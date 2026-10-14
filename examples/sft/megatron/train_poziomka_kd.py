#!/usr/bin/env python3
"""Off-policy KD for Poziomka: reverse KL to the teacher's top-k on a packed SFT cache.

Same model, data path and packing as train_poziomka_sft.py (packed mode only); what changes
is the loss. The model runs without labels, so the last pipeline stage returns logits
instead of Megatron's fused CE, and kd_loss_func computes the bucketed reverse KL of
poziomka_kd_loss against the teacher rows that MMapKDDataset attaches to every window.
Every stage reads its own identical batch (DP=TP=1, deterministic dataset), as in SFT;
only the last stage loads teacher rows.
"""
from functools import partial
from pathlib import Path

import torch

from poziomka_data import load_manifest, sha256
from poziomka_kd_data import MMapKDDataset, label_coverage
from poziomka_kd_loss import topk_reverse_kl
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import get_args, pretrain, print_rank_0
from pretrain_gpt import model_provider, has_nvidia_modelopt


def extra_args(parser):
    if has_nvidia_modelopt:  # model_provider reads these args when ModelOpt is installed
        from pretrain_gpt import add_modelopt_args
        parser = add_modelopt_args(parser)
    parser.add_argument("--sft-manifest", required=True)
    parser.add_argument("--teacher-topk", required=True,
                        help="teacher-topk-v1 directory computed from exactly this cache")
    parser.add_argument("--kd-logit-chunk", type=int, default=1024,
                        help="positions per log-softmax chunk in the KD loss")
    return parser


def kd_loss_func(loss_mask, labels, topk_ids, topk_logp, output_tensor):
    """Sum of per-token reverse KL over supervised positions, plus monitoring metrics.

    Returned in --calculate-per-token-loss form: (sum, tokens, {name: [sum, tokens]}).
    `student ce` is plain next-token CE on the same logits (comparable with SFT lm loss);
    `student tail` is the student's mass outside the teacher's top-k, which the bucketed
    loss under-penalises -- if it grows, the top-k approximation is being exploited.
    """
    args = get_args()
    vocab = output_tensor.shape[-1]
    mask = loss_mask.view(-1) > 0
    kl, ce, tail = topk_reverse_kl(
        output_tensor.reshape(-1, vocab), topk_ids.view(-1, topk_ids.shape[-1]),
        topk_logp.view(-1, topk_logp.shape[-1]), labels.view(-1), mask, args.kd_logit_chunk)
    loss = kl.sum()
    if args.check_for_nan_in_loss_and_grad:
        rerun = get_rerun_state_machine()
        for reject, what in ((torch.isnan, "NaN"), (torch.isinf, "Inf")):
            rerun.validate_result(result=loss, rejection_func=reject, tolerance=0.0, fatal=True,
                                  message=f"found {what} in local KD loss calculation")
    num_tokens = mask.sum().to(torch.int)
    n = num_tokens.float().view(1)
    report = {"kd loss": torch.cat([loss.detach().view(1), n]),
              "student ce": torch.cat([ce.sum().view(1), n]),
              "student tail": torch.cat([tail.sum().view(1), n])}
    return loss, num_tokens, report


def kd_forward_step(data_iterator, model):
    """packed_forward_step of train_poziomka_sft.py with labels=None (logits out)."""
    args = get_args()
    batch = next(data_iterator)
    device = torch.cuda.current_device()
    tokens = batch["tokens"].to(device, non_blocking=True)
    position_ids = batch["position_ids"].to(device, non_blocking=True)
    cu_seqlens = batch["cu_seqlens"][0].to(device, non_blocking=True).to(torch.int32)
    kept = int((cu_seqlens < args.seq_length).sum().item()) + 1
    cu_seqlens = cu_seqlens[:kept].contiguous()
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens, cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen)
    output_tensor = model(tokens, position_ids, None, labels=None,
                          packed_seq_params=packed_seq_params)
    if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        return output_tensor, None  # only the last stage calls the loss function
    teacher = [batch[key].to(device, non_blocking=True)
               for key in ("loss_mask", "labels", "topk_ids", "topk_logp")]
    return output_tensor, partial(kd_loss_func, *teacher)


def datasets_provider(sample_counts):
    args = get_args()
    manifest_path = Path(args.sft_manifest).resolve()
    manifest = load_manifest(manifest_path)
    if args.seq_length != manifest["seq_length"]:
        raise ValueError("Training seq-length must match prepared cache")
    if args.context_parallel_size != 1 or args.tensor_model_parallel_size != 1 or args.expert_model_parallel_size != 1:
        raise ValueError("KD supports TP=CP=EP=1, like the Poziomka SFT adapter")
    if args.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Virtual/interleaved pipeline parallelism is not supported here")
    if args.reset_position_ids or args.reset_attention_mask or args.eod_mask_loss:
        raise ValueError("Do not reset on im_end: it ends a turn, not a conversation")
    if args.dataloader_type != "single":
        raise ValueError("Use the single sampler; the dataset handles deterministic epoch shuffling")
    if args.micro_batch_size != 1:
        raise ValueError("Packing (always on in KD) requires micro-batch 1: thd drops the batch dimension")
    if not args.calculate_per_token_loss:
        raise ValueError("kd_loss_func returns per-token sums; use --calculate-per-token-loss")
    tokenizer_dir = Path(args.tokenizer_model).resolve()
    if sha256(tokenizer_dir / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Training tokenizer differs from cache tokenizer")
    if sha256(manifest_path.parent / "chat_template.jinja") != manifest["template_sha256"]:
        raise ValueError("Cache chat template changed after preparation")
    last = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    train = MMapKDDataset(manifest_path, "train", args.teacher_topk, sample_counts[0], args.seed,
                          pack=True, with_teacher=last)
    valid = MMapKDDataset(manifest_path, "validation", args.teacher_topk, sample_counts[1], args.seed,
                          shuffle=False, pack=True, with_teacher=last)
    if last:
        # Alignment guard: labels must sit among the teacher's top-k almost always.
        coverage = label_coverage(train)
        print(f"KD: label in teacher top-{train.k} for {100 * coverage:.2f}% of supervised "
              f"tokens (first samples)", flush=True)
        if coverage < 0.9:
            raise ValueError("Teacher top-k does not line up with the cache labels")
    print_rank_0(f"KD: {train.record_count} train conversations packed into {train.sample_count} "
                 f"sequences ({100 * train.packing_efficiency():.1f}% real tokens); "
                 f"{valid.record_count} validation; teacher {train.teacher['model']} k={train.k}")
    return train, valid, None


if __name__ == "__main__":
    datasets_provider.is_distributed = True
    pretrain(datasets_provider, model_provider, ModelType.encoder_or_decoder, kd_forward_step,
             extra_args_provider=extra_args,
             args_defaults={"tokenizer_type": "HuggingFaceTokenizer"})
