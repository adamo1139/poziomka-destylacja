#!/usr/bin/env python3
"""Etap 2 OPRD-Bridge: destylacja reprezentacji on-policy przez zamrozony most (rown. 14-15).

Krok (alg. 1, linie 16-19):
  1. rank 0: rollouty ucznia z SGLanga (/generate, t=1.0) - tokeny i logproby polityki
  2. rank 0: nauczyciel (teacher_server.py) - z_T = P_T (h_T - mu_T) i logproby tokenow
  3. wszystkie rangi: forward/backward ucznia, strata = srednia po warstwach z
     ||norm(P_S h_S) - norm(z_T)||^2 / r po tokenach odpowiedzi; krok AdamW
  4. pelny state_dict -> safetensors w ukladzie HF -> SGLang /update_weights_from_disk

Most, LM head, koncowy norm i router MoE sa zamrozone (decyzja: najpierw sam Bridge,
bez OPD; router zamrozony, zeby strata na stanach posrednich nie rozjechala routingu).
Metryka: KL(uczen || nauczyciel) na rolloutach ucznia, estymatory k1 i k3 z logprobow
wylosowanych tokenow, na kazdym kroku (rollouty treningowe) i co --eval-every na val.

Rig (8x 3090 Ti): GPU0 SGLang, GPU1 teacher_server, GPU2-7 ten skrypt:
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc-per-node 6 oprd_bridge/train.py ...
Lokalnie bez SGLanga i nauczyciela: --offline-cache (z teacher_server.py --dump),
--train-layers do ograniczenia pamieci.
"""

import argparse
import json
import math
import os
import random
import shutil
import time
import urllib.request
from pathlib import Path

import common  # noqa: F401  (USE_TF=0)
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import LayerCapture, read_jsonl, render_prompt_ids
from moe_fast import moe_blocks, patch_model, split_fused_state_dict
from teacher_server import decode as decode_teacher


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def log0(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


# ---------------------------------------------------------------- rollouty i nauczyciel (rank 0)

class Rollouts:
    def __init__(self, args, tok):
        self.args, self.tok = args, tok
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.prompts = [r for r in read_jsonl(args.prompts) if r.get("split", "train") == "train"]
        self.eval_prompts = [r for r in read_jsonl(args.eval_prompts or args.prompts) if r.get("split") == "val"]
        self.rng = random.Random(args.seed)
        self.order, self.pos = [], 0
        if args.offline_cache:
            self.cache = torch.load(args.offline_cache)["seqs"]

    def _next_prompts(self, n):
        out = []
        while len(out) < n:
            if self.pos >= len(self.order):
                self.order = list(range(len(self.prompts)))
                self.rng.shuffle(self.order)
                self.pos = 0
            out.append(self.prompts[self.order[self.pos]])
            self.pos += 1
        return out

    def generate(self, prompts, seed=None):
        a = self.args
        ids = [render_prompt_ids(self.tok, p["messages"], False) for p in prompts for _ in range(a.samples)]
        sp = {"temperature": a.temperature, "top_p": 1.0, "max_new_tokens": a.max_new_tokens,
              "stop_token_ids": [self.im_end]}
        if seed is not None:
            sp["sampling_seed"] = seed
        res = post(a.sglang + "/generate", {"input_ids": ids, "sampling_params": sp, "return_logprob": True})
        seqs = []
        for prompt_ids, r in zip(ids, res):
            lp = r["meta_info"]["output_token_logprobs"]  # [(logprob, token_id, tekst)]
            out_ids = [t[1] for t in lp]
            if not out_ids:
                continue
            seqs.append({"ids": prompt_ids + out_ids, "start": len(prompt_ids), "s_logp": [t[0] for t in lp],
                         "finish": (r["meta_info"].get("finish_reason") or {}).get("type")})
        return seqs

    def score(self, seqs):
        a = self.args
        res = decode_teacher(post(a.teacher + "/score", {
            "seqs": [{"ids": s["ids"], "start": s["start"]} for s in seqs],
            "layers": a.layers, "rank": a.rank})["results"])
        for s, r in zip(seqs, res):
            s["z"], s["t_logp"] = torch.from_numpy(r["z"].copy()), r["logp"]
        return seqs

    def batch(self):
        if self.args.offline_cache:  # stale rollouty (test mechaniki, nie on-policy)
            return [self.cache[self.rng.randrange(len(self.cache))] for _ in range(self.args.prompts_per_step * self.args.samples)]
        return self.score(self.generate(self._next_prompts(self.args.prompts_per_step)))


def kl_stats(seqs):
    """Reverse KL(uczen || nauczyciel) z logprobow wylosowanych tokenow (rollouty z t=1)."""
    parts = [np.asarray(s["s_logp"]) - np.asarray(s["t_logp"]) for s in seqs if s.get("s_logp")]
    if not parts:  # offline: brak logprobow polityki
        return {}
    d = np.concatenate(parts)
    return {"kl_k1": float(d.mean()), "kl_k3": float((np.exp(-d) - 1 + d).mean()),
            "kl_seq": float(d.sum() / len(seqs))}


# ---------------------------------------------------------------- model

def build_model(args, device):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, attn_implementation=args.attn, trust_remote_code=True).to(device)
    patch_model(model)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_layers = set(map(int, args.train_layers.split(","))) if args.train_layers else None
    for name, p in model.named_parameters():
        frozen = (
            name.startswith("lm_head") or name == "model.norm.weight"  # strata nie dochodzi do wyjscia
            or ".mlp.gate.weight" in name  # router
            or (train_layers is not None and not any(name.startswith(f"model.layers.{l}.") for l in train_layers))
        )
        p.requires_grad_(not frozen)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # master wagi w fp32; rzutujemy warstwa po warstwie, zeby nie trzymac calego modelu w fp32 na jednej karcie.
    # Forward idzie przez model.model (bez LM head), wiec to on jest korzeniem FSDP; lm_head zostaje
    # w bf16 bez shardowania - jest zamrozony i nie bierze udzialu w stracie.
    # Zamrozone czesci zostaja w bf16 (przy --train-layers caly model w fp32 nie zmiescilby sie lokalnie).
    trainable = lambda m: any(p.requires_grad for p in m.parameters())
    for layer in model.model.layers:
        if trainable(layer):
            layer.float()
        fully_shard(layer, mp_policy=mp)
    if trainable(model.get_input_embeddings()):  # w Bailingu: model.word_embeddings
        model.get_input_embeddings().float()
    fully_shard(model.model, mp_policy=mp)
    return model


def export_hf(model, out_dir, src_dir):
    """Pelne wagi (wszystkie rangi wolaja) -> rank 0 zapisuje katalog HF ladowalny przez SGLang/HF."""
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dist.get_rank() != 0:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(src_dir).iterdir():  # config, tokenizer, remote code - bez wag i indeksu
        if f.is_file() and not f.name.endswith((".safetensors", ".safetensors.index.json", ".bin")):
            shutil.copy2(f, out_dir / f.name)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in split_fused_state_dict(sd).items()}
    tmp = out_dir / "model.safetensors.tmp"
    save_file(sd, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, out_dir / "model.safetensors")


# ---------------------------------------------------------------- strata

def microbatches(seqs, max_tokens):
    seqs = sorted(seqs, key=lambda s: len(s["ids"]))
    out, cur = [], []
    for s in seqs:
        if cur and len(s["ids"]) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return out


def bridge_loss(model, mb, P_S, layers, device, pad_id):
    """Suma po tokenach (srednia po warstwach) + kosinusy per warstwa; normalizacja po stronie wywolujacego."""
    width = max(len(s["ids"]) for s in mb)
    ids = torch.full((len(mb), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(mb):
        ids[i, : len(s["ids"])] = torch.tensor(s["ids"])
        mask[i, : len(s["ids"])] = 1
    with LayerCapture(model, layers) as cap:
        model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    loss = 0.0
    cos = torch.zeros(len(layers), device=device)
    for j, l in enumerate(layers):
        h = torch.cat([cap.outputs[l][i, s["start"]: len(s["ids"])] for i, s in enumerate(mb)])
        zs = F.normalize(h.float() @ P_S[l].T, dim=-1)
        zt = F.normalize(torch.cat([s["z"][j] for s in mb]).to(device).float(), dim=-1)
        loss = loss + ((zs - zt) ** 2).mean(-1).sum() / len(layers)  # jak normalized_mse_loss w kodzie OPRD
        cos[j] = (zs * zt).sum(-1).sum().detach()
    return loss, cos


# ---------------------------------------------------------------- petla

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", required=True)
    p.add_argument("--bridge", required=True)
    p.add_argument("--prompts", required=True, help="JSONL z prompts.py (split=train).")
    p.add_argument("--eval-prompts", help="JSONL z promptami val (domyslnie split=val z --prompts).")
    p.add_argument("--out", required=True)
    p.add_argument("--sglang", default="http://127.0.0.1:30000")
    p.add_argument("--teacher", default="http://127.0.0.1:30100")
    p.add_argument("--sync-dir", help="Katalog wag dla SGLanga (lokalny szybki dysk); domyslnie OUT/sync.")
    p.add_argument("--offline-cache", help="Cache z teacher_server.py --dump zamiast SGLanga i nauczyciela.")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--prompts-per-step", type=int, default=32)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--layers", default=",".join(map(str, range(1, 16))), help="Warstwy ucznia w stracie (bez 0).")
    p.add_argument("--train-layers", help="Tylko te warstwy trenowalne (lokalne testy pamieciowe).")
    p.add_argument("--micro-tokens", type=int, default=2048,
                   help="Tokeny (z paddingiem) na mikrobatch; wejscie ekspertow to x32 tyle wierszy.")
    p.add_argument("--grad-ckpt", action="store_true", help="Gradient checkpointing warstw (mniej pamieci).")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-n", type=int, default=200)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--no-sync", action="store_true", help="Bez aktualizacji SGLanga (tylko testy).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    sync_dir = Path(args.sync_dir or out / "sync")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    pad_id = tok.pad_token_id or 0
    bridge = torch.load(args.bridge)
    P_S = {l: bridge["P_S"][l][: args.rank].to(device).float() for l in args.layers}
    model = build_model(args, device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)  # DTensor: numel globalny
    log0(f"world {world}, trenowalne parametry {n_train / 1e9:.2f}B, warstwy straty {args.layers}, r={args.rank}")
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    warm = max(1, int(args.warmup * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))

    # obciazenie ekspertow: histogram wyborow routera na warstwe
    load = {}
    def gate_hook(name):
        def hook(_m, _i, o):
            load[name] = load.get(name, 0) + torch.bincount(o[0].reshape(-1), minlength=_m.num_experts)
        return hook
    for name, block in moe_blocks(model):
        block.gate.register_forward_hook(gate_hook(name))

    ro = Rollouts(args, tok) if rank == 0 else None
    metrics_f = open(out / "metrics.jsonl", "a") if rank == 0 else None

    def evaluate(step):
        if rank != 0 or args.offline_cache:
            return
        prompts = ro.eval_prompts[: args.eval_n]
        seqs = ro.score(ro.generate(prompts, seed=12345))
        m = {"step": step, "eval": True, **kl_stats(seqs),
             "len": float(np.mean([len(s["ids"]) - s["start"] for s in seqs])),
             "truncated": float(np.mean([s["finish"] == "length" for s in seqs]))}
        print("EVAL", json.dumps(m), flush=True)
        metrics_f.write(json.dumps(m) + "\n")
        metrics_f.flush()

    model.train()
    evaluate(0)
    for step in range(1, args.steps + 1):
        t0 = time.time()
        seqs = [ro.batch() if rank == 0 else None]
        t_roll = time.time() - t0
        dist.broadcast_object_list(seqs, src=0)
        seqs = seqs[0]
        # rownowazenie tokenow miedzy rangami, potem wspolna liczba mikrobatchy (FSDP wymaga rownych krokow)
        seqs_sorted = sorted(seqs, key=lambda s: -len(s["ids"]))
        mine = [s for i, s in enumerate(seqs_sorted) if i % world == rank]
        mbs = microbatches(mine, args.micro_tokens)
        n_mb = torch.tensor(len(mbs), device=device)
        dist.all_reduce(n_mb, op=dist.ReduceOp.MAX)
        n_tok = torch.tensor(float(sum(len(s["ids"]) - s["start"] for s in seqs)), device=device)
        load.clear()
        cos_sum = torch.zeros(len(args.layers), device=device)
        loss_sum = torch.zeros((), device=device)
        for k in range(int(n_mb)):
            if k < len(mbs):
                loss, cos = bridge_loss(model, mbs[k], P_S, args.layers, device, pad_id)
                scale = 1.0
            else:  # pusty mikrobatch tej rangi - forward/backward z waga 0 dla zgodnosci kolektywow FSDP
                loss, cos = bridge_loss(model, [seqs_sorted[-1]], P_S, args.layers, device, pad_id)
                scale, cos = 0.0, cos * 0
            (loss * scale / n_tok).backward()
            loss_sum += loss.detach() * scale
            cos_sum += cos
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        for t in (loss_sum, cos_sum):
            dist.all_reduce(t)
        expert = torch.stack([v for _, v in sorted(load.items())]).float()
        dist.all_reduce(expert)
        t_train = time.time() - t0 - t_roll

        t1 = time.time()
        if not args.no_sync and not args.offline_cache:
            export_hf(model, sync_dir, args.student)
            if rank == 0:
                r = post(args.sglang + "/update_weights_from_disk", {"model_path": str(sync_dir)})
                if not r.get("success"):
                    raise RuntimeError(f"SGLang nie przyjal wag: {r}")
            dist.barrier()
        t_sync = time.time() - t1

        if rank == 0:
            cos_l = (cos_sum / n_tok).tolist()
            share = expert / expert.sum(1, keepdim=True)
            m = {"step": step, "loss": float(loss_sum / n_tok), "cos": float(np.mean(cos_l)),
                 "cos_layers": [round(c, 4) for c in cos_l], "grad_norm": float(grad_norm),
                 "lr": sched.get_last_lr()[0], "tokens": int(n_tok), **kl_stats(seqs),
                 "len": float(n_tok) / len(seqs),
                 "truncated": float(np.mean([s.get("finish") == "length" for s in seqs])),
                 "expert_max_share": float(share.max(1).values.mean() * share.shape[1]),  # 1.0 = rowno
                 "experts_unused": float((expert == 0).float().sum(1).mean()),
                 "t_rollout": round(t_roll, 1), "t_train": round(t_train, 1), "t_sync": round(t_sync, 1)}
            print(json.dumps(m), flush=True)
            metrics_f.write(json.dumps(m) + "\n")
            metrics_f.flush()
        if step % args.eval_every == 0:
            evaluate(step)
        if step % args.save_every == 0 or step == args.steps:
            export_hf(model, out / f"step_{step:05d}", args.student)
            log0(f"checkpoint {out / f'step_{step:05d}'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
