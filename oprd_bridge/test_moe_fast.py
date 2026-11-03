#!/usr/bin/env python3
"""Zgodnosc moe_fast z remote code Poziomki (forward, backward, eksport wag) + czas.

  python3 oprd_bridge/test_moe_fast.py --model $OPRD/poziomka_... --rollouts $W/rollouts.jsonl
"""

import argparse
import copy
import time

import common  # noqa: F401  (USE_TF=0)
import torch

from common import load_model, read_jsonl, render_prompt_ids
from moe_fast import FusedExperts, _fast_block_forward, hf_state_dict, moe_blocks, patch_model


def rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()  # na CPU - na 16 GB brakuje miejsca
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--layers", default="1,8,15")
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device
    torch.manual_seed(0)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dev)
    ids = []
    for r in read_jsonl(args.rollouts):
        ids += render_prompt_ids(tok, r["messages"], False) + tok.encode(r["response"], add_special_tokens=False)
        if len(ids) >= args.tokens:
            break
    ids = torch.tensor([ids[: args.tokens]], device=dev)

    blocks = dict(moe_blocks(model))
    names = {int(n.split(".")[2]): n for n in blocks}
    layers = [int(x) for x in args.layers.split(",")]

    # wejscia blokow MoE i stany warstw z oryginalnego modelu (eval)
    inputs, hs_ref = {}, {}
    hooks = [blocks[names[l]].register_forward_hook(
        lambda m, i, o, l=l: inputs.__setitem__(l, i[0].detach().clone())) for l in layers]
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    for h in hooks:
        h.remove()
    logits_ref = out.logits.float()
    hs_ref = [h.float() for h in out.hidden_states]
    del out

    print(f"Tokenow: {ids.shape[1]}; grouped_mm: "
          f"{'F.grouped_mm' if hasattr(torch.nn.functional, 'grouped_mm') else 'torch._grouped_mm' if hasattr(torch, '_grouped_mm') else 'petla'}")
    print("\n== blok MoE w trybie train: remote code vs fused (rel. blad = ||a-b|| / ||ref||)")
    for l in layers:
        ref = blocks[names[l]]
        ref.requires_grad_(True)  # load_model zamraza wszystko
        fast = copy.deepcopy(ref)
        fast.fused_experts = FusedExperts(fast.experts)
        del fast.experts
        fast.forward = _fast_block_forward.__get__(fast)
        ref.train(); fast.train()
        x = inputs[l]
        R = torch.randn(x.shape, device=dev)

        def run(block):
            xx = x.clone().requires_grad_(True)
            y, _ = block(xx)
            (y.float() * R).sum().backward()
            return y.detach(), xx.grad

        for m in (ref, fast):
            m.zero_grad(set_to_none=True)
        y_r, gx_r = run(ref)
        y_f, gx_f = run(fast)
        g = lambda p: p.grad if p.grad is not None else torch.zeros_like(p)  # ekspert bez tokenow
        g_gu_ref = torch.stack([torch.cat([g(e.gate_proj.weight), g(e.up_proj.weight)]) for e in ref.experts])
        g_dn_ref = torch.stack([g(e.down_proj.weight) for e in ref.experts])
        print(f"warstwa {l:>2}: y {rel(y_f, y_r):.2e} | grad x {rel(gx_f, gx_r):.2e} | "
              f"grad gate_up {rel(fast.fused_experts.gate_up.grad, g_gu_ref):.2e} | "
              f"grad down {rel(fast.fused_experts.down.grad, g_dn_ref):.2e} | "
              f"grad shared {rel(fast.shared_experts.down_proj.weight.grad, ref.shared_experts.down_proj.weight.grad):.2e} | "
              f"grad bramki {rel(fast.gate.weight.grad, ref.gate.weight.grad):.2e}")

        def step(block):
            block.zero_grad(set_to_none=True)
            run(block)
        t_r, t_f = timed(lambda: step(ref)), timed(lambda: step(fast))
        print(f"           fwd+bwd: remote {t_r * 1000:.0f} ms, fused {t_f * 1000:.0f} ms ({t_r / t_f:.1f}x)")
        ref.eval()
        ref.requires_grad_(False)
        ref.zero_grad(set_to_none=True)
        del fast, y_r, gx_r, y_f, gx_f, g_gu_ref, g_dn_ref
        torch.cuda.empty_cache()

    print("\n== caly model (eval): oryginal vs po patch_model")
    sd_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if ".experts." in k}
    print(f"podmienione bloki: {patch_model(model)}")
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    worst = max(rel(a, b) for a, b in zip(out.hidden_states, hs_ref))
    agree = float((out.logits.argmax(-1) == logits_ref.argmax(-1)).float().mean())
    print(f"stany warstw: max rel. blad {worst:.2e}; logity: rel {rel(out.logits, logits_ref):.2e}, zgodnosc argmax {agree:.4f}")
    sd_after = hf_state_dict(model)
    exact = all(torch.equal(sd_after[k].cpu(), v) for k, v in sd_before.items())
    print(f"eksport do kluczy per ekspert: {len(sd_before)} tensorow, identyczne: {exact}; "
          f"klucze jak w oryginale: {set(sd_after) >= set(sd_before)}")


if __name__ == "__main__":
    main()
