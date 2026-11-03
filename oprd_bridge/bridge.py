#!/usr/bin/env python3
"""Etap 1 OPRD-Bridge (arXiv 2606.06021 v4, par. 3.3.2) na jednym GPU.

Nauczyciel 7B i Poziomka nie mieszcza sie razem na 16 GB, a w Etapie 1 oba sa
zamrozone, wiec liczymy je osobno, po jednym modelu na przebieg:

  1. teacher-pca      nauczyciel na rolloutach train -> mu_T i baza PCA P_T (top r_max)
                      dla kazdej warstwy nauczyciela sparowanej z warstwa ucznia (phi, rown. 11)
  2. teacher-project  nauczyciel na train+val -> z_T = P_T (h_T - mu_T) na dysk (fp16)
  3. student-fit      uczen -> P_S z ridge'owych najmniejszych kwadratow (rown. 13),
                      kosinus/R^2 na val dla kazdej rangi -> bridge.pt + report.md

Roznica wzgledem paperu: P_S liczymy w zamknietej postaci zamiast 20 epok Adama.
Cel jest ten sam (liniowe MSE do z_T), a LS daje dokladne minimum na wszystkich
tokenach naraz i - bo kolumny z_T sa niezalezne - jednym rozwiazaniem dla
wszystkich rang (P_S rangi r = pierwsze r wierszy).

Oba modele dostaja te same tokeny odpowiedzi (wspolny tokenizer APT4); prompt
nauczyciela mozna renderowac jego wlasnym szablonem (--teacher-template own) albo
szablonem ucznia jak w paperze (student). Stany bierzemy hakami z wyjsc blokow,
czyli residual stream przed koncowym normem.

Przyklad:
  B=oprd_bridge/bridge.py; W=$OPRD/bridge-bielik7b-poziomka-baza
  python3 $B teacher-pca     --teacher $OPRD/bielik --student $OPRD/poziomka --rollouts R.jsonl --work $W
  python3 $B teacher-project --teacher $OPRD/bielik --student $OPRD/poziomka --rollouts R.jsonl --work $W
  python3 $B student-fit     --teacher $OPRD/bielik --student $OPRD/poziomka --rollouts R.jsonl --work $W
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (LayerCapture, check_same_vocab, layer_map, load_model, read_jsonl,
                    render_prompt_ids)

THINKING = {"on": True, "off": False, "none": None}
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


# ---------------------------------------------------------------- dane

def build_examples(args, student_tok, teacher_tok=None):
    """Sekwencje dla ucznia i (opcjonalnie) nauczyciela z identycznymi tokenami odpowiedzi."""
    rollouts = read_jsonl(args.rollouts)
    if args.max_rollouts:
        rollouts = rollouts[: args.max_rollouts]
    im_end = student_tok.convert_tokens_to_ids("<|im_end|>")
    examples = []
    for idx, r in enumerate(rollouts):
        thinking = THINKING[r.get("thinking", "none")]
        resp = student_tok.encode(r["response"], add_special_tokens=False)
        if r.get("finish_reason") == "stop":
            resp = resp + [im_end]
        if not resp:
            continue
        s_prompt = render_prompt_ids(student_tok, r["messages"], thinking)
        if teacher_tok is not None and args.teacher_template == "own":
            t_prompt = render_prompt_ids(teacher_tok, r["messages"], thinking)
        else:
            t_prompt = s_prompt
        budget = args.max_seq_len - max(len(s_prompt), len(t_prompt))
        if budget < 16:
            continue
        resp = resp[:budget]
        positions = list(range(len(resp)))
        if args.positions == "last_k":
            positions = positions[-args.last_k:]
        examples.append({
            "idx": idx, "split": r["split"], "resp": resp, "pos": positions,
            "s_prompt": s_prompt, "t_prompt": t_prompt,
        })
    return examples


def plan_batches(examples, max_tokens):
    """Deterministyczny podzial na batche (te same we wszystkich przebiegach)."""
    order = sorted(examples, key=lambda e: (len(e["s_prompt"]) + len(e["resp"]), e["idx"]))
    batches, cur, cur_max = [], [], 0
    for ex in order:
        n = max(len(ex["s_prompt"]), len(ex["t_prompt"])) + len(ex["resp"])
        if cur and max(cur_max, n) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(ex)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def forward_rows(model, batch, which, layer_indices, pad_id, device):
    """Zwraca {warstwa: (sum_tokenow, d)} stanow na wybranych pozycjach odpowiedzi."""
    seqs = [ex[which] + ex["resp"] for ex in batch]
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    with LayerCapture(model, layer_indices) as cap:
        model.get_decoder()(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    rows = {}
    for layer in cap.indices:
        h = cap.outputs[layer]
        parts = []
        for i, ex in enumerate(batch):
            start = len(ex[which])
            pos = torch.tensor(ex["pos"], device=h.device) + start
            parts.append(h[i, pos])
        # przy offloadzie czesc warstw liczy sie na CPU - wszystko zbieramy na device
        rows[layer] = torch.cat(parts).to(device).float()
    return rows


def load_teacher(args):
    if args.teacher_max_gpu_mem:
        # dokladne bf16: warstwy, ktore sie nie mieszcza, ida na CPU (accelerate przerzuca je przy forwardzie)
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
            device_map="auto", max_memory={int(args.device.split(":")[-1]): args.teacher_max_gpu_mem, "cpu": "48GiB"},
            trust_remote_code=True,
        )
        on_cpu = sum(1 for d in model.hf_device_map.values() if d in ("cpu", "disk"))
        print(f"Nauczyciel: {on_cpu}/{len(model.hf_device_map)} modulow na CPU")
        return model.eval()
    if args.teacher_8bit:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            args.teacher, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.bfloat16, attn_implementation=args.attn,
            device_map={"": args.device}, trust_remote_code=True,
        )
        return model.eval()
    return load_model(args.teacher, args.device, attn=args.attn)


def tokenizers(args, need_teacher):
    from transformers import AutoTokenizer, AutoConfig

    s_tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    t_tok = None
    if need_teacher:
        t_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        check_same_vocab(s_tok, t_tok)
    s_cfg = AutoConfig.from_pretrained(args.student, trust_remote_code=True)
    return s_tok, t_tok, s_cfg


PLAN_ARGS = ("student", "teacher", "rollouts", "teacher_template", "max_seq_len", "batch_tokens",
             "positions", "last_k", "max_rollouts")


def plan_key(args):
    return {k: getattr(args, k) for k in PLAN_ARGS}


def check_plan(pca, args):
    """Wszystkie przebiegi musza widziec te same przyklady i batche."""
    diff = {k: (v, getattr(args, k)) for k, v in pca["plan"].items() if v != getattr(args, k)}
    if diff:
        raise SystemExit(f"Parametry inne niz w teacher-pca (bylo, jest): {diff}")


# ---------------------------------------------------------------- 1. PCA nauczyciela

def cmd_teacher_pca(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    s_tok, t_tok, s_cfg = tokenizers(args, need_teacher=True)
    model = load_teacher(args)
    n_teacher = model.config.num_hidden_layers
    phi = layer_map(s_cfg.num_hidden_layers, n_teacher)
    t_layers = sorted(set(phi))
    examples = [e for e in build_examples(args, s_tok, t_tok) if e["split"] == "train"]
    batches = plan_batches(examples, args.batch_tokens)
    print(f"phi (uczen->nauczyciel, 0-based): {phi}\n{len(examples)} rolloutow train, {len(batches)} batchy")

    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    sums = {t: None for t in t_layers}
    count = 0
    buf = {t: None for t in t_layers}
    keys = {t: None for t in t_layers}
    for b, batch in enumerate(batches):
        rows = forward_rows(model, batch, "t_prompt", t_layers, s_tok.pad_token_id or 0, args.device)
        n = next(iter(rows.values())).shape[0]
        k = torch.rand(n, generator=gen)
        for t in t_layers:
            x = rows[t].cpu()
            s = x.double().sum(0)
            sums[t] = s if sums[t] is None else sums[t] + s
            # rezerwuar: zostaja wiersze z najmniejszymi losowymi kluczami
            bx = x if buf[t] is None else torch.cat([buf[t], x])
            bk = k if keys[t] is None else torch.cat([keys[t], k])
            if bk.numel() > args.pca_rows:
                keep = torch.topk(bk, args.pca_rows, largest=False).indices
                bx, bk = bx[keep], bk[keep]
            buf[t], keys[t] = bx, bk
        count += n
        if b % 20 == 0 or b == len(batches) - 1:
            dt = time.time() - t0
            print(f"  batch {b + 1}/{len(batches)}, tokenow {count}, {count / dt:.0f} tok/s, {dt / 60:.1f} min", flush=True)

    print(f"PCA ({len(t_layers)} warstw)...", flush=True)
    out = {"phi": phi, "teacher_layers": t_layers, "n_tokens": count, "r_max": args.r_max,
           "plan": plan_key(args), "mean": {}, "P_T": {}, "eigvals": {}}
    for t in t_layers:
        mu = sums[t] / count
        x = buf[t].double() - mu
        cov = x.T @ x / max(x.shape[0] - 1, 1)
        vals, vecs = torch.linalg.eigh(cov)
        vals, vecs = vals.flip(0), vecs.flip(1)
        out["mean"][t] = mu.float()
        out["P_T"][t] = vecs[:, : args.r_max].T.contiguous().float()  # (r_max, d_T)
        out["eigvals"][t] = vals.float()
    torch.save(out, work / "teacher_pca.pt")
    print(f"Zapisano {work / 'teacher_pca.pt'} ({count} tokenow, PCA z <= {args.pca_rows} wierszy/warstwe)")


# ---------------------------------------------------------------- 2. projekcja nauczyciela

def cmd_teacher_project(args):
    work = Path(args.work)
    pca = torch.load(work / "teacher_pca.pt")
    check_plan(pca, args)
    s_tok, t_tok, s_cfg = tokenizers(args, need_teacher=True)
    model = load_teacher(args)
    phi = pca["phi"]
    examples = build_examples(args, s_tok, t_tok)
    batches = plan_batches(examples, args.batch_tokens)
    zdir = work / "z_T"
    zdir.mkdir(exist_ok=True)
    P = {t: pca["P_T"][t].to(args.device) for t in pca["teacher_layers"]}
    mu = {t: pca["mean"][t].to(args.device) for t in pca["teacher_layers"]}
    for b, batch in enumerate(batches):
        path = zdir / f"batch_{b:05d}.pt"
        if path.exists():
            continue
        rows = forward_rows(model, batch, "t_prompt", pca["teacher_layers"], s_tok.pad_token_id or 0, args.device)
        z = {t: ((rows[t] - mu[t]) @ P[t].T).half().cpu() for t in rows}
        # z dla kazdej warstwy UCZNIA (kilka warstw ucznia moze dzielic warstwe nauczyciela)
        torch.save({"idx": [e["idx"] for e in batch], "z": torch.stack([z[t] for t in phi])}, path)
        if b % 20 == 0:
            print(f"  batch {b + 1}/{len(batches)}", flush=True)
    print(f"z_T w {zdir} ({len(batches)} batchy)")


# ---------------------------------------------------------------- 3. dopasowanie P_S

def cmd_student_fit(args):
    work = Path(args.work)
    pca = torch.load(work / "teacher_pca.pt")
    check_plan(pca, args)
    # tokenizer nauczyciela tylko po to, zeby uklad przykladow/batchy byl identyczny jak w teacher-project
    s_tok, t_tok, s_cfg = tokenizers(args, need_teacher=True)
    model = load_model(args.student, args.device, attn=args.attn)
    n_layers = s_cfg.num_hidden_layers
    s_layers = list(range(n_layers))
    examples = build_examples(args, s_tok, t_tok)
    batches = plan_batches(examples, args.batch_tokens)
    r_max = pca["r_max"]
    dev = args.device
    pad = s_tok.pad_token_id or 0

    def batch_data(b, batch):
        saved = torch.load(work / "z_T" / f"batch_{b:05d}.pt")
        if saved["idx"] != [e["idx"] for e in batch]:
            raise RuntimeError("Batche ucznia i z_T sie rozjechaly - uzyj tych samych --rollouts/--max-seq-len/--batch-tokens.")
        return forward_rows(model, batch, "s_prompt", s_layers, pad, dev), saved["z"].to(dev).float()

    # Uwaga: batch moze miec mieszane splity, wiec filtrujemy po wierszach.
    def split_mask(batch, split):
        return torch.cat([torch.full((len(e["pos"]),), e["split"] == split, dtype=torch.bool) for e in batch]).to(dev)

    # --- statystyki do najmniejszych kwadratow (train)
    d_s = s_cfg.hidden_size
    A = torch.zeros(n_layers, d_s, d_s, dtype=torch.float64, device=dev)
    Bm = torch.zeros(n_layers, d_s, r_max, dtype=torch.float64, device=dev)
    sh = torch.zeros(n_layers, d_s, dtype=torch.float64, device=dev)
    sz = torch.zeros(n_layers, r_max, dtype=torch.float64, device=dev)
    n_train = 0
    for b, batch in enumerate(batches):
        m = split_mask(batch, "train")
        if not m.any():
            continue
        rows, z = batch_data(b, batch)
        for l in s_layers:
            h, zl = rows[l][m], z[l][m]
            A[l] += (h.T @ h).double()
            Bm[l] += (h.T @ zl).double()
            sh[l] += h.sum(0).double()
            sz[l] += zl.sum(0).double()
        n_train += int(m.sum())
        if b % 20 == 0:
            print(f"  fit batch {b + 1}/{len(batches)}, tokenow train {n_train}", flush=True)

    W = torch.empty(n_layers, d_s, r_max, dtype=torch.float64, device=dev)
    bias = torch.zeros(n_layers, r_max, dtype=torch.float64, device=dev)
    eye = torch.eye(d_s, dtype=torch.float64, device=dev)
    for l in s_layers:
        a, bb = A[l], Bm[l]
        if args.student_bias:
            mh, mz = sh[l] / n_train, sz[l] / n_train
            a = a - n_train * torch.outer(mh, mh)
            bb = bb - n_train * torch.outer(mh, mz)
        lam = args.ridge * torch.trace(a) / d_s
        W[l] = torch.linalg.solve(a + lam * eye, bb)
        if args.student_bias:
            bias[l] = mz - mh @ W[l]
    del A, Bm

    # --- ocena na val dla kazdej rangi
    ranks = [r for r in RANKS if r <= r_max]
    cos_sum = torch.zeros(n_layers, len(ranks), dtype=torch.float64, device=dev)
    sse = torch.zeros(n_layers, len(ranks), dtype=torch.float64, device=dev)
    sst = torch.zeros(n_layers, len(ranks), dtype=torch.float64, device=dev)
    n_val = 0
    for b, batch in enumerate(batches):
        m = split_mask(batch, "val")
        if not m.any():
            continue
        rows, z = batch_data(b, batch)
        for l in s_layers:
            zt = z[l][m].double()
            zh = rows[l][m].double() @ W[l] + bias[l]
            for j, r in enumerate(ranks):
                cos_sum[l, j] += F.cosine_similarity(zh[:, :r], zt[:, :r], dim=-1).sum()
                sse[l, j] += ((zh[:, :r] - zt[:, :r]) ** 2).sum()
                sst[l, j] += (zt[:, :r] ** 2).sum()  # z_T jest wycentrowane srednia train
        n_val += int(m.sum())

    cos = (cos_sum / max(n_val, 1)).cpu()
    r2 = (1 - sse / sst).cpu()
    phi = pca["phi"]
    evr = {}
    for t in pca["teacher_layers"]:
        ev = pca["eigvals"][t].clamp_min(0)
        evr[t] = [float(ev[:r].sum() / ev.sum()) for r in ranks]

    bridge = {
        "phi": phi, "r_max": r_max, "ranks": ranks,
        "P_T": [pca["P_T"][phi[l]] for l in s_layers],
        "mu_T": [pca["mean"][phi[l]] for l in s_layers],
        "P_S": [W[l].T.float().cpu().contiguous() for l in s_layers],  # (r_max, d_S)
        "b_S": [bias[l].float().cpu() for l in s_layers] if args.student_bias else None,
        "val_cosine": cos, "val_r2": r2,
        "meta": {k: v for k, v in vars(args).items() if k != "func"} | {"n_train_tokens": n_train, "n_val_tokens": n_val},
    }
    torch.save(bridge, work / "bridge.pt")
    write_report(work, bridge, evr, ranks)


def write_report(work, bridge, evr, ranks):
    cos, r2, phi = bridge["val_cosine"], bridge["val_r2"], bridge["phi"]
    lines = [
        "# OPRD-Bridge, Etap 1 - raport", "",
        f"Tokeny: train {bridge['meta']['n_train_tokens']}, val {bridge['meta']['n_val_tokens']}. "
        f"Uczen {bridge['meta']['student']}, ridge {bridge['meta']['ridge']}, bias P_S: {bridge['meta']['student_bias']}.", "",
        "## Sredni kosinus na val (po warstwach) vs ranga", "",
        "| r | " + " | ".join(map(str, ranks)) + " |",
        "|---|" + "---|" * len(ranks),
        "| kosinus | " + " | ".join(f"{float(c):.3f}" for c in cos.mean(0)) + " |",
        "| R^2 | " + " | ".join(f"{float(c):.3f}" for c in r2.mean(0)) + " |", "",
        "## Kosinus na val per warstwa ucznia", "",
        "| warstwa S | warstwa T | " + " | ".join(f"r={r}" for r in ranks) + " | wariancja T w top-8 |",
        "|---|---|" + "---|" * (len(ranks) + 1),
    ]
    i8 = ranks.index(8) if 8 in ranks else None
    for l in range(cos.shape[0]):
        ev = f"{evr[phi[l]][i8]:.3f}" if i8 is not None else "-"
        lines.append(f"| {l} | {phi[l]} | " + " | ".join(f"{float(c):.3f}" for c in cos[l]) + f" | {ev} |")
    (work / "report.md").write_text("\n".join(lines) + "\n")
    summary = {"ranks": ranks, "cosine_mean": cos.mean(0).tolist(), "r2_mean": r2.mean(0).tolist(),
               "cosine_per_layer": cos.tolist(), "phi": phi, "teacher_evr": {str(k): v for k, v in evr.items()}}
    (work / "report.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines[:9]))
    print(f"\nPelny raport: {work / 'report.md'}")


# ---------------------------------------------------------------- 4. kontrole

def cmd_controls(args):
    """Czy kosinus mostu mierzy wspolna strukture, a nie tylko to, ktory to token?

    Na val porownuje: most z bridge.pt (niezaleznie przeliczony), P_S dopasowane do
    z_T przemieszanych w obrebie batcha (powinno dac ~0), liniowa mape z embeddingu
    tokenu ucznia oraz tablice sredniego z_T dla id tokenu (sama tozsamosc tokenu).
    """
    work = Path(args.work)
    pca = torch.load(work / "teacher_pca.pt")
    check_plan(pca, args)
    bridge = torch.load(work / "bridge.pt")
    s_tok, t_tok, s_cfg = tokenizers(args, need_teacher=True)
    model = load_model(args.student, args.device, attn=args.attn)
    n_layers, d_s, dev = s_cfg.num_hidden_layers, s_cfg.hidden_size, args.device
    r = min(args.r_max, pca["r_max"])
    ranks = [k for k in RANKS if k <= r]
    examples = build_examples(args, s_tok, t_tok)
    batches = plan_batches(examples, args.batch_tokens)
    embed = model.get_input_embeddings()
    vocab = embed.weight.shape[0]
    gen = torch.Generator(device=dev).manual_seed(0)

    def load(b, batch, split):
        m = torch.cat([torch.full((len(e["pos"]),), e["split"] == split, dtype=torch.bool) for e in batch]).to(dev)
        if not m.any():
            return None
        saved = torch.load(work / "z_T" / f"batch_{b:05d}.pt")
        assert saved["idx"] == [e["idx"] for e in batch]
        rows = forward_rows(model, batch, "s_prompt", list(range(n_layers)), s_tok.pad_token_id or 0, dev)
        tok = torch.cat([torch.tensor([e["resp"][p] for p in e["pos"]]) for e in batch]).to(dev)
        z = saved["z"][:, :, :r].to(dev).float()
        return m, rows, tok, z

    # --- train: statystyki dla kontroli
    A = torch.zeros(n_layers, d_s, d_s, dtype=torch.float64, device=dev)
    B_shuf = torch.zeros(n_layers, d_s, r, dtype=torch.float64, device=dev)
    A_emb = torch.zeros(d_s, d_s, dtype=torch.float64, device=dev)
    B_emb = torch.zeros(n_layers, d_s, r, dtype=torch.float64, device=dev)
    tok_sum = torch.zeros(vocab, n_layers, r, device=dev)
    tok_cnt = torch.zeros(vocab, device=dev)
    for b, batch in enumerate(batches):
        got = load(b, batch, "train")
        if got is None:
            continue
        m, rows, tok, z = got
        tok, z = tok[m], z[:, m]
        perm = torch.randperm(tok.numel(), generator=gen, device=dev)
        e = embed(tok).float()
        A_emb += (e.T @ e).double()
        tok_sum.index_add_(0, tok, z.permute(1, 0, 2))
        tok_cnt.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        for l in range(n_layers):
            h = rows[l][m]
            A[l] += (h.T @ h).double()
            B_shuf[l] += (h.T @ z[l][perm]).double()
            B_emb[l] += (e.T @ z[l]).double()
        if b % 40 == 0:
            print(f"  train batch {b + 1}/{len(batches)}", flush=True)

    def ridge(a, bb):
        lam = args.ridge * torch.trace(a) / a.shape[0]
        return torch.linalg.solve(a + lam * torch.eye(a.shape[0], dtype=a.dtype, device=dev), bb)

    W_shuf = torch.stack([ridge(A[l], B_shuf[l]) for l in range(n_layers)])
    W_emb = torch.stack([ridge(A_emb, B_emb[l]) for l in range(n_layers)])
    tok_mean = tok_sum / tok_cnt.clamp_min(1)[:, None, None]
    seen = tok_cnt > 0
    W_real = torch.stack([bridge["P_S"][l][:r].T.to(dev).double() for l in range(n_layers)])
    del A, B_shuf, B_emb, tok_sum

    # --- val
    names = ["most", "przemieszane", "embedding", "srednia_tokenu"]
    cos_all = {k: [[[] for _ in ranks] for _ in range(n_layers)] for k in names}
    n_unseen = n_val = 0
    for b, batch in enumerate(batches):
        got = load(b, batch, "val")
        if got is None:
            continue
        m, rows, tok, z = got
        tok, z = tok[m], z[:, m]
        e = embed(tok).double()
        n_val += tok.numel()
        n_unseen += int((~seen[tok]).sum())
        for l in range(n_layers):
            zt = z[l].double()
            preds = {
                "most": rows[l][m].double() @ W_real[l],
                "przemieszane": rows[l][m].double() @ W_shuf[l],
                "embedding": e @ W_emb[l],
                "srednia_tokenu": tok_mean[tok, l].double(),
            }
            for k, zh in preds.items():
                for j, rr in enumerate(ranks):
                    cos_all[k][l][j].append(F.cosine_similarity(zh[:, :rr], zt[:, :rr], dim=-1).float().cpu())

    out = {"ranks": ranks, "n_val_tokens": n_val, "val_tokens_unseen_in_train": n_unseen, "kontrole": {}}
    lines = ["# OPRD-Bridge - kontrole kosinusa", "",
             f"Val: {n_val} tokenow ({n_unseen} z id nieobecnym w train - dla nich srednia_tokenu = 0).",
             "Sredni kosinus po warstwach (val); w nawiasie p10 / mediana per token.", "",
             "| predyktor | " + " | ".join(f"r={k}" for k in ranks) + " |", "|---|" + "---|" * len(ranks)]
    for k in names:
        per_rank = []
        for j in range(len(ranks)):
            c = torch.cat([torch.cat(cos_all[k][l][j]) for l in range(n_layers)])
            per_layer = [float(torch.cat(cos_all[k][l][j]).mean()) for l in range(n_layers)]
            per_rank.append({"mean": float(sum(per_layer) / n_layers), "p10": float(c.quantile(0.1)),
                             "median": float(c.median()), "per_layer": per_layer})
        out["kontrole"][k] = per_rank
        lines.append(f"| {k} | " + " | ".join(f"{p['mean']:.3f} ({p['p10']:.2f}/{p['median']:.2f})" for p in per_rank) + " |")
    j8 = ranks.index(8)
    lines += ["", "## r=8 per warstwa ucznia", "", "| warstwa | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for l in range(n_layers):
        lines.append(f"| {l} | " + " | ".join(f"{out['kontrole'][k][j8]['per_layer'][l]:.3f}" for k in names) + " |")
    (work / "controls.md").write_text("\n".join(lines) + "\n")
    (work / "controls.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    def common(p, teacher):
        p.add_argument("--student", required=True)
        p.add_argument("--teacher", required=True, help="W student-fit potrzebny tylko tokenizer.")
        p.add_argument("--teacher-template", choices=["own", "student"], default="own")
        if teacher:
            p.add_argument("--teacher-max-gpu-mem", help="Np. 12GiB: bf16 z offloadem nadmiarowych warstw na CPU (dokladne).")
            p.add_argument("--teacher-8bit", action="store_true", help="Nauczyciel w int8 (bitsandbytes) - zmienia stany, raczej offload.")
        p.add_argument("--rollouts", required=True)
        p.add_argument("--work", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--attn", default="sdpa")
        p.add_argument("--max-seq-len", type=int, default=4096)
        p.add_argument("--batch-tokens", type=int, default=8192, help="Padded tokeny na batch.")
        p.add_argument("--positions", choices=["all", "last_k"], default="all")
        p.add_argument("--last-k", type=int, default=2000)
        p.add_argument("--max-rollouts", type=int, default=0, help="Do szybkich testow.")

    p = sub.add_parser("teacher-pca")
    common(p, teacher=True)
    p.add_argument("--r-max", type=int, default=256)
    p.add_argument("--pca-rows", type=int, default=16384, help="Paper: <= 16 384 wierszy do kowariancji.")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_teacher_pca)

    p = sub.add_parser("teacher-project")
    common(p, teacher=True)
    p.set_defaults(func=cmd_teacher_project)

    p = sub.add_parser("student-fit")
    common(p, teacher=False)
    p.add_argument("--ridge", type=float, default=1e-4, help="lambda wzgledem sredniej wartosci wlasnej h_S^T h_S.")
    p.add_argument("--student-bias", action="store_true", help="P_S z wyrazem wolnym (paper: bez).")
    p.set_defaults(func=cmd_student_fit)

    p = sub.add_parser("controls")
    common(p, teacher=False)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--r-max", type=int, default=64, help="Rangi kontroli (tablica tokenow rosnie z r).")
    p.set_defaults(func=cmd_controls)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
