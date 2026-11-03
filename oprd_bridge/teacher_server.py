#!/usr/bin/env python3
"""Nauczyciel OPRD-Bridge jako serwis: rzuty przez most + logproby tokenow odpowiedzi.

POST /score {"seqs": [{"ids": [...], "start": int}], "layers": [1, ..., 15], "rank": 8}
  -> {"results": [{"z": <base64 fp16 (len(layers), R, rank)>, "logp": [R floats]}]}
gdzie R = len(ids) - start (wszystkie tokeny odpowiedzi). z = P_T (h_T - mu_T) dla
warstwy nauczyciela phi(l) z bridge.pt; logp[j] = log p_T(ids[start+j] | ids[:start+j]).
Nauczyciel dostaje dokladnie te same tokeny co uczen (wspolny tokenizer i szablon).

Tryb --dump liczy to samo dla pliku rolloutow i zapisuje cache (testy offline
train.py --offline-cache bez SGLanga i serwera).

  python3 oprd_bridge/teacher_server.py --teacher $OPRD/instruktor-lotnictwa-7B-2-hf \
      --bridge $W/bridge.pt --port 30100
"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import numpy as np
import torch

from common import LayerCapture, load_model, read_jsonl, render_prompt_ids


class Teacher:
    def __init__(self, args):
        self.device = args.device
        if args.max_gpu_mem:
            from transformers import AutoModelForCausalLM

            idx = int(args.device.split(":")[-1])
            self.model = AutoModelForCausalLM.from_pretrained(
                args.teacher, dtype=torch.bfloat16, attn_implementation=args.attn,
                device_map="auto", max_memory={idx: args.max_gpu_mem, "cpu": "48GiB"},
            ).eval()
        else:
            self.model = load_model(args.teacher, args.device, attn=args.attn)
        bridge = torch.load(args.bridge)
        self.phi = bridge["phi"]
        self.P_T = [p.to(self.device) for p in bridge["P_T"]]  # (r_max, d_T) dla warstwy ucznia l
        self.mu_T = [m.to(self.device) for m in bridge["mu_T"]]
        self.batch_tokens = args.batch_tokens
        self.lock = threading.Lock()

    @torch.no_grad()
    def score(self, seqs, layers, rank):
        t_layers = sorted({self.phi[l] for l in layers})
        out = [None] * len(seqs)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["ids"]))
        batches, cur = [], []
        for i in order:  # batche po budzecie tokenow (z paddingiem)
            if cur and len(seqs[i]["ids"]) * (len(cur) + 1) > self.batch_tokens:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        decoder, head = self.model.get_decoder(), self.model.get_output_embeddings()
        for batch in batches:
            width = max(len(seqs[i]["ids"]) for i in batch)
            ids = torch.zeros((len(batch), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for b, i in enumerate(batch):
                s = seqs[i]["ids"]
                ids[b, : len(s)] = torch.tensor(s)
                mask[b, : len(s)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            with LayerCapture(self.model, t_layers) as cap:
                last = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            for b, i in enumerate(batch):
                start, end = seqs[i]["start"], len(seqs[i]["ids"])
                z = torch.stack([
                    (cap.outputs[self.phi[l]][b, start:end].to(self.device).float() - self.mu_T[l])
                    @ self.P_T[l][:rank].T for l in layers
                ])
                logits = head(last[b, start - 1: end - 1]).float()  # przy offloadzie accelerate sam przenosi wejscie
                logp = torch.log_softmax(logits, -1).gather(-1, ids[b, start:end, None].to(logits.device))[:, 0]
                out[i] = {"z": z.half().cpu().numpy(), "logp": logp.cpu().tolist()}
        return out


def encode(results):
    return [{"z": base64.b64encode(r["z"].tobytes()).decode(), "shape": list(r["z"].shape), "logp": r["logp"]}
            for r in results]


def decode(results):
    return [{"z": np.frombuffer(base64.b64decode(r["z"]), dtype=np.float16).reshape(r["shape"]), "logp": r["logp"]}
            for r in results]


def serve(teacher, args):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with teacher.lock:
                res = teacher.score(req["seqs"], req["layers"], req["rank"])
            body = json.dumps({"results": encode(res)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # health check
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    print(f"Nauczyciel gotowy na {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


def dump(teacher, args):
    """Cache dla train.py --offline-cache: rollouty (tokeny ucznia) + wynik nauczyciela."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    layers = [int(x) for x in args.layers.split(",")]
    rows = read_jsonl(args.dump)[: args.dump_limit or None]
    seqs = []
    for r in rows:
        prompt = render_prompt_ids(tok, r["messages"], False if r.get("thinking", "off") == "off" else None)
        resp = tok.encode(r["response"], add_special_tokens=False)
        if r.get("finish_reason") == "stop":
            resp.append(im_end)
        seqs.append({"ids": prompt + resp, "start": len(prompt), "s_logp": None})
    res = teacher.score(seqs, layers, args.rank)
    for s, r in zip(seqs, res):
        s["z"], s["t_logp"] = torch.from_numpy(r["z"].copy()), r["logp"]
    torch.save({"layers": layers, "rank": args.rank, "seqs": seqs}, args.dump_out)
    print(f"Zapisano {len(seqs)} sekwencji do {args.dump_out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True)
    p.add_argument("--bridge", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--max-gpu-mem", help="Np. 11GiB: offload nadmiarowych warstw na CPU (lokalnie na 16 GB).")
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30100)
    p.add_argument("--dump", help="Plik rolloutow - zamiast serwera zapisz cache i wyjdz.")
    p.add_argument("--dump-out")
    p.add_argument("--dump-limit", type=int, default=0)
    p.add_argument("--student", help="Tokenizer ucznia (dla --dump).")
    p.add_argument("--layers", default=",".join(map(str, range(1, 16))))
    p.add_argument("--rank", type=int, default=8)
    args = p.parse_args()
    teacher = Teacher(args)
    if args.dump:
        dump(teacher, args)
    else:
        serve(teacher, args)


if __name__ == "__main__":
    main()
