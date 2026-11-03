#!/usr/bin/env python3
"""Rollouty ucznia (Poziomki) na SGLangu do budowy mostu OPRD-Bridge.

Prompt renderujemy lokalnie szablonem ucznia i wysylamy jako tokeny na
/v1/completions (jak chat/chat_poziomka.py). Paper probkuje z temperatura 1.0
bez obcinania rozkladu - to sa rollouty on-policy, nie odpowiedzi "do czytania",
wiec domyslnie nie uzywamy ustawien z czatu (t0,6 / min_p).

Adres serwera: --base-url, albo POZIOMKA_BASE_URL / SGLANG_BASE_URL (tez z .env
w chat/ lub w korzeniu repo).

Skrypt jest wznawialny: pomija id, ktore juz sa w --out.

Przyklad:
  python3 oprd_bridge/rollouts.py --prompts $OPRD/prompts-v8-2k.jsonl \
      --tokenizer $OPRD/poziomka-baza --out $OPRD/rollouts-poziomka-baza.jsonl
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request

from common import append_jsonl, read_jsonl, render_prompt_ids, wczytaj_dotenv

REPO = Path(__file__).resolve().parent.parent
THINKING = {"on": True, "off": False, "none": None}


def api(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--tokenizer", required=True, help="Katalog ucznia (tokenizer + szablon czatu).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--thinking", choices=list(THINKING), default="off",
                        help="enable_thinking dla szablonu (v13: off -> prefill <think></think>); none = nie przekazywac.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=1, help="Rollouty na prompt.")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wczytaj_dotenv(REPO / "chat" / ".env", REPO / ".env")
    base_url = args.base_url or os.environ.get("POZIOMKA_BASE_URL") or os.environ.get("SGLANG_BASE_URL")
    if not base_url:
        raise SystemExit("Brak adresu SGLanga: --base-url albo POZIOMKA_BASE_URL.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = api(base_url, "/v1/models")["data"][0]["id"]
    thinking = THINKING[args.thinking]

    done = set()
    if Path(args.out).exists():
        done = {(r["id"], r["sample"]) for r in read_jsonl(args.out)}
    jobs = []
    for row in read_jsonl(args.prompts):
        for s in range(args.samples):
            if (row["id"], s) not in done:
                jobs.append((row, s))
    print(f"SGLang {base_url}, model {model}: {len(jobs)} rollautow do zrobienia, {len(done)} gotowych")

    def run(row, sample):
        ids = render_prompt_ids(tokenizer, row["messages"], thinking)
        budget = min(args.max_tokens, args.context_length - len(ids) - 32)
        if budget <= 0:
            return None
        payload = {
            "model": model, "prompt": ids, "max_tokens": budget,
            "temperature": args.temperature, "top_p": args.top_p,
            "stop": ["<|im_end|>"], "seed": args.seed * 1000003 + sample,
        }
        choice = api(base_url, "/v1/completions", payload)["choices"][0]
        return {
            "id": row["id"], "sample": sample, "split": row["split"],
            "messages": row["messages"], "thinking": args.thinking,
            "response": choice["text"], "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": len(ids),
        }

    n_ok = n_skip = 0
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(run, row, s) for row, s in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result is None:
                n_skip += 1
            else:
                append_jsonl(args.out, [result])
                n_ok += 1
            if i % 100 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} (zapisane {n_ok}, za dlugie prompty {n_skip})", flush=True)


if __name__ == "__main__":
    main()
