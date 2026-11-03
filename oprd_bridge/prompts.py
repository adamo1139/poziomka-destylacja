#!/usr/bin/env python3
"""Losuje prompty do OPRD-Bridge z polskie-sprawy-v8-jakosc50 (lokalny sft.jsonl).

Rozmowy sa wieloturowe, wiec prompt to historia ucieta przed wybrana tura
uzytkownika (odpowiedzi asystenta z historii zostaja takie, jak w zbiorze), a
odpowiedz na ostatnia ture wygeneruje uczen. Podzial train/val jest po wylosowaniu
- val sluzy do pomiaru kosinusa mostu na rozmowach, na ktorych P_S nie byl dopasowany.

Plik ma ~2,2 GB, wiec czytamy go strumieniowo: kazda rozmowa dostaje klucz z
hasha (seed, id) i zostaje n rozmow z najmniejszymi kluczami - losowanie jest
deterministyczne i nie trzyma zbioru w pamieci.

Przyklad:
  python3 oprd_bridge/prompts.py --out $OPRD/prompts-v8-jakosc50-2k.jsonl --n 2000
"""

import argparse
import hashlib
import heapq
import json
import random

from common import read_jsonl, write_jsonl

DOMYSLNY_ZBIOR = "/media/adamo/21c5549f-7ee2-497f-9c44-8fec44fce4ba/projects/OPRD/polskie-sprawy-v8-jakosc50/sft.jsonl"


def klucz(seed, conv_id):
    return int.from_bytes(hashlib.sha256(f"{seed}:{conv_id}".encode()).digest()[:8], "big")


def make_row(ex, args):
    """Prompt z jednej rozmowy albo None, gdy odpada na filtrach."""
    meta = ex.get("meta") or {}
    if args.thinking != "any" and meta.get("thinking") != args.thinking:
        return None
    messages = ex["messages"]
    user_turns = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if not user_turns:
        return None
    if args.turn == "first":
        cut = user_turns[0]
    elif args.turn == "last":
        cut = user_turns[-1]
    else:
        cut = random.Random(klucz(args.seed + 1, ex["id"])).choice(user_turns)
    prompt = [{"role": m["role"], "content": m["content"]} for m in messages[: cut + 1]]
    if sum(len(m["content"]) for m in prompt) > args.max_prompt_chars:
        return None
    return {
        "id": f"{ex['id']}:{cut}",
        "messages": prompt,
        "n_user_turns": sum(m["role"] == "user" for m in prompt),
        "topic": meta.get("topic"),
        "jakosc": (ex.get("jakosc") or {}).get("ocena"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DOMYSLNY_ZBIOR, help="Lokalny JSONL (id, messages, meta).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=2000, help="Liczba promptow (paper: 2K).")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking", choices=["off", "on", "any"], default="off",
                        help="Filtr po meta.thinking rozmowy zrodlowej.")
    parser.add_argument("--turn", choices=["random", "first", "last"], default="random",
                        help="Przed ktora tura uzytkownika ucinamy rozmowe.")
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    parser.add_argument("--exclude", action="append", default=[],
                        help="JSONL promptow, ktorych rozmow nie brac (np. zestaw mostu z val do KL).")
    args = parser.parse_args()
    excluded = {r["id"].split(":")[0] for path in args.exclude for r in read_jsonl(path)}

    heap = []  # (-klucz, id, wiersz): n rozmow z najmniejszymi kluczami
    seen = 0
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            seen += 1
            if ex["id"] in excluded:
                continue
            row = make_row(ex, args)
            if row is None:
                continue
            item = (-klucz(args.seed, ex["id"]), row["id"], row)
            if len(heap) < args.n:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    rows = [row for _, _, row in sorted(heap, reverse=True)]

    n_val = int(round(len(rows) * args.val_frac))
    for i, row in enumerate(rows):
        row["split"] = "val" if i < n_val else "train"
    write_jsonl(args.out, rows)
    print(f"Przejrzano {seen} rozmow (wykluczone {len(excluded)}); zapisano {len(rows)} promptow ({n_val} val) do {args.out}")


if __name__ == "__main__":
    main()
