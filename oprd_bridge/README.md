# OPRD-Bridge: Bielik → Poziomka

Destylacja reprezentacji on-policy przez zamrożony most niskiego rzędu,
według [arXiv 2606.06021 **v4**](https://arxiv.org/abs/2606.06021v4) (par. 3.3, alg. 1).
Bielik i Poziomka mają wspólny tokenizer APT4, więc pozycje tokenów pokrywają się 1:1
i nie ma retokenizacji. Różni je tylko architektura (dense vs MoE, głębokość, szerokość).

Modele i wyniki leżą w `/media/adamo/21c5549f-7ee2-497f-9c44-8fec44fce4ba/projects/OPRD`.

## Etap 1: budowa mostu (oba modele zamrożone)

Kroki są pisane pod jedno GPU 16 GB, z jednym modelem na przebieg.

```bash
OPRD=/media/adamo/21c5549f-7ee2-497f-9c44-8fec44fce4ba/projects/OPRD
T=$OPRD/<bielik>; S=$OPRD/<poziomka>; W=$OPRD/bridge-<para>

# 1. prompty z polskie-sprawy-v8-jakosc50 (rozmowa ucięta przed losową turą użytkownika)
python3 prompts.py --out $OPRD/prompts-v8-jakosc50-2k.jsonl --n 2000   # domyślnie OPRD/polskie-sprawy-v8-jakosc50/sft.jsonl

# 2. rollouty ucznia na SGLangu (POZIOMKA_BASE_URL albo --base-url), t=1.0 jak w paperze
python3 rollouts.py --prompts $OPRD/prompts-v8-jakosc50-2k.jsonl --tokenizer $S --out $W/rollouts.jsonl

# 3. most
A="--teacher $T --student $S --rollouts $W/rollouts.jsonl --work $W"
python3 bridge.py teacher-pca $A        # mu_T, baza PCA (<= 16 384 wierszy / warstwę)
python3 bridge.py teacher-project $A    # z_T = P_T (h_T - mu_T) na dysk
python3 bridge.py student-fit $A        # P_S (ridge LS), kosinus na val dla każdej rangi
```

Wynikiem jest `bridge.pt`, czyli P_T, mu_T i P_S dla każdej warstwy ucznia w randze `r_max`.
Most rangi r to pierwsze r wierszy. Do tego `report.md`/`report.json` z krzywą kosinus vs ranga
(odpowiednik rys. 12 w paperze) i rozbiciem na warstwy.

Jeśli nauczyciel w bf16 nie mieści się na karcie, jest `--teacher-8bit`.

### Różnice względem paperu

- **P_S w postaci zamkniętej** (ridge) zamiast 20 epok Adama. Funkcja celu jest ta sama
  (równ. 13), ale dostajemy dokładne minimum i jedno rozwiązanie dla wszystkich rang naraz.
- **Stany z hooków na wyjściach bloków.** HF `hidden_states[-1]` ma już nałożony końcowy norm.
- **Prompt nauczyciela jego własnym szablonem** (`--teacher-template own`). Paper podaje obu
  modelom te same `input_ids`, co odpowiada opcji `student`. Tokeny odpowiedzi są identyczne
  w obu wariantach.
- **Nadzorowane są wszystkie tokeny odpowiedzi** (`--positions all`). Paper używa last-k=2000,
  ale tam chodziło o długie CoT matematyczne.

### Kontrole kosinusa

`python3 bridge.py controls <te same argumenty>` porównuje most z przemieszanymi parami,
liniową mapą z embeddingu tokenu i tablicą średniego `z_T` dla id tokenu. Wyniki trafiają do
`controls.md`. Dla pary instruktor 7B → bieg15 przy r=8: most 0.920, sama leksyka ~0.71.
Warstwa 0 jest czysto leksykalna.

## Etap 2: destylacja on-policy (`train.py`)

Krok wygląda tak:

1. Uczeń generuje rollouty na SGLangu (`/generate`, t=1.0, tokeny i logproby polityki).
2. `teacher_server.py` zwraca rzuty `P_T (h_T − mu_T)` i logproby tokenów odpowiedzi.
3. Wszystkie rangi FSDP liczą stratę z równ. 14–15 (L2-norm rzutów, MSE, warstwy 1–15) i robią krok AdamW.
4. Pełny state dict jest zapisywany jako safetensors w układzie HF, a SGLang wczytuje go przez `/update_weights_from_disk`.

Most, router MoE, końcowy norm i LM head są zamrożone. Metryka to KL(uczeń ‖ nauczyciel)
na rolloutach ucznia (k1, k3). Jest liczona na każdym kroku i co `--eval-every` na 200 promptach val.

`moe_fast.py` łączy 128 ekspertów w jeden tensor i liczy je przez grouped GEMM. Jest ok. 11.5×
szybszy w fwd+bwd niż remote code (test: `test_moe_fast.py`). Różnice są na poziomie bf16,
a eksport do kluczy per ekspert jest bezstratny.

Rig: `run_stage2_rig.sh [argumenty train.py]`. GPU0 to SGLang, GPU1 nauczyciel, GPU2–7 trening.
Lokalnie można przetestować samą mechanikę: `teacher_server.py --dump`, a potem
`train.py --offline-cache ... --train-layers 15 --no-sync`.
