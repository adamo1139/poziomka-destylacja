#!/bin/bash
# Etap 2 OPRD-Bridge na riggu (8x 3090 Ti): GPU0 SGLang, GPU1 nauczyciel, GPU2-7 trening (FSDP, 6 rang).
# Wszystkie sciezki mozna nadpisac zmiennymi srodowiskowymi - domyslne sa z maszyny, na ktorej budowalismy most.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPRD=${OPRD:-/media/adamo/21c5549f-7ee2-497f-9c44-8fec44fce4ba/projects/OPRD}
STUDENT=${STUDENT:-$OPRD/poziomka_instruct_eksperyment_bieg15_5214.}
TEACHER=${TEACHER:-$OPRD/instruktor-lotnictwa-7B-2-hf}
BRIDGE=${BRIDGE:-$OPRD/bridge-instruktor7b-bieg15/bridge.pt}
TRAIN_PROMPTS=${TRAIN_PROMPTS:-$OPRD/prompts-v8-jakosc50-train-16k.jsonl}
EVAL_PROMPTS=${EVAL_PROMPTS:-$OPRD/prompts-v8-jakosc50-2k.jsonl}   # split=val (200), jak przy budowie mostu
OUT=${OUT:-$OPRD/stage2-instruktor7b-bieg15-run1}
SYNC_DIR=${SYNC_DIR:-$OUT/sync}           # 8 GB zapisywane co krok - najlepiej lokalny NVMe
SGLANG_PY=${SGLANG_PY:-$OPRD/sglang-venv/bin/python}
PY=${PY:-python3}                         # srodowisko z torch>=2.10 (F.grouped_mm na SM86) i transformers
CUDA_HOME=${CUDA_HOME_SGLANG:-/usr/local/cuda-12.8}
mkdir -p "$OUT"

cleanup() { kill ${SG_PID:-} ${T_PID:-} 2>/dev/null || true; }
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 CUDA_HOME=$CUDA_HOME PATH=$CUDA_HOME/bin:$PATH SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  "$SGLANG_PY" -m sglang.launch_server --model-path "$STUDENT" --trust-remote-code \
  --host 127.0.0.1 --port 30000 --context-length 8192 --mem-fraction-static 0.85 --dtype bfloat16 \
  > "$OUT/sglang.log" 2>&1 &
SG_PID=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$HERE/teacher_server.py" --teacher "$TEACHER" --bridge "$BRIDGE" \
  --port 30100 > "$OUT/teacher.log" 2>&1 &
T_PID=$!

until curl -sf -m 2 http://127.0.0.1:30000/v1/models >/dev/null; do kill -0 $SG_PID || { echo "SGLang padl, zob. $OUT/sglang.log"; exit 1; }; sleep 3; done
until curl -sf -m 2 http://127.0.0.1:30100/ >/dev/null; do kill -0 $T_PID || { echo "Nauczyciel padl, zob. $OUT/teacher.log"; exit 1; }; sleep 3; done
echo "SGLang i nauczyciel gotowi, start treningu $(date)"

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 "$PY" -m torch.distributed.run --nproc-per-node 6 "$HERE/train.py" \
  --student "$STUDENT" --bridge "$BRIDGE" --prompts "$TRAIN_PROMPTS" --eval-prompts "$EVAL_PROMPTS" \
  --out "$OUT" --sync-dir "$SYNC_DIR" "$@" 2>&1 | tee -a "$OUT/train.log"
