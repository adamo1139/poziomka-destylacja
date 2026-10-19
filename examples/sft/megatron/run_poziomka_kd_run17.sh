#!/usr/bin/env bash
set -euo pipefail

# Run 17: destylacja off-policy Poziomki po run 15 od Bielika-PL-Minitron-7B-v3.0-Instruct
# (oryginalny Instruct, nie instruktor; W8A8 INT8). 100% reverse KL do top-10 z kubelkiem ogona.
#
# Run 16 (KD od instruktora, LR 2e-4) wyszedl gorzej niz run 15. Zmiany wzgledem run 16:
# - nauczyciel: Bielik v3 Instruct zamiast instruktora (ten napisal duza czesc v8, wiec jego
#   rozklad byl glownie jego wlasnym stylem);
# - dane: polskie-sprawy-v8 BEZ rozmow z rozumowaniem (v8-nothink; Bielik v3 nie rozumuje),
#   3 z 16 shardow train = 18,75% (subset_cache.py), walidacja cala;
# - loss tylko na odpowiedziach asystenta: cache z --loss-roles assistant (tury uzytkownika
#   i naglowki rol poza lossem - Bielik byl trenowany tylko na odpowiedziach) oraz
#   --kd-mask-think (blok "<think></think>\n" v13, ktorego Bielik nigdy nie widzial);
# LR, GBS, warmup i loss (reverse KL, top-10) jak w run 16.
# Benchmark nauczyciela (1000 rekordow, pozycje KD): NLL 0,6025, top-1 0,820, masa top-10 0,9835.
#
# Start: koncowy checkpoint run 15 (iter 5214), --finetune.
# TRAIN_ITERS (1 epoka) i EVAL_ITERS (cala walidacja) liczone z cache przed startem.
# Wznowienie: RESUME=1 bash run_poziomka_kd_run17.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

if [[ -z "${MEGATRON_PATH:-}" ]]; then
    MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
    [[ -f "${MEGATRON_PATH}/pretrain_gpt.py" ]] || MEGATRON_PATH="${WORK_DIR}/Ling-V2/Megatron-LM-core_v0.13.0"
fi
export MEGATRON_PATH
export TRAIN_SCRIPT=train_poziomka_kd.py
export SFT_DATA="/media/nvme_2tb/maked/poziomka_train/poziomka-sft-cache-polskie-sprawy-v8-nothink-19pct-v13-16384"
export TEACHER_TOPK="/media/nvme_2tb/maked/bielik_train/teacher-topk-bielik-v3-instruct-v8-nothink-19pct-k10"
RUN15="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run15_polskie_sprawy_v8_v13_16384_gbs16"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_kd_run17_bielik_v3_instruct_v8_nothink_19pct_rkl_k10_gbs16"
export RESUME="${RESUME:-0}"
if [[ "${RESUME}" == 1 ]]; then
    export LOAD_CHECKPOINT="${SAVE_CHECKPOINT}"
else
    export LOAD_CHECKPOINT="${RUN15}"
fi
# Staging i cache artefaktow W&B na nvme (katalog runu Megatron daje do <SAVE_CHECKPOINT>/wandb).
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/media/nvme_2tb/wandb/data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/media/nvme_2tb/wandb/cache}"
mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}"

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384
export ROTARY_BASE=640000
export PACKING=0   # KD jest zawsze pakowane
export GLOBAL_BATCH_SIZE=16
# Jak run 16 (stale 2e-4), zeby porownanie z run 16 dotyczylo nauczyciela i danych, nie LR.
export LR=2e-4
export WARMUP_ITERS=20
# Tylko koncowy checkpoint: interwal wiekszy niz trening, a Megatron i tak zapisuje na koncu
# (iteration % save_interval != 0). Bez posrednich zapisow nie ma od czego wznowic po awarii.
export SAVE_INTERVAL=1000000
export EVAL_INTERVAL=250
export DATALOADER_WORKERS=2
export ASYNC_SAVE=0
export ROUTER_BIAS_UPDATE_RATE=0

[[ -f "${SFT_DATA}/manifest.json" ]] || { echo "No cache at ${SFT_DATA}" >&2; exit 1; }
[[ -f "${TEACHER_TOPK}/manifest.json" ]] || { echo "No teacher top-k at ${TEACHER_TOPK}" >&2; exit 1; }
if [[ "${RESUME}" != 1 ]]; then
    [[ "$(cat "${RUN15}/latest_checkpointed_iteration.txt")" == 5214 ]] || {
        echo "Expected run 15 final iteration 5214 in ${RUN15}" >&2; exit 1; }
fi

# Sprawdzenie danych i nauczyciela; z cache liczymy 1 epoke i cala walidacje (kubelki pakowania
# sa deterministyczne, te same co w treningu).
# Podstawienie (nie read < <(...)): przy set -e blad sprawdzenia przerywa launcher.
ITERS="$(PYTHONPATH="${SCRIPT_DIR}" python3 - \
    "${SFT_DATA}/manifest.json" "${TEACHER_TOPK}/manifest.json" "${SEQ_LENGTH}" "${GLOBAL_BATCH_SIZE}" <<'PY'
import hashlib, json, math, sys
from poziomka_data import MMapSFTDataset
cache_path, teacher_path, seq, gbs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cache, teacher = json.load(open(cache_path)), json.load(open(teacher_path))
if cache["seq_length"] != seq:
    sys.exit(f"Cache is {cache['seq_length']} tokens, training at {seq}")
if cache["loss_roles"] != ["assistant"]:
    sys.exit(f"Expected assistant-only loss (v8-nothink), got {cache['loss_roles']}")
if cache["totals"]["train"]["records"] != 3 * 75303:
    sys.exit(f"Expected 225,909 train records (3 of 16 shards of v8-nothink), found {cache['totals']['train']['records']:,}")
if teacher["cache_manifest_sha256"] != hashlib.sha256(open(cache_path, "rb").read()).hexdigest():
    sys.exit("Teacher top-k was computed for a different cache")
if teacher["k"] != 10:
    sys.exit(f"Expected top-10 teacher, found k={teacher['k']}")
train = MMapSFTDataset(cache_path, "train", pack=True)
valid = MMapSFTDataset(cache_path, "validation", pack=True, shuffle=False)
iters, eval_iters = math.ceil(len(train.bins) / gbs), math.ceil(len(valid.bins) / gbs)
kd = teacher["totals"]["train"].get("kd_teacher_nll")
print(f"Cache OK: {cache['totals']['train']['records']:,} train records -> {len(train.bins):,} packed "
      f"sequences = {iters} steps at GBS {gbs}; validation {len(valid.bins)} sequences = {eval_iters} x {gbs}; "
      f"teacher {teacher['model']} k={teacher['k']}"
      + (f", NLL on KD positions {kd:.4f}" if kd is not None else ""), file=sys.stderr)
print(iters, eval_iters)
PY
)"
read -r TRAIN_ITERS EVAL_ITERS <<< "${ITERS}"
(( TRAIN_ITERS > 0 && EVAL_ITERS > 0 )) || { echo "Could not compute TRAIN_ITERS/EVAL_ITERS" >&2; exit 1; }
export TRAIN_ITERS EVAL_ITERS

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-destylacja"
export WANDB_NAME="poziomka_kd_run17_bielik_v3_instruct_v8_nothink_19pct_rkl_k10_gbs16_lr${LR}_wu20_${TRAIN_ITERS}steps"
export WANDB_MODE="online"

save_parent="$(dirname "${SAVE_CHECKPOINT}")"
free_gb=$(df -BG --output=avail "${save_parent}" | tail -1 | tr -dc '0-9')
needed_gb=$(( (TRAIN_ITERS / SAVE_INTERVAL + 2) * 8 ))
if (( free_gb < needed_gb )); then
    echo "Checkpoints need ~${needed_gb} GB, ${free_gb} GB free on ${save_parent}." >&2
    exit 1
fi

exec bash "${SCRIPT_DIR}/run_poziomka.sh" \
    --teacher-topk "${TEACHER_TOPK}" --kd-mask-think \
    --wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_NAME}" "$@"
