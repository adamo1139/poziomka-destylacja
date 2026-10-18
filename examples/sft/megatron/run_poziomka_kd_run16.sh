#!/usr/bin/env bash
set -euo pipefail

# Run 16: destylacja off-policy Poziomki po run 15 (SFT na polskie-sprawy-v8) od INSTRUKTORA
# (Bielik-PL-Minitron-7B po nauka-lotnictwa-2, W8A8 INT8). 100% reverse KL do top-10
# nauczyciela z kubelkiem ogona (poziomka_kd_loss.py), bez domieszki CE.
#
# Dane: ten sam cache co run 15 (poziomka-sft-cache-polskie-sprawy-v8-v13-16384, 1,312,556 rozmow
# train, 83,423 kubelkow), top-10 z polskie-dane-instruktazowe/destylacja/teacher_topk.py
# --backend sglang na tym cache: NLL instruktora 0.5179 (walidacja 0.5752), top-1 0.859,
# masa top-10 0.9814. KD na wszystkich tokenach poza BOS (jak loss run 15).
#
# Start: koncowy checkpoint run 15 (iter 5214), --finetune (liczniki od zera).
# Przepis jak run 15 (GBS 16, LR 1,03e-4, warmup 20, 1 epoka, walidacja i zapisy w tych samych
# odstepach), wiec jedyna roznica wobec dalszego SFT to loss.
#
# Wznowienie po przerwaniu: RESUME=1 bash run_poziomka_kd_run16.sh (ostatni checkpoint tego runu,
# bez optymalizatora i bez warmupu, jak w run 15).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# On rigga this resolves to ~/projects/pretrain, independently of the shell cwd.
WORK_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

# Patched Megatron (training/megatron/apply_patch.sh) is not tracked in git. Use this
# repo's copy if it has one, else Ling-V2's on the same machine (read only).
if [[ -z "${MEGATRON_PATH:-}" ]]; then
    MEGATRON_PATH="${REPO_DIR}/Megatron-LM-core_v0.13.0"
    [[ -f "${MEGATRON_PATH}/pretrain_gpt.py" ]] || MEGATRON_PATH="${WORK_DIR}/Ling-V2/Megatron-LM-core_v0.13.0"
fi
export MEGATRON_PATH
export TRAIN_SCRIPT=train_poziomka_kd.py
export SFT_DATA="/media/nvme_2tb/maked/poziomka_train/poziomka-sft-cache-polskie-sprawy-v8-v13-16384"
export TEACHER_TOPK="/media/nvme_2tb/maked/bielik_train/teacher-topk-polskie-sprawy-v8-k10"
RUN15="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run15_polskie_sprawy_v8_v13_16384_gbs16"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_kd_run16_polskie_sprawy_v8_rkl_k10_16384_gbs16"
export RESUME="${RESUME:-0}"
if [[ "${RESUME}" == 1 ]]; then
    export LOAD_CHECKPOINT="${SAVE_CHECKPOINT}"
else
    export LOAD_CHECKPOINT="${RUN15}"
fi
# Artefakty W&B na nvme: na / zabraklo miejsca w run 15 (crash po zapisie checkpointu).
export WANDB_DIR="${WANDB_DIR:-/media/nvme_2tb/wandb}"

export SEQ_LENGTH=16384
export MAX_POSITION_EMBEDDINGS=16384
export ROTARY_BASE=640000
# KD jest zawsze pakowane (train_poziomka_kd.py); --sft-packing nie istnieje w KD.
export PACKING=0
export GLOBAL_BATCH_SIZE=16
# Jak run 15: GBS 16 x 15,569 = 249,1k tokenow na krok -> 3e-4 * sqrt(0,119) = 1,03e-4.
export LR=1.03e-4
export WARMUP_ITERS=20
# Jedna epoka: 83,423 / 16 -> 5214.
export TRAIN_ITERS=5214
# Cala walidacja: 830 sekwencji, 52 x 16 = 832.
export EVAL_ITERS=52
export SAVE_INTERVAL=652
export EVAL_INTERVAL=326
export DATALOADER_WORKERS=2
export ASYNC_SAVE=0
export ROUTER_BIAS_UPDATE_RATE=0

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-destylacja"
export WANDB_NAME="poziomka_kd_run16_polskie_sprawy_v8_rkl_k10_16384_gbs16_lr${LR}_wu20_${TRAIN_ITERS}steps"
export WANDB_MODE="online"

[[ -f "${SFT_DATA}/manifest.json" ]] || { echo "No cache at ${SFT_DATA}" >&2; exit 1; }
[[ -f "${TEACHER_TOPK}/manifest.json" ]] || { echo "No teacher top-k at ${TEACHER_TOPK}" >&2; exit 1; }
if [[ "${RESUME}" != 1 ]]; then
    [[ "$(cat "${RUN15}/latest_checkpointed_iteration.txt")" == 5214 ]] || {
        echo "Expected run 15 final iteration 5214 in ${RUN15}" >&2; exit 1; }
fi

# Cache, teacher and this run's step count must agree before 8 GPUs start.
python3 - "${SFT_DATA}/manifest.json" "${TEACHER_TOPK}/manifest.json" "${SEQ_LENGTH}" <<'PY'
import hashlib, json, sys
cache_path, teacher_path, seq = sys.argv[1], sys.argv[2], int(sys.argv[3])
cache, teacher = json.load(open(cache_path)), json.load(open(teacher_path))
if cache["seq_length"] != seq:
    sys.exit(f"Cache is {cache['seq_length']} tokens, training at {seq}")
if cache["totals"]["train"]["records"] != 1312556:
    sys.exit(f"Expected 1,312,556 train records (polskie-sprawy-v8), found {cache['totals']['train']['records']:,}")
if teacher["cache_manifest_sha256"] != hashlib.sha256(open(cache_path, "rb").read()).hexdigest():
    sys.exit("Teacher top-k was computed for a different cache")
if teacher["k"] != 10:
    sys.exit(f"Expected top-10 teacher, found k={teacher['k']}")
print(f"Cache OK: {cache['totals']['train']['records']:,} train records; teacher {teacher['model']} "
      f"k={teacher['k']}, NLL {teacher['totals']['train']['teacher_nll']:.4f}")
PY

save_parent="$(dirname "${SAVE_CHECKPOINT}")"
free_gb=$(df -BG --output=avail "${save_parent}" | tail -1 | tr -dc '0-9')
needed_gb=$(( (TRAIN_ITERS / SAVE_INTERVAL + 2) * 8 ))
if (( free_gb < needed_gb )); then
    echo "Checkpoints need ~${needed_gb} GB, ${free_gb} GB free on ${save_parent}." >&2
    exit 1
fi

exec bash "${SCRIPT_DIR}/run_poziomka.sh" \
    --teacher-topk "${TEACHER_TOPK}" \
    --wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_NAME}" "$@"
