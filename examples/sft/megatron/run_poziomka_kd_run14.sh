#!/usr/bin/env bash
set -euo pipefail

# Run 14: pierwsza destylacja off-policy UCZNIA (Poziomka po run 13) od INSTRUKTORA
# (Bielik-PL-Minitron-7B po nauka-lotnictwa-2). 100% reverse KL do top-40 nauczyciela
# z kubelkiem ogona (poziomka_kd_loss.py), bez domieszki CE.
#
# Dane: cpral/destylowanie-1 — 787,659 rozmow wygenerowanych przez instruktora, ktorych
# uczen nie widzial. Cache bielik-sft-cache-destylowanie-1-v13-8192 (polskie-dane-instruktazowe,
# build_nauka_lotnictwa_1_cache.sh, NAME=destylowanie-1, 8192, nic nie odrzucone), top-40
# z polskie-dane-instruktazowe/destylacja/teacher_topk.py na tym samym cache (bf16 HF + FA2):
# NLL instruktora 0.516, top-1 0.853, masa top-40 0.9845.
# KD na wszystkich tokenach poza BOS (jak loss w run 13 i u instruktora).
#
# Start: koncowy checkpoint run 13 (iter 1350; HF: cpral/uczen-lotnictwa-4B-2-hf).
# Reszta przepisu jak run 13: GBS 64, micro-batch 1, PP8, warmup 20, stale LR, bias routera
# zamrozony, RoPE 640000.

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
export SFT_DATA="${WORK_DIR}/bielik-sft-cache-destylowanie-1-v13-8192"
export TEACHER_TOPK="/media/nvme_2tb/maked/bielik_train/teacher-topk-destylowanie-1-k40"
export LOAD_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_sft_run13_nauka_lotnictwa_2_uczen_v13_6144_gbs64"
export SAVE_CHECKPOINT="/media/nvme_2tb/maked/poziomka_train/poziomka_kd_run14_destylowanie_1_rkl_k40_8192_gbs64"
export RESUME=0

export SEQ_LENGTH=8192
export MAX_POSITION_EMBEDDINGS=8192
export ROTARY_BASE=640000
# KD jest zawsze pakowane (train_poziomka_kd.py); --sft-packing nie istnieje w KD.
export PACKING=0
export GLOBAL_BATCH_SIZE=64
# Regula pierwiastkowa Poziomki jak w run 13: 40,603 kubelkow, pakowanie 96,6% ->
# 7,914 tokenow na sekwencje, przy GBS 64 to 506,5k tokenow na krok = 0,2415 pretreningu
# -> 3e-4 * sqrt(0,2415) = 1,47e-4. Regula byla ustalona dla CE; dla reverse KL to punkt startu.
export LR=1.47e-4
export WARMUP_ITERS=20
# Jedna epoka: 40,603 kubelkow / 64 = 634,4 -> 635.
export TRAIN_ITERS=635
# Zapisy na 127, 254, 381, 508 i koncowy 635; walidacja tak samo (~1/5 epoki).
export SAVE_INTERVAL=127
export EVAL_INTERVAL=127
# Walidacja: 3,261,502 pozycji / 7,914 na kubelek ~ 412 kubelkow; 6 x 64 = 384.
export EVAL_ITERS=6
export DATALOADER_WORKERS=2
export ASYNC_SAVE=0
export ROUTER_BIAS_UPDATE_RATE=0

export WANDB_ENTITY="adamo1139-no"
export WANDB_PROJECT="poziomka-destylacja"
export WANDB_NAME="poziomka_kd_run14_destylowanie_1_rkl_k40_8192_gbs64_lr${LR}_const_wu20_${TRAIN_ITERS}steps"
export WANDB_MODE="online"

[[ -f "${SFT_DATA}/manifest.json" ]] || { echo "No cache at ${SFT_DATA}" >&2; exit 1; }
[[ -f "${TEACHER_TOPK}/manifest.json" ]] || { echo "No teacher top-k at ${TEACHER_TOPK}" >&2; exit 1; }
[[ "$(cat "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt")" == 1350 ]] || {
    echo "Expected run 13 final iteration 1350 in ${LOAD_CHECKPOINT}" >&2; exit 1; }

# Cache, teacher and this run's step count must agree before 8 GPUs start.
python3 - "${SFT_DATA}/manifest.json" "${TEACHER_TOPK}/manifest.json" "${SEQ_LENGTH}" <<'PY'
import hashlib, json, sys
cache_path, teacher_path, seq = sys.argv[1], sys.argv[2], int(sys.argv[3])
cache, teacher = json.load(open(cache_path)), json.load(open(teacher_path))
if cache["seq_length"] != seq:
    sys.exit(f"Cache is {cache['seq_length']} tokens, training at {seq}")
if cache["totals"]["train"]["records"] != 779823:
    sys.exit(f"Expected 779,823 train records (destylowanie-1), found {cache['totals']['train']['records']:,}")
if cache["loss_roles"] != ["all"]:
    sys.exit(f"Expected loss on all tokens, got {cache['loss_roles']}")
if teacher["cache_manifest_sha256"] != hashlib.sha256(open(cache_path, "rb").read()).hexdigest():
    sys.exit("Teacher top-k was computed for a different cache")
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
