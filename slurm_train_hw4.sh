#!/bin/bash
# SLURM submission script for HW4 Image Restoration (PromptIR)
# Usage: sbatch slurm_train_hw4.sh

#SBATCH -J vr_hw4_ir
#SBATCH -o /home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw4/logs/slurm_%j.out
#SBATCH -e /home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw4/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=h200q

MODE="full"  # set to "smoke" for a quick test

set -euo pipefail

module purge
module load anaconda
module load slurm
module load nvidia-hpc
module list

eval "$(conda shell.bash hook)"
conda deactivate || true
conda activate Visual_Recognition

export PYTHONNOUSERSITE=1

PROJECT_DIR="/home/a00021/sherry890910.cs13/NYCU_Computer_Vision_2026_HW1/visual_recognition_hw4"
# dataset provided in hw4_realse_dataset/hw4_realse_dataset
DATA_ROOT="${PROJECT_DIR}/hw4_realse_dataset/hw4_realse_dataset"
OUTPUT_BASE="${PROJECT_DIR}/outputs"
RUN_DIR="${OUTPUT_BASE}/run_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_BASE}"
mkdir -p "${RUN_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

if [ "${MODE}" = "smoke" ]; then
    EPOCHS=2
    BATCH_SIZE=4
    PATCH=128
    BASE_CH=32
    NUM_PROMPTS=8
    PROMPT_DIM=64
    WARMUP=1
else
    EPOCHS=600
    BATCH_SIZE=8
    PATCH=256
    BASE_CH=64
    NUM_PROMPTS=16
    PROMPT_DIM=128
    WARMUP=20
fi

echo "Running MODE=${MODE} epochs=${EPOCHS} batch=${BATCH_SIZE} patch=${PATCH} base_ch=${BASE_CH}"

python -u train.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RUN_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --patch_size ${PATCH} \
    --num_workers 6 \
    --amp \
    --loss combined \
    --base_ch ${BASE_CH} \
    --num_prompts ${NUM_PROMPTS} \
    --prompt_dim ${PROMPT_DIM} \
    --warmup_epochs ${WARMUP:-10} \
    --sched cosine

# Run inference using best checkpoint
BEST_CKPT="${RUN_DIR}/checkpoints/best.pt"
if [ ! -f "${BEST_CKPT}" ]; then
    echo "best.pt not found, falling back to latest.pt"
    BEST_CKPT="${RUN_DIR}/checkpoints/latest.pt"
fi
python -u infer.py \
    --data_root "${DATA_ROOT}" \
    --checkpoint "${BEST_CKPT}" \
    --output_dir "${RUN_DIR}" \
    --batch_size 1 --tta \
    --base_ch ${BASE_CH} --num_prompts ${NUM_PROMPTS} --prompt_dim ${PROMPT_DIM}

# Make pred.npz and zip submission (use Python zipfile to avoid missing `zip` binary)
RESTORED="${RUN_DIR}/restored_images"
python3 scripts/make_submission.py --restored_dir "${RESTORED}" --out "${RUN_DIR}/pred.npz"
if [ -f "${RUN_DIR}/pred.npz" ]; then
    python3 - <<PYCODE
import zipfile, os
out = os.path.join('${RUN_DIR}', 'codabench_submission.zip')
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(os.path.join('${RUN_DIR}', 'pred.npz'), 'pred.npz')
print('submission zip created:', out)
PYCODE
else
    echo "[ERROR] pred.npz not found at ${RUN_DIR}/pred.npz — skipping zip"
    exit 1
fi
