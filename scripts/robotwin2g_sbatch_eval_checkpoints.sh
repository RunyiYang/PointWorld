#!/bin/bash
#SBATCH --job-name=pw_r2g_eval_ckpt
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/work/runyi_yang/FloWAM
REPO=${ROOT}/code/PointWorld
WDS_ROOT=${ROOT}/data/robotwin2g_wds
PRETRAINED=${REPO}/pretrained_checkpoints/large-droid+behavior/model-best.pt
OUT_JSON=${ROOT}/outputs/robotwin2g_eval/stage1_stage2_full_test.json

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd "${REPO}"
mkdir -p "$(dirname "${OUT_JSON}")"

export PYTHONUNBUFFERED=1
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${ROOT}/.cache"
export HF_HOME="${ROOT}/.cache/huggingface"
export TORCH_HOME="${ROOT}/.cache/torch"

python tools/robotwin2g/evaluate_action_checkpoint.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --output-json "${OUT_JSON}" \
  --split test \
  --batch-size 8 \
  --num-workers 4 \
  --max-batches 0 \
  --amp \
  --checkpoints \
    train_logs/robotwin2g_action_decoder/checkpoint-best.pt \
    train_logs/robotwin2g_all/checkpoint-best.pt \
    train_logs/robotwin2g_all/checkpoint-last.pt
