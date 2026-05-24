#!/bin/bash
#SBATCH --job-name=pw_r2g_viser_pred
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/work/runyi_yang/FloWAM
REPO=${ROOT}/code/PointWorld
WDS_ROOT=${WDS_ROOT:-${ROOT}/data/robotwin2g_wds}
PRETRAINED=${PRETRAINED:-${REPO}/pretrained_checkpoints/large-droid+behavior/model-best.pt}
CHECKPOINT=${CHECKPOINT:-${REPO}/train_logs/robotwin2g_all/checkpoint-best.pt}
PORT=${PORT:-8091}
SAMPLE_INDEX=${SAMPLE_INDEX:-0}

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd "${REPO}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${ROOT}/.cache"
export HF_HOME="${ROOT}/.cache/huggingface"
export TORCH_HOME="${ROOT}/.cache/torch"

python tools/robotwin2g/viser_prediction_viewer.py \
  --wds-root "${WDS_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --split test \
  --sample-index "${SAMPLE_INDEX}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-points 2048 \
  --line-point-stride 8 \
  --include-raw \
  --amp
