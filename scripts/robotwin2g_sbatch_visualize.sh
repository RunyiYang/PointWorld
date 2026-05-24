#!/bin/bash
#SBATCH --job-name=pw_r2g_viz
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
WDS_ROOT=${WDS_ROOT:-${ROOT}/data/robotwin2g_wds}
PRETRAINED=${PRETRAINED:-${REPO}/pretrained_checkpoints/large-droid+behavior/model-best.pt}
STAGE_NAME=${STAGE_NAME:-stage2_best}
CHECKPOINT=${CHECKPOINT:-${REPO}/train_logs/robotwin2g_all/checkpoint-best.pt}
OUT_DIR=${OUT_DIR:-${ROOT}/outputs/robotwin2g_flow_videos/${STAGE_NAME}}
NUM_SAMPLES=${NUM_SAMPLES:-6}
MAX_POINTS=${MAX_POINTS:-1024}
MAX_ROBOT_POINTS=${MAX_ROBOT_POINTS:-512}
FPS=${FPS:-2}
HOLD_FINAL=${HOLD_FINAL:-4}

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd "${REPO}"

test -d "${WDS_ROOT}"
test -f "${PRETRAINED}"
test -f "${CHECKPOINT}"

export PYTHONUNBUFFERED=1
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${ROOT}/.cache"
export HF_HOME="${ROOT}/.cache/huggingface"
export TORCH_HOME="${ROOT}/.cache/torch"

python tools/robotwin2g/render_flow_videos.py \
  --wds-root "${WDS_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --output-dir "${OUT_DIR}" \
  --split test \
  --num-samples "${NUM_SAMPLES}" \
  --max-points "${MAX_POINTS}" \
  --max-robot-points "${MAX_ROBOT_POINTS}" \
  --fps "${FPS}" \
  --hold-final "${HOLD_FINAL}" \
  --amp
