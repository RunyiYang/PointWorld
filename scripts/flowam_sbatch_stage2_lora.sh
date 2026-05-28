#!/bin/bash
#SBATCH --job-name=pw_flowam_s2_lora
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

cd /work/runyi_yang/FloWAM/code/PointWorld
mkdir -p /work/runyi_yang/FloWAM/logs/slurm train_logs

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WDS_ROOT=${WDS_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM_PointWorld}
PRETRAINED=${PRETRAINED:-pretrained_checkpoints/large-droid+behavior/model-best.pt}
RESUME=${RESUME:-train_logs/flowam_dexterous_action_decoder/checkpoint-best.pt}
OUT=${OUT:-train_logs/flowam_dexterous_lora}

test -f "${RESUME}"

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --resume "${RESUME}" \
  --output-dir "${OUT}" \
  --stage lora \
  --reset-optimizer \
  --reset-progress \
  --action-dim 54 \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout "${LORA_DROPOUT:-0.0}" \
  --scene-loss-weight "${SCENE_LOSS_WEIGHT:-1.0}" \
  --scene-cd-max-points "${SCENE_CD_MAX_POINTS:-1024}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --num-epochs "${NUM_EPOCHS:-40}" \
  --max-steps "${MAX_STEPS:--1}" \
  --lr "${LR:-1e-4}" \
  --world-lr "${WORLD_LR:-2e-5}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --action-metric-layout "${ACTION_METRIC_LAYOUT:-robot_flow_nn}" \
  --require-test \
  --amp

test -f "${OUT}/checkpoint-last.pt"
test -f "${OUT}/checkpoint-best.pt"
