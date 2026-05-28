#!/bin/bash
#SBATCH --job-name=pw_flowam_2stage
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
STAGE1_OUT=${STAGE1_OUT:-train_logs/flowam_dexterous_action_decoder}
STAGE2_OUT=${STAGE2_OUT:-train_logs/flowam_dexterous_lora}

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --output-dir "${STAGE1_OUT}" \
  --stage action_decoder \
  --action-dim 54 \
  --batch-size "${STAGE1_BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --num-epochs "${STAGE1_NUM_EPOCHS:-40}" \
  --max-steps "${STAGE1_MAX_STEPS:--1}" \
  --lr "${STAGE1_LR:-1e-4}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --action-metric-layout "${ACTION_METRIC_LAYOUT:-robot_flow_nn}" \
  --require-test \
  --amp

test -f "${STAGE1_OUT}/checkpoint-best.pt"

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --resume "${STAGE1_OUT}/checkpoint-best.pt" \
  --output-dir "${STAGE2_OUT}" \
  --stage lora \
  --reset-optimizer \
  --reset-progress \
  --action-dim 54 \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout "${LORA_DROPOUT:-0.0}" \
  --scene-loss-weight "${SCENE_LOSS_WEIGHT:-1.0}" \
  --scene-cd-max-points "${SCENE_CD_MAX_POINTS:-1024}" \
  --batch-size "${STAGE2_BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --num-epochs "${STAGE2_NUM_EPOCHS:-40}" \
  --max-steps "${STAGE2_MAX_STEPS:--1}" \
  --lr "${STAGE2_LR:-1e-4}" \
  --world-lr "${WORLD_LR:-2e-5}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --action-metric-layout "${ACTION_METRIC_LAYOUT:-robot_flow_nn}" \
  --require-test \
  --amp

test -f "${STAGE2_OUT}/checkpoint-last.pt"
test -f "${STAGE2_OUT}/checkpoint-best.pt"
