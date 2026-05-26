#!/usr/bin/env bash
set -euo pipefail

# Stage 2: resume from stage 1, train LoRA adapters in PointWorld plus the
# 54-D FlowAM dexterous action head.  Scene loss is enabled so the PointWorld
# dynamics head receives adaptation signal.
# Usage:
#   bash scripts/flowam_finetune_all.sh WDS_ROOT RESUME OUT [PRETRAINED]

WDS_ROOT=${1:-${WDS_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM_PointWorld}}
RESUME=${2:-${RESUME:-train_logs/flowam_dexterous_action_decoder/checkpoint-best.pt}}
OUT=${3:-${OUT:-train_logs/flowam_dexterous_lora}}
PRETRAINED=${4:-${PRETRAINED:-pretrained_checkpoints/large-droid+behavior/model-best.pt}}

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --resume "${RESUME}" \
  --output-dir "${OUT}" \
  --stage lora \
  --reset-optimizer \
  --action-dim 54 \
  --lora-rank "${LORA_RANK:-8}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  --lora-dropout "${LORA_DROPOUT:-0.0}" \
  --scene-loss-weight "${SCENE_LOSS_WEIGHT:-1.0}" \
  --scene-cd-max-points "${SCENE_CD_MAX_POINTS:-1024}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --num-epochs "${NUM_EPOCHS:-20}" \
  --lr "${LR:-1e-4}" \
  --world-lr "${WORLD_LR:-2e-5}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --amp
