#!/usr/bin/env bash
set -euo pipefail

# Stage 1: freeze PointWorld, train only the 54-D FlowAM dexterous action head.
# Usage:
#   bash scripts/flowam_finetune_action_decoder.sh WDS_ROOT PRETRAINED OUT

WDS_ROOT=${1:-${WDS_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM_PointWorld}}
PRETRAINED=${2:-${PRETRAINED:-pretrained_checkpoints/large-droid+behavior/model-best.pt}}
OUT=${3:-${OUT:-train_logs/flowam_dexterous_action_decoder}}

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --output-dir "${OUT}" \
  --stage action_decoder \
  --action-dim 54 \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --num-epochs "${NUM_EPOCHS:-20}" \
  --lr "${LR:-1e-4}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --amp
