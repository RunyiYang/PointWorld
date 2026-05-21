#!/usr/bin/env bash
set -euo pipefail

# Stage 1: freeze PointWorld, train only the new action decoder.
# Usage:
#   bash scripts/robotwin2g_finetune_action_decoder.sh WDS_ROOT PRETRAINED OUT
WDS_ROOT=${1:-${WDS_ROOT:-/work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds}}
PRETRAINED=${2:-${PRETRAINED:-pretrained_checkpoints/large-droid+behavior/model-best.pt}}
OUT=${3:-${OUT:-train_logs/robotwin2g_action_decoder}}

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --output-dir "${OUT}" \
  --stage action_decoder \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --num-epochs "${NUM_EPOCHS:-20}" \
  --lr "${LR:-1e-4}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --amp
