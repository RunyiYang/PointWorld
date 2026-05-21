#!/usr/bin/env bash
set -euo pipefail

# Stage 2: resume from stage 1, unfreeze PointWorld non-DINO modules + action decoder.
WDS_ROOT=${WDS_ROOT:-/work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds}
PRETRAINED=${PRETRAINED:-pretrained_checkpoints/large-droid+behavior/model-best.pt}
RESUME=${RESUME:-train_logs/robotwin2g_action_decoder/checkpoint-best.pt}
OUT=${OUT:-train_logs/robotwin2g_all}

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --resume "${RESUME}" \
  --output-dir "${OUT}" \
  --stage all \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --num-epochs "${NUM_EPOCHS:-20}" \
  --lr "${LR:-1e-4}" \
  --world-lr "${WORLD_LR:-2e-5}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --amp
