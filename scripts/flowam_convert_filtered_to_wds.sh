#!/usr/bin/env bash
set -euo pipefail

# Convert filtered FloWAM flow HDF5s into PointWorld-style WDS shards with
# 54-D dexterous action labels:
#   left_arm(7), right_arm(7), left_hand(20), right_hand(20)

INPUT_ROOT=${1:-${INPUT_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM/flow_data_filtered}}
OUTPUT_ROOT=${2:-${OUTPUT_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM_PointWorld}}
ACTION_ROOT=${3:-${ACTION_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM/origin}}

python tools/flowam/convert_flow_filtered_to_wds.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --action-root "${ACTION_ROOT}" \
  --overwrite \
  --keep-going \
  --test-ratio "${TEST_RATIO:-0.02}" \
  --split-scope "${SPLIT_SCOPE:-task}" \
  --seed "${SEED:-42}" \
  --clip-horizon "${CLIP_HORIZON:-11}" \
  --clip-stride "${CLIP_STRIDE:-5}" \
  --max-samples-per-shard "${MAX_SAMPLES_PER_SHARD:-512}" \
  --robot-groups "${ROBOT_GROUPS:-arm,eef,hand}" \
  --num-cameras "${NUM_CAMERAS:-2}"
