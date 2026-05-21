#!/usr/bin/env bash
set -euo pipefail

# Run from the PointWorld repository root.
DATA_ROOT=${DATA_ROOT:-/work/runyi_yang/FloWAM/data}
WDS_ROOT=${WDS_ROOT:-/work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds}
INTEGRITY_JSON=${INTEGRITY_JSON:-${WDS_ROOT}/integrity_check.json}

mkdir -p "${WDS_ROOT}"

python tools/robotwin2g/robotwin_integrity_check.py \
  --input-dir "${DATA_ROOT}" \
  --output "${INTEGRITY_JSON}" \
  --clip-horizon "${CLIP_HORIZON:-11}" \
  --clip-stride "${CLIP_STRIDE:-5}" \
  --num-mp-workers "${NUM_MP_WORKERS:-0}" \
  --action-key "${ACTION_KEY:-/joint_action/vector}" \
  --action-dim "${ACTION_DIM:-14}" \
  --pointcloud-key "${POINTCLOUD_KEY:-/pointcloud}" \
  --camera-names ${CAMERA_NAMES:-cam0 cam1}
