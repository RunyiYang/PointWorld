#!/usr/bin/env bash
set -euo pipefail

# Run from the PointWorld repository root.
DATA_ROOT=${DATA_ROOT:-/work/runyi_yang/FloWAM/data}
WDS_ROOT=${WDS_ROOT:-/work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds}

python tools/robotwin2g/convert_robotwin_to_wds.py \
  --input-root "${DATA_ROOT}" \
  --output-root "${WDS_ROOT}" \
  --test-ratio 0.02 \
  --split-scope task \
  --clip-horizon 11 \
  --clip-stride 5 \
  --scene-flow-mode repeat_t0 \
  --camera-names cam0 cam1

python tools/robotwin2g/smoke_read_wds.py \
  --wds-root "${WDS_ROOT}" \
  --split train \
  --batch-size 2
