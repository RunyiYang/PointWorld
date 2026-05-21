#!/usr/bin/env bash
set -euo pipefail

# Run from the PointWorld repository root.
DATA_ROOT=${DATA_ROOT:-/work/runyi_yang/FloWAM/data}
WDS_ROOT=${WDS_ROOT:-/work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds}
INTEGRITY_JSON=${INTEGRITY_JSON:-${WDS_ROOT}/integrity_check.json}
TEST_PERCENTAGE=${TEST_PERCENTAGE:-0.02}
MANIFEST_JSON=${MANIFEST_JSON:-${WDS_ROOT}/wds_manifest_seed${SEED:-42}_test${TEST_PERCENTAGE}.json}

mkdir -p "${WDS_ROOT}"

python tools/robotwin2g/make_robotwin_wds_manifest.py \
  --input-dir "${DATA_ROOT}" \
  --integrity-check-file "${INTEGRITY_JSON}" \
  --output-manifest "${MANIFEST_JSON}" \
  --seed "${SEED:-42}" \
  --test-percentage "${TEST_PERCENTAGE}" \
  --split-unit "${SPLIT_UNIT:-episode}" \
  --split-scope "${SPLIT_SCOPE:-global}"
