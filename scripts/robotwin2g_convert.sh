#!/usr/bin/env bash
# Official-style PointWorld data flow for RoboTwin two-gripper data:
#   raw RoboTwin HDF5 -> generated clip H5 -> integrity JSON -> split manifest -> WDS shards.
set -euo pipefail

RAW_ROOT=${1:?"Usage: robotwin2g_convert.sh RAW_ROOT WDS_ROOT [GENERATED_H5_ROOT]"}
WDS_ROOT=${2:?"Usage: robotwin2g_convert.sh RAW_ROOT WDS_ROOT [GENERATED_H5_ROOT]"}
GENERATED_H5_ROOT=${3:-"${WDS_ROOT%/}_generated_h5"}

TEST_RATIO=${TEST_RATIO:-0.02}
SPLIT_SCOPE=${SPLIT_SCOPE:-task}
SEED=${SEED:-42}
CLIP_HORIZON=${CLIP_HORIZON:-11}
CLIP_STRIDE=${CLIP_STRIDE:-5}
MAX_SCENE_POINTS=${MAX_SCENE_POINTS:-2048}
MAX_CLIPS_PER_EPISODE=${MAX_CLIPS_PER_EPISODE:--1}
MAX_SAMPLES_PER_SHARD=${MAX_SAMPLES_PER_SHARD:-512}
SCENE_FLOW_MODE=${SCENE_FLOW_MODE:-repeat_t0}
IMAGE_WIDTH=${IMAGE_WIDTH:-320}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-180}

EXTRA_H5_ARGS=()
if [[ "${POLICY_ROBOT_INPUT:-1}" == "1" || "${POLICY_ROBOT_INPUT:-true}" == "true" ]]; then
  EXTRA_H5_ARGS+=(--policy-robot-input)
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

mkdir -p "${GENERATED_H5_ROOT}" "${WDS_ROOT}"

python "${REPO_ROOT}/tools/robotwin2g/convert_robotwin_to_pointworld_h5.py" \
  --input-root "${RAW_ROOT}" \
  --output-root "${GENERATED_H5_ROOT}" \
  --overwrite \
  --seed "${SEED}" \
  --clip-horizon "${CLIP_HORIZON}" \
  --clip-stride "${CLIP_STRIDE}" \
  --max-scene-points "${MAX_SCENE_POINTS}" \
  --max-clips-per-episode "${MAX_CLIPS_PER_EPISODE}" \
  --scene-flow-mode "${SCENE_FLOW_MODE}" \
  --image-width "${IMAGE_WIDTH}" \
  --image-height "${IMAGE_HEIGHT}" \
  "${EXTRA_H5_ARGS[@]}"

python "${REPO_ROOT}/tools/robotwin2g/integrity_check_robotwin_h5.py" \
  --input-dir "${GENERATED_H5_ROOT}/flows" \
  --output "${GENERATED_H5_ROOT}/flows/integrity_check.json"

MANIFEST="${GENERATED_H5_ROOT}/flows/wds_manifest_seed${SEED}_test${TEST_RATIO}.json"
python "${REPO_ROOT}/tools/robotwin2g/make_robotwin_wds_manifest.py" \
  --input-dir "${GENERATED_H5_ROOT}/flows" \
  --integrity-check-file "${GENERATED_H5_ROOT}/flows/integrity_check.json" \
  --output-manifest "${MANIFEST}" \
  --test-ratio "${TEST_RATIO}" \
  --split-scope "${SPLIT_SCOPE}" \
  --seed "${SEED}"

python "${REPO_ROOT}/tools/robotwin2g/convert_robotwin_h5_to_wds.py" \
  --input-dir "${GENERATED_H5_ROOT}/flows" \
  --output-dir "${WDS_ROOT}" \
  --manifest "${MANIFEST}" \
  --max-samples-per-shard "${MAX_SAMPLES_PER_SHARD}" \
  --overwrite

printf '\nGenerated H5 root: %s\nWDS root: %s\nManifest: %s\n' "${GENERATED_H5_ROOT}" "${WDS_ROOT}" "${MANIFEST}"
