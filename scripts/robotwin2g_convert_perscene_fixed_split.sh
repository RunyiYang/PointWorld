#!/usr/bin/env bash
set -euo pipefail

# Convert one RoboTwin task/scene into PointWorld WDS with an explicit split:
# train episodes [0, 50), test episodes [50, 60).

SCENE=${1:?"Usage: robotwin2g_convert_perscene_fixed_split.sh SCENE [RAW_ROOT] [OUT_ROOT]"}
RAW_ROOT=${2:-${RAW_ROOT:-/work/runyi_yang/FloWAM/data/robotwin_sim}}
OUT_ROOT=${3:-${OUT_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM/FloWAM_PointWorld_PerScene}}

TRAIN_RANGE=${TRAIN_RANGE:-0:50}
TEST_RANGE=${TEST_RANGE:-50:60}
SELECT_END=${SELECT_END:-60}
SEED=${SEED:-42}
CLIP_HORIZON=${CLIP_HORIZON:-11}
CLIP_STRIDE=${CLIP_STRIDE:-5}
MAX_SCENE_POINTS=${MAX_SCENE_POINTS:-2048}
MAX_CLIPS_PER_EPISODE=${MAX_CLIPS_PER_EPISODE:--1}
MAX_SAMPLES_PER_SHARD=${MAX_SAMPLES_PER_SHARD:-512}
SCENE_FLOW_MODE=${SCENE_FLOW_MODE:-repeat_t0}
IMAGE_WIDTH=${IMAGE_WIDTH:-320}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-180}
POINTCLOUD_KEY=${POINTCLOUD_KEY:-/pointcloud_single_camera/head_camera}
HAND_QPOS_KEY=${HAND_QPOS_KEY:-/preprocessed/endpose_derived/gripper_open}
ENDPOSE_KEY=${ENDPOSE_KEY:-/endpose}
CAMERA_NAMES=${CAMERA_NAMES:-"head_camera front_camera"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

SCENE_RAW="${RAW_ROOT%/}/${SCENE}/demo_clean/data"
SCENE_ROOT="${OUT_ROOT%/}/${SCENE}"
SELECTED_ROOT="${SCENE_ROOT}/raw_selected"
SELECTED_TASK_DIR="${SELECTED_ROOT}/${SCENE}"
GENERATED_H5_ROOT="${SCENE_ROOT}/generated_h5"
WDS_ROOT="${SCENE_ROOT}/wds"

if [[ ! -d "${SCENE_RAW}" ]]; then
  echo "Missing raw scene data directory: ${SCENE_RAW}" >&2
  exit 2
fi

mkdir -p "${SELECTED_TASK_DIR}" "${SCENE_ROOT}"
find "${SELECTED_TASK_DIR}" -maxdepth 1 -type l -name 'episode*.hdf5' -delete

missing=()
selected=0
for ep in $(seq 0 $((SELECT_END - 1))); do
  src="${SCENE_RAW}/episode${ep}.hdf5"
  dst="${SELECTED_TASK_DIR}/episode${ep}.hdf5"
  if [[ -f "${src}" ]]; then
    ln -sfn "${src}" "${dst}"
    selected=$((selected + 1))
  else
    missing+=("${ep}")
  fi
done

python - <<PY
import json
from pathlib import Path
payload = {
    "scene": "${SCENE}",
    "raw_scene_data": "${SCENE_RAW}",
    "selected_root": "${SELECTED_ROOT}",
    "train_range": "${TRAIN_RANGE}",
    "test_range": "${TEST_RANGE}",
    "selected_episode_upper_bound_exclusive": ${SELECT_END},
    "selected_existing_episodes": ${selected},
    "missing_requested_episodes": [int(x) for x in "${missing[*]:-}".split()],
}
path = Path("${SCENE_ROOT}") / "selection_metadata.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
PY

if [[ ${selected} -eq 0 ]]; then
  echo "No episodes selected for ${SCENE}" >&2
  exit 3
fi

case "${GENERATED_H5_ROOT}" in
  "${SCENE_ROOT}/generated_h5") rm -rf "${GENERATED_H5_ROOT}" ;;
  *) echo "Refusing to remove unexpected generated H5 path: ${GENERATED_H5_ROOT}" >&2; exit 4 ;;
esac
mkdir -p "${GENERATED_H5_ROOT}" "${WDS_ROOT}"

read -r -a CAMERA_NAME_ARGS <<< "${CAMERA_NAMES}"
EXTRA_H5_ARGS=()
if [[ "${POLICY_ROBOT_INPUT:-1}" == "1" || "${POLICY_ROBOT_INPUT:-true}" == "true" ]]; then
  EXTRA_H5_ARGS+=(--policy-robot-input)
fi

python "${REPO_ROOT}/tools/robotwin2g/convert_robotwin_to_pointworld_h5.py" \
  --input-root "${SELECTED_ROOT}" \
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
  --pointcloud-key "${POINTCLOUD_KEY}" \
  --hand-qpos-key "${HAND_QPOS_KEY}" \
  --endpose-key "${ENDPOSE_KEY}" \
  --camera-names "${CAMERA_NAME_ARGS[@]}" \
  "${EXTRA_H5_ARGS[@]}"

python "${REPO_ROOT}/tools/robotwin2g/integrity_check_robotwin_h5.py" \
  --input-dir "${GENERATED_H5_ROOT}/flows" \
  --output "${GENERATED_H5_ROOT}/flows/integrity_check.json"

MANIFEST="${GENERATED_H5_ROOT}/flows/wds_manifest_${SCENE}_train${TRAIN_RANGE//:/-}_test${TEST_RANGE//:/-}.json"
python "${REPO_ROOT}/tools/robotwin2g/make_robotwin_wds_manifest_episode_ranges.py" \
  --input-dir "${GENERATED_H5_ROOT}/flows" \
  --integrity-check-file "${GENERATED_H5_ROOT}/flows/integrity_check.json" \
  --output-manifest "${MANIFEST}" \
  --train-range "${TRAIN_RANGE}" \
  --test-range "${TEST_RANGE}" \
  --task-key "${SCENE}"

python "${REPO_ROOT}/tools/robotwin2g/convert_robotwin_h5_to_wds.py" \
  --input-dir "${GENERATED_H5_ROOT}/flows" \
  --output-dir "${WDS_ROOT}" \
  --manifest "${MANIFEST}" \
  --max-samples-per-shard "${MAX_SAMPLES_PER_SHARD}" \
  --overwrite

python "${REPO_ROOT}/tools/robotwin2g/smoke_read_wds.py" \
  --wds-root "${WDS_ROOT}" \
  --split train \
  --batch-size 2

python "${REPO_ROOT}/tools/robotwin2g/smoke_read_wds.py" \
  --wds-root "${WDS_ROOT}" \
  --split test \
  --batch-size 2

printf '\nScene: %s\nGenerated H5 root: %s\nWDS root: %s\nManifest: %s\n' \
  "${SCENE}" "${GENERATED_H5_ROOT}" "${WDS_ROOT}" "${MANIFEST}"
