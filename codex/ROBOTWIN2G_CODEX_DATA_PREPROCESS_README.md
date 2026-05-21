# Codex README: run RoboTwin2G data preprocessing after the full PointWorld patch is already unzipped

This README is for Codex or another coding agent working inside the already-patched `NVlabs/PointWorld` repository.

The goal is **data preprocessing only**:

```text
raw RoboTwin HDF5 episodes
  -> PointWorld-like generated clip H5
  -> integrity_check.json
  -> episode-level train/test WDS manifest
  -> WDS train/test tar shards
```

Do **not** start action-decoder training or full-model fine-tuning in this step.

---

## 0. Preconditions

From the PointWorld repository root, these files should already exist because the full training patch was already unzipped:

```bash
ls tools/robotwin2g/convert_robotwin_to_pointworld_h5.py
ls tools/robotwin2g/integrity_check_robotwin_h5.py
ls tools/robotwin2g/make_robotwin_wds_manifest.py
ls tools/robotwin2g/convert_robotwin_h5_to_wds.py
ls scripts/robotwin2g_convert.sh
```

If any of these files are missing, stop and report that the full patch was not applied correctly. Do not reimplement the full training code unless explicitly requested.

Set `PYTHONPATH` from the repo root before running scripts:

```bash
cd /path/to/PointWorld
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Use the Python/conda environment that can import at least:

```text
h5py
numpy
cv2 or opencv-python
PIL or pillow
torch, only needed for the optional WDS smoke reader
```

The preprocessing scripts write WDS shards with Python tar utilities, so `webdataset` is not required for conversion.

---

## 1. Important data-root rule

The preprocessing script recursively scans `*.hdf5` and `*.h5` under `RAW_ROOT`.

Therefore, **do not pass a mixed data root** that also contains PointWorld-BEHAVIOR, generated outputs, old samples, or unrelated HDF5 files. Use a clean `RAW_ROOT` containing only RoboTwin task folders like this:

```text
RAW_ROOT/
  bartending/
    episode0.hdf5
    episode1.hdf5
    ...
  dough_rolling/
    episode0.hdf5
    ...
  handover/
    episode0.hdf5
    ...
  pick_and_place/
    episode0.hdf5
    ...
  rope_folding/
    episode0.hdf5
    ...
```

If the local workspace is `/work/runyi_yang/FloWAM/data` and that folder contains other datasets, create a clean root with symlinked episode files:

```bash
RAW_SOURCE=/work/runyi_yang/FloWAM/data
RAW_ROOT=/work/runyi_yang/FloWAM/robotwin2g_raw_tasks_only

rm -rf "$RAW_ROOT"
mkdir -p "$RAW_ROOT"

for task in bartending dough_rolling handover pick_and_place rope_folding; do
  mkdir -p "$RAW_ROOT/$task"
  find "$RAW_SOURCE/$task" -maxdepth 1 -type f -name 'episode*.hdf5' -print0 \
    | sort -z \
    | while IFS= read -r -d '' f; do
        ln -s "$f" "$RAW_ROOT/$task/$(basename "$f")"
      done
done

find "$RAW_ROOT" -maxdepth 2 -name 'episode*.hdf5' | head
```

Some `rope_folding` HDF5 files may be unreadable in the local scan. The converter should skip unreadable HDF5 files and log them instead of crashing the whole run.

---

## 2. Smoke test on the compact RoboTwin sample

Run this first before full preprocessing.

```bash
cd /path/to/PointWorld
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SMALL_RAW_ROOT=/work/runyi_yang/FloWAM/data/robotwin_small_sample
DEBUG_H5_ROOT=/tmp/robotwin2g_h5_debug
DEBUG_WDS_ROOT=/tmp/robotwin2g_wds_debug

rm -rf "$DEBUG_H5_ROOT" "$DEBUG_WDS_ROOT"

TEST_RATIO=0.34 \
MAX_CLIPS_PER_EPISODE=2 \
MAX_SCENE_POINTS=128 \
MAX_SAMPLES_PER_SHARD=16 \
IMAGE_WIDTH=160 \
IMAGE_HEIGHT=120 \
POLICY_ROBOT_INPUT=1 \
SCENE_FLOW_MODE=repeat_t0 \
bash scripts/robotwin2g_convert.sh \
  "$SMALL_RAW_ROOT" \
  "$DEBUG_WDS_ROOT" \
  "$DEBUG_H5_ROOT"
```

Why `TEST_RATIO=0.34` for the smoke test: the compact sample has only a few episodes per task, so this verifies that both train and test paths can be created. For the real full-data run, use `TEST_RATIO=0.02`.

Expected files after the smoke test:

```bash
ls "$DEBUG_H5_ROOT/flows/integrity_check.json"
ls "$DEBUG_H5_ROOT/flows"/wds_manifest_seed42_test0.34.json
ls "$DEBUG_WDS_ROOT/action_stats.json"
ls "$DEBUG_WDS_ROOT/metadata_rank0.json"
find "$DEBUG_WDS_ROOT" -name '*.tar' | sort | head
```

Check the integrity summary:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/robotwin2g_h5_debug/flows/integrity_check.json')
data = json.loads(p.read_text())
print('num_files:', data.get('num_files'))
print('num_valid_clips:', data.get('num_valid_clips'))
print('num_invalid_items:', data.get('num_invalid_items'))
if data.get('num_valid_clips', 0) <= 0:
    raise SystemExit('No valid clips generated')
PY
```

Check the manifest split:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/robotwin2g_h5_debug/flows/wds_manifest_seed42_test0.34.json')
m = json.loads(p.read_text())
print(json.dumps(m['stats'], indent=2, sort_keys=True))
assert m['stats']['num_train_clips'] > 0
assert m['stats']['num_clips'] == m['stats']['num_train_clips'] + m['stats']['num_test_clips']
PY
```

Check one WDS tar manually without importing the training dataset:

```bash
python - <<'PY'
import io
import tarfile
from pathlib import Path

import numpy as np

root = Path('/tmp/robotwin2g_wds_debug')
tars = sorted((root / 'train').glob('*.tar'))
assert tars, 'No train tar found'

with tarfile.open(tars[0], 'r') as tf:
    names = tf.getnames()
    print('tar:', tars[0])
    print('first files:', names[:12])
    sample_prefix = names[0].split('.', 1)[0]
    needed = [
        'scene_flows.npy',
        'robot_flows.npy',
        'action_state.npy',
        'action_target.npy',
        'action_mask.npy',
        'metadata.json',
    ]
    for suffix in needed:
        member_name = f'{sample_prefix}.{suffix}'
        assert member_name in names, f'missing {member_name}'
    def load_npy(suffix):
        f = tf.extractfile(f'{sample_prefix}.{suffix}')
        assert f is not None
        return np.load(io.BytesIO(f.read()))
    print('scene_flows:', load_npy('scene_flows.npy').shape)
    print('robot_flows:', load_npy('robot_flows.npy').shape)
    print('action_state:', load_npy('action_state.npy').shape)
    print('action_target:', load_npy('action_target.npy').shape)
    print('action_mask:', load_npy('action_mask.npy').shape)
PY
```

Expected shapes for the smoke test are approximately:

```text
scene_flows:   (11, 128, 3)
robot_flows:   (11, Nr, 3)
action_state:  (11, 14)
action_target: (10, 14)
action_mask:   (10,)
```

Optional repo-level smoke reader:

```bash
python tools/robotwin2g/smoke_read_wds.py \
  --wds-root "$DEBUG_WDS_ROOT" \
  --split train \
  --batch-size 2
```

If this optional reader fails with a PointWorld import error such as `dataset_components`, do not treat that as a preprocessing failure. Use the manual tarfile check above as the conversion smoke test.

---

## 3. Full data preprocessing run

After the smoke test passes, run the real conversion with the requested 2% test split.

Recommended output locations outside the git repo:

```bash
RAW_ROOT=/work/runyi_yang/FloWAM/robotwin2g_raw_tasks_only
H5_ROOT=/work/runyi_yang/FloWAM/data/robotwin2g_generated_h5
WDS_ROOT=/work/runyi_yang/FloWAM/data/robotwin2g_wds
```

Run:

```bash
cd /path/to/PointWorld
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

rm -rf "$H5_ROOT" "$WDS_ROOT"

TEST_RATIO=0.02 \
SPLIT_SCOPE=task \
SEED=42 \
CLIP_HORIZON=11 \
CLIP_STRIDE=5 \
MAX_SCENE_POINTS=2048 \
MAX_CLIPS_PER_EPISODE=-1 \
MAX_SAMPLES_PER_SHARD=512 \
IMAGE_WIDTH=320 \
IMAGE_HEIGHT=180 \
POLICY_ROBOT_INPUT=1 \
SCENE_FLOW_MODE=repeat_t0 \
bash scripts/robotwin2g_convert.sh \
  "$RAW_ROOT" \
  "$WDS_ROOT" \
  "$H5_ROOT"
```

Expected final outputs:

```text
$H5_ROOT/
  task_map.json
  generated_h5_manifest.jsonl
  metadata.json
  flows/
    task-0000/episode_00000000.hdf5
    task-0001/episode_00000000.hdf5
    integrity_check.json
    wds_manifest_seed42_test0.02.json

$WDS_ROOT/
  train/train-000000.tar
  train/train-000001.tar
  ...
  test/test-000000.tar
  ...
  action_stats.json
  metadata_rank0.json
  train_sources.txt
  test_sources.txt
```

---

## 4. Full-run validation commands

Integrity summary:

```bash
python - <<'PY'
import json, os
from pathlib import Path
h5_root = Path(os.environ['H5_ROOT'])
p = h5_root / 'flows' / 'integrity_check.json'
data = json.loads(p.read_text())
print(json.dumps({
    'num_files': data.get('num_files'),
    'num_valid_clips': data.get('num_valid_clips'),
    'num_invalid_items': data.get('num_invalid_items'),
}, indent=2))
assert data.get('num_valid_clips', 0) > 0
PY
```

Manifest summary:

```bash
python - <<'PY'
import json, os
from pathlib import Path
h5_root = Path(os.environ['H5_ROOT'])
p = h5_root / 'flows' / 'wds_manifest_seed42_test0.02.json'
m = json.loads(p.read_text())
print(json.dumps(m['stats'], indent=2, sort_keys=True))
assert m['stats']['num_train_clips'] > 0
# For the full dataset, this should normally be > 0.
assert m['stats']['num_test_clips'] > 0
PY
```

WDS metadata summary:

```bash
python - <<'PY'
import json, os
from pathlib import Path
wds_root = Path(os.environ['WDS_ROOT'])
meta = json.loads((wds_root / 'metadata_rank0.json').read_text())
stats = json.loads((wds_root / 'action_stats.json').read_text())
print('processed train:', meta['train']['processed_count'])
print('processed test:', meta['test']['processed_count'])
print('errors:', len(meta.get('errors', [])))
print('action count:', stats['count'])
print('action mean dim:', len(stats['mean']))
print('action std dim:', len(stats['std']))
assert meta['train']['processed_count'] > 0
assert meta['test']['processed_count'] > 0
assert stats['count'] > 0
assert len(stats['mean']) == 14
assert len(stats['std']) == 14
PY
```

Manual WDS tar check:

```bash
python - <<'PY'
import io, os, tarfile
from pathlib import Path
import numpy as np

wds_root = Path(os.environ['WDS_ROOT'])
for split in ['train', 'test']:
    tars = sorted((wds_root / split).glob('*.tar'))
    print(split, 'num_tars:', len(tars))
    assert tars, f'No {split} tar files'
    with tarfile.open(tars[0], 'r') as tf:
        names = tf.getnames()
        prefix = names[0].split('.', 1)[0]
        def arr(name):
            f = tf.extractfile(f'{prefix}.{name}')
            assert f is not None, name
            return np.load(io.BytesIO(f.read()))
        print(split, 'scene_flows', arr('scene_flows.npy').shape)
        print(split, 'robot_flows', arr('robot_flows.npy').shape)
        print(split, 'action_state', arr('action_state.npy').shape)
        print(split, 'action_target', arr('action_target.npy').shape)
PY
```

---

## 5. What the preprocessing is supposed to do

The converter slices every RoboTwin full episode into overlapping clips:

```text
clip 0: frames 0:11
clip 1: frames 5:16
clip 2: frames 10:21
...
```

Default clip parameters:

```text
CLIP_HORIZON=11
CLIP_STRIDE=5
```

Action labels are preserved for two-gripper action fine-tuning:

```text
action_state[t]  = /joint_action/vector[start + t]
action_target[t] = /joint_action/vector[start + t + 1]
```

So one 11-frame clip gives:

```text
action_state:  (11, 14)
action_target: (10, 14)
action_mask:   (10,)
```

Train/test split must be episode-level, not clip-level. Adjacent overlapping clips from the same episode must never appear in both train and test.

Default real split:

```text
TEST_RATIO=0.02
SPLIT_SCOPE=task
SEED=42
```

The default scene handling is intentionally conservative:

```text
SCENE_FLOW_MODE=repeat_t0
```

This repeats the initial observed scene points through the clip. It avoids fabricating true scene-flow supervision because RoboTwin observed point clouds do not guarantee persistent point identity across frames.

For policy learning, keep:

```text
POLICY_ROBOT_INPUT=1
```

This repeats the `t=0` gripper surrogate geometry through the clip and avoids leaking future robot trajectories into the action target.

---

## 6. Do not commit generated data

Generated H5 files and WDS tar shards are large data artifacts. Do not add them to git.

Codex may commit code-only fixes if it had to patch minor bugs, but it should not commit:

```text
*.hdf5
*.h5
*.tar
robotwin2g_wds/
robotwin2g_generated_h5/
```

Recommended git check:

```bash
git status --short
```

Expected code-only changes should be limited to files under:

```text
tools/robotwin2g/
scripts/robotwin2g_*.sh
ROBOTWIN2G_README.md
```

---

## 7. Common failure modes

### `No readable HDF5 episodes found`

`RAW_ROOT` is wrong or does not contain `episode*.hdf5`. Check:

```bash
find "$RAW_ROOT" -maxdepth 2 -name 'episode*.hdf5' | head
```

### Mixed PointWorld/RoboTwin data accidentally scanned

If logs show files from `pointworld_behavior_restored`, `robotwin2g_generated_h5`, or another generated folder, the wrong `RAW_ROOT` was used. Create a clean root with only RoboTwin task folders.

### Test split empty

This is acceptable for a one-episode debug run. It should not be acceptable for the full dataset unless the full dataset really has only one episode per split scope. For the full run with the known local task counts, `num_test_clips` should be greater than zero.

### Optional `smoke_read_wds.py` import failure

Use the manual tarfile check. The conversion itself does not depend on the full PointWorld training import stack.

### Very slow or high memory usage

Reduce debug parameters first:

```bash
MAX_CLIPS_PER_EPISODE=2 MAX_SCENE_POINTS=128 MAX_SAMPLES_PER_SHARD=16
```

For the real run, restore:

```bash
MAX_CLIPS_PER_EPISODE=-1 MAX_SCENE_POINTS=2048 MAX_SAMPLES_PER_SHARD=512
```

---

## 8. Copy-paste prompt for Codex

Use this prompt if handing the task to Codex:

```text
You are inside an already-patched NVlabs/PointWorld repo. The full RoboTwin2G training patch is already unzipped. Do not reapply the patch and do not start training.

Your task is to run and verify the data preprocessing only:
raw RoboTwin HDF5 episodes -> generated PointWorld-like clip H5 -> integrity_check.json -> episode-level train/test WDS manifest -> WDS tar shards.

First verify that scripts/robotwin2g_convert.sh and tools/robotwin2g/{convert_robotwin_to_pointworld_h5.py,integrity_check_robotwin_h5.py,make_robotwin_wds_manifest.py,convert_robotwin_h5_to_wds.py} exist. Then run the compact sample smoke test from ROBOTWIN2G_CODEX_DATA_PREPROCESS_README.md. After it passes, create/use a clean RAW_ROOT containing only these RoboTwin task folders: bartending, dough_rolling, handover, pick_and_place, rope_folding. Do not pass a mixed /work/runyi_yang/FloWAM/data root directly if it also contains PointWorld or generated HDF5 files.

Run the full conversion with TEST_RATIO=0.02, SPLIT_SCOPE=task, CLIP_HORIZON=11, CLIP_STRIDE=5, MAX_SCENE_POINTS=2048, SCENE_FLOW_MODE=repeat_t0, POLICY_ROBOT_INPUT=1. Keep generated H5 and WDS outputs outside the git repo or do not add them to git. Validate integrity_check.json, manifest stats, action_stats.json, metadata_rank0.json, and inspect one train/test tar manually. Report the final H5 root, WDS root, number of train/test clips, train/test tar count, action_stats count, and any skipped/bad episodes.
```
