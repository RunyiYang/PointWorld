# RoboTwin Small Sample Data

This note documents the current RoboTwin HDF5 data in this workspace and the compact sample generated from it.

## Generated Sample

Small sample root:

```bash
/work/runyi_yang/FloWAM/data/robotwin_small_sample
```

Contents:

```text
robotwin_small_sample/
  manifest.json
  bartending/episode0.hdf5
  bartending/episode1.hdf5
  bartending/episode2.hdf5
  dough_rolling/episode0.hdf5
  dough_rolling/episode1.hdf5
  dough_rolling/episode2.hdf5
  handover/episode0.hdf5
  handover/episode1.hdf5
  handover/episode2.hdf5
  pick_and_place/episode0.hdf5
  pick_and_place/episode1.hdf5
  pick_and_place/episode2.hdf5
  rope_folding/episode0.hdf5
  rope_folding/episode1.hdf5
  rope_folding/episode2.hdf5
```

Current size is about `4.7M`. Each task has 3 compact episodes. Each compact episode keeps the first 16 frames, downsamples point clouds to 128 points, resizes RGB to `160x120`, omits depth by default, and gzip-compresses numeric arrays.

The script that generated it is:

```bash
/work/runyi_yang/FloWAM/code/RoboTwin/script/make_robotwin_small_sample.py
```

Generation command:

```bash
source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate flowam-act

python /work/runyi_yang/FloWAM/code/RoboTwin/script/make_robotwin_small_sample.py \
  --data-root /work/runyi_yang/FloWAM/data \
  --out /work/runyi_yang/FloWAM/data/robotwin_small_sample \
  --episodes-per-task 3 \
  --max-frames 16 \
  --max-points 128 \
  --image-width 160 \
  --image-height 120 \
  --overwrite
```

Add `--keep-depth` if you want downsampled depth frames in the sample. The default omits depth because the current ACT, DP3, pi0, and pi0.5 RoboTwin pipelines use RGB, joint vectors, and point clouds, not depth.

## Full Data Location

Full RoboTwin data root:

```bash
/work/runyi_yang/FloWAM/data
```

Task directories:

```text
data/
  bartending/
  dough_rolling/
  handover/
  pick_and_place/
  rope_folding/
```

Each task directory contains files named like:

```text
episode0.hdf5
episode1.hdf5
...
```

Observed episode counts and lengths:

| task | files | readable | bad files | min T | median T | mean T | max T |
|---|---:|---:|---:|---:|---:|---:|---:|
| bartending | 28 | 28 | 0 | 296 | 384.5 | 399.75 | 640 |
| dough_rolling | 35 | 35 | 0 | 311 | 411 | 410.60 | 622 |
| handover | 30 | 30 | 0 | 237 | 308.5 | 307.20 | 423 |
| pick_and_place | 50 | 50 | 0 | 103 | 147.5 | 155.24 | 215 |
| rope_folding | 50 | 46 | 4 | 241 | 349 | 363.20 | 766 |

The unreadable `rope_folding` files in this scan were:

```text
episode41.hdf5
episode47.hdf5
episode48.hdf5
episode49.hdf5
```

## Full HDF5 Structure

Representative full file:

```bash
/work/runyi_yang/FloWAM/data/bartending/episode0.hdf5
```

Typical layout:

```text
episode*.hdf5
  endpose                              (T, 14) float64

  joint_action/
    vector                            (T, 14) float64
    position                          (T, 14) float64
    hand_qpos                         (T, 40) float64

  joint_state/
    vector                            (T, 14) float64
    position                          (T, 14) float64
    velocity                          (T, 14) float64
    effort                            (T, 14) float64
    hand_qpos                         (T, 40) float64

  observation/
    cam0/
      rgb                             (T,) JPEG byte strings
      depth                           (T, 800, 1280) uint16
      pcd                             (T, 2048, 6) float32
    cam1/
      rgb                             (T,) JPEG byte strings
      depth                           (T, 800, 1280) uint16
      pcd                             (T, 2048, 6) float32

  pointcloud                          (T, 2048, 6) float32
```

Important exception: `rope_folding` has `/joint_action/vector`, but some files do not include `/joint_action/position` or `/joint_action/hand_qpos`. The current loader handles this by using `/joint_action/vector` directly.

## Field Meaning

`T` is the number of frames in one full demonstration episode.

`/joint_action/vector` is the 14-D action/state vector used by the current RoboTwin training wrappers. The training conversion treats:

```text
qpos/input state at time t:      joint_action/vector[t]
action target at time t:         joint_action/vector[t + 1]
```

So an episode with `T` frames produces `T - 1` supervised action pairs.

`/joint_state/*` is observed robot state. In the current wrappers, `/joint_action/vector` is the canonical vector for ACT, DP3, pi0, and pi0.5 data conversion.

`/observation/cam0/rgb` and `/observation/cam1/rgb` are compressed JPEG byte strings. The loader decodes them with OpenCV. In the local two-camera files, the common camera mapping is:

```text
cam_high:        observation/cam0/rgb
cam_left_wrist:  observation/cam1/rgb
cam_right_wrist: observation/cam1/rgb
```

`/pointcloud` and `/observation/<camera>/pcd` are shaped `(T, N, 6)`. The first 3 channels are XYZ. The remaining 3 channels are per-point color or point features. For DP3 processing, the shared `/pointcloud` dataset is used.

Depth is present in full data, but not consumed by the current RoboTwin finetune/eval scripts.

## Small Sample HDF5 Structure

Representative sample file:

```bash
/work/runyi_yang/FloWAM/data/robotwin_small_sample/bartending/episode0.hdf5
```

Typical sample layout:

```text
episode0.hdf5
  endpose                              (16, 14) float64
  joint_action/vector                  (16, 14) float64
  joint_action/position                (16, 14) float64
  joint_action/hand_qpos               (16, 40) float64
  joint_state/vector                   (16, 14) float64
  joint_state/position                 (16, 14) float64
  joint_state/velocity                 (16, 14) float64
  joint_state/effort                   (16, 14) float64
  joint_state/hand_qpos                (16, 40) float64
  observation/cam0/rgb                 (16,) JPEG byte strings, 160x120 decoded
  observation/cam0/pcd                 (16, 128, 6) float32
  observation/cam1/rgb                 (16,) JPEG byte strings, 160x120 decoded
  observation/cam1/pcd                 (16, 128, 6) float32
  pointcloud                           (16, 128, 6) float32
```

For `rope_folding`, the sample follows the source file and may only contain:

```text
joint_action/vector                    (16, 14)
```

without `joint_action/position` or `joint_action/hand_qpos`.

The sample files include HDF5 attributes:

```text
source_file
source_length
sampled_frames
max_points
image_size
keep_depth
```

`manifest.json` records the source and output file for every sampled episode.

## Minimal Reader Script

This script reads one compact episode, decodes RGB, and prints the arrays used by the training code.

```python
from pathlib import Path

import cv2
import h5py
import numpy as np

path = Path("/work/runyi_yang/FloWAM/data/robotwin_small_sample/bartending/episode0.hdf5")

with h5py.File(path, "r") as f:
    joint = f["/joint_action/vector"][()].astype(np.float32)
    pointcloud = f["/pointcloud"][()].astype(np.float32)

    rgb_bytes = f["/observation/cam0/rgb"][0]
    rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)

print("joint:", joint.shape, joint.dtype)
print("pointcloud:", pointcloud.shape, pointcloud.dtype)
print("cam0 rgb:", rgb.shape, rgb.dtype)

qpos = joint[:-1]
action = joint[1:]
pcd_obs = pointcloud[:-1]

print("qpos/action pairs:", qpos.shape, action.shape)
print("dp3 pointcloud obs:", pcd_obs.shape)
```

Expected output for the generated sample:

```text
joint: (16, 14) float32
pointcloud: (16, 128, 6) float32
cam0 rgb: (120, 160, 3) uint8
qpos/action pairs: (15, 14) (15, 14)
dp3 pointcloud obs: (15, 128, 6)
```

## Using The Sample With RoboTwin Code

Use `ROBOTWIN_DATA_ROOT` to point the existing loaders at the compact sample:

```bash
export ROBOTWIN_DATA_ROOT=/work/runyi_yang/FloWAM/data/robotwin_small_sample
```

For direct processing smoke tests, call the policy-specific processors with a small `expert_data_num`. With 3 sample episodes per task and `--test-episodes 2`, the train split has 1 episode and the test split has 2 episodes:

```bash
cd /work/runyi_yang/FloWAM/code/RoboTwin/policy/DP3
ROBOTWIN_DATA_ROOT=/work/runyi_yang/FloWAM/data/robotwin_small_sample \
python scripts/process_data.py bartending demo_clean 1 --split train --test-episodes 2
```

The high-level finetune wrapper `flowam_train_eval_one.sh` is configured for the full data counts such as 26, 33, 28, and 48. Do not use it unchanged for the compact sample.

## PointWorld Compatibility Note

This RoboTwin sample is still RoboTwin-style data. It is not PointWorld scene-flow data.

RoboTwin stores observed point clouds per frame:

```text
pointcloud[t] = observed points at time t
```

PointWorld scene-flow training needs persistent point identities over a clip:

```text
scene_flows[t, n] = position or flow of the same physical point n through time
```

So this sample is useful for IO debugging, RoboTwin preprocessing, ACT/pi0 image-state checks, and DP3 point-cloud-state checks. To become PointWorld-like supervision, the data still needs tracked scene points or simulator object mesh points transformed through time.
