# Data Structure Summary

This note summarizes the two HDF5 data layouts currently present in this workspace:

- PointWorld-BEHAVIOR restored annotations:
  `/work/runyi_yang/FloWAM/data/pointworld_behavior_restored/behavior/flows/task-0000/episode_00000010.hdf5`
- RoboTwin demonstration data:
  `/work/runyi_yang/FloWAM/data/bartending/episode0.hdf5`

They are not the same format. PointWorld-BEHAVIOR is a BEHAVIOR-derived 3D annotation dataset organized by short clip groups. RoboTwin is a robot imitation dataset organized as full demonstrations with time-aligned joint actions, camera frames, and point clouds.

## PointWorld-BEHAVIOR

Restored canonical root:

```text
/work/runyi_yang/FloWAM/data/pointworld_behavior_restored/behavior
```

Typical file:

```text
behavior/
  flows/
    task-0000/
      episode_00000010.hdf5
```

Top-level keys are clip windows, named by frame ranges:

```text
125:136
130:141
135:146
...
```

Each clip group contains robot state, camera calibration, initial RGB-D observations, and scene geometry annotations:

```text
<clip>/
  base_pose                         (T, 7) float32
  world_to_robot                    (4, 4) float32
  joint_names                       (J,) object/string
  joint_positions                   (T, J) float32
  left_gripper_open                 (T, 1) bool
  left_gripper_pose                 (T, 7) float32
  left_is_grasping                  (T, 1) bool
  right_gripper_open                (T, 1) bool
  right_gripper_pose                (T, 7) float32
  right_is_grasping                 (T, 1) bool

  camera_head/
    initial_rgb                     (1,) object, JPEG bytes
    initial_depth                   (180, 320) uint16, millimeters
    intrinsic                       (3, 3) float32
    extrinsic                       (4, 4) float32
    extrinsic_trajectory            (T, 4, 4) float32
    local_scene_points/<mesh>       (N, 3) float16
    local_scene_colors/<mesh>       (N, 3) uint8
    local_scene_normals/<mesh>      (N, 3) int8
    scene_mesh_trajectories/<mesh>  (T, 7) float32

  camera_left/
    ...

  camera_right/
    ...
```

For `episode_00000010.hdf5`, the inspected clip length is usually `T = 11`, and `joint_positions` has shape `(11, 22)`.

## Clip and Episode Horizon

PointWorld-BEHAVIOR should be treated as a clip dataset, not a full-episode
dataset. In the restored files, each top-level HDF5 group is one clip. The group
name is a frame window such as `125:136`; the end is exclusive, so this window
contains `136 - 125 = 11` timesteps.

A sampled scan over 20 restored BEHAVIOR HDF5 files, spread across the 7,842
files in this workspace, found:

```text
clip length T:              always 11 in the sample
usual clip stride:          5 frames between neighboring clip starts
clips per HDF5 episode:     median 185, min 25, max 481 in the sample
covered episode span:       median 1386 frames, sample range 181..4056 frames
```

So PointWorld usually runs on overlapping 11-frame windows from a longer
episode. The full episode is only the source container that supplies many
overlapping clip samples.

For the current RoboTwin-style data in this workspace, the full sequence length
is the HDF5 episode length, read from `/joint_action/vector` or `/pointcloud`.
Observed local lengths:

```text
bartending:       28 episodes, median 388, range 296..640
dough_rolling:   35 episodes, median 411, range 311..622
handover:         30 episodes, median 308, range 237..423
pick_and_place:   50 episodes, median 147, range 103..215
rope_folding:     46 readable episodes, median 347, range 241..766
all readable:    189 episodes, median 333, range 103..766
```

Three local `rope_folding` HDF5 files did not open cleanly during this metadata
scan: `episode47.hdf5`, `episode48.hdf5`, and `episode49.hdf5`.

## PointWorld Inference Granularity

The released PointWorld training/evaluation code consumes WebDataset samples.
Each sample is already a fixed-horizon clip, not a whole raw episode. Evaluation
iterates batches of clips, runs `model(batch, training=False)`, and compares the
predicted scene flow against the clip ground truth.

The model input/output horizon is therefore the clip horizon `T`, typically
`11`. The forward pass uses:

```text
initial scene points:   scene_flows[:, 0]
robot trajectory:       robot_flows[:, 0:T]
scene/camera features:  initial RGB-D plus camera intrinsics/extrinsics
output:                 predicted scene_flows[:, 0:T]
```

If full-episode inference is needed, it must be implemented by sliding or
sampling 11-frame windows over the episode and then optionally stitching or
aggregating the per-clip predictions. The current code path does not run one
forward pass over a 300-700 frame episode.

## Converting RoboTwin to PointWorld-Like Data

It is possible, but the hard part is not HDF5 formatting. The hard part is
producing PointWorld-compatible scene flow supervision and robot flow inputs.

Required pieces for a useful PointWorld-like conversion:

```text
1. Split each full RoboTwin episode into 11-frame clips, usually stride 5.
2. Provide initial RGB-D, intrinsics, and extrinsics per selected camera.
3. Provide scene points at t=0 with consistent identities through the 11 frames.
4. Provide scene_flows shaped (T, N, 3), where point n is the same physical
   scene point at every timestep.
5. Provide robot_flows shaped (T, Nr, 3), either by adding a robot sampler/URDF
   for the RoboTwin robot or by modifying the pipeline to accept precomputed
   robot point trajectories.
```

The main blocker is point identity. RoboTwin stores per-frame observed point
clouds like `/pointcloud` `(T, 2048, 6)`, but those are not guaranteed to be the
same physical points across time. PointWorld's target is tracked scene flow:
the same initial point is moved through time. BEHAVIOR gets this from mesh
points plus mesh pose trajectories.

A practical path is:

```text
best:       export object/mesh point samples and object poses from the simulator,
            then build exact scene_flows like BEHAVIOR;
acceptable: if simulator object poses are available, sample object-local points
            once at clip start and transform them through time;
risky:      approximate correspondences between per-frame point clouds with
            nearest neighbors or optical/scene flow estimation.
```

For RoboTwin, converting to a DROID-like PointWorld domain may be easier than
pretending it is BEHAVIOR, but the robot model still has to match. The existing
DROID path assumes a single Franka/Panda-style arm; the BEHAVIOR path assumes
the BEHAVIOR bimanual robot joint names and base pose. RoboTwin's 14-DoF action
vectors and 40-DoF hand state need either a matching URDF sampler or a custom
precomputed `robot_flows` path.

PointWorld-BEHAVIOR does not store a single dense `/pointcloud` dataset. Scene geometry is stored per camera and per mesh as local mesh point samples plus a rigid pose trajectory for each mesh.

## RoboTwin Demonstrations

Typical file:

```text
/work/runyi_yang/FloWAM/data/bartending/episode0.hdf5
```

Top-level keys:

```text
endpose
joint_action
joint_state
observation
pointcloud
```

Observed structure:

```text
endpose                              (T, 14) float64

joint_action/
  vector                             (T, 14) float64
  position                           (T, 14) float64
  hand_qpos                          (T, 40) float64

joint_state/
  vector                             (T, 14) float64
  position                           (T, 14) float64
  velocity                           (T, 14) float64
  effort                             (T, 14) float64
  hand_qpos                          (T, 40) float64

observation/
  cam0/
    rgb                              (T,) JPEG bytes
    depth                            (T, 800, 1280) uint16
    pcd                              (T, 2048, 6) float32
  cam1/
    rgb                              (T,) JPEG bytes
    depth                            (T, 800, 1280) uint16
    pcd                              (T, 2048, 6) float32

pointcloud                           (T, 2048, 6) float32
```

For `bartending/episode0.hdf5`, the inspected trajectory length is `T = 510`.

## Key Differences

| Field | PointWorld-BEHAVIOR | RoboTwin |
| --- | --- | --- |
| Top-level unit | Clip groups like `125:136` | One full episode |
| Joint action | No `/joint_action/vector`; has `joint_positions` per clip | `/joint_action/vector` `(T, 14)` |
| Cameras | `camera_head`, `camera_left`, `camera_right` inside each clip | `observation/cam0`, `observation/cam1` over full episode |
| RGB | One `initial_rgb` JPEG per camera per clip | RGB sequence with one JPEG per timestep |
| Depth | One initial depth image per camera per clip | Depth sequence over all timesteps |
| Point cloud | Per-mesh local scene points plus mesh trajectories | Dense time sequence `/pointcloud` `(T, 2048, 6)` |
| Typical purpose | 3D annotation / world-model data | Imitation learning demo data |

Because of these differences, PointWorld-BEHAVIOR cannot be used directly by the RoboTwin ACT/DP3/pi0/pi0.5 data loaders without a converter.

## Reading PointWorld-BEHAVIOR

Use a Python environment with `h5py`, for example:

```bash
/work/runyi_yang/miniconda3/envs/flowam-act/bin/python read_pointworld.py
```

Minimal read example:

```python
import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def decode_jpeg_object(entry) -> np.ndarray:
    if isinstance(entry, np.ndarray):
        jpeg_bytes = entry.astype(np.uint8, copy=False).tobytes()
    elif isinstance(entry, (bytes, bytearray, memoryview)):
        jpeg_bytes = bytes(entry)
    elif hasattr(entry, "tobytes"):
        jpeg_bytes = entry.tobytes()
    else:
        jpeg_bytes = bytes(entry)
    return np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))


path = Path(
    "/work/runyi_yang/FloWAM/data/"
    "pointworld_behavior_restored/behavior/flows/task-0000/episode_00000010.hdf5"
)

with h5py.File(path, "r") as f:
    clip_key = sorted(f.keys())[0]
    clip = f[clip_key]

    joint_names = [name.decode() if isinstance(name, bytes) else str(name) for name in clip["joint_names"][:]]
    joint_positions = clip["joint_positions"][:]
    base_pose = clip["base_pose"][:]

    cam = clip["camera_head"]
    rgb = decode_jpeg_object(cam["initial_rgb"][0])
    depth_m = cam["initial_depth"][:].astype(np.float32) / 1000.0
    intrinsic = cam["intrinsic"][:]
    extrinsic_trajectory = cam["extrinsic_trajectory"][:]

    mesh_name = sorted(cam["local_scene_points"].keys())[0]
    local_points = cam["local_scene_points"][mesh_name][:].astype(np.float32)
    local_colors = cam["local_scene_colors"][mesh_name][:]
    mesh_trajectory = cam["scene_mesh_trajectories"][mesh_name][:]

print("clip:", clip_key)
print("joint_names:", joint_names)
print("joint_positions:", joint_positions.shape)
print("rgb:", rgb.shape)
print("depth_m:", depth_m.shape)
print("intrinsic:", intrinsic.shape)
print("extrinsic_trajectory:", extrinsic_trajectory.shape)
print("mesh:", mesh_name)
print("local_points:", local_points.shape)
print("local_colors:", local_colors.shape)
print("mesh_trajectory:", mesh_trajectory.shape)
```

## Reading RoboTwin

Minimal read example:

```python
import cv2
import h5py
import numpy as np


path = "/work/runyi_yang/FloWAM/data/bartending/episode0.hdf5"

with h5py.File(path, "r") as f:
    action = f["/joint_action/vector"][:].astype(np.float32)
    state = f["/joint_state/vector"][:].astype(np.float32)
    pointcloud = f["/pointcloud"][:].astype(np.float32)

    encoded = f["/observation/cam0/rgb"][0]
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)

print("action:", action.shape)
print("state:", state.shape)
print("pointcloud:", pointcloud.shape)
print("image:", image.shape)
```

## Conversion Notes

To adapt PointWorld-BEHAVIOR into a RoboTwin-style loader, a converter would need to define:

1. How to map PointWorld `joint_positions` `(T, 22)` into a RoboTwin-like action/state vector, if that is meaningful for the target model.
2. How to create time-indexed RGB observations. PointWorld stores `initial_rgb` per clip camera, not a full RGB frame sequence.
3. How to reconstruct a dense point cloud from per-mesh `local_scene_points` and `scene_mesh_trajectories`.
4. Whether each PointWorld clip should become one training episode, or whether multiple clips from the same source HDF5 should be concatenated.

For PointWorld training, prefer the native PointWorld data pipeline rather than converting to RoboTwin format.
