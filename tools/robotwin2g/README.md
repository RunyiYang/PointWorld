# RoboTwin two-gripper PointWorld data + action fine-tuning

This patch adds a RoboTwin two-gripper path to PointWorld while following the PointWorld `data` branch workflow:

```text
raw RoboTwin HDF5 episodes
  -> generated PointWorld-like clip H5 files
  -> integrity_check.json
  -> train/test WDS manifest
  -> WDS shards
  -> two-stage action fine-tuning
```

The training target is a 14-D two-gripper action vector.  The patch does **not** claim to create exact scene-flow supervision from RoboTwin observed point clouds.  It uses PointWorld as the 3D/visual backbone and adds an action decoder for RoboTwin.

## Files added

```text
tools/robotwin2g/convert_robotwin_to_pointworld_h5.py   # raw episodes -> clip H5
tools/robotwin2g/integrity_check_robotwin_h5.py         # validate generated H5 clips
tools/robotwin2g/make_robotwin_wds_manifest.py          # episode-level 0.02 train/test split
tools/robotwin2g/convert_robotwin_h5_to_wds.py          # generated H5 -> WDS shards
tools/robotwin2g/convert_robotwin_to_wds.py             # older direct converter, kept for debugging
tools/robotwin2g/train_robotwin_action.py               # two-stage finetuning
tools/robotwin2g/action_model.py                        # PointWorld + action decoder
tools/robotwin2g/wds_dataset.py                         # custom WDS reader/collator
tools/robotwin2g/smoke_read_wds.py                      # read smoke test
scripts/robotwin2g_convert.sh                           # complete data pipeline wrapper
scripts/robotwin2g_finetune_action_decoder.sh           # stage 1 wrapper
scripts/robotwin2g_finetune_all.sh                      # stage 2 wrapper
```

## Install

Unzip at the PointWorld repository root:

```bash
cd /path/to/PointWorld
unzip /path/to/pointworld_robotwin2g_data_branch_patch.zip

git add tools/robotwin2g scripts/robotwin2g_*.sh ROBOTWIN2G_README.md
git commit -m "Add RoboTwin two-gripper PointWorld data conversion and action finetuning"
```

## Data conversion

Run the full official-style pipeline:

```bash
bash scripts/robotwin2g_convert.sh \
  /path/to/robotwin_data \
  /path/to/robotwin2g_wds \
  /path/to/robotwin2g_generated_h5
```

Useful environment variables:

```bash
TEST_RATIO=0.02              # default
SPLIT_SCOPE=task             # task or global
SEED=42
CLIP_HORIZON=11
CLIP_STRIDE=5
MAX_SCENE_POINTS=2048
MAX_CLIPS_PER_EPISODE=-1     # debug cap; <=0 disables
MAX_SAMPLES_PER_SHARD=512
SCENE_FLOW_MODE=repeat_t0    # repeat_t0 or by_index
POLICY_ROBOT_INPUT=1         # safer for policy learning: repeat t0 robot geometry
IMAGE_WIDTH=320
IMAGE_HEIGHT=180
```

Example debug run:

```bash
MAX_CLIPS_PER_EPISODE=2 MAX_SCENE_POINTS=128 \
bash scripts/robotwin2g_convert.sh \
  /path/to/robotwin_small_sample \
  /tmp/robotwin2g_wds_debug \
  /tmp/robotwin2g_h5_debug
```

## Generated-H5 layout

The first stage writes H5 files like:

```text
robotwin2g_generated_h5/
  task_map.json
  generated_h5_manifest.jsonl
  metadata.json
  flows/
    task-0000/
      episode_00000000.hdf5
      episode_00000001.hdf5
    task-0001/
      episode_00000000.hdf5
```

Each H5 episode contains clip groups named by frame range:

```text
episode_00000000.hdf5
  0:11/
  5:16/
  10:21/
  ...
```

Each clip group contains both direct arrays used by the trainer and BEHAVIOR-like fields used for data-contract compatibility/debugging:

```text
<clip>/
  scene_flows                         (11, N, 3)
  scene_colors                        (11, N, 3) uint8
  scene_normals                       (11, N, 3)
  scene_visibility                    (11, N) bool
  scene_depth_valid_mask              (11, N) bool

  robot_flows                         (11, Nr, 3)
  robot_colors                        (11, Nr, 3) uint8
  robot_normals                       (11, Nr, 3)
  left_gripper_pose                   (11, 7)
  right_gripper_pose                  (11, 7)
  left_gripper_open                   (11, 1)
  right_gripper_open                  (11, 1)

  action_state                        (11, 14)
  action_target                       (10, 14)
  action_mask                         (10,)

  base_pose
  world_to_robot
  joint_names
  joint_positions

  camera_cam0/
    initial_rgb                       JPEG bytes
    initial_depth                     (180, 320) uint16, millimeters
    intrinsic                         (3, 3)
    extrinsic                         (4, 4)
    extrinsic_trajectory              (11, 4, 4)
    local_scene_points/robotwin_observed
    local_scene_colors/robotwin_observed
    local_scene_normals/robotwin_observed
    scene_mesh_trajectories/robotwin_observed
  camera_cam1/
    ...
```

## WDS layout

The final stage writes:

```text
robotwin2g_wds/
  train/train-000000.tar
  train/train-000001.tar
  test/test-000000.tar
  metadata_rank0.json
  action_stats.json
  train_sources.txt
  test_sources.txt
```

Each WDS sample contains:

```text
scene_flows.npy
scene_colors.npy
scene_normals.npy
scene_visibility.npy
scene_depth_valid_mask.npy
robot_flows.npy
robot_colors.npy
robot_normals.npy
left_gripper_pose.npy
right_gripper_pose.npy
left_gripper_open.npy
right_gripper_open.npy
action_state.npy
action_target.npy
action_mask.npy
cam0_initial_rgb.jpg
cam0_initial_depth.npy
cam0_intrinsic.npy
cam0_extrinsic.npy
cam1_initial_rgb.jpg
cam1_initial_depth.npy
cam1_intrinsic.npy
cam1_extrinsic.npy
metadata.json
```

The test split is episode-level.  Adjacent overlapping clips from one episode cannot appear in both train and test.  With only one episode, the test split is empty by design.

## Action target

The default target is next-step action over the clip:

```text
action_state[t]    = /joint_action/vector[start + t]
action_target[t]   = /joint_action/vector[start + t + 1]
```

So a clip of 11 frames produces:

```text
action_state:  (11, 14)
action_target: (10, 14)
```

`action_stats.json` is computed from train `action_target` only and is used to normalize actions in training.

## Scene-flow caveat

RoboTwin `/pointcloud` stores observed point clouds per frame.  PointWorld scene-flow training expects the same physical scene point to be tracked through the whole clip.  Therefore the default is:

```bash
SCENE_FLOW_MODE=repeat_t0
```

This keeps PointWorld-style scene tensors available for the backbone without using fake scene-flow supervision.  Use `SCENE_FLOW_MODE=by_index` only when your simulator export guarantees stable point identities.

## Robot input caveat

For closed-loop policy training, keep:

```bash
POLICY_ROBOT_INPUT=1
```

This repeats the t=0 gripper geometry through the clip and avoids leaking future endpose trajectory into the action target.  Setting `POLICY_ROBOT_INPUT=0` uses the full gripper trajectory, which is closer to the PointWorld world-model input but less strict for imitation policy learning.

## Smoke test

```bash
python tools/robotwin2g/smoke_read_wds.py \
  --wds-root /path/to/robotwin2g_wds \
  --split train \
  --batch-size 2
```

Expected shapes include:

```text
scene_flows      (B, 11, N, 3)
scene_features   (B, 1, N, 33)
robot_flows      (B, 11, Nr, 3)
robot_features   (B, 11, Nr, 14)
action_state     (B, 11, 14)
action_target    (B, 10, 14)
```

## Stage 1: train only action decoder

```bash
bash scripts/robotwin2g_finetune_action_decoder.sh \
  /path/to/robotwin2g_wds \
  /path/to/pointworld_checkpoint.pth \
  /path/to/output_stage1
```

Equivalent direct command:

```bash
python tools/robotwin2g/train_robotwin_action.py \
  --wds-root /path/to/robotwin2g_wds \
  --pretrained-checkpoint /path/to/pointworld_checkpoint.pth \
  --output-dir /path/to/output_stage1 \
  --stage action_decoder \
  --batch-size 8 \
  --num-workers 4 \
  --num-epochs 20 \
  --lr 1e-4 \
  --amp
```

## Stage 2: finetune PointWorld non-DINO modules + action decoder

```bash
bash scripts/robotwin2g_finetune_all.sh \
  /path/to/robotwin2g_wds \
  /path/to/output_stage1/checkpoint-best.pt \
  /path/to/output_stage2
```

Equivalent direct command:

```bash
python tools/robotwin2g/train_robotwin_action.py \
  --wds-root /path/to/robotwin2g_wds \
  --pretrained-checkpoint /path/to/pointworld_checkpoint.pth \
  --resume /path/to/output_stage1/checkpoint-best.pt \
  --output-dir /path/to/output_stage2 \
  --stage all \
  --batch-size 4 \
  --num-workers 4 \
  --num-epochs 20 \
  --lr 1e-4 \
  --world-lr 2e-5 \
  --amp
```

DINO/DINOv3 is frozen by default.  Add `--unfreeze-dinov3` only when you explicitly want to tune the visual backbone and have enough data/GPU memory.

## Checkpoints

The trainer writes:

```text
checkpoint-last.pt
checkpoint-best.pt
checkpoint-step<step>.pt
```

The checkpoint stores `action_mean` and `action_std` from `action_stats.json`.
