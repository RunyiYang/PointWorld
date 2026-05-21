# RoboTwin two-gripper WDS + PointWorld action fine-tuning

This patch adds a non-invasive RoboTwin two-gripper path to the PointWorld `main` branch:

- `convert_robotwin_to_wds.py` converts RoboTwin HDF5 demonstrations into fixed-horizon WDS clips.
- `train_robotwin_action.py` loads a released PointWorld checkpoint, adds a 14-D action decoder, and fine-tunes in two stages.
- `smoke_read_wds.py` checks that the generated WDS has the tensors the action trainer expects.

The PointWorld scene-flow release consumes fixed clips, not full episodes. The default clip horizon here is `11` with stride `5`, matching the BEHAVIOR-style clip size used in the release. The converter splits by episode before clipping, so overlapping clips from one episode cannot leak between train and test.

## Install location

Unzip this patch at the PointWorld repository root:

```bash
cd /path/to/PointWorld
unzip /path/to/code.zip
```

Then commit the new files:

```bash
git add tools scripts
git commit -m "Add RoboTwin two-gripper WDS conversion and action fine-tuning"
```

## Convert RoboTwin HDF5 to WDS

For the full data root:

```bash
python tools/robotwin2g/convert_robotwin_to_wds.py \
  --input-root /work/runyi_yang/FloWAM/data \
  --output-root /work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds \
  --test-ratio 0.02 \
  --split-scope task \
  --clip-horizon 11 \
  --clip-stride 5 \
  --scene-flow-mode repeat_t0 \
  --camera-names cam0 cam1
```

Or use the wrapper:

```bash
DATA_ROOT=/work/runyi_yang/FloWAM/data \
WDS_ROOT=/work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds \
bash scripts/robotwin2g_convert.sh
```

Generated layout:

```text
robotwin2g_pointworld_wds/
  train/train-000000.tar
  test/test-000000.tar
  metadata_rank0.json
  action_stats.json
  manifest.jsonl
  splits/train_episodes.txt
  splits/test_episodes.txt
```

The split is deterministic. `--split-scope task` applies the 0.02 test ratio within each task. Use `--split-scope global` when you want an overall global 2% split instead.

## What the converter writes

Each WDS sample contains:

```text
scene_flows.npy                 (11, N, 3)
scene_colors.npy                (11, N, 3) uint8
scene_normals.npy               (11, N, 3)
scene_visibility.npy            (11, N) bool
scene_depth_valid_mask.npy      (11, N) bool
robot_flows.npy                 (11, Nr, 3)
robot_colors.npy                (11, Nr, 3) uint8
robot_normals.npy               (11, Nr, 3)
right_gripper_pose.npy          (11, 7)
left_gripper_pose.npy           (11, 7)
right_gripper_open.npy          (11, 1)
left_gripper_open.npy           (11, 1)
action_state.npy                (11, 14)
action_target.npy               (10, 14)
action_mask.npy                 (10,)
cam0_initial_rgb.jpg            320x180
cam0_initial_depth.npy          180x320 float32 meters
cam0_intrinsic.npy              (3, 3)
cam0_extrinsic.npy              (4, 4), world-to-camera
cam1_initial_rgb.jpg            320x180
...
metadata.json
```

The 14-D action target is `joint_action/vector[t+1]` for each future step in the clip. `action_state` is `joint_action/vector[t]` for all 11 frames.

The two-gripper robot input is a precomputed surrogate point trajectory: by default 64 small sphere points around the right gripper and 64 around the left gripper, using `endpose[:, 7:14]` and `endpose[:, 0:7]` respectively. If your `endpose` layout differs, change:

```bash
--left-pose-slice 0:7 --right-pose-slice 7:14
```

## Important scene-flow note

RoboTwin `/pointcloud` is an observed point cloud sequence. It usually does not guarantee that point index `n` is the same physical point over time. The converter therefore defaults to:

```bash
--scene-flow-mode repeat_t0
```

This uses the initial scene cloud as repeated scene context and avoids pretending there is exact scene-flow supervision. Use:

```bash
--scene-flow-mode by_index
```

only if your exported point clouds have persistent point identities.

## Smoke test

```bash
python tools/robotwin2g/smoke_read_wds.py \
  --wds-root /work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds \
  --split train \
  --batch-size 2
```

Expected key shapes include:

```text
scene_flows      (B, 11, N, 3)
scene_features   (B, 1, N, 33)
robot_flows      (B, 11, Nr, 3)
robot_features   (B, 11, Nr, 14)
action_state     (B, 11, 14)
action_target    (B, 10, 14)
```

## Stage 1: action decoder only

```bash
python tools/robotwin2g/train_robotwin_action.py \
  --wds-root /work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds \
  --pretrained-checkpoint pretrained_checkpoints/large-droid+behavior/model-best.pt \
  --output-dir train_logs/robotwin2g_action_decoder \
  --stage action_decoder \
  --batch-size 8 \
  --num-workers 4 \
  --num-epochs 20 \
  --lr 1e-4 \
  --amp
```

Wrapper:

```bash
WDS_ROOT=/work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds \
PRETRAINED=pretrained_checkpoints/large-droid+behavior/model-best.pt \
bash scripts/robotwin2g_finetune_action_decoder.sh
```

## Stage 2: unfreeze PointWorld non-DINO modules

```bash
python tools/robotwin2g/train_robotwin_action.py \
  --wds-root /work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds \
  --pretrained-checkpoint pretrained_checkpoints/large-droid+behavior/model-best.pt \
  --resume train_logs/robotwin2g_action_decoder/checkpoint-best.pt \
  --output-dir train_logs/robotwin2g_all \
  --stage all \
  --batch-size 4 \
  --num-workers 4 \
  --num-epochs 20 \
  --lr 1e-4 \
  --world-lr 2e-5 \
  --amp
```

Wrapper:

```bash
WDS_ROOT=/work/runyi_yang/FloWAM/data/robotwin2g_pointworld_wds \
PRETRAINED=pretrained_checkpoints/large-droid+behavior/model-best.pt \
RESUME=train_logs/robotwin2g_action_decoder/checkpoint-best.pt \
bash scripts/robotwin2g_finetune_all.sh
```

The DINOv3 image backbone remains frozen by default, matching the released PointWorld code. Add `--unfreeze-dinov3` only if you have enough data and GPU memory.

## Checkpoints

The trainer writes:

```text
checkpoint-last.pt
checkpoint-best.pt
checkpoint-step<step>.pt
```

The action decoder predicts normalized actions internally. `action_mean` and `action_std` are saved in the checkpoint and come from `action_stats.json` computed on the train split only.

## Practical defaults

For small sample debugging, cap the conversion:

```bash
python tools/robotwin2g/convert_robotwin_to_wds.py \
  --input-root /work/runyi_yang/FloWAM/data/robotwin_small_sample \
  --output-root /tmp/robotwin2g_wds_debug \
  --max-clips-per-episode 2
```

For full fine-tuning, use the `large-droid+behavior` checkpoint rather than a DROID-only checkpoint because this path uses bimanual gripper features. DROID-only checkpoints can have incompatible feature dimensions for the two-gripper features.
