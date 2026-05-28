# RoboTwin2G PointWorld Per-Scene Runbook

This runbook documents the per-scene PointWorld finetuning workflow used for the RoboTwin2G tasks in this workspace. It intentionally excludes generated data and checkpoints from the codebase; those artifacts should live in external storage or the HF dataset repo.

## Tasks

```text
place_bread_skillet
move_stapler_pad
pick_diverse_bottles
place_phone_stand
stamp_seal
rotate_qrcode
adjust_bottle
beat_block_hammer
click_bell
lift_pot
place_bread_basket
```

## Split

For each scene:

```text
train episodes: 0-49
eval episodes: 50-59
```

`adjust_bottle` was missing train episodes `16` and `36`, so it used 48 train episodes and 10 eval episodes.

## Data Conversion

From the PointWorld repo root:

```bash
cd /work/runyi_yang/FloWAM/code/PointWorld
source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RAW_ROOT=/work/runyi_yang/FloWAM/data/robotwin_sim
OUT_ROOT=/work/runyi_yang/FloWAM/data/FloWAM/FloWAM_PointWorld_PerScene

for scene in \
  place_bread_skillet move_stapler_pad pick_diverse_bottles place_phone_stand \
  stamp_seal rotate_qrcode adjust_bottle beat_block_hammer click_bell \
  lift_pot place_bread_basket
do
  bash scripts/robotwin2g_convert_perscene_fixed_split.sh "$scene" "$RAW_ROOT" "$OUT_ROOT"
done
```

Important settings in `scripts/robotwin2g_convert_perscene_fixed_split.sh`:

```text
TRAIN_RANGE=0:50
TEST_RANGE=50:60
CLIP_HORIZON=11
CLIP_STRIDE=5
MAX_SCENE_POINTS=2048
POINTCLOUD_KEY=/pointcloud_single_camera/head_camera
HAND_QPOS_KEY=/preprocessed/endpose_derived/gripper_open
ENDPOSE_KEY=/endpose
CAMERA_NAMES="head_camera front_camera"
POLICY_ROBOT_INPUT=1
```

The policy robot input setting repeats the initial gripper geometry across the clip so future gripper trajectories are not leaked into the policy input.

## Training

Each scene is trained independently with two stages:

1. `--stage action_decoder`: freeze PointWorld and train only the action decoder.
2. `--stage all`: resume the best stage-1 checkpoint, reset optimizer/progress, and train the full PointWorld wrapper plus action head. DINO remains frozen unless explicitly unfrozen.

Submit independent non-array jobs:

```bash
cd /work/runyi_yang/FloWAM/code/PointWorld

for scene in \
  place_bread_skillet move_stapler_pad pick_diverse_bottles place_phone_stand \
  stamp_seal rotate_qrcode adjust_bottle beat_block_hammer click_bell \
  lift_pot place_bread_basket
do
  sbatch --job-name="pw_${scene}" scripts/robotwin2g_perscene_sbatch_train_one.sh "$scene"
done
```

The training script writes:

```text
train_logs/robotwin2g_perscene/<scene>_action_decoder/
train_logs/robotwin2g_perscene/<scene>_all/
```

Final eval JSONs are written to:

```text
/work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval/<scene>.json
```

Collect tables:

```bash
python tools/robotwin2g/collect_per_scene_results.py \
  --eval-dir /work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval \
  --data-root /work/runyi_yang/FloWAM/data/FloWAM/FloWAM_PointWorld_PerScene \
  --output-csv /work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval/per_scene_scores.csv \
  --output-md /work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval/per_scene_scores.md
```

## Metrics

`action_vector_rmse` and `action_vector_mae` are raw 14D vector metrics. `action_rmse_cm` and `action_cd_cm` use the configured action metric layout. For this run the layout was:

```text
two_gripper_xyz
```

That layout interprets action slots `[0:3]` and `[7:10]` as left/right XYZ positions in meters and converts to centimeters. If the target action source is `/joint_action/vector`, confirm that these slots are physically meaningful before treating the cm metrics as end-effector pose metrics. For true end-effector pose metrics, rebuild action targets from `/endpose/left_endpose` and `/endpose/right_endpose`.

## Artifact Policy

Keep the git checkout code-only:

```text
do not commit data/
do not commit train_logs/
do not commit outputs/
do not commit deployment/
do not commit checkpoints or *.pt
```

Upload completed training artifacts to HF before local cleanup.
