#!/bin/bash
#SBATCH --job-name=pw_r2g_scene
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

TASK="${1:-${TASK:-}}"
if [[ -z "${TASK}" ]]; then
  echo "Usage: sbatch robotwin2g_perscene_sbatch_train_one.sh SCENE" >&2
  exit 2
fi

cd /work/runyi_yang/FloWAM/code/PointWorld
mkdir -p /work/runyi_yang/FloWAM/logs/slurm train_logs/robotwin2g_perscene /work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WDS_ROOT="${OUT_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM/FloWAM_PointWorld_PerScene}/${TASK}/wds"
PRETRAINED="${PRETRAINED:-pretrained_checkpoints/large-droid+behavior/model-best.pt}"
STAGE1_OUT="train_logs/robotwin2g_perscene/${TASK}_action_decoder"
STAGE2_OUT="train_logs/robotwin2g_perscene/${TASK}_all"
EVAL_JSON="/work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval/${TASK}.json"

test -d "${WDS_ROOT}/train"
test -d "${WDS_ROOT}/test"
test -f "${WDS_ROOT}/action_stats.json"

if [[ "${RESET_OUTPUTS:-1}" == "1" || "${RESET_OUTPUTS:-true}" == "true" ]]; then
  rm -rf "${STAGE1_OUT}" "${STAGE2_OUT}"
  rm -f "${EVAL_JSON}"
fi

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --output-dir "${STAGE1_OUT}" \
  --stage action_decoder \
  --action-dim 14 \
  --batch-size "${STAGE1_BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --eval-num-workers "${EVAL_NUM_WORKERS:-4}" \
  --num-epochs "${STAGE1_NUM_EPOCHS:-50}" \
  --max-steps "${STAGE1_MAX_STEPS:-1000}" \
  --lr "${STAGE1_LR:-1e-4}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --scene-cd-max-points "${SCENE_CD_MAX_POINTS:-1024}" \
  --action-cd-max-points "${ACTION_CD_MAX_POINTS:-1024}" \
  --action-metric-layout "${ACTION_METRIC_LAYOUT:-two_gripper_xyz}" \
  --require-test \
  --amp

test -f "${STAGE1_OUT}/checkpoint-best.pt"

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --resume "${STAGE1_OUT}/checkpoint-best.pt" \
  --output-dir "${STAGE2_OUT}" \
  --stage all \
  --reset-optimizer \
  --reset-progress \
  --action-dim 14 \
  --scene-loss-weight "${SCENE_LOSS_WEIGHT:-1.0}" \
  --scene-cd-max-points "${SCENE_CD_MAX_POINTS:-1024}" \
  --action-cd-max-points "${ACTION_CD_MAX_POINTS:-1024}" \
  --batch-size "${STAGE2_BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --eval-num-workers "${EVAL_NUM_WORKERS:-4}" \
  --num-epochs "${STAGE2_NUM_EPOCHS:-50}" \
  --max-steps "${STAGE2_MAX_STEPS:-2000}" \
  --lr "${STAGE2_LR:-1e-4}" \
  --world-lr "${WORLD_LR:-2e-5}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-20}" \
  --action-metric-layout "${ACTION_METRIC_LAYOUT:-two_gripper_xyz}" \
  --require-test \
  --amp

test -f "${STAGE2_OUT}/checkpoint-best.pt"
test -f "${STAGE2_OUT}/checkpoint-last.pt"

python tools/robotwin2g/evaluate_action_checkpoint.py \
  --wds-root "${WDS_ROOT}" \
  --checkpoints \
    "${STAGE1_OUT}/checkpoint-best.pt" \
    "${STAGE2_OUT}/checkpoint-best.pt" \
    "${STAGE2_OUT}/checkpoint-last.pt" \
  --output-json "${EVAL_JSON}" \
  --split test \
  --batch-size "${EVAL_BATCH_SIZE:-8}" \
  --num-workers "${EVAL_NUM_WORKERS:-4}" \
  --max-batches 0 \
  --scene-cd-max-points "${SCENE_CD_MAX_POINTS:-1024}" \
  --action-cd-max-points "${ACTION_CD_MAX_POINTS:-1024}" \
  --action-metric-layout "${ACTION_METRIC_LAYOUT:-two_gripper_xyz}" \
  --amp
