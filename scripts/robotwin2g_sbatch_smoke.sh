#!/bin/bash
#SBATCH --job-name=pw_r2g_smoke
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

cd /work/runyi_yang/FloWAM/code/PointWorld
mkdir -p /work/runyi_yang/FloWAM/logs/slurm train_logs

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WDS_ROOT=/work/runyi_yang/FloWAM/data/robotwin2g_wds
PRETRAINED=pretrained_checkpoints/large-droid+behavior/model-best.pt
OUT=train_logs/robotwin2g_smoke_action_decoder

python tools/robotwin2g/smoke_read_wds.py \
  --wds-root "${WDS_ROOT}" \
  --split train \
  --batch-size 2

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --output-dir "${OUT}" \
  --stage action_decoder \
  --batch-size 1 \
  --num-workers 0 \
  --num-epochs 1 \
  --max-steps 10 \
  --eval-every 5 \
  --eval-batches 2 \
  --lr 1e-4 \
  --require-test \
  --amp

test -f "${OUT}/checkpoint-last.pt"
test -f "${OUT}/checkpoint-best.pt"
