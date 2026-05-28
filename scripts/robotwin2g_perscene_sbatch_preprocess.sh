#!/bin/bash
#SBATCH --job-name=pw_r2g_prep_scene
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-10%4
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%A_%a.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%A_%a.err

set -euo pipefail

TASKS=(
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
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

cd /work/runyi_yang/FloWAM/code/PointWorld
mkdir -p /work/runyi_yang/FloWAM/logs/slurm

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

bash scripts/robotwin2g_convert_perscene_fixed_split.sh \
  "${TASK}" \
  "${RAW_ROOT:-/work/runyi_yang/FloWAM/data/robotwin_sim}" \
  "${OUT_ROOT:-/work/runyi_yang/FloWAM/data/FloWAM/FloWAM_PointWorld_PerScene}"
