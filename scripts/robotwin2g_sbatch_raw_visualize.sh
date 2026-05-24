#!/bin/bash
#SBATCH --job-name=pw_r2g_raw_viz
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/work/runyi_yang/FloWAM
REPO=${ROOT}/code/PointWorld
WDS_ROOT=${WDS_ROOT:-${ROOT}/data/robotwin2g_wds}
OUT_DIR=${OUT_DIR:-${ROOT}/outputs/robotwin2g_flow_videos/raw_observed_test}
NUM_SAMPLES=${NUM_SAMPLES:-6}
MAX_POINTS=${MAX_POINTS:-1024}

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd "${REPO}"

python tools/robotwin2g/render_raw_pointcloud_videos.py \
  --wds-root "${WDS_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --split test \
  --num-samples "${NUM_SAMPLES}" \
  --max-points "${MAX_POINTS}" \
  --fps 2 \
  --hold-final 4
