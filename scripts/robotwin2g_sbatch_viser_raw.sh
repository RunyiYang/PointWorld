#!/bin/bash
#SBATCH --job-name=pw_r2g_viser_raw
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/work/runyi_yang/FloWAM
REPO=${ROOT}/code/PointWorld
RAW_HDF5=${RAW_HDF5:-${ROOT}/robotwin2g_raw_tasks_only/bartending/episode25.hdf5}
PORT=${PORT:-8090}

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd "${REPO}"

python visualization/visualize_behavior_hdf5_flow.py \
  --hdf5 "${RAW_HDF5}" \
  --format robotwin \
  --point-slice 0:2048:8 \
  --background-max-points 2048 \
  --flow-window 1 \
  --line-point-stride 4 \
  --host 0.0.0.0 \
  --port "${PORT}"
