#!/bin/bash
#SBATCH --job-name=pw_r2g_scene_collect
#SBATCH --partition=batch
#SBATCH --nodelist=hala
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

cd /work/runyi_yang/FloWAM/code/PointWorld
mkdir -p /work/runyi_yang/FloWAM/logs/slurm /work/runyi_yang/FloWAM/outputs/robotwin2g_perscene_eval

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python tools/robotwin2g/collect_per_scene_results.py
