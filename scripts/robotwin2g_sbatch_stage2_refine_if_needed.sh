#!/bin/bash
#SBATCH --job-name=pw_r2g_stage2_refine
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-[0-7]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.out
#SBATCH --error=/work/runyi_yang/FloWAM/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/work/runyi_yang/FloWAM
REPO=${ROOT}/code/PointWorld
WDS_ROOT=${ROOT}/data/robotwin2g_wds
PRETRAINED=${REPO}/pretrained_checkpoints/large-droid+behavior/model-best.pt
EVAL_JSON=${ROOT}/outputs/robotwin2g_eval/stage1_stage2_full_test.json
RESUME=${REPO}/train_logs/robotwin2g_all/checkpoint-best.pt
OUT=${REPO}/train_logs/robotwin2g_all_refined

source /work/runyi_yang/miniconda3/etc/profile.d/conda.sh
conda activate pointworld-env

cd "${REPO}"

test -f "${EVAL_JSON}"
test -f "${RESUME}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${ROOT}/.cache"
export HF_HOME="${ROOT}/.cache/huggingface"
export TORCH_HOME="${ROOT}/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DECISION=$(python - <<'PY'
import json
from pathlib import Path

path = Path("/work/runyi_yang/FloWAM/outputs/robotwin2g_eval/stage1_stage2_full_test.json")
data = json.loads(path.read_text())
results = data["results"]

def find(name):
    for item in results:
        if name in item["checkpoint"]:
            return item
    raise SystemExit(f"missing checkpoint result containing {name}")

stage1 = find("robotwin2g_action_decoder/checkpoint-best.pt")
stage2 = find("robotwin2g_all/checkpoint-best.pt")
print(f"stage1_mae={stage1['mae']:.10f} stage2_best_mae={stage2['mae']:.10f}", flush=True)
print("skip" if stage2["mae"] <= stage1["mae"] else "train", flush=True)
PY
)

echo "${DECISION}"
ACTION=$(printf '%s\n' "${DECISION}" | tail -n 1)

if [[ "${ACTION}" == "skip" ]]; then
  echo "[decision] stage2 best is no worse than stage1 best on full test; skipping refinement."
  exit 0
fi

echo "[decision] stage2 best did not beat stage1 best on full test; running lower-LR refinement."
mkdir -p "${OUT}"

python tools/robotwin2g/train_robotwin_action.py \
  --wds-root "${WDS_ROOT}" \
  --pretrained-checkpoint "${PRETRAINED}" \
  --resume "${RESUME}" \
  --output-dir "${OUT}" \
  --stage all \
  --reset-optimizer \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --num-epochs "${NUM_EPOCHS:-30}" \
  --lr "${LR:-3e-5}" \
  --world-lr "${WORLD_LR:-3e-6}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-batches "${EVAL_BATCHES:-0}" \
  --require-test \
  --amp

test -f "${OUT}/checkpoint-last.pt"
test -f "${OUT}/checkpoint-best.pt"
