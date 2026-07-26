#!/bin/bash
#SBATCH -J d23_smoke
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o /work/scratch/%u/keypoint_preoperator_logs/%x_%j.out
#SBATCH -e /work/scratch/%u/keypoint_preoperator_logs/%x_%j.err

set -euo pipefail

EXPECTED_COMMIT="${DECISION23_EXPECTED_COMMIT:?set DECISION23_EXPECTED_COMMIT}"
PROJECT_ROOT="/work/scratch/$USER/keypoint_preoperator_gates_20260726"
DATA_ROOT="/work/scratch/$USER/keypoint_learning/_tdw_world_z_roll_base_panel_512_v2"
VENV_DIR="/work/scratch/$USER/venvs/keypoint_learning"
OUTPUT_ROOT="/work/scratch/$USER/keypoint_preoperator_runs/decision23_$EXPECTED_COMMIT"
SCRIPT="keypoint_net/diagnostics/decision23_diagnostic_head.py"

module purge
module --ignore_cache load gcc
module --ignore_cache load openmpi
module load python
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
ACTUAL_BRANCH="$(git branch --show-current)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "commit mismatch: $ACTUAL_COMMIT != $EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ "$ACTUAL_BRANCH" != "agent/preoperator-gates-20260726" ]]; then
  echo "branch mismatch: $ACTUAL_BRANCH" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree is dirty" >&2
  git status --short >&2
  exit 2
fi

echo "gate=D1 job_id=$SLURM_JOB_ID host=$(hostname)"
echo "branch=$ACTUAL_BRANCH commit=$ACTUAL_COMMIT"
nvidia-smi

srun python -u "$SCRIPT" lock \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda

for ARM in raw_linear probability_linear fixed_expectation; do
  srun python -u "$SCRIPT" train \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda \
    --arm "$ARM" \
    --seed 42 \
    --run-scope smoke \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --min-epochs 2 \
    --max-epochs 2 \
    --eval-every 1 \
    --plateau-patience 1 \
    --relative-improvement 0.01
done

srun python -u keypoint_net/diagnostics/summarize_decision23_smoke.py \
  --output-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/DECISION23_D1_SMOKE_REPORT.json"
