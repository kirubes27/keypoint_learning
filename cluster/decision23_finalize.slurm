#!/bin/bash
#SBATCH -J d23_finalize
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
VENV_DIR="/work/scratch/$USER/venvs/keypoint_learning"
OUTPUT_ROOT="/work/scratch/$USER/keypoint_preoperator_runs/decision23_$EXPECTED_COMMIT"
RUN_ROOT="$OUTPUT_ROOT/full/runs"
SCRIPT="keypoint_net/diagnostics/decision23_diagnostic_head.py"

RUN_DIRS=(
  "$RUN_ROOT/e2e_raw_linear_standard64_k10_seed42"
  "$RUN_ROOT/e2e_raw_linear_standard64_k10_seed43"
  "$RUN_ROOT/e2e_raw_linear_standard64_k10_seed44"
  "$RUN_ROOT/e2e_probability_linear_standard64_k10_seed42"
  "$RUN_ROOT/e2e_probability_linear_standard64_k10_seed43"
  "$RUN_ROOT/e2e_probability_linear_standard64_k10_seed44"
  "$RUN_ROOT/e2e_fixed_expectation_standard64_k10_seed42"
  "$RUN_ROOT/e2e_fixed_expectation_standard64_k10_seed43"
  "$RUN_ROOT/e2e_fixed_expectation_standard64_k10_seed44"
)

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

echo "gate=D3 job_id=$SLURM_JOB_ID host=$(hostname)"
echo "branch=$ACTUAL_BRANCH commit=$ACTUAL_COMMIT"
nvidia-smi

srun python -u "$SCRIPT" finalize \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --run-dirs "${RUN_DIRS[@]}"
