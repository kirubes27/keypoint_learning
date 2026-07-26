#!/bin/bash
#SBATCH -J d23_ext_final
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o /work/scratch/%u/keypoint_preoperator_logs/%x_%j.out
#SBATCH -e /work/scratch/%u/keypoint_preoperator_logs/%x_%j.err

set -euo pipefail

EXPECTED_COMMIT="${DECISION23_EXPECTED_COMMIT:?set DECISION23_EXPECTED_COMMIT}"
INITIAL_REPORT="${DECISION23_INITIAL_REPORT:?set DECISION23_INITIAL_REPORT}"
PROVISIONAL_ARMS="${DECISION23_PROVISIONAL_ARMS:?set DECISION23_PROVISIONAL_ARMS}"
PROJECT_ROOT="/work/scratch/$USER/keypoint_preoperator_gates_20260726"
VENV_DIR="/work/scratch/$USER/venvs/keypoint_learning"
OUTPUT_ROOT="/work/scratch/$USER/keypoint_preoperator_runs/decision23_$EXPECTED_COMMIT"
RUN_ROOT="$OUTPUT_ROOT/full/runs"
SCRIPT="keypoint_net/diagnostics/decision23_diagnostic_head.py"

IFS=',' read -r -a ARMS <<< "$PROVISIONAL_ARMS"
RUN_DIRS=()
for ARM in "${ARMS[@]}"; do
  if [[ "$ARM" != "raw_linear" && "$ARM" != "probability_linear" && "$ARM" != "fixed_expectation" ]]; then
    echo "invalid provisional arm: $ARM" >&2
    exit 2
  fi
  RUN_DIRS+=("$RUN_ROOT/e2e_${ARM}_standard64_k10_seed45")
  RUN_DIRS+=("$RUN_ROOT/e2e_${ARM}_standard64_k10_seed46")
done

module purge
module --ignore_cache load gcc
module --ignore_cache load openmpi
module load python
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

ACTUAL_BRANCH="$(git branch --show-current)"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "commit mismatch" >&2
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
if [[ ! -f "$INITIAL_REPORT" ]]; then
  echo "missing initial report: $INITIAL_REPORT" >&2
  exit 2
fi

echo "gate=D3_extension_finalization job_id=$SLURM_JOB_ID host=$(hostname)"
echo "commit=$EXPECTED_COMMIT provisional_arms=$PROVISIONAL_ARMS"
nvidia-smi

srun python -u "$SCRIPT" finalize-extension \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --initial-report "$INITIAL_REPORT" \
  --run-dirs "${RUN_DIRS[@]}"
