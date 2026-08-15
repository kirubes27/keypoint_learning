#!/bin/bash
#SBATCH -J selfeq_smoke
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH --array=0-1%2
#SBATCH -A l0003424
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%A_%a.err

set -euo pipefail

EXPECTED_COMMIT="${SELF_EQUIVARIANCE_EXPECTED_COMMIT:?set SELF_EQUIVARIANCE_EXPECTED_COMMIT}"
PROJECT_ROOT="${SELF_EQUIVARIANCE_PROJECT_ROOT:?set SELF_EQUIVARIANCE_PROJECT_ROOT}"
VENV_DIR="${SELF_EQUIVARIANCE_VENV_DIR:?set SELF_EQUIVARIANCE_VENV_DIR}"
MANIFEST="${SELF_EQUIVARIANCE_MANIFEST:?set SELF_EQUIVARIANCE_MANIFEST}"
MANIFEST_SHA256="${SELF_EQUIVARIANCE_MANIFEST_SHA256:?set SELF_EQUIVARIANCE_MANIFEST_SHA256}"
EXPECTED_SCRIPT_SHA256="${SELF_EQUIVARIANCE_SLURM_SCRIPT_SHA256:?set SELF_EQUIVARIANCE_SLURM_SCRIPT_SHA256}"
EXPECTED_ENV_LOCK_SHA256="${SELF_EQUIVARIANCE_ENV_LOCK_SHA256:?set SELF_EQUIVARIANCE_ENV_LOCK_SHA256}"
ENV_LOCK="$VENV_DIR/FRESH_ROLL_ENVIRONMENT.json"
EXECUTED_SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_PATH="$(realpath "$PROJECT_ROOT/cluster/same_frame_equivariance_smoke.slurm")"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
for VALUE in "$MANIFEST_SHA256" "$EXPECTED_SCRIPT_SHA256" "$EXPECTED_ENV_LOCK_SHA256"; do
  [[ "$VALUE" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
for REQUIRED in "$PROJECT_ROOT" "$VENV_DIR"; do
  [[ -d "$REQUIRED" ]] || { echo "missing directory: $REQUIRED" >&2; exit 2; }
done
for REQUIRED in "$MANIFEST" "$ENV_LOCK" "$SCRIPT_PATH" "$EXECUTED_SCRIPT_PATH"; do
  [[ -f "$REQUIRED" && ! -L "$REQUIRED" ]] || {
    echo "missing regular file: $REQUIRED" >&2
    exit 2
  }
done

[[ "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)" == "$MANIFEST_SHA256" ]] || exit 2
[[ "$(sha256sum "$SCRIPT_PATH" | cut -d ' ' -f 1)" == "$EXPECTED_SCRIPT_SHA256" ]] || exit 2
[[ "$(sha256sum "$EXECUTED_SCRIPT_PATH" | cut -d ' ' -f 1)" == "$EXPECTED_SCRIPT_SHA256" ]] || exit 2
[[ "$(sha256sum "$ENV_LOCK" | cut -d ' ' -f 1)" == "$EXPECTED_ENV_LOCK_SHA256" ]] || exit 2

module purge
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || exit 2
[[ "$(git branch --show-current)" == "agent/sliding-wobble-eradication-20260814" ]] || exit 2
git diff --quiet HEAD -- || { echo "tracked source differs from HEAD" >&2; exit 2; }
git diff --cached --quiet || { echo "staged source differs from HEAD" >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?missing Slurm array task ID}" in
  0) RECIPE=task55_clean ;;
  1) RECIPE=task80_assisted ;;
  *) echo "Slurm array task is outside the two smoke recipes" >&2; exit 2 ;;
esac

echo "gate=same_frame_equivariance_one_epoch_smoke recipe=$RECIPE job_id=$SLURM_JOB_ID host=$(hostname)"
nvidia-smi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SELF_EQUIVARIANCE_STAGE_LAUNCH_LOCK="$MANIFEST"
export SELF_EQUIVARIANCE_STAGE_LAUNCH_LOCK_SHA256="$MANIFEST_SHA256"
export SELF_EQUIVARIANCE_ENV_LOCK="$ENV_LOCK"
export SELF_EQUIVARIANCE_ENV_LOCK_SHA256="$EXPECTED_ENV_LOCK_SHA256"
export SELF_EQUIVARIANCE_SLURM_SCRIPT="$SCRIPT_PATH"
export SELF_EQUIVARIANCE_SLURM_SCRIPT_SHA256="$EXPECTED_SCRIPT_SHA256"

srun python -B -u -m keypoint_net.run_same_frame_equivariance_smoke execute \
  --repo-root "$PROJECT_ROOT" \
  --manifest-path "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --recipe "$RECIPE"
