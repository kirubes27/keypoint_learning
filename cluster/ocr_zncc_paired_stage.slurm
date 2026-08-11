#!/bin/bash
#SBATCH -J ocr_zncc_pairs
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH --array=0-2%3
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%A_%a.err

set -euo pipefail

EXPECTED_COMMIT="${OCR_EXPECTED_COMMIT:?set OCR_EXPECTED_COMMIT}"
PROJECT_ROOT="${OCR_PROJECT_ROOT:?set OCR_PROJECT_ROOT}"
VENV_DIR="${OCR_VENV_DIR:?set OCR_VENV_DIR}"
MANIFEST="${OCR_MANIFEST:?set OCR_MANIFEST}"
MANIFEST_SHA256="${OCR_MANIFEST_SHA256:?set OCR_MANIFEST_SHA256}"
STAGE_LOCK="${OCR_STAGE_LAUNCH_LOCK:?set OCR_STAGE_LAUNCH_LOCK}"
STAGE_LOCK_SHA256="${OCR_STAGE_LAUNCH_LOCK_SHA256:?set OCR_STAGE_LAUNCH_LOCK_SHA256}"
EXPECTED_SCRIPT_SHA256="${OCR_SLURM_SCRIPT_SHA256:?set OCR_SLURM_SCRIPT_SHA256}"
EXPECTED_ENV_LOCK_SHA256="${OCR_ENV_LOCK_SHA256:?set OCR_ENV_LOCK_SHA256}"
ENV_LOCK="$VENV_DIR/FRESH_ROLL_ENVIRONMENT.json"
EXECUTED_SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_PATH="$(realpath "$PROJECT_ROOT/cluster/ocr_zncc_paired_stage.slurm")"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "invalid expected commit" >&2
  exit 2
}
for VALUE in "$MANIFEST_SHA256" "$STAGE_LOCK_SHA256" \
             "$EXPECTED_SCRIPT_SHA256" "$EXPECTED_ENV_LOCK_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid OCR paired-stage SHA-256 input" >&2
    exit 2
  fi
done
for REQUIRED in "$PROJECT_ROOT" "$VENV_DIR"; do
  [[ -d "$REQUIRED" ]] || { echo "missing directory: $REQUIRED" >&2; exit 2; }
done
for REQUIRED in "$MANIFEST" "$STAGE_LOCK" "$ENV_LOCK" "$SCRIPT_PATH" \
                "$EXECUTED_SCRIPT_PATH"; do
  [[ -f "$REQUIRED" && ! -L "$REQUIRED" ]] || { echo "missing regular file: $REQUIRED" >&2; exit 2; }
done
[[ "$PROJECT_ROOT" == /* && "$VENV_DIR" == /* && "$MANIFEST" == /* \
   && "$STAGE_LOCK" == /* ]] || {
  echo "OCR paired-stage paths must be absolute" >&2
  exit 2
}

ACTUAL_SCRIPT_SHA256="$(sha256sum "$SCRIPT_PATH" | cut -d ' ' -f 1)"
EXECUTED_SCRIPT_SHA256="$(sha256sum "$EXECUTED_SCRIPT_PATH" | cut -d ' ' -f 1)"
ACTUAL_ENV_LOCK_SHA256="$(sha256sum "$ENV_LOCK" | cut -d ' ' -f 1)"
[[ "$ACTUAL_SCRIPT_SHA256" == "$EXPECTED_SCRIPT_SHA256" ]] || exit 2
[[ "$EXECUTED_SCRIPT_SHA256" == "$EXPECTED_SCRIPT_SHA256" ]] || exit 2
[[ "$ACTUAL_ENV_LOCK_SHA256" == "$EXPECTED_ENV_LOCK_SHA256" ]] || exit 2
[[ "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)" == "$MANIFEST_SHA256" ]] || exit 2
[[ "$(sha256sum "$STAGE_LOCK" | cut -d ' ' -f 1)" == "$STAGE_LOCK_SHA256" ]] || exit 2

module purge
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || exit 2
[[ "$(git branch --show-current)" == "agent/ocr-zncc-training-20260811" ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || exit 2

RECIPE="$(python -B -u -m keypoint_net.ocr_zncc_launch_validation \
  --repo-root "$PROJECT_ROOT" \
  --expected-commit "$EXPECTED_COMMIT" \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --stage-launch-lock "$STAGE_LOCK" \
  --stage-launch-lock-sha256 "$STAGE_LOCK_SHA256" \
  --environment-lock "$ENV_LOCK" \
  --environment-lock-sha256 "$EXPECTED_ENV_LOCK_SHA256" \
  --slurm-script "$SCRIPT_PATH" \
  --slurm-script-sha256 "$EXPECTED_SCRIPT_SHA256")"
[[ "$RECIPE" == "task55_clean" || "$RECIPE" == "task80_assisted" ]] || exit 2

case "${SLURM_ARRAY_TASK_ID:?missing Slurm array task ID}" in
  0) SEED=42 ;;
  1) SEED=43 ;;
  2) SEED=44 ;;
  *) echo "Slurm array task is outside the frozen seed set" >&2; exit 2 ;;
esac
echo "gate=ocr_zncc_paired_stage recipe=$RECIPE seed=$SEED job_id=$SLURM_JOB_ID host=$(hostname)"
nvidia-smi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OCR_STAGE_LAUNCH_LOCK="$STAGE_LOCK"
export OCR_STAGE_LAUNCH_LOCK_SHA256="$STAGE_LOCK_SHA256"
export OCR_ENV_LOCK="$ENV_LOCK"
export OCR_ENV_LOCK_SHA256="$EXPECTED_ENV_LOCK_SHA256"
export OCR_SLURM_SCRIPT="$SCRIPT_PATH"
export OCR_SLURM_SCRIPT_SHA256="$EXPECTED_SCRIPT_SHA256"
srun python -B -u -m keypoint_net.run_ocr_zncc_paired_training run-pair \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --recipe "$RECIPE" \
  --seed "$SEED" \
  --stage-launch-lock "$STAGE_LOCK" \
  --stage-launch-lock-sha256 "$STAGE_LOCK_SHA256"
