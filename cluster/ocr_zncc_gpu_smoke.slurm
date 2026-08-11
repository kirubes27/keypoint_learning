#!/bin/bash
#SBATCH -J ocr_zncc_smoke
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%j.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%j.err

set -euo pipefail

EXPECTED_COMMIT="${OCR_EXPECTED_COMMIT:?set OCR_EXPECTED_COMMIT}"
PROJECT_ROOT="${OCR_PROJECT_ROOT:?set OCR_PROJECT_ROOT}"
VENV_DIR="${OCR_VENV_DIR:?set OCR_VENV_DIR}"
MANIFEST="${OCR_MANIFEST:?set OCR_MANIFEST}"
MANIFEST_SHA256="${OCR_MANIFEST_SHA256:?set OCR_MANIFEST_SHA256}"
OUTPUT="${OCR_SMOKE_OUTPUT:?set OCR_SMOKE_OUTPUT}"
SMOKE_LOCK="${OCR_SMOKE_LAUNCH_LOCK:?set OCR_SMOKE_LAUNCH_LOCK}"
SMOKE_LOCK_SHA256="${OCR_SMOKE_LAUNCH_LOCK_SHA256:?set OCR_SMOKE_LAUNCH_LOCK_SHA256}"
EXPECTED_SCRIPT_SHA256="${OCR_SLURM_SCRIPT_SHA256:?set OCR_SLURM_SCRIPT_SHA256}"
EXPECTED_ENV_LOCK_SHA256="${OCR_ENV_LOCK_SHA256:?set OCR_ENV_LOCK_SHA256}"
ENV_LOCK="$VENV_DIR/FRESH_ROLL_ENVIRONMENT.json"
EXECUTED_SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_PATH="$(realpath "$PROJECT_ROOT/cluster/ocr_zncc_gpu_smoke.slurm")"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "invalid expected commit" >&2
  exit 2
}
for VALUE in "$MANIFEST_SHA256" "$SMOKE_LOCK_SHA256" \
             "$EXPECTED_SCRIPT_SHA256" "$EXPECTED_ENV_LOCK_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid OCR smoke SHA-256 input" >&2
    exit 2
  fi
done
for REQUIRED in "$PROJECT_ROOT" "$VENV_DIR"; do
  [[ -d "$REQUIRED" ]] || { echo "missing directory: $REQUIRED" >&2; exit 2; }
done
for REQUIRED in "$MANIFEST" "$SMOKE_LOCK" "$ENV_LOCK" "$SCRIPT_PATH" \
                "$EXECUTED_SCRIPT_PATH"; do
  [[ -f "$REQUIRED" && ! -L "$REQUIRED" ]] || { echo "missing regular file: $REQUIRED" >&2; exit 2; }
done
[[ "$PROJECT_ROOT" == /* && "$VENV_DIR" == /* && "$MANIFEST" == /* \
   && "$SMOKE_LOCK" == /* && "$OUTPUT" == /* ]] || {
  echo "OCR smoke paths must be absolute" >&2
  exit 2
}
[[ ! -e "$OUTPUT" ]] || {
  echo "smoke output already exists: $OUTPUT" >&2
  exit 2
}

ACTUAL_SCRIPT_SHA256="$(sha256sum "$SCRIPT_PATH" | cut -d ' ' -f 1)"
EXECUTED_SCRIPT_SHA256="$(sha256sum "$EXECUTED_SCRIPT_PATH" | cut -d ' ' -f 1)"
ACTUAL_ENV_LOCK_SHA256="$(sha256sum "$ENV_LOCK" | cut -d ' ' -f 1)"
ACTUAL_MANIFEST_SHA256="$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
ACTUAL_SMOKE_LOCK_SHA256="$(sha256sum "$SMOKE_LOCK" | cut -d ' ' -f 1)"
[[ "$ACTUAL_SCRIPT_SHA256" == "$EXPECTED_SCRIPT_SHA256" ]] || exit 2
[[ "$EXECUTED_SCRIPT_SHA256" == "$EXPECTED_SCRIPT_SHA256" ]] || exit 2
[[ "$ACTUAL_ENV_LOCK_SHA256" == "$EXPECTED_ENV_LOCK_SHA256" ]] || exit 2
[[ "$ACTUAL_MANIFEST_SHA256" == "$MANIFEST_SHA256" ]] || exit 2
[[ "$ACTUAL_SMOKE_LOCK_SHA256" == "$SMOKE_LOCK_SHA256" ]] || exit 2

module purge
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
ACTUAL_BRANCH="$(git branch --show-current)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || { echo "commit differs" >&2; exit 2; }
[[ "$ACTUAL_BRANCH" == "agent/ocr-zncc-training-20260811" ]] || { echo "branch differs" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || {
  echo "source checkout is not completely clean" >&2
  exit 2
}

OCR_ENV_LOCK_PATH="$ENV_LOCK" \
OCR_ENV_LOCK_EXPECTED_SHA256="$EXPECTED_ENV_LOCK_SHA256" \
OCR_MANIFEST_PATH="$MANIFEST" \
OCR_MANIFEST_EXPECTED_SHA256="$MANIFEST_SHA256" \
OCR_SCRIPT_PATH="$SCRIPT_PATH" \
OCR_SCRIPT_EXPECTED_SHA256="$EXPECTED_SCRIPT_SHA256" \
OCR_SMOKE_LOCK_PATH="$SMOKE_LOCK" \
OCR_SMOKE_LOCK_EXPECTED_SHA256="$SMOKE_LOCK_SHA256" \
OCR_SOURCE_COMMIT="$EXPECTED_COMMIT" python - <<'PY'
import hashlib, importlib.metadata, json, os, platform, subprocess
from pathlib import Path
import numpy, torch

def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()

def record(variable, expected_hash_variable):
    path = Path(os.environ[variable])
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{variable} is not an absolute regular file")
    value = {
        "absolute_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    if value["sha256"] != os.environ[expected_hash_variable]:
        raise RuntimeError(f"{variable} hash differs")
    return value

environment_record = record("OCR_ENV_LOCK_PATH", "OCR_ENV_LOCK_EXPECTED_SHA256")
manifest_record = record("OCR_MANIFEST_PATH", "OCR_MANIFEST_EXPECTED_SHA256")
script_record = record("OCR_SCRIPT_PATH", "OCR_SCRIPT_EXPECTED_SHA256")
smoke_record = record("OCR_SMOKE_LOCK_PATH", "OCR_SMOKE_LOCK_EXPECTED_SHA256")
smoke_lock = json.loads(Path(smoke_record["absolute_path"]).read_text())
smoke_payload = dict(smoke_lock)
smoke_claimed = smoke_payload.pop("content_hash_sha256", None)
if canonical_hash(smoke_payload) != smoke_claimed:
    raise RuntimeError("smoke launch-lock content hash differs")
if (smoke_lock.get("schema_version") != "ocr_zncc_gpu_smoke_launch_lock.v1"
        or smoke_lock.get("artifact_type") != "ocr_zncc_gpu_smoke_launch_lock"
        or smoke_lock.get("source_commit") != os.environ["OCR_SOURCE_COMMIT"]
        or smoke_lock.get("source_branch")
        != "agent/ocr-zncc-training-20260811"
        or smoke_lock.get("scientific_result_authorized") is not False):
    raise RuntimeError("smoke launch-lock semantics differ")
locked_manifest = smoke_lock.get("manifest")
if (not isinstance(locked_manifest, dict)
        or locked_manifest.get("absolute_path") != manifest_record["absolute_path"]
        or locked_manifest.get("sha256") != manifest_record["sha256"]
        or locked_manifest.get("size_bytes") != manifest_record["size_bytes"]):
    raise RuntimeError("smoke launch-lock manifest record differs")
if smoke_lock.get("environment_lock") != environment_record:
    raise RuntimeError("smoke launch-lock environment record differs")
if smoke_lock.get("slurm_script") != script_record:
    raise RuntimeError("smoke launch-lock Slurm record differs")

manifest = json.loads(Path(manifest_record["absolute_path"]).read_text())
manifest_payload = dict(manifest)
manifest_claimed = manifest_payload.pop("content_hash_sha256", None)
if canonical_hash(manifest_payload) != manifest_claimed:
    raise RuntimeError("manifest content hash differs")
recipe = manifest.get("decision_lock", {}).get("authorized_recipe")
if recipe not in {"task55_clean", "task80_assisted"}:
    raise RuntimeError("smoke manifest recipe is invalid")
expected_cells = [
    f"{recipe}__control__seed42__smoke",
    f"{recipe}__ocr_zncc__seed42__smoke",
]
if smoke_lock.get("cells") != expected_cells:
    raise RuntimeError("smoke launch-lock cell set differs")
if smoke_lock.get("recipe") != recipe:
    raise RuntimeError("smoke launch-lock recipe differs")
if (locked_manifest.get("content_hash_sha256") != manifest_claimed
        or manifest.get("source_commit") != os.environ["OCR_SOURCE_COMMIT"]
        or manifest.get("source_branch")
        != "agent/ocr-zncc-training-20260811"
        or manifest.get("decision_lock", {}).get("authorized_recipe") != recipe):
    raise RuntimeError("smoke manifest lineage differs")

lock = json.loads(Path(environment_record["absolute_path"]).read_text())
if (lock.get("schema_version") != "fresh_roll_cluster_environment.v1"
        or lock.get("artifact_type")
        != "fresh_roll_pinned_cluster_environment"
        or lock.get("scope")
        != "shared_unchanged_environment_for_cuda_smoke_and_frozen_matrix"):
    raise RuntimeError("environment-lock semantics differ")
live = {
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "torch": str(torch.__version__),
    "torchvision": importlib.metadata.version("torchvision"),
    "numpy": str(numpy.__version__),
    "Pillow": importlib.metadata.version("Pillow"),
    "pytorch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
}
if live != lock.get("versions"):
    raise RuntimeError(f"environment versions differ: {live!r}")
freeze = subprocess.run(
    [os.sys.executable, "-m", "pip", "freeze", "--all"], check=True,
    capture_output=True, text=True,
).stdout.encode()
if hashlib.sha256(freeze).hexdigest() != lock["pip_freeze"]["sha256"]:
    raise RuntimeError("pip freeze differs")
PY

echo "gate=ocr_zncc_gpu_smoke job_id=$SLURM_JOB_ID host=$(hostname)"
nvidia-smi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OCR_STAGE_LAUNCH_LOCK="$SMOKE_LOCK"
export OCR_STAGE_LAUNCH_LOCK_SHA256="$SMOKE_LOCK_SHA256"
export OCR_ENV_LOCK="$ENV_LOCK"
export OCR_ENV_LOCK_SHA256="$EXPECTED_ENV_LOCK_SHA256"
export OCR_SLURM_SCRIPT="$SCRIPT_PATH"
export OCR_SLURM_SCRIPT_SHA256="$EXPECTED_SCRIPT_SHA256"
srun python -B -u -m keypoint_net.diagnostics.run_ocr_zncc_gpu_smoke \
  --expected-commit "$EXPECTED_COMMIT" \
  --experiment-manifest "$MANIFEST" \
  --experiment-manifest-sha256 "$MANIFEST_SHA256" \
  --output "$OUTPUT"
