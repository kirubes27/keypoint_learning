#!/bin/bash
#SBATCH -J collect_hammer
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -o /work/scratch/%u/phd_phase_a/slurm_logs/%x.out.%j
#SBATCH -e /work/scratch/%u/phd_phase_a/slurm_logs/%x.err.%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/scratch/$USER/phd_phase_a}"
VENV_DIR="${VENV_DIR:-/work/scratch/$USER/venvs/phd_keypoint}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/keypoint_net/runs_full_sweep_hammer_shared_affine}"
SWEEP_OUT="${SWEEP_OUT:-$RUNS_DIR/sweep}"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$SWEEP_OUT"

module purge
module load gcc openmpi
module load python
source "$VENV_DIR/bin/activate"

cd "$PROJECT_ROOT"

srun python -u keypoint_net/sweep.py \
  --data_root "$DATA_ROOT" \
  --object engineers_hammer_vray \
  --operator_type shared_affine \
  --collect_only \
  --runs_dir "$RUNS_DIR" \
  --sweep_output_dir "$SWEEP_OUT"

echo "Summary files:"
ls -lh "$SWEEP_OUT"
