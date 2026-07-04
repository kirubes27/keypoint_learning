# Lichtenberg Cluster Runbook

This folder prepares the Phase A keypoint project for Lichtenberg without
running the full sweep blindly.

## Semantic Lock

Must be true before a full sweep:

- Code runs through Slurm, not directly on a login node.
- The job script omits `#SBATCH -A`; your successful CPU test showed the
  submit plugin assigns project `10003029` automatically.
- CUDA is visible inside a Slurm GPU allocation.
- The 512 roll dataset exists on the cluster at the path used by the job.
- A one-epoch GPU training smoke completes and records `device: cuda`.
- A one-config Stage 1 sweep gate completes before launching the 432-config
  hammer sweep.

Must not happen:

- Do not use local Mac paths like `/Users/kirubeso.r/...` inside Slurm jobs.
- Do not run `python keypoint_net/sweep.py ...` on the login node.
- Do not sync old `keypoint_net/runs/` artifacts unless you explicitly need
  them; they are large and not required for new cluster runs.

Evidence required:

- `cuda_smoke.out.<jobid>` shows `torch.cuda.is_available=True`.
- `gpu_train_smoke.out.<jobid>` shows training selected `Device: cuda`.
- The one-config gate produces a run directory with `best_model.pt` and rollout
  metrics before the full sweep is submitted.

## Local To Cluster Paths

Local Mac project:

```bash
/Users/kirubeso.r/Documents/PhD
```

Cluster project root used by these scripts:

```bash
/work/scratch/$USER/phd_phase_a
```

Cluster dataset root used by these scripts:

```bash
/work/scratch/$USER/phd_phase_a/tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2
```

The trailing space in `tdw_phase_a_starter ` is real in this repository, so all
scripts quote the path.

## Step 1: Sync Code And Dataset From Your Mac

Run this on your Mac, not on the cluster:

```bash
bash cluster/sync_to_lcluster.sh
```

This copies source code, cluster scripts, docs, and the 512 roll dataset to
`/work/scratch/$USER/phd_phase_a` on Lichtenberg. It excludes old run artifacts.

Royal TSX can do the same file-transfer job through SFTP if you prefer a GUI,
but `rsync` is safer for repeated updates because it only transfers changes.

## Step 2: Create The Cluster Python Environment

SSH into Lichtenberg:

```bash
ssh -i ~/.ssh/id_ed25519_lichtenberg ko75kamy@lcluster13.hrz.tu-darmstadt.de
```

Then run on the cluster:

```bash
cd /work/scratch/$USER/phd_phase_a
bash cluster/setup_env_on_lcluster.sh
```

This creates:

```bash
/work/scratch/$USER/venvs/phd_keypoint
```

## Step 3: CUDA Smoke Test

Run on the cluster:

```bash
cd /work/scratch/$USER/phd_phase_a
sbatch cluster/cuda_smoke.slurm
```

Check:

```bash
squeue -u $USER
cat /work/scratch/$USER/phd_phase_a/slurm_logs/cuda_smoke.out.*
cat /work/scratch/$USER/phd_phase_a/slurm_logs/cuda_smoke.err.*
```

Pass condition: output says CUDA is available and prints a GPU name.

## Step 4: One-Epoch GPU Training Smoke

Run:

```bash
sbatch cluster/gpu_train_smoke.slurm
```

This is not a real experiment. It runs `train.py` for one epoch at `512x512` on
the hammer object. It validates imports, data paths, CUDA, batch shape, model
forward/backward, and write permissions.

Pass condition: output shows `Device: cuda` and completes with exit code `0`.

## Step 5: One-Config Stage 1 Gate

Only after the one-epoch smoke passes:

```bash
sbatch cluster/sweep_hammer_one_config_stage1.slurm
```

This runs the exact `sweep.py --stage 1` path for one filtered config:

```text
lambda_act=1.0,lambda_cycle=0.1,lambda_disp=0.1,lambda_ent=0.01,
lambda_inv=0.1,lambda_loc=0.0,lambda_smooth=0.0
```

Pass condition: a run directory contains `best_model.pt`, `history.json`, and
`rollout/rollout_metrics.json`.

## Step 6: Full Hammer Sweep

Only after the one-config gate passes:

```bash
python cluster/make_hammer_sweep_filters.py cluster/hammer_sweep_filters.txt
sbatch cluster/sweep_hammer_array_stage1.slurm
```

The array job runs 432 hammer configs as separate GPU jobs. The default array
concurrency is `%2`, meaning at most two configs run at once. Edit the
`#SBATCH --array=1-432%2` line only if you intentionally want a different
concurrency.

## Step 7: Collect Summary

After the array jobs finish:

```bash
sbatch cluster/collect_hammer_sweep.slurm
```

Inspect:

```bash
/work/scratch/$USER/phd_phase_a/keypoint_net/runs_full_sweep_hammer_shared_affine/sweep/sweep_summary.csv
/work/scratch/$USER/phd_phase_a/keypoint_net/runs_full_sweep_hammer_shared_affine/sweep/sweep_results.json
/work/scratch/$USER/phd_phase_a/keypoint_net/runs_full_sweep_hammer_shared_affine/sweep/sweep_summary.png
```
