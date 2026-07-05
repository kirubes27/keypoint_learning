# Conditional dead-zone fallback R1 cluster lock — 2026-07-05

## Frozen launch

- branch: `fitted-operator-diagnostics-20260704`;
- run count: three GPU tasks, seeds 42/43/44;
- object/frames: hammer, frames 0/3/6/9;
- K=10, standard 64x64 architecture;
- coordinate MSE plus `conditional_deadzone`;
- shape weight: `8.723808430294485e-06`;
- maximum updates: 5,000, evaluation every 100;
- expected active GPU runtime: under five minutes total based on the previous
  R1 gate; scheduler queue time is unknown;
- cluster script: `cluster/stage_r1_deadzone_gate.slurm`;
- collector: `cluster/fetch_stage_r1_deadzone_gate_to_mac.sh`;
- Mac destination:
  `/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r1_deadzone_gate_<timestamp>`.

## Gate

A seed passes only if coordinate, heatmap-shape, dominant-mode-mass and
counterfactual-gradient conditions all pass. Aggregate PASS requires at least
2/3 seed passes and no physical target failing in at least 2/3 seeds.

Failure is terminal for this instrument design under the frozen fallback rule;
do not tune the loss or launch R2. Success authorizes R2 planning, not Stage B
directly.

The exact launch commit is recorded by the job log and must match the pushed
branch head before interpretation.
