# Representative coordinate-instrument pilot lock — 2026-07-05

## Question

Does plain coordinate supervision through the existing 64x64 soft-argmax
instrument converge reliably enough when trained on the representative
60-frame training split, rather than the four-frame tiny-overfit diagnostic?

This pilot tests whether the catastrophic tiny-regime saturation is also the
active blocker in the representative regime. It does not test the paper's
operator hypothesis.

## Must be true

1. One pilot only: seed 41, K=10, standard 64x64 architecture.
2. Plain coordinate MSE only: no JS, dead-zone or other heatmap-shape repair.
3. Use `split_phase_mod6.json`: 60 train, 60 validation and 60 committed test
   frames. The test split remains untouched during the pilot.
4. Train with the existing fixed augmentation, Adam (`lr=1e-4`, weight decay
   `1e-5`) and batch size 16.
5. Evaluate every 25 epochs; require at least 1,000 epochs; stop after 400
   epochs without a 1% relative validation improvement; hard cap 3,000.
6. Select the checkpoint only by the worse of unaugmented and fixed-augmented
   validation median-of-channel median localization error.
7. At every validation evaluation, store per-channel heatmap maximum
   probability, effective support and a half-cell counterfactual coordinate
   gradient. No target or test result changes checkpoint selection.
8. Report every channel. Targets 3, 6 and 9 may be described but cannot be
   excluded or given different thresholds after observing outcomes.

## Must not happen

- The four-frame R1 failure cannot veto this pilot.
- The dense heatmap control cannot be presented as evidence that coordinate
  supervision passed.
- A successful single seed cannot authorize Stage B or support a population
  claim.
- Reaching the hard cap cannot be called convergence.
- Test frames cannot be evaluated until a later three-seed recipe is frozen.

## Pilot interpretation

The pilot is **viable for a frozen three-seed confirmation** only if the best
validation checkpoint satisfies all of:

- stop reason is `validation_plateau`;
- both unaugmented and fixed-augmented median-of-channel medians are `<=0.50`
  cell64;
- both p90 localization errors are `<=1.50` cell64;
- both on-mask fractions are `>=0.95`;
- no channel is simultaneously inaccurate (median error `>0.75` cell64) and
  saturated (median maximum probability `>=0.99`);
- no channel's median counterfactual-gradient norm is below `0.01x` its
  initialization value.

These pilot thresholds decide whether a three-seed confirmation is worth
running. They do not replace the stricter frozen R2 test gate.

If viable, freeze the unchanged recipe and run seeds 42, 43 and 44 with
one-shot test evaluation after all three checkpoints are frozen. If not viable,
do not add another heatmap penalty; compare alternative coordinate instruments.

## Statistical scope

This is one optimization seed on one object and one correlated cyclic orbit.
All frame/channel summaries are descriptive. There are no error bars,
hypothesis tests or generalization claims.
