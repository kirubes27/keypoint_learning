# Next Steps: Instrument Repair to Main Hypothesis Test (2026-07-05)

**Status:** proposed execution plan. This replaces only Stage A of
`FINAL_MATERIAL_KEYPOINT_DECISION_PLAN_2026-07-04.md`. The five-arm Stage-B
design and its decision branches remain unchanged.

## 1. Research question

> Once minimal image-grounded constraints address known trivial solutions,
> does pressure to make transformations simple and predictable provide
> additional pressure toward precise, distinct and materially stable
> keypoints?

The image-level causal test has not yet occurred. It is blocked by one measured
instrument problem: coordinate-MSE training through soft-argmax permits
saturated or malformed heatmaps and can lose a useful coordinate gradient.

The purpose of the next stage is not to improve keypoints directly. It is to
make the coordinate instrument reliable enough that Stage B can test the
research question without this confound.

## 2. Constraints on the repair

The repair must:

- use no true keypoint, mask, transform or semantic annotation in Stage B;
- constrain heatmap **shape**, not choose its image location;
- discourage delta spikes, diffuse maps and separated multiple peaks;
- preserve a finite-width coordinate gradient;
- be identical in every Stage-B arm;
- introduce one primary candidate and no sweep. Any fallback requires a new
  semantic proof that it changes the failed gradient path.

The optional native-quarter stride mode remains off. Current evidence rejects
resolution as the active A0 blocker, so no 64/128 experiment is authorized.

## 3. Stage R0 — freeze the repair semantics

### Primary candidate

For each predicted heatmap distribution `p`, calculate its soft-argmax mean
`mu`. Construct a fixed-width Gaussian `q` centred at `stop_gradient(mu)` and
penalize `JS(p || q)`.

- Gaussian width: `sigma = 1.0` heatmap cell.
- Centre: predicted coordinate, detached before rendering `q`.
- Divergence: Jensen-Shannon.
- Ground-truth coordinate appears only in the separate supervised coordinate
  loss used by Stage A; it is not used by the shape term.
- In Stage B, the same shape term is entirely ground-truth-free.

This is a DSNT-style distribution regularizer adapted from a ground-truth-
centred supervised form to a prediction-centred shape constraint. The adapted
form must not be described as directly validated by the original DSNT paper.

### Weight calibration

No weight sweep. On one fixed seed-42 A0 probe batch at initialization:

`lambda_shape = ||d L_coordinate / d logits|| / max(||d L_shape / d logits||, 1e-12)`

Store the raw norms, formula, sigma and resulting weight before training. Freeze
them for all seeds and all later arms.

### Semantic unit tests

The primary candidate must pass all of these before image training:

1. a matching one-cell Gaussian has near-zero shape loss;
2. a delta spike receives a gradient that broadens it;
3. a uniform map receives a gradient that concentrates it;
4. two separated symmetric peaks receive a gradient toward one unimodal blob;
5. translating an interior heatmap and its detached centre preserves the loss;
6. no gradient flows through the rendered Gaussian centre;
7. all values and gradients remain finite at exact symmetry and collapse.

Also re-measure the existing successful heatmap positive controls in both train
and evaluation mode before repair training. Record their shape distributions;
do not use repair-run outcomes to set the healthy ranges.

**R0 gate:** all tests pass, calibration is recorded and no core training file
has changed. Otherwise stop and repair the implementation only.

## 4. Stage R1 — tiny supervised coordinate gate

Implement the candidate first inside the diagnostic control, not `train.py`.
Use the exact frozen A0 setup:

- one hammer;
- frames 0, 3, 6 and 9;
- K=10;
- standard 64x64 architecture;
- coordinate MSE plus the frozen shape term;
- seeds 42, 43 and 44;
- maximum 5,000 updates;
- no target permutation or new resolution arm.

### R1 pass

At least 2/3 seeds must satisfy all of:

- median training error `<= 0.10` cell64;
- every channel median `<= 0.20` cell64;
- no physical target fails in at least 2/3 seeds;
- every channel's median maximum probability is in `[0.08, 0.30]`;
- every channel's median effective support is in `[8, 32]` cells;
- on a frozen `0.5`-cell counterfactual target shift, run-level median
  coordinate-gradient norm is at least `0.10x` its initialization value and no
  channel median is below `0.01x`.

The shape ranges are deliberately wider than the measured successful heatmap
control (maximum probability 0.130--0.177; effective support 15.9--20.4 cells).

### R1 failure branches

- **Shape is still spiked/diffuse/multimodal and A0 fails:** the penalty did not
  enforce its stated semantics. Stop and review the implementation/calibration.
- **Shape metrics pass but coordinates fail:** the proposed mechanism is
  incomplete. Stop and review.
- **Only 1/3 or 0/3 passes:** fail, regardless of aggregate median.

### Fallback policy

Jakab-style architectural re-rendering remains in the parking lot, not as an
authorized automatic fallback. In the present architecture the operator already
consumes coordinates; re-rendering those coordinates may leave the original
soft-argmax gradient bottleneck unchanged. It can be considered only after a
synthetic gradient-path proof shows that the proposed downstream rewiring
actually bypasses or repairs the measured failure. Any fallback is a new core
architecture, semantic lock and experiment ID.

## 5. Stage R2 — full supervised instrument gate

Only after R1 passes:

1. run one convergence pilot;
2. require the frozen validation plateau rule (minimum 1,000 epochs, evaluate
   every 25, 1% relative-improvement patience of 400 epochs, hard cap 3,000);
3. freeze the complete recipe;
4. train seeds 42, 43 and 44;
5. select checkpoints using validation localization only;
6. evaluate the test split exactly once after all checkpoints are frozen.

The full R2 group goes to the cluster. Before submission, notify the user and
record branch, commit, job script, expected run count and Mac destination.

### Per-seed R2 pass

A seed passes only if all conditions hold:

- stop reason is a genuine validation plateau, not merely the hard cap;
- test median-of-channel median localization error `<= 0.35` cell64;
- test p90 localization error `<= 1.00` cell64;
- no channel median localization error exceeds `0.75` cell64;
- median temporal high-pass jitter `<= 0.30` cell64;
- p90 channel jitter `<= 0.40` cell64;
- on-mask occupancy `>= 0.95`;
- every channel's median maximum probability remains in `[0.08, 0.30]`;
- every channel's median effective support remains in `[8, 32]` cells;
- the frozen counterfactual gradient probe meets the R1 thresholds;
- no NaN, non-finite gradient or persistent catastrophic target.

**R2 PASS:** at least 2/3 seeds pass every condition. Seed is the replication
unit (`n=3`); frames/channels are correlated descriptive measurements.

### Gray zone and failure

- **Gray zone:** median jitter `>0.30` and `<=0.50` in at least 2/3 seeds while
  every other gate passes. Permit one diagnosis-selected improvement iteration;
  write its semantic lock before implementation and do not change thresholds.
- **Fail:** jitter `>0.50`, any other critical gate failure, or failure after
  the one gray-zone iteration. Verdict: instrument-limited; stop and review
  with Angela.

## 6. Stage R3 — freeze the instrument

After R2 passes:

- port the already validated shape term into the shared training loss;
- add equivalence tests against the diagnostic implementation;
- record every core-file change in `docs/CORE_FILE_CHANGELOG.md`;
- freeze architecture, sigma, shape weight, optimizer, splits, stopping rule
  and checkpoint selection in `PRELAUNCH_LOCK.json`;
- use this identical repaired instrument in all five Stage-B arms.

The Stage-B baseline is a **new-instrument baseline**. Historical Task-80 and
smoke results are context, not direct causal comparators.

**R3 gate:** all arm configurations differ only by their named scientific loss
addition. No test result exists yet.

## 7. Stage B — main hypothesis experiment

Run paired seeds 42, 43 and 44 for:

1. repaired-instrument baseline;
2. baseline + non-oracle grounding;
3. baseline + similarity minimal-subset consistency;
4. baseline + grounding + subset consistency;
5. privileged true-mask + true-rotation oracle ceiling.

The oracle is a capability ceiling, not a peer method or component ablation.
All existing Stage-B metrics, checkpoint rules and success/partial/failure
branches remain those in `FINAL_MATERIAL_KEYPOINT_DECISION_PLAN_2026-07-04.md`.

Before the 15-run launch:

- unit-test exact arm composition and finite gradients;
- run a five-epoch meaning smoke for every arm;
- calibrate added scientific loss weights by the existing fixed-probe gradient
  formula and freeze them;
- push the frozen diagnostic branch;
- submit through the standard cluster protocol.

When Stage B completes, automatically validate, archive and copy all outputs and
logs to `PhD/cluster_downloads/<experiment_id>/`; generate the summary locally.
No scientific JSON is transferred through terminal copy/paste.

## 8. Stage-B decision

- **Subset+grounding succeeds:** replicate the frozen configuration across
  multiple objects under the same rotation.
- **Geometric layout improves but canonical attachment does not:** permit the
  already declared descriptor-consistency follow-up.
- **Static-in-support dominates:** permit the already declared fixed-point-aware
  activity follow-up.
- **No meaningful improvement:** stop modifying this subset loss; report the
  negative result.
- **Oracle also fails:** architecture/instrument/evaluation remains inadequate.
- **Oracle succeeds but self-supervised arms fail:** desired coordinates are
  representable but not identifiable from the available self-supervised
  signals—a scientifically useful negative.

No additional diagnostic expansion is allowed after R2 passes. Stage B launches
immediately after R3 implementation equivalence and meaning-smoke gates.

## 9. Runtime and artifact policy

| Stage | Compute | Location | Persistent output |
|---|---|---|---|
| R0 tests/calibration | minutes | Mac | diagnostic outputs + lock |
| R1 one-seed smoke | minutes | Mac | local smoke artifact |
| R1 three seeds | likely under one hour on GPU | cluster | cluster + automatic Mac copy |
| R2 convergence, three seeds | over one hour | cluster | cluster + automatic Mac copy |
| R3 equivalence/smokes | minutes | Mac | tests + prelaunch lock |
| Stage B, 15 runs | over one hour | cluster | cluster + automatic Mac copy |

Every stage ends with a dated Markdown verdict before the next starts. Cluster
state is checked with scheduler commands only; scientific interpretation uses
the copied Mac artifacts.

## 10. Expected sequence

1. R0 implementation and tests: approximately one working day.
2. R1 tiny gate: same day or next cluster window.
3. R2 convergence and one-shot test: approximately one to two days including
   queue time.
4. R3 port/freeze and Stage-B smoke: approximately one working day.
5. Stage-B cluster run and decision report: dependent on measured runtime and
   queue, expected overnight rather than weeks.

These are estimates, not deadlines. Any gate failure follows its declared
branch rather than silently extending the schedule.
