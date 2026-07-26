# Decision 2.3 final result — 2026-07-26

Status: **complete; closes negative for a coordinate-head redesign**.

## Plain-language result

All three coordinate readouts passed the frozen supervised test in all three
fresh optimization seeds:

- A: a learned shared linear decoder from centered raw logits — **3/3 pass**;
- B: a learned shared linear decoder from softmax probabilities — **3/3
  pass**;
- C: the unchanged production spatial-softmax expectation — **3/3 pass**.

The current fixed expectation is therefore capable of accurate supervised
material-point localization on this one hammer object's 180-frame, full-circle
in-plane world-Z roll dataset within the matched training budget. The seed-41
Gate 0 failure is not an architectural inevitability of that readout. No
coordinate-head redesign is justified by this panel, and the frozen
seeds-45/46 extension is not triggered.

The calibrated explanation is narrower than the aggregate JSON's shorthand
phrase "matched seed variability dominates": the evidence localizes the prior
failure to the specific seed-41 realization — its random seed and/or training
trajectory — but does not distinguish those two explanations.

## Decision lock

**Decision.** Decide whether bypassing the fixed spatial-softmax expectation
works when learned probability-space and unchanged fixed-expectation controls
do not, strongly enough to justify redesigning the coordinate head.

**Current claim.** Fresh matched training passes the supervised coordinate
thresholds with A, B, and C. The seed-41 failure is not architecturally forced;
no head redesign is supported.

**Known evidence.**

- Gate 0 remains closed at 6/150 correct-dominant-mode high-error pairs
  (4.0 percent versus the frozen 50 percent rule). This result still describes
  that frozen seed-41 checkpoint and continues to demote windowing.
- D0 semantic/unit evidence, the final D1 smoke, all nine D2 training jobs,
  the one-shot D3 test finalization, and the post-finalization immutable
  artifact validator passed.
- Every A/B/C seed passed both unaugmented and fixed-augmented test conditions.
- Five runs reached the 3,000-epoch hard cap without satisfying the frozen
  400-epoch plateau rule.

**Critical unknown.** This experiment does not determine representation
identifiability, material attachment/distinctness of unsupervised keypoints, or
whether those coordinates improve the no-coordinate-label operator objective.

**Constraints.** One object; one correlated world-Z roll orbit; three
optimization seeds; supervised coordinate labels; no yaw, pitch, scale,
translation, other object, or operator training.

**Next gate.** Freeze `COINCIDENCE_RUNFILE.md` with Gate 3a's numeric
antisymmetric-gradient, separation, on-mask, assignment-churn, stochastic-draw,
and tie-breaking criteria before executing Gate 3a. Passing all frozen 3a
criteria authorizes only Gate 3b; any failed critical criterion stops that
branch and requires re-planning.

## Frozen test rule

For each seed, both the unaugmented and fixed-augmented conditions had to meet:

- median-of-channel median error `<= 0.50 cell64`;
- pooled p90 error `<= 1.50 cell64`;
- on-mask fraction `>= 0.95`.

The table reports the worse median, worse p90, and lower on-mask fraction
across those two conditions. Every row had in-range fraction `1.0` and zero
out-of-range coordinates.

| Arm | Seed | Selected epoch | Completed epoch / stop | Worst median | Worst p90 | Min on-mask | Result |
|---|---:|---:|---|---:|---:|---:|---|
| A raw-linear | 42 | 2,000 | 2,400 / plateau | 0.209304 | 0.439496 | 1.000000 | pass |
| A raw-linear | 43 | 2,500 | 2,900 / plateau | 0.168814 | 0.420607 | 1.000000 | pass |
| A raw-linear | 44 | 2,775 | 3,000 / hard cap | 0.161905 | 0.347384 | 1.000000 | pass |
| B probability-linear | 42 | 1,650 | 2,050 / plateau | 0.441296 | 1.123679 | 0.996667 | pass |
| B probability-linear | 43 | 1,700 | 2,100 / plateau | 0.432749 | 1.225730 | 1.000000 | pass |
| B probability-linear | 44 | 2,825 | 3,000 / hard cap | 0.368330 | 1.005395 | 0.998333 | pass |
| C fixed-expectation | 42 | 2,650 | 3,000 / hard cap | 0.205936 | 0.578365 | 1.000000 | pass |
| C fixed-expectation | 43 | 2,875 | 3,000 / hard cap | 0.263691 | 0.654590 | 1.000000 | pass |
| C fixed-expectation | 44 | 2,900 | 3,000 / hard cap | 0.236398 | 0.496381 | 1.000000 | pass |

Arm rule:

- `3/3`: pass;
- exactly `2/3`: provisional, requiring unchanged seeds 45 and 46;
- `0-1/3`: fail.

All arms are `3/3`, so seeds 45 and 46 must not be run as a Decision 2.3
extension. They would be permissible only under a separately preregistered new
experiment.

## What the epoch cap means

Four runs stopped on a genuine validation plateau; five were
`hard_cap_unconverged`:

- A: two plateau, one cap;
- B: two plateau, one cap;
- C: zero plateau, three cap.

This does not overturn the frozen capability verdict: all nine selected
validation checkpoints passed their one-shot test thresholds within the same
matched budget. It does prevent an asymptotic architecture ranking. A's
numerical errors are smaller than B's in this snapshot, but there was no
preregistered superiority margin or hypothesis test, and the architectures
were at different convergence states. The report therefore makes no claim
that A or B is better than C.

## Statistical scope

- **Quantity:** raw per-seed descriptive localization metrics; no error bars.
- **Sample unit:** optimization seed.
- **n:** 3 seeds per arm.
- **Within-seed data:** 60 test frames forming one correlated cyclic orbit;
  frames and channel-frame pairs are not independent replicates.
- **Scope:** one-object instrument capability.
- **Inference:** descriptive only. There is no population-level hypothesis
  test, confidence interval, causal claim, or unseen-object generalization.

## Relation to Gate 0

The two results answer different questions and are not contradictory:

- Gate 0 asked whether changing a frozen seed-41 checkpoint to local/windowed
  expectation matched its representative failure mechanism. Only 4.0 percent
  of high-error pairs had the correct dominant mode, so windowing remains
  demoted.
- Decision 2.3 asked whether fixed versus learned readouts are capable after
  fresh, matched end-to-end training. The unchanged fixed expectation passed
  3/3.

Gate 0 is neither rerun nor retroactively invalidated. Decision 2.3 shows only
that the fixed expectation did not force the seed-41 checkpoint's failure.

## Relevance for training other objects

For the next object, keep the fixed expectation as the default coordinate
readout. A learned A/B head adds parameters but has no demonstrated decision-
level benefit here.

This result does **not** establish:

- transfer of weights or keypoints to another object;
- a common error distribution across objects;
- materially attached, distinct, or identifiable keypoints;
- benefit to the later shared-operator objective;
- readiness to start operator training on new objects.

Object is an untested replication level. Any multi-object conclusion needs a
separately frozen object-level protocol rather than pooling correlated frames.
Under the existing programme, coincidence and geometry/estimator gates still
precede other-object operator training.

## Relevance for yaw and pitch

There is no direct yaw/pitch evidence in this result. In-plane world-Z roll
largely preserves visible surfaces and produces image-plane rotation.
Out-of-plane yaw/pitch introduce perspective, depth-dependent motion,
foreshortening, changing visibility, and self-occlusion. A shared 2D affine
operator can therefore have a geometry floor even with accurate keypoints.

Before yaw/pitch training, the programme still requires:

1. coincidence Gates 3a and 3b;
2. an explicit transformation-geometry and estimator-oracle package;
3. code-path proof of the requested output-space axes plus visual/metric proof
   on generated frames;
4. frozen direction, stride, visibility, pairing, split, seed, budget, and
   shared-affine error-floor rules;
5. dataset manifest and upload verification.

Passing Decision 2.3 removes an unnecessary coordinate-head-redesign detour;
it does not pre-clear any yaw/pitch claim.

## Independent Fable check

Read-only Fable 5 High found no remaining audit blocker and agreed that:

- A, B, and C each pass 3/3;
- seeds 45/46 are off-protocol for this frozen decision;
- no head redesign or architecture ranking is justified;
- hard-cap runs limit convergence/ranking claims but not the threshold
  capability result;
- no other-object, identifiability, operator, or yaw/pitch claim follows.

Fable challenged the machine aggregate's phrase "matched seed variability
dominates the seed-41 result." The final human interpretation adopts Fable's
better-calibrated disjunction: the failure belongs to the specific seed-41
realization, but this panel does not separate random-seed effects from its
training trajectory.

Raw review:
`FABLE_5_HIGH_DECISION23_FINAL_REVIEW_2026-07-26.md`.

## Execution and provenance

- Public branch: `agent/preoperator-gates-20260726`.
- Frozen scientific commit:
  `490bffac39f67cca2a8f2d0f363ea413c0ec6fb0`.
- Bound dataset: `_tdw_world_z_roll_base_panel_512_v2`.
- Clean cluster checkout:
  `/work/scratch/ko75kamy/keypoint_preoperator_gates_20260726`.
- Immutable output root:
  `/work/scratch/ko75kamy/keypoint_preoperator_runs/decision23_490bffac39f67cca2a8f2d0f363ea413c0ec6fb0`.
- Final D1 smoke: job `53632077`, passed.
- D2 exact `0-8%2` matrix: job `53632209`, all nine tasks exited zero.
- D3 one-shot finalizer: job `53632862`, exited zero.
- Test-content manifest:
  `fc033324b777ec4e80a526209d17db2b5929b68378f5985b3e656832fef5b60c`.
- Finalization ledger event:
  `3fd214630a0ef180cb473627263610c00fdbb80a59015157741f4b384e64906b`.
- D1 report SHA-256:
  `dd018c1609017cfbe94d08a96297a7673fb818464391eb4c919fd7faefc3f671`.
- Initial test report SHA-256:
  `45f967066bf0f5bae08a747e6a44f6622a21ef736cfffd8c39bf671247e9873d`.
- Initial test ledger SHA-256:
  `cc9c9c886e5d70bac25da96e5fba8cf969e54dbc3a67db2e9c35126590308c77`.

Repository copies:

- `decision23_results/DECISION23_D1_SMOKE_REPORT.json`;
- `decision23_results/DECISION23_INITIAL_TEST_REPORT.json`;
- `decision23_results/DECISION23_INITIAL_TEST_LEDGER.jsonl`.

The repository validator re-read and hash-checked the exact 3x3 matrix,
configs, checkpoints, selected validation artifacts, one-shot test metrics and
predictions, per-run claim ledgers, aggregate ledger, source/spec/runfile
bindings, and arm-status recomputation after D3. It returned `PASS`.

## Final branch decision

1. Keep C, the current fixed expectation, as the baseline readout.
2. Do not promote A or B and do not reopen windowing.
3. Do not run the Decision 2.3 seed extension.
4. Do not start other-object or yaw/pitch operator training from this result.
5. Freeze the Gate 3a coincidence runfile next; pass authorizes only Gate 3b,
   while failure redirects the coincidence branch.
