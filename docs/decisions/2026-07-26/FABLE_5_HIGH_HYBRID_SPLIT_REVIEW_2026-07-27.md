# Fable 5 High independent hybrid-split review — 2026-07-27

Review mode: Fable 5, high effort, read-only `Read`, `Glob`, and `Grep` tools,
`--print`, safe mode, plan permission mode, no session persistence, Claude Max
subscription authentication, and no API-key billing. Fable could inspect files
but could not edit or invoke a shell.

The text below is the raw substantive review returned by Fable.

---

I inspected `keypoint_preoperator_gates/keypoint_net/dataset.py` and `train.py`
read-only. The stated code facts check out exactly: `IndexPairDataset`
(`dataset.py:256`) reads only rotation-style fields (`src_theta_deg`,
`delta_theta_deg`), hardcodes `action_label: 0` for every original pair, and
validates nothing about family/axis/sign/stride; `train.py` sets
`val_dataset = train_dataset` in the index path (`train.py:442`), always builds
a val loader, writes `best_model.pt` on minimum validation loss
(`train.py:670–678`), and has no test-index or fixed-epoch confirmation mode.

## 1. Is the hybrid protocol defensible?

Yes, it is defensible, with two conditions. Development object(s) may use
validation to fix recipe, duration, and checkpoint policy. Confirmation and
final objects then run the frozen recipe once against an untouched test
partition.

The strongest supported claim is: under a recipe, duration, and checkpoint
policy fixed in advance on a development object, fresh object-specific models
trained on within-object state subsets can be evaluated on spatially separated
held-out pairs of the same object, descriptively across non-development
objects and seeds.

It does not support cross-object weight generalization, category-level
generalization, generalization beyond the trained band except for a separately
sealed probe, or inferential statistics treating frames/pairs as independent.

The fixed-final duration/checkpoint rule must be absolute. If a confirmation
result changes the recipe, that object is spent as evidence and becomes
development; final objects must remain untouched by that loop.

## 2. Independent split recommendation

Fable recommended:

- roll skip 3: 138 train frames, 36 holdout frames, two 3-frame wrap-aware
  guards; 135 train and 33 holdout pairs;
- yaw/pitch skip 6: two 45-frame train arms, a 19-frame interior holdout, and
  two 6-frame guards; 78 train and 13 holdout pairs per direction;
- scale skip 4: two 48-frame train arms, a 17-frame interior holdout, and two
  4-frame guards; 88 train and 13 holdout pairs per direction;
- translation stride 3: orthogonal row/column bands with six train bands,
  three guard bands, and two holdout bands; 48 train and 16 holdout pairs per
  axis/direction.

Fable marked the exact proportions as methodological judgment. The final
reconciliation preserves more training frames, using the previously discussed
10–15 percent primary holdout where the one-dimensional corpora permit it,
while retaining Fable's interior-block, wrap-aware, and orthogonal-band
principles. Exact final counts are frozen in
`REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.md`.

## 3. Translation patch versus orthogonal bands

Fable found orthogonal bands decisively preferable. A compact interior patch
with a stride-sized guard ring either retains very few stride-3 holdout edges
or consumes the full 11-by-11 grid. Orthogonal bands retain substantially more
train and holdout pairs, preserve the full transformed-axis range, and remain
valid while x and y operators are never pooled. A pooled x/y model would
require a new joint split.

## 4. Secondary strides

Fable recommended fixed post-freeze stress/composition diagnostics instead of
separately trained models. This probes how a primary-step operator composes
without tripling tuning regimes. The claim cost is explicit: the project may
describe how the primary-step model behaves at other step sizes, but not claim
that separately trained other-stride models work.

## 5. Minimum safe code behavior

Fable required:

1. strict transform/axis/sign/stride/object metadata validation;
2. explicit and disjoint train plus validation indices for development;
3. a fixed-final mode that refuses validation, builds no validation loader,
   runs the frozen epoch count, makes the best-checkpoint branch unreachable,
   and writes an authoritative final checkpoint;
4. test evaluation in a separate one-shot path after training;
5. no partition-unaware whole-directory auto-evaluation;
6. CPU tests for exact counts, endpoint disjointness, guard exclusion,
   translation grid decoding, mode failures, scale-holdout exclusion, and
   byte-identical regeneration.

## 6. Decisions Fable identified

Fable identified four choices requiring an explicit lock:

- the consequence of confirmation failure;
- the exact form of the frozen duration rule;
- whether reverse pairs enter primary training;
- whether yaw/pitch/scale holdouts measure interpolation or endpoint
  extrapolation.

The reconciled specification freezes the confirmation burn rule and interior
interpolation holdouts. Forward and reverse remain separate transformation
families and are never silently pooled. The numeric duration and checkpoint
rule will be frozen in the later training specification before any GPU work.

## Bounded gate

Fable's recommended preparation gate passes only if the generator reproduces
every frozen count; train and holdout endpoint sets are disjoint; no guard
frame enters a retained pair; the scale absolute holdout and secondary strides
enter no selection manifest; regeneration is byte-identical; and fixed-final
mode cannot accept validation or create/select `best_model.pt`. Any single
violation blocks training.
