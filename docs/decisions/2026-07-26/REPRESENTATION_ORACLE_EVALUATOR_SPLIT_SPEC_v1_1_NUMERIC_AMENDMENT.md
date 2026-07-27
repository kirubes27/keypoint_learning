# Representation-oracle specification v1.1: numeric calibration amendment

Date: 2026-07-27

Parent specification:
`REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.md`

Status: **independently reviewed; approved with the required changes recorded
below; not execution authority until this file and the authoritative
calibration artifact are committed**

## 1. Decision lock

**Decision.** Freeze the numerical implementation-correctness tolerances that
the production evaluator and deterministic oracle harness must use.

**Current claim.** The independently calibrated tolerances below are all
tighter than the provisional ceilings in v1 and are suitable for detecting
implementation error; they are not scientific quality thresholds.

**Known evidence.**

- The calibration fixtures were preregistered in commit
  `c694af75e803a1ed2f4f3abf2eee141f9ff0d486`.
- The independent reference implementation does not import the production
  evaluator.
- Its exact-expected-magnitude rule is covered by CPU unit tests.
- Every resulting tolerance is below the matching v1 provisional ceiling.

**Critical unknown.** Whether the production evaluator reproduces the
preregistered answers and rejects the preregistered falsifiers. No learned
checkpoint result may answer this question.

**Constraints.** These values may not select a model, define a representation
as scientifically useful, or be widened after a production failure without
stopping for user approval.

**Next gate.** After this amendment is committed, complete the split gate:
content-bound corpus inventories, deterministic generation, byte-identical
regeneration, independent verification of every frozen count/predicate, and
loader anti-aliasing proof. Only after that gate passes may the deterministic
planted cases run through the one production evaluator path. Any unexpected
pass, unexpected failure, hash mismatch, or non-deterministic output stops the
gate.

## 2. Authoritative artifact and provenance

The only authoritative numeric registry is:

`docs/decisions/2026-07-26/representation_oracle_calibration/NUMERIC_CALIBRATION_v1_1.json`

- canonical-content SHA-256:
  `24a2d0eb28d1ec8f26c14064abdb4afcaa51e7cf63d9560e1c2f1c9b4a8118c8`
- file SHA-256:
  `40115c7325858137595ba6e90e36fbd52c9a72da2f16b64d9fd1ad4f7764f420`
- fixture manifest commit:
  `c694af75e803a1ed2f4f3abf2eee141f9ff0d486`
- fixture manifest SHA-256:
  `40435922cad4f877697f4aa026dc64f28b2863c86b025f0b332d3b6b53508b90`
- independent reference commit:
  `cce25862d93238c3df080c25aa5452c4029d699c`
- independent reference source SHA-256:
  `300cbd4289646ce82be72e7477ffaf2f84b4d144cd3bb90ca513b314c3596a1c`
- environment: CPython 3.10.19, NumPy 2.2.6,
  macOS 15.7.2 arm64, little-endian.

The uncommitted file named `NUMERIC_CALIBRATION.json` is an earlier,
non-authoritative draft generated before complete per-metric registry binding.
Its file SHA-256 is
`4e7f79b533fd316fbec04fa0a9348c3d86389136327a65e2dfa5e4a35e5d31d3`
and its claimed internal content SHA-256 is
`709be0f7530593474bbfe07bc3f5e18f87b2ab6d8b5bffda5b8aedd3a3ecef7b`.
It is superseded but is not deleted or moved.
Its presence must never be resolved by filename preference: any consumer that
does not bind the exact v1.1 path, file hash, content hash, and schema above
must fail closed.

The bindings above were verified before this amendment commit:

- `git show c694af75...:CALIBRATION_FIXTURE_MANIFEST.json` hashes to
  `40435922...`;
- the working-tree fixture manifest has the same hash;
- `git show cce25862...:calibrate_representation_oracle_tolerances.py` hashes
  to `300cbd42...`;
- the working-tree reference source has the same hash;
- removing `content_sha256` from the candidate registry, encoding the remaining
  object as canonical sorted compact JSON, and hashing it reproduces
  `24a2d0eb...`;
- the complete candidate file hashes to `40115c73...`.

## 3. Formula correction and lock

The v1 formula remains:

`T = max(32 * machine_epsilon * S, 4 * E_ref)`.

`S` is exactly:

`max(1, maximum absolute exact expected value of that reported metric over the
preregistered calibration fixtures)`.

Therefore a reported error whose exact expected value is zero uses `S=1`.
It does not inherit the scale of the non-error quantity from which it was
calculated. In particular:

- `signed_angle_error_deg` uses `S=1`, not the planted 6-degree angle;
- `composition_bias_error_l2` uses `S=1`, not the 59-step target bias.

The corresponding non-error quantities retain their own scales:

- `proper_rotation_angle_deg` uses `S=6`;
- `composition_bias_element` uses `S=4.72`.

This correction tightens two draft tolerances. It does not widen any tolerance
and was made before an authorized production-oracle run.

## 4. Frozen numerical registry

All comparisons are absolute unless a case manifest explicitly names an exact
categorical/count/hash comparison. Array-valued metrics apply the named
tolerance elementwise.

| Registry key | Frozen tolerance |
|---|---:|
| `float32::affine_coordinate` | `3.814697265625e-06` |
| `float32::spatial_expectation_coordinate` | `3.814697265625e-06` |
| `float64::affine_bias_element` | `7.105427357601002e-15` |
| `float64::affine_matrix_element` | `7.105427357601002e-15` |
| `float64::anisotropy` | `7.105427357601002e-15` |
| `float64::bias_error_l2` | `7.105427357601002e-15` |
| `float64::canonical_drift_rms` | `7.105427357601002e-15` |
| `float64::closure_model_mse` | `7.105427357601002e-15` |
| `float64::composition_bias_element` | `3.353761712787673e-14` |
| `float64::composition_bias_error_l2` | `1.1916163935586911e-14` |
| `float64::composition_matrix_element` | `7.105427357601002e-15` |
| `float64::determinant_A` | `7.105427357601002e-15` |
| `float64::full_rollout_auc` | `7.105427357601002e-15` |
| `float64::matrix_error_fro` | `7.105427357601002e-15` |
| `float64::mean_scale` | `7.602807272633072e-15` |
| `float64::off_diagonal_shear_residual` | `7.105427357601002e-15` |
| `float64::proper_rotation_angle_deg` | `4.263256414560601e-14` |
| `float64::proper_rotation_matrix_element` | `7.105427357601002e-15` |
| `float64::rollout_identity_normalized_ratio` | `7.105427357601002e-15` |
| `float64::rollout_model_mse` | `7.105427357601002e-15` |
| `float64::signed_angle_error_deg` | `7.105427357601002e-15` |
| `float64::spatial_expectation_coordinate` | `7.105427357601002e-15` |

Fractions, counts, categories, hashes, split predicates, and reflection
categories use exact equality.

The exact production lookup-key mapping is frozen:

- manifest class `float32`, metric `float32_affine_coordinate` maps to
  `float32::affine_coordinate`;
- manifest class `float32`, metric `spatial_expectation_coordinate` maps to
  `float32::spatial_expectation_coordinate`;
- manifest class `float64_estimator`, metric
  `spatial_expectation_coordinate` maps to
  `float64::spatial_expectation_coordinate`;
- every manifest class `float64` metric `M` maps to `float64::M`.

An unknown key, duplicate key, missing key, or different mapping is fatal. A
consumer may not fall back to a dtype-wide or provisional ceiling.

The production evaluator must not accept a caller-chosen coordinate
consistency tolerance. It must resolve the tolerance from this exact registry:

- production float32 checkpoint logits:
  `float32::spatial_expectation_coordinate`;
- float64 planted logits:
  `float64::spatial_expectation_coordinate`.

## 5. Separate replay and scientific registries

This amendment does not turn historical checkpoint values into exact answers.
Before Task 20/55/80 checkpoint execution, a separate committed replay registry
must bind:

- exact `best_model.pt`, config, dataset inventory, pair-index, metadata, frame,
  and mask hashes;
- definition-identical historical fields only;
- the already-used replay rule
  `absolute_difference <= 2e-6 OR relative_difference <= 5e-4`.

New all-channel collapse, absolute-activity, attachment, and switching fields
are not definition-identical to the historical reanalysis and must not be
forced to reproduce its eligible-only values.

Scientific decision thresholds remain a third, separate registry. They remain
unfrozen and unauthorized by this amendment.

Checkpoint replay therefore remains blocked after this numeric amendment. A
further committed replay-registry amendment is required before Task 20, 55, or
80 is opened by the production path.

## 6. Early-call deviation and non-contamination rule

On 2026-07-27, during a read-only implementation audit, the production
`evaluate_bundle` function was called twice on synthetic in-memory bundles
before this amendment was committed. No checkpoint, model, dataset, or saved
result was opened or run, and no result was written. The calls planted one
wrong-sign roll and one non-roll reflection.

This was an ordering error. Those observed outputs:

- did not contribute to `E_ref`, `S`, or any tolerance above;
- may not be used to alter an expected case classification;
- may not justify widening a tolerance;
- must be rerun only after this amendment is independently reviewed and
  committed, as preregistered regression tests;
- require a code review proving that future audit instructions explicitly
  prohibit production-function calls before an execution lock opens.

The observed classifications are quarantined as non-evidence. Both falsifiers
must be reconstructed from the committed oracle-case manifest and rerun fresh
only in the official post-split-gate oracle suite.

## 7. Environment and scope lock

This registry is valid only for the exact fixture-manifest hash, reference-code
hash, lookup-key mapping, and environment recorded above. On a different
environment:

1. first verify the committed registry unchanged;
2. do not silently regenerate it;
3. if numerical behavior exceeds a frozen tolerance, stop and report the
   environment difference;
4. any proposal to recalibrate or widen a value requires a new reviewed
   amendment and user approval before another production call.

The numeric registry provides implementation-correctness bounds only. It
cannot be reused as:

- a historical replay tolerance;
- an activity, attachment, separation, coincidence, drift, or collapse
  scientific boundary;
- a yaw/pitch approximation-quality threshold;
- a recipe selector, noninferiority margin, or winner tie-break.

No further production-evaluator call is authorized until both this amendment
is committed and the split gate passes.

## 8. Independent review

Fable 5 at high effort independently reviewed the parent specification,
fixture manifest, reference implementation, candidate registry, and CPU
contract test without seeing this amendment draft. Its verdict was
`APPROVE WITH REQUIRED CHANGES`.

Fable independently supported the formula, corrected `S` semantics, arithmetic,
ceilings, registry separation, and the conclusion that the early calls require
an erratum/quarantine rather than recalibration. It required the replay-gap,
old-artifact hashes, exact lookup-key mapping, Git/blob hash verification,
environment lock, and split-before-oracle ordering now recorded above.

Raw review:
`FABLE_5_HIGH_NUMERIC_AMENDMENT_REVIEW_2026-07-27.md`, SHA-256
`6f9b6cf27411133fcb1798df4532627d4837e39255524833018f1afa929b7d87`.
