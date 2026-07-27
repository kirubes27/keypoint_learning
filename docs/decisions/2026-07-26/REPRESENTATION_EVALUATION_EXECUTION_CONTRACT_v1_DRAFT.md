# Representation evaluation execution contract v1 (draft)

Date: 2026-07-27

Status: **draft semantic lock; not oracle execution authority**

Parent specifications:

- `REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.md`
- `REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1_1_NUMERIC_AMENDMENT.md`

Frozen split bundle:

- source commit:
  `d325cdb9bdf07dd3d215b1f9e3153012e2065f5b`
- artifact commit:
  `78071297ba46bb22b637a00e1c141eb6e9a8f2de`
- `SPLIT_MANIFEST.json` file SHA-256:
  `9b70db2b2f30e151c1322d7839715167b9dbf750f4caa0b6986a31218ac7ed55`
- `SPLIT_MANIFEST.json` canonical-content SHA-256:
  `aae48879593e6bb7e4c9fdd4706a71f09785b8ebee9d1a7662c0fce62d640a7d`

## 1. Decision lock

**Decision.** Freeze the executable boundary between the verified split
artifacts, planted geometry cases, saved-checkpoint replay inputs, and the one
production representation evaluator.

**Current claim.** No official oracle result is valid unless a versioned,
hash-bound adapter constructs the evaluator bundle without changing the
physical meaning of the frozen split fields, and the evaluator independently
rejects inconsistent geometry, provenance, partitions, or registry bindings.

**Known evidence.**

- The 40-file split bundle passed live-corpus inventory validation,
  byte-identical regeneration, and independent split verification.
- The authoritative numeric registry is committed and independently reviewed.
- The first evaluator draft passed synthetic contract tests, but a release audit
  found that its scale, translation, roll, and yaw/pitch vocabulary did not
  match the frozen split artifacts.
- The first evaluator draft accepted caller-supplied state transforms and a
  lexically valid but unverified source-commit string.

**Critical unknown.** Whether the corrected evaluator reproduces the
preregistered estimator, exact-roll, and representation-health planted
answers, rejects their falsifiers, and reruns byte-identically in the frozen
Python 3.10.19 / NumPy 2.2.6 environment. Dataset-backed geometry and saved
checkpoint replay are separate later gates.

**Main failure mode.** A self-consistent but semantically false bundle could
declare the wrong scale quantity, camera sign, physical state, canonical
transform, split partition, or source provenance and still produce plausible
metrics.

**Constraints.**

- No learned checkpoint may be opened until the separate replay registry is
  committed.
- No camera/world-to-image translation magnitude may be invented. Rendered
  translation remains blocked until its calibration artifact is frozen.
- No activity, attachment, separation, collapse, or yaw/pitch approximation
  threshold may be taken from the numeric-correctness registry.
- No oracle output may modify a tolerance or expected case classification.
- This draft does not change the frozen programme order or authorize training.

**Next gate.** Implement and test the statements below without opening a model
or checkpoint. Then obtain a fresh, substantive, read-only Fable 5 High review
of this contract and the exact implementation. Any unresolved P0 finding,
unexpected semantic test result, or non-substantive Fable response blocks the
official planted suite.

## 2. Mandatory split-to-evaluator adapter

The adapter must accept the exact committed split manifest, one exact committed
evaluation-pair artifact, and its exact committed corpus inventory. If and only
if the evaluator is asked to fit an operator from pairs, the adapter must also
accept the exact matching committed `train` pair artifact. It must call the
independent split-bundle validator before constructing a dataset-backed
evaluation bundle. A held-out evaluation artifact must never be treated as a
source of fitting rows, and a single-artifact held-out construction must reject
`fit_from_pairs`.

The following field mapping is exact:

| Frozen pair artifact | Evaluation pair row |
|---|---|
| `src_frame_index` | `source_frame` |
| `dst_frame_index` | `target_frame` |
| `split` | `partition` |
| `src_state` | source per-frame physical state |
| `dst_state` | target per-frame physical state |
| `direction` | `direction`, unchanged |
| `stride` | `stride`, unchanged |

The adapter output keeps `evaluation.frames` / `evaluation.rows` and optional
`fit.frames` / `fit.rows` as structurally separate sections. It must not
concatenate either frames or rows into one collection whose meaning depends on
partition labels.
The adapter must retain and bind:

- split-manifest file and content hashes;
- evaluation-pair-artifact file and content hashes;
- fit-pair-artifact file and content hashes when fitting is requested;
- corpus-inventory file and content hashes;
- dataset binding and source-pair-index hashes;
- generator commit and config hash;
- object, role, transform, direction, stride, and partition;
- equality of object, transform, physical axis, direction, stride, generator
  semantics, dataset binding, inventory binding, and source-pair-index binding
  between the evaluation and optional fit artifacts.

An unversioned dictionary conversion, filename-based guess, or caller-provided
replacement value is forbidden.

### 2.1 Transform normalization

The frozen split meaning remains authoritative:

- **roll:** `signed_generator` is signed physical degrees. The exact image map
  is `R_img(g)`, about the separately bound projected centre.
- **yaw/pitch:** `signed_generator` is signed physical degrees. The expected
  family is a local affine approximation, never an exact finite-arc planar
  rotation.
- **translation:** `signed_generator` is signed world displacement. It is not
  an image-space bias. The target image bias must come from a separate frozen
  camera-calibration artifact.
- **scale:** `signed_generator` is signed log scale. The multiplicative image
  ratio is exactly `r = exp(signed_generator)`. The evaluator must never treat
  the log step itself as the diagonal of `A`.

The adapter may add derived fields only when their formula and source artifact
are named and hash-bound. It may not overwrite the original physical fields.
In particular, the split bundle alone does not supply a projected object
centre, a rendered-translation world-to-image calibration, or yaw/pitch
projection/depth labels. Dataset-backed evaluation requiring one of those
quantities remains blocked until a separate frozen geometry artifact supplies
and binds it. That geometry artifact must also bind the inventory, object,
coordinate convention, crop, resize, and `align_corners` preprocessing to
which its values apply.

## 3. Evaluation strata and fitting

Every result represents one
`(object, seed, transform family, physical axis, direction, stride,
evaluation partition)` stratum.

- Metrics use only `evaluation.frames` and `evaluation.rows` from the named
  evaluation partition.
- A `fit_from_pairs` operator may use only the separately supplied `fit.frames`
  and `fit.rows`, all of which must come from the matching `train` artifact.
- If the evaluation partition is `validation` or `test`, training endpoints and
  evaluation endpoints must be disjoint.
- A row labelled `test`, `validation`, or `full_corpus` must never enter the
  affine fit.
- Directions, strides, axes, families, and object identities may not be pooled.
- Full-primary roll is a separate diagnostic protocol containing exactly the
  180 cyclic `+3` edges, each once, all labelled `full_corpus`.
- A generic roll bundle must not emit metrics labelled full-corpus AUC,
  role-scoped holdout AUC, or closure.

## 4. Physical-state and geometry contract

The bundle must include a finite, explicit physical-state record for every
frame. Evaluation pair rows obtain source and target states by frame lookup;
the caller does not repeat or override them.

Required state meanings are:

- roll: physical `theta_deg`;
- scale: positive scale relative to the canonical object state;
- translation: physical world position plus a calibrated normalized-image
  offset from a frozen camera-calibration artifact;
- yaw/pitch: physical `theta_deg`, projection model, and named depth
  configuration.

For exact planar families, the evaluator derives canonical state transforms
from these states:

- roll: `A_i = R_img(theta_i)`, `b_i = c - A_i c`;
- scale: `A_i = s_i I`, `b_i = c - A_i c`;
- translation: `A_i = I`, `b_i = calibrated_image_offset_i`.

Caller-supplied per-frame `state_A` or `state_b` is forbidden for these
families. Yaw/pitch canonical drift is not applicable. Their pair residual rows
must explicitly store source angle, target angle, projection model, depth
configuration, direction, stride, and evaluation partition.

Switching uses the locked exact pair transform for exact planar fixtures and
the explicitly labelled fitted-reference transform for yaw/pitch. It must not
use an arbitrary per-frame transform supplied by the caller.

## 5. Provenance contract

The source commit must be a full lowercase 40-hex Git commit that exists in the
repository. The running evaluator source must be byte-identical to the blob at
that commit.

Every case must bind mandatory, uniquely named file roles:

- all cases: authoritative numeric registry, evaluator source, oracle case
  manifest, and oracle harness/adapter source;
- planted cases: calibration fixture manifest and the planted-case definition;
- dataset-backed cases: split manifest, pair artifact, and corpus inventory;
- checkpoint cases: checkpoint, checkpoint config, dataset-backed bindings,
  metadata, frame/mask inventory, and replay registry.

Every file is rehashed from its absolute path. A resealed JSON object containing
only plausible hash strings is insufficient. Duplicate roles, duplicate paths,
unknown mandatory-role substitutions, missing Git blobs, or byte mismatches are
fatal.

The only accepted numeric registry is the exact v1.1 path, file hash, content
hash, and schema frozen in the numeric amendment. Coordinate tolerance is
resolved internally from the estimator softmax dtype. A caller-provided
coordinate tolerance is forbidden.

## 6. Representation-health semantics

- `motion_fraction_min` must be strictly positive. Zero cannot classify a
  static channel as active.
- Motion activity, on-object attachment, and heatmap-flat/dead status remain
  separate fields.
- The active/on-object eligibility denominator is exactly activity AND
  attachment. Heatmap death is reported separately and must not silently change
  that denominator.
- All-channel pair metrics remain mandatory for collapse detection.
- Pairs with no jointly visible frame are explicitly void and cannot count as
  healthy, separate, or persistent duplicates.
- Visibility gaps break close-run duration.
- The Task 20 structural negative-control flag is the frozen all-channel
  persistent-duplicate category. It is not the later scientific coincidence
  trigger.

## 7. Required pre-execution evidence

### 7.1 Immediate estimator/roll/representation-health gate

Before the current official planted command, the
authoritative-environment contract suite must prove at least:

1. exact numeric-registry binding and caller-tolerance rejection;
2. the actual 64- and 128-resolution estimator path in both float32 and
   float64;
3. exact positive world-Z roll in image coordinates, a wrong-sign falsifier,
   exact off-centre bias, and reflection rejection;
4. separated, coincident, sliding, static/inactive, off-object, peak-switching,
   dead-heatmap, and coordinate-only/void evidence behavior;
5. zero activity threshold rejection and separate dead-heatmap reporting;
6. all-channel collapse when eligible-only evidence is void;
7. generic-roll AUC/closure suppression;
8. strict JSON, source-commit/blob, role, path, and file-hash failures;
9. byte-identical canonical results across two fresh official constructions.

The saved Task 20/55/80 replay is not part of this planted command. It remains
blocked until the committed replay registry and a reviewed dataset-backed roll
geometry binding have both passed their own gates.

### 7.2 Deferred family-specific evidence

The following checks remain mandatory before their respective transformation
families are opened, but are not prerequisites for the immediate roll-only
planted command:

1. exact split transform normalization, including log-scale-to-ratio;
2. train-only fitting and test/validation leakage rejection;
3. calibrated translation sign and swapped-axis rejection;
4. family-specific reflection handling;
5. explicit yaw/pitch angle, projection, depth, and visibility strata.

Passing tests, valid JSON, correct file counts, or a zero process exit status do
not by themselves pass the oracle gate.

## 8. Review status

An attempted interim Fable review on 2026-07-27 returned only
`Execution error`; it is not a review and supplies no evidence. A fresh Fable 5
High call, using subscription authentication and the final raw contract/code
boundary, remains mandatory before official oracle execution.
