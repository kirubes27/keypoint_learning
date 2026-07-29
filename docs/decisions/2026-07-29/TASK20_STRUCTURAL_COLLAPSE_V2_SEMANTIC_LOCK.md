# Task 20 Structural-Collapse v2 Semantic Lock

Date: 2026-07-29
Status: candidate semantic amendment under implementation; becomes frozen when
committed, before any v2 checkpoint replay

## Parent documents and precedence

- Project decision synthesis:
  `docs/decisions/2026-07-26/DECISION_SYNTHESIS_v2_2026-07-26.md`
- Evaluator and representation-health definitions:
  `docs/decisions/2026-07-26/REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.md`
- Evaluation execution contract:
  `docs/decisions/2026-07-26/REPRESENTATION_EVALUATION_EXECUTION_CONTRACT_v1_DRAFT.md`
- Checkpoint replay execution specification:
  `docs/decisions/2026-07-26/representation_oracle_replay/CHECKPOINT_REPLAY_EXECUTION_SPEC_v1.md`
- Frozen replay registry:
  `docs/decisions/2026-07-26/representation_oracle_replay/REPLAY_REGISTRY_v1.json`
- Numeric implementation-correctness registry:
  `docs/decisions/2026-07-26/representation_oracle_calibration/NUMERIC_CALIBRATION_v1_1.json`
- Existing reviewed replay boundary:
  `docs/decisions/2026-07-26/FABLE_5_HIGH_CHECKPOINT_REPLAY_RUNTIME_REVIEW_2026-07-28.md`
- Immutable Task 20 v1 result:
  `docs/decisions/2026-07-26/representation_oracle_replay/results/TASK20_CHECKPOINT_REPLAY_RESULT_v1.json`

This document supersedes only the old all-channel structural-collapse
classification predicate as the replay classification gate for Tasks 20, 55,
and 80. Mandatory all-channel reporting and the active/on-object diagnostic
denominator remain unchanged.

The amendment is motivated by Task 20, but the separately named v2 predicate is
emitted for every future v2 evaluator result and is used uniformly by the Task
20/55/80 replay runtime. Task 20 must return `true`; Tasks 55 and 80 must return
exactly `false`, not `null`. A `false` result does not establish that a
representation is healthy.

No programme order, checkpoint identity, historical replay comparison, pair
category, numeric threshold, heatmap-death definition, geometry, estimator,
split, or training semantic changes. All parent rules outside this narrow
conflict retain precedence.

## Decision

Decide whether the saved Task 20 checkpoint is correctly rejected as the
registered collapsed negative control before Tasks 55 or 80 may be opened.

## Current claim

A channel whose heatmap is categorically `heatmap_flat_dead=true` supplies no
reliable localized evidence under the frozen diagnostic and therefore must not
be allowed to certify that the remaining channels are structurally separated.

## Known evidence

- The immutable v1 replay result is
  `docs/decisions/2026-07-26/representation_oracle_replay/results/TASK20_CHECKPOINT_REPLAY_RESULT_v1.json`.
- Its file SHA-256 is
  `e44ec8b839d6b39377a8acf8b5b2997334ad3c518530675cadcc322e696d2675`.
- Its canonical content SHA-256 is
  `5087f5538f728647774fc9e88b4b55ac384c162ebcd3f44136e555bb258258fb`.
- Its recorded runtime `source_commit` is
  `426d1dbe94a655e6a90b4441b8e368be7338a4ae`; the result artifact itself was
  committed at `db002cdb452a6216b3862d275927653d137745fe`.
- All 245 frozen definition-identical historical comparisons passed.
- The v1 all-channel rule observed 36 persistent-duplicate pairs and nine
  separate pairs. Every separate pair involved channel 8.
- Channel 8 was categorically `heatmap_flat_dead=true`. The other nine
  channels formed one persistent-duplicate clique.
- The existing v1 rule consequently returned
  `structural_negative_control_collapse=false`.
- The frozen v1 specification quoted a signature sourced from an unversioned
  historical reanalysis: duplicate rate `1.0` and median distance approximately
  `1.27e-7`. The production v1 result instead reports all-channel rate `0.8`
  and median `1.8024939031909698e-7`; its eligible-only diagnostic reports rate
  `1.0` and median `7.102422897126104e-8`. This is not a failure of the
  245-record replay: those newly introduced representation-health fields were
  explicitly outside the definition-identical historical comparison.

## Postmortem and post-outcome disclosure

The wrong v1 assumption was that every bad channel in the registered collapsed
negative control would participate in the same duplicate cluster. The
implementation therefore allowed any separated channel to refute structural
collapse, even when that channel's heatmap contained no localized evidence.

The planted suite did not catch this because it tested an all-coincident case
and an all-flat-dead case separately. It did not contain the mixed mechanism
observed here: a large persistent-duplicate clique plus a distant flat-dead
channel.

This amendment is post-outcome with respect to the v1 replay and is not an
untouched out-of-sample validation of Task 20. Task 20 is now a repaired
development fixture. This lock was first drafted before implementation changes;
an uncommitted worktree candidate now exists. No v2 checkpoint replay, v2
execution authorization, Task 55/80 replay, training, or GPU work has occurred.
The rule and its near-miss cases are to be frozen by committing this amendment
before that candidate can be accepted or authorized.

## Critical unknown

Can a categorical v2 rule recognize the duplicate-cluster-plus-dead-channel
failure without incorrectly calling a representation collapsed when it
contains a genuinely localized, separated channel?

## Constraints

- The v1 artifact remains valid and immutable as evidence that checkpoint
  reconstruction succeeded, all 245 historical fields matched, and the
  original v1 predicate evaluated `false`. The v1 predicate is inadequate for
  the intended mixed-pathology negative-control meaning.
- No existing artifact may be overwritten or deleted.
- No new numeric threshold may be selected from Task 20.
- The existing categorical `heatmap_flat_dead` decision and the existing
  minimum-channel floor are reused unchanged.
- Reusing `minimum_eligible_channels=2` as the v2 structural-subset floor is a
  new semantic role for that inherited value. The value is not fitted to Task
  20.
- Motion activity and on-object eligibility do not determine membership in the
  v2 structural subset.
- A non-flat-dead but inactive, off-object, or otherwise ineligible channel
  remains in the v2 structural subset unless it is explicitly
  `heatmap_flat_dead=true`.
- This flag diagnoses one specific collapse mode. `false` does not mean that a
  representation is healthy; activity, attachment, switching, sliding and
  other representation evidence remain separate.
- Tasks 55 and 80 remain blocked until a separately committed v2 Task 20 result
  passes both the v2 classification and all 245 historical comparisons.
- The user's instruction to proceed authorizes this bounded repair programme;
  it is not advance approval of unseen implementation details or permission to
  bypass the review and execution gates.

## Exact v2 definition

For each channel `i`, use the evaluator's already-computed categorical field:

`flat_dead(i) := channel_health.channels[i].heatmap_flat_dead is true`

Define:

`S := {i : flat_dead(i) is not true}`

The use of `is not true` is deliberate. When heatmap evidence is unavailable
and the field is `null`, the channel is retained; absence of evidence may not
be treated as proof that a channel is dead.

Let `m` be the existing `minimum_eligible_channels` configuration value,
currently `2`. Using it for this structural subset is a new semantic role, but
the inherited value is unchanged and was not selected from Task 20.

- If `|S| < m`, the v2 decision is `null` with status
  `void_below_minimum_channels_not_confirmed_flat_dead`.
- Otherwise, run the unchanged frozen pair-category calculation on exactly
  `S`.
- The v2 decision is `true` if and only if every possible pair in `S` is
  evaluable and every such pair has category `persistent_duplicate`.
- Otherwise, the v2 decision is `false`.

The result must retain the v1 all-channel decision as a separately named
legacy diagnostic. New outputs must not silently relabel the v1 field.

## Required planted cases

| Case | Construction | Required v2 result | Purpose |
|---|---|---:|---|
| Existing all-coincident | All localized channels coincide | `true` | Preserves the original positive collapse case |
| Separated channels | At least two localized channels are separated | `false` | Prevents collapse from becoming a generic badness flag |
| Separated plus confirmed flat-dead | Separated localized channels plus one distant confirmed flat-dead channel | `false` | Proves dead-channel filtering does not erase real separation |
| Duplicate cluster plus dead | Coincident localized channels plus one distant flat-dead channel | `true` | Covers the Task 20 failure mechanism |
| Duplicate cluster plus distinct retained channel | Coincident cluster plus one distant channel not confirmed flat-dead, even if that channel is inactive or off-object | `false` | Prevents an eligible-only shortcut |
| Below minimum | Fewer than `m` channels not confirmed flat-dead | `null` | Prevents all-dead or nearly all-dead checkpoints from passing as non-collapse |
| Missing heatmap evidence | Coordinate-only channels with `heatmap_flat_dead=null` | Same pairwise decision as all-channel v1 | Prevents missing evidence from silently excluding channels |

## Versioning and authorization

- The legacy decision is reported under an explicitly v1/all-channel name.
- The new tri-state decision is reported under an explicitly
  `v2_excluding_confirmed_flat_dead` name. Channels with
  `heatmap_flat_dead=null` are retained and are therefore not described as
  proven non-dead.
- The Task 20 v2 replay writes only to
  `TASK20_CHECKPOINT_REPLAY_RESULT_v2.json` using exclusive creation.
- Authorization for Tasks 55/80 must verify the exact committed v1 failure and
  a separately committed v2 pass.
- Every v2 replay uses this predicate. Task 20 requires `true`; Tasks 55/80
  require non-void `false`. The predicate is a negative-control stop test, not
  the later coincidence trigger, a model-selection score, or a general
  representation-health verdict.
- The v2 result must record its source commit, source hashes, checkpoint hash,
  unchanged inference boundary, and the 245-record historical comparison.

## Next gate

Implement the additive v1/v2 diagnostics and all planted cases, then run only
the focused CPU test suites. Stop on any semantic or regression failure.
Before any checkpoint is opened again under the v2 replay chain, obtain a
bounded read-only Fable review of the exact specification, implementation and
tests, then create fresh authorization bound to the reviewed commit.
