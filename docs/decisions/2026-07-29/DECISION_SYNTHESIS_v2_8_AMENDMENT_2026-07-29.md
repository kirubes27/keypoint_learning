# Decision Synthesis v2.8 Amendment

Date: 2026-07-29
Status: additive change-control record; no old decision artifact is rewritten

## Scope

This amendment is motivated by the failed Task 20 structural-collapse
negative-control predicate. The separately named v2 predicate applies
uniformly to every future Task 20/55/80 v2 replay result. It does not change the
research programme order:

1. geometry, estimator and evaluator semantics;
2. leakage-safe splits;
3. deterministic and negative-control oracles;
4. Task 55/80 head-package comparison;
5. descriptor consistency;
6. frozen-recipe transfer to new full-roll objects;
7. conditional coincidence gates;
8. later transformation families.

## Trigger

The committed Task 20 v1 replay faithfully reproduced all 245 frozen
historical comparison values. Its detailed evidence showed nine channels in
one persistent-duplicate clique and one categorically flat-dead channel.

The v1 top-level rule returned `false` because it required every
all-channel pair to be a persistent duplicate. The nine pairs involving the
dead channel were separate. The v1 result and verdict remain valid under the
v1 definition and are retained as an immutable postmortem artifact. More
precisely, the artifact proves that checkpoint reconstruction succeeded, all
245 historical fields matched, and the inadequate original predicate evaluated
`false`.

The frozen v1 specification quoted a signature sourced from an unversioned
historical reanalysis: Task 20 duplicate rate `1.0` and median distance
approximately `1.27e-7`. The production v1 result reports all-channel rate
`0.8` and median `1.8024939031909698e-7`, while its eligible-only diagnostic
reports rate `1.0` and median `7.102422897126104e-8`. This is not a 245-record
replay mismatch because these new representation-health fields were explicitly
non-definition-identical.

## Amendment

- Task 20 remains the collapsed negative-control fixture and remains forbidden
  as a candidate.
- The v1 all-channel classification remains available under its original
  meaning.
- A separately named tri-state v2 classification excludes only channels whose
  existing `heatmap_flat_dead` field is exactly `true`.
- Channels whose heatmap state is `null`, and non-flat-dead channels that are
  inactive, off-object or otherwise ineligible, remain in the v2 structural
  subset.
- Below the unchanged minimum-channel floor, v2 is `null`, not `false`.
- The inherited value `minimum_eligible_channels=2` gains this new
  structural-subset role but is not changed or fitted to Task 20.
- Otherwise v2 is `true` only when every possible retained-channel pair is
  evaluable and has the unchanged `persistent_duplicate` category.
- Task 20 must return `true`; Tasks 55/80 must return exactly `false`, not
  `null`. A `false` classification is not a general health verdict.

The executable semantic definition for this amendment is
`docs/decisions/2026-07-29/TASK20_STRUCTURAL_COLLAPSE_V2_SEMANTIC_LOCK.md`.

No numeric threshold, pair category, flat-dead definition, geometry, split,
dataset, checkpoint identity or historical replay comparison is changed.
Mandatory all-channel evidence and the active/on-object diagnostic denominator
remain unchanged. The amendment supersedes only the old all-channel
structural-collapse predicate as the replay classification gate for Tasks 20,
55, and 80.

## Post-outcome boundary

This is a post-outcome repair of a development negative-control fixture. It is
not presented as untouched validation. The exact rule and planted near-misses
will become frozen when this amendment is committed, before a v2 Task 20
replay. A worktree implementation candidate now exists, but no v2 replay,
v2 execution authorization, Task 55/80 replay, training, or GPU work has
occurred. The candidate must be committed and independently reviewed before it
can be authorized.

The user's instruction to proceed authorizes the bounded repair and later
Phase 1-5 programme. It is not advance approval of unseen implementation
details and does not waive any review, checkpoint-execution, or training gate.

## Authorization boundary

- Preserve the v1 runtime manifest, review, authorization and result.
- Create a separate v2 runtime manifest, Fable review, authorization and
  `TASK20_CHECKPOINT_REPLAY_RESULT_v2.json`.
- Tasks 55 and 80 remain blocked until the committed v2 Task 20 result reports
  the expected v2 classification and all 245 historical comparisons pass.
- No checkpoint may be opened again under the v2 replay chain until fresh
  authorization binds the exact reviewed v2 source commit and artifacts.
