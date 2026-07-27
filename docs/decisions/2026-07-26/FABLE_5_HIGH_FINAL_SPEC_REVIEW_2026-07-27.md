# Fable 5 High final specification review — 2026-07-27

Review mode: read-only `Read`, `Glob`, and `Grep`; high effort; `--print`;
safe mode; plan permission; no session persistence; Claude Max subscription;
no API-key billing; no edit or shell tools.

## Verdict

**APPROVE** freezing
`REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.md` for deterministic
split-generator and loader implementation. This review authorizes no training
or GPU work.

## Independently recomputed arithmetic

| Family | Train frames | Guard frames | Holdout frames | Train pairs | Holdout pairs | Match |
|---|---:|---:|---:|---:|---:|:--:|
| roll, cyclic skip 3 | 150 | 6 | 24 | 147 | 21 | yes |
| yaw, non-cyclic skip 6 | 94 | 12 | 15 | 82 | 9 | yes |
| pitch, non-cyclic skip 6 | 94 | 12 | 15 | 82 | 9 | yes |
| scale, non-cyclic skip 4 | 98 | 8 | 15 | 90 | 11 | yes |
| translation x/y, stride 3 | 77 | 22 | 22 | 56 | 16 | yes |

Fable additionally verified:

- all 180 cyclic roll edges account as 147 train, 21 holdout, and 12 dropped;
- all roll wrap edges are excluded;
- minimum train/holdout index distances meet the primary strides;
- forward/reverse files can share the same endpoint partition;
- the 15-frame scale holdout contains zero skip-20 pairs;
- the yaw/pitch block is exactly the interior `-7..+7` degree band;
- orthogonal translation bands are honest and preserve usable stride-3 pairs.

## Agreements

- The hybrid is faithfully encoded: hammer uses train+validation;
  confirmation/final objects use train+test, no validation, fixed epochs, and
  the final checkpoint.
- Fresh object-specific training is not weight transfer or zero-shot object
  generalization.
- The current code defects named by the specification are present:
  train-as-validation aliasing, roll-specific pair metadata, partition-unaware
  whole-directory auto-evaluation, and warning-only evaluation failures.
- Object roles and the confirmation burn rule agree across the amendment,
  specification, and contamination ledger.
- The preparation-only execution boundary is consistent and does not authorize
  training, rendering, GPU smoke, matrices, or cluster work.

## Required corrections and disposition

Fable identified three non-blocking corrections:

1. The historical full-orbit `k=1..59` rollout AUC mixes train and holdout
   frames under a blocked split. The specification now separates it from
   role-scoped held-out evidence and defines holdout `k=1..7` for the frozen
   24-frame roll block.
2. Frozen documents must be committed and hash-bound. This is satisfied by the
   specification commit that includes this review.
3. "Future datasets" was ambiguous because the four corpora are already
   rendered. The specification now clarifies that they are future only in the
   experimental order and generalizes the zero-holdout rule to any secondary
   stride spanning the holdout.

## Deferred but not blocking

- numeric oracle tolerances and scientific margins;
- descriptor and 64-vs-128 implementation details;
- yaw/pitch scientific acceptability thresholds;
- coincidence-trigger numeric threshold;
- the exact epoch-count derivation rule, which must be frozen before the first
  confirmation run;
- source-generator provenance imports and dated dataset errata, which remain
  inside the split inventory gate.

## Next gate

Implement the deterministic generator, independent verifier, and two indexed
loader modes.

Pass only if every frozen count matches; endpoints are disjoint; guards,
directions, wrap behavior, and scale-holdout exclusion are exact;
regeneration is byte-identical; paths resolve; and the loader rejects overlap,
wrong role arguments, fixed-final validation/best-checkpoint behavior, and
partition-unaware auto-evaluation before creating a run directory.
