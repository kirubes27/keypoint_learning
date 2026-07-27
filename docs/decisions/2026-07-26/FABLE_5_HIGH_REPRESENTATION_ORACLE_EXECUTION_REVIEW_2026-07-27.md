# Fable 5 High Review: Representation Oracle Execution Gate

Date: 2026-07-27
Review mode: Claude App subscription, model alias `fable`, effort `high`,
read-only plan permissions, print mode, no session persistence
Reviewed candidate: `0b161ccde6ae674272ece96b23bb35777bf91614`
Official planted run status when the review was requested: not run

## Independent verdict

`PASS_TO_RUN`

Fable reported no P0/P1 defect after independently inspecting the committed
evaluator, provenance boundary, split/geometry and replay gates, harness,
manifest, and planted case definitions.

## Substantive response (lightly formatted for the decision record)

> I audited commit `0b161ccde6ae674272ece96b23bb35777bf91614`
> independently: I read the full evaluator, hand-checked the rotation math,
> and used three read-only audits over the provenance/replay boundary, the
> harness/manifests/16 case fixtures, and the split adapter/governing specs.
> HEAD matches the candidate exactly and the tree is clean apart from the stale
> untracked calibration file the briefing flagged. No P0/P1 defect found.
>
> **Physical roll/sign/centre semantics — adequate.** The spec freezes
> `R_img(delta) = [[cos, -sin], [sin, cos]]` on the y-down image grid, with
> `b = c - A c`. The code implements this convention. The off-centre case's
> expected bias `[-0.016509874690203957, -0.024972824338953842]` reproduces
> from `c - R(6 degrees)c` with `c = [0.23, -0.17]`. Reflections cannot be
> reported as a proper angle, the wrong sign is tested against the locked
> positive-six-degree generator, and planted physical states must reproduce
> the target transform within the frozen tolerance. Whether the renderer's
> world-Z convention maps to this image convention remains correctly deferred
> to rendered geometry evidence; dataset-backed roll is blocked.
>
> **Collapse/health/void/assignment/switching — adequate.** Dedicated cases
> cover separated/attached points, coincident collapse, static inactivity,
> controlled sliding, off-object attachment, dead/flat heatmaps,
> peak-channel switching, and coordinate-only void evidence. Exact assignment
> ties remain ambiguous. Switching durations follow the directed pair graph,
> decompose paths/cycles, break on void or ambiguity, join a cycle boundary
> once, reject branch/merge graphs, and validate the three 60-edge cycles of
> the plus-3/180-frame graph. The legacy role subset is explicitly not promoted
> to held-out evidence without proven non-exposure.
>
> **Provenance boundary — adequate.** Every committed role requires equality
> between the claimed hash, the working bytes, and the Git blob at the claimed
> commit. The repository root is derived from the production source rather than
> supplied by a caller. Numeric files, manifest, case definitions, evaluator,
> provenance validator, and harness are path- and hash-bound. Injected
> provenance and caller-selected repositories cannot bypass the public path.
>
> **Dataset/checkpoint stop boundaries — adequate.** Checkpoint cases are
> rejected before external files are referenced. The geometry registry records
> `saved_checkpoint_replay_authorized: false`, and the replay preflight checks
> that exact commit-bound registry before the first checkpoint `lstat`, open,
> or hash. Dataset-backed geometry remains blocked without a reviewed registry
> entry. The planted harness contains no model, CUDA, or dataset execution path.
>
> **P2 caveats (non-blocking).**
>
> 1. Record the review as the ratifying step because the execution contract
>    self-describes as a draft.
> 2. The stale untracked `NUMERIC_CALIBRATION.json` is a human-confusion hazard,
>    although nothing in the candidate references it.
> 3. Assignment tie detection uses exact floating-point equality; near-ties are
>    unique with a small margin. This is acceptable for planted fixtures but
>    should not be over-read on real data.
> 4. Direct calibration-file hash recomputation was unavailable in Fable's
>    sandbox; runtime hash checks fail closed.
> 5. Provenance includes machine-specific absolute paths.
> 6. Dead-heatmap thresholds should be registry-bound before dataset use.
>
> **Next gate.** Run the planted suite once from the exact reviewed commit into
> a fresh output directory, construct every case twice, and require canonical
> byte equality. Stop on any case mismatch, hash/provenance failure, or
> nondeterminism. Do not widen tolerances or proceed to training, datasets, or
> checkpoint work while saved replay remains blocked.

## Reconciliation and disposition

- The review occurred before the official command, while HEAD was the exact
  reviewed candidate.
- The authoritative calibration and fixture-manifest file hashes were
  independently recomputed after the review and matched the manifest bindings.
- The stale untracked calibration file remains excluded and untouched. It was
  not deleted or moved.
- The near-tie, absolute-path, and future heatmap-threshold caveats do not alter
  planted implementation-correctness semantics. Dataset-backed use remains
  blocked.
- Fable's next gate and stop criterion were followed without amendment.
