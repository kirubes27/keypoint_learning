# Fable High review: roll head-package training lock

Date: 2026-07-29

Reviewed candidate commit:
`2eb1cc14f73850ce254fe5670c088a9e6b8e91b2`

Reviewed file:
`docs/decisions/2026-07-29/ROLL_HEAD_PACKAGE_TRAINING_SEMANTIC_LOCK_v1_DRAFT.md`

## Verdict

`PASS_WITH_NONBLOCKING_FINDINGS`

No condition had to change before implementation. P0: none. The sole P1 was
that the fail-closed fresh-run binding and authorization path is mostly new
code and its CPU semantic/mutation tests must not be abbreviated.

## Independently verified

- The 64 path uses the `/8` encoder output plus a `1x1` heatmap head.
- The 128 path adds bilinear feature upsampling, a learned `3x3` convolution,
  batch normalization and ReLU before its `1x1` head.
- At 10 keypoints and 32 base channels the head parameter counts are 1,290 and
  74,570, a difference of exactly 73,280.
- The entropy loss is raw Shannon entropy over `H*W` cells and is not divided
  by `log(HW)`.
- The frozen development path selects `best_model.pt` by minimum total
  validation loss, evaluates at epoch 1 and multiples of 10, and has no test
  loader.
- The six-object train artifact reduces to 147 hammer pairs after the mandatory
  object filter; the validation artifact has 21 hammer pairs. Endpoints are
  disjoint and the split verifier records structural, live-corpus and
  byte-regeneration passes.
- The production evaluator contains every required representation/dynamics
  axis in the lock.
- The existing secure authorization path accepts only the three immutable
  Task 20/55/80 fixtures. No fresh-checkpoint path exists yet.
- The Task 20/55/80 v3 claims quoted in the lock match their committed results.

## Nonblocking findings reconciled

1. The validation axes are mildly optimistic because the same development
   block selects the checkpoint. This remains permitted matched development
   selection, not confirmation evidence; later objects use fixed epochs and
   untouched tests.
2. Only AUC/drift 2-of-3 outcomes trigger the seeds-45/46 extension. The lock
   now says this explicitly.
3. Section 8's gate ordering governs; the next gate is the fresh-checkpoint
   authority extension before the CPU training smoke.
4. The lock now states that roll pair files contain all objects while the
   quoted counts are post-object-filter, and that legacy CLI defaults must be
   overridden.
5. Checkpoint reconstruction means exact architecture/config plus exact weight
   reload, not bitwise-reproducible retraining. Determinism settings must be
   enforced and recorded.

## Required next gate

Extend evaluator authority only to complete, hash-bound fresh-run manifests
while retaining the immutable fixture path. Mutation of any checkpoint,
config, history, cell identity, role, path, size, source commit, pair binding,
dataset binding or embedded config must fail closed. The candidate requires
its own source manifest, tests and independent review before training.
