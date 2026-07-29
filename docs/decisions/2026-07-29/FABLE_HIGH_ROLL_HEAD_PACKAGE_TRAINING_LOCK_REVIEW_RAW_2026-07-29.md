# Audit report — roll head-package training semantic lock v1 (draft)

## Verdict: PASS_WITH_NONBLOCKING_FINDINGS

No condition must change before implementation. Every numeric and structural claim in the candidate at commit `2eb1cc14` verified exactly against the committed code and artifacts. I confirmed HEAD is the exact candidate commit, the spec is committed in it, and the worktree is clean except the excluded untracked `NUMERIC_CALIBRATION.json` (never read; it overlaps no bound path). One limitation: the retained independent report at `/private/tmp/...` is outside this session's allowed read roots, so I resolved everything directly against code, artifacts, and parent decisions — which the task mandates anyway. The full report with all citations is saved at `~/.claude/plans/review-task-independently-audit-modular-dove.md`.

## Key verifications

- **Head packages**: 64 path = /8 encoder + 1×1 head (`model.py:154-157`); 128 path = bilinear upsample + 3×3 conv(128→64) + BN + ReLU + 1×1 head (`model.py:142-153`). Recomputed head-only parameters from the layer definitions: 1,290 vs 74,570, delta **exactly +73,280**. Output shapes 64×64 / 128×128 at 512 input. Encoder module identical between packages.
- **Entropy**: raw Shannon entropy over H·W softmax cells, not divided by log(HW) (`model.py:396-421`) — the spec's scale-mismatch statement and grid-only-causal-claim prohibition are correct and necessary.
- **Training semantics**: development mode = validation-selected `best_model.pt` by minimum total validation loss, no test loader (`train.py:651-661, 1200-1208`); validation at epoch 1 and multiples of 10; all frozen hyperparameters match existing defaults/flags, including hardcoded `eta_min=1e-6` (`train.py:1038`); both recipes fully expressible via existing lambda flags.
- **Splits**: frame partition train 27–176 / holdout 0–23 / guard 6 frames, endpoints disjoint; `gate_pass=true`, `structurally_valid=true`, byte-identical regeneration in `SPLIT_VERIFIER_REPORT.json`; binding hash `acfa8358…` present in the pair artifacts. One subtlety I resolved directly: the train artifact holds **882 pairs across six objects**; the hammer subset is exactly 147 (validation is 21, hammer-only). train.py's mandatory `--object` + role lock + recorded `pair_count_after_object_filter` (`train.py:444-446, 543-550, 574`) make the spec's 147/21 counts correct post-filter.
- **Evaluator**: `eval_representation.py` emits every Section 5 axis (angle/improper detection, role-scoped k=1..7 AUC, drift, both separation variants, duplicate identities, channel health, flat-dead, mode switching, full-corpus k=1..59 + k=60 closure as diagnostic-only) and hard-fails via `EvaluationContractError`.
- **Authorization gap is real**: `representation_checkpoint_authorization.py` permits exactly the three immutable Task 20/55/80 fixtures via frozen hash bindings; no fresh-checkpoint path exists — Gate 2 is correctly a pre-training gate.
- **Section 2 claims** match the committed v3 replay JSONs (Task 55: flat-dead [8], 2 persistent duplicates; Task 80: flat-dead [8], 1; Task 20 negative-control v2 true; 245/245 records each).

## Coherence

Causal claim, 12-cell matrix, checkpoint policy, multi-axis decision rule, extension rule, evaluation boundary, provenance, CPU smoke, and CUDA/full-run stop gates are all **internally coherent**. On reconciliation item 3: retaining validation-selected checkpoints is a **caveat, not a blocker** — the parent split spec (lines 738–742) freezes it for the development role, no parent decision forbids it, both packages face the identical selection rule paired by (recipe, seed), and confirmation/final objects later use fixed epochs with untouched tests.

## Findings

- **P0**: none.
- **P1-1**: Section 4's fail-closed run binding is almost entirely **new code** — train.py records provenance (commit, index/dataset-binding hashes, embedded config) but enforces nothing: no dirty-worktree rejection, manifest-hash verification, unique-cell check, or untracked-overlap rejection; the legacy train-as-validation path still exists (indexed mode avoids it). Gate 3's CPU semantic tests are the necessary proof and must not be abbreviated.
- **P2-1**: decision axes are computed on the same 24-frame block whose loss selected the checkpoint — absolute values mildly selection-optimistic; comparative use sound. Suggest one caveat sentence.
- **P2-2**: only AUC/drift (items 2–3) trigger the seeds-45/46 extension; items 4–5 at exactly 2/3 are final. Appears intentional — should be stated explicitly.
- **P2-3**: Section 2's "Next gate" describes the CPU smoke, but Section 8 inserts Gate 2 before it; declare Section 8's ordering governing. Also "every 10 epochs thereafter" literally means multiples of 10 plus epoch 1.
- **P2-4**: note in Section 3 that the pair artifacts are six-object files and 147/21 are post-object-filter counts; `--img_size` defaults to 256 and must be pinned to 512 (already covered by run-binding item 5 and Gate 4's shape assertion).
- **P2-5**: define "checkpoint can be reconstructed" as architecture+weights reload, not bitwise retraining — seeds are set but torch/cudnn determinism is not enforced, and Section 4 only requires recording determinism settings.

## Smallest next gate

**Gate 2: fresh-checkpoint evaluator authority extension.** Pass requires all of: (1) the three fixture bindings remain byte-identical and all fixture authorizations still pass; (2) mutation tests — tampering any bound field (checkpoint/config/history sha256, size, fixture id, role, path) causes fail-closed rejection; (3) the fresh path refuses any candidate lacking a complete hash-bound run manifest (commit, cell ID, pair-file hashes, dataset binding `acfa8358…`, checkpoint sha256, embedded-config match to Section 3), each missing/mismatched field an independent hard fail; (4) no authorization possible outside the fixture set plus manifest-bound fresh cells, no dynamic registration; (5) its own source manifest plus independent review. Any accepted mutation, fixture-behavior change, or missing-field acceptance fails the gate and blocks Gates 3–7.
