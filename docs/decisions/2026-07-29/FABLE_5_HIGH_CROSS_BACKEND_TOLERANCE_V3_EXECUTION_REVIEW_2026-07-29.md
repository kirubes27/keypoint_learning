# Fable execution review — cross-backend coordinate-tolerance candidate `6ddb2b8`

Reviewed read-only at the frozen candidate `6ddb2b87d978dea8514475e0fa97a6ca62225442`, which is the current HEAD with a clean tree (the only deviation is the out-of-scope untracked `NUMERIC_CALIBRATION.json`, which the evaluator can never consult because the registry path, file hash, and canonical content hash are all pinned to `NUMERIC_CALIBRATION_v1_1.json`). No files were edited, no checkpoint inference or training was run, and no external checkpoint/run artifacts were opened.

## 1. Verdict

**PASS_TO_RUN.** The candidate does exactly what the decision record authorizes — a narrowly scoped, checkpoint-only widening of one implementation-consistency check — and every fail-closed constraint I audited is intact. No P0 or P1 findings.

## 2. Findings

**P0 — none.**

**P1 — none.**

**P2 (nonblocking):**

1. **Role-scoping is unit-tested at the policy function, not end-to-end for checkpoint bundles.** `test_checkpoint_cross_backend_tolerance_is_narrow_and_role_scoped` proves `_coordinate_consistency_policy` returns 1e-4 with the checkpoint key and leaves planted cases on the registry value, and brackets the observed 1.3053417205810547e-05 below and 1e-3 above the tolerance. The wiring inside `validate_bundle` is eight lines feeding a single shared enforcement site (`eval_representation.py:3456` and `:3553`), and the planted rejection path is already exercised by existing suites, but a direct failure-injection test (a checkpoint-kind bundle whose supplied points differ by more than 1e-4 being rejected) would close the last gap. The Task 20 v3 shadow replay exercises the live checkpoint path with a frozen known answer, which substantially mitigates this.
2. **1e-4 is a round policy number, not a derived one.** The registry derives its tolerances from `max(32·eps·S, 4·E_ref)`; the cross-backend constant sits outside that formula and above the registry's float32 provisional ceiling of 1e-5. Keeping it out of the hash-frozen v1.1 registry file under a visibly different key is the correct call for now; a future registry revision could formalize a cross-backend entry with its own derivation. Cosmetic.
3. **Minor cosmetic indentation irregularities** in the expected-paths dict of `tests/test_checkpoint_replay_manifests.py` (no semantic effect).
4. **Inert placeholders:** `_validate_committed_task20_v3_result` builds a provisional authorization with `"0"*64` for the execution-authorization and review-file hash fields. I confirmed `_validate_checkpoint_load_record` never reads those fields, so this is harmless (and those files are independently validated as committed and unchanged since the reviewed commit), but a comment would prevent future confusion.

**Behaviors verified as claimed (audit checklist):**

- The registry tolerance 3.814697265625e-06 is unchanged for planted and all non-checkpoint cases: the registry file is untouched by the candidate (last modified in `bc0b660`) and is triple-hash-pinned; the policy function returns registry values for every `case_kind != "checkpoint"`; transform and bbox consistency checks still use the registry tolerance even inside checkpoint cases.
- Only saved-checkpoint PyTorch-versus-NumPy coordinate consistency uses 1e-4, and only for float32 logits/softmax — any other dtype fails closed rather than falling back. A planted case cannot masquerade as a checkpoint: `case_kind == "checkpoint"` hard-requires the full authorization chain (committed manifests, review binding, receipt, provenance load receipt), and no bundle field can enable that path on its own.
- The 1e-4 constant is labeled implementation-consistency at every layer: a distinct tolerance key (`float32::checkpoint_pytorch_numpy_spatial_expectation_coordinate`), a docstring saying "not a quality threshold", a dedicated execution-boundary key in the v3 manifest, and a required `scientific_quality_threshold_changed: False` assertion in the execution-authorization boundary. No scientific threshold references the constant (its only uses are the definition and the policy function).
- Supplied checkpoint points are checked and then **replaced** by the evaluator's own NumPy recomputation (`evaluation_points = derived`, `fit_points = derived_fit_points`) before every scientific metric, and checkpoint cases must supply logits for both evaluation and fit sections.
- Temperature (pinned 1.0), dtype, endpoint grid (`linspace(-1,1)`, `endpoint_grid is True`), xy axis order, source hashes (git-blob plus import-time digests), checkpoint identity (pinned sha256/size, same-descriptor hash-and-load, weights-only, cpu, no unsafe fallback), and no-training/no-selection flags all remain fail-closed; `_git` runs with `check=True` and `--no-replace-objects`; replay results are exclusive-create with an explicit refuse-to-overwrite.
- Task 20 must shadow-replay under v3 before Tasks 55/80 (details in §4), and v1/v2 evidence is preserved: the candidate touches no `results/` files, v3 result filenames are distinct from v1/v2, and the immutable v1 false-negative audit is validated byte-exact with its failed-gate content pinned.

## 3. Is 1e-4 proportionate?

Yes. Three ratios anchor it:

- **Above the noise it must tolerate:** 7.66× the observed Task 55 maximum (1.3053417205810547e-05), leaving modest headroom for the not-yet-measured Task 80 checkpoint without renegotiating the lock. The observed maximum itself is float32 reduction-order noise, corroborated by the exact-zero fresh PyTorch recomputation.
- **Far below any convention error it must catch:** one 64-grid cell step is 2/63 ≈ 3.175e-2 (≈317× the tolerance); a half-cell / align-corners convention error is ≈1.6e-2 (≈159×); an axis swap or sign flip produces errors of order 0.1–1; a temperature mismatch on non-degenerate heatmaps is far above 1e-4. Every plausible coordinate-convention defect lands two or more orders of magnitude above the tolerance, so a material error cannot hide under it. The lock document's arithmetic (0.00315 pixels, ~7.7×, ~317×) checks out.
- **Structurally incapable of changing science:** the widened check is a stop/go consistency assertion only — the supplied coordinates are discarded and replaced by the NumPy derivation before all metrics. Widening 3.8e-6 → 1e-4 cannot alter a single scientific number; it can only permit a run to continue that identical logits would have produced anyway.

This is the proportionate small repair the decision record asked for, not a mechanism.

## 4. Is the Task 20 v3 shadow replay a sufficient next gate?

Yes. Task 20 (itself a checkpoint-kind case) is exempt from the gate so it can run first, and its capability must carry no gate fields. Tasks 55/80 then hard-require both the immutable v1 audit and a committed v3 Task 20 result whose scientific verdict is pinned field-by-field to the frozen v2 verdict — flat-dead channel [8], retained channels [0–7, 9], 36/36 persistent-duplicate pairs with the full category-count breakdown, collapse-v2 true / legacy-v1 false, 245/245 historical record comparisons, gate passed — which I confirmed matches the actually committed v2 result byte-for-byte on those fields. The v3 result must additionally be canonical JSON with a verified content hash, carry loaded-source digests matching the v3 manifest and the git blobs at its source commit, and that commit must be an ancestor of the running commit. So the changed evaluator path is exercised live on a checkpoint with a frozen known answer, and any scientific drift stops Tasks 55/80 before a checkpoint is loaded. Combined with the unchanged-tolerance planted suites and the role-scoping unit test, this is a sufficient and well-ordered next gate.

## 5. Assumptions and limits of independent verification

- **I could not recompute SHA256 values myself** — the read-only sandbox blocked `shasum`, `openssl`, `python3`, and `node`. I cross-checked the supplied marker values against committed artifacts instead: the preflight hash appears identically in both prior execution authorizations and both Task 20 results; the v2 lock and amendment hashes appear in `CHECKPOINT_EXECUTION_AUTHORIZATION_v2.json`; the oracle-manifest hash appears throughout the committed v2 planted results; the immutable v1 hash is the pinned constant in the authorization source, harness, and manifest test; the manifest content hash is embedded in the v3 manifest and schema-validated at authorization. Four values (the v3 manifest file hash, the new semantic-lock hash, the new harness hash, the reviewed-fileset hash) have no committed cross-reference yet — but all four are recomputed from committed bytes by `authorize_checkpoint_runtime` and compared against this report's marker lines at run time, so an incorrect value fails closed and cannot enable anything.
- I took the stated test evidence as given (109 relevant tests plus 21 manifest/provenance tests passed in the frozen environment; the v3 manifest independently rebuilt byte-identical) and did not re-run tests.
- Per the review constraints I opened no checkpoint or run artifacts; checkpoint identity was verified only as the pinned hash chain, and the Task 55 discrepancy statistics were taken from the decision record, not re-derived.

---

FABLE_REVIEW_BINDING_SCHEMA: cross_backend_tolerance_v3_fable_execution_review_binding.v1
REVIEWED_CANDIDATE_COMMIT: 6ddb2b87d978dea8514475e0fa97a6ca62225442
RUNTIME_SOURCE_MANIFEST_FILE_SHA256: 15afd4a72446aacdcdde350aa7dfaeeb236d581cffcaa7e542a7a6593d47c452
RUNTIME_SOURCE_MANIFEST_CONTENT_SHA256: fb1177b2aac19996168b81572e7658b1a8cc26866e2829621ad00b14d7e7e6b6
CHECKPOINT_HASH_PREFLIGHT_FILE_SHA256: d333d6295c8ba6bab5af90e59a54291de37cb8f82379db4d7a8f231b5bffc99a
TASK20_V2_SEMANTIC_LOCK_FILE_SHA256: add948e69655991e4aa5bef55ea60a8826cf50af1068b9bae4a539d403bc22c8
DECISION_SYNTHESIS_V2_8_AMENDMENT_FILE_SHA256: 6ab5c6ea585920ee6579a91b3b82650c734dee93507ac29b43ae92bd3b4b858d
CROSS_BACKEND_TOLERANCE_SEMANTIC_LOCK_FILE_SHA256: ca985ac94e13f300c921f49df0ed47d96e230b245189bef0749f290a731cb8e7
TASK20_V2_ORACLE_HARNESS_FILE_SHA256: 6065ae24a8897fa25cfca009b45be9c098d698a3d500926f7160962bd7f02e74
TASK20_V2_ORACLE_MANIFEST_FILE_SHA256: 6ef8d58dc637effe02fc5a4416fd714e1e03cc70e761aa9915cb02dc0dc30ac0
PLANTED_V2_REVIEWED_FILESET_SHA256: b8eaee99b663c8dc3d968b33323b6afa920a4739bf7ac9d76578c04358d99817
IMMUTABLE_TASK20_V1_FILE_SHA256: e44ec8b839d6b39377a8acf8b5b2997334ad3c518530675cadcc322e696d2675
MODEL: fable
EFFORT: high
PERMISSION_MODE: read_only
SESSION_PERSISTENCE: false
VERDICT: PASS_TO_RUN
UNRESOLVED_P0_COUNT: 0
UNRESOLVED_P1_COUNT: 0
