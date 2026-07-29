# Fable 5 high-effort checkpoint-replay runtime review

Date: 2026-07-28

Reviewed candidate commit:
`bac87c4faeacbb71683c7323ecda29f841c2ce35`

Invocation boundary:

- model: `fable`
- effort: `high`
- permission mode: read-only plan mode
- safe mode: enabled
- session persistence: disabled
- real checkpoint access: forbidden
- project-file mutation: forbidden

## Independent report

The audit is complete.

### Independent Review — Saved-Checkpoint Replay Gate, commit `bac87c4`

I audited the exact committed candidate read-only: the runtime,
authorization, replay-registry, comparator, provenance, evaluator, model, and
manifest-builder sources (~15k lines), the five replay JSON artifacts, the
geometry bindings, and the focused test suites. No file was modified, no
checkpoint was opened, no training or GPU job was run. Three parallel
read-only audit agents covered the registry, the evaluator/provenance path,
and the model/manifest bindings; I read the runtime, authorization,
comparator, and key tests directly.

#### Findings

**P0 — none.**

**P1 — none.**

**P2 (observations; none blocks the replay decision):**

1. **Capability tokens are in-process sentinels, not cryptographic**
   (`representation_checkpoint_authorization.py:77-79`). Any in-process code
   can forge them via `object.__setattr__` (the tests do exactly this). This is
   the strongest guarantee Python permits; the bundle itself can never
   authorize, and the committed entry point uses the tokens correctly.
   Accepted limitation.
2. **Weak substring check on the Fable review file**
   (`representation_checkpoint_authorization.py:1050`):
   `"PASS_TO_RUN" in review_text` would also match a review saying
   "not PASS_TO_RUN". The real gate is the content-hash-bound
   execution-authorization JSON asserting `verdict: "PASS_TO_RUN"` with 0/0
   counts (`:1013-1027`). Suggest an exact verdict-line pattern when that
   artifact is authored.
3. **TOCTOU window between import-time self-digests and later git-byte
   re-verification** (`representation_checkpoint_runtime.py:96-99` versus
   `representation_checkpoint_authorization.py:481-513`); a racing writer or
   hand-crafted `__pycache__` entry could theoretically decouple loaded code
   from verified bytes. Outside the specification's drift-focused threat
   model.
4. **Third-party packages pinned by version only**, not bytes
   (`representation_checkpoint_runtime.py:296-318`).
5. **`batch_size` is caller-selectable** (default 8; validated only as a
   positive integer at
   `representation_checkpoint_authorization.py:1094-1098`). It is recorded in
   the result and CPU numerics are deterministic per run, but if the
   `2e-6`/`5e-4` tolerance is ever missed, it is a variable to rule out;
   consider freezing it.
6. **Untracked worktree file**
   `representation_oracle_calibration/NUMERIC_CALIBRATION.json`; the replay
   path reads only the committed `NUMERIC_CALIBRATION_v1_1.json`. Hygiene only.
7. **`visibility` is fabricated all-ones by the runtime**
   (`representation_checkpoint_runtime.py:1038-1040`) and trusted by the
   evaluator; attachment is independently recomputed from hash-bound masks,
   so this matches the fully-visible historical protocol and cannot fake
   on-object evidence.
8. **No end-to-end test of `run_authorized_checkpoint_replay`** (impossible
   pre-authorization by design); ordering, one-shot, and forgery properties
   are unit-tested with mocks. Residual integration risk fails closed.
9. **Verification note:** sandbox approval limits prevented my own
   `shasum`/`git hash-object` recomputation of the 11 source-manifest hashes.
   Any mismatch is caught fail-closed at run time by three independent chains
   (import digest to rebuilt manifest to committed manifest bytes to git blobs
   at both source and reviewed commits), so this does not affect the verdict.

#### Is the specification's meaning actually satisfied?

Yes — not merely schema-consistent, but semantically:

- **Source authorization (Q1):** `source_commit` must equal live HEAD; every
  consumed committed artifact is byte-compared to
  `git show <source_commit>:path`; the 11 runtime sources are additionally
  byte-compared to the reviewed candidate commit, which must be an ancestor.
  Registry content hash, corpus-inventory, and pair-index hashes are
  hard-coded in reviewed source, so post-review data drift cannot pass. The
  repo-local import closure is exactly the 11 manifest roles.
- **Checkpoint open (Q2):** one `O_NOFOLLOW` descriptor — fstat, size,
  streamed SHA-256, `seek(0)`,
  `torch.load(map_location="cpu", weights_only=True)` on that same
  descriptor; any failure aborts with no fallback; provenance later consumes
  a one-shot receipt instead of reopening
  (`representation_checkpoint_runtime.py:798-869`,
  `representation_checkpoint_authorization.py:2025-2082`).
- **Architecture (Q3):** 512 RGB, 10 keypoints, base 32, 64x64 logits with
  `head_upsample is None`, shared affine operator, no action head, inverse
  operator exactly for Tasks 20/80 (derived from lambdas and cross-checked
  against hard-coded bindings), strict state dict with per-tensor
  shape/dtype/finiteness checks, eval mode, CPU, inference_mode, deterministic
  algorithms, and identical parameter+buffer digest before/after inference.
- **Dataset binding (Q4):** 180 frames+masks each hash/size-verified through
  no-follow descriptors against manifests rebuilt from hashes frozen in
  reviewed source; theta = 2*frame over 0..358 degrees;
  `pairs_skip3_cyclic.json` binds 180 cyclic `(source+3)%180` pairs at
  +6 degrees; geometry binding pins centre `(0,0)` and `-theta` unrotation. No
  write path into the dataset exists on the runtime call graph.
- **Evaluator authorization (Q5):** one-shot contextvar receipt, minted only
  after a registered same-descriptor load and unchanged-state inference,
  cleared before validation (a modified bundle burns it); caller bundle
  fields alone can never authorize; the loaded checkpoint crosses into
  provenance via the consumed receipt without a reopen.
- **Logits (Q6):** raw `180x10x64x64` float32 logits are mandatory for
  checkpoint cases, strictly decoded, soft-argmax recomputed on
  endpoint-aligned `linspace(-1,1)` grids, compared to runtime points within
  the calibrated tolerance, and the evaluator's own recomputation is what is
  used thereafter. Representation quality (channel health, trajectory
  separation, data-computed collapse evidence) is reported separately from
  operator recovery and rollout.
- **Historical comparator (Q7):** all four registered historical files per
  fixture byte-verified; 245 = 5 operator records + 60 horizons x 4 fields;
  absolute difference <= `2e-6` OR relative difference <= `5e-4` for
  numerics, exact integers for sample counts; both the 245-record count and
  the pointer-contract count are independently enforced, against a registry
  re-validated at comparison time against its source-frozen content hash.
- **Task ordering and immutability (Q8):** Tasks 55/80 cannot mint a runtime
  capability without a committed, canonical, content-hash-verified,
  all-true-gate Task 20 result at the frozen path whose source commit is an
  ancestor and whose loaded-source digests match the current manifest;
  results are written only to frozen per-task paths with `O_CREAT|O_EXCL`.
- **Q9:** no selection leak (Task 20 `forbidden`, 55/80
  `replay_only_not_selection`, `selection_use_authorized: false` throughout,
  no ranking code path); no unsafe mutation or provenance gap found beyond
  the P2 notes.

The absent execution-authorization JSON and Fable review file are correctly
future artifacts: the code paths that will validate them are exact
(content-hash, git-byte, ancestor, and verdict checks) and fail closed while
they are absent.

#### Verdict

**PASS_TO_RUN** — unresolved P0 count: **0**; unresolved P1 count: **0**.

---

## Independent delta review — 2026-07-29

Reviewed candidate commit:
`6411547918691ec1abde09abfd4731db13bf48bf`

Reviewed predecessor:
`bac87c4faeacbb71683c7323ecda29f841c2ce35`

Invocation boundary:

- model: `fable`
- effort: `high`
- permission mode: read-only plan mode
- safe mode: enabled
- session persistence: disabled
- registered checkpoint access: forbidden
- project-file mutation: forbidden

### Scope

The review covered the complete four-file delta:

- `docs/decisions/2026-07-26/representation_oracle_replay/CHECKPOINT_REPLAY_EXECUTION_SPEC_v1.md`
- `docs/decisions/2026-07-26/representation_oracle_replay/CHECKPOINT_RUNTIME_SOURCE_MANIFEST_v1.json`
- `keypoint_net/representation_checkpoint_runtime.py`
- `tests/test_representation_checkpoint_runtime.py`

It also inspected the historical checkpoint writer at Git commit
`952064e28f7661907382a203920628310c8ccbbe`, the registered external
configuration bindings, and the surrounding runtime and authorization path.
No registered checkpoint was opened or deserialized.

### Historical justification

- The legacy checkpoint writer's embedded `ckpt_config` contained neither
  `center_crop` nor `heatmap_res`.
- The same writer stored the full run-level `config.json` from `vars(args)`,
  which necessarily contains `center_crop`, with `null` meaning no crop.
- Each registered run-level config is separately path-, size-, and
  SHA-256-bound and is validated before the checkpoint is opened.
- The legacy model had no `heatmap_res` argument. At 512-pixel input its
  fixed `/8` head is intrinsically 64 by 64.
- An embedded non-null `center_crop` remains fatal.
- An embedded `heatmap_res` other than 64 remains fatal.
- The strict state-dict key/shape checks and the explicit absence of a
  128-resolution upsampling head provide independent architecture checks.
- Because checkpoint bytes are already size- and SHA-256-bound, the relaxed
  metadata equivalence admits only the registered historical checkpoints and
  cannot authorize different bytes.

### Manifest and provenance

The runtime-source manifest delta changes only:

1. the manifest content hash; and
2. the `runtime_source` hash for
   `keypoint_net/representation_checkpoint_runtime.py`.

The other ten runtime-source records and all frozen preprocessing,
environment, and execution-boundary fields are unchanged. The runtime still
performs no crop. The prior execution authorization remains correctly
fail-closed because it binds the predecessor commit and predecessor manifest.

### P0 findings

None.

### P1 findings

None.

### P2 observations

1. `_validate_external_config` uses `config.get("center_crop")`, so a missing
   external key would be treated like explicit null. This is historically
   impossible for these hash-bound `vars(args)` configs, but an explicit
   key-presence check would match the specification wording more literally.
2. Embedded `heatmap_res` equality is not separately type-strict; `64.0`
   compares equal to `64`. This has no architectural effect because the
   reconstructed model and strict state dictionary are bound to integer 64.
3. The focused tests reject embedded `center_crop=256`, but do not separately
   test embedded `heatmap_res != 64` or explicit embedded
   `center_crop: null`.
4. Exact-set validation of embedded configuration keys is not performed.
   This predates the candidate and is harmless for the hash-bound checkpoint
   bytes.
5. The reviewer could not independently recompute the new manifest hashes
   within its sandbox. A mismatch can only block at runtime because working
   bytes are compared with both the rebuilt manifest and Git blobs.
6. A refreshed execution authorization is required before replay.
7. The unrelated untracked
   `representation_oracle_calibration/NUMERIC_CALIBRATION.json` remains
   outside the replay input closure.

None of these P2 observations blocks the Task 20 replay decision.

### Delta verdict

**PASS_TO_RUN** — unresolved P0 count: **0**; unresolved P1 count: **0**.

Scope: this authorizes at most the Task 20 CPU replay stop gate. It authorizes
no training, GPU work, model selection, Tasks 55/80 before a committed passing
Task 20 result, dataset mutation, or overwrite of an existing result.
