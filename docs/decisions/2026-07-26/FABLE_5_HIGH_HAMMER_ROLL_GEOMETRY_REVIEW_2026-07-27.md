# Fable 5 High: hammer-roll geometry and hash-preflight review

- Date: 2026-07-27
- Candidate commit reviewed: `cbff50d`
- Model alias: `fable`
- Effort: `high`
- Mode: read-only safe mode, no session persistence, plan permissions
- Decision requested: whether the exact hammer-roll geometry evidence and
  fail-closed boundary are sufficient to permit opaque checkpoint-byte hashing
  only

## Raw independent review

> # Verdict: PASS_TO_HASH_PREFLIGHT
>
> Commit cbff50d provides adequate, non-circular evidence for the hammer
> world-Z roll geometry, and its code fails closed so that at most opaque
> checkpoint-byte hash preflight is authorized. I found **no P0 or P1 issues**.
> Full review record is in the plan file; the repository was not modified.
>
> ## The five requested assessments
>
> **1. Centre (255.5, 255.5) / normalized (0, 0) — derived, and independently
> corroborated.** The verified generator snapshot shows the camera at
> `(0, object_base_y, camera_dist)` looking at `(0, object_base_y, 0)`
> (`generate_tdw_world_z_roll_dataset_post_update.py:161-162`), and
> `_recenter_object` (`compare_hammer_rotation_operators.py:84-108`) teleports
> the object centre onto that look-at point. `meta.jsonl` records
> `center_error_world ≤ 7.8e-8` and `use_centroid=True, is_world=True`, so the
> world-Z rotation axis lies on the optical axis and projects to the image
> centre; at 512×512 with no crop under the endpoint-aligned `(W−1)/2` rule
> that is pixel (255.5, 255.5) = normalized (0, 0). This is then corroborated
> from rendered masks: `_local_center_audit` (`smoke_geometry.py:118-197`) tests
> a 3×3 pixel grid on 4 decisive frames and the exact centre is the unique best
> (mean canonical IoU 0.9895 vs 0.9521 for the nearest neighbour). Not assumed,
> and not circular — the six sample-mask hashes and the meta.jsonl hash in the
> evidence report all match the corpus inventory frozen at the earlier artifact
> commit 78071297, and `_sign_check_hammer.png` is never cited as evidence.
>
> **2. Sign is unambiguous.** `_warp_mask_same_eval_sign`
> (`smoke_geometry.py:81-115`) implements the forward map q = R(+θ)p with
> R = [[c,−s],[s,c]] in x-right/y-down coordinates (positive angle appears
> clockwise), via inverse sampling — I verified the matrix algebra. On
> decisive frames (60°, 120°, 240°, 300° — all four quadrants) unrotation by
> −θ recovers frame 0 at IoU ≥ 0.989 while +θ gives ≤ 0.035; frames at
> θ%180 = 0 are correctly flagged `ambiguous_for_sign` and excluded from the
> locked pass. The binding records the forward transform R_img(+6°)
> (skip3 × 2°/frame, matching `dataset_index.json` skip3
> `delta_theta_deg: 6.0`) and canonical unrotation R_img(−θ) as distinct
> quantities.
>
> **3. Cross-checks before any checkpoint touch — yes.**
> `preflight_checkpoint_candidates`
> (`representation_checkpoint_replay.py:1545-1589`) enforces, in order:
> capability-bearing provenance receipt (identity-checked sentinel, lines
> 1157-1182), registry byte revalidation, then
> `_require_saved_checkpoint_replay_geometry_authorization` (lines 1242-1542),
> which verifies the geometry registry's frozen file and content hashes, exact
> execution boundary, the binding's hashes, the semantic-lock and inventory
> hashes, evidence roles, and byte-equality against the source commit's Git
> blobs for the registry, binding, and all eight evidence files — all of which
> I confirmed are Git-tracked and clean at HEAD. Only after all of that does
> `_hash_checkpoint_candidate` touch a path, with symlink rejection and
> size/hash mismatch stops.
>
> **4. Everything else fails closed.** Scale, translation, yaw, and pitch are
> `"missing"` in the registry's `required_evidence` and have no registered
> binding, so the adapter's `_authorize_registered_dataset_geometry` (which
> runs before `_load_geometry`, `representation_split_adapter.py:974-980`)
> rejects them with "no unique frozen registry entry" — tested. The translation
> camera sign (−1/−1) is preserved and enforced. Checkpoint loading, replay,
> model selection, and training are `false` in both registries and re-asserted
> in the preflight's returned receipt
> (`checkpoint_load_authorized_by_this_receipt: False`, lines 1583-1588).
> Nothing in the commit imports a model.
>
> **5. Tests are sufficient for this narrow authorization.** Mutation tests
> freeze roles, hashes, paths, tolerances, and boundary flags; forged receipts
> are rejected; a mutated geometry binding fails before any checkpoint touch
> (proven with checkpoint hashing mocked to raise); the adapter has both allow
> and block tests plus translation sign falsifiers; and the prior
> planted-oracle suite includes wrong-sign and reflection falsifier cases.
>
> ## Non-blocking P2 notes and limitations
>
> For the *later* replay-runtime gate (explicitly not reasons to reject hash
> preflight): `align_corners` is null in the binding while the pixel rule is
> endpoint-aligned semantics — pin it before any runtime resampling; the centre
> audit has 1-px granularity (sub-half-pixel offsets excluded only by the
> metadata derivation); the evidence report doesn't embed its generating
> command; and the adapter has a minor read-after-hash TOCTOU window. One review
> limitation: the sandbox denied hashing tools without approval, so I verified
> the SHA-256 constants by cross-artifact consistency (binding ↔ registry ↔
> inventory ↔ code constants ↔ snapshot manifest, all mutually consistent)
> rather than recomputation — every gate recomputes at runtime and fails closed,
> so a wrong constant could only cause refusal, never false authorization. You
> can confirm with `shasum -a 256` on the geometry docs and by running the two
> test modules. The untracked `NUMERIC_CALIBRATION.json` in the working tree is
> outside this commit and read by no gate.
>
> Geometry completion is not checkpoint-replay completion: the replay execution
> runtime is intentionally absent and remains blocked behind its own separately
> committed, Fable-reviewed gate.

## Codex reconciliation

Fable independently supported the same narrow decision reached from the code
and artifacts: the exact hammer-roll centre/sign evidence is sufficient, and
only opaque checkpoint hashing may proceed. Its P2 runtime notes are retained
for the later checkpoint-loading implementation. They do not weaken this
hash-only gate because the receipt explicitly keeps model loading, replay,
training, and selection unauthorized.
