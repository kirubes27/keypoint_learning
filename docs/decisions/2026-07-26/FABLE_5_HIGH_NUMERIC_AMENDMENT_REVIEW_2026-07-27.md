# Fable 5 high: independent numeric-amendment review

Date: 2026-07-27

Mode: Fable model alias, high effort, read-only plan permission, no session
persistence, Claude Max subscription authentication.

Independence: Fable received the original task restatement and raw v1
specification, fixture manifest, independent calibrator, candidate numeric
registry, and CPU contract test. It was explicitly prohibited from reading the
Codex amendment draft.

## Verdict: APPROVE WITH REQUIRED CHANGES

The numeric calibration in `NUMERIC_CALIBRATION_v1_1.json` is arithmetically
exact and methodologically sound as registry-1 content, and the two early
synthetic `evaluate_bundle` calls do **not** contaminate it or require a restart
— but the amendment commit must add the six items below.

### What I verified from the raw files

**Formula and S.** The calibrator implements the frozen formula
`T = max(32*eps*max(1,S), 4*E_ref)`. `S` is derived only from preregistered
fixture parameters, never from computed outputs, so there are no tunable
degrees of freedom: value metrics get their exact expected magnitude (6.0
degrees for `proper_rotation_angle_deg`, 1.07 for `mean_scale`, and
`abs(59 * -0.08) = 4.72` for `composition_bias_element`), while error metrics
with exact expected value zero correctly stay at `S=1`. The contract test
locks this distinction. Remaining `S=1` defaults are valid because every other
expected magnitude lies in `[-1,1]`.

**Arithmetic.** Every tolerance re-derives exactly: float64 `32*eps` is
`7.105427357601002e-15`; multiplication by 1.07, 6, and 4.72 matches the
registry; `composition_bias_error_l2` is the only entry where the `4*E_ref`
branch governs (`1.1916163935586911e-14`); and float32 `32*eps` is
`3.814697265625e-06` for both float32 entries. Every `E_ref` equals the maximum
of its fixture rows. All values are below the v1 ceilings. Exact categorical
predicates carry no tolerance, and no replay or scientific threshold leaks
into the registry.

**Early evaluator calls.** They breached the preregistered execution lock and
the split-before-oracle order, but cannot have contaminated the numbers:
fixtures were preregistered first; tolerances are a deterministic,
parameter-free function of the manifest and environment; the calibrator never
touches the evaluator; and wrong-sign/reflection outcomes are categorical or
many orders of magnitude from a numeric boundary. This is a
documentation-and-quarantine matter, not a calibration defect.

### Required changes before or at commit

1. Record the two `evaluate_bundle` calls, quarantine their outcomes as
   non-evidence, and rerun both falsifiers fresh in the official post-commit
   oracle gate.
2. State that Task 20/55/80 replay tolerances are not covered, live in a
   separate registry, and keep all checkpoint replay blocked until that
   registry is frozen in a further amendment.
3. Declare `NUMERIC_CALIBRATION_v1_1.json` solely authoritative, mark the older
   untracked `NUMERIC_CALIBRATION.json` superseded without deletion, and record
   both hashes.
4. Freeze the exact production lookup-key mapping, including
   `float32_affine_coordinate` to `float32::affine_coordinate` and
   `float64_estimator` to `float64::spatial_expectation_coordinate`.
5. Verify the fixture and reference hashes against their named commits and
   recompute the candidate internal content hash before commit.
6. Restate the widening lock, no-scientific-reuse rule, exact-equality fields,
   environment-specific scope, and verify-without-silent-regeneration policy.

### Not blockers for this numeric commit

Evaluator/loader implementation, split generation, Task 55/80 replay fixture
binding, yaw/pitch floors, the coincidence-trigger threshold, and scientific
Registry 2 are later ordered work and must not be pulled into this commit.

### Next gate

After the amendment commit, complete the split gate: deterministic generation,
byte-identical regeneration, independent verification of all frozen counts,
and loader anti-aliasing proof in both modes. Only then run the oracle suite
under these frozen tolerances. Any accepted falsifier or Task 20 non-collapse
is a hard stop. Checkpoint replay additionally waits on the separate replay
tolerance freeze; 64-versus-128 and new-transformation training come later.

### Access note

The original brief attachment was outside Fable's allowed directory and could
not be opened. The review used the task restatement plus the five in-repository
raw sources. Fable launched no search agents and did not inspect the amendment
draft.
