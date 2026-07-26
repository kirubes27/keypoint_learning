# Fable 5 High final Decision 2.3 review — 2026-07-26

Invocation: Claude Code `2.1.191`, model alias `fable`, effort `high`,
subscription authentication, `--print`, safe mode, no session persistence,
and no tools. Fable received the frozen specification and raw audited
measurements, not Codex's conclusion. The response below is retained verbatim.

---

# Final Review — Decision 2.3 (Coordinate-Head Bypass Diagnostic)

**Decision outcome: Decision 2.3 closes NEGATIVE for a head redesign.** All three arms, including the unchanged fixed spatial-softmax expectation, pass the frozen thresholds on 3/3 seeds, and the frozen interpretation matrix explicitly routes the all-pass cell away from redesign. Detail below, with evidence tiers separated at the end.

## 1. Sufficiency and audit status

The supplied evidence is internally sufficient to close Decision 2.3. The chain is complete: frozen commit → D1 smoke (checkpoint restore, gradient paths, no test access) → D2 blind 3×3 training → D3 single test-content access → independent validator PASS over the aggregate, ledger, configs, checkpoints, metrics, hashes, and recomputed arm statuses. No audit blocker remains. The five hard-cap-unconverged runs are correctly labelled and are explicitly permitted by the frozen pass criterion (threshold capability within the matched budget), so they are a documented limitation, not a blocker. The one thing sufficiency does *not* extend to is any claim beyond the matrix cell that was reached — see §9.

## 2. Frozen seed rule per arm

Checking each seed row against the joint criteria (median-of-channel medians ≤0.50, pooled p90 ≤1.50, on-mask ≥0.95, worst of both conditions):

| Arm | Seed results | Arm status |
|---|---|---|
| A raw_linear | 0.209/0.169/0.162 median; 0.439/0.421/0.347 p90; on-mask 1.0 | **3/3 pass** |
| B probability_linear | 0.441/0.433/0.368 median; 1.124/1.226/1.005 p90; on-mask ≥0.9967 | **3/3 pass** |
| C fixed_expectation | 0.206/0.264/0.236 median; 0.578/0.655/0.496 p90; on-mask 1.0 | **3/3 pass** |

Seed B-42's 0.441 is the closest call and still clears 0.50. Seeds 45/46 are triggered only by an exactly-2/3 provisional outcome, which no arm produced. They are therefore **not required, and running them under the Decision 2.3 label would be off-protocol (forbidden within this frozen decision)**. They remain optional only as a separately preregistered new experiment, which nothing here motivates.

## 3. Strongest justified one-sentence conclusion

Under the frozen matched-budget protocol on a single object's 360-degree in-plane roll panel, freshly trained models pass all supervised coordinate thresholds on 3/3 seeds regardless of whether the coordinate head is a raw-logit linear decoder, a learned probability-space linear decoder, or the unchanged fixed spatial-softmax expectation — so the seed-41 Gate 0 failure does not reflect an architectural incapacity of the fixed expectation and does not justify a coordinate-head redesign.

## 4. Calibration of "matched seed variability dominates the seed-41 result"

**This claim should be weakened.** Two problems:

- The frozen matrix itself is disjunctive: all-pass routes to "seed variability **or** the fixed seed-41 trajectory." The panel retrained from scratch under the frozen protocol; it did not rerun the seed-41 checkpoint's original training conditions with new seeds. It therefore cannot separate "seed 41 was an unlucky draw" from "the seed-41 production trajectory (its budget, selection, or other unmatched details) produced the failure." The panel establishes the disjunction, not the first disjunct.
- "Dominates" is an effect-size and inference claim, and the briefing states the design is descriptive with n=3 correlated cyclic orbits, no error bars, no inferential test.

Properly calibrated: *"The Gate 0 failure is not architecturally forced; matched retraining with three fresh seeds passes with the identical fixed head, localizing the failure to the seed-41 checkpoint's specific realization (seed and/or training trajectory), which this panel does not further decompose."*

## 5. What the five hard-cap runs change and do not change

**Do not change:** every seed's pass status. The frozen criterion is capability within the matched budget, and all nine runs met it; a run that clears thresholds without plateauing still cleared them. The decision outcome is unaffected.

**Do change:** any ability to rank the architectures. All three C seeds and one seed each of A and B were still improving at epoch 3000, so the observed ordering (A ≈ C numerically better than B) is a snapshot of unequal convergence states, not asymptotic performance. This cuts in a specific direction worth noting: B is numerically worst *despite* being initialized at C's fixed expectation and stopping on plateau in 2/3 seeds, while C was still improving at the cap — plausibly the learned probability-space decoder drifted to a worse solution, but with no preregistered effect-size margin or inferential test, no ranking claim of any kind is licensed. Threshold capability (established) and asymptotic architecture ranking (not established, not preregistered) must be kept strictly apart in the report.

## 6. Implications for other objects

Implied: the diagnostic pipeline, provenance machinery, and pass criteria are executable and can be applied per-object. Not implied: anything about any object other than engineers_hammer_vray. One object, one panel; there is no transfer, no population inference, and each future object requires its own gated evaluation under the locked programme order.

## 7. Implications for yaw/pitch

Essentially none. The panel is pure in-plane world-Z roll, which preserves the object's visible surface up to image-plane rotation. Yaw/pitch introduce out-of-plane appearance change, aspect variation, and self-occlusion — qualitatively different challenges the fixed expectation was never tested against here. The locked programme correctly sequences yaw/pitch last, and this result advances toward it only by closing Decision 2.3 without a redesign detour.

## 8. Single next programme action

Per the locked order: **execute the post-Decision-2.3 re-plan, whose required artifact is a frozen COINCIDENCE_RUNFILE.md specifying Gate 3a's numeric criteria** (hash-bound before any Gate 3a data access, consistent with the provenance discipline used here). Branch criterion: Gate 3a cannot run until that freeze exists; once run, Gate 3a's own pass/fail routes to Gate 3b and onward to the transformation-geometry and estimator-oracle package. Nothing else in the programme is unblocked before that freeze.

## 9. Claims the final report must avoid

1. That a coordinate-head redesign is justified or that A or B is "better than" C (no preregistered margin; five runs unconverged).
2. That seed variability *specifically* — rather than the seed-41 trajectory — explains Gate 0 (disjunction unresolved).
3. That Gate 0 was erroneous or is retroactively overturned; it is closed and its result stands for that checkpoint.
4. Any identifiability, distinct/materially-attached-keypoint, or operator-benefit claim (supervised success explicitly does not license these).
5. Any transfer claim: other objects, translation/scale, yaw/pitch, or unseen data.
6. Any population or inferential statement (error bars, significance, "robust across seeds" in a statistical sense) — n=3 descriptive, correlated orbits.
7. Any statement that downstream gates (3a/3b, geometry package) are pre-cleared or that roll-control readiness is established.

---

**Evidence tiers.**
- *Verified facts (as supplied, cross-checked arithmetically against the frozen thresholds):* all nine seed rows clear all three criteria in both conditions; zero out-of-range coordinates; the 3/3–3/3–3/3 arm statuses; the D1→D2→D3 access ordering and validator PASS.
- *Conclusions licensed by the frozen rules:* Decision 2.3 closes with no redesign; seeds 45/46 not triggered; routing to the seed-variability-or-seed-41-trajectory cell; hard-cap runs count as passes.
- *Plausible interpretations (flag as such):* the learned probability-space decoder (B) may converge to worse solutions than the fixed expectation it was initialized to; C's numbers would likely improve past the cap.
- *Unresolved questions:* seed-draw versus seed-41-trajectory decomposition; asymptotic A/B/C ranking; everything downstream of the boundary in §9.
