# Fable 5 high audit of completed Gate 0

Review mode: Fable 5, high effort, `--print`, `--tools ""`, safe mode, no
session persistence, and no API-key environment. The reviewer received only
the completed `GATE0_REPLAY_RESULTS.md` artifact plus the audit question.

**Audit verdict: the Gate 0 evidence supports the frozen demotion, the arithmetic is internally consistent, and no caveat I can identify from the document could plausibly reverse the routing to Decision 2.3.** The observed 6/150 is a factor of 12.5 below the 75/150 rank-retention threshold, and the worst-case unfrozen-convention sensitivity (23/150) is still a factor of 3.3 below it. Details below, followed by the diagnostic-head question I recommend freezing next.

## Arithmetic and denominators

All checks pass:

- 60 frames × 10 channels = 600 pairs; strictly-above-75th-percentile on 600 values yields at most 150, and exactly 150 absent ties, which the document asserts and the partition check (600 unique records, two strata partitioning all 150) corroborates.
- 6/150 = 4.0%, 144/150 = 96.0%, 6/600 = 1.0%, 144/600 = 24.0% — all correct.
- The frame ID list is a clean arithmetic progression (1, 4, …, 178; 60 values), consistent with a phase-mod split over the 180-frame orbit.
- The high-error mask is defined once from the global readout and held fixed across variants, so no variant benefits from denominator recomputation. Note this conditioning is not a bias against windowing — it is the hypothesis itself: pairs where the dominant mode is correct but the global expectation is dragged off would land exactly in the correct-dominant high-error stratum, and only 6 did.
- Internal table consistency holds: the 150-pair global median (1.9310) sits slightly below the 144-pair stratum median (1.9710), as expected when 6 low values (~1.35) are mixed in; all global high-error stratum medians exceed the 1.277964 threshold; windowed medians in the high-error strata are unconstrained by the threshold since the mask comes from the global readout.

## Robustness of the branch

The decision has enormous margin in every direction I can stress it:

- **Geometry convention.** The "within 1 cell" ambiguity is the only unfrozen definitional gap, and the document bounds it: the four interpretations give 1, 6, 13, and 23 out of 150. Reversal would require a convention yielding ≥75; no defensible reading of "1 cell" gets there. The floor-cell variant (13/150) also rules out a half-cell grid-alignment error silently deflating the count.
- **Numeric replay drift.** The CPU-vs-CUDA discrepancy (median 0.0034 cell against a 0.01-cell QA tolerance) could at most swap a handful of pairs across the 1.278 threshold. Flipping the branch would require 69 additional correct-dominant pairs; drift of this magnitude cannot produce that.
- **Window radius and temperature.** These affect the descriptive error tables but not the counting rule — mode correctness is radius-independent — so the frozen choices of r ∈ {2, 4, 8} and temperature 1.0 cannot be the hinge of the decision.

## Leakage and replay fidelity

- The "test untouched" claim is consistent with everything shown: all paths are the frozen validation split, and the split hash matched the hash frozen in the checkpoint. The validation orbit is interleaved every third frame with training data, but this is disclosed, and for a within-model readout ranking it actually strengthens the conclusion — the wrong-mode failure occurs even on near-training-distribution frames, so it is not a distribution-shift artifact that windowing might repair elsewhere.
- Transport fidelity is well supported: frame-0 targets regenerated exactly from the local mask, on-mask grounding 1.0 over all 180 frames (a wrong roll sign or center would break this for an asymmetric object like a hammer), and the global readout's 0.65-cell median error against transported targets independently confirms the target geometry.
- Replay integrity checks are thorough: bitwise-unchanged parameters, identical checkpoint hash pre/post, exact agreement with the extractor forward pass, sliced-vs-vectorized window agreement at float precision, and a hash-identical second execution.
- Statistical scope is stated correctly and matches the rule's nature: a rank demotion on fixed descriptive evidence from one seed and one object, with no population inference smuggled in.

## Non-blocking gaps

Two things are absent but do not affect the branch: the nearest pair-to-threshold margin is not reported (immaterial given the 69-pair gap), and mode correctness on the 450 low-error pairs is not reported (irrelevant to the frozen denominator, though it would contextualize the mechanism). More importantly, "wrong-or-diffuse" deliberately conflates two mechanistically different failures — a confidently wrong unimodal peak versus diffuse or multimodal mass — because no diffuse cutoff was frozen. That conflation is exactly what the diagnostic head must resolve.

## Narrowest diagnostic-head question to freeze next

Gate 0 established that the *dominant* mode is usually not target-local, but says nothing about whether the correct mode exists in the heatmap at all. That is the single fork that determines re-planning: a readout/selection problem is repairable by a mode-selection head; a representation problem is not repairable by any readout. I would freeze:

> **Among the fixed 144 wrong-or-diffuse-mode high-error pairs (same checkpoint d4777e3a…, same 150-pair mask, no training, no weight updates), what fraction of frozen heatmaps contain a target-local secondary mode — defined as any local maximum of the 64×64 heatmap (strict maximum over its 3×3 neighborhood, edge-clipped) whose cell index is within 1 cell Euclidean of the target cell, using the identical target-cell and distance conventions as Gate 0?**

Freeze alongside it: the denominator (144, with the 6 correct-dominant pairs reported separately, not pooled), the local-maximum definition, and a branch rule symmetric to Gate 0's — for example, if fewer than 50% of the 144 pairs have a target-local secondary mode, the failure is classified as representational (no target-local mass to select) and readout-side repairs of any kind lose rank; otherwise a mode-selection/re-ranking head keeps rank. A useful secondary descriptive (not a gate): the rank and relative mass of that target-local mode versus the dominant mode, which quantifies how hard selection would be. This is the narrowest question because it is binary per pair, reuses every frozen convention from Gate 0, requires no new thresholds beyond the local-maximum definition, and its two outcomes map one-to-one onto the two re-planning branches.
