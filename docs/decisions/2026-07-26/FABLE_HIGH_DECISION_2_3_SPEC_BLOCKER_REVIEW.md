# Fable 5 high blocker review of Decision 2.3 spec v1

Review mode: Fable 5, high effort, `--print`, `--tools ""`, safe mode, no
session persistence, and no API-key environment.

## Verdict

**BLOCK**, on three blockers. The core arm design, shared-head math, and
initialization reasoning were judged sound.

## Blockers returned

1. **Undefined post-extension pass rule.** The draft defined 3/3 pass, 2/3
   provisional with two added seeds, and 0–1/3 fail, but did not define the
   five-seed outcome. This left a post-test forking path.
2. **Contradictory extension finalizer.** The draft made the initial finalizer
   refuse any existing test artifact while also requiring a later test
   evaluation for seeds 45/46. It did not name per-seed artifact isolation or
   restate the freeze-before-test rule for the added seeds.
3. **Gradient-audit contamination risk.** The draft called for validation-loss
   gradient audits during training without requiring that parameter gradients,
   optimizer state, RNG state, and the next training step remain unchanged.
   A careless `backward()` could silently train on validation data.

## Verified as sound

- one shared `Linear(4096,2)` has exactly 8,194 parameters;
- sharing it across channels preserves channel permutation and prevents
  cross-channel mixing;
- Arm B with coordinate-grid weights and zero bias equals the fixed
  spatial expectation when flatten order matches;
- Arm A's x/y-grid-divided-by-64 initialization has a reasonable output scale
  and nonzero upstream gradient after spatial centering;
- spatial centering gives Arm A the same additive-logit gauge invariance that
  softmax already has;
- Arm C is always in range because it is a convex combination of grid points;
- the A/B/C factorization is a valid causal ladder.

## Optional improvements returned

- verify Arm B/C equivalence in float64 and use a looser float32 parity
  tolerance;
- pin Arm A's grid orientation and flatten order explicitly;
- state that coordinate errors use unclipped outputs;
- quarantine smoke artifacts;
- record decoder weight norms because matched weight decay acts on different
  initial magnitudes in Arms A and B.

The specification was amended to close the three blockers and incorporate
these low-cost safeguards before requesting re-review.
