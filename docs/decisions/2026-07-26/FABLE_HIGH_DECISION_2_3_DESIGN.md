# Fable 5 high independent Decision 2.3 design

Review mode: Fable 5, high effort, `--print`, `--tools ""`, safe mode, no
session persistence, and no API-key environment. The reviewer received the
raw Decision 2.3 constraints and architecture facts, not Codex's draft.

## Decision 2.3 pre-implementation specification: direct-coordinate diagnostic head

**Bottom line:** I recommend a two-stage design — a cheap frozen-encoder linear probe on the seed-41 checkpoint, then the decision-bearing end-to-end run — using a **per-channel full-resolution linear readout of the raw 64×64 logits** as the direct head (arm A), paired with a **capacity-matched control that reads the softmax probabilities through an identical learned linear map** (arm B). Everything else (encoder, loss, data, optimizer, stopping, selection, thresholds, seeds 42/43/44) is held exactly at the existing control protocol. The A-vs-B contrast is what turns "a different head did better" into "bypassing the spatial-softmax gradient path did better." Three ambiguities require decision-document changelog entries before launch, most importantly that the current decision text has **no branch for diagnostic FAIL** and no operational definition of "downstream benefit."

### Constraints taken as fixed

Gate 0 is frozen and not rerun; local/windowed readout is demoted. Dataset, splits (train 0,3,…,177; val 1,4,…,178; test 2,5,…,179), augmentation (±5° rotation, ±8 px translation), optimizer (Adam 1e-4, wd 1e-5, batch 16), stopping (min 1000 / max 3000 epochs, eval every 25, patience 400, 1% relative), validation selection score, the four instrument thresholds, the ≥2/3-seed rule, and seed set {42,43,44} are all reused unchanged. Test frames stay untouched until three-seed finalization. No yaw/pitch/arc60 data, no temperature changes (explicitly a non-answer), no readout+loss confound: the loss stays coordinate MSE in all arms. All claims are descriptive: one object, one correlated orbit; seeds replicate optimization only.

### 1. Head choice: direct linear regression, not parametric Gaussian mean

Use direct coordinate regression. The parametric-Gaussian-mean head is rejected as the diagnostic because it is not a single intervention: its mean μ must come from somewhere — if μ is produced by a spatial expectation the softmax Jacobian is back in the path, and if μ is regressed directly the head *is* direct regression plus a rendering/shape term that adds a loss component, violating the no-readout+loss-confound constraint. The Gaussian head remains the natural candidate for the later *repair* phase (it restores the spatial prior), which keeps the diagnostic cleanly separated from the headline method.

"Sequential pair" in the sense of Gaussian-then-direct is unnecessary. Sequential in the sense of probe-then-end-to-end is adopted.

### 2. Head architecture, initialization, capacity

**Arm A (bypass head, decision-bearing).** Attach after the existing 1×1 conv, on the raw 10×64×64 logits. Per channel k: ŷ_k = W_k · vec(H_k) + b_k, with W_k ∈ ℝ^{2×4096}, no nonlinearity, no cross-channel mixing. Output is in the normalized coordinate convention. No range constraint: tanh/sigmoid would introduce a new saturation path into a Jacobian diagnosis. Out-of-range predictions are reported as a fraction, not clamped during training.

- Init: W = 0, b = 0 (image center); gradients are nonzero at step 0 (dL/dW = residual ⊗ H).
- Parameter count: 10 × (2·4096 + 2) = **81,940**, vs ≈ 246.7k encoder parameters (~33%); both numbers reported in every run artifact.

Why full resolution rather than pooled: the 0.50 cell64 median threshold demands sub-cell precision; a pooled readout quantizes sharply peaked bumps and could fail for reasons unrelated to the hypothesis. Why per-channel rather than flattened-all-channels FC (~820k params) or GAP: per-channel structure retains spatial layout (blocking a GAP pose shortcut) while denying the head cross-keypoint/global-pose information, and the existing ±8 px translation augmentation forces any generalizing linear readout toward translation-equivariant, centroid-like structure — a strong regularizer against arbitrary 60-frame memorization. A predeclared memorization flag covers the residual risk.

**Arm B (softmax-path control, attribution-bearing).** Identical shape and parameter count, but reads probabilities: ŷ_k = W_k · vec(softmax(H_k)) + b_k. Init W_k to the [-1,1] coordinate grid, b = 0, so B is *exactly* the baseline soft-argmax at initialization (unit-tested), and differs from A only in whether gradients to the logits pass through the softmax Jacobian p_i(δ_ij − p_j). This init asymmetry with A is a deliberate design choice, recorded as such.

**Escalation variant (only if arm A fails the tiny-overfit gate):** per-channel MLP 4096→64→2 with ReLU. Not run otherwise.

### 3. Frozen vs end-to-end

- **Stage P (probe, descriptive only, runs first, cheap).** Freeze the seed-41 representative checkpoint. Train arm-A heads only, same protocol, head-init seeds 42/43/44. Evaluate on train/validation only and never touch test. It asks whether the failed checkpoint already linearly encodes the coordinates. Probe pass sharpens attribution; probe fail is explicitly uninformative about end-to-end and gates nothing.
- **Stage E (end-to-end, decision-bearing).** Train encoder + head from scratch, arms A and B, seeds 42/43/44 each. Decision 2.3 is a claim about optimization through the head, so end-to-end is the primary experiment.

Prerequisite: the A/B comparison also needs the frozen soft-argmax baseline's three-seed instrument outcome under this identical protocol. If it does not exist, baseline arm C must run concurrently.

### 4. Gates, seeds, selection, test policy, branch rules

**G-unit** → **G-tiny:** 8 train-split frames spread around the orbit, no augmentation, ≤2000 epochs; pass = train median-of-channel median ≤ 0.25 cell64, per arm and probe head. Failure means the head is underpowered and is not a scientific result. → **G-smoke:** full 60-frame train, 200 epochs; loss finite and < 0.5× initial, validation evaluation and gradient logs written, ETA within budget. → **G-full:** the end-to-end and probe runs under the frozen protocol.

**Selection:** best checkpoint by the existing validation selection score. **Test policy:** one test evaluation per arm, only after all three seeds' checkpoints and the analysis code are hash-frozen; results recorded regardless of outcome.

**Pass/fail:**

- Diagnostic **PASS** = arm A meets all instrument thresholds in at least 2/3 seeds.
- **Memorization flag:** train median ≤ 0.10 cell64 with validation median > 1.0 cell64 means invalid, not FAIL; escalate to a pooled/lower-capacity variant before concluding anything.
- Diagnostic **FAIL** = anything else, provided G-tiny passed and gradient logs show the head trained.

**Branching:** PASS plus the recorded Gate 0 local-readout failure → redesign head, then a downstream-benefit gate before any identifiability claim. FAIL → the softmax Jacobian is not the bottleneck; combined with Gate 0, evidence points upstream, but the current decision text must add that branch. If A and B pass, the softmax gradient path is exonerated and the culprit was the fixed expectation readout itself; that branch is also currently uncovered.

### 5. Gradient health and attribution

Log every evaluation interval with identical code across arms: per-channel ‖∂L/∂H_k‖ over the batch; dead-gradient fraction; conv1 weight-gradient norm and its ratio to the last conv's; for arm B, additionally the analytic soft-argmax Jacobian scale proxy and per-channel max probability.

Strong attribution would require:

1. Arm A passes and capacity-matched arm B fails.
2. At matched epochs, B's dead-gradient fraction is at least five times A's, concentrated on high-error channels.
3. Stage P recovers coordinates on validation from the frozen failed checkpoint.

Any weaker combination downgrades to "a learned linear readout improves the instrument (descriptive)" without mechanism attribution.

### 6. Downstream benefit and oracle guard

The diagnostic head's own supervised fit can never count as downstream benefit. Before a "stop decoder work" branch can fire, the decision document must predeclare: the headline unsupervised/operator metric computed with no coordinate labels in the loop; matched data/budget/seed policy old-head versus new-head; a margin; and a degeneracy guard pairing the operator metric with on-mask fraction so collapse cannot count as benefit. The exact metric is absent, so the identifiability branch is unexecutable until it is defined.

### 7. Minimum artifacts and pre-launch tests

Tests: coordinate-convention round-trip; arm B at initialization reproduces baseline soft-argmax to ≤1e-6; evaluation reproduces checkpoint metrics; on-mask parity; fixed augmentation reproducibility; nonzero conv1 gradients through each head at initialization; asserted parameter counts; deterministic short-run replay; and a test-loader guard that hard-fails before finalization.

Artifacts: this versioned spec; per-arm configs and hashes; split manifest; Git commit and config hash embedded in checkpoints; tiny and smoke reports; per-seed training curves and gradient logs; parameter-count report; validation selection records; one frozen test report per arm; and a decision-branch record.

### Changelog ambiguities requiring decisions before launch

- No branch exists for diagnostic FAIL; "downstream benefit" is undefined.
- Confirm whether a matched three-seed soft-argmax baseline exists; otherwise add arm C.
- Arm B extends the one-head decision document and must be recorded.
- Define the interpretation when both A and B pass.
- Record that Stage P is validation-only and outside the optimization-seed policy.
- Record out-of-range handling.
- Confirm the normalized coordinate range `[-1,1]`.
