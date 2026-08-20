# Consolidated keypoint-instrument diagnostic report

**Date:** 2026-07-05

**Purpose:** expert review of the CNN–heatmap–soft-argmax instrument before the
operator/keypoint hypothesis is tested

**Branch:** `fitted-operator-diagnostics-20260704`

## 1. Executive summary

The scientific question is whether pressure to make transformations simple and
predictable helps select image-grounded, stable and distinct keypoints. That
question has **not yet been cleanly tested**. Before comparing scientific
losses, we tested whether the underlying keypoint extractor can reliably move
points in response to an unambiguous coordinate-learning signal.

The extractor is a convolutional network that emits one spatial heatmap per
keypoint. Spatial softmax converts each heatmap to a probability distribution,
and soft-argmax returns its expected `(x,y)` coordinate. The operator and other
scientific losses see only these coordinates.

The diagnostic programme established two different facts:

1. **Four-frame regime:** coordinate-only supervision often creates wrong,
   nearly one-hot heatmaps whose useful gradients collapse. Dense Gaussian
   heatmap supervision learns the same targets, proving that the architecture
   can represent them. Two attempted ground-truth-free heatmap-shape repairs
   did not solve coordinate training.
2. **Representative 60-frame regime:** data diversity prevents the catastrophic
   saturation/gradient-collapse state, but several channels still plateau with
   large localization errors. Therefore saturation is a real small-data
   failure mechanism, but it is **not the primary explanation of the current
   representative-regime error**.

The current result is not “nothing works and we do not know anything.” We have
ruled out many concrete explanations and rejected two specific repairs. The
remaining causal ambiguity is narrower: the representative failure is most
consistent with some combination of underconstrained expectation supervision,
weak/local visual observability at certain targets, and interference through a
shared convolutional representation. Their relative contributions have not yet
been separated.

This matters scientifically because a poor result under the operator losses is
currently ambiguous: it could mean the losses prefer poor coordinates, or that
the extractor cannot reliably realize useful coordinates implied by those
losses. Until this instrument confound is removed, a negative result cannot be
attributed to the operator hypothesis.

## 2. What was actually tested

### Main project objective

The intended Stage-B comparison is approximately:

- minimal image grounding alone;
- grounding plus fitted-operator/minimal-subset consistency;
- privileged oracle controls.

The desired inference is that differences in keypoint quality are caused by
the scientific losses.

### Instrument diagnostic

The diagnostic deliberately used privileged target coordinates generated from
known object masks and the known rotation. This does **not** constitute the
paper method. It asks a simpler capability question:

> If the desired coordinate is supplied directly, can the unchanged extractor
> learn it accurately and consistently?

This is a deliberately favourable test. Direct coordinate MSE is clearer than
the real operator loss, which generally constrains relationships among points
rather than specifying a unique pixel for each channel. Failure of this control
therefore creates a confound for the harder unsupervised experiment.

### Why standard heatmap literature does not automatically answer this

Supervised pose/keypoint systems commonly train every heatmap pixel against a
Gaussian target. This provides dense spatial credit: the target region receives
a strong gradient even when the current heatmap is wrong. Our intended method
cannot use a true target heatmap. Coordinate and operator losses act through the
heatmap expectation, which is much less informative: many sharp, diffuse or
multimodal distributions share the same expectation.

Thus the forward operation is standard, but the supervision regime is not the
one under which heatmap models are usually easiest to train.

## 3. Chronology of experiments and what each contributed

### Phase 0 — fitted-operator algebra and localization-noise risk

The coordinate-only Block-0 tests established that minimal-subset fitted
operators can expose duplicates/collapse in idealized settings, but their
affine rung is sensitive to heterogeneous coordinate noise near the measured
image-model level. Empirical Task-80 jitter was approximately one heatmap cell.
The diagnostic-week review then found:

- mask transport geometry was valid for the hammer orbit;
- Hungarian channel reassignment did not explain the dominant error;
- hard argmax did not improve Task-80 equivariance;
- halving simulated residual amplitude changed similarity separation from 3/5
  to 5/5 optimization seeds;
- a 128-head smoke had better equivariance but was not a controlled
  architecture comparison and had worse grounding.

**Contribution:** established that upstream coordinate quality is a
load-bearing confound for the proposed fitted-operator experiment. It did not
identify the upstream cause.

Primary review: `keypoint_net/diagnostics/DIAGNOSIS.md`.

### Phase 1 — initial full-data supervised controls

Two coordinate-supervised and two dense-heatmap-supervised 64-resolution runs
used 90 training frames and 500 epochs. The coordinate runs produced:

| Seed | Median of channel medians | P90 | Median jitter | On mask |
|---:|---:|---:|---:|---:|
| 42 | 0.497 | 1.367 | 0.344 | 1.000 |
| 43 | 0.642 | 2.169 | 0.302 | 0.993 |

The dense heatmap runs had lower jitter (`0.265--0.271`) but worse held-out
coordinate localization (`0.875--0.971` median of channel medians). These runs
were capped at 500 epochs and did not use the later validation-plateau
protocol.

**Contribution:** showed that the instrument was neither catastrophically
broken nor reliably accurate. Dense supervision improved stability but did not
by itself solve full-orbit generalization. The mixed result motivated more
controlled attribution rather than immediate Stage B.

Artifacts:
`cluster_downloads/day45_supervised_53023161/`.

### Phase 2 — keypoint-count sweep on a four-frame overfit task

We trained K in `{5,10,15,20}`, three seeds each, for 5,000 optimizer updates
on frames 0, 3, 6 and 9: 12 runs total. Every strict all-channel gate failed,
but failure fraction was not monotonic with K. Targets 3 and 6 repeatedly
failed, and target 9 usually failed.

**Ruled out:** choosing K=10 as the primary cause; simple 64-cell quantization;
insufficient update count for the observed bad basins.

**Did not yet distinguish:** bad numerical channels from target-dependent
optimization.

Report: `docs/STAGE_A_K_SWEEP_RESULTS_2026-07-05.md`.

### Phase 3 — target/channel permutation and dense heatmap positive control

For K=10 and three seeds, we added:

- coordinate MSE with target assignment cyclically shifted by one;
- Gaussian target-heatmap supervision with the original assignment.

The shifted failures did not stay on numerical channels 3, 6 and 9. Some moved
with target identity, but not perfectly: target 9 was rescued and target 1
became difficult. The dense heatmap condition passed all ten targets in all
three seeds within 700--900 updates, with worst-channel errors
`0.142--0.193` cell64.

**Ruled out:** a fixed broken channel/head index; intrinsically impossible
targets; lack of basic model capacity; forward soft-argmax choosing a point
different from the heatmap expectation.

**Supported:** an interaction between target/image content and the
coordinate-only optimization path.

Report: `docs/STAGE_A_ATTRIBUTION_RESULTS_2026-07-05.md`.

### Phase 4 — direct soft-argmax gradient audit

We differentiated coordinate MSE and dense heatmap supervision with respect to
the same saved logits. The analytic coordinate gradient matched PyTorch
autograd to `3.73e-9` maximum absolute error.

For failed coordinate-trained units:

- median coordinate-gradient final/initial ratio was `0.00836` under identity
  assignment and `0.000418` under shifted assignment;
- dense heatmap gradients on the same logits remained strong (`2.68x` and
  `3.07x` their initialization values);
- 75% of failed units met the saturated-wrong-peak condition;
- the stricter target-region-starvation hypothesis was mixed (`~60%`, below
  its preregistered 75% support threshold).

A read-only temperature increase restored gradient magnitude but worsened
coordinate error by factors of approximately `2.4--7.3`; it was rejected as a
repair.

**Established for the four-frame task:** coordinate-only expectation
supervision permits wrong or overly concentrated maps whose useful gradients
can become negligible. This is not a numerical implementation error in
soft-argmax.

Report: `docs/STAGE_A_GRADIENT_AUDIT_RESULTS_2026-07-05.md`.

### Phase 5 — repair attempt 1: prediction-centred Gaussian JS

The first ground-truth-free repair penalized divergence between each predicted
heatmap and a one-cell Gaussian centred at its own detached predicted mean.

Synthetic semantic tests passed: the loss broadened spikes, concentrated
uniform maps, handled separated modes, was translation invariant and remained
finite. Note the scope of that invariance: the loss VALUE is unchanged when a
map and its (detached) centre translate together, but within a single gradient
step the detached centre is frozen, so the transient asymmetric mass
redistribution required to move the expectation is penalized. Value-level
translation invariance therefore did not imply gradient-level
non-interference — which is exactly the failure the R1 audit later measured.
A 200-update smoke also produced healthy heatmaps and usable counterfactual
gradients.

In the authoritative three-seed, 5,000-update R1:

- heatmap shape and gradient gates passed;
- coordinate localization failed 0/3 seeds;
- median coordinate errors were `2.34`, `2.34`, and `3.03` cell64.

A read-only parameter-gradient audit found that the shape loss removed about
`90--95%` of coordinate first-order descent at the heatmap head in all three
seeds. The network did transmit coordinate gradients; the simultaneous losses
were opposing each other.

**Contribution:** proved that “make the heatmaps Gaussian” is not sufficient.
An always-active Gaussian around the current prediction anchors the current
location and slows movement toward the target.

Reports:

- `docs/STAGE_R0_SHAPE_CONSTRAINT_RESULTS_2026-07-05.md`
- `docs/STAGE_R1_SHAPE_GATE_RESULTS_2026-07-05.md`
- `docs/STAGE_R1_GRADIENT_PATH_AUDIT_RESULTS_2026-07-05.md`

### Phase 6 — repair attempt 2: conditional dead-zone shape penalty

The fallback penalized only malformed maps using frozen bands on maximum
probability, effective support and dominant local mass. It was exactly silent
on successful dense-supervision controls. Direct free-logit prototypes repaired
spike, diffuse, uniform and separated-mode cases and then became silent.

The 200-update implementation smoke reduced coordinate error but had not
entered healthy shape ranges. It was initially described too positively; in
retrospect it proved wiring only and already contained the eventual warning.

The authoritative three-seed, 5,000-update R1 again passed 0/3:

- median errors: `0.043`, `0.439`, `0.066` cell64;
- worst-channel errors: `4.396`, `4.110`, `5.494`;
- targets 3, 6 and 9 failed in all three seeds;
- some channels were near-delta spikes, while others were excessively diffuse;
- shape and counterfactual-gradient gates failed.

The post-mortem found that the weight was calibrated globally at initialization
on very diffuse maps (`8.72e-6`). Its relative strength changed as heatmap shape
changed, and it became weak for several failed channels. Because the repair was
also computed after softmax, near-delta maps attenuated its gradients through
the same softmax Jacobian.

**Contribution:** rejected a conditional post-softmax shape penalty as a robust
repair under this calibration. It also exposed a methodological error: direct
optimization of free logits with the shape loss alone did not predict joint
optimization through the shared CNN.

Reports:

- `docs/STAGE_R1_DEADZONE_R0_RESULTS_2026-07-05.md`
- `docs/STAGE_R1_DEADZONE_GATE_RESULTS_2026-07-05.md`
- `docs/STAGE_R1_DEADZONE_FAILURE_POSTMORTEM_2026-07-05.md`

### Phase 7 — challenge to the four-frame gate

Reviewers correctly challenged the inference from the tiny task to the real
training regime. Existing 90-frame results showed much smaller errors than the
four-frame catastrophic failures. The claim that targets 3, 6 and 9 were
physically unlearnable was considered and then rejected because dense
supervision learned them in the same four-frame setting.

**Correction:** the four-frame task is useful for exposing a failure mechanism,
but failure there cannot by itself establish that the mechanism dominates
representative training.

### Phase 8 — representative validation-plateau pilot

We then ran the unchanged plain-coordinate instrument with:

- 60 train, 60 validation and 60 untouched test frames;
- seed 41, K=10, standard 64x64 architecture;
- minimum 1,000 epochs, validation every 25, 400-epoch 1%-improvement patience,
  hard cap 3,000;
- per-channel heatmap-shape and counterfactual-gradient probes throughout.

It reached a genuine plateau at epoch 1,175; the best checkpoint was epoch 775.

| Validation condition | Median of channel medians | P90 | On mask |
|---|---:|---:|---:|
| Unaugmented | 0.502 | 2.066 | 0.987 |
| Fixed augmentation | 0.512 | 2.068 | 0.985 |

Channels 1, 3, 7 and 8 had the principal errors. Channels 6 and 9, which were
catastrophic in the four-frame task, were now adequate.

Crucially:

- no channel was saturated at the best checkpoint;
- no inaccurate channel was saturated;
- no channel's counterfactual-gradient ratio fell below `0.01x` initialization;
- the minimum ratio was `0.185x`;
- no saturation or collapsed-gradient channel was observed from epoch 200
  through the plateau.

**Two-phenotype structure of the failing channels.** The per-channel shape
probes at the best checkpoint show that the four failing channels split into
two distinct regimes rather than one:

| Channels | Median error | Max probability | Effective support (cells) | Gradient ratio |
|---|---:|---:|---:|---:|
| 1, 3 | 2.18 / 1.81 | 0.88 / 0.95 | 1.7 / 1.3 | 0.18 / 0.19 |
| 7, 8 | 1.18 / 1.21 | 0.03 / 0.18 | 217.8 / 38.3 | 1.29 / 3.25 |

Channels 1 and 3 are compact-but-wrong: nearly concentrated maps below the
frozen `>=0.99` saturation threshold, holding the two lowest gradient ratios
in the run — a weakened form of the tiny-regime concentrated-wrong-peak state
rather than its absence. Channels 7 and 8 are diffuse with healthy gradients:
their error is consistent with off-peak probability mass dragging the global
expectation. Channels 0, 4 and 5 pass despite very large support (176–401
cells), showing diffuse maps are tolerable when the mass is symmetric about
the target. Any candidate replacement instrument should therefore be assessed
against both phenotypes separately, not only against aggregate error.

**Conclusion:** representative data diversity escapes the tiny-regime
saturation trap as formally defined. The unchanged instrument nevertheless
remains insufficiently accurate; the residual failure comprises at least two
distinct heatmap phenotypes, and cannot be attributed primarily to vanished
soft-argmax gradients.

Report: `docs/REPRESENTATIVE_COORDINATE_PILOT_RESULTS_2026-07-05.md`.

## 4. Explanations now ruled out or substantially weakened

| Candidate explanation | Current status | Evidence |
|---|---|---|
| Incorrect target transport/indexing | Ruled out as primary | Grounding checks pass; shifted assignment and dense control are internally consistent |
| Permanently broken channels 3/6/9 | Ruled out | Shifted targets change failures; dense supervision makes every channel valid |
| Targets 3/6/9 are impossible to represent | Ruled out in the tested setting | Dense heatmap control fits all ten |
| K=10 is the cause | Ruled out as primary | K=5/10/15/20 failure fraction is not monotonic |
| 64-cell rounding/quantization | Ruled out for large errors | Failures are multiple cells and wrong-region predictions |
| Accidental optional stride/128 path | Ruled out | Failed gates used the unchanged standard-64 path |
| Too few tiny-task updates | Ruled out for the observed bad basins | Failures persist after 5,000 updates |
| Soft-argmax forward implementation bug | Ruled out | Analytic/autograd agreement and successful dense-control readout |
| Global temperature is the fix | Rejected | Gradients increase but localization worsens substantially |
| Always-active prediction-centred JS is the fix | Rejected | It cancels coordinate descent and fails 0/3 |
| Conditional post-softmax dead-zone is the fix | Rejected under frozen design | It fails 0/3 and loses state-robust corrective strength |
| Saturation is the representative primary blocker | Ruled out for the seed-41 pilot | No saturated/collapsed-gradient channels, yet validation accuracy fails |

One minor metric issue was found: historical “cell64” reporting used `2/64`
while the exact spacing of `linspace(-1,1,64)` is `2/63`, a 1.6% difference.
This cannot explain multi-cell failures.

## 5. What is established

1. The model and target pipeline can represent all ten target locations under
   dense spatial supervision.
2. Coordinate-only expectation supervision is substantially less reliable than
   dense heatmap supervision on the four-frame task.
3. One concrete small-data failure mechanism is softmax saturation with weak
   target-directed coordinate gradients.
4. The first repair failed because it directly opposed coordinate movement.
5. The second repair failed because synthetic free-logit success and
   initialization-only global calibration did not transfer to shared-CNN
   training across changing heatmap states.
6. Representative data prevents catastrophic saturation, but does not produce
   sufficiently accurate or uniform per-channel localization.
7. The main operator/keypoint hypothesis remains untested; none of these
   supervised diagnostics establishes whether fitted-operator pressure selects
   semantic parts.

## 6. What remains genuinely unknown

The representative pilot narrows the unresolved cause to three non-exclusive
possibilities:

1. **Underconstrained spatial representation.** Coordinate MSE constrains only
   the heatmap expectation. Heatmap shapes remain extremely heterogeneous, so
   the representation may use unstable or nonlocal probability arrangements
   even without complete saturation.
2. **Local observability/architecture mismatch.** The convolutional output cell
   has an approximately 35-pixel receptive field and no explicit positional
   coordinates. Some FPS targets lie on long edges or weakly textured regions.
   Dense supervision proves subtle information is sufficient for memorization,
   but not that coordinate-only learning can generalize robustly there.
   Direct inspection of the target image content
   (`keypoint_net/diagnostics/check_failed_targets.py`; overlay and patch
   contact sheet in `outputs/target_content_analysis/`) supports a graded
   version of this cause: target 3 lies on the smooth hammer-head face against
   a nearly iso-luminant background, target 1 on the rounded, near-symmetric
   handle end, and targets 6/9 on long straight head edges — while robust
   targets sit on corners, junctions and tapering structures. Difficulty
   ordering is stable across every arm ever run: target 3 was the worst or
   near-worst channel under tiny coordinate training, full-data coordinate
   training, dense heatmap training (1.55–1.72 cell64 in the Phase-1 runs) and
   the representative pilot. Local content thus predicts WHICH channels fail,
   even though (per the dense control) it does not make them unlearnable.
3. **Shared-representation interference.** Ten heads share one encoder.
   Improvements for one channel can alter features needed by another. This has
   not been isolated with independent heads/frozen features or per-channel
   training.

The current evidence does not rank these three causes. Claiming one as “the
reason” would exceed the data.

## 7. Reasoning and process errors that caused unnecessary work

1. The 200-update dead-zone smoke was described as encouraging although its
   shape gate already failed. It demonstrated execution, not repair efficacy.
2. Synthetic free-logit optimization was treated as stronger evidence than it
   was. It did not test the shared CNN or the joint objective.
3. A globally calibrated loss weight at initialization was assumed to remain
   meaningful across radically different heatmap states.
4. The four-frame diagnostic was initially treated as a decisive instrument
   gate. The later representative pilot showed that its saturation mechanism
   does not dominate full-data training.
5. “5,000” was sometimes discussed imprecisely; it meant optimizer updates in
   the tiny task, not epochs.
6. Aggregate medians were repeatedly tempting but misleading: good medians can
   coexist with several unusable channels. Per-channel and tail metrics are
   mandatory.
7. (Review chain.) The prediction-centred JS design originated in external
   review as a ground-truth-free adaptation of DSNT's distribution
   regularizer, and was ratified without analyzing its gradient dynamics under
   translation; the stop-gradient added to prevent one pathology created the
   anti-translation force. The failure was a design error upstream of
   implementation.
8. (Review chain.) The target-content ("observability") hypothesis was first
   retracted on the basis of the dense arm's healthy GRADIENTS rather than its
   per-target LOCALIZATION, then over-restored as "targets are inherently
   unlearnable" before the dense control refuted that too. The surviving,
   evidence-consistent form is graded difficulty (Section 6, cause 2).

These are interpretation/design errors, not evidence of fabricated results.
They are recorded because an expert reviewer should not have to reconstruct
them from the conversation history.

## 8. Recommended next decision

Do not run another post-softmax heatmap-shape loss. Do not launch the five-arm
operator experiment yet.

The next bounded study should compare a small number of alternative coordinate
instruments under the **same representative validation-only protocol**, with
the test split untouched. Before implementation, an expert should review:

1. whether the required instrument accuracy is scientifically appropriate for
   the downstream fitted-operator sensitivity;
2. whether arbitrary FPS mask targets are a valid capability proxy for the
   kinds of visually identifiable keypoints the unsupervised method should
   discover;
3. which comparison best separates the remaining causes—for example direct
   coordinate regression with global spatial context, an explicitly
   position-aware heatmap architecture, or independent/per-channel capacity;
4. whether a structural heatmap parameterization that guarantees a valid
   distribution is preferable to another penalty on an unconstrained map;
5. **a Stage-B transfer constraint on candidates:** the final instrument must
   be trainable by coordinate-level, ground-truth-free gradients, because the
   Stage-B losses (operator fitting, grounding) supply only those. Instruments
   that pass the supervised gate only WITH dense target-heatmap training or
   ground-truth-centred regularization cannot carry that training signal into
   Stage B and are eligible as capacity references only — unless the project
   explicitly adopts a ground-truth-free dense signal (e.g., reconstruction),
   which changes the scientific claim and must be declared as such;
6. per-arm reporting of BOTH failure phenotypes from the representative pilot
   (compact-but-wrong vs diffuse-drag; Section 3, Phase 8), so the comparison
   attributes WHICH phenotype each instrument fixes rather than only ranking
   aggregate error.

The success criterion must remain semantic: accurate, stable coordinates on
held-out validation frames, with every channel reported. File validity,
decreasing training loss or healthy-looking heatmaps are insufficient.

Only after one instrument passes a frozen three-seed representative gate should
the project resume the grounding-versus-operator-loss comparison.

## 9. Artifact and provenance index

- Core-file changes: `docs/CORE_FILE_CHANGELOG.md`
- Diagnostic-week review: `keypoint_net/diagnostics/DIAGNOSIS.md`
- K sweep: `docs/STAGE_A_K_SWEEP_RESULTS_2026-07-05.md`
- Attribution/dense control: `docs/STAGE_A_ATTRIBUTION_RESULTS_2026-07-05.md`
- Gradient audit: `docs/STAGE_A_GRADIENT_AUDIT_RESULTS_2026-07-05.md`
- JS repair R0/R1: `docs/STAGE_R0_SHAPE_CONSTRAINT_RESULTS_2026-07-05.md`,
  `docs/STAGE_R1_SHAPE_GATE_RESULTS_2026-07-05.md`
- JS parameter-gradient audit:
  `docs/STAGE_R1_GRADIENT_PATH_AUDIT_RESULTS_2026-07-05.md`
- Dead-zone repair and post-mortem:
  `docs/STAGE_R1_DEADZONE_GATE_RESULTS_2026-07-05.md`,
  `docs/STAGE_R1_DEADZONE_FAILURE_POSTMORTEM_2026-07-05.md`
- Representative pilot:
  `docs/REPRESENTATIVE_COORDINATE_PILOT_RESULTS_2026-07-05.md`
- Representative pilot Mac archive:
  `/Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r2_representative_pilot_20260705_210253/`
- Target image-content analysis (script + overlay + patch contact sheet):
  `keypoint_net/diagnostics/check_failed_targets.py`,
  `keypoint_net/diagnostics/outputs/target_content_analysis/`
- Project-level conceptual explainer (hypothesis, pipeline, literature,
  readout ladder): `/Users/kirubeso.r/Documents/PhD/docs/PROJECT_EXPLAINER_2026-07-05.md`

All reported uncertainty is descriptive. The experiments concern one hammer
object and correlated frames from one cyclic orbit. Optimization seed is the
replication unit where multiple seeds exist. No population-level inference,
hypothesis test or causal claim beyond the stated controlled interventions is
made.
