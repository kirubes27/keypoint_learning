# Decision synthesis v2 - 2026-07-26
Supersedes DECISION_SYNTHESIS_MERGED_2026-07-26.md (v1). Change control: edits only via a new version with a changelog line. Status: draft pending dual review (opus + gpt-5.6-sol); becomes actionable when both report no remaining blockers. Owner: Kirubes.
Changelog v1 to v2: fixed both artifact paths; reinstated statistical scope and per-seed spread; narrowed Decision 1 closure language and removed the asymmetric reopen prerequisite; added Gate 0 representative-replay pre-gate; split Decision 3 into three gates; demoted citation layer from binding to verified-as-of-date; softened two overstated phrases; added provenance manifest; inlined gates into actions.

## STATISTICAL SCOPE (applies to every gate below)
All existing diagnostics are descriptive, not inferential: n=3 optimization seeds, one object, and the Stage-A probe used four correlated frames. Observed per-seed median-error spread in the deadzone gate: seed 42 = 0.043, seed 43 = 0.439, seed 44 = 0.066 cells - an order of magnitude. Pre-registration: 3/3 seeds passing = adopt; 2/3 = adopt provisionally and add two seeds before any downstream claim; 0-1/3 = fail. No population-level inference is claimed from n=3.

## DECISION 1 - Ordering: readout repair first; post-softmax fixed-weight shape losses timeboxed out
What is empirically established: ONE loss-side design - the conditional-deadzone shape constraint, post-softmax, single globally calibrated fixed weight - was run and failed 0/3 seeds (artifact: /Users/kirubeso.r/Documents/PhD/cluster_downloads/stage_r1_deadzone_gate_20260705_160834/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_deadzone_gate/R1_DEADZONE_GATE_SUMMARY.json - verdict: fallback R1 fails; stop this instrument design and review). Post-mortem (artifact: /Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_deadzone_failure_audit/R1_DEADZONE_FAILURE_AUDIT.json): fixed weight calibrated on diffuse initial maps; saturated maps attenuate both losses through the same softmax Jacobian.
Policy consequence (a time-budget decision, NOT empirical elimination of all loss-side mechanisms): post-softmax fixed-weight shape losses are timeboxed out of the current cycle. Pre-softmax constraints, state-adaptive weighting, and other loss-side mechanisms remain admissible but deprioritized; any such proposal must pass its own synthetic gate before image training and is evaluated independently of the readout track - readout failure is NOT a prerequisite for considering them. Readout repair goes first in the queue because it attacks the measurement layer directly and Runs A and B ranked it first from independent grounding. Never combine readout and loss changes in one experiment matrix.

## GATE 0 - Representative-replay pre-gate (run BEFORE any retraining; zero training cost)
Motivation (sol finding, verified): the Stage-A failure subtypes (near-target saturation, far-peak starvation) come from a 4-frame supervised probe; the 60-frame representative pilot found NO saturated channels and NO collapsed coordinate gradients at the validation plateau (artifact: keypoint_learning_fitted_operator/docs/REPRESENTATIVE_COORDINATE_PILOT_RESULTS_2026-07-05.md), leaving weak observability, CNN interference, and underconstrained heatmaps as live suspects. Therefore the mechanism local windowing repairs may not be the mechanism producing representative errors.
Procedure: replay all three candidate readouts (current global, local/windowed, coarse+offset) on the frozen representative-pilot checkpoints; stratify high-error cases by whether the dominant mode is correct and error comes from secondary-mode pull or sub-cell saturation, versus wrong or diffuse modes.
Decision rule: if at least half of high-error cases are correct-dominant-mode failures, proceed to Action 2 with local windowing ranked first. If most high-error cases are wrong-mode or diffuse, local windowing loses rank, and the diagnosis shifts to representation/observability - route to the diagnostic head (Decision 2.3) and re-plan. Closing artifact: GATE0_REPLAY_RESULTS.md with per-stratum counts.

## DECISION 2 - Ranked instrument repairs (conditional on Gate 0)
1. LOCAL/WINDOWED DOMINANT-MODE SOFT-ARGMAX - readout-only swap, everything else frozen. Addresses correct-dominant-mode failures (sub-cell saturation, secondary-mode pull); does not fix far-wrong-peak starvation and can entrench a wrong mode, so far-peak escape and translation compatibility are explicit gates. Grounding: Lin et al. CVPR 2021 (arXiv:2104.02273); Zhang et al. CVPR 2024 (arXiv:2311.17034) - NOTE their own supplement reports window soft-argmax during training helped the strictest PCK threshold while hurting looser ones, so this grounding justifies testing, not presuming success. The current head is a DSNT-style global expectation (model.py, softmax over flattened heatmap into grid expectation, near line 28-30).
   PREREQUISITE: a frozen spec document (READOUT_SPEC_v1.md) fixing window algorithm and radius rule, peak selector, gradient path (straight-through or differentiable), tie and boundary behavior, data split and its preservation, seeds, optimizer, budget, baseline replay protocol, aggregation unit for median and p90, numeric equivalence margins, far-wrong definition, and the translation test. No launch without the spec.
   GATE: median at most 0.50 cell AND p90 at most 1.50 cells on the held-out split, per the 3/3 - 2/3 - fail rule in Statistical Scope; stratified far-wrong vs near-target; translation-equivariance residual not worse than baseline by more than the equivalence margin; no increase in persistent far-wrong peaks. Fail via wrong-window trapping -> repair 2. Fail despite correct mode selection -> repair 3.
2. COARSE CELL + LOCAL CONTINUOUS OFFSET (CenterNet decomposition, arXiv:1904.07850). Falsifier: the unsupervised objective may provide no legitimate coarse teaching signal; wrong-cell selection or boundary instability fails it.
3. PARAMETRIC GAUSSIAN-MEAN / DIRECT-COORDINATE HEAD - DIAGNOSTIC ONLY. Diagnostic pass + local-readout failure -> redesign head. Diagnostic pass + no downstream benefit -> bottleneck is representation identifiability; stop decoder work.
EXCLUDED THIS CYCLE (grounds): temperature (Stage A: 49.6x gradient, 2.42x worse error); prediction-centred Gaussian shaping (R1: 90-95 percent cancellation); plain global-expectation swap (already the implementation); resolution-first (Stage A excluded 64x64 as primary cause); VICReg covariance on coordinates (rank-2 rigid orbit); conditional deadzone (0/3 seeds).

## DECISION 3 - Coincidence stationary point: three gates, no shortcut into the loss stack
Mechanism selected: hard one-to-one assignment of channels to distinct on-object candidate modes with a fixed channel-specific tie-breaking rule - the selected remover among the mechanisms considered (isotropic penalties merely penalize: for L = phi(d^2), grad_i L = 2 phi-prime (p_i - p_j) = 0 at coincidence; queries, noise, slot competition, and entropic Sinkhorn alone only destabilize). Grounding: DETR arXiv:2005.12872; MESH arXiv:2301.13197; Gumbel-Sinkhorn OpenReview Byt3oJ-0W. Claim cost: architectural anti-collapse import - the claim becomes prediction pressure plus exclusive allocation. Status: untested in this project.
GATE 3a - toy synthetic (one day, no cluster): two channels, identical logits, two separated on-mask candidate peaks, zero noise; conditions: current loss, KeypointNet hinge, B-KinD exponential, random query perturbation, entropic Sinkhorn, hard assignment + fixed tie-break; measure step-zero antisymmetric gradient norm, then after 100 matched steps min pairwise distance, on-mask rate, assignment churn (churn threshold to be fixed in the runfile before execution); 100 draws for stochastic variants. Authorizes ONLY Gate 3b.
GATE 3b - full-CNN candidate-generation smoke test: define the candidate-extraction mechanism in the real model; verify at least K valid on-object candidates exist across representative frames; verify translation stability of extraction and assignment. Authorizes ONLY Gate 3c.
GATE 3c - integration gate: hard-assignment variant enters the loss stack only after 3a and 3b pass, evaluated with the downstream coordinate and operator gates, claim cost recorded in the paper draft.

## DECISION 4 - Novelty defense (verified as of 2026-07-26; re-verify at submission - not binding)
Conjunction defined: (a) explicit 2D spatial keypoints from image heatmaps via a differentiable spatial expectation; (b) ONE shared learned affine coordinate map k-next = A k + b, A in R2x2, applied pointwise to all keypoints; (c) no pixel or observation reconstruction loss anywhere in training; (d) no explicit transformation parameters or action labels supplied to the network (temporal pairing itself is weak supervision - never claim label-free).
Null: no exact match found in the searches performed (Runs A, B, C - partially overlapping seed lists, so independence is partial). Search result, not proof of nonexistence.
Framing: the moat is (b) inside the (a)+(c)+(d) system; (a)+(c)+(d) alone is MINT territory. Mechanism-level defense of sharedness required: the shared-vs-per-keypoint operator ablation. Koopman/STN inoculation paragraph due before submission.
Pre-empt order: 1. MINT (PMLR 202:40225-40253). 2. EbC (Schmidt/Schneider/Bethge, NeurIPS 2025, arXiv:2510.21706, OpenReview kvI0QTVRQD). 3. Ruiz-Morales (AAAI 2026, DOI 10.1609/aaai.v40i30.39708) and CARE (ICLR 2024). Boundary set: Transporter, B-KinD, HAE, MSP, NFT, C-SWM, Compositional Koopman (arXiv:1910.08264), KOROL, Seq-JEPA (arXiv:2505.03176), PlaySlot, RP-GSSM.
Citation state (verified 2026-07-26): Hromadka arXiv:2505.23569 = ICML 2026 accepted per ICML catalogue and UCL Discovery camera-ready - no PMLR pages until published. LPWM arXiv:2603.04553 = ICLR 2026 Oral. Delta-JEPA arXiv:2606.31232 = real, action-reconstructing, not label-free. BCIR: Gu, Yang, Mi, Yao, TPAMI 2023, DOI 10.1109/TPAMI.2023.3264742; related Gu et al. ICCV 2021 pp. 11067-11076.

## IMMEDIATE ACTIONS (each with its closing artifact; documents stop after this one, except the two spec/runfiles named)
0. Gate 0 replay on representative checkpoints -> GATE0_REPLAY_RESULTS.md.
1. Freeze READOUT_SPEC_v1.md (Decision 2.1 prerequisite list).
2. Run the readout-only gate per spec -> READOUT_GATE_RESULTS.md.
3. Gate 3a toy synthetic (runfile with churn threshold fixed first) -> COINCIDENCE_3A_RESULTS.md.
4. On 3a pass: Gate 3b smoke test -> COINCIDENCE_3B_RESULTS.md. Integration only via Gate 3c.
5. Near submission: re-run novelty scan; write EbC and CARE paragraphs; re-verify citation state.

## PROVENANCE MANIFEST
On disk in this folder: prompt-runB.md, prompt-runA.md, runA-output.md, DECISION_SYNTHESIS_MERGED_2026-07-26.md (v1, superseded), this file. NOT yet on disk: Run B output (exists in the Claude chat transcript and in ChatGPT - EXPORT REQUIRED, save as runB-output.md); Run C report (Claude artifact - export as runC-output.md); opus and sol critiques (in /tmp/rate_merge/ and /tmp/runC_check/ - copy into this folder before /tmp is purged). Claims of three-arm convergence are provisional until these land here.
Overturn ledger: shape-constraint-first ordering overturned by Run A artifact; Run C Hromadka disposition overturned by sol primary evidence; opus Delta-JEPA suspicion resolved false alarm; Run B Q2 answer sharpened by Run A; EbC missed by all three runs, found in critique layer.

## v2.1 AMENDMENTS (2026-07-26, post dual-review; changelog per change-control rule)
1. GATE 0 FEASIBILITY FIX (sol): the replay compares ONLY readouts computable from existing frozen heatmaps - current global expectation vs local/windowed variants (several window radii). Coarse+offset CANNOT be replayed (no trained offset head exists); its coarse half is approximated by cell-argmax as a mode-correctness probe only, and it enters ranking solely through Decision 2 gates after training.
2. GATE 0 DEFINITIONS FROZEN (opus + sol): sample unit = channel-frame pair over all validation frames of the representative pilot (one seed - descriptive only, noted per Statistical Scope); high-error = per-pair coordinate error above the 75th percentile of that distribution; correct-dominant-mode = heatmap argmax cell within 1 cell of the supervised target cell; strata and counts reported over that fixed denominator. Decision rule unchanged (at least half of high-error pairs correct-dominant-mode -> windowing keeps rank).
3. SCOPE CORRECTION: the representative pilot is ONE seed (41), not three; every Gate 0 conclusion is descriptive and feeds ranking only, never a pass/fail claim.
4. GATE 3a/3b NUMERIC CRITERIA: to be frozen in COINCIDENCE_RUNFILE.md before execution (antisymmetric-gradient nonzero margin, separation target, churn threshold, on-mask minimum, K value) - execution is blocked until that runfile exists, same rule as READOUT_SPEC_v1.md.
5. MANIFEST CORRECTION: v1 and v2 reviews by both critics are now on disk in this folder (v1-review-sol.md, v1-review-opus.md, v2-review-sol.md, v2-review-opus.md). Still pending export: runB-output.md, runC-output.md. No repository commit recorded because docs/ is not under version control - noted as a known gap.

## v2.2 AMENDMENTS (2026-07-26, after executed Gate 0 and Fable 5 high audit)
1. GATE 0 IS CLOSED: the completed frozen replay observed 6/150 (4.0 percent) correct-dominant-mode high-error pairs, versus 75/150 required. Local/windowed readout is demoted. Gate 0 must not be rerun or bypassed.
2. DECISION 2.3 PANEL: before implementation, freeze `DECISION_2_3_DIAGNOSTIC_HEAD_SPEC_v1.md`. The primary panel is (A) one channel-shared learned coordinate map from spatially centered raw logits, (B) the same-capacity channel-shared learned map from spatial-softmax probabilities, and (C) the fixed production expectation. All other training factors remain matched. This records the addition of attribution control B and the mandatory matched baseline C.
3. SEED RULE CORRECTION: the Decision 2.3 finalizer must implement Statistical Scope exactly: 3/3 pass; 2/3 provisional plus two unchanged seeds before any downstream claim; 0-1/3 fail. The older at-least-2/3 Stage-A finalizer must not be reused unchanged.
4. BRANCH COMPLETION: A pass with B/C failure supports bypassing softmax conditioning; A/B pass with C failure supports a learned decoder but not a specifically non-softmax mechanism; all three passing points to seed variability; A/B/C failure shows the tested logit-level readouts are insufficient but does not prove encoder incapability. A compact direct-from-feature control, if needed, requires a new versioned spec.
5. SECONDARY-MODE NON-GATE: no target-local-secondary-mode analysis is inserted before Decision 2.3. The saved Gate 0 artifact lacks nondominant peaks, and the proposed 50 percent rule was not frozen. Any later topology audit is descriptive and requires its own changelog.
6. TINY-GATE ROLE: the historical four-frame control remains mechanism evidence but is not a scientific stop gate for the representative 60-frame Decision 2.3 panel. D0 semantic tests and a three-arm Slurm smoke are the full-run gate.
7. DOWNSTREAM BOUNDARY: supervised diagnostic success is not downstream benefit. No identifiability or operator claim is authorized until a separate versioned spec fixes a no-coordinate-label metric, margin, matched seeds/budget, and collapse/off-object guards.

## v2.3 AMENDMENTS (2026-07-26, Decision 2.3 execution control)

1. PUBLICATION AUTHORITY: Kirubes explicitly approved using the existing
   public repository and branch `agent/preoperator-gates-20260726`. This
   supersedes the earlier private-remote prerequisite; it does not relax the
   one-branch, clean-checkout, immutable-artifact, or reviewer-read-only rules.
2. PHYSICAL TEST ISOLATION: Decision 2.3 lock/train/probe may enumerate frozen
   split metadata but may open only train/validation image and mask contents.
   Test image/mask contents may first be opened inside the one frozen
   finalization command after all run artifacts pass preflight.
3. D1 BUDGET: one sequential A/B/C seed-42 Slurm job; exactly two epochs per
   arm; evaluation/audit at epochs 0, 1, and 2; one GPU, eight CPUs,
   `--mem-per-cpu=5000`, 30 minutes, no explicit account or partition.
4. D2 RESOURCES: one `0-8%2` Slurm array for A/B/C crossed with seeds
   42/43/44; at most two GPUs concurrently; each task requests one GPU, eight
   CPUs, `--mem-per-cpu=5000`, and one hour. No interim arm tuning, dropping,
   or promotion is permitted.
5. SOURCE BINDING: configs and finalization bind the Decision 2.3 script,
   imported model/data/augmentation/checkpoint/evaluation modules, exact Git
   commit, dataset semantic/index/operator metadata hashes, and clean-worktree
   state.
6. PROBE BINDING: the optional A/B frozen probe accepts only the canonical
   seed-41 checkpoint and config hashes recorded in the provenance manifest.
7. ONE-SHOT TEST ARTIFACTS: preflight refuses either an existing test-metrics
   or test-predictions artifact, verifies the selected checkpoint, history,
   validation metric, and gradient audit, then evaluates each frozen
   arm/seed checkpoint exactly once. Seeds 45/46 require a hash-bound immutable
   initial report with exactly 2/3 for that arm.

## v2.4 AMENDMENTS (2026-07-26, pre-D1 blocker closure)

1. RUNTIME MATCHING: record complete GPU identity, but compare only the
   software and determinism environment across D2 tasks, resumes, extensions,
   and finalization. Scheduler-selected GPU hardware need not be identical.
2. ONE COMMITTED TEST RESULT: an interrupted finalizer with no test artifact
   may resume unchanged; two matching artifacts are recovered without
   reevaluation; any observable partial or inconsistent state fails closed.
   This operationally defines "exactly once" as one immutable committed result
   per frozen checkpoint and forbids adaptive changes between attempts.
3. D1 EPOCH ZERO: store full unaugmented/fixed-augmented validation metrics at
   epoch 0 separately from the epoch-1/2 training history.
4. ARRAY ISOLATION AND RESUME: each D2/extension task runs from its own clean
   clone at the expected commit and passes `--resume` only when a matching
   incomplete run has a last checkpoint.
5. RUNFILE BINDING: all five Decision 2.3 Slurm runfiles are SHA-256-bound
   into the prelaunch lock, configs, D1 report, finalization plans, and
   aggregate reports before any cluster execution.
