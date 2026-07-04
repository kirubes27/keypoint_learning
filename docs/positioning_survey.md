# Positioning the Keypoint-Operator Approach

**A survey and scope assessment for the "keypoints that linearize object transformations" project**

Prepared for: Kirube · Computational Cognitive Science PhD
Anchors: Yu (ToI, PNAS resub. 2026) · Liebenow et al. (CCN 2026, "Paul") · target ECCV geometric-intelligence workshop (4-page, ~June 13)
Companion files: `positioning_refs.csv` (27 verified references), `positioning_table.csv` / `positioning_table.png` (axis grid)

---

## 0. The one-sentence positioning

> **You are learning a *spatial, interpretable* coordinate chart (keypoints) in which a rigid object's transformation becomes a single *global linear* operator — and you learn both the chart and the operator jointly from *temporal prediction alone*, with no reconstruction and no supplied transformation labels on the operator.**

That specific combination is not occupied by any prior method I could find. The positioning grid makes this visible: yours is the only row where all six design axes take their distinctive value simultaneously.

*A sharper restatement (added after the operator-spectrum analysis):* the deepest version of the claim is not "prediction-only keypoints," but **keypoints as the coordinates that make an operator's effects legible** — the coordinates where operator-simplicity and physical (k=1) grounding coincide. Under that framing your current Phase-A result is a *nuanced* one: the operator matrix is recovered cleanly (a near-exact small-angle rotation), yet operator-simplicity alone is satisfied by charts whose keypoints are duplicated or drifting — so recovering the operator is **not** sufficient to fix which coordinates constitute a stable representation. That honest negative is the contribution, not a blemish on it (see §3, §4).

{{artifact:ab234334-3173-415d-b420-89988634200c}}

*Figure 1. Sixteen representative methods (rows) scored on the six axes that define the design space (columns). Deep blue = the cell matches your design choice; light blue = same family but not identical; grey = a different choice. Your row (red border) is the only one that is deep-blue across all six columns. The nearest neighbours — Minderer '19 (keypoints + dynamics), the Koopman line (learned linear operator), and the ToI demo (fitted linear operator on TDW rotation) — each match on only one or two axes, and each differs from you on the axis that matters most for its family. Full per-method notes in `positioning_refs.csv`.*

---

## 1. How the field is organized, and where the empty cell is

I grouped the literature into seven method families plus the three lab-adjacent anchors (and a short methods note). The point of the grouping is not completeness — it is to show that **every family makes one assumption you are dropping.**

**A. Unsupervised object landmarks** (Thewlis '17, Jakab '18, Zhang '18, Lorenz '19).
These *do* produce spatial, interpretable keypoints — the representation you want. But the learning signal is either **equivariance to a known synthetic warp** (Thewlis: you apply the transform, so you already know it) or **image reconstruction** (Jakab/Zhang/Lorenz: the keypoints are an information bottleneck for regenerating the image). Crucially, **none of them learn a transformation operator.** The keypoints are a static descriptor; there is no object in these papers analogous to your `W`. So they answer "can keypoints emerge without labels?" (yes) but not "can keypoints emerge *because they linearize dynamics?*" (your question).

**B. Keypoints for dynamics / control** (KeypointNet '18, Transporter '19, Minderer '19, Chen '21, B-KinD '21/'22).
This is your closest family and the one to cite most carefully. These methods put keypoints *and* motion in the same system. But look at the operator column: KeypointNet needs **known relative pose**; Transporter and B-KinD **reconstruct** and have **no operator at all** (keypoints feed a downstream policy or a behavior classifier); Minderer '19 is the sharpest comparison — it learns keypoints *and* a dynamics model — but the dynamics model is a **stochastic nonlinear decoder trained to reconstruct future video frames** (a VRNN), not a single linear operator, and it is not tested for compositional linearity. **Your claim that a *shared, global, linear* operator suffices is exactly what this family does not test.** This is your headline contrast.

**C. Equivariance by construction** (G-CNN '16, Harmonic '16, Capsules '17, canonicalization '22, Marchetti '23).
Here the operator/linearity is **built in for a group you specify in advance.** A G-CNN is equivariant to a chosen discrete group because its filters are designed that way; canonicalization networks achieve equivariance by learning to undo pose against a known group. You are doing the opposite: the operator is **discovered from data**, not wired in, and it lives in an **interpretable spatial chart**, not an opaque feature space. Cite these to say "we do not assume the group; we let prediction pressure produce it."

**D. Koopman / linearizing operators** (E2C '15, Takeishi '17, Lusch '17).
This is the mathematically honest name for "find coordinates where the dynamics are linear," and it is the family your intuition keeps gesturing at. The critical technical distinction (expanded in §3): classical Koopman **lifts to a higher-dimensional (often infinite) observable space** to achieve *global* linearity, and is trained by **reconstruction of observables**. You are doing the reverse — a **low-dimensional** chart — and you get away with it only because your generator (rigid motion) is genuinely low-dimensional. So Koopman is a *framing you can borrow* but not a method you are extending; being precise about this protects you from an obvious reviewer objection.

**E. Symmetry / generator discovery** (Higgins '18, Zhou '20, L-conv '21, LieGAN '23, Homomorphism-AE '22).
These *discover* the group/generators from data — the most ambitious framing, and the one you flagged as a "maybe." But they operate on **full representations**: LieGAN is generative (it must model the whole data distribution), Homomorphism-AE and L-conv reconstruct or act on complete feature vectors. Your worry is correct and important: **keypoints are a partial, non-reconstructive representation, so "discover the symmetry group" is not obviously well-posed from them** (§3). Note LieGAN's author list — Yang, Walters, **Dehmamy, Yu** — this is Angela's line; it is the natural place your project would eventually connect to symmetry discovery, *if* you can make it well-posed.

**G. Emergent localized units — "why keypoints at all"** (Bau '17 / Network Dissection, Locatello '20 / Slot Attention, Caron '21 / DINO).
A body of work shows that spatially-localized, instance-aligned units *emerge implicitly* across CNNs, object-centric models, and self-supervised transformers, with no keypoint supervision. This is the family that motivates your project's opening question — but every one of these papers **describes** the emergence and none **explains** it. Your contribution reframes them: keypoints emerge not as a perceptual heuristic but as the coordinates that make an operator's effects *legible* (easy to express, compose, and compare across instances). Cite this cluster to say "emergence is observed everywhere and explained nowhere; here is a functional account."

**H. Irreducible-representation / frequency selection** (Koyama '23 / Neural Fourier Transform).
This is the sharpest framing of your novelty, and its most important prior art. Linear-predictability alone leaves *which frequency / irrep* the representation uses as a free degeneracy; soft-argmax keypoints are physically pinned to the **k=1 fundamental**, whereas a generic latent is free to pick any harmonic. So the learned operator's block-spectrum becomes a *measurement of which chart the representation selected.* The Neural Fourier Transform is the paper you must cite and distinguish: it establishes that capacity-limited latents select dominant irreps **given a fixed invariant kernel** — anticipating the "degeneracy among harmonics" half — but it does **not** address physical k=1 pinning of spatial coordinates, nor the keypoint-vs-generic-latent comparison. That comparison is the open, essentially-unpublished contribution.

**The empty cell.** Reading down the operator and supervision columns: the methods that give you a *learned linear operator* (D, parts of B/E) all pay for it with **reconstruction and opaque high-dimensional latents**; the methods that give you *spatial interpretable keypoints* (A, B) mostly have **no operator** or a **nonlinear reconstructive** one. Nobody sits where you sit: **spatial keypoints + jointly-learned global-linear operator + prediction-only.** That is a real, defensible gap — not a crowded corner.

There is a sharper way to name the gap, and it is worth stating precisely because it is the version most resistant to a "this has been done" objection. Prediction-only linearity does **not** uniquely determine a representation — it leaves a *degeneracy over which harmonic / irrep* the chart uses (cluster H). What breaks the degeneracy in your setup is that soft-argmax keypoints are **physically pinned to the k=1 fundamental** in image space, whereas a generic latent (a VAE code, a Koopman observable, a slot) is free to select any irrep. So your contribution is not merely "prediction-only supervision"; it is that **spatial keypoints are the coordinates on which operator-simplicity and physical grounding coincide**, and the operator's block-spectrum then *reads out which chart was selected.* The Neural Fourier Transform (Koyama '23) is the one paper that partially anticipates this — it shows capacity-limited latents select dominant irreps — so the honest, defensible framing is: NFT settled the degeneracy question for abstract latents given a fixed kernel; the open question is the **keypoint-vs-generic-latent irrep-selection comparison**, which no one has run.

---

## 2. The three lab anchors — and why you are not redundant with any of them

These three matter most because a reviewer (and your committee) will ask "isn't this just Angela's demo / Paul's paper with keypoints?" You need a crisp answer for each. You have one.

### 2.1 Angela's ToI ("The Art of Making Problems Simple")

ToI is your **theoretical parent**, and you should embrace that framing rather than distance from it. The paper's thesis is that intelligence = the ability to construct **representational normal forms**: internal coordinate systems in which prediction/comparison/inference become simple, built by (1) discovering lawful **generators** of variation, (2) **quotienting** irrelevant distinctions, (3) constructing **measurable structure**, (4) reorganizing so computation is cheap.

Your project is a **concrete, spatial instantiation of exactly this program**:

| ToI concept | Your instantiation |
|---|---|
| Measurable structure / coordinate chart | The 10 keypoints `p ∈ ℝ²ᴺ` |
| Discovered generator | The learned linear operator `W` (≈ +6° rotation) |
| Normal-form simplification | Making the generator's action *linear and compositional* over the full orbit |
| Learning pressure | Multi-step prediction / operator predictability |

But here is the **decisive difference** from ToI's own constructive demonstration (Section 5 of the PNAS draft). Angela's demo:
- uses the **TDW control parameters as known transformation supervision** ("structured self-supervision from *controlled* transformations"), i.e. the operator is fit against transforms the system is *told*;
- represents rotation as a **discrete orbit of prototypes with a circulant shift operator estimated by pooled least-squares**, then quotients by **orbit pooling** over that discretized ring;
- lives in an **abstract latent code**, not spatial keypoints.

**You differ on all three.** You want the operator learned from **prediction alone** (no control-parameter supervision on `W`); you use a **continuous linear operator in a spatial keypoint chart**, not a discrete prototype ring with pooling; and your coordinates are **on-object, interpretable, and eventually semantic** (eyes/mouth for faces), which the latent orbit code is not. So your contribution to the ToI program is: *"generator discovery can be done in an interpretable spatial chart, driven purely by predictability, without being handed the transforms."* That is a genuine addition, not a re-run.

### 2.2 Paul's paper (Liebenow, German, Bauer, Yu — CCN 2026)

Paul's paper is the **most direct methodological sibling**, so this is the comparison to get exactly right. Same lab, same "cognitive operators on internal representations" framing, same mental-rotation / SO(3) setting. His approach:
- an encoder → **VAE latent**, trained with **reconstruction + KL** (β-VAE) **plus curvature-regularization** terms (`ℒCurv`, `ℒCurv-Var`) defined on triplets of points;
- formalizes transformations as **Lie group actions in SO(3)**, cognitive operators as **vector fields on the latent manifold**, and pushes the latent toward **locally linear, low-curvature** geometry;
- goal: a latent where operators are efficiently *learnable and accurate*, demonstrated on synthetic mental-rotation stimuli.

The contrasts that make you non-redundant:

1. **Reconstruction vs. prediction.** Paul's latent is anchored by a reconstruction VAE (it must be able to regenerate the image). Yours is anchored by *prediction of the next state* with **no decoder**. This is the single biggest methodological fork in the whole survey — reconstruction is the assumption almost everyone makes and you are the one dropping it.
2. **Opaque latent vs. spatial keypoints.** Paul's coordinates are VAE latents (interpretable only via PCA projection). Yours are literally points on the object — the interpretability and the eventual *semantic* grounding (which keypoint is "the eye") come for free from the representation choice.
3. **Locally linear vs. globally linear.** Paul regularizes **curvature** to get *local* linearity (a vector field, valid in a neighborhood). Your result — a single shared `W` matching a +6° rotation across the whole 360° orbit — is a claim of *global* linearity of the operator over the entire generator range. These are different and complementary geometric claims. In fact, **Paul's curvature framework may be the right tool for your non-rigid future** (§3), where a single global `W` will break down and you will need locally-linear patches.

Framing to use: *"Paul asks what latent geometry makes operators efficient (curvature regularization on a reconstructive latent); we ask whether the operator-predictability pressure alone can produce an interpretable spatial chart in which the operator is globally linear. The approaches are complementary — his geometry tools apply directly when we relax rigidity."*

### 2.3 The ECCV geometric-intelligence workshop (target, 4 pages, ~June 13)

You have nothing written yet, so the useful thing here is **scoping what the 4 pages should claim**, because a workshop paper rewards one sharp result, not a system. Based on where the project actually is (per your 27 June handoff: the operator result is solid — shared 2×2 ≈ +6° rotation across all 324 runs, ‖A−R(6°)‖ median ≈0.03 on the clean `inv=cyc=0` cells; the open problem is representation quality — collapse, dead channels, sliding), the defensible 4-page claim is:

> **"Keypoints that make rigid motion linear emerge from prediction alone."** A single figure showing (a) unsupervised keypoints on the object, (b) the learned `W` recovering the true rotation angle across the full orbit, (c) compositional k-step rollout staying stable, and (d) the honest limitation (representation-quality failure modes and the fixes in progress).

This is *exactly* the size of result a geometric-intelligence workshop wants, and it is already in hand. The survey's §4 verdict argues you should write the workshop paper on the **rigid-motion result you already have**, and treat the non-rigid/faces/symmetry directions as the PhD programme behind it — not as things you need before submitting. See §4.

---

---

## 3. Stress-testing the trajectory: is each next hop a real contribution or an increment?

You listed a sequence of "maybe" directions. Here is an honest per-hop assessment — nearest prior work, whether it is a genuine research question or an engineering increment, and the key risk. This is the evidence base for the narrow-vs-programme verdict in §4.

### Hop 1 — yaw + pitch + roll → full SO(3)
**Genuine, and load-bearing.** Right now you have a 1-parameter group (in-plane rotation ≈ SO(2)), and a single 2×2 block does the job. Moving to full 3-DOF rotation is the test of whether your central claim *scales*: does one shared linear operator per generator still emerge, or do you need one `W` per axis and a composition rule `W_yaw · W_pitch`? **This is exactly the "compositional" in your thesis becoming non-trivial** — SO(3) is non-commutative, so operator composition order matters, and showing your learned operators respect that is a real result. Nearest prior: KeypointNet (3D, but supervised pose); Paul (SO(3) but latent+reconstruction). Risk: 2D keypoints from a single view cannot disambiguate all of SO(3) (out-of-plane rotation causes self-occlusion and depth ambiguity) — which forces Hop 2.

### Hop 2 — 3D keypoints, or depth-from-2D
**Genuine but this is where you must choose, and it changes the project's identity.** Two sub-paths:
- *3D keypoints* (à la KeypointNet / BKinD-3D) buys you honest SO(3) but classically needs **multi-view or known pose** — which reintroduces the supervision you are proud to avoid. The open question "can 3D keypoints emerge from *prediction* + single-view video alone?" is genuinely open and would be a strong result.
- *Depth-from-2D* keeps single-view but makes the operator act on inferred depth — closer to how biological vision handles this, which fits your cognitive-science framing.

You marked this "idk yet, not planned" — that is the correct posture. **Do not commit here until Hop 1 tells you whether 2D genuinely breaks.** Risk: this is the hop most likely to quietly turn the project into a 3D-vision engineering problem and away from the cognitive/representational question. Guard against that.

### Hop 3 — Koopman framing
**Borrow the vocabulary; do not claim to extend the method.** As established in §1.D, classical Koopman achieves *global* linearity by **lifting to a higher-dimensional observable space**, trained by reconstruction. Your keypoints are a *low-dimensional* chart, and you achieve linearity because the *generator itself* is low-dimensional (rigid motion), not because you lifted. So:
- **Honest framing:** "our keypoints are learned observables in which the transfer operator restricted to the object's orbit is finite-dimensional and linear" — this is Koopman-*flavored* and defensible.
- **Overclaim to avoid:** calling `W` "the Koopman operator." The Koopman operator is infinite-dimensional and acts on *functions*; your `W` is a finite matrix on coordinates. A reviewer who knows Koopman will pounce.
- **Where Koopman genuinely helps:** when you hit non-rigid objects (Hop 5), the "one global `W`" assumption fails, and Koopman's spectral/eigenfunction machinery (Lusch, Takeishi) becomes the *right* tool for finding a linearizing lift. So Koopman is best positioned as **the tool you graduate to when rigidity breaks**, not the frame for the current rigid result. Verdict: a real methodological direction for Phase 2, a slogan (and a trap) for Phase 1.
- **A live neighbour to cite (not a scoop):** Ruiz-Morales et al. 2026 (arXiv:2511.09783) train a **near-identity linear predictor** in a JEPA and show it forces the encoder to learn Koopman-invariant coordinates — the *same mechanism* you rely on (linear-predictability pressure selects structured latents), on a *different substrate* (multivariate time-series, no images, no spatial keypoints, no irrep-selection claim). It confirms the mechanism is real and publishable; it does not occupy your cell. Cite it to show the mechanism has independent support, and distinguish on substrate (spatial keypoints) and on the irrep/chart-selection question (§1.H).

### Hop 4 — symmetry / generator discovery
**The most ambitious hop, and the one with a genuine well-posedness question you have correctly identified.** Your worry — "keypoints won't necessarily have a full representation of the input, so discovering a symmetry group from them may not make sense" — is exactly right and is the crux. Existing symmetry-discovery methods (LieGAN, L-conv, Homomorphism-AE) all lean on a **complete/generative** representation: LieGAN must generate the data distribution; Homomorphism-AE reconstructs. **A non-reconstructive partial keypoint code does not obviously determine a group action** — two different global symmetries can act identically on a sparse set of keypoints, so the group is underdetermined by the keypoints alone.

Two ways this becomes well-posed (both are real research contributions, not increments):
1. **Discover the generator, not the group.** You are already doing a weak form of this — `W` *is* an infinitesimal generator (log of the group element). "Symmetry discovery" for you could mean *identifying the Lie-algebra element* your operator corresponds to and testing closure/commutation, which **is** well-posed from keypoints because you only need the operator, not the full representation. This connects directly to LieGAN's lineage (Yu lab) without needing generative modeling.
2. **Add enough keypoints to make the action faithful.** If the keypoints span the object richly enough that the group action on them is faithful (injective), the group *is* recoverable. This is a concrete, testable condition — and deciding how many keypoints make a rigid/articulated object's symmetry faithful is itself a nice result.

Verdict: genuine, high-risk, high-reward. **Frame it as generator/Lie-algebra discovery (path 1), which is well-posed, rather than group discovery, which is not — from keypoints.**

### Hop 5 — non-rigid objects → faces, with *semantic* keypoints
**This is the thesis-defining hop, and it is where the project stops being narrow.** Everything before it is a controlled proof-of-concept; this is the payoff that makes it a *cognitive-science* contribution rather than a computer-vision curiosity. Two hard problems fuse here:
- **Non-rigid → the single global `W` must break into something structured.** Faces deform (expression, identity, viewpoint). The right object is probably a *low-rank or block-structured* operator, or Paul's *locally-linear* (curvature-regularized) patches, or a small dictionary of operators (viewpoint vs. expression). This is where Paul's tools and yours merge.
- **Semantic keypoints.** Your requirement that keypoints "make semantic sense" (eye, mouth corner, nose) is **not guaranteed by predictability alone** — a predictable chart need not be a semantic one. The open question "does operator-predictability pressure *also* push keypoints toward semantically meaningful parts, or do you need an extra inductive bias?" is a genuine and *falsifiable* cognitive-science hypothesis. If yes, that is a strong claim about why brains might use parts. If no, characterizing what extra pressure is needed is still a result.

Nearest prior: the unsupervised-landmark line (A) gets semantic face keypoints *from reconstruction*; you would be claiming to get them *from predictability*. That contrast is the spine of a thesis.

### Trajectory summary

| Hop | Real contribution? | Nearest prior | Key risk |
|---|---|---|---|
| 1. SO(3) | Yes — tests compositionality claim | KeypointNet (supervised), Paul (latent) | 2D single-view underdetermines SO(3) |
| 2. 3D / depth | Yes, but identity-changing | KeypointNet, BKinD-3D | drifts into 3D-vision engineering |
| 3. Koopman | Frame only in Phase 1; tool in Phase 2 | Lusch, Takeishi, E2C | overclaiming `W` = Koopman operator |
| 4. Symmetry | Yes — reframe as *generator* discovery | LieGAN, L-conv (Yu lab) | ill-posed from partial keypoints unless reframed |
| 5. Non-rigid faces + semantics | **Thesis-defining** | unsup. landmarks (recon-based) | semantics not guaranteed by predictability |

---

## 4. Verdict: narrow exploration, or a PhD-sized programme?

**A PhD-sized programme — provided you sequence it as one clean result plus a principled ladder, not as a single paper.**

Here is the honest reasoning, both directions.

**Why it could look narrow.** Right now the demonstrated result is: *on one rigid object under one 1-parameter rotation, unsupervised keypoints emerge in which a single linear operator predicts motion.* Taken alone, that is a workshop paper, not a thesis — and it is close to (a) the ToI demo's rotation phase and (b) the dynamics half of Minderer '19. If the project stopped at rigid yaw, "narrow" would be a fair criticism.

**Why it is actually a programme.** The thesis is not the rigid result — it is the **hypothesis that predictability-of-transformation is a sufficient learning principle for interpretable, eventually-semantic, part-based representations**, tested along a ladder of increasing generator complexity (SO(2) → SO(3) → non-rigid → faces) and connected to a formal theory (ToI) and a sibling method (Paul). Three things make it thesis-sized rather than a single result:

1. **A falsifiable central claim with a clear failure point.** "Operator-predictability alone yields interpretable/semantic keypoints without reconstruction" is a real hypothesis that could *fail* at Hop 5 (semantics may need extra bias). A thesis that could genuinely come out either way is a real thesis.
2. **Distinct, non-trivial sub-results along the ladder.** Compositional SO(3) operators (Hop 1), emergence of 3D/depth from prediction (Hop 2), generator/Lie-algebra discovery from keypoints (Hop 4), and structured/local operators for deformation (Hop 5) are each publishable and each answers a different question. That is 3–4 papers, which is a thesis.
3. **It occupies an empty cell against a theory.** The positioning grid shows the design point is unoccupied, and ToI gives it a principled reason to exist. "Unoccupied *and* theory-motivated" is the definition of a programme rather than a gap-filling exercise.

**Is the expected improvement "much bigger" than existing methods?** Be careful how you claim this. You will **not** win on standard landmark benchmarks (keypoint localization accuracy) — reconstruction-based methods are heavily tuned for that and you are not optimizing it. Your improvement is on a **different axis that you should define and own**: *operator simplicity / compositional predictability* — "our keypoints make the transformation linear and compositional over the full orbit; theirs do not, because they were never asked to." Design the metric (operator linearity, k-step compositional error, generator recovery error like your ‖A−R(6°)‖≈0.03) and you win by construction on the thing you actually care about. Do **not** frame it as "better keypoints"; frame it as "keypoints that are better *coordinates for the dynamics*."

**Concrete recommendation (sequencing).**
1. **Now → ECCV workshop (~June 13):** write the 4-pager on the **rigid SO(2) result you already have** (§2.3). One figure, one clean claim, honest limitations. Do not wait for SO(3) or the collapse fixes — the result is submission-ready and the workshop rewards a sharp single result.
2. **Fix representation quality in parallel** (your min-distance hinge, overlap penalty, channel-init, 64→128 resolution) — needed for clean figures, but not a blocker for the *operator* claim which is already solid.
3. **Paper 2 (SO(3) + compositionality):** the first real thesis chapter. Non-commutative operator composition is the natural "bigger" result.
4. **Choose the branch after SO(3):** 3D-vs-depth (Hop 2) or straight to non-rigid (Hop 5). Let Hop 1's results decide.
5. **Keep Koopman as a Phase-2 tool and symmetry as generator-discovery** (§3) — both real, both later, both currently traps if claimed early.

**Bottom line:** the idea is not narrow. It is a well-posed, theory-anchored, falsifiable programme sitting in a genuinely empty region of the design space, with a submission-ready first result already in hand and a clear ladder of 3–4 distinct contributions above it. The two risks to manage are (a) not letting Hop 2 turn it into a 3D-vision engineering project, and (b) not overclaiming the Koopman/symmetry connections before they are well-posed. Manage those and this is a PhD.

---

## Appendix — reference clusters

Full verified list with per-paper "what they assume" notes in `positioning_refs.csv` (34 references). Clusters: **A** unsupervised landmarks (6) · **B** keypoints for dynamics/control (6) · **C** equivariance by construction (5) · **D** Koopman/linearizing operators (4) · **E** symmetry/generator discovery (5) · **F** lab-adjacent anchors: ToI, Paul, van der Pol (3) · **G** emergent localized units — "why keypoints at all": Bau/Network Dissection, Locatello/Slot Attention, Caron/DINO (3) · **H** irrep / frequency selection: Koyama/Neural Fourier Transform (1) · **I** estimator / methods notes: Fang/AlphaPose soft-argmax bias (1).

*Additions (this revision).* Seven references were added after a second-pass review that incorporated the project's theory document and an external critique:
- **Cluster G** (emergent localized units) motivates the "why keypoints at all" question — units functionally equivalent to keypoints emerge implicitly across CNNs (Bau '17), object-centric models (Locatello '20), and self-supervised ViTs (Caron '21), yet none of these papers explains the emergence. This work supplies the functional account (operator legibility).
- **Cluster H** (Koyama '23, Neural Fourier Transform) is the closest prior art to the sharpened irrep-selection claim and must be cited and distinguished (it settles harmonic degeneracy for abstract latents given a fixed kernel; it does not address physical k=1 pinning or the keypoint-vs-latent comparison).
- **Ruiz-Morales '26** (arXiv:2511.09783, added to cluster D) is a live neighbour: a near-identity linear predictor forcing Koopman-invariant latents — same mechanism, different substrate (time-series, no keypoints). Pedigree, not scoop.
- **Fang '22 / AlphaPose** (cluster I) documents soft-argmax estimator bias, a candidate mechanistic cause of the operator's sub-unity singular values.
- **Hedlin '24** (arXiv:2312.00065, cluster A) is a field-maturity marker (modern keypoint correspondence now comes from frozen diffusion/DINO features) — flagged for ID re-verification, see note below.

*Retrieval note: references were verified by direct arXiv ID metadata fetch and Crossref lookup. All references except Hedlin '24 were confirmed this session (title, authors, year). Ruiz-Morales, Koyama/NFT, Fang/AlphaPose, Bau, Locatello, and Caron were each fetched by ID and matched. **Hedlin '24 (arXiv:2312.00065) could not be re-verified before saving because the arXiv API was timing out; confirm this ID before citing.** The earlier OpenAlex 503 outage still applies to keyword search; all verification used ID-level fetches.*
