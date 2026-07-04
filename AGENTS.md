# AGENTS.md

## General, Cross-Project Delivery Guardrails

These rules are intentionally **not project-specific**. Apply them to any future technical task.

1. **Semantic lock before implementation**
   - Convert stakeholder language into explicit testable statements before coding.
   - Example structure: “must be true”, “must not happen”, “evidence required”.
   - Do not proceed to full runs until this lock is written and checked.

2. **Meaning-first verification (not consistency-first)**
   - Passing scripts, matching counts, and valid JSON are insufficient.
   - First verify that outputs satisfy the requested semantics.
   - If semantics and implementation disagree, semantics win.

3. **Transform tasks require axis/geometry proof**
   - For rotation/pose/camera requests, verify the intended axis in output space.
   - Use both:
     - explicit code-path check (which commands/axes are applied), and
     - output-level check (visual or metric evidence of the intended motion).
   - Never infer correctness from variable names alone.

4. **Full-run gate policy**
   - Before expensive/full generation, require a smoke gate that proves the critical semantic requirement.
   - If any critical semantic check fails, stop and report; do not proceed.

5. **Approval gate for ambiguous semantics**
   - If the user request can reasonably map to multiple technical interpretations, pause and ask a single disambiguation question before coding.
   - Do not assume interpretation in silence.

6. **Report uncertainty explicitly**
   - When certainty is below high confidence, say exactly what is unverified.
   - Do not present plausible outputs as confirmed outputs.

7. **Post-mortem discipline after misses**
   - If a mismatch is reported, document:
     - what assumption was wrong,
     - why checks failed to catch it,
     - what new gate prevents recurrence.
   - Apply the new gate as default going forward.

8. **No “structure-only success” claims**
   - Never claim success based only on:
     - frame counts,
     - file integrity,
     - background consistency,
     - absence of runtime errors.
   - Include semantic evidence in every success claim.

9. **Concise accountability standard**
   - Explanations must be plain-language, direct, and non-defensive.
   - Include concrete correction plan with pass/fail criteria.

10. **Statistical reporting guardrail (mandatory for plots/tables)**
   - For any error bars or uncertainty bands, explicitly state:
     - quantity shown (`std`, `sem`, or `CI`),
     - exact computation (e.g., `sample std with ddof=1`, `sem = std/sqrt(n)`),
     - sample unit and `n` (e.g., frames, runs, objects, seeds),
     - whether the result is descriptive or inferential.
   - Never use ambiguous labels like `+/- error`; always name the statistic.
   - Never present `std` as `sem` (or vice versa). If requested statistic differs from current implementation, update code before reporting.
   - If samples are correlated (time series, overlapping windows, same sequence), do not claim population-level inference from naive error bars; either:
     - label them as descriptive variability, or
     - use an appropriate method (e.g., block bootstrap / hierarchical aggregation) and state it.
   - Store the error-bar definition in artifacts (JSON/metadata) so plots are auditable.

11. **Statistical test discipline**
   - For every hypothesis test, report:
     - test name, null hypothesis, tail direction, alpha,
     - exact sample unit and independence assumption,
     - multiplicity correction if multiple comparisons are made,
     - effect size and confidence interval (not p-value alone).
   - If assumptions are doubtful, switch to robust/nonparametric alternatives and state why.
   - Do not make causal or generalization claims from single-run, single-object, or non-independent samples.
