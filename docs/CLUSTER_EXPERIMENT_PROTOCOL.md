# Cluster Experiment Protocol (Frozen 2026-07-05)

This is the single operating procedure for every future cluster experiment in
this project. Do not invent experiment-specific transfer or review workflows.

## Non-negotiable rules

1. Cluster terminal output is for job health only. Scientific results are never
   copied from the terminal or interpreted from screenshots.
2. Every experiment ends with a complete artifact copy on the Mac under
   `/Users/kirubeso.r/Documents/PhD/cluster_downloads/<experiment_id>/`.
3. The copy must contain outputs, scheduler logs, an archive checksum and the
   machine-readable summary.
4. The next experiment cannot start until the current artifact has been
   inspected locally and a result report has been saved in `docs/`.
5. Any change to `model.py`, `train.py`, losses, datasets or target generation
   must be recorded in `docs/CORE_FILE_CHANGELOG.md` in the same commit.
6. The exact branch, commit, run count, supervision, resolution and any
   non-default architecture flag must be recorded before submission.

## Stage 1 — Define

- Write the semantic lock: hypotheses, must/must-not statements, evidence and
  pass/fail interpretations.
- Declare the experiment ID, expected run count, cluster output path and Mac
  artifact path.
- Decide whether a core-file change is required. If yes, document its default
  and checkpoint implications before running.

**Gate:** no implementation or run until the semantic interpretation is
unambiguous.

## Stage 2 — Implement and smoke-test locally

- Implement on `fitted-operator-diagnostics-20260704`.
- Run unit tests and a minimal smoke test that exercises the critical semantic
  distinction.
- Confirm that defaults used by ordinary training have not silently changed.
- Estimate runtime. Any local run expected to exceed one hour goes to the
  cluster after informing the user.

**Gate:** tests pass and the smoke output has the intended meaning.

## Stage 3 — Freeze the run

- Commit intentionally and push the diagnostic branch.
- Record the commit hash and complete launch command.
- Do not edit experiment code while its jobs are running. A correction requires
  a new commit and a new experiment ID.

**Gate:** cluster checkout equals the frozen commit.

## Stage 4 — Run on the cluster

- Pull the frozen branch and submit the recorded job script.
- Use `squeue`/`sacct` only to establish running/completed/failed state.
- Outputs and logs remain on the cluster until collection succeeds.
- Do not print full JSON results for manual transfer.

**Gate:** expected number of jobs complete with exit code zero and the expected
number of result files exist.

## Stage 5 — Collect on the Mac

- First leave the cluster session with `exit`; the prompt must be the Mac.
- Run the experiment's checked-in fetch script once.
- The script validates completeness, downloads results and logs, records a
  SHA-256 checksum and generates the local summary.
- Keep the remote copy; collection is a copy, not a move.

**Gate:** a complete local artifact exists under `cluster_downloads/` and its
summary can be read without cluster access.

## Stage 6 — Review and report

- Inspect the local raw metrics, not terminal excerpts.
- Apply only the preregistered interpretation first; label any additional
  analysis post-hoc.
- Record sample unit, `n`, descriptive/inferential status and uncertainty
  definition where applicable.
- Save the conclusion and exact artifact path in a dated Markdown report under
  `docs/`, then commit it.

**Gate:** only after this report may the next experiment be selected.

## Current experiment registry

| Field | Value |
|---|---|
| Experiment ID | `stage_a_attribution_20260705` |
| Branch | `fitted-operator-diagnostics-20260704` |
| Cluster output | `keypoint_net/diagnostics/outputs/final_material_keypoints/stage_a_attribution/` |
| Expected runs | 6 |
| Mac collector | `cluster/fetch_stage_a_attribution_to_mac.sh` |
| Architecture | Standard 64x64 path; `true_quarter_res=False` |
| Current state | Collected and reviewed locally; result report saved in `docs/STAGE_A_ATTRIBUTION_RESULTS_2026-07-05.md` |

## Current collection command

This command is run from the **Mac**, not from inside SSH:

```bash
cd "/Users/kirubeso.r/Documents/PhD/keypoint_learning_fitted_operator"
./cluster/fetch_stage_a_attribution_to_mac.sh
```
