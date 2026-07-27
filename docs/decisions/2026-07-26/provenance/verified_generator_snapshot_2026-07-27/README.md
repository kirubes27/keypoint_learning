# Dataset generator — SELF-CONTAINED, VERIFIED (read this first)

This ONE folder contains everything about the generator of the working dataset
`tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2`
(6 objects x 180 frames, world-Z roll 0..358 deg in 2-deg steps, 512x512, masks).

**If you only remember one thing:** the 4 `.py` files at the top level of this folder,
plus `generate_tdw_world_z_roll_dataset_post_update.py`, ARE the verified generator.
Everything else here is evidence and history.

The sibling folder `../tdw_phase_a_starter_recovered/` is now FULLY REDUNDANT —
a verbatim copy lives in `june2_recovery_archive/` here. Safe to delete the sibling.
Likewise `tdw_phase_a_starter /generate_tdw_world_z_roll_dataset.py` (top level of the
starter dir) is an UNFAITHFUL rewrite — do not use it; kept there only to avoid
breaking anything that references it.

## Timeline (why this folder exists)

| Date | Event |
|---|---|
| 2026-05-30 | Original generator written/run. Python auto-cached its bytecode in `tdw_phase_a_starter /__pycache__/` (nobody planned this — it saved the project). |
| 2026-06-02 | Original `.py` sources accidentally deleted. Same-day recovery attempt wrote `../tdw_phase_a_starter_recovered/` (from patches/memory) and REGENERATED the dataset at 17:15. All June-2026 training runs use that regenerated data. |
| 2026-07-10 | Discovery that the June-2 recovery was partly wrong (see below). GPT-5.5 (Codex, xhigh) reconstructed exact source from the bytecode fossils; Claude (Fable 5) orchestrated and independently verified. This folder created. |

## What was wrong with the June-2 recovery (../tdw_phase_a_starter_recovered/)

- `compare_hammer_rotation_operators.py` — STALE: missing the `tdw_world_z_roll`
  operator entirely (crashes on this dataset's operator).
- `create_dataset_affine_xy_scale_rotation.py` — NEVER RECOVERED (no source anywhere;
  only bytecode survived).
- `create_true_yaw_from_base_panel_sanity.py`, `generate_tdw_world_z_roll_dataset.py`
  — these two were actually faithful (verified 2026-07-10 as bytecode-identical).
- Also confusing: it held multiple variants (`_exact_candidate`, `_post_update`)
  with no verdict on which was real.

## What is in THIS folder

| Item | What it is | Status |
|---|---|---|
| `create_dataset_affine_xy_scale_rotation.py` | Base module (controller, capture, masks, scene). Was fully lost; rebuilt from disassembly. | VERIFIED bytecode-identical |
| `create_true_yaw_from_base_panel_sanity.py` | ObjectSpec/RenderConfig, scene setup, helpers | VERIFIED bytecode-identical |
| `compare_hammer_rotation_operators.py` | `_apply_tdw_operator` incl. `tdw_world_z_roll` | VERIFIED bytecode-identical |
| `generate_tdw_world_z_roll_dataset.py` | Full-dataset driver, May-30 state | VERIFIED bytecode-identical |
| `generate_tdw_world_z_roll_dataset_post_update.py` | **The driver that actually produced the on-disk dataset** (same rendering; richer pair-index/metadata files matching the dataset's actual `indices/`) | schema-matched to dataset; rendering path pixel-verified |
| `scale_probe/` | The probe scripts that determined the approved per-object scales (`scale_summary.json` values). Needed if you ever add objects / rebuild from scratch. | from June-2 recovery |
| `bytecode_fossils/` | The 4 surviving `.cpython-310.pyc` files = ground truth. Version-locked to Python 3.10. DO NOT DELETE. | original |
| `verification_evidence/` | Sanity images (real vs replica vs diff x20), pixel-diff JSONs, the verification scripts, Codex's report | 2026-07-10 |
| `june2_recovery_archive/` | Verbatim copy of the entire June-2 recovery folder (incl. the patch .txt files = provenance of the post_update changes). History only — top-level files supersede it. | archive |

## How the verification worked (2026-07-10)

1. **Bytecode equivalence (exact):** each top-level module compiles under cpython 3.10
   to code instruction-for-instruction identical to `bytecode_fossils/` — checked
   independently with `verification_evidence/verify_equivalence.py`. ALL FOUR IDENTICAL.
2. **Pixel replication (sampled):** 3 objects (hammer, kettle, toy monkey) x 3 angles
   (0/60/180 deg) rendered with fossil bytecode AND with the reconstructed source,
   compared to the real dataset frames (0/30/90): all 9 masks byte-identical in both
   runs; RGB identical except <=7.3% of pixels on glossy surfaces at <=11/255 channel
   difference — GPU render nondeterminism (the SAME code differs run-to-run on the
   same pixel class). See `verification_evidence/sanity_*_vs_dataset.png`.

Honest caveats: pixel check sampled 3/6 objects, 3/180 angles; full 1080-frame
regeneration was deliberately not rerun (user decision, 2026-07-10).

## To regenerate the dataset

Python 3.10 (`conda activate phd`), tdw==1.13.0 (build at `~/tdw_build/TDW/TDW.app`).
Run `generate_tdw_world_z_roll_dataset_post_update.py` with the 4 top-level modules on
the import path, from a folder containing `models_affine_final_6.txt`; approved scales
are in the dataset's own `scale_summary.json` (or re-derive with `scale_probe/`).
Note: some original input folders (e.g. the affine source-index panel) were partially
lost; the dataset's own `dataset_index.json` + `meta.jsonl` record every parameter
needed to reconstruct the specs (see `verification_evidence/render_and_compare.py`
for a worked example of building ObjectSpec/RenderConfig from the dataset itself).
