# TDW Phase-A Starter Pack (dataset generation)

This folder is meant to be a minimal, reproducible scaffold for generating Phase-A data:
- rigid objects
- fixed lighting/background/scale
- smooth pose changes with small increments
- yaw-only first (optionally add pitch later)

## Files you should edit
- `models_4.txt` or `models_12.txt` — one TDW model name per line.
  These are *templates*. Replace with the model names you actually have in TDW.
- `run_yaw_only.sh` — your main command.

## Quick start
1) (Optional) clean your model list:
   ```bash
   python check_models_file.py models_4.txt
   ```
   Then use `models_4_cleaned.txt`.

2) Generate yaw-only dataset:
   ```bash
   ./run_yaw_only.sh
   ```

## What outputs you should expect
- `phase_a_yaw_only/`
  - per-object image sequences (png)
  - `meta.jsonl` (or similar metadata file; depends on script)
  - train/test split directories or metadata fields (depends on script)

## Phase-A “does this obey the action plan?”
It does **if** you run it with:
- `yaw_step` ~ 1–2 degrees
- **only one** rotation axis (set pitch/roll lists to 0)
- fixed `dist` (no scale changes)
- fixed lighting/background settings (script defaults)
- sequences saved in *ordered* yaw steps so adjacent frames are smooth

The one thing you should always sanity-check after the first run:
- **metadata ↔ image indexing** matches (some TDW capture pipelines start counting after warmup frames).
  Just spot-check: does the metadata for t=0 correspond to the first saved image?

If you want, send me one `meta.jsonl` line + the first 2 filenames and I’ll tell you if it’s aligned.
