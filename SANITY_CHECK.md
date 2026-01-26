# Quick sanity checks (Phase A dataset)

## 1) Confirm your TDW versions match
Run:
```bash
python -c "import tdw; print('tdw python:', tdw.__version__)"
```
Make sure your Unity build version matches (shown in the build download name / log).

## 2) Smoke test: generate 5 frames for 1 object
```bash
python create_dataset.py \
  --out_dir ./_smoke \
  --models_file ./models_example_4.txt \
  --mode rotate_object \
  --yaw_min -2 --yaw_max 2 --yaw_step 1 \
  --pitch_list 0 \
  --img_size 256 \
  --dist 1.5
```

## 3) Are lighting + scale fixed?
- **Lighting**: this script uses `create_empty_room(...)` and does **not** randomize lights/HDRI. For strict reproducibility across machines/versions, consider adding explicit light intensity + skybox settings later.
- **Scale**: the script **does not** call `scale_object`. So object *mesh sizes* differ across object identities. That’s OK for *per-object* training; for multi-object training, you may want optional normalization later.

## 4) “Fixed camera” vs “orbit camera”
- `--mode rotate_object`: camera stays fixed; object rotates (simplest).
- `--mode orbit_camera`: camera moves on a sphere; object is fixed (equivalent images, different interpretation).
