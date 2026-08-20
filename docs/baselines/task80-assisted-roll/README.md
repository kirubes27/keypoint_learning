# Task-80 assisted world-Z roll baseline

This directory defines the repository's curated pre-OCR, pre-descriptor, pre-sliding/wobble baseline. It binds the scientific recipe to exact source, configuration, checkpoint, dataset-index, input, and forward-output hashes.

## What the model does

```text
frame t ── shared CNN ── soft-argmax ── 10 keypoints ── shared affine ── predicted keypoints at t+1
frame t+1 ─ shared CNN ─ soft-argmax ── 10 observed keypoints
```

Both frames use the same CNN weights. There is no decoder or image-reconstruction objective. The operator applies one shared 2D affine transform to all keypoints.

## Exact active recipe

- Source base: `788ee14a9fdf8aa396a52507f7aa1d84416def60`.
- Object: `engineers_hammer_vray`.
- Data motion: world-Z roll, frames separated by 3 source steps = +6 degrees, cyclic pairs.
- Input: 512 x 512 RGB, no crop, ImageNet normalization.
- Extractor: four convolutional layers, base width 32, reflect padding.
- Head: legacy 64 x 64 heatmaps, 10 channels, temperature-1 soft-argmax.
- Operator: shared affine forward operator plus shared affine inverse operator.
- Training: 1,000 epochs, batch size 16, Adam learning rate `1e-4`, weight decay `1e-5`, seed 42.

| Term | Weight | Active? |
|---|---:|---|
| prediction | implicit primary term | yes |
| smoothness | 0.001 | yes |
| displacement | 0.1 | yes |
| entropy | 0.01 | yes |
| inverse prediction | 0.5 | yes |
| cycle consistency | 0.5 | yes |
| action classification | 0.0 | no |
| localization | 0.0 | no |

Two legacy fields are easy to misread:

- `learn_inverse_operator` is `false`, but the inverse operator is active because `lambda_inv` and `lambda_cycle` are nonzero. The saved config records `learn_inverse_operator_effective: true`.
- `num_action_classes` is 2, but no action classifier is instantiated because `lambda_act` is zero.

The config predates the later `heatmap_res` and `true_quarter_res` options. Their absence is part of the recipe and means the legacy 64-resolution path; it must not be interpreted using later experimental defaults.

## What is established—and what is not

The checkpoint is source-compatible with this baseline: it strict-loads with no missing or unexpected keys and instantiates 246,678 trainable parameters, including 6 forward-affine and 6 inverse-affine parameters.

The scientific conclusion is deliberately narrow. This is an assisted, provisional, single-seed result. The recorded representation screen found 5 of 10 keypoints clean, 4 sliding, 1 dead/off-object, and one near-duplicate pair. Therefore:

- operator learning under this stated setup is reproducible;
- stable material attachment is **not** established;
- held-out object or seed generalization is **not** established;
- the later OCR-ZNCC, descriptor, sliding, and wobble code is **not** part of this baseline.

## Files

- `config.json`: byte-for-byte copy of the bound training config.
- `MANIFEST.json`: hashes, source contract, expected forward outputs, and limitations.
- `../../../tests/test_task80_baseline_contract.py`: static and optional external-artifact checks.

The checkpoint and dataset are intentionally not committed to Git. `MANIFEST.json` binds the currently verified copies, but their durable publication/storage locators are still pending. Until those locators exist, describe the baseline as locally reproduced—not independently downloadable.

## Verification

Static contract only:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Bound checkpoint/data check:

```bash
TASK80_CHECKPOINT=/absolute/path/to/best_model.pt \
TASK80_DATA_ROOT=/absolute/path/to/_tdw_world_z_roll_base_panel_512_v2 \
TASK80_REQUIRE_EXACT_OUTPUT_HASHES=1 \
python -m unittest tests.test_task80_baseline_contract
```

The exact-hash mode is the pinned local reference gate. The test also performs tolerance-based comparisons of keypoint coordinates so that any future portability decision is explicit rather than silently accepting different behavior.
