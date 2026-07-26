import copy
import io
import itertools
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = KEYPOINT_ROOT.parent
DATA_ROOT = REPOSITORY_ROOT / "_tdw_world_z_roll_base_panel_512_v2"
sys.path.insert(0, str(KEYPOINT_ROOT))

from model import KeypointExtractor, spatial_softmax  # noqa: E402
from diagnostics import decision23_diagnostic_head as d23  # noqa: E402
from diagnostics.decision23_diagnostic_head import (  # noqa: E402
    ARMS,
    EXPECTED_CENTER_X,
    EXPECTED_CENTER_Y,
    EXPECTED_OBJECT,
    EXPECTED_PROBE_CHECKPOINT_SHA256,
    EXPECTED_PROBE_CONFIG_SHA256,
    EXPECTED_SPLIT_SHA256,
    EXPECTED_TRAINVAL_CONTENT_MANIFEST_SHA256,
    FULL_RECIPE,
    SMOKE_RECIPE,
    Decision23Extractor,
    PhaseSplit,
    SharedSpatialReadout,
    _claim_path,
    _complete_finalization_event,
    _prepare_finalization_event,
    assert_finite_training_state,
    assert_json_finite,
    assert_probe_checkpoint,
    arm_status,
    assert_expected_upstream_gradients,
    assert_recipe_scope,
    assert_semantic_scope,
    assert_split_access,
    coordinate_grid,
    evaluate,
    evaluation_epoch_due,
    gradient_audit,
    instrument_pass,
    load_scoped_problem,
    build_training_datasets,
    make_scoped_dataset,
    validate_resume_records,
    write_json,
)
from diagnostics.stage_a_supervised_control import (  # noqa: E402
    load_phase_split,
    restore_checkpoint,
    save_checkpoint,
    seed_everything,
)


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _assert_numpy_rng_equal(left, right) -> None:
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _metric_pair(median: float, p90: float, on_mask: float) -> dict:
    condition = {
        "median_of_channel_medians_cells64": median,
        "p90_error_cells64": p90,
        "on_mask_fraction": on_mask,
    }
    return {
        "unaugmented": dict(condition),
        "fixed_augmented": dict(condition),
    }


def _write_synthetic_test_artifacts(
    run_dir: Path,
    record: dict,
    *,
    metrics_override: dict | None = None,
) -> tuple[dict, Path, Path]:
    delta = 0.25 * d23.CELL64_NORM
    prediction = np.array(
        [[[delta, 0.0]], [[delta, 0.0]]], dtype=np.float64
    )
    target = np.zeros_like(prediction)
    frame = np.array([120, 121], dtype=np.int64)
    in_range = np.ones((2, 1), dtype=bool)
    on_mask = np.ones((2, 1), dtype=bool)
    error = np.full((2, 1), 0.25, dtype=np.float64)
    condition = {
        "median_error_cells64": 0.25,
        "median_of_channel_medians_cells64": 0.25,
        "channel_median_error_cells64": [0.25],
        "p90_error_cells64": 0.25,
        "on_mask_fraction": 1.0,
        "in_range_fraction": 1.0,
        "out_of_range_count": 0,
        "n_frames": 2,
        "n_channel_frame_pairs": 2,
        "sample_unit": "synthetic test frame x channel",
        "uncertainty": "descriptive synthetic fixture",
    }
    metrics = {
        "selection_score_cells64": 0.25,
        "selection_rule": "synthetic max of two conditions",
        "unaugmented": dict(condition),
        "fixed_augmented": dict(condition),
        "arm": record["config"]["arm"],
        "seed": int(record["config"]["seed"]),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "config_sha256": record["config_sha256"],
        "instrument_pass": True,
    }
    if metrics_override:
        metrics.update(metrics_override)
    metrics_path = run_dir / "test_metrics.json"
    predictions_path = run_dir / "test_predictions.npz"
    write_json(metrics_path, metrics)
    arrays = {}
    for prefix in ("plain", "augmented"):
        arrays.update(
            {
                f"{prefix}_prediction": prediction,
                f"{prefix}_target": target,
                f"{prefix}_frame": frame,
                f"{prefix}_in_range": in_range,
                f"{prefix}_on_mask": on_mask,
                f"{prefix}_error_cells64": error,
            }
        )
    np.savez_compressed(predictions_path, **arrays)
    return metrics, metrics_path, predictions_path


def test_cross_job_runtime_match_ignores_gpu_hardware_but_not_software() -> None:
    first = {
        "python": "3.11.0",
        "torch": "2.7.0",
        "numpy": "2.0.0",
        "pillow": "11.0.0",
        "cuda_runtime": "12.6",
        "cudnn_version": 90501,
        "cuda_device_index": 0,
        "cuda_device_name": "NVIDIA H100 80GB HBM3",
        "cuda_device_capability": [9, 0],
        "cuda_device_total_memory_bytes": 80_000_000_000,
        "deterministic_algorithms_enabled": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    second = {
        **first,
        "cuda_device_index": 1,
        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
        "cuda_device_capability": [8, 0],
        "cuda_device_total_memory_bytes": 79_000_000_000,
    }
    assert d23.runtime_software_identity(first) == d23.runtime_software_identity(
        second
    )
    assert d23.matched_config_value(
        {"runtime": first}, "runtime"
    ) == d23.matched_config_value({"runtime": second}, "runtime")
    incompatible = {**second, "torch": "2.7.1"}
    assert d23.runtime_software_identity(first) != d23.runtime_software_identity(
        incompatible
    )


def test_runtime_identity_is_safe_for_weights_only_checkpoint_loading() -> None:
    runtime = d23.runtime_identity(None)
    for key in ("python", "torch", "numpy", "pillow"):
        assert type(runtime[key]) is str
    if runtime["cuda_runtime"] is not None:
        assert type(runtime["cuda_runtime"]) is str
    buffer = io.BytesIO()
    torch.save({"config": {"runtime": runtime}}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    assert restored["config"]["runtime"] == runtime


def test_slurm_runfiles_are_frozen_into_immutable_provenance() -> None:
    hashes = d23.slurm_runfile_hashes()
    assert set(hashes) == {
        "d1_smoke",
        "d2_full",
        "d3_finalize",
        "extension",
        "extension_finalize",
    }
    assert all(len(value) == 64 for value in hashes.values())
    assert "slurm_runfiles_sha256" in d23._immutable_config_keys()


def test_evaluation_schedule_matches_frozen_full_and_smoke_intervals() -> None:
    assert [
        epoch
        for epoch in range(1, 51)
        if evaluation_epoch_due(epoch, FULL_RECIPE["eval_every"])
    ] == [25, 50]
    assert [
        epoch
        for epoch in range(1, 3)
        if evaluation_epoch_due(epoch, SMOKE_RECIPE["eval_every"])
    ] == [1, 2]
    with pytest.raises(ValueError):
        evaluation_epoch_due(1, 0)


def test_shared_readout_initialization_symmetry_and_active_gradients() -> None:
    height, width = 8, 8
    raw = SharedSpatialReadout("raw_linear", height, width)
    probability = SharedSpatialReadout("probability_linear", height, width)
    fixed = SharedSpatialReadout("fixed_expectation", height, width)
    independent_grid = torch.tensor(
        [
            [
                -1.0 + 2.0 * column / (width - 1)
                for row in range(height)
                for column in range(width)
            ],
            [
                -1.0 + 2.0 * row / (height - 1)
                for row in range(height)
                for column in range(width)
            ],
        ]
    )

    assert raw.decoder is not None and probability.decoder is not None
    assert sum(p.numel() for p in raw.decoder.parameters()) == 2 * 64 + 2
    assert raw.added_parameter_count == probability.added_parameter_count == 130
    production_raw = SharedSpatialReadout("raw_linear")
    production_probability = SharedSpatialReadout("probability_linear")
    assert sum(
        p.numel() for p in production_raw.parameters()
    ) == 8194
    assert sum(
        p.numel() for p in production_probability.parameters()
    ) == 8194
    assert len([module for module in raw.modules() if isinstance(module, nn.Linear)]) == 1
    assert raw.decoder.weight.shape == (2, height * width)
    assert torch.allclose(raw.decoder.weight, independent_grid / 8, atol=1e-7)
    assert torch.count_nonzero(raw.decoder.bias) == 0
    assert torch.allclose(probability.decoder.weight, independent_grid, atol=1e-7)
    assert torch.count_nonzero(probability.decoder.bias) == 0
    assert torch.equal(coordinate_grid(height, width)[:, 0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(coordinate_grid(height, width)[:, width - 1], torch.tensor([1.0, -1.0]))
    assert torch.equal(coordinate_grid(height, width)[:, width], torch.tensor([-1.0, -1.0 + 2.0 / 7.0]))
    assert torch.equal(coordinate_grid(height, width)[:, -1], torch.tensor([1.0, 1.0]))

    logits = torch.randn(2, 4, height, width)
    assert torch.equal(fixed(logits), spatial_softmax(logits, temperature=1.0))
    shifted = logits + torch.randn(2, 4, 1, 1)
    assert torch.allclose(raw(logits), raw(shifted), atol=1e-6, rtol=0)

    for readout in (raw, probability):
        for values in itertools.permutations(range(logits.shape[1])):
            permutation = torch.tensor(values)
            expected = readout(logits)[:, permutation]
            actual = readout(logits[:, permutation])
            assert torch.allclose(actual, expected, atol=1e-6, rtol=0)
        for source_channel in range(logits.shape[1]):
            perturbed = logits.clone()
            perturbed[:, source_channel] += torch.randn_like(
                perturbed[:, source_channel]
            )
            before = readout(logits)
            after = readout(perturbed)
            untouched = [
                channel
                for channel in range(logits.shape[1])
                if channel != source_channel
            ]
            assert torch.equal(before[:, untouched], after[:, untouched])

    for arm in ARMS:
        active_logits = torch.randn(2, 3, height, width, requires_grad=True)
        output = SharedSpatialReadout(arm, height, width)(active_logits)
        gradient = torch.autograd.grad(output.square().sum(), active_logits)[0]
        assert float(torch.linalg.vector_norm(gradient)) > 0.0


@pytest.mark.parametrize(
    ("dtype", "atol"),
    ((torch.float64, 1e-6), (torch.float32, 1e-5)),
)
def test_probability_arm_matches_fixed_expectation_at_initialization(
    dtype: torch.dtype,
    atol: float,
) -> None:
    logits = torch.randn(3, 5, 64, 64, dtype=dtype)
    learned = SharedSpatialReadout("probability_linear").to(dtype=dtype)
    fixed = SharedSpatialReadout("fixed_expectation").to(dtype=dtype)
    assert torch.allclose(learned(logits), fixed(logits), atol=atol, rtol=0)


def test_fixed_arm_reproduces_full_production_extractor() -> None:
    torch.manual_seed(7)
    production = KeypointExtractor(
        num_keypoints=2,
        base_channels=1,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    ).eval()
    diagnostic = Decision23Extractor(
        arm="fixed_expectation",
        num_keypoints=2,
        base_channels=1,
    ).eval()
    diagnostic.backbone.load_state_dict(production.state_dict(), strict=True)
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        production_coordinates, production_logits = production(image)
        diagnostic_coordinates, diagnostic_logits = diagnostic(image)
    assert torch.equal(diagnostic_logits, production_logits)
    assert torch.equal(diagnostic_coordinates, production_coordinates)


class _FixedPredictionExtractor(nn.Module):
    def __init__(self, prediction: torch.Tensor):
        super().__init__()
        self.register_buffer("prediction", prediction)

    def get_keypoint_coords(self, image: torch.Tensor) -> torch.Tensor:
        return self.prediction.expand(image.shape[0], -1, -1)


def test_out_of_range_is_unclipped_for_error_and_always_off_mask() -> None:
    prediction = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
    extractor = _FixedPredictionExtractor(prediction)
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        "target": torch.zeros(1, 2, 2),
        "frame": torch.tensor([7]),
        "mask": torch.ones(1, 512, 512, dtype=torch.bool),
    }
    metrics, arrays = evaluate(
        extractor,
        [batch],
        torch.device("cpu"),
        sample_unit="synthetic frame x channel",
    )
    assert metrics["out_of_range_count"] == 1
    assert metrics["in_range_fraction"] == pytest.approx(0.5)
    assert metrics["on_mask_fraction"] == pytest.approx(0.5)
    assert not arrays["on_mask"][0, 0]
    assert arrays["prediction"][0, 0, 0] == 2.0
    assert arrays["error_cells64"][0, 0] == pytest.approx(64.0)


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_predictions_json_and_training_state_abort(
    bad_value: float,
    tmp_path: Path,
) -> None:
    extractor = _FixedPredictionExtractor(
        torch.tensor([[[bad_value, 0.0], [0.0, 0.0]]])
    )
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        "target": torch.zeros(1, 2, 2),
        "frame": torch.tensor([7]),
        "mask": torch.ones(1, 512, 512, dtype=torch.bool),
    }
    with pytest.raises(FloatingPointError):
        evaluate(
            extractor,
            [batch],
            torch.device("cpu"),
            sample_unit="synthetic",
        )
    with pytest.raises(FloatingPointError):
        assert_json_finite({"nested": [1.0, bad_value]})
    output = tmp_path / "nonfinite.json"
    with pytest.raises(FloatingPointError):
        write_json(output, {"value": bad_value})
    assert not output.exists()
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([bad_value])
    with pytest.raises(FloatingPointError):
        assert_finite_training_state([parameter], require_gradients=True)


def test_gradient_gate_rejects_zero_and_nonfinite_values() -> None:
    baseline = {
        "backbone_frozen": False,
        "parameter_gradient_norms": {
            "encoder_first_conv_weight": 1.0,
            "encoder_final_conv_weight": 1.0,
            "heatmap_head_weight": 1.0,
            "decoder_weight": 1.0,
        },
        "pooled_logit_gradient_l2_norm": 1.0,
    }
    assert_expected_upstream_gradients(baseline)
    for field, value in (
        ("encoder_first_conv_weight", 0.0),
        ("encoder_final_conv_weight", float("nan")),
        ("heatmap_head_weight", float("inf")),
    ):
        bad = copy.deepcopy(baseline)
        bad["parameter_gradient_norms"][field] = value
        with pytest.raises(RuntimeError):
            assert_expected_upstream_gradients(bad)
    bad = copy.deepcopy(baseline)
    bad["pooled_logit_gradient_l2_norm"] = float("nan")
    with pytest.raises(RuntimeError):
        assert_expected_upstream_gradients(bad)


def _synthetic_problem() -> dict:
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[..., 0] = np.arange(512, dtype=np.uint8)[None, :]
    mask = np.ones((512, 512), dtype=bool)
    targets = np.zeros((180, 2, 2), dtype=np.float64)
    split = PhaseSplit(train=(0,), validation=(1,), test=(2,), sha256="synthetic")
    return {
        "split": split,
        "images": {0: image.copy(), 1: image.copy()},
        "masks": {0: mask.copy(), 1: mask.copy()},
        "targets": targets,
        "center": (255.5, 255.5),
        "loaded_frame_indices": [0, 1],
    }


def test_split_authority_and_fixed_validation_augmentation_are_enforced() -> None:
    problem = _synthetic_problem()
    first = make_scoped_dataset(
        problem,
        mode="train",
        split_name="validation",
        augment=True,
        seed=900001,
    )
    second = make_scoped_dataset(
        problem,
        mode="train",
        split_name="validation",
        augment=True,
        seed=900001,
    )
    first.set_epoch(4)
    second.set_epoch(4)
    first_item = first[0]
    second_item = second[0]
    for key in ("image", "mask", "target", "angle_deg", "translation"):
        assert torch.equal(first_item[key], second_item[key])

    with pytest.raises(RuntimeError, match="not authorized"):
        make_scoped_dataset(
            problem,
            mode="train",
            split_name="test",
            augment=False,
            seed=1,
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        assert_split_access("probe", "test")
    with pytest.raises(TypeError):
        make_scoped_dataset(
            problem,
            problem["split"].test,
            mode="train",
            split_name="train",
            augment=False,
            seed=1,
        )


def _scope_args(**updates) -> Namespace:
    values = {
        "data_root": DATA_ROOT,
        "object": "engineers_hammer_vray",
        "num_keypoints": 10,
        "base_channels": 32,
        "seed": 42,
        "target_shift": 0,
        "roll_sign": 1,
        "center_x": EXPECTED_CENTER_X,
        "center_y": EXPECTED_CENTER_Y,
        "mode": "train",
        "run_scope": "full",
    }
    values.update(updates)
    return Namespace(**values)


def _scope_problem() -> dict:
    split = load_phase_split(
        DATA_ROOT / "indices" / "split_phase_mod6.json",
        "engineers_hammer_vray",
    )
    return {
        "split": split,
        "discovered_frame_count": 180,
        "discovered_mask_count": 180,
        "loaded_frame_indices": sorted(set(split.train) | set(split.validation)),
        "loaded_content_manifest_sha256": (
            EXPECTED_TRAINVAL_CONTENT_MANIFEST_SHA256
        ),
    }


def test_semantic_scope_recipe_seed_and_hashes_fail_closed() -> None:
    problem = _scope_problem()
    assert problem["split"].sha256 == EXPECTED_SPLIT_SHA256
    assert_semantic_scope(_scope_args(), problem)
    assert_recipe_scope(
        _scope_args(
            batch_size=FULL_RECIPE["batch_size"],
            lr=FULL_RECIPE["lr"],
            weight_decay=FULL_RECIPE["weight_decay"],
            min_epochs=FULL_RECIPE["min_epochs"],
            max_epochs=FULL_RECIPE["max_epochs"],
            eval_every=FULL_RECIPE["eval_every"],
            plateau_patience=FULL_RECIPE["plateau_patience"],
            relative_improvement=FULL_RECIPE["relative_improvement"],
        )
    )

    for field, value in (
        ("seed", 41),
        ("target_shift", 1),
        ("roll_sign", -1),
        ("center_x", EXPECTED_CENTER_X + 1e-6),
        ("center_y", EXPECTED_CENTER_Y + 1e-6),
        ("object", "wrong_object"),
        ("num_keypoints", 9),
        ("base_channels", 16),
    ):
        with pytest.raises(RuntimeError):
            assert_semantic_scope(_scope_args(**{field: value}), problem)
    with pytest.raises(RuntimeError):
        assert_semantic_scope(
            _scope_args(data_root=DATA_ROOT.parent / "wrong_dataset"),
            problem,
        )
    with pytest.raises(RuntimeError):
        assert_semantic_scope(
            _scope_args(mode="probe", seed=45),
            problem,
        )
    with pytest.raises(RuntimeError):
        assert_semantic_scope(
            _scope_args(run_scope="smoke", seed=43),
            problem,
        )

    bad_split = PhaseSplit(
        train=problem["split"].train,
        validation=problem["split"].validation,
        test=problem["split"].test,
        sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="split SHA-256"):
        assert_semantic_scope(
            _scope_args(),
            {**problem, "split": bad_split},
        )
    for mutation in (
        {"discovered_frame_count": 179},
        {"discovered_mask_count": 179},
        {"loaded_frame_indices": [*problem["loaded_frame_indices"], problem["split"].test[0]]},
        {"loaded_content_manifest_sha256": "0" * 64},
    ):
        with pytest.raises(RuntimeError):
            assert_semantic_scope(_scope_args(), {**problem, **mutation})

    smoke_args = _scope_args(
        mode="train",
        run_scope="smoke",
        **SMOKE_RECIPE,
    )
    assert_recipe_scope(smoke_args)
    recipe_mutations = {
        "batch_size": 15,
        "lr": 2e-4,
        "weight_decay": 2e-5,
        "min_epochs": 1,
        "max_epochs": 3,
        "eval_every": 2,
        "plateau_patience": 2,
        "relative_improvement": 0.02,
    }
    for field, value in recipe_mutations.items():
        with pytest.raises(RuntimeError, match="recipe mismatch"):
            assert_recipe_scope(
                Namespace(**{**vars(smoke_args), field: value})
            )


def test_actual_dataset_builder_wires_frozen_validation_seed_and_epoch() -> None:
    problem = _synthetic_problem()
    access_log: list[str] = []
    train, validation_plain, validation_augmented = build_training_datasets(
        problem,
        Namespace(mode="train", seed=42),
        access_log,
    )
    assert train.seed == 42
    assert validation_plain.seed == d23.DEFAULT_VALIDATION_AUGMENT_SEED
    assert validation_augmented.seed == d23.DEFAULT_VALIDATION_AUGMENT_SEED
    assert validation_plain.epoch == validation_augmented.epoch == 0
    assert access_log == ["train", "validation", "validation"]


def test_scoped_loader_physically_opens_only_authorized_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    split = PhaseSplit(
        train=tuple(range(0, 60)),
        validation=tuple(range(60, 120)),
        test=tuple(range(120, 180)),
        sha256="synthetic",
    )
    frame_paths = [
        tmp_path / "train" / EXPECTED_OBJECT / "frames" / "a" / f"img_{i:04d}.png"
        for i in range(180)
    ]
    mask_paths = [
        tmp_path / "train" / EXPECTED_OBJECT / "masks" / "a" / f"mask_{i:04d}.png"
        for i in range(180)
    ]
    opened_rgb: list[int] = []
    opened_mask: list[int] = []

    monkeypatch.setattr(d23, "load_phase_split", lambda *_: split)
    monkeypatch.setattr(
        d23, "_dataset_paths", lambda *_: (frame_paths, mask_paths)
    )
    monkeypatch.setattr(
        d23,
        "_read_rgb",
        lambda path: (
            opened_rgb.append(int(path.stem.split("_")[-1]))
            or np.zeros((512, 512, 3), dtype=np.uint8)
        ),
    )
    monkeypatch.setattr(
        d23,
        "_read_mask",
        lambda path: (
            opened_mask.append(int(path.stem.split("_")[-1]))
            or np.ones((512, 512), dtype=bool)
        ),
    )
    monkeypatch.setattr(
        d23,
        "sha256_file",
        lambda path: f"{sum(path.as_posix().encode()) % (1 << 256):064x}",
    )
    points = np.stack(
        (
            np.linspace(220.0, 290.0, 10),
            np.linspace(230.0, 280.0, 10),
        ),
        axis=1,
    )
    monkeypatch.setattr(d23, "farthest_interior_points", lambda *_args, **_kw: points)
    base_args = Namespace(
        split_json=None,
        data_root=tmp_path,
        object=EXPECTED_OBJECT,
        center_x=EXPECTED_CENTER_X,
        center_y=EXPECTED_CENTER_Y,
        num_keypoints=10,
        target_shift=0,
        roll_sign=1,
    )

    train_problem = load_scoped_problem(base_args, mode="train")
    assert opened_rgb == list(range(120))
    assert opened_mask == list(range(120))
    assert not (set(opened_rgb) & set(split.test))
    assert len(train_problem["loaded_content_sha256"]) == 240

    opened_rgb.clear()
    opened_mask.clear()
    frozen_config = {
        "physical_frame0_targets_px": points.tolist(),
        "channel_to_physical_target": list(range(10)),
    }
    test_problem = load_scoped_problem(
        base_args,
        mode="finalize",
        frozen_config=frozen_config,
    )
    assert opened_rgb == list(split.test)
    assert opened_mask == list(split.test)
    assert not (set(opened_rgb) & (set(split.train) | set(split.validation)))
    assert len(test_problem["loaded_content_sha256"]) == 120


def test_probe_checkpoint_is_bound_to_exact_hash_and_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model.pt"
    config_path = tmp_path / "config.json"
    checkpoint.write_bytes(b"checkpoint")
    config_path.write_text(
        json.dumps(
            {
                "seed": 41,
                "object": EXPECTED_OBJECT,
                "split_sha256": EXPECTED_SPLIT_SHA256,
                "supervision": "coordinate",
            }
        )
    )

    def frozen_hash(path: Path) -> str:
        if Path(path) == checkpoint:
            return EXPECTED_PROBE_CHECKPOINT_SHA256
        if Path(path) == config_path:
            return EXPECTED_PROBE_CONFIG_SHA256
        raise AssertionError(path)

    monkeypatch.setattr(d23, "sha256_file", frozen_hash)
    assert assert_probe_checkpoint(checkpoint) == config_path
    config_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "object": EXPECTED_OBJECT,
                "split_sha256": EXPECTED_SPLIT_SHA256,
                "supervision": "coordinate",
            }
        )
    )
    with pytest.raises(RuntimeError, match="config mismatch"):
        assert_probe_checkpoint(checkpoint)


def test_instrument_and_seed_replication_rules_are_exact() -> None:
    assert instrument_pass(_metric_pair(0.50, 1.50, 0.95))
    assert not instrument_pass(_metric_pair(0.5001, 1.50, 0.95))
    assert not instrument_pass(_metric_pair(0.50, 1.5001, 0.95))
    assert not instrument_pass(_metric_pair(0.50, 1.50, 0.9499))
    assert [arm_status(count, 3) for count in range(4)] == [
        "fail",
        "fail",
        "provisional",
        "pass",
    ]
    assert arm_status(4, 5) == "pass"
    assert arm_status(3, 5) == "fail"


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
) -> None:
    order = torch.randperm(4, generator=loader_generator)
    image = torch.randn(4, 3)[order]
    target = torch.randn(4, 2)[order]
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(image), target)
    loss.backward()
    optimizer.step()


def test_checkpoint_restore_reproduces_next_two_optimizer_steps(
    tmp_path: Path,
) -> None:
    seed_everything(123)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader_generator = torch.Generator().manual_seed(456)
    _optimizer_step(model, optimizer, loader_generator)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        extractor=model,
        optimizer=optimizer,
        epoch=1,
        config={"test": "decision23"},
        best_score=1.0,
        significant_best=1.0,
        last_significant_epoch=1,
        loader_generator=loader_generator,
    )

    expected = []
    for _ in range(2):
        _optimizer_step(model, optimizer, loader_generator)
        expected.append(
            (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(optimizer.state_dict()),
                torch.get_rng_state().clone(),
                loader_generator.get_state().clone(),
            )
        )

    restored_model = nn.Linear(3, 2)
    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(), lr=1e-3, weight_decay=1e-5
    )
    restored_generator = torch.Generator().manual_seed(999)
    restore_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_generator,
        torch.device("cpu"),
    )
    for expected_state in expected:
        _optimizer_step(restored_model, restored_optimizer, restored_generator)
        _assert_nested_equal(restored_model.state_dict(), expected_state[0])
        _assert_nested_equal(restored_optimizer.state_dict(), expected_state[1])
        assert torch.equal(torch.get_rng_state(), expected_state[2])
        assert torch.equal(restored_generator.get_state(), expected_state[3])


def test_checkpoint_restore_keeps_rng_tensors_on_cpu_for_cuda_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader_generator = torch.Generator().manual_seed(456)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        extractor=model,
        optimizer=optimizer,
        epoch=1,
        config={"test": "cuda-target-cpu-rng"},
        best_score=1.0,
        significant_best=1.0,
        last_significant_epoch=1,
        loader_generator=loader_generator,
    )
    real_load = torch.load
    observed: dict[str, object] = {}

    def recording_load(*args, **kwargs):
        observed["map_location"] = kwargs.get("map_location")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    restored_model = nn.Linear(3, 2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    restored_generator = torch.Generator().manual_seed(999)
    restore_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_generator,
        torch.device("cuda"),
    )
    assert observed["map_location"] == "cpu"
    assert torch.equal(
        restored_generator.get_state(), loader_generator.get_state()
    )


def test_resume_records_reject_duplicates_and_checkpoint_skew() -> None:
    history = [{"epoch": 1}, {"epoch": 25}]
    audits = [{"epoch": 0}, {"epoch": 1}, {"epoch": 25}]
    validate_resume_records(history, audits, checkpoint_epoch=25)
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_resume_records(
            [{"epoch": 1}, {"epoch": 1}],
            [{"epoch": 0}, {"epoch": 1}, {"epoch": 1}],
            checkpoint_epoch=1,
        )
    with pytest.raises(RuntimeError, match="not aligned"):
        validate_resume_records(history, audits, checkpoint_epoch=1)
    with pytest.raises(RuntimeError, match="not aligned"):
        validate_resume_records(
            history,
            [{"epoch": 0}, {"epoch": 1}],
            checkpoint_epoch=25,
        )


def test_real_decision23_checkpoint_reproduces_two_next_steps(
    tmp_path: Path,
) -> None:
    seed_everything(910)
    model = Decision23Extractor(
        arm="probability_linear",
        num_keypoints=2,
        base_channels=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(911)
    batch = {
        "image": torch.randn(1, 3, 512, 512),
        "target": torch.randn(1, 2, 2).clamp(-1, 1),
    }
    _training_step(model, optimizer, batch)
    checkpoint_path = tmp_path / "decision23_checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        extractor=model,
        optimizer=optimizer,
        epoch=1,
        config={"test": "real_decision23"},
        best_score=1.0,
        significant_best=1.0,
        last_significant_epoch=1,
        loader_generator=generator,
    )

    expected = []
    for _ in range(2):
        torch.rand((), generator=generator)
        _training_step(model, optimizer, batch)
        expected.append(
            (
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(optimizer.state_dict()),
                torch.get_rng_state().clone(),
                generator.get_state().clone(),
            )
        )

    restored = Decision23Extractor(
        arm="probability_linear",
        num_keypoints=2,
        base_channels=1,
    )
    restored_optimizer = torch.optim.Adam(
        restored.parameters(), lr=1e-4, weight_decay=1e-5
    )
    restored_generator = torch.Generator().manual_seed(1)
    restore_checkpoint(
        checkpoint_path,
        restored,
        restored_optimizer,
        restored_generator,
        torch.device("cpu"),
    )
    for expected_state in expected:
        torch.rand((), generator=restored_generator)
        _training_step(restored, restored_optimizer, batch)
        _assert_nested_equal(restored.state_dict(), expected_state[0])
        _assert_nested_equal(restored_optimizer.state_dict(), expected_state[1])
        assert torch.equal(torch.get_rng_state(), expected_state[2])
        assert torch.equal(restored_generator.get_state(), expected_state[3])


def test_exact_once_test_claim_is_recoverable_and_fails_on_partial_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "arm": "raw_linear",
        "seed": 42,
        "git_commit": "a" * 40,
        "source_sha256": "b" * 64,
        "source_dependencies_sha256": {"model": "c" * 64},
        "slurm_runfiles_sha256": {"d1_smoke": "3" * 64},
        "decision_spec_sha256": "d" * 64,
        "prelaunch_lock": {"sha256": "e" * 64},
        "d1_report": {"sha256": "f" * 64},
    }
    record = {
        "run_dir": run_dir,
        "config": config,
        "config_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
    }
    copied_record = {**record, "run_dir": tmp_path / "copied_run"}
    assert _claim_path(tmp_path, record) == _claim_path(tmp_path, copied_record)
    output = tmp_path / "aggregate.json"
    template = Namespace(output_root=tmp_path)
    event = _prepare_finalization_event(
        [record],
        template,
        event_kind="SYNTHETIC",
        output_paths=(output,),
    )
    assert event["state"] == "fresh"
    retry = _prepare_finalization_event(
        [record],
        template,
        event_kind="SYNTHETIC",
        output_paths=(output,),
    )
    assert retry["state"] == "resume_incomplete"
    assert retry["pending_records"] == [record]
    assert retry["staged"] == []

    metrics, metrics_path, predictions_path = _write_synthetic_test_artifacts(
        run_dir, record
    )
    claim_path = _claim_path(tmp_path, record)
    recovered = _prepare_finalization_event(
        [record],
        template,
        event_kind="SYNTHETIC",
        output_paths=(output,),
    )
    assert recovered["state"] == "recover_aggregate"
    assert recovered["staged"][0]["test_metrics"] == metrics
    assert d23.read_jsonl(claim_path)[1]["recovered_existing_artifacts"] is True

    write_json(output, {"status": "complete"})
    _complete_finalization_event(recovered, (output,))
    completed = _prepare_finalization_event(
        [record],
        template,
        event_kind="SYNTHETIC",
        output_paths=(output,),
    )
    assert completed["state"] == "completed"

    partial_root = tmp_path / "partial"
    partial_run_dir = partial_root / "run"
    partial_run_dir.mkdir(parents=True)
    partial_record = {**record, "run_dir": partial_run_dir}
    partial_template = Namespace(output_root=partial_root)
    partial_event = _prepare_finalization_event(
        [partial_record],
        partial_template,
        event_kind="SYNTHETIC_PARTIAL",
        output_paths=(partial_root / "aggregate.json",),
    )
    write_json(partial_run_dir / "test_metrics.json", metrics)
    with pytest.raises(RuntimeError, match="partial state"):
        _prepare_finalization_event(
            [partial_record],
            partial_template,
            event_kind="SYNTHETIC_PARTIAL",
            output_paths=(partial_root / "aggregate.json",),
        )

    corrupt_root = tmp_path / "corrupt"
    corrupt_run_dir = corrupt_root / "run"
    corrupt_run_dir.mkdir(parents=True)
    corrupt_record = {**record, "run_dir": corrupt_run_dir}
    corrupt_template = Namespace(output_root=corrupt_root)
    corrupt_event = _prepare_finalization_event(
        [corrupt_record],
        corrupt_template,
        event_kind="SYNTHETIC_CORRUPT",
        output_paths=(corrupt_root / "aggregate.json",),
    )
    assert corrupt_event["state"] == "fresh"
    write_json(corrupt_run_dir / "test_metrics.json", metrics)
    (corrupt_run_dir / "test_predictions.npz").write_bytes(b"truncated")
    corrupt_claim = _claim_path(corrupt_root, corrupt_record)
    with pytest.raises(RuntimeError, match="invalid test predictions NPZ"):
        _prepare_finalization_event(
            [corrupt_record],
            corrupt_template,
            event_kind="SYNTHETIC_CORRUPT",
            output_paths=(corrupt_root / "aggregate.json",),
        )
    assert len(d23.read_jsonl(corrupt_claim)) == 1


def test_recovery_completion_requires_an_exclusive_lock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = {
        "run_dir": run_dir,
        "config": {
            "arm": "raw_linear",
            "seed": 42,
            "git_commit": "a" * 40,
            "source_sha256": "b" * 64,
            "source_dependencies_sha256": {"model": "c" * 64},
            "slurm_runfiles_sha256": {"d1_smoke": "3" * 64},
            "decision_spec_sha256": "d" * 64,
            "prelaunch_lock": {"sha256": "e" * 64},
            "d1_report": {"sha256": "f" * 64},
        },
        "config_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
    }
    output = tmp_path / "aggregate.json"
    template = Namespace(output_root=tmp_path)
    event = _prepare_finalization_event(
        [record],
        template,
        event_kind="SYNTHETIC_LOCK",
        output_paths=(output,),
    )
    assert event["state"] == "fresh"
    _write_synthetic_test_artifacts(run_dir, record)
    claim_path = _claim_path(tmp_path, record)
    recovery_lock = claim_path.with_suffix(".recovery.lock")
    d23.write_jsonl_exclusive(
        recovery_lock,
        {"status": "competing_recovery_reservation"},
    )
    with pytest.raises(FileExistsError):
        _prepare_finalization_event(
            [record],
            template,
            event_kind="SYNTHETIC_LOCK",
            output_paths=(output,),
        )
    assert len(d23.read_jsonl(claim_path)) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"arm": "probability_linear"},
        {"seed": 43},
        {"config_sha256": "9" * 64},
        {"checkpoint_sha256": "8" * 64},
        {"instrument_pass": False},
    ],
    ids=(
        "wrong-arm",
        "wrong-seed",
        "wrong-config",
        "wrong-checkpoint",
        "wrong-instrument-result",
    ),
)
def test_recovery_rejects_wrong_metric_identity_without_completion(
    tmp_path: Path,
    override: dict,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = {
        "run_dir": run_dir,
        "config": {
            "arm": "raw_linear",
            "seed": 42,
            "git_commit": "a" * 40,
            "source_sha256": "b" * 64,
            "source_dependencies_sha256": {"model": "c" * 64},
            "slurm_runfiles_sha256": {"d1_smoke": "3" * 64},
            "decision_spec_sha256": "d" * 64,
            "prelaunch_lock": {"sha256": "e" * 64},
            "d1_report": {"sha256": "f" * 64},
        },
        "config_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
    }
    output = tmp_path / "aggregate.json"
    template = Namespace(output_root=tmp_path)
    event = _prepare_finalization_event(
        [record],
        template,
        event_kind="SYNTHETIC_IDENTITY",
        output_paths=(output,),
    )
    assert event["state"] == "fresh"
    _write_synthetic_test_artifacts(
        run_dir,
        record,
        metrics_override=override,
    )
    claim_path = _claim_path(tmp_path, record)
    with pytest.raises(RuntimeError, match="metric identity mismatch"):
        _prepare_finalization_event(
            [record],
            template,
            event_kind="SYNTHETIC_IDENTITY",
            output_paths=(output,),
        )
    assert len(d23.read_jsonl(claim_path)) == 1


def _training_step(
    model: Decision23Extractor,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    coordinates = model.get_keypoint_coords(batch["image"])
    loss = F.mse_loss(coordinates, batch["target"])
    loss.backward()
    optimizer.step()


def test_gradient_audit_is_observational_and_next_step_is_bitwise_identical() -> None:
    seed_everything(321)
    audited = Decision23Extractor(
        arm="raw_linear",
        num_keypoints=2,
        base_channels=1,
    ).train()
    control = copy.deepcopy(audited)
    audited_optimizer = torch.optim.Adam(audited.parameters(), lr=1e-4)
    control_optimizer = torch.optim.Adam(control.parameters(), lr=1e-4)
    batch = {
        "image": torch.randn(1, 3, 512, 512),
        "target": torch.randn(1, 2, 2).clamp(-1, 1),
    }
    _training_step(audited, audited_optimizer, batch)
    _training_step(control, control_optimizer, batch)
    _assert_nested_equal(audited.state_dict(), control.state_dict())
    _assert_nested_equal(
        audited_optimizer.state_dict(), control_optimizer.state_dict()
    )
    for index, (audited_parameter, control_parameter) in enumerate(
        zip(audited.parameters(), control.parameters(), strict=True)
    ):
        if index % 3 == 0:
            audited_parameter.grad = None
            control_parameter.grad = None
        elif index % 3 == 1:
            audited_parameter.grad = torch.zeros_like(audited_parameter)
            control_parameter.grad = torch.zeros_like(control_parameter)
        else:
            audited_parameter.grad = torch.full_like(audited_parameter, 0.125)
            control_parameter.grad = torch.full_like(control_parameter, 0.125)

    python_rng = random.getstate()
    numpy_rng = copy.deepcopy(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    loader_generator = torch.Generator().manual_seed(777)
    loader_rng = loader_generator.get_state().clone()
    modes = [module.training for module in audited.modules()]
    parameter_state = copy.deepcopy(audited.state_dict())
    optimizer_state = copy.deepcopy(audited_optimizer.state_dict())
    gradients = [
        None if parameter.grad is None else parameter.grad.clone()
        for parameter in audited.parameters()
    ]

    audit = gradient_audit(audited, batch, torch.device("cpu"))
    assert_expected_upstream_gradients(audit)
    assert audit["pooled_logit_gradient_l2_norm"] > 0
    assert len(audit["per_channel_logit_gradient_l2_norm"]) == 2
    assert audit["fixed_expectation_coordinate_error_cells64"]["n"] == 2
    _assert_nested_equal(audited.state_dict(), parameter_state)
    _assert_nested_equal(audited_optimizer.state_dict(), optimizer_state)
    assert modes == [module.training for module in audited.modules()]
    assert random.getstate() == python_rng
    _assert_numpy_rng_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    if cuda_rng is not None:
        for observed, expected in zip(
            torch.cuda.get_rng_state_all(), cuda_rng, strict=True
        ):
            assert torch.equal(observed, expected)
    assert torch.equal(loader_generator.get_state(), loader_rng)
    for parameter, expected_gradient in zip(
        audited.parameters(), gradients, strict=True
    ):
        if expected_gradient is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, expected_gradient)

    audited_generator = torch.Generator()
    audited_generator.set_state(loader_rng)
    control_generator = torch.Generator()
    control_generator.set_state(loader_rng)

    def branch_step(model, optimizer, generator) -> None:
        random.random()
        np.random.random()
        torch.rand(())
        torch.rand((), generator=generator)
        _training_step(model, optimizer, batch)

    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    branch_step(audited, audited_optimizer, audited_generator)
    audited_post_rng = (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
        (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if cuda_rng is not None
            else None
        ),
    )
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    branch_step(control, control_optimizer, control_generator)
    _assert_nested_equal(audited.state_dict(), control.state_dict())
    _assert_nested_equal(
        audited_optimizer.state_dict(),
        control_optimizer.state_dict(),
    )
    for audited_parameter, control_parameter in zip(
        audited.parameters(), control.parameters(), strict=True
    ):
        assert torch.equal(audited_parameter.grad, control_parameter.grad)
    assert random.getstate() == audited_post_rng[0]
    _assert_numpy_rng_equal(np.random.get_state(), audited_post_rng[1])
    assert torch.equal(torch.get_rng_state(), audited_post_rng[2])
    if cuda_rng is not None:
        for observed, expected in zip(
            torch.cuda.get_rng_state_all(), audited_post_rng[3], strict=True
        ):
            assert torch.equal(observed, expected)
    assert torch.equal(
        audited_generator.get_state(), control_generator.get_state()
    )


def test_frozen_probe_audit_reports_backbone_gradients_as_not_applicable() -> None:
    model = Decision23Extractor(
        arm="raw_linear",
        num_keypoints=2,
        base_channels=1,
    )
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    batch = {
        "image": torch.randn(1, 3, 512, 512),
        "target": torch.randn(1, 2, 2).clamp(-1, 1),
    }
    audit = gradient_audit(model, batch, torch.device("cpu"))
    assert audit["backbone_frozen"]
    assert audit["parameter_gradient_norms"]["encoder_first_conv_weight"] is None
    assert audit["parameter_gradient_norms"]["decoder_weight"] > 0
    assert audit["pooled_logit_gradient_l2_norm"] > 0
    assert_expected_upstream_gradients(audit)
