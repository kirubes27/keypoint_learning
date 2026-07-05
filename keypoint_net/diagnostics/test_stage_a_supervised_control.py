import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from model import KeypointExtractor, spatial_softmax  # noqa: E402
from diagnostics.day45_supervised_control import supervised_loss  # noqa: E402
from diagnostics.stage_a_supervised_control import (  # noqa: E402
    add_probe_ratios,
    build_extractor,
    channel_to_target_indices,
    coordinate_path_probe,
    counterfactual_gradient_norms,
    instrument_gate,
    load_phase_split,
    r1_probe_metrics,
    run_name,
    supervised_objective,
)


def _write_split(path: Path, *, overlap: bool = False) -> None:
    rows = {"train": [], "val": [], "test": []}
    destination = {0: "train", 1: "val", 2: "test", 3: "train", 4: "val", 5: "test"}
    for frame in range(180):
        key = destination[frame % 6]
        rows[key].append(
            {
                "model_name": "engineers_hammer_vray",
                "frame_index": frame,
                "theta_deg": 2.0 * frame,
                "image_relpath": f"frame_{frame:04d}.png",
            }
        )
    if overlap:
        rows["val"].append(dict(rows["train"][0]))
    path.write_text(json.dumps(rows))


def test_phase_split_is_exact_partition(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    _write_split(path)
    split = load_phase_split(path, "engineers_hammer_vray")
    assert len(split.train) == len(split.validation) == len(split.test) == 60
    assert set(split.train) | set(split.validation) | set(split.test) == set(range(180))
    assert not (set(split.train) & set(split.validation))


def test_phase_split_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    _write_split(path, overlap=True)
    with pytest.raises(ValueError):
        load_phase_split(path, "engineers_hammer_vray")


def _test_metrics(median: float, p90: float, on_mask: float) -> dict:
    one = {
        "median_of_channel_medians_cells64": median,
        "p90_error_cells64": p90,
        "on_mask_fraction": on_mask,
    }
    return {"unaugmented": dict(one), "fixed_augmented": dict(one)}


def test_instrument_gate_requires_all_joint_conditions() -> None:
    assert instrument_gate(_test_metrics(0.50, 1.50, 0.95))
    assert not instrument_gate(_test_metrics(0.51, 1.50, 0.95))
    assert not instrument_gate(_test_metrics(0.50, 1.51, 0.95))
    assert not instrument_gate(_test_metrics(0.50, 1.50, 0.94))


def test_instrument_gate_uses_worse_augmented_condition() -> None:
    metrics = _test_metrics(0.4, 1.0, 0.99)
    metrics["fixed_augmented"]["median_of_channel_medians_cells64"] = 0.6
    assert not instrument_gate(metrics)


def test_native_quarter_is_encoder_resolution_not_upsampling() -> None:
    standard = KeypointExtractor(num_keypoints=10, base_channels=4, heatmap_res=64)
    upsampled = KeypointExtractor(num_keypoints=10, base_channels=4, heatmap_res=128)
    native = KeypointExtractor(
        num_keypoints=10,
        base_channels=4,
        heatmap_res=128,
        true_quarter_res=True,
    )
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        _, standard_heatmaps = standard(image)
        _, upsampled_heatmaps = upsampled(image)
        _, native_heatmaps = native(image)
    assert standard_heatmaps.shape[-2:] == (8, 8)
    assert upsampled_heatmaps.shape[-2:] == (16, 16)
    assert native_heatmaps.shape[-2:] == (16, 16)
    assert upsampled.head_upsample is not None
    assert native.head_upsample is None
    assert native.encoder[6].stride == (1, 1)


def test_true_quarter_requires_128_heatmaps() -> None:
    with pytest.raises(ValueError):
        KeypointExtractor(heatmap_res=64, true_quarter_res=True)


@pytest.mark.parametrize("k", [5, 10, 15, 20])
def test_stage_a_extractor_and_run_name_follow_k(k: int) -> None:
    args = Namespace(
        num_keypoints=k,
        base_channels=4,
        native_quarter=False,
        supervision="coordinate",
        target_shift=0,
        seed=43,
    )
    extractor = build_extractor(args, torch.device("cpu"))
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        coords, heatmaps = extractor(image)
    assert coords.shape == (1, 2 * k)
    assert heatmaps.shape[1] == k
    assert run_name(args) == f"coordinate_standard64_k{k}_seed43"


def test_cyclic_target_assignment_moves_hard_targets_off_same_channels() -> None:
    mapping = channel_to_target_indices(10, 1)
    assert mapping == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    assert mapping.index(3) == 2
    assert mapping.index(6) == 5
    assert mapping.index(9) == 8


def test_run_name_records_supervision_and_shift() -> None:
    args = Namespace(
        num_keypoints=10,
        base_channels=4,
        native_quarter=False,
        supervision="heatmap",
        target_shift=0,
        seed=42,
    )
    assert run_name(args) == "heatmap_standard64_k10_seed42"
    args.supervision = "coordinate"
    args.target_shift = 1
    assert run_name(args) == "coordinate_standard64_k10_shift1_seed42"


def test_shape_constraint_run_name_is_explicit_and_default_name_is_unchanged() -> None:
    args = Namespace(
        num_keypoints=10,
        native_quarter=False,
        supervision="coordinate",
        target_shift=0,
        seed=42,
        shape_constraint="none",
    )
    assert run_name(args) == "coordinate_standard64_k10_seed42"
    args.shape_constraint = "prediction_centered_js"
    assert run_name(args) == "coordinate_standard64_k10_shapejs_seed42"
    args.shape_constraint = "conditional_deadzone"
    assert run_name(args) == "coordinate_standard64_k10_deadzone_seed42"


def test_default_supervised_objective_is_exact_legacy_loss() -> None:
    coordinates = torch.randn(2, 3, 2)
    heatmaps = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 2)
    args = Namespace(
        supervision="coordinate",
        shape_constraint="none",
        shape_weight=0.0,
        shape_sigma_cells=1.0,
    )
    total, parts = supervised_objective(args, coordinates, heatmaps, target)
    expected = supervised_loss("coordinate", coordinates, heatmaps, target)
    assert torch.equal(total, expected)
    assert parts["shape_loss"] == 0.0


def test_conditional_deadzone_objective_is_zero_on_healthy_heatmap() -> None:
    size = 33
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    heatmaps = (-0.5 * ((x - 16) ** 2 + (y - 16) ** 2)).float()[None, None]
    coordinates = spatial_softmax(heatmaps)
    target = coordinates.detach().clone()
    args = Namespace(
        supervision="coordinate",
        shape_constraint="conditional_deadzone",
        shape_weight=8.723808430294485e-06,
        shape_sigma_cells=1.0,
    )
    total, parts = supervised_objective(args, coordinates, heatmaps, target)
    assert parts["shape_loss"] == 0.0
    assert float(total) == 0.0


class _FixedHeatmapExtractor(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, image: torch.Tensor):
        logits = self.logits.expand(image.shape[0], -1, -1, -1)
        coords = spatial_softmax(logits)
        return coords.flatten(1), logits


def test_r1_probe_accepts_healthy_one_cell_gaussian_shape_and_gradient() -> None:
    size = 33
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    logits = (-0.5 * ((x - 16) ** 2 + (y - 16) ** 2)).float()[None, None]
    extractor = _FixedHeatmapExtractor(logits)
    image = torch.zeros(4, 3, 16, 16)
    flat, heatmaps = extractor(image)
    coordinates = flat.view(4, 1, 2)
    initial = counterfactual_gradient_norms(heatmaps, coordinates).detach()
    metrics = r1_probe_metrics(extractor, image, initial)
    assert metrics["shape_gate_pass"]
    assert metrics["counterfactual_gradient_gate_pass"]
    assert min(metrics["per_channel_median_dominant_mass_r2"]) >= 0.70
    assert metrics["run_median_counterfactual_gradient_final_initial_ratio"] == pytest.approx(1.0)


def test_coordinate_path_probe_records_shape_and_relative_sensitivity() -> None:
    size = 33
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    logits = (-0.5 * ((x - 16) ** 2 + (y - 16) ** 2)).float()[None, None]
    extractor = _FixedHeatmapExtractor(logits)
    extractor.train()
    loader = [{"image": torch.zeros(4, 3, 16, 16)}]
    initial = coordinate_path_probe(extractor, loader, torch.device("cpu"))
    relative = add_probe_ratios(initial, initial)
    assert extractor.training
    assert initial["n_frames"] == 4
    assert len(initial["per_channel_median_max_probability"]) == 1
    assert initial["per_channel_median_counterfactual_gradient_l2"][0] > 0
    assert relative["minimum_channel_counterfactual_gradient_initial_ratio"] == pytest.approx(1.0)
    assert relative["collapsed_gradient_channel_indices"] == []
