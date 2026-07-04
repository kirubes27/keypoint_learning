import json
import sys
from pathlib import Path

import pytest
import torch


KEYPOINT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KEYPOINT_ROOT))

from model import KeypointExtractor  # noqa: E402
from diagnostics.stage_a_supervised_control import (  # noqa: E402
    instrument_gate,
    load_phase_split,
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
