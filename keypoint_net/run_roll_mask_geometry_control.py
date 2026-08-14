"""Verify world-Z roll sign, pivot, and +2-degree step on the bound masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .frozen_wobble_forensics import warp_image_by_standard_rotation
except ImportError:
    from frozen_wobble_forensics import warp_image_by_standard_rotation  # type: ignore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    if union == 0:
        raise ValueError("empty mask union")
    return float(np.count_nonzero(left & right) / union)


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "maximum": float(np.max(values)),
        "sample_unit": "correlated rendered mask frame or adjacent mask pair",
        "descriptive_not_inferential": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    object_root = args.data_root.resolve(strict=True) / "train" / "engineers_hammer_vray"
    meta_path = object_root / "meta.jsonl"
    rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if len(rows) != 180:
        raise ValueError("expected 180 metadata rows")
    masks = []
    for index, row in enumerate(rows):
        if int(row["frame_index"]) != index or float(row["theta_deg"]) != 2.0 * index:
            raise ValueError("metadata angle/index lock differs")
        if not (row["operator_name"] == "tdw_world_z_roll" and row["tdw_axis"] == "roll" and row["is_world"] is True and row["use_centroid"] is True):
            raise ValueError("world-Z roll metadata differs")
        with Image.open(object_root / row["mask_relpath"]) as handle:
            masks.append(np.asarray(handle.convert("L")) > 0)
    masks_array = np.stack(masks)
    base = masks_array[0]
    pivots = {
        "locked_0_0": (0.0, 0.0),
        "x_plus_0_05": (0.05, 0.0),
        "x_minus_0_05": (-0.05, 0.0),
        "y_plus_0_05": (0.0, 0.05),
        "y_minus_0_05": (0.0, -0.05),
    }
    canonical_iou: dict[str, list[float]] = {name: [] for name in pivots}
    wrong_sign_iou: list[float] = []
    for index, row in enumerate(rows):
        theta = float(row["theta_deg"])
        for name, pivot in pivots.items():
            recovered = warp_image_by_standard_rotation(
                masks_array[index].astype(float), -theta, pivot=pivot, order=0
            ) > 0.5
            canonical_iou[name].append(_iou(base, recovered))
        wrong = warp_image_by_standard_rotation(
            masks_array[index].astype(float), theta, pivot=(0.0, 0.0), order=0
        ) > 0.5
        wrong_sign_iou.append(_iou(base, wrong))

    forward_iou: list[float] = []
    reverse_sign_iou: list[float] = []
    for index in range(179):
        predicted = warp_image_by_standard_rotation(
            masks_array[index].astype(float), 2.0, order=0
        ) > 0.5
        wrong = warp_image_by_standard_rotation(
            masks_array[index].astype(float), -2.0, order=0
        ) > 0.5
        forward_iou.append(_iou(masks_array[index + 1], predicted))
        reverse_sign_iou.append(_iou(masks_array[index + 1], wrong))

    correct = np.asarray(canonical_iou["locked_0_0"])
    wrong_sign = np.asarray(wrong_sign_iou)
    forward = np.asarray(forward_iou)
    reverse = np.asarray(reverse_sign_iou)
    pivot_medians = {name: float(np.median(values)) for name, values in canonical_iou.items()}
    best_pivot = max(pivot_medians, key=pivot_medians.get)
    result: dict[str, Any] = {
        "schema_version": "roll_mask_geometry_control.v1",
        "artifact_type": "bound_renderer_mask_geometry_control",
        "training_or_weight_update_performed": False,
        "metadata": {
            "frame_count": 180,
            "theta_deg": "0..358 in exact +2-degree increments",
            "operator": "TDW world-Z roll about centroid",
            "metadata_file": {"absolute_path": str(meta_path), "sha256": _sha256(meta_path)},
        },
        "canonical_mask_iou_locked_minus_theta_pivot_0_0": _summary(correct),
        "canonical_mask_iou_wrong_plus_theta_pivot_0_0": _summary(wrong_sign),
        "pivot_candidate_median_ious": pivot_medians,
        "highest_median_pivot_candidate": best_pivot,
        "adjacent_plus2_mask_prediction_iou": _summary(forward),
        "adjacent_wrong_minus2_mask_prediction_iou": _summary(reverse),
        "adjacent_plus2_iou_by_source_frame_mod3": {
            str(residue): _summary(forward[np.arange(179) % 3 == residue])
            for residue in range(3)
        },
        "semantic_checks": {
            "locked_sign_better_than_wrong_sign_median": float(np.median(correct)) > float(np.median(wrong_sign)),
            "locked_pivot_is_best_predeclared_candidate": best_pivot == "locked_0_0",
            "plus2_step_better_than_wrong_minus2_median": float(np.median(forward)) > float(np.median(reverse)),
        },
        "statistical_scope": {
            "statistics": "empirical mean, median, minimum, q05, and maximum IoU",
            "sample_unit": "correlated rendered mask frame or overlapping adjacent pair",
            "descriptive_not_inferential": True,
            "correlation_caveat": "all masks belong to one closed orbit; no SEM or population CI is reported",
        },
    }
    result["all_semantic_checks_pass"] = all(result["semantic_checks"].values())
    output_dir = args.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "MASK_GEOMETRY_CONTROL.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
    axes[0].plot(correct, label="locked -theta, pivot (0,0)", lw=1.2)
    axes[0].plot(wrong_sign, label="wrong +theta", lw=0.8, alpha=0.75)
    axes[0].set_ylabel("IoU with frame-0 mask")
    axes[0].set_xlabel("frame index")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(np.arange(1, 180), forward, label="predict next with +2°", lw=1.2)
    axes[1].plot(np.arange(1, 180), reverse, label="wrong -2°", lw=0.8, alpha=0.75)
    axes[1].set_ylabel("adjacent mask IoU")
    axes[1].set_xlabel("target frame index")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        "Renderer-level geometry check on bound hammer masks\n"
        "IoU traces are descriptive over one correlated 180-frame orbit; no inferential band is shown."
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plot_path = output_dir / "MASK_GEOMETRY_CONTROL.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({
        "report": {"absolute_path": str(result_path), "sha256": _sha256(result_path)},
        "plot": {"absolute_path": str(plot_path), "sha256": _sha256(plot_path)},
        "all_semantic_checks_pass": result["all_semantic_checks_pass"],
    }, indent=2, sort_keys=True))
    if not result["all_semantic_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
