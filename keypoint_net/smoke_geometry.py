"""
Dataset-level geometry smoke checks for the TDW World-Z roll dataset.

This script verifies the semantic assumptions used by eval_rollout_viz:

1. Metadata says the dataset is TDW-rendered World-Z roll.
2. Masks exist and are non-empty.
3. Rotating each mask back to frame 0 around the image center has high IoU.
4. The canonical sign is locked to -theta; +theta is audit-only.

It intentionally does not inspect learned keypoints or choose the sign by
variance. This is a dataset geometry check, not a model-selection procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    return arr > 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _bbox(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "nonempty": False,
            "x0": None,
            "x1": None,
            "y0": None,
            "y1": None,
            "area_frac": 0.0,
            "margin_frac": 0.0,
            "centroid_xy": None,
        }
    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    margin = min(x0, y0, w - 1 - x1, h - 1 - y1) / min(w, h)
    return {
        "nonempty": True,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "area_frac": float(mask.mean()),
        "margin_frac": float(margin),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _warp_mask_same_eval_sign(
    mask: np.ndarray,
    theta_deg: float,
    *,
    center_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Apply the same image-coordinate point transform used by eval_rollout_viz,
    but with inverse sampling to avoid holes.

    Pixel coordinates use x right, y down, and pivot at the normalized-coordinate
    image origin, i.e. ((W - 1) / 2, (H - 1) / 2).
    """
    h, w = mask.shape
    if center_xy is None:
        center_xy = ((w - 1) / 2.0, (h - 1) / 2.0)
    cx, cy = (float(value) for value in center_xy)
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs - cx
    dy = ys - cy

    # Inverse sample for output q = R(+theta) source_p.
    src_x = cx + c * dx + s * dy
    src_y = cy - s * dx + c * dy
    src_x = np.rint(src_x).astype(np.int64)
    src_y = np.rint(src_y).astype(np.int64)

    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    out = np.zeros_like(mask, dtype=bool)
    out[valid] = mask[src_y[valid], src_x[valid]]
    return out


def _local_center_audit(
    *,
    object_dir: Path,
    by_frame: dict[int, dict[str, Any]],
    reference_mask: np.ndarray,
    sample_frame_indices: list[int],
) -> dict[str, Any]:
    """Check that mask motion locally selects the metadata-derived image centre.

    The camera/object code path derives the centre.  This small, fixed 3x3
    pixel grid is an independent rendered-output check, not a centre-fitting
    procedure used by the model or evaluator.
    """

    h, w = reference_mask.shape
    derived_center = ((w - 1) / 2.0, (h - 1) / 2.0)
    decisive_indices = [
        frame_index
        for frame_index in sample_frame_indices
        if abs(float(by_frame[frame_index]["theta_deg"]) % 180.0) > 1e-9
    ]
    candidates = []
    for y_offset in (-1.0, 0.0, 1.0):
        for x_offset in (-1.0, 0.0, 1.0):
            center = (
                derived_center[0] + x_offset,
                derived_center[1] + y_offset,
            )
            values = []
            for frame_index in decisive_indices:
                metadata = by_frame[frame_index]
                mask = _load_mask(object_dir / metadata["mask_relpath"])
                canonical = _warp_mask_same_eval_sign(
                    mask,
                    -float(metadata["theta_deg"]),
                    center_xy=center,
                )
                values.append(_iou(canonical, reference_mask))
            candidates.append(
                {
                    "center_pixel_xy": [center[0], center[1]],
                    "mean_canonical_iou": float(np.mean(values)),
                    "minimum_canonical_iou": float(np.min(values)),
                    "n_decisive_frames": len(values),
                }
            )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["mean_canonical_iou"],
            item["center_pixel_xy"][1],
            item["center_pixel_xy"][0],
        ),
    )
    best = ranked[0]
    next_best = ranked[1]
    exact_center_selected = best["center_pixel_xy"] == [
        derived_center[0],
        derived_center[1],
    ]
    unique_best = (
        best["mean_canonical_iou"] > next_best["mean_canonical_iou"]
    )
    return {
        "purpose": (
            "independent_rendered_output_check_of_metadata_derived_center_"
            "not_model_or_evaluator_fitting"
        ),
        "candidate_grid_pixel_offsets_xy": [-1.0, 0.0, 1.0],
        "decisive_frame_indices": decisive_indices,
        "derived_center_pixel_xy": [
            derived_center[0],
            derived_center[1],
        ],
        "derived_center_normalized_xy": [0.0, 0.0],
        "ranked_candidates": ranked,
        "exact_center_selected": bool(exact_center_selected),
        "unique_best": bool(unique_best),
        "pass": bool(exact_center_selected and unique_best),
    }


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _resolve_objects(dataset_index: dict[str, Any], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    objects = dataset_index.get("objects", [])
    if objects and isinstance(objects[0], dict):
        return [o["model_name"] for o in objects]
    if objects and isinstance(objects[0], str):
        return objects
    splits = dataset_index.get("splits", {})
    if "train" in splits:
        return list(splits["train"])
    raise ValueError("Could not infer objects from dataset_index.json")


def check_dataset(
    data_root: Path,
    objects: list[str],
    sample_frame_indices: list[int],
    iou_threshold: float,
) -> dict[str, Any]:
    object_results = []

    for obj in objects:
        obj_dir = data_root / "train" / obj
        meta_path = obj_dir / "meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata for {obj}: {meta_path}")
        meta = _load_jsonl(meta_path)
        if not meta:
            raise ValueError(f"Empty metadata file: {meta_path}")

        by_frame = {int(m["frame_index"]): m for m in meta}
        ref_meta = by_frame[0]
        ref_mask = _load_mask(obj_dir / ref_meta["mask_relpath"])
        ref_bbox = _bbox(ref_mask)
        local_center_audit = _local_center_audit(
            object_dir=obj_dir,
            by_frame=by_frame,
            reference_mask=ref_mask,
            sample_frame_indices=sample_frame_indices,
        )
        plus_ious = []
        minus_ious = []
        sign_samples = []
        bbox_samples = []

        for frame_idx in sample_frame_indices:
            if frame_idx not in by_frame:
                raise ValueError(f"{obj} missing frame {frame_idx}")
            m = by_frame[frame_idx]
            theta = float(m["theta_deg"])
            mask = _load_mask(obj_dir / m["mask_relpath"])
            bb = _bbox(mask)
            bbox_samples.append({"frame_index": frame_idx, "theta_deg": theta, **bb})

            plus = _warp_mask_same_eval_sign(mask, theta)
            minus = _warp_mask_same_eval_sign(mask, -theta)
            plus_iou = _iou(plus, ref_mask)
            minus_iou = _iou(minus, ref_mask)

            plus_ious.append(plus_iou)
            minus_ious.append(minus_iou)
            ambiguous = abs((theta % 180.0)) < 1e-9
            sign_samples.append({
                "frame_index": frame_idx,
                "theta_deg": theta,
                "plus_theta_iou": plus_iou,
                "minus_theta_iou": minus_iou,
                "ambiguous_for_sign": bool(ambiguous),
            })

        decisive = [s for s in sign_samples if not s["ambiguous_for_sign"]]
        decisive_plus = [s["plus_theta_iou"] for s in decisive]
        decisive_minus = [s["minus_theta_iou"] for s in decisive]
        minus_beats_plus = all(
            s["minus_theta_iou"] >= s["plus_theta_iou"] for s in decisive
        )
        min_decisive_minus = min(decisive_minus) if decisive_minus else 0.0

        metadata_semantics = {
            "operator_names": sorted({m.get("operator_name") for m in meta}),
            "tdw_axes": sorted({m.get("tdw_axis") for m in meta}),
            "is_world_values": sorted({bool(m.get("is_world")) for m in meta}),
            "rotation_is_tdw_rerendered_values": sorted(
                {bool(m.get("rotation_is_tdw_rerendered")) for m in meta}
            ),
            "use_centroid_values": sorted({bool(m.get("use_centroid")) for m in meta}),
            "theta_step_values": sorted({float(m.get("theta_step_deg")) for m in meta}),
        }
        semantics_pass = (
            metadata_semantics["operator_names"] == ["tdw_world_z_roll"]
            and metadata_semantics["tdw_axes"] == ["roll"]
            and metadata_semantics["is_world_values"] == [True]
            and metadata_semantics["rotation_is_tdw_rerendered_values"] == [True]
            and metadata_semantics["use_centroid_values"] == [True]
            and metadata_semantics["theta_step_values"] == [2.0]
        )

        all_nonempty = all(s["nonempty"] for s in bbox_samples)
        min_margin = min(s["margin_frac"] for s in bbox_samples)
        result = {
            "object": obj,
            "n_frames": len(meta),
            "reference_frame": {
                "frame_index": 0,
                "theta_deg": float(ref_meta["theta_deg"]),
                "bbox": ref_bbox,
            },
            "metadata_semantics": metadata_semantics,
            "metadata_semantics_pass": bool(semantics_pass),
            "sample_bbox": bbox_samples,
            "sign_samples": sign_samples,
            "plus_theta_iou_summary_all_samples": _summarize(plus_ious),
            "minus_theta_iou_summary_all_samples": _summarize(minus_ious),
            "decisive_plus_theta_iou_summary": _summarize(decisive_plus),
            "decisive_minus_theta_iou_summary": _summarize(decisive_minus),
            "locked_minus_theta_pass": bool(
                decisive
                and minus_beats_plus
                and min_decisive_minus >= iou_threshold
            ),
            "all_sample_masks_nonempty": bool(all_nonempty),
            "min_sample_margin_frac": float(min_margin),
            "center_pivot_pixel_xy": [
                (ref_mask.shape[1] - 1) / 2.0,
                (ref_mask.shape[0] - 1) / 2.0,
            ],
            "local_center_audit": local_center_audit,
            "input_evidence": {
                "metadata": {
                    "absolute_path": str(meta_path.resolve()),
                    "sha256": _sha256_file(meta_path),
                },
                "sample_masks": [
                    {
                        "frame_index": frame_index,
                        "absolute_path": str(
                            (
                                obj_dir
                                / by_frame[frame_index]["mask_relpath"]
                            ).resolve()
                        ),
                        "sha256": _sha256_file(
                            obj_dir / by_frame[frame_index]["mask_relpath"]
                        ),
                    }
                    for frame_index in sample_frame_indices
                ],
            },
        }
        object_results.append(result)

    return {
        "data_root": str(data_root),
        "check": "world_z_roll_mask_iou_sign_and_pivot",
        "coordinate_convention": {
            "keypoint_space": "normalized [-1, 1]",
            "normalized_pivot": [0.0, 0.0],
            "pixel_pivot_rule": "((W - 1) / 2, (H - 1) / 2)",
            "locked_canonical_sign": "-theta",
            "audit_sign": "+theta",
            "theta_source": "meta.jsonl theta_deg",
            "forward_three_frame_transform": "R_img(+6_deg)",
            "canonical_unrotation": "R_img(-theta)",
            "image_axes": "x_right_y_down_positive_angle_appears_clockwise",
        },
        "sample_frame_indices": sample_frame_indices,
        "iou_threshold": iou_threshold,
        "objects": object_results,
        "pass": bool(
            all(o["metadata_semantics_pass"] for o in object_results)
            and all(o["all_sample_masks_nonempty"] for o in object_results)
            and all(o["locked_minus_theta_pass"] for o in object_results)
            and all(o["local_center_audit"]["pass"] for o in object_results)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check TDW World-Z roll geometry")
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument(
        "--sample_frames",
        nargs="*",
        type=int,
        default=[0, 30, 60, 90, 120, 150],
        help="Frame indices to test. With 2 deg/frame: 30=60 deg, 150=300 deg.",
    )
    parser.add_argument("--iou_threshold", type=float, default=0.75)
    args = parser.parse_args()

    index_path = args.data_root / "dataset_index.json"
    with open(index_path) as f:
        dataset_index = json.load(f)
    objects = _resolve_objects(dataset_index, args.objects)

    results = check_dataset(
        data_root=args.data_root,
        objects=objects,
        sample_frame_indices=args.sample_frames,
        iou_threshold=args.iou_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write("\n")

    print(json.dumps({
        "pass": results["pass"],
        "output": str(args.output),
        "objects": [
            {
                "object": o["object"],
                "metadata": o["metadata_semantics_pass"],
                "locked_minus_theta": o["locked_minus_theta_pass"],
                "minus_min": o["decisive_minus_theta_iou_summary"]["min"],
                "plus_max": o["decisive_plus_theta_iou_summary"]["max"],
                "center_pixel_xy": o["center_pivot_pixel_xy"],
                "local_center_pass": o["local_center_audit"]["pass"],
            }
            for o in results["objects"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
