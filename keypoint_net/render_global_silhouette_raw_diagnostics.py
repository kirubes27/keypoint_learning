"""Render the lowest-IoU RGB-only silhouette pose events without truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .global_silhouette_teacher import IMAGE_SIZE, warp_mask
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from global_silhouette_teacher import IMAGE_SIZE, warp_mask
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )


EXPECTED_MANIFEST_SHA256 = "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
HEADER_HEIGHT = 72
ROWS = 4


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _boundary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(value, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (value > 0) & (eroded == 0)


def render(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    manifest_record = file_record(args.manifest)
    require(manifest_record["sha256"] == EXPECTED_MANIFEST_SHA256, "manifest SHA-256 differs")
    raw_record = file_record(args.raw_predictions)
    receipt_record = file_record(args.raw_receipt)
    receipt = load_json(args.raw_receipt)
    require(receipt["schema_version"] == "raw_global_rgb_silhouette_teacher.v1", "raw schema differs")
    require(receipt["raw_predictions"] == raw_record, "raw arrays binding differs")
    require(receipt["controls"]["supplied_masks_opened"] is False, "raw stage opened masks")
    require(receipt["controls"]["non_frame_zero_truth_opened"] is False, "raw stage opened truth")
    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)

    with np.load(args.raw_predictions) as raw:
        masks = np.asarray(raw["foreground_mask"], dtype=bool)
        matrices = np.asarray(raw["matrix"], dtype=np.float64)
        selected_angle = np.asarray(raw["selected_angle_rad"], dtype=np.float64)
        selected_iou = np.asarray(raw["selected_iou"], dtype=np.float64)
        candidate_angle = np.asarray(raw["candidate_angle_rad"], dtype=np.float64)
        candidate_iou = np.asarray(raw["candidate_iou"], dtype=np.float64)
    require(masks.shape == (len(rgb_paths), IMAGE_SIZE, IMAGE_SIZE), "mask shape differs")
    selected_frames = np.argsort(selected_iou, kind="stable")[:ROWS]

    canvas = Image.new("RGB", (IMAGE_SIZE, ROWS * (IMAGE_SIZE + HEADER_HEIGHT)), "white")
    event_rows: list[dict[str, Any]] = []
    reference_mask = masks[0]
    for row, frame_value in enumerate(selected_frames):
        frame = int(frame_value)
        with Image.open(rgb_paths[frame]) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        warped_reference = warp_mask(reference_mask, matrices[frame])
        target_boundary = _boundary(masks[frame])
        reference_boundary = _boundary(warped_reference)
        both = target_boundary & reference_boundary
        rgb[target_boundary] = np.asarray([0, 255, 0], dtype=np.uint8)
        rgb[reference_boundary] = np.asarray([255, 0, 255], dtype=np.uint8)
        rgb[both] = np.asarray([255, 220, 0], dtype=np.uint8)
        panel = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE + HEADER_HEIGHT), "white")
        panel.paste(Image.fromarray(rgb), (0, HEADER_HEIGHT))
        title = (
            f"frame {frame}; selected IoU {selected_iou[frame]:.5f}; "
            f"angle {np.degrees(selected_angle[frame]):.3f} deg\n"
            f"candidates: {np.degrees(candidate_angle[frame, 0]):.3f} deg / "
            f"{candidate_iou[frame, 0]:.5f}, {np.degrees(candidate_angle[frame, 1]):.3f} deg / "
            f"{candidate_iou[frame, 1]:.5f}; green=target, magenta=warped f0, yellow=overlap"
        )
        ImageDraw.Draw(panel).multiline_text(
            (8, 7), title, fill="black", font=ImageFont.load_default(), spacing=3
        )
        canvas.paste(panel, (0, row * (IMAGE_SIZE + HEADER_HEIGHT)))
        event_rows.append(
            {
                "frame_index": frame,
                "selected_angle_deg": float(np.degrees(selected_angle[frame])),
                "selected_iou": float(selected_iou[frame]),
                "candidate_angle_deg": np.degrees(candidate_angle[frame]).tolist(),
                "candidate_iou": candidate_iou[frame].tolist(),
                "rgb": file_record(rgb_paths[frame]),
            }
        )

    args.output_dir.mkdir(parents=True)
    visual_path = args.output_dir / "LOWEST_SILHOUETTE_IOU_EVENTS.png"
    canvas.save(visual_path)
    output = {
        "schema_version": "global_silhouette_raw_diagnostic_visual.v1",
        "artifact_type": "rgb_only_failed_pose_visual",
        "selection_rule": "four lowest selected silhouette IoUs; stable frame order for ties",
        "events": event_rows,
        "sources": {
            "raw_predictions": raw_record,
            "raw_receipt": receipt_record,
            "sanitized_manifest": manifest_record,
        },
        "implementation_head": implementation_head,
        "implementation_source": file_record(Path(__file__)),
        "visual": file_record(visual_path),
        "supplied_masks_or_non_frame_zero_truth_opened": False,
        "training_or_weight_update_performed": False,
    }
    receipt_path = args.output_dir / "RAW_SILHOUETTE_VISUAL_RECEIPT.json"
    write_json(receipt_path, output)
    return {**output, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", required=True, type=Path)
    parser.add_argument("--raw-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(render(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
