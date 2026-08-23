"""Render the worst certified-witness failures of the fixed RGB transport field."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from material_transport_gate_io import (
        file_record,
        load_json,
        require,
        resolve_rgb_paths,
        write_json,
    )


EXPECTED_ARRAYS_SHA256 = "6227a50533284d0ef03bb95b52fefdebde828e6e601e524d8bb8e015f83ba120"
EXPECTED_RESULT_SHA256 = "2909b1bbfeb864a81544ac400cd8ff3a93d4565351b268c300bc2f35e1bcbafa"
EXPECTED_MANIFEST_SHA256 = "f4a6d27922bcaa6446590a227bbf75c9c38730b5288fef763f0e164225098f29"
IMAGE_SIZE = 512
HEADER_HEIGHT = 62
ROWS = 4
COLS = 3


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mark(image: Image.Image, xy: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    draw = ImageDraw.Draw(image)
    x, y = (float(xy[0]), float(xy[1]))
    radius = 10
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=5)
    draw.line((x - 14, y, x + 14, y), fill=color, width=3)
    draw.line((x, y - 14, x, y + 14), fill=color, width=3)
    draw.text((x + 13, y - 13), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))


def _panel(image: Image.Image, title: str) -> Image.Image:
    panel = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE + HEADER_HEIGHT), "white")
    panel.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(panel)
    draw.multiline_text((10, 8), title, fill="black", font=ImageFont.load_default(), spacing=4)
    return panel


def _destination_crop(
    image: Image.Image,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    low = np.minimum(truth, prediction) - 45.0
    high = np.maximum(truth, prediction) + 45.0
    center = 0.5 * (low + high)
    extent = max(float(np.max(high - low)), 150.0)
    x0 = int(np.floor(center[0] - 0.5 * extent))
    y0 = int(np.floor(center[1] - 0.5 * extent))
    x0 = max(0, min(IMAGE_SIZE - int(np.ceil(extent)), x0))
    y0 = max(0, min(IMAGE_SIZE - int(np.ceil(extent)), y0))
    x1 = min(IMAGE_SIZE, x0 + int(np.ceil(extent)))
    y1 = min(IMAGE_SIZE, y0 + int(np.ceil(extent)))
    box = (x0, y0, x1, y1)
    crop = image.crop(box).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    scale_x = IMAGE_SIZE / (x1 - x0)
    scale_y = IMAGE_SIZE / (y1 - y0)
    truth_crop = np.asarray([(truth[0] - x0) * scale_x, (truth[1] - y0) * scale_y])
    prediction_crop = np.asarray(
        [(prediction[0] - x0) * scale_x, (prediction[1] - y0) * scale_y]
    )
    _mark(crop, truth_crop, (0, 200, 0), "TRUE")
    _mark(crop, prediction_crop, (255, 0, 255), "PRED")
    ImageDraw.Draw(crop).line(
        (*truth_crop, *prediction_crop), fill=(255, 210, 0), width=4
    )
    return crop, box


def render(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    implementation_head = _git(args.repo_root, "rev-parse", "HEAD")
    require(_git(args.repo_root, "status", "--porcelain") == "", "repository is dirty")
    require(file_record(args.arrays)["sha256"] == EXPECTED_ARRAYS_SHA256, "arrays SHA-256 differs")
    require(file_record(args.result)["sha256"] == EXPECTED_RESULT_SHA256, "result SHA-256 differs")
    require(file_record(args.manifest)["sha256"] == EXPECTED_MANIFEST_SHA256, "manifest SHA-256 differs")
    result = load_json(args.result)
    require(result["decision"]["branch"] == "reject_exact_r64_centered_field", "unexpected replay branch")
    manifest = load_json(args.manifest)
    rgb_paths = resolve_rgb_paths(manifest, object_root_override=args.object_root)

    with np.load(args.arrays) as archive:
        witness_id = np.asarray(archive["witness_id"], dtype=np.int64)
        source_px = np.asarray(archive["source_coordinate_px"], dtype=np.float64)
        target_px = np.asarray(archive["target_coordinate_px"], dtype=np.float64)
        forward_prediction = np.asarray(archive["forward_prediction_px"], dtype=np.float64)
        reverse_prediction = np.asarray(archive["reverse_prediction_px"], dtype=np.float64)
        forward_error = np.asarray(archive["forward_error_px"], dtype=np.float64)
        reverse_error = np.asarray(archive["reverse_error_px"], dtype=np.float64)

    candidates: list[dict[str, Any]] = []
    for witness_index, identifier in enumerate(witness_id):
        for direction, errors in (("forward", forward_error), ("reverse", reverse_error)):
            frame = int(np.argmax(errors[:, witness_index]))
            candidates.append(
                {
                    "direction": direction,
                    "frame": frame,
                    "witness_index": witness_index,
                    "witness_id": int(identifier),
                    "error_px": float(errors[frame, witness_index]),
                }
            )
    selected = sorted(candidates, key=lambda row: row["error_px"], reverse=True)[:ROWS]

    canvas = Image.new("RGB", (COLS * IMAGE_SIZE, ROWS * (IMAGE_SIZE + HEADER_HEIGHT)), "white")
    event_rows: list[dict[str, Any]] = []
    for row_index, event in enumerate(selected):
        frame = int(event["frame"])
        witness = int(event["witness_index"])
        if event["direction"] == "forward":
            query_frame = frame
            destination_frame = (frame + 1) % len(rgb_paths)
            query_coordinate = source_px[frame, witness]
            truth = target_px[frame, witness]
            prediction = forward_prediction[frame, witness]
        else:
            query_frame = (frame + 1) % len(rgb_paths)
            destination_frame = frame
            query_coordinate = target_px[frame, witness]
            truth = source_px[frame, witness]
            prediction = reverse_prediction[frame, witness]

        with Image.open(rgb_paths[query_frame]) as opened:
            query_image = opened.convert("RGB")
        with Image.open(rgb_paths[destination_frame]) as opened:
            destination_image = opened.convert("RGB")
        require(query_image.size == (IMAGE_SIZE, IMAGE_SIZE), "query RGB dimensions differ")
        require(destination_image.size == (IMAGE_SIZE, IMAGE_SIZE), "destination RGB dimensions differ")
        crop, crop_box = _destination_crop(destination_image.copy(), truth, prediction)
        _mark(query_image, query_coordinate, (0, 180, 255), "QUERY")
        _mark(destination_image, truth, (0, 200, 0), "TRUE")
        _mark(destination_image, prediction, (255, 0, 255), "PRED")
        ImageDraw.Draw(destination_image).line((*truth, *prediction), fill=(255, 210, 0), width=4)

        common = (
            f"{event['direction'].upper()}  witness {event['witness_id']}  "
            f"edge {frame}->{(frame + 1) % len(rgb_paths)}  error {event['error_px']:.2f}px"
        )
        panels = (
            _panel(query_image, common + f"\nquery frame {query_frame}; cyan = source query"),
            _panel(destination_image, common + f"\ndestination frame {destination_frame}; green = true, magenta = predicted"),
            _panel(crop, common + f"\ndestination crop {crop_box}; yellow = error vector"),
        )
        y = row_index * (IMAGE_SIZE + HEADER_HEIGHT)
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * IMAGE_SIZE, y))

        event_rows.append(
            {
                **event,
                "query_frame": query_frame,
                "destination_frame": destination_frame,
                "query_coordinate_px": query_coordinate.tolist(),
                "true_destination_px": truth.tolist(),
                "predicted_destination_px": prediction.tolist(),
                "destination_crop_box_xyxy": list(crop_box),
                "query_rgb": file_record(rgb_paths[query_frame]),
                "destination_rgb": file_record(rgb_paths[destination_frame]),
            }
        )

    args.output_dir.mkdir(parents=True)
    montage_path = args.output_dir / "WORST_R64_TRANSPORT_FAILURES.png"
    canvas.save(montage_path)
    receipt = {
        "schema_version": "material_transport_witness_failure_visual.v1",
        "artifact_type": "posthash_visual_geometry_audit",
        "implementation_head": implementation_head,
        "implementation_source": file_record(Path(__file__)),
        "selection_rule": "largest per-witness per-direction error, then top four globally",
        "events": event_rows,
        "sources": {
            "arrays": file_record(args.arrays),
            "result": file_record(args.result),
            "sanitized_manifest": file_record(args.manifest),
        },
        "visual": file_record(montage_path),
        "training_or_weight_update_performed": False,
        "statistical_scope": "four largest descriptive failure events from ten fixed witnesses",
    }
    receipt_path = args.output_dir / "WITNESS_FAILURE_VISUAL_RECEIPT.json"
    write_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--object-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(render(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
