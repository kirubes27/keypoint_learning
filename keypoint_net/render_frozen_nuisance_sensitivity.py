"""Render deterministic worst-event RGB/heatmap panels for one nuisance role."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .frozen_nuisance_sensitivity import (
        IMAGE_SIZE,
        LOCKED_SPECS,
        apply_nuisance,
        apply_nuisance_coordinates,
        normalized_to_rgb,
    )
    from .run_frozen_feature_decode_raw import _image_paths, _load_manifest, _preprocess
    from .run_frozen_nuisance_sensitivity import _infer
    from .run_frozen_wobble_forensics import (
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )
except ImportError:  # pragma: no cover
    from frozen_nuisance_sensitivity import (  # type: ignore
        IMAGE_SIZE,
        LOCKED_SPECS,
        apply_nuisance,
        apply_nuisance_coordinates,
        normalized_to_rgb,
    )
    from run_frozen_feature_decode_raw import _image_paths, _load_manifest, _preprocess  # type: ignore
    from run_frozen_nuisance_sensitivity import _infer  # type: ignore
    from run_frozen_wobble_forensics import (  # type: ignore
        _construct_frozen_model,
        _same_fd_checkpoint_load,
        _state_sha256,
    )


class NuisanceRenderError(ValueError):
    """Raised when visual evidence differs from the bound raw result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NuisanceRenderError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path_value: Path | str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    return {"absolute_path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _xy_pixels(coordinate: np.ndarray) -> tuple[float, float]:
    value = np.asarray(coordinate, dtype=np.float64)
    return float((value[0] + 1.0) * (IMAGE_SIZE - 1) / 2.0), float(
        (value[1] + 1.0) * (IMAGE_SIZE - 1) / 2.0
    )


def _xy_heatmap(coordinate: np.ndarray) -> tuple[float, float]:
    value = np.asarray(coordinate, dtype=np.float64)
    return float((value[0] + 1.0) * 63.0 / 2.0), float((value[1] + 1.0) * 63.0 / 2.0)


def _overlay(ax: Any, image: np.ndarray, points: Sequence[tuple[np.ndarray, str, str, str]]) -> None:
    ax.imshow(image)
    for coordinate, color, marker, label in points:
        x, y = _xy_pixels(coordinate)
        ax.scatter([x], [y], s=90, c=color, marker=marker, linewidths=2, label=label)
    ax.set_xlim(0, IMAGE_SIZE - 1)
    ax.set_ylim(IMAGE_SIZE - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)


def render(
    *,
    manifest_path: Path,
    role_key: str,
    raw_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    _require(not output_directory.exists(), "output directory already exists")
    receipt_path = (raw_directory / "nuisance_sensitivity_receipt.json").resolve(strict=True)
    arrays_path = (raw_directory / "nuisance_sensitivity_arrays.npz").resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("role", {}).get("role_key") == role_key, "raw role differs")
    _require(receipt.get("arrays") == _record(arrays_path), "raw arrays binding differs")
    arrays = np.load(arrays_path, allow_pickle=False)
    names = [str(value) for value in arrays["nuisance_name"]]
    expected_names = [spec.name for spec in LOCKED_SPECS]
    _require(names == expected_names, "nuisance order differs")

    manifest, manifest_record = _load_manifest(manifest_path.resolve(strict=True))
    matches = [item for item in manifest["roles"] if item.get("role_key") == role_key]
    _require(len(matches) == 1, "role key is absent or duplicated")
    role = matches[0]
    checkpoint = role["checkpoint"]
    payload, checkpoint_record = _same_fd_checkpoint_load(
        checkpoint["absolute_path"],
        expected_sha256=checkpoint["sha256"],
        expected_size=checkpoint["size_bytes"],
    )
    model, _config = _construct_frozen_model(
        payload,
        cell_id=role["cell_id"],
        checkpoint_role=role["checkpoint_role"],
        expected_epoch=role["checkpoint_epoch"],
    )
    image_paths = _image_paths(manifest)
    state_before = _state_sha256(model)

    shown_specs = [spec for spec in LOCKED_SPECS if spec.kind != "identity"]
    figure, axes = plt.subplots(len(shown_specs), 5, figsize=(22, 12), constrained_layout=True)
    events: list[dict[str, Any]] = []
    for row, spec in enumerate(shown_specs):
        nuisance_index = names.index(spec.name)
        event = receipt["summary"][spec.name]["worst_soft_event"]
        frame = int(event["frame_index"])
        channel = int(event["channel_index"])
        source_input = _preprocess([image_paths[frame]])
        target_input = apply_nuisance(source_input, spec)
        baseline = _infer(model, source_input)
        perturbed = _infer(model, target_input)

        baseline_soft = baseline["soft_coordinate"][0, channel].cpu().numpy()
        baseline_hard = baseline["hard_coordinate"][0, channel].cpu().numpy()
        perturbed_soft = perturbed["soft_coordinate"][0, channel].cpu().numpy()
        perturbed_hard = perturbed["hard_coordinate"][0, channel].cpu().numpy()
        expected_soft = apply_nuisance_coordinates(
            baseline["soft_coordinate"][0, channel], spec
        ).cpu().numpy()

        _require(
            np.array_equal(baseline_soft, arrays["baseline_soft_coordinate"][frame, channel]),
            "recomputed baseline soft coordinate differs",
        )
        _require(
            np.array_equal(perturbed_soft, arrays["perturbed_soft_coordinate"][nuisance_index, frame, channel]),
            "recomputed perturbed soft coordinate differs",
        )

        source_rgb = normalized_to_rgb(source_input).clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
        target_rgb = normalized_to_rgb(target_input).clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
        baseline_logits = baseline["logits"][0, channel].cpu().numpy()
        perturbed_logits = perturbed["logits"][0, channel].cpu().numpy()

        _overlay(
            axes[row, 0],
            source_rgb,
            ((baseline_soft, "cyan", "o", "baseline soft"), (baseline_hard, "yellow", "x", "baseline hard")),
        )
        axes[row, 0].set_title(f"Source frame {frame}, KP{channel}")
        _overlay(
            axes[row, 1],
            target_rgb,
            (
                (expected_soft, "lime", "o", "expected material"),
                (perturbed_soft, "red", "o", "actual soft"),
                (perturbed_hard, "yellow", "x", "actual hard"),
            ),
        )
        residual = float(arrays["soft_residual_pixels"][nuisance_index, frame, channel])
        axes[row, 1].set_title(f"{spec.name}\nsoft residual {residual:.2f} px")

        for column, values, title in (
            (2, baseline_logits, "Baseline heatmap logits"),
            (3, perturbed_logits, "Perturbed heatmap logits"),
            (4, perturbed_logits - baseline_logits, "Perturbed − baseline logits"),
        ):
            axes[row, column].imshow(values, cmap="magma" if column < 4 else "coolwarm")
            axes[row, column].set_title(title)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        for column, coordinate, color, marker in (
            (2, baseline_soft, "cyan", "o"),
            (3, expected_soft, "lime", "o"),
            (3, perturbed_soft, "red", "o"),
        ):
            x, y = _xy_heatmap(coordinate)
            axes[row, column].scatter([x], [y], s=70, c=color, marker=marker, linewidths=2)

        events.append(
            {
                "nuisance": spec.name,
                "frame_index": frame,
                "channel_index": channel,
                "soft_residual_pixels": residual,
                "source_image": _record(image_paths[frame]),
            }
        )

    figure.suptitle(f"Frozen nuisance sensitivity — {role_key}", fontsize=16)
    output_directory.mkdir(parents=True, exist_ok=False)
    image_path = output_directory / "worst_nuisance_events.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)
    state_after = _state_sha256(model)
    _require(state_before == state_after, "model state changed while rendering")
    render_receipt = {
        "schema_version": "frozen_nuisance_sensitivity_visual.v1",
        "artifact_type": "frozen_nuisance_sensitivity_worst_event_visual",
        "role_key": role_key,
        "events": events,
        "raw_receipt": _record(receipt_path),
        "raw_arrays": _record(arrays_path),
        "manifest": manifest_record,
        "checkpoint": checkpoint_record,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": True,
        "training_or_weight_update_performed": False,
        "visual": _record(image_path),
    }
    render_path = output_directory / "visual_receipt.json"
    render_path.write_text(json.dumps(render_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return render_receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role-key", required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = render(
        manifest_path=args.manifest,
        role_key=args.role_key,
        raw_directory=args.raw_directory,
        output_directory=args.output_directory,
    )
    print(json.dumps({"visual": receipt["visual"], "events": receipt["events"]}, sort_keys=True))


if __name__ == "__main__":
    main()
