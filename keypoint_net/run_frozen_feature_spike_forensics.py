"""Run frozen spike-versus-control encoder-feature forensics over all 24 roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

try:
    from .frozen_feature_forensics import (
        coordinate_on_mask,
        feature_match_metrics,
        select_low_wobble_centres,
    )
    from .frozen_wobble_forensics import rotate_points
    from .plot_frozen_wobble_pair import _load_run
    from .run_frozen_wobble_forensics import (
        COORDINATE_CONSISTENCY_TOLERANCE,
        _construct_frozen_model,
        _load_corpus,
        _same_fd_checkpoint_load,
        _state_sha256,
    )
    from .summarize_frozen_wobble_matrix import _all_expected_roles, _content_hash, _role_key
except ImportError:
    from frozen_feature_forensics import (  # type: ignore
        coordinate_on_mask,
        feature_match_metrics,
        select_low_wobble_centres,
    )
    from frozen_wobble_forensics import rotate_points  # type: ignore
    from plot_frozen_wobble_pair import _load_run  # type: ignore
    from run_frozen_wobble_forensics import (  # type: ignore
        COORDINATE_CONSISTENCY_TOLERANCE,
        _construct_frozen_model,
        _load_corpus,
        _same_fd_checkpoint_load,
        _state_sha256,
    )
    from summarize_frozen_wobble_matrix import _all_expected_roles, _content_hash, _role_key  # type: ignore


SCHEMA_VERSION = "frozen_feature_spike_forensics.v1_1"
EXPECTED_MATRIX_SUMMARY_SCHEMA = "frozen_wobble_complete_matrix_summary.v1"
EXPECTED_FRAMES = 180
EXPECTED_CHANNELS = 10
FEATURE_CHANNELS = 128
FEATURE_SIZE = 64
IMPLEMENTATION_SOURCES = (
    "keypoint_net/run_frozen_feature_spike_forensics.py",
    "keypoint_net/frozen_feature_forensics.py",
    "keypoint_net/run_frozen_wobble_forensics.py",
    "keypoint_net/frozen_wobble_forensics.py",
    "keypoint_net/model.py",
)


class FeatureSpikeRunError(ValueError):
    """Fail-closed source, lineage, or feature-semantic error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureSpikeRunError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "absolute_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _implementation_binding(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *IMPLEMENTATION_SOURCES],
        cwd=repo_root,
        check=False,
    )
    _require(clean.returncode == 0, "feature-forensic implementation differs from its recorded HEAD")
    return {
        "implementation_head": head,
        "implementation_sources": {
            relative: _file_record(repo_root / relative) for relative in IMPLEMENTATION_SOURCES
        },
    }


def _cell_to_coordinate(x_cell: float, y_cell: float) -> np.ndarray:
    return np.asarray((-1.0 + 2.0 * x_cell / 63.0, -1.0 + 2.0 * y_cell / 63.0))


def _worst_channel(report: Mapping[str, Any]) -> int:
    rows = report["soft_coordinate_metrics"]["full_nonseam"]["noncyclic_second_difference"]["per_channel"]
    return int(max(range(EXPECTED_CHANNELS), key=lambda channel: rows[channel]["q99"]))


def _selected_centres(
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    channel: int,
) -> dict[str, list[int]]:
    spikes = [int(row["centre_frame_index"]) for row in report["spike_frames"]["soft"][channel]]
    controls = select_low_wobble_centres(
        arrays["soft_canonical_coordinate"][:, channel],
        spikes,
        count=len(spikes),
        exclusion_radius=2,
    )
    return {"spike": spikes, "low_wobble_control": controls}


def _infer_selected_features(
    model: torch.nn.Module,
    images: list[np.ndarray],
    frames: list[int],
    expected_points: np.ndarray,
    expected_logits: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    _require(not model.training and all(not parameter.requires_grad for parameter in model.parameters()), "model is not frozen")
    before = _state_sha256(model)
    means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    features: dict[int, np.ndarray] = {}
    maximum_coordinate_error = 0.0
    maximum_logit_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(frames), 8):
            selected = frames[start : start + 8]
            array = np.stack([images[frame] for frame in selected])
            batch = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
            batch = batch.sub_(means).div_(stds)
            flattened, logits, encoder = model.extractor(batch, return_descriptor_features=True)
            points = flattened.reshape(len(selected), EXPECTED_CHANNELS, 2).cpu().numpy()
            logits_numpy = logits.cpu().numpy()
            maximum_coordinate_error = max(
                maximum_coordinate_error,
                float(np.max(np.abs(points - expected_points[np.asarray(selected)]))),
            )
            maximum_logit_error = max(
                maximum_logit_error,
                float(np.max(np.abs(logits_numpy - expected_logits[np.asarray(selected)]))),
            )
            encoder_numpy = encoder.cpu().numpy()
            _require(
                encoder_numpy.shape[1:] == (FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
                "encoder feature shape differs from retained descriptor field",
            )
            for batch_index, frame in enumerate(selected):
                features[frame] = encoder_numpy[batch_index].copy()
    _require(maximum_coordinate_error <= COORDINATE_CONSISTENCY_TOLERANCE, "selected feature pass coordinates differ")
    _require(maximum_logit_error <= 1e-6, "selected feature pass logits differ from frozen forensic arrays")
    after = _state_sha256(model)
    _require(before == after, "model state changed during feature inference")
    return features, {
        "selected_frames": frames,
        "selected_frame_count": len(frames),
        "batch_size": 8,
        "device": "cpu",
        "inference_mode": True,
        "optimizer_constructed": False,
        "training_or_weight_update_performed": False,
        "maximum_coordinate_error_against_frozen_run": maximum_coordinate_error,
        "coordinate_tolerance": COORDINATE_CONSISTENCY_TOLERANCE,
        "maximum_logit_error_against_frozen_run": maximum_logit_error,
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "model_state_unchanged": True,
    }


def _one_basis_match(
    arrays: Mapping[str, np.ndarray],
    features: Mapping[int, np.ndarray],
    masks: np.ndarray,
    *,
    source_frame: int,
    target_frame: int,
    channel: int,
    basis: str,
) -> tuple[dict[str, Any], np.ndarray]:
    if basis == "hard_peak":
        source_coordinate = arrays["hard_coordinate"][source_frame, channel]
        detector_coordinate = arrays["hard_coordinate"][target_frame, channel]
    elif basis == "soft_coordinate":
        source_coordinate = arrays["soft_coordinate"][source_frame, channel]
        detector_coordinate = arrays["soft_coordinate"][target_frame, channel]
    else:
        raise FeatureSpikeRunError("unknown feature coordinate basis")
    physical_target = rotate_points(source_coordinate[None], 2.0)[0]
    second_peak = _cell_to_coordinate(
        arrays["separated_peak_r4_second_peak_x_cell"][target_frame, channel],
        arrays["separated_peak_r4_second_peak_y_cell"][target_frame, channel],
    )
    metrics, correlation = feature_match_metrics(
        features[source_frame],
        features[target_frame],
        masks[target_frame],
        source_coordinate=source_coordinate,
        physical_target_coordinate=physical_target,
        detector_target_coordinate=detector_coordinate,
        separated_second_peak_coordinate=second_peak,
    )
    source_on_object = coordinate_on_mask(masks[source_frame], source_coordinate)
    physical_target_on_object = coordinate_on_mask(masks[target_frame], physical_target)
    detector_target_on_object = coordinate_on_mask(masks[target_frame], detector_coordinate)
    return {
        "basis": basis,
        "source_coordinate": [float(value) for value in source_coordinate],
        "physical_target_coordinate": [float(value) for value in physical_target],
        "detector_target_coordinate": [float(value) for value in detector_coordinate],
        "separated_second_peak_coordinate": [float(value) for value in second_peak],
        "source_on_object": source_on_object,
        "physical_target_on_object": physical_target_on_object,
        "detector_target_on_object": detector_target_on_object,
        "grounded_physical_edge": source_on_object and physical_target_on_object,
        **metrics,
    }, correlation


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.isfinite(array).all(), "cannot summarize empty or non-finite values")
    return {
        "n": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def _aggregate_edges(edges: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "physical_target_similarity",
        "detector_target_similarity",
        "detector_minus_physical_similarity",
        "physical_target_mask_percentile",
        "mask_cells_higher_than_physical_fraction",
        "object_best_minus_physical_similarity",
        "object_argmax_distance_to_physical_cells",
        "detector_distance_to_physical_cells",
        "object_top1_top2_margin",
    )
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        for basis in ("hard_peak", "soft_coordinate"):
            key = (
                edge["task"],
                edge["arm"],
                edge["checkpoint_role"],
                edge["condition"],
                basis,
            )
            grouped[key].append(edge[basis])
    output = []
    for key, rows in sorted(grouped.items()):
        task, arm, role, condition, basis = key
        output.append(
            {
                "task": task,
                "arm": arm,
                "checkpoint_role": role,
                "condition": condition,
                "basis": basis,
                "metrics": {
                    name: _numeric_summary([float(row[name]) for row in rows]) for name in metric_names
                },
                "descriptive_not_inferential": True,
            }
        )
    overall = []
    for subset in ("all_selected_edges", "grounded_physical_edges"):
        for condition in ("spike", "low_wobble_control"):
            for basis in ("hard_peak", "soft_coordinate"):
                rows = [
                    edge[basis]
                    for edge in edges
                    if edge["condition"] == condition
                    and (subset == "all_selected_edges" or edge[basis]["grounded_physical_edge"])
                ]
                overall.append(
                    {
                        "subset": subset,
                        "condition": condition,
                        "basis": basis,
                        "selected_edge_count": len(rows),
                        "metrics": {
                            name: _numeric_summary([float(row[name]) for row in rows])
                            for name in metric_names
                        }
                        if rows
                        else None,
                        "descriptive_not_inferential": True,
                    }
                )
    return {"task_arm_role_condition": output, "overall_condition_and_grounding": overall}


def _image(data_root: Path, frame: int) -> np.ndarray:
    path = data_root / "train" / "engineers_hammer_vray" / "frames" / "a" / f"img_{frame:04d}.png"
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _visualize_event(
    role_key: str,
    visual_reason: str,
    edge: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    features: Mapping[int, np.ndarray],
    masks: np.ndarray,
    data_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    source_frame = int(edge["source_frame"])
    target_frame = int(edge["target_frame"])
    channel = int(edge["channel"])
    hard, correlation = _one_basis_match(
        arrays,
        features,
        masks,
        source_frame=source_frame,
        target_frame=target_frame,
        channel=channel,
        basis="hard_peak",
    )
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    axes[0].imshow(_image(data_root, source_frame), extent=(-1, 1, 1, -1))
    axes[0].scatter(*hard["source_coordinate"], s=70, facecolors="none", edgecolors="#00FFFF", lw=1.8)
    axes[0].set_title(f"source RGB frame {source_frame}")
    axes[1].imshow(_image(data_root, target_frame), extent=(-1, 1, 1, -1))
    axes[1].scatter(*hard["physical_target_coordinate"], s=70, facecolors="none", edgecolors="#00FF00", lw=1.8, label="physical +2°")
    axes[1].scatter(*hard["detector_target_coordinate"], s=65, marker="x", color="#00FFFF", lw=1.8, label="detector")
    axes[1].scatter(*hard["object_argmax_coordinate"], s=45, marker="D", facecolors="none", edgecolors="#FF00FF", lw=1.5, label="feature argmax")
    axes[1].set_title(f"target RGB frame {target_frame}")
    axes[1].legend(loc="best", fontsize=7)
    axes[2].imshow(arrays["logits"][source_frame, channel], cmap="magma", extent=(-1, 1, 1, -1))
    axes[2].scatter(*hard["source_coordinate"], s=55, facecolors="none", edgecolors="#00FFFF", lw=1.5)
    axes[2].set_title(f"source KP{channel} heatmap")
    axes[3].imshow(arrays["logits"][target_frame, channel], cmap="magma", extent=(-1, 1, 1, -1))
    axes[3].scatter(*hard["physical_target_coordinate"], s=55, facecolors="none", edgecolors="#00FF00", lw=1.5)
    axes[3].scatter(*hard["detector_target_coordinate"], s=50, marker="x", color="#00FFFF", lw=1.5)
    axes[3].set_title(f"target KP{channel} heatmap")
    image = axes[4].imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, extent=(-1, 1, 1, -1))
    axes[4].scatter(*hard["physical_target_coordinate"], s=55, facecolors="none", edgecolors="#00FF00", lw=1.5)
    axes[4].scatter(*hard["detector_target_coordinate"], s=50, marker="x", color="#00FFFF", lw=1.5)
    axes[4].scatter(*hard["object_argmax_coordinate"], s=40, marker="D", facecolors="none", edgecolors="#FF00FF", lw=1.3)
    axes[4].set_title("source-feature cosine map")
    fig.colorbar(image, ax=axes[4], fraction=0.046, pad=0.03)
    for ax in axes:
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1)
        ax.axis("off")
    fig.suptitle(
        f"{role_key}: {visual_reason}\n"
        f"{edge['condition']} {edge['direction']} for KP{channel}\n"
        f"physical percentile={hard['physical_target_mask_percentile']:.3f}; "
        f"detector-physical cosine={hard['detector_minus_physical_similarity']:+.3f}; "
        f"detector distance={hard['detector_distance_to_physical_cells']:.2f} cells; "
        f"grounded={hard['grounded_physical_edge']}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return _file_record(output_path)


def run(
    matrix_root: Path,
    matrix_summary_path: Path,
    expected_matrix_summary_sha256: str,
    data_root: Path,
    output_dir: Path,
    role_key_filter: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    implementation = _implementation_binding(repo_root)
    summary_record = _file_record(matrix_summary_path)
    _require(summary_record["sha256"] == expected_matrix_summary_sha256, "matrix summary SHA-256 differs")
    matrix_summary = json.loads(matrix_summary_path.read_text(encoding="utf-8"))
    _require(matrix_summary["schema_version"] == EXPECTED_MATRIX_SUMMARY_SCHEMA, "matrix summary schema differs")
    _require(matrix_summary["content_hash_sha256"] == _content_hash(matrix_summary), "matrix summary content hash differs")
    _require(matrix_summary["inventory"]["matrix_complete"] is True, "matrix summary is incomplete")
    expected_roles = _all_expected_roles()
    if role_key_filter is not None:
        _require(role_key_filter in expected_roles, "requested smoke role is not one of the 24 frozen roles")
        expected_roles = {role_key_filter}

    images, masks, theta, frame_indices, corpus = _load_corpus(data_root)
    _require(np.array_equal(theta, np.arange(EXPECTED_FRAMES) * 2.0), "corpus theta differs")
    _require(frame_indices == list(range(EXPECTED_FRAMES)), "corpus frame order differs")
    output_dir.mkdir(parents=True, exist_ok=False)
    visual_root = output_dir / "correlation_montages"
    visual_root.mkdir()

    edges: list[dict[str, Any]] = []
    role_records = []
    visuals = {}
    observed_roles = set()
    for report_path in sorted(matrix_root.resolve(strict=True).glob("*/forensic_report.json")):
        if role_key_filter is not None and report_path.parent.name != role_key_filter:
            continue
        report, arrays = _load_run(report_path.parent)
        role_key = _role_key(report["recipe"], report["arm"], int(report["seed"]), report["checkpoint_role"])
        _require(role_key == report_path.parent.name, "role directory differs from report")
        observed_roles.add(role_key)
        worst = _worst_channel(report)
        channel_labels: dict[int, list[str]] = defaultdict(list)
        channel_labels[2].append("kp2")
        channel_labels[worst].append("worst_second_difference_q99")
        centres = {
            channel: _selected_centres(report, arrays, channel) for channel in sorted(channel_labels)
        }
        selected_frames = sorted(
            {
                frame
                for by_condition in centres.values()
                for condition_centres in by_condition.values()
                for centre in condition_centres
                for frame in (centre - 1, centre, centre + 1)
            }
        )
        checkpoint = report["bindings"]["checkpoint"]
        payload, live_checkpoint = _same_fd_checkpoint_load(
            checkpoint["absolute_path"],
            expected_sha256=checkpoint["sha256"],
            expected_size=checkpoint["size_bytes"],
        )
        model, _ = _construct_frozen_model(
            payload,
            cell_id=report["cell_id"],
            checkpoint_role=report["checkpoint_role"],
            expected_epoch=int(report["checkpoint_epoch"]),
        )
        _require(_state_sha256(model) == report["inference"]["model_state_sha256_before"], "model state differs from frozen run")
        features, inference = _infer_selected_features(
            model,
            images,
            selected_frames,
            arrays["soft_coordinate"],
            arrays["logits"],
        )
        role_edges = []
        for channel, by_condition in centres.items():
            for condition, condition_centres in by_condition.items():
                for centre in condition_centres:
                    for direction, source_frame, target_frame in (
                        ("incoming", centre - 1, centre),
                        ("outgoing", centre, centre + 1),
                    ):
                        row: dict[str, Any] = {
                            "role_key": role_key,
                            "task": report["recipe"],
                            "arm": report["arm"],
                            "seed": int(report["seed"]),
                            "checkpoint_role": report["checkpoint_role"],
                            "checkpoint_epoch": int(report["checkpoint_epoch"]),
                            "channel": int(channel),
                            "channel_selection_labels": channel_labels[channel],
                            "condition": condition,
                            "centre_frame": int(centre),
                            "direction": direction,
                            "source_frame": int(source_frame),
                            "target_frame": int(target_frame),
                        }
                        for basis in ("hard_peak", "soft_coordinate"):
                            match, _ = _one_basis_match(
                                arrays,
                                features,
                                masks,
                                source_frame=source_frame,
                                target_frame=target_frame,
                                channel=channel,
                                basis=basis,
                            )
                            row[basis] = match
                        edges.append(row)
                        role_edges.append(row)
        grounded_spike_edges = [
            row
            for row in role_edges
            if row["condition"] == "spike" and row["hard_peak"]["grounded_physical_edge"]
        ]
        visual_candidates = grounded_spike_edges or [
            row for row in role_edges if row["condition"] == "spike"
        ]
        largest_detector_jump_edge = max(
            visual_candidates,
            key=lambda row: row["hard_peak"]["detector_distance_to_physical_cells"],
        )
        farthest_feature_argmax_edge = max(
            visual_candidates,
            key=lambda row: row["hard_peak"]["object_argmax_distance_to_physical_cells"],
        )
        visual_edges = {
            "largest_grounded_detector_jump": largest_detector_jump_edge,
            "farthest_grounded_feature_argmax": farthest_feature_argmax_edge,
        }
        visuals[role_key] = {}
        for visual_reason, visual_edge in visual_edges.items():
            visual_path = visual_root / f"{role_key}__{visual_reason}.png"
            visuals[role_key][visual_reason] = _visualize_event(
                role_key,
                visual_reason.replace("_", " "),
                visual_edge,
                arrays,
                features,
                masks,
                data_root,
                visual_path,
            )
        role_records.append(
            {
                "role_key": role_key,
                "worst_channel": worst,
                "selected_channels": {str(channel): labels for channel, labels in sorted(channel_labels.items())},
                "selected_centres": {str(channel): value for channel, value in centres.items()},
                "edge_count": len(role_edges),
                "checkpoint": live_checkpoint,
                "forensic_report": _file_record(report_path),
                "raw_arrays": _file_record(report_path.parent / "raw_forensic_arrays.npz"),
                "inference": inference,
                "visual_events": {
                    visual_reason: {
                        key: visual_edge[key]
                        for key in (
                            "channel",
                            "condition",
                            "centre_frame",
                            "direction",
                            "source_frame",
                            "target_frame",
                            "hard_peak",
                        )
                    }
                    for visual_reason, visual_edge in visual_edges.items()
                },
            }
        )
        print(
            f"completed {len(role_records):02d}/{len(expected_roles):02d} {role_key}: "
            f"{len(role_edges)} selected adjacent edges, no training",
            flush=True,
        )
    _require(observed_roles == expected_roles, "feature forensics does not contain its exact requested role set")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": (
            "source_bound_complete_frozen_feature_spike_forensics"
            if role_key_filter is None
            else "source_bound_single_role_frozen_feature_spike_smoke"
        ),
        "training_or_weight_update_performed": False,
        "implementation": implementation,
        "bindings": {
            "matrix_summary": summary_record,
            "expected_matrix_summary_sha256": expected_matrix_summary_sha256,
            "matrix_root": str(matrix_root.resolve(strict=True)),
            "corpus": corpus,
        },
        "feature_field": {
            "tensor": "post-ReLU encoder output before 1x1 heatmap head",
            "shape_per_frame": [FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE],
            "sampling": "bilinear, endpoint-aligned, align_corners=True, L2-normalized",
            "correlation": "cosine similarity against every target feature cell",
            "rank_domain": "endpoint-aligned feature cells whose corresponding pixel is inside the bound hammer mask",
            "new_or_replacement_descriptor_used": False,
        },
        "selection": {
            "role_scope": "all 24 frozen roles" if role_key_filter is None else role_key_filter,
            "channels": "blue KP2 and largest canonical second-difference-q99 channel; evaluated once if identical",
            "spikes": "five pre-existing largest non-seam second-difference centres",
            "controls": "five lowest centres after excluding every spike centre +/-2 frames; magnitude then frame-index tie-break",
            "edges_per_centre": ["incoming centre-1 to centre", "outgoing centre to centre+1"],
        },
        "role_records": role_records,
        "edges": edges,
        "summaries": _aggregate_edges(edges),
        "visuals": visuals,
        "statistical_scope": {
            "statistics": "complete selected-edge values plus descriptive mean, median, q90, q99, extrema",
            "sample_unit": "selected adjacent edge within one correlated orbit/checkpoint role",
            "descriptive_not_inferential": True,
            "correlation_caveat": "incoming/outgoing edges overlap, spike/control selections share orbits, and checkpoint roles share training runs",
            "error_bars_or_uncertainty_bands": "none",
        },
    }
    result["content_hash_sha256"] = _content_hash(result)
    result_path = output_dir / "FROZEN_FEATURE_SPIKE_FORENSICS.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "result": _file_record(result_path),
        "checkpoint_roles": len(role_records),
        "selected_edges": len(edges),
        "correlation_montages": sum(len(role_visuals) for role_visuals in visuals.values()),
        "training_or_weight_update_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--expected-matrix-summary-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--role-key",
        help="run one exact frozen role as a production-path smoke; omit for the complete 24-role matrix",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.matrix_root,
                args.matrix_summary,
                args.expected_matrix_summary_sha256,
                args.data_root,
                args.output_dir,
                args.role_key,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
