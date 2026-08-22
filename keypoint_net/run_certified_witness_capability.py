"""Train and evaluate the exact-ten-track supervised capability gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, __version__ as pillow_version
from torch.utils.data import DataLoader, Dataset

from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    CapabilityContractError,
    bilinear_planted_logits,
    dense_heatmap_cross_entropy,
    evaluate_predictions,
    evaluation_score,
    file_record,
    model_state_sha256,
    nearest_r64_grid_prediction,
    normalized_to_pixel,
    require,
    sha256_file,
)
from model import KeypointExtractor, spatial_softmax


MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
COLORS = (
    "#ff2d2d",
    "#00b84a",
    "#2878ff",
    "#e6c700",
    "#d62dff",
    "#00bfc7",
    "#ff8c1a",
    "#8b4bd9",
    "#48a832",
    "#244a9b",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _summary(values: np.ndarray) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    require(vector.size > 0 and bool(np.isfinite(vector).all()), "invalid summary vector")
    return {
        "n": int(vector.size),
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
        "q90": float(np.quantile(vector, 0.9)),
        "maximum": float(vector.max()),
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class BoundCapabilityDataset(Dataset):
    def __init__(
        self,
        images_uint8: np.ndarray,
        masks: np.ndarray,
        targets_normalized: np.ndarray,
        frame_indices: np.ndarray,
    ) -> None:
        self.images = np.asarray(images_uint8, dtype=np.uint8)
        self.masks = np.asarray(masks, dtype=bool)
        self.targets = np.asarray(targets_normalized, dtype=np.float32)
        self.frames = np.asarray(frame_indices, dtype=np.int64)
        require(self.images.shape == (len(self.frames), 512, 512, 3), "unexpected image array shape")
        require(self.masks.shape == (len(self.frames), 512, 512), "unexpected mask array shape")
        require(self.targets.shape == (len(self.frames), EXPECTED_WITNESSES, 2), "unexpected target array shape")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[item].copy()).permute(2, 0, 1).float() / 255.0
        return {
            "image": (image - MEAN) / STD,
            "target": torch.from_numpy(self.targets[item].copy()),
            "mask": torch.from_numpy(self.masks[item].copy()),
            "frame": torch.tensor(int(self.frames[item]), dtype=torch.int64),
        }


def _verify_sources(manifest: dict[str, Any], repo_root: Path) -> None:
    for relative, record in manifest["implementation_sources"].items():
        path = repo_root / relative
        require(path.is_file(), f"implementation source missing: {relative}")
        require(sha256_file(path) == record["sha256"], f"implementation source SHA-256 differs: {relative}")


def _load_bound_inputs(
    manifest_path: Path,
    tracks_path: Path,
    data_object_root: Path,
    repo_root: Path,
    expected_manifest_sha256: str,
    expected_tracks_sha256: str,
    frame_limit: int,
) -> tuple[dict[str, Any], BoundCapabilityDataset, np.ndarray, np.ndarray, dict[str, Any]]:
    require(sha256_file(manifest_path) == expected_manifest_sha256, "manifest SHA-256 differs from command lock")
    require(sha256_file(tracks_path) == expected_tracks_sha256, "tracks SHA-256 differs from command lock")
    manifest = _load_json(manifest_path)
    require(manifest["schema_version"] == "certified_witness_supervised_capability_manifest.v1", "manifest schema differs")
    require(manifest["portable_tracks"]["sha256"] == expected_tracks_sha256, "manifest tracks binding differs")
    require(manifest["preservation_phase_authorized"] is False, "manifest unexpectedly authorizes preservation")
    _verify_sources(manifest, repo_root)

    with np.load(tracks_path) as arrays:
        frame_index = np.asarray(arrays["frame_index"], dtype=np.int64)
        witness_id = np.asarray(arrays["witness_id"], dtype=np.int64)
        target_px = np.asarray(arrays["target_coordinate_px"], dtype=np.float64)
        target_normalized = np.asarray(arrays["target_coordinate_normalized"], dtype=np.float32)
        physical_valid = np.asarray(arrays["physical_valid"], dtype=bool)
        target_on_object_recorded = np.asarray(arrays["target_on_object"], dtype=bool)
    require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), "tracks frame index differs")
    require(tuple(witness_id.tolist()) == EXPECTED_WITNESS_IDS, "tracks witness identity/order differs")
    require(target_px.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "tracks target shape differs")
    require(bool(physical_valid.all()), "tracks contain an invalid physical target")
    require(bool(target_on_object_recorded.all()), "tracks record an off-object target")
    require(1 <= frame_limit <= EXPECTED_FRAMES, "frame_limit outside 1..180")

    images = np.empty((frame_limit, 512, 512, 3), dtype=np.uint8)
    masks = np.empty((frame_limit, 512, 512), dtype=bool)
    verified_rows: list[dict[str, Any]] = []
    for row in manifest["dataset"]["frames"]:
        frame = int(row["frame_index"])
        if frame >= frame_limit:
            continue
        image_path = data_object_root / row["image_relpath"]
        mask_path = data_object_root / row["mask_relpath"]
        require(sha256_file(image_path) == row["image_sha256"], f"frame {frame} RGB SHA-256 differs")
        require(sha256_file(mask_path) == row["mask_sha256"], f"frame {frame} mask SHA-256 differs")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        require(image.shape == (512, 512, 3), f"frame {frame} RGB shape differs")
        require(mask.shape == (512, 512), f"frame {frame} mask shape differs")
        images[frame] = image
        masks[frame] = mask
        verified_rows.append({"frame_index": frame, "image_sha256": row["image_sha256"], "mask_sha256": row["mask_sha256"]})
    require(len(verified_rows) == frame_limit, "verified frame inventory incomplete")

    rounded_target = np.rint(target_px[:frame_limit]).astype(np.int64)
    target_on_object = masks[
        np.arange(frame_limit)[:, None],
        rounded_target[..., 1],
        rounded_target[..., 0],
    ]
    require(bool(target_on_object.all()), "a bound target is off object in loaded masks")
    require(np.array_equal(target_on_object, target_on_object_recorded[:frame_limit]), "target mask replay differs")
    dataset = BoundCapabilityDataset(
        images,
        masks,
        target_normalized[:frame_limit],
        frame_index[:frame_limit],
    )

    # Frame-keyed content must be identical when traversed in the opposite order.
    forward_hashes: dict[int, str] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        payload = (
            item["image"].numpy().tobytes()
            + item["target"].numpy().tobytes()
            + item["mask"].numpy().tobytes()
        )
        forward_hashes[int(item["frame"])] = _sha256_bytes(payload)
    reverse_hashes: dict[int, str] = {}
    for index in reversed(range(len(dataset))):
        item = dataset[index]
        payload = (
            item["image"].numpy().tobytes()
            + item["target"].numpy().tobytes()
            + item["mask"].numpy().tobytes()
        )
        reverse_hashes[int(item["frame"])] = _sha256_bytes(payload)
    require(forward_hashes == reverse_hashes, "forward/reverse loader replay differs")

    planted_logits = bilinear_planted_logits(target_normalized[:frame_limit])
    planted_prediction = normalized_to_pixel(spatial_softmax(planted_logits, temperature=1.0).numpy())
    planted_report, _ = evaluate_predictions(
        planted_prediction,
        target_px[:frame_limit],
        masks,
    )
    require(planted_report["strict_capability_pass"] is True, "planted-softargmax evaluator falsifier failed")
    hard_grid_prediction = nearest_r64_grid_prediction(target_px[:frame_limit])
    hard_grid_report, _ = evaluate_predictions(
        hard_grid_prediction,
        target_px[:frame_limit],
        masks,
    )
    hard_grid_violations = hard_grid_report["violations"]
    require(hard_grid_violations["outside_half_cell_count"] == 0, "nearest-grid localization geometry failed")
    require(hard_grid_violations["wrong_identity_count"] == 0, "nearest-grid identity geometry failed")
    require(hard_grid_violations["collapsed_pair_count"] == 0, "nearest-grid distinctness geometry failed")
    controls = {
        "manifest_sha256_verified": True,
        "tracks_sha256_verified": True,
        "implementation_source_sha256_verified": True,
        "rgb_and_mask_sha256_verified_count": frame_limit,
        "all_targets_physical_valid": True,
        "all_targets_on_object": True,
        "forward_reverse_loader_replay_exact": True,
        "planted_bilinear_logit_softargmax_evaluator_control": planted_report,
        "nearest_hard_grid_diagnostic": {
            "report": hard_grid_report,
            "interpretation": "hard cells pass localization, identity, and distinctness; off-mask cells are allowed only in this diagnostic because the trained readout is continuous soft-argmax",
        },
    }
    return manifest, dataset, target_px[:frame_limit], masks, controls


@torch.no_grad()
def _evaluate_model(
    model: KeypointExtractor,
    dataset: BoundCapabilityDataset,
    target_px: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    entropy: list[np.ndarray] = []
    peak_probability: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for batch in loader:
        flat, logits = model(batch["image"].to(device))
        coordinates = flat.view(-1, EXPECTED_WITNESSES, 2)
        probability = torch.softmax(logits.flatten(-2), dim=-1)
        predictions.append(coordinates.cpu().numpy())
        entropy.append((-(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)).cpu().numpy())
        peak_probability.append(probability.max(dim=-1).values.cpu().numpy())
        frames.append(batch["frame"].numpy())
    prediction_normalized = np.concatenate(predictions)
    frame_index = np.concatenate(frames)
    order = np.argsort(frame_index)
    require(np.array_equal(frame_index[order], np.arange(len(dataset))), "evaluation frame order incomplete")
    prediction_normalized = prediction_normalized[order]
    entropy_array = np.concatenate(entropy)[order]
    peak_array = np.concatenate(peak_probability)[order]
    prediction_px = normalized_to_pixel(prediction_normalized)
    report, derived = evaluate_predictions(prediction_px, target_px, masks)
    report["heatmap_entropy"] = _summary(entropy_array)
    report["heatmap_peak_probability"] = _summary(peak_array)
    derived.update(
        {
            "frame_index": np.arange(len(dataset), dtype=np.int64),
            "prediction_coordinate_normalized": prediction_normalized,
            "prediction_coordinate_px": prediction_px,
            "target_coordinate_px": target_px,
            "heatmap_entropy": entropy_array,
            "heatmap_peak_probability": peak_array,
        }
    )
    return report, derived


def _save_worst_montage(
    images: np.ndarray,
    prediction_px: np.ndarray,
    target_px: np.ndarray,
    material_error_px: np.ndarray,
    output_path: Path,
) -> None:
    worst_flat = np.argsort(material_error_px.reshape(-1), kind="stable")[-6:][::-1]
    tiles: list[Image.Image] = []
    for flat_index in worst_flat:
        frame, focus_channel = np.unravel_index(int(flat_index), material_error_px.shape)
        tile = Image.fromarray(images[frame]).convert("RGB")
        draw = ImageDraw.Draw(tile)
        for channel in range(EXPECTED_WITNESSES):
            color = COLORS[channel]
            tx, ty = map(float, target_px[frame, channel])
            px, py = map(float, prediction_px[frame, channel])
            draw.line((tx - 5, ty, tx + 5, ty), fill=color, width=2)
            draw.line((tx, ty - 5, tx, ty + 5), fill=color, width=2)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline=color, width=2)
            if channel == focus_channel:
                draw.line((tx, ty, px, py), fill="#ffffff", width=3)
        label = f"frame {frame} witness {EXPECTED_WITNESS_IDS[focus_channel]} error {material_error_px[frame, focus_channel]:.2f}px"
        draw.rectangle((0, 0, 512, 24), fill="#000000")
        draw.text((6, 5), label, fill="#ffffff")
        tile.thumbnail((384, 384), Image.Resampling.LANCZOS)
        tiles.append(tile)
    canvas = Image.new("RGB", (384 * 3, 384 * 2), color="#202020")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 3) * 384, (index // 3) * 384))
    canvas.save(output_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh run")
    require(args.max_updates >= 1, "max_updates must be positive")
    require(args.eval_every >= 1, "eval_every must be positive")
    require(args.batch_size >= 1, "batch_size must be positive")
    manifest, dataset, target_px, masks, controls = _load_bound_inputs(
        args.manifest,
        args.tracks,
        args.data_object_root,
        args.repo_root,
        args.expected_manifest_sha256,
        args.expected_tracks_sha256,
        args.frame_limit,
    )
    args.output_dir.mkdir(parents=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")

    model = KeypointExtractor(
        num_keypoints=EXPECTED_WITNESSES,
        base_channels=32,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=64,
        true_quarter_res=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    initial_state_hash = model_state_sha256(model)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    config = {
        "schema_version": "certified_witness_supervised_capability_run_config.v1",
        "manifest": file_record(args.manifest),
        "tracks": file_record(args.tracks),
        "manifest_implementation_head": manifest["implementation_head"],
        "seed": args.seed,
        "device": str(device),
        "frame_limit": args.frame_limit,
        "scientific_full_orbit_run": args.frame_limit == EXPECTED_FRAMES,
        "max_updates": args.max_updates,
        "eval_every": args.eval_every,
        "batch_size": args.batch_size,
        "loss": "gaussian_target_distribution_cross_entropy_only",
        "sigma_input_px": 8.0,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "augmentation": "none",
        "preservation_loss_or_intervention": "none",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": pillow_version,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "semantic_controls.json").write_text(json.dumps(controls, indent=2, sort_keys=True) + "\n")

    history: list[dict[str, Any]] = []
    best_score: tuple[int, int, int, int, float, float] | None = None
    best_checkpoint_path: Path | None = None
    parameter_change_proved = False
    update = 0
    epoch = 0
    start = time.perf_counter()
    stop = False
    while update < args.max_updates and not stop:
        epoch += 1
        model.train()
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            _, logits = model(images)
            loss = dense_heatmap_cross_entropy(logits, targets, sigma_input_px=8.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update += 1
            if update == 1:
                parameter_change_proved = model_state_sha256(model) != initial_state_hash
                require(parameter_change_proved, "optimizer step did not change model state")
            should_evaluate = update == 1 or update % args.eval_every == 0 or update == args.max_updates
            if should_evaluate:
                report, _ = _evaluate_model(
                    model,
                    dataset,
                    target_px,
                    masks,
                    device,
                    args.batch_size,
                )
                score = evaluation_score(report)
                row = {
                    "update": update,
                    "epoch": epoch,
                    "train_batch_loss": float(loss.detach().cpu()),
                    "strict_capability_pass": report["strict_capability_pass"],
                    "outside_half_cell_count": report["violations"]["outside_half_cell_count"],
                    "wrong_identity_count": report["violations"]["wrong_identity_count"],
                    "collapsed_pair_count": report["violations"]["collapsed_pair_count"],
                    "off_object_count": report["violations"]["off_object_count"],
                    "median_material_error_px": report["material_error_px"]["median"],
                    "maximum_material_error_px": report["material_error_px"]["maximum"],
                    "within_half_cell_rate": report["within_half_cell_rate"],
                    "on_object_rate": report["on_object_rate"],
                    "identity_assignment_rate": report["identity_assignment_rate"],
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                if best_score is None or score < best_score:
                    best_score = score
                    checkpoint_path = checkpoint_dir / f"update_{update:06d}.pt"
                    checkpoint_state_hash = model_state_sha256(model)
                    torch.save(
                        {
                            "schema_version": "certified_witness_supervised_capability_checkpoint.v1",
                            "extractor_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "update": update,
                            "epoch": epoch,
                            "seed": args.seed,
                            "score": list(score),
                            "model_state_sha256": checkpoint_state_hash,
                            "config": config,
                        },
                        checkpoint_path,
                    )
                    best_checkpoint_path = checkpoint_path
                if report["strict_capability_pass"] and args.stop_on_pass:
                    stop = True
                if not stop:
                    model.train()
            if update >= args.max_updates or stop:
                break

    require(best_checkpoint_path is not None and best_score is not None, "no checkpoint was selected")
    best_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["extractor_state_dict"], strict=True)
    loaded_state_hash = model_state_sha256(model)
    require(
        loaded_state_hash == best_payload["model_state_sha256"],
        "checkpoint round-trip state hash differs",
    )
    final_report, derived = _evaluate_model(
        model,
        dataset,
        target_px,
        masks,
        device,
        args.batch_size,
    )
    require(evaluation_score(final_report) == best_score, "reloaded checkpoint score differs")
    best_copy = args.output_dir / "best_model.pt"
    shutil.copy2(best_checkpoint_path, best_copy)

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    predictions_path = args.output_dir / "predictions.npz"
    np.savez_compressed(predictions_path, **derived)
    montage_path = args.output_dir / "worst_events.png"
    _save_worst_montage(
        dataset.images,
        derived["prediction_coordinate_px"],
        derived["target_coordinate_px"],
        derived["material_error_px"],
        montage_path,
    )

    result = {
        "schema_version": "certified_witness_supervised_capability_result.v1",
        "artifact_type": "source_bound_supervised_capability_result",
        "scientific_full_orbit_run": args.frame_limit == EXPECTED_FRAMES,
        "strict_capability_pass": final_report["strict_capability_pass"],
        "seed": args.seed,
        "best_update": int(best_payload["update"]),
        "completed_updates": update,
        "runtime_seconds": time.perf_counter() - start,
        "device": str(device),
        "initial_model_state_sha256": initial_state_hash,
        "best_model_state_sha256": loaded_state_hash,
        "optimizer_parameter_change_proved": parameter_change_proved,
        "checkpoint_round_trip_exact": True,
        "semantic_controls": controls,
        "evaluation": final_report,
        "decision_branch": (
            "strict_seed_capability_pass"
            if final_report["strict_capability_pass"] and args.frame_limit == EXPECTED_FRAMES
            else "bounded_smoke_only_not_scientific"
            if args.frame_limit != EXPECTED_FRAMES
            else "strict_seed_capability_not_reached_within_budget"
        ),
        "preservation_phase_authorized_by_this_result": False,
        "statistical_scope": {
            "inference": "descriptive_only",
            "object_count": 1,
            "orbit_count": 1,
            "optimization_seed_count": 1,
            "frame_values_independent": False,
        },
    }
    result_path = args.output_dir / "CAPABILITY_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "certified_witness_supervised_capability_run_receipt.v1",
        "result": file_record(result_path),
        "config": file_record(args.output_dir / "config.json"),
        "semantic_controls": file_record(args.output_dir / "semantic_controls.json"),
        "best_model": file_record(best_copy),
        "selected_checkpoint": file_record(best_checkpoint_path),
        "history": file_record(history_path),
        "predictions": file_record(predictions_path),
        "worst_events_visual": file_record(montage_path),
        "strict_capability_pass": final_report["strict_capability_pass"],
        "scientific_full_orbit_run": args.frame_limit == EXPECTED_FRAMES,
        "preservation_phase_authorized": False,
    }
    (args.output_dir / "RUN_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tracks-sha256", required=True)
    parser.add_argument("--data-object-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frame-limit", type=int, default=EXPECTED_FRAMES)
    parser.add_argument("--max-updates", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stop-on-pass", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except CapabilityContractError as error:
        raise SystemExit(f"CAPABILITY CONTRACT FAILURE: {error}") from error


if __name__ == "__main__":
    main()
