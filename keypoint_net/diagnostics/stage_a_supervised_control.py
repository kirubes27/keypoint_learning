"""Final Stage-A supervised coordinate-instrument control.

This replaces the execution role of ``day45_supervised_control.py`` while
keeping that script unchanged as an audit record.

Semantic contract
-----------------
* The existing phase-modulo split is used: residues {0,3} train, {1,4}
  validation, and {2,5} test.
* Training never evaluates the test split. ``finalize`` requires three frozen
  runs and touches test exactly once.
* Validation checkpoint selection uses the worse of unaugmented and fixed-
  augmentation median localization errors.
* The tiny-subset gate must pass before a convergence pilot is meaningful.
* ``--native-quarter`` changes the encoder's third stride to one. It is not
  the older feature-upsampling 128-head experiment.

The result is an architecture/instrument control, not a landmark-discovery or
cross-object generalization result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
KEYPOINT_ROOT = HERE.parent
sys.path.insert(0, str(KEYPOINT_ROOT))

from model import KeypointExtractor  # noqa: E402
from diagnostics.day45_supervised_control import (  # noqa: E402
    SupervisedRollDataset,
    evaluate,
    farthest_interior_points,
    load_arrays,
    save_overlay,
    supervised_loss,
    target_mask_fraction,
    transported_targets,
)


NUM_KEYPOINTS = 10
DEFAULT_VALIDATION_AUGMENT_SEED = 2026070401
DEFAULT_TEST_AUGMENT_SEED = 2026070402
EXPECTED_RESIDUES = {
    "train": {0, 3},
    "validation": {1, 4},
    "test": {2, 5},
}


@dataclass(frozen=True)
class PhaseSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    sha256: str


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_phase_split(path: Path, object_name: str) -> PhaseSplit:
    raw = json.loads(path.read_text())
    source_keys = {"train": "train", "validation": "val", "test": "test"}
    split: dict[str, tuple[int, ...]] = {}
    for destination, source in source_keys.items():
        if source not in raw:
            raise ValueError(f"split file has no {source!r} key")
        indices = sorted(
            int(row["frame_index"])
            for row in raw[source]
            if row.get("model_name") == object_name
        )
        if not indices:
            raise ValueError(f"split {source!r} has no rows for {object_name!r}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate frame in {source!r} split")
        residues = {index % 6 for index in indices}
        if residues != EXPECTED_RESIDUES[destination]:
            raise ValueError(
                f"{destination} residues {sorted(residues)} != "
                f"{sorted(EXPECTED_RESIDUES[destination])}"
            )
        split[destination] = tuple(indices)

    sets = [set(split[name]) for name in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train/validation/test frame sets overlap")
    union = sets[0] | sets[1] | sets[2]
    if union != set(range(180)):
        missing = sorted(set(range(180)) - union)
        extra = sorted(union - set(range(180)))
        raise ValueError(f"split does not partition 0..179; missing={missing}, extra={extra}")
    return PhaseSplit(
        train=split["train"],
        validation=split["validation"],
        test=split["test"],
        sha256=sha256_file(path),
    )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_extractor(args: argparse.Namespace, device: torch.device) -> KeypointExtractor:
    return KeypointExtractor(
        num_keypoints=NUM_KEYPOINTS,
        base_channels=args.base_channels,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=128 if args.native_quarter else 64,
        true_quarter_res=args.native_quarter,
    ).to(device)


def load_problem(args: argparse.Namespace) -> dict[str, Any]:
    split_path = args.split_json or (args.data_root / "indices" / "split_phase_mod6.json")
    split = load_phase_split(split_path, args.object)
    images, masks, frame_paths = load_arrays(args.data_root, args.object)
    center = (args.center_x, args.center_y)
    frame0_points = farthest_interior_points(masks[0])
    targets = transported_targets(
        frame0_points,
        len(images),
        center_xy=center,
        roll_sign=args.roll_sign,
    )
    grounding = target_mask_fraction(targets, masks)
    if grounding < 0.98:
        raise RuntimeError(f"transported target grounding failed: {grounding:.6f} < 0.98")
    return {
        "split": split,
        "split_path": split_path,
        "images": images,
        "masks": masks,
        "frame_paths": frame_paths,
        "targets": targets,
        "frame0_points": frame0_points,
        "target_grounding": grounding,
        "center": center,
    }


def make_dataset(
    problem: dict[str, Any],
    indices: Iterable[int],
    *,
    augment: bool,
    seed: int,
) -> SupervisedRollDataset:
    dataset = SupervisedRollDataset(
        problem["images"],
        problem["masks"],
        problem["targets"],
        list(indices),
        augment=augment,
        seed=seed,
        center_xy=problem["center"],
    )
    dataset.set_epoch(0)
    return dataset


def make_eval_loader(dataset: SupervisedRollDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_pair(
    extractor: KeypointExtractor,
    plain_loader: DataLoader,
    augmented_loader: DataLoader,
    device: torch.device,
    *,
    split_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    plain, plain_pred, plain_target, plain_frames = evaluate(extractor, plain_loader, device)
    augmented, aug_pred, aug_target, aug_frames = evaluate(
        extractor, augmented_loader, device
    )
    plain["sample_unit"] = f"{split_name} frame x supervised channel"
    augmented["sample_unit"] = (
        f"fixed digitally augmented {split_name} frame x supervised channel"
    )
    score = max(
        plain["median_of_channel_medians_cells64"],
        augmented["median_of_channel_medians_cells64"],
    )
    metrics = {
        "selection_score_cells64": float(score),
        "selection_rule": (
            "max(unaugmented, fixed-augmented median-of-channel median error)"
        ),
        "unaugmented": plain,
        "fixed_augmented": augmented,
    }
    arrays = {
        "plain_prediction": plain_pred,
        "plain_target": plain_target,
        "plain_frame": plain_frames,
        "augmented_prediction": aug_pred,
        "augmented_target": aug_target,
        "augmented_frame": aug_frames,
    }
    return metrics, arrays


def run_name(args: argparse.Namespace) -> str:
    architecture = "native_quarter" if args.native_quarter else "standard64"
    return f"coordinate_{architecture}_seed{args.seed}"


def base_config(args: argparse.Namespace, problem: dict[str, Any], device: torch.device) -> dict:
    split: PhaseSplit = problem["split"]
    return {
        "schema_version": 1,
        "semantic_role": "supervised_coordinate_instrument_control",
        "object": args.object,
        "data_root": str(args.data_root.resolve()),
        "split_json": str(problem["split_path"].resolve()),
        "split_sha256": split.sha256,
        "train_frames": list(split.train),
        "validation_frames": list(split.validation),
        "test_frames_committed_not_evaluated": list(split.test),
        "seed": args.seed,
        "device": str(device),
        "base_channels": args.base_channels,
        "architecture": "native_quarter" if args.native_quarter else "standard64",
        "heatmap_res": 128 if args.native_quarter else 64,
        "true_quarter_res": bool(args.native_quarter),
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "min_epochs": args.min_epochs,
        "max_epochs": args.max_epochs,
        "eval_every": args.eval_every,
        "plateau_patience_epochs": args.plateau_patience,
        "relative_improvement": args.relative_improvement,
        "checkpoint_selection": (
            "minimum validation max(unaugmented,fixed-augmented) "
            "median-of-channel median error"
        ),
        "train_augmentation": {"rotation_deg": [-5, 5], "translation_px": [-8, 8]},
        "validation_augmentation_seed": DEFAULT_VALIDATION_AUGMENT_SEED,
        "test_augmentation_seed": DEFAULT_TEST_AUGMENT_SEED,
        "frame0_targets_px": problem["frame0_points"].tolist(),
        "transported_target_on_mask_fraction_all_frames": problem["target_grounding"],
        "test_policy": "not evaluated during train; finalize requires three frozen runs",
        "statistical_scope": (
            "single object and correlated cyclic orbit; seed replicates optimization only"
        ),
    }


def save_checkpoint(
    path: Path,
    *,
    extractor: KeypointExtractor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    best_score: float,
    significant_best: float,
    last_significant_epoch: int,
    loader_generator: torch.Generator,
) -> None:
    payload: dict[str, Any] = {
        "extractor_state_dict": extractor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "epoch": int(epoch),
        "config": config,
        "best_score": float(best_score),
        "significant_best": float(significant_best),
        "last_significant_epoch": int(last_significant_epoch),
        "loader_generator_state": loader_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def restore_checkpoint(
    path: Path,
    extractor: KeypointExtractor,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    extractor.load_state_dict(checkpoint["extractor_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loader_generator.set_state(checkpoint["loader_generator_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        result.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_score_cells64": float(row["val_score_cells64"]),
                "val_plain_median_cells64": float(row["val_plain_median_cells64"]),
                "val_augmented_median_cells64": float(
                    row["val_augmented_median_cells64"]
                ),
            }
        )
    return result


def train_control(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = choose_device(args.device)
    problem = load_problem(args)
    split: PhaseSplit = problem["split"]
    run_dir = args.output_root / "runs" / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = base_config(args, problem, device)
    config_path = run_dir / "config.json"

    if config_path.exists() and not args.resume:
        raise FileExistsError(f"run already exists: {run_dir}; use --resume explicitly")
    if not config_path.exists():
        write_json(config_path, config)
    else:
        existing = json.loads(config_path.read_text())
        immutable = (
            "split_sha256", "seed", "base_channels", "architecture", "learning_rate",
            "weight_decay", "batch_size", "min_epochs", "max_epochs", "eval_every",
            "plateau_patience_epochs", "relative_improvement",
        )
        mismatches = {key: (existing[key], config[key]) for key in immutable if existing[key] != config[key]}
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        config = existing

    train_data = make_dataset(
        problem, split.train, augment=True, seed=args.seed
    )
    val_plain = make_dataset(
        problem, split.validation, augment=False, seed=DEFAULT_VALIDATION_AUGMENT_SEED
    )
    val_augmented = make_dataset(
        problem, split.validation, augment=True, seed=DEFAULT_VALIDATION_AUGMENT_SEED
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    val_plain_loader = make_eval_loader(val_plain, args.batch_size)
    val_aug_loader = make_eval_loader(val_augmented, args.batch_size)

    extractor = build_extractor(args, device)
    optimizer = torch.optim.Adam(
        extractor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    best_score = float("inf")
    significant_best = float("inf")
    last_significant_epoch = 0
    history = read_history(run_dir / "history.csv")
    last_path = run_dir / "last_checkpoint.pt"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"cannot resume; missing {last_path}")
        checkpoint = restore_checkpoint(
            last_path, extractor, optimizer, loader_generator, device
        )
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint["best_score"])
        significant_best = float(checkpoint["significant_best"])
        last_significant_epoch = int(checkpoint["last_significant_epoch"])

    stop_reason = "hard_cap_unconverged"
    start_time = time.perf_counter()
    for epoch in range(start_epoch, args.max_epochs):
        train_data.set_epoch(epoch)
        extractor.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            flat, heatmaps = extractor(image)
            coordinates = flat.view(-1, NUM_KEYPOINTS, 2)
            loss = supervised_loss("coordinate", coordinates, heatmaps, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        current_epoch = epoch + 1
        should_eval = current_epoch == 1 or current_epoch % args.eval_every == 0
        if not should_eval:
            continue
        val_metrics, _ = evaluate_pair(
            extractor,
            val_plain_loader,
            val_aug_loader,
            device,
            split_name="validation",
        )
        score = float(val_metrics["selection_score_cells64"])
        row = {
            "epoch": current_epoch,
            "train_loss": float(np.mean(losses)),
            "val_score_cells64": score,
            "val_plain_median_cells64": val_metrics["unaugmented"][
                "median_of_channel_medians_cells64"
            ],
            "val_augmented_median_cells64": val_metrics["fixed_augmented"][
                "median_of_channel_medians_cells64"
            ],
        }
        history.append(row)
        write_history(run_dir / "history.csv", history)
        print(json.dumps(row), flush=True)

        if score < best_score:
            best_score = score
            save_checkpoint(
                run_dir / "best_model.pt",
                extractor=extractor,
                optimizer=optimizer,
                epoch=current_epoch,
                config=config,
                best_score=best_score,
                significant_best=significant_best,
                last_significant_epoch=last_significant_epoch,
                loader_generator=loader_generator,
            )
            write_json(run_dir / "best_validation_metrics.json", val_metrics)

        threshold = significant_best * (1.0 - args.relative_improvement)
        if not np.isfinite(significant_best) or score <= threshold:
            significant_best = score
            last_significant_epoch = current_epoch

        save_checkpoint(
            last_path,
            extractor=extractor,
            optimizer=optimizer,
            epoch=current_epoch,
            config=config,
            best_score=best_score,
            significant_best=significant_best,
            last_significant_epoch=last_significant_epoch,
            loader_generator=loader_generator,
        )
        if (
            current_epoch >= args.min_epochs
            and current_epoch - last_significant_epoch >= args.plateau_patience
        ):
            stop_reason = "validation_plateau"
            break

    summary = {
        "run_dir": str(run_dir.resolve()),
        "seed": args.seed,
        "architecture": config["architecture"],
        "completed_epoch": int(history[-1]["epoch"]),
        "best_epoch": int(
            torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)[
                "epoch"
            ]
        ),
        "best_validation_score_cells64": best_score,
        "stop_reason": stop_reason,
        "runtime_seconds_this_invocation": time.perf_counter() - start_time,
        "test_evaluated": False,
        "interpretation": (
            "validation converged" if stop_reason == "validation_plateau"
            else "hard cap reached; capability may be shown by test but no ceiling claim is allowed"
        ),
    }
    write_json(run_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return run_dir


def tiny_overfit(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = choose_device(args.device)
    problem = load_problem(args)
    indices = list(problem["split"].train[: args.tiny_frames])
    dataset = make_dataset(problem, indices, augment=False, seed=args.seed)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    extractor = build_extractor(args, device)
    optimizer = torch.optim.Adam(
        extractor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    run_dir = args.output_root / "tiny_overfit" / run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"tiny-overfit result already exists: {run_dir}")

    batch = next(iter(loader))
    image = batch["image"].to(device)
    target = batch["target"].to(device)
    start = time.perf_counter()
    passed = False
    metrics: dict[str, Any] = {}
    completed_steps = 0
    for step in range(1, args.tiny_max_steps + 1):
        extractor.train()
        flat, heatmaps = extractor(image)
        coordinates = flat.view(-1, NUM_KEYPOINTS, 2)
        loss = supervised_loss("coordinate", coordinates, heatmaps, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        completed_steps = step
        if step == 1 or step % args.tiny_eval_every == 0:
            eval_metrics, _, _, _ = evaluate(extractor, loader, device)
            channel = np.asarray(eval_metrics["channel_median_error_cells64"])
            metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "median_error_cells64": float(np.median(channel)),
                "max_channel_median_error_cells64": float(np.max(channel)),
            }
            print(json.dumps(metrics), flush=True)
            passed = (
                metrics["median_error_cells64"] <= 0.10
                and metrics["max_channel_median_error_cells64"] <= 0.20
            )
            if passed:
                break

    result = {
        "gate": "A0_tiny_subset_overfit",
        "passed": passed,
        "thresholds": {
            "median_error_cells64_max": 0.10,
            "max_channel_median_error_cells64_max": 0.20,
        },
        "tiny_frames": indices,
        "completed_steps": completed_steps,
        "metrics": metrics,
        "device": str(device),
        "runtime_seconds": time.perf_counter() - start,
        "semantic_read": (
            "implementation can fit the fixed coordinate targets"
            if passed
            else "critical implementation/optimization gate failed; do not launch convergence runs"
        ),
    }
    write_json(run_dir / "metrics.json", result)
    torch.save(
        {"extractor_state_dict": extractor.state_dict(), "result": result},
        run_dir / "model.pt",
    )
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("A0 tiny-subset gate failed")
    return run_dir


def args_from_run_config(config: dict[str, Any], template: argparse.Namespace) -> argparse.Namespace:
    values = vars(template).copy()
    values.update(
        {
            "data_root": Path(config["data_root"]),
            "split_json": Path(config["split_json"]),
            "object": config["object"],
            "seed": int(config["seed"]),
            "base_channels": int(config["base_channels"]),
            "native_quarter": bool(config["true_quarter_res"]),
            "batch_size": int(config["batch_size"]),
        }
    )
    return argparse.Namespace(**values)


def instrument_gate(test_metrics: dict[str, Any]) -> bool:
    plain = test_metrics["unaugmented"]
    augmented = test_metrics["fixed_augmented"]
    return bool(
        max(
            plain["median_of_channel_medians_cells64"],
            augmented["median_of_channel_medians_cells64"],
        )
        <= 0.50
        and max(plain["p90_error_cells64"], augmented["p90_error_cells64"])
        <= 1.50
        and min(plain["on_mask_fraction"], augmented["on_mask_fraction"])
        >= 0.95
    )


def finalize_three_runs(args: argparse.Namespace) -> Path:
    if len(args.run_dirs) != 3:
        raise ValueError("finalize requires exactly three frozen run directories")
    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    if len(set(run_dirs)) != 3:
        raise ValueError("run directories must be distinct")
    records = []
    configs = []
    for run_dir in run_dirs:
        if (run_dir / "test_metrics.json").exists():
            raise RuntimeError(f"test was already evaluated for {run_dir}")
        config = json.loads((run_dir / "config.json").read_text())
        summary = json.loads((run_dir / "training_summary.json").read_text())
        if summary.get("test_evaluated"):
            raise RuntimeError(f"training summary says test was already evaluated: {run_dir}")
        if not (run_dir / "best_model.pt").exists():
            raise FileNotFoundError(run_dir / "best_model.pt")
        configs.append(config)
        records.append((run_dir, config, summary))

    seeds = {int(config["seed"]) for config in configs}
    if len(seeds) != 3:
        raise ValueError(f"expected three distinct seeds, got {sorted(seeds)}")
    comparable = (
        "split_sha256", "architecture", "base_channels", "learning_rate",
        "weight_decay", "batch_size", "min_epochs", "max_epochs", "eval_every",
        "plateau_patience_epochs", "relative_improvement",
    )
    reference = configs[0]
    for config in configs[1:]:
        mismatch = {key: (reference[key], config[key]) for key in comparable if reference[key] != config[key]}
        if mismatch:
            raise ValueError(f"runs do not share a frozen recipe: {mismatch}")

    device = choose_device(args.device)
    staged_results = []
    for run_dir, config, summary in records:
        run_args = args_from_run_config(config, args)
        problem = load_problem(run_args)
        split: PhaseSplit = problem["split"]
        test_plain = make_dataset(
            problem, split.test, augment=False, seed=DEFAULT_TEST_AUGMENT_SEED
        )
        test_augmented = make_dataset(
            problem, split.test, augment=True, seed=DEFAULT_TEST_AUGMENT_SEED
        )
        extractor = build_extractor(run_args, device)
        checkpoint = torch.load(
            run_dir / "best_model.pt", map_location=device, weights_only=True
        )
        extractor.load_state_dict(checkpoint["extractor_state_dict"], strict=True)
        metrics, arrays = evaluate_pair(
            extractor,
            make_eval_loader(test_plain, run_args.batch_size),
            make_eval_loader(test_augmented, run_args.batch_size),
            device,
            split_name="test",
        )
        metrics.update(
            {
                "seed": int(config["seed"]),
                "architecture": config["architecture"],
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "stop_reason": summary["stop_reason"],
                "instrument_pass": instrument_gate(metrics),
                "thresholds": {
                    "both_median_of_channel_medians_cells64_max": 0.50,
                    "both_p90_error_cells64_max": 1.50,
                    "both_on_mask_fraction_min": 0.95,
                },
                "statistical_scope": (
                    "descriptive single object/correlated orbit; this seed is one optimization replicate"
                ),
            }
        )
        staged_results.append((run_dir, metrics, arrays, problem))

    pass_count = sum(int(metrics["instrument_pass"]) for _, metrics, _, _ in staged_results)
    aggregate = {
        "gate": "A1_supervised_coordinate_instrument",
        "passed": pass_count >= 2,
        "pass_count": pass_count,
        "required": "at least 2 of 3 optimization seeds",
        "seeds": sorted(seeds),
        "architecture": reference["architecture"],
        "per_seed": [metrics for _, metrics, _, _ in staged_results],
        "test_touched_once": True,
        "statistical_reporting": {
            "quantity": "raw per-seed descriptive localization metrics; no error bars or hypothesis test",
            "sample_unit": "optimization seed",
            "n": 3,
            "frame_dependence": "60 test frames per seed are one correlated cyclic orbit",
            "scope": "instrument capability on one object; not population inference",
        },
        "decision": (
            "Stage B may proceed"
            if pass_count >= 2
            else "STOP before Stage B; standard instrument failed"
        ),
    }

    output = args.output_root / f"instrument_gate_{reference['architecture']}.json"
    if output.exists():
        raise FileExistsError(f"aggregate test output already exists: {output}")
    # Stage all evaluations in memory; write only after every run evaluated successfully.
    for run_dir, metrics, arrays, problem in staged_results:
        write_json(run_dir / "test_metrics.json", metrics)
        np.savez_compressed(run_dir / "test_predictions.npz", **arrays)
        plain_frames = arrays["plain_frame"]
        overlay_position = int(np.argmin(np.abs(plain_frames - 44)))
        frame = int(plain_frames[overlay_position])
        save_overlay(
            problem["frame_paths"][frame],
            arrays["plain_prediction"][overlay_position],
            arrays["plain_target"][overlay_position],
            run_dir / f"test_frame{frame:03d}_overlay.png",
        )
        summary_path = run_dir / "training_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["test_evaluated"] = True
        write_json(summary_path, summary)
    write_json(output, aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)
    return output


def write_prelaunch_lock(args: argparse.Namespace) -> Path:
    problem = load_problem(args)
    payload = {
        "schema_version": 1,
        "created_before_stage_a_full_runs": True,
        "split_json": str(problem["split_path"].resolve()),
        "split_sha256": problem["split"].sha256,
        "semantic_gate": {
            "tiny_median_cells64_max": 0.10,
            "tiny_max_channel_median_cells64_max": 0.20,
            "instrument_pass_seeds": 2,
            "instrument_total_seeds": 3,
            "test_median_cells64_max_both_plain_and_augmented": 0.50,
            "test_p90_cells64_max_both_plain_and_augmented": 1.50,
            "test_on_mask_min_both_plain_and_augmented": 0.95,
        },
        "stopping": {
            "minimum_epochs": args.min_epochs,
            "maximum_epochs": args.max_epochs,
            "evaluation_every_epochs": args.eval_every,
            "plateau_patience_epochs": args.plateau_patience,
            "relative_improvement": args.relative_improvement,
        },
        "checkpoint_selection": (
            "minimum validation max(unaugmented,fixed-augmented) "
            "median-of-channel median localization error"
        ),
        "test_policy": "finalize exactly three frozen runs; refuse repeat evaluation",
        "statistical_scope": (
            "single object; correlated orbit; n=3 optimization seeds; descriptive only"
        ),
    }
    output = args.output_root / "PRELAUNCH_STAGE_A_LOCK.json"
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise RuntimeError(f"prelaunch lock exists with different content: {output}")
    else:
        write_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("lock", "tiny-overfit", "train", "finalize")
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--object", default="engineers_hammer_vray")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs" / "stage_a")
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--native-quarter", action="store_true")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-epochs", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--plateau-patience", type=int, default=400)
    parser.add_argument("--relative-improvement", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tiny-frames", type=int, default=4)
    parser.add_argument("--tiny-max-steps", type=int, default=5000)
    parser.add_argument("--tiny-eval-every", type=int, default=100)
    parser.add_argument("--center-x", type=float, default=255.49998435893767)
    parser.add_argument("--center-y", type=float, default=255.50001568508694)
    parser.add_argument("--roll-sign", type=int, choices=(-1, 1), default=1)
    args = parser.parse_args()
    if args.mode != "finalize" and args.data_root is None:
        parser.error("--data-root is required for lock, tiny-overfit, and train")
    if args.mode == "finalize" and not args.run_dirs:
        parser.error("--run-dirs is required for finalize")
    if args.eval_every <= 0 or args.plateau_patience % args.eval_every != 0:
        parser.error("plateau patience must be a positive multiple of eval-every")
    if args.min_epochs > args.max_epochs:
        parser.error("min-epochs cannot exceed max-epochs")
    return args


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    if args.mode == "lock":
        write_prelaunch_lock(args)
    elif args.mode == "tiny-overfit":
        tiny_overfit(args)
    elif args.mode == "train":
        train_control(args)
    else:
        finalize_three_runs(args)


if __name__ == "__main__":
    main()
