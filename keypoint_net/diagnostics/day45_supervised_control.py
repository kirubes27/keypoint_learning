"""Days 4-5 supervised localization control.

Semantic contract:
* Ten deterministic, farthest-point-sampled mask-interior targets are defined
  in frame 0 and transported with the independently verified image-plane roll.
* Only even frames train; odd frames evaluate.
* RGB, mask, and targets receive the exact same seeded digital transform.
* ``coordinate`` trains soft-argmax coordinates with MSE.
* ``heatmap`` trains per-channel heatmap distributions with Gaussian-target CE.
* Unaugmented and fixed-augmented held-out metrics are reported separately.

This is an architecture/instrument control, not a keypoint discovery result.
One seed or one object cannot establish population generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
KEYPOINT_ROOT = HERE.parent
sys.path.insert(0, str(KEYPOINT_ROOT))
from model import KeypointExtractor  # noqa: E402


IMG_SIZE = 512
NUM_KEYPOINTS = 10
CELL64_NORM = 2.0 / 64.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#FFA500", "#800080", "#008000", "#000080",
]


def to_px(coords_norm: np.ndarray) -> np.ndarray:
    return (np.asarray(coords_norm, dtype=np.float64) + 1.0) * 0.5 * (IMG_SIZE - 1)


def to_norm(coords_px: np.ndarray) -> np.ndarray:
    return np.asarray(coords_px, dtype=np.float64) / (IMG_SIZE - 1) * 2.0 - 1.0


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """Forward matrix in image (x-right, y-down) coordinates."""
    angle = math.radians(angle_deg)
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )


def transform_points_px(
    points_px: np.ndarray,
    *,
    angle_deg: float,
    translate_xy: tuple[float, float],
    center_xy: tuple[float, float],
) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    translation = np.asarray(translate_xy, dtype=np.float64)
    return (points - center) @ rotation_matrix(angle_deg).T + center + translation


def shared_digital_transform(
    image: torch.Tensor,
    mask: torch.Tensor,
    targets_norm: torch.Tensor,
    *,
    angle_deg: float,
    translate_xy: tuple[int, int],
    center_xy: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one proven forward affine to RGB, mask, and target coordinates."""
    rows, cols = torch.meshgrid(
        torch.arange(IMG_SIZE, dtype=torch.float32),
        torch.arange(IMG_SIZE, dtype=torch.float32),
        indexing="ij",
    )
    output = torch.stack([cols, rows], dim=-1)
    center = torch.tensor(center_xy, dtype=torch.float32)
    translation = torch.tensor(translate_xy, dtype=torch.float32)
    matrix = torch.tensor(rotation_matrix(angle_deg), dtype=torch.float32)
    # grid_sample requires the inverse map from output q to source p.
    source = (output - center - translation) @ matrix + center
    grid = torch.empty_like(source)
    grid[..., 0] = source[..., 0] / (IMG_SIZE - 1) * 2.0 - 1.0
    grid[..., 1] = source[..., 1] / (IMG_SIZE - 1) * 2.0 - 1.0
    grid = grid.unsqueeze(0)

    background = torch.stack(
        [image[:, 0, 0], image[:, 0, -1], image[:, -1, 0], image[:, -1, -1]]
    ).median(dim=0).values[:, None, None]
    warped_image = F.grid_sample(
        (image - background).unsqueeze(0), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0] + background
    warped_mask = F.grid_sample(
        mask.float()[None, None], grid,
        mode="nearest", padding_mode="zeros", align_corners=True,
    )[0, 0] > 0.5
    transformed_px = transform_points_px(
        to_px(targets_norm.numpy()),
        angle_deg=angle_deg,
        translate_xy=translate_xy,
        center_xy=center_xy,
    )
    transformed_targets = torch.tensor(to_norm(transformed_px), dtype=torch.float32)
    return warped_image.clamp(0.0, 1.0), warped_mask, transformed_targets


def eroded_interior(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    foreground = torch.tensor(np.asarray(mask, dtype=np.float32))[None, None]
    background_nearby = F.max_pool2d(
        1.0 - foreground, kernel_size=2 * radius + 1, stride=1, padding=radius
    )
    interior = (foreground > 0.5) & (background_nearby < 0.5)
    result = interior[0, 0].numpy()
    if int(result.sum()) < NUM_KEYPOINTS:
        raise RuntimeError("mask has fewer than ten pixels after interior erosion")
    return result


def farthest_interior_points(mask: np.ndarray, count: int = NUM_KEYPOINTS) -> np.ndarray:
    rows, cols = np.nonzero(eroded_interior(mask))
    candidates = np.stack([cols, rows], axis=1).astype(np.float64)
    centroid = candidates.mean(axis=0)
    first = int(np.argmin(np.sum((candidates - centroid) ** 2, axis=1)))
    selected = [first]
    minimum_sq = np.sum((candidates - candidates[first]) ** 2, axis=1)
    for _ in range(1, count):
        next_index = int(np.argmax(minimum_sq))
        selected.append(next_index)
        distance_sq = np.sum((candidates - candidates[next_index]) ** 2, axis=1)
        minimum_sq = np.minimum(minimum_sq, distance_sq)
    return candidates[selected]


def transported_targets(
    frame0_points_px: np.ndarray,
    frames: int,
    *,
    center_xy: tuple[float, float],
    roll_sign: int,
) -> np.ndarray:
    return np.stack(
        [
            to_norm(
                transform_points_px(
                    frame0_points_px,
                    angle_deg=roll_sign * 2.0 * frame,
                    translate_xy=(0.0, 0.0),
                    center_xy=center_xy,
                )
            )
            for frame in range(frames)
        ]
    ).astype(np.float32)


def highpass_residual(coords: np.ndarray, window: int = 9) -> np.ndarray:
    pad = window // 2
    padded = np.concatenate([coords[:pad][::-1], coords, coords[-pad:][::-1]], axis=0)
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.stack(
        [
            np.stack(
                [np.convolve(padded[:, channel, axis], kernel, mode="valid") for axis in range(2)],
                axis=-1,
            )
            for channel in range(coords.shape[1])
        ],
        axis=1,
    )
    return coords - smooth


class SupervisedRollDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        indices: list[int],
        *,
        augment: bool,
        seed: int,
        center_xy: tuple[float, float],
    ) -> None:
        self.images = images
        self.masks = masks
        self.targets = targets
        self.indices = indices
        self.augment = augment
        self.seed = seed
        self.center_xy = center_xy
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        frame = self.indices[item]
        image = torch.tensor(self.images[frame], dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(self.masks[frame], dtype=torch.bool)
        target = torch.tensor(self.targets[frame], dtype=torch.float32)
        angle, dx, dy = 0.0, 0, 0
        if self.augment:
            rng = np.random.default_rng(
                self.seed + 1000003 * self.epoch + 9176 * frame + 37 * item
            )
            angle = float(rng.uniform(-5.0, 5.0))
            dx, dy = int(rng.integers(-8, 9)), int(rng.integers(-8, 9))
            image, mask, target = shared_digital_transform(
                image, mask, target,
                angle_deg=angle, translate_xy=(dx, dy), center_xy=self.center_xy,
            )
        normalized = (image - MEAN) / STD
        return {
            "image": normalized,
            "mask": mask,
            "target": target,
            "frame": torch.tensor(frame),
            "angle_deg": torch.tensor(angle, dtype=torch.float32),
            "translation": torch.tensor([dx, dy], dtype=torch.float32),
        }


def gaussian_target_distribution(
    target_norm: torch.Tensor, height: int, width: int, sigma_input_px: float = 8.0
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=target_norm.device)
    x = torch.linspace(-1.0, 1.0, width, device=target_norm.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    sigma_norm = 2.0 * sigma_input_px / (IMG_SIZE - 1)
    squared = (
        (xx[None, None] - target_norm[..., 0, None, None]) ** 2
        + (yy[None, None] - target_norm[..., 1, None, None]) ** 2
    )
    logits = -0.5 * squared / (sigma_norm**2)
    return torch.softmax(logits.flatten(-2), dim=-1)


def supervised_loss(
    arm: str,
    coordinates: torch.Tensor,
    heatmaps: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if arm == "coordinate":
        return F.mse_loss(coordinates, target)
    if arm == "heatmap":
        distribution = gaussian_target_distribution(target, heatmaps.shape[-2], heatmaps.shape[-1])
        log_prob = F.log_softmax(heatmaps.flatten(-2), dim=-1)
        return -(distribution * log_prob).sum(dim=-1).mean()
    raise ValueError(arm)


@torch.no_grad()
def evaluate(
    extractor: KeypointExtractor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    extractor.eval()
    predictions, targets, frames = [], [], []
    on_mask = []
    for batch in loader:
        image = batch["image"].to(device)
        prediction = extractor.get_keypoint_coords(image)
        target = batch["target"].to(device)
        predictions.append(prediction.cpu().numpy())
        targets.append(target.cpu().numpy())
        frames.append(batch["frame"].numpy())
        pixels = np.rint(to_px(prediction.cpu().numpy())).astype(int)
        pixels = np.clip(pixels, 0, IMG_SIZE - 1)
        masks = batch["mask"].numpy()
        inside = np.stack(
            [masks[i, pixels[i, :, 1], pixels[i, :, 0]] for i in range(len(masks))]
        )
        on_mask.append(inside)
    pred = np.concatenate(predictions)
    truth = np.concatenate(targets)
    frame_ids = np.concatenate(frames)
    inside = np.concatenate(on_mask)
    order = np.argsort(frame_ids)
    pred, truth, frame_ids, inside = pred[order], truth[order], frame_ids[order], inside[order]
    error_cells = np.linalg.norm(pred - truth, axis=-1) / CELL64_NORM
    channel_medians = np.median(error_cells, axis=0)
    metrics = {
        "median_error_cells64": float(np.median(error_cells)),
        "median_of_channel_medians_cells64": float(np.median(channel_medians)),
        "channel_median_error_cells64": channel_medians.tolist(),
        "p90_error_cells64": float(np.quantile(error_cells, 0.9)),
        "on_mask_fraction": float(inside.mean()),
        "n_frames": int(len(frame_ids)),
        "sample_unit": "held-out odd frame for each supervised channel",
        "uncertainty": "descriptive medians; frames belong to one correlated cyclic orbit",
    }
    return metrics, pred, truth, frame_ids


def sequence_jitter(
    extractor: KeypointExtractor,
    dataset: SupervisedRollDataset,
    device: torch.device,
    *,
    center_xy: tuple[float, float],
    roll_sign: int,
    batch_size: int,
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    _, prediction, _, frames = evaluate(extractor, loader, device)
    derotated = []
    for points, frame in zip(prediction, frames):
        derotated.append(
            to_norm(
                transform_points_px(
                    to_px(points),
                    angle_deg=-roll_sign * 2.0 * int(frame),
                    translate_xy=(0.0, 0.0),
                    center_xy=center_xy,
                )
            )
        )
    residual = highpass_residual(np.asarray(derotated))
    sigma_axis = residual.std(axis=0, ddof=0)
    sigma_channel = np.sqrt(np.mean(sigma_axis**2, axis=1)) / CELL64_NORM
    return {
        "channel_sigma_cells64": sigma_channel.tolist(),
        "median_channel_sigma_cells64": float(np.median(sigma_channel)),
        "definition": "width-9 high-pass after ground-truth derotation; population std ddof=0 over 180 frames",
        "sample_unit": "frame in one correlated cyclic orbit; descriptive only",
    }


def save_overlay(
    image_path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, (pred_px, target_px) in enumerate(zip(to_px(prediction), to_px(target))):
        color = COLORS[index]
        px, py = map(float, pred_px)
        tx, ty = map(float, target_px)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=color, width=3)
        draw.line((tx - 5, ty, tx + 5, ty), fill=color, width=2)
        draw.line((tx, ty - 5, tx, ty + 5), fill=color, width=2)
    image.save(output_path)


def load_arrays(data_root: Path, object_name: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    object_dir = data_root / "train" / object_name
    frame_paths = sorted((object_dir / "frames" / "a").glob("img_*.png"))
    mask_paths = sorted((object_dir / "masks" / "a").glob("mask_*.png"))
    if len(frame_paths) != 180 or len(mask_paths) != 180:
        raise RuntimeError(f"expected 180 frames/masks, found {len(frame_paths)}/{len(mask_paths)}")
    images = np.stack([np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in frame_paths])
    masks = np.stack([np.asarray(Image.open(path).convert("L")) > 0 for path in mask_paths])
    return images, masks, frame_paths


def target_mask_fraction(targets: np.ndarray, masks: np.ndarray) -> float:
    pixels = np.rint(to_px(targets)).astype(int)
    pixels = np.clip(pixels, 0, IMG_SIZE - 1)
    inside = [masks[f, pixels[f, :, 1], pixels[f, :, 0]] for f in range(len(masks))]
    return float(np.asarray(inside).mean())


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    center = (args.center_x, args.center_y)
    images, masks, frame_paths = load_arrays(args.data_root, args.object)
    frame0_points = farthest_interior_points(masks[0])
    targets = transported_targets(
        frame0_points, len(images), center_xy=center, roll_sign=args.roll_sign
    )
    fraction = target_mask_fraction(targets, masks)
    if fraction < 0.98:
        raise RuntimeError(f"transported target grounding failed: {fraction:.4f} < 0.98")

    train_indices = list(range(0, 180, 2))
    eval_indices = list(range(1, 180, 2))
    all_indices = list(range(180))
    train_data = SupervisedRollDataset(
        images, masks, targets, train_indices,
        augment=True, seed=args.seed, center_xy=center,
    )
    eval_plain = SupervisedRollDataset(
        images, masks, targets, eval_indices,
        augment=False, seed=args.seed + 500000, center_xy=center,
    )
    eval_aug = SupervisedRollDataset(
        images, masks, targets, eval_indices,
        augment=True, seed=args.seed + 500000, center_xy=center,
    )
    all_plain = SupervisedRollDataset(
        images, masks, targets, all_indices,
        augment=False, seed=args.seed, center_xy=center,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0,
    )
    plain_loader = DataLoader(eval_plain, batch_size=args.batch_size, shuffle=False, num_workers=0)
    aug_loader = DataLoader(eval_aug, batch_size=args.batch_size, shuffle=False, num_workers=0)

    extractor = KeypointExtractor(
        num_keypoints=NUM_KEYPOINTS,
        base_channels=args.base_channels,
        temperature=1.0,
        padding_mode="reflect",
        heatmap_res=args.heatmap_res,
    ).to(device)
    optimizer = torch.optim.Adam(extractor.parameters(), lr=args.lr, weight_decay=1e-5)
    run_dir = args.output_root / f"{args.arm}_h{args.heatmap_res}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update(
        {
            "device": str(device),
            "train_frames": train_indices,
            "eval_frames": eval_indices,
            "augmentation": {"rotation_deg": [-5, 5], "translation_px": [-8, 8]},
            "frame0_targets_px": frame0_points.tolist(),
            "transported_target_on_mask_fraction": fraction,
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    history = []
    best_augmented = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        extractor.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            flat, heatmaps = extractor(image)
            coordinates = flat.view(-1, NUM_KEYPOINTS, 2)
            loss = supervised_loss(args.arm, coordinates, heatmaps, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % args.eval_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            plain_metrics, _, _, _ = evaluate(extractor, plain_loader, device)
            aug_metrics, _, _, _ = evaluate(extractor, aug_loader, device)
            row = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "eval_plain_median_cells64": plain_metrics["median_of_channel_medians_cells64"],
                "eval_aug_median_cells64": aug_metrics["median_of_channel_medians_cells64"],
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if row["eval_aug_median_cells64"] < best_augmented:
                best_augmented = row["eval_aug_median_cells64"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "extractor_state_dict": extractor.state_dict(),
                        "epoch": best_epoch,
                        "config": config,
                    },
                    run_dir / "best_model.pt",
                )

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    extractor.load_state_dict(checkpoint["extractor_state_dict"], strict=True)
    plain_metrics, plain_pred, plain_target, plain_frames = evaluate(extractor, plain_loader, device)
    aug_metrics, _, _, _ = evaluate(extractor, aug_loader, device)
    jitter = sequence_jitter(
        extractor, all_plain, device,
        center_xy=center, roll_sign=args.roll_sign, batch_size=args.batch_size,
    )
    result = {
        "arm": args.arm,
        "heatmap_res": args.heatmap_res,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "device": str(device),
        "runtime_seconds": time.perf_counter() - start_time,
        "unaugmented_heldout": plain_metrics,
        "augmented_heldout": aug_metrics,
        "unaugmented_sequence_jitter": jitter,
        "preregistered_read": (
            "capable" if aug_metrics["median_of_channel_medians_cells64"] <= 0.4
            else "architecture_or_readout_bottleneck"
            if aug_metrics["median_of_channel_medians_cells64"] >= 0.8
            else "mixed_requires_true_quarter_resolution_arm"
        ),
        "statistical_scope": "descriptive single object, one cyclic orbit, one optimization seed",
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        run_dir / "heldout_predictions.npz",
        prediction=plain_pred,
        target=plain_target,
        frame=plain_frames,
    )
    overlay_index = int(np.where(plain_frames == 45)[0][0])
    save_overlay(
        frame_paths[45], plain_pred[overlay_index], plain_target[overlay_index],
        run_dir / "heldout_frame45_overlay.png",
    )
    print(json.dumps(result, indent=2), flush=True)
    return run_dir


def self_test() -> None:
    image = torch.zeros(3, IMG_SIZE, IMG_SIZE)
    mask = torch.zeros(IMG_SIZE, IMG_SIZE, dtype=torch.bool)
    source_px = np.array([[300.0, 200.0]])
    sx, sy = map(int, source_px[0])
    image[:, sy, sx] = 1.0
    mask[sy, sx] = True
    target = torch.tensor(to_norm(source_px), dtype=torch.float32)
    transformed_image, transformed_mask, transformed_target = shared_digital_transform(
        image, mask, target,
        angle_deg=0.0, translate_xy=(7, -4), center_xy=(255.5, 255.5),
    )
    expected = source_px + [7, -4]
    assert np.max(np.abs(to_px(transformed_target.numpy()) - expected)) < 1e-4
    rows, cols = np.nonzero(transformed_mask.numpy())
    assert len(rows) == 1 and abs(cols[0] - expected[0, 0]) <= 1 and abs(rows[0] - expected[0, 1]) <= 1
    peak = np.unravel_index(int(transformed_image[0].argmax()), transformed_image[0].shape)
    assert abs(peak[1] - expected[0, 0]) <= 1 and abs(peak[0] - expected[0, 1]) <= 1
    rotated = transform_points_px(
        np.array([[300.0, 255.5]]), angle_deg=90.0,
        translate_xy=(0.0, 0.0), center_xy=(255.5, 255.5),
    )
    assert np.allclose(rotated, [[255.5, 300.0]], atol=1e-8)
    print("DAY45 SHARED-TRANSFORM SEMANTIC SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--object", default="engineers_hammer_vray")
    parser.add_argument("--arm", choices=("coordinate", "heatmap"), default="coordinate")
    parser.add_argument("--heatmap-res", type=int, choices=(64, 128), default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--center-x", type=float, default=255.49998435893767)
    parser.add_argument("--center-y", type=float, default=255.50001568508694)
    parser.add_argument("--roll-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs" / "day45_runs")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.data_root is None:
        parser.error("--data-root is required unless --self-test is used")
    train(args)


if __name__ == "__main__":
    main()
