"""Shared, meaning-first utilities for the diagnostic week.

The critical semantic contract is:
* checkpoints are reconstructed from their authoritative config.json;
* coordinates use model.py's (x, y) convention;
* foreground means the true per-frame TDW instance mask;
* equivariance uses a rotation centre and sign validated by output mask IoU.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
KEYPOINT_ROOT = HERE.parent
PROJECT_ROOT = KEYPOINT_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "tdw_phase_a_starter " / "_tdw_world_z_roll_base_panel_512_v2"
OBJECT_DIR = DATA_ROOT / "train" / "engineers_hammer_vray"
FRAMES_DIR = OBJECT_DIR / "frames" / "a"
MASKS_DIR = OBJECT_DIR / "masks" / "a"
OUTPUTS = HERE / "outputs"
SMOKE_RUNS = KEYPOINT_ROOT / "runs_res_smoke"
TASK80_RUN = (
    PROJECT_ROOT
    / "cluster_downloads"
    / "hammer_full360_shared_complete"
    / "keypoint_net"
    / "runs_hammer_full360_shared"
    / "phase_a_engineers_hammer_vray_20260606_151123_941396_seed42_pid3289780"
)

IMG_SIZE = 512
CELL64_NORM = 2.0 / 64.0
CELL64_PX = IMG_SIZE / 64.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# Exact first-ten palette used by keypoint_net/visualize.py and the sweeps.
COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#FFA500", "#800080", "#008000", "#000080",
]

sys.path.insert(0, str(KEYPOINT_ROOT))
from model import KeypointExtractor  # noqa: E402


@dataclass(frozen=True)
class RotationModel:
    center_x_px: float
    center_y_px: float
    sign: int
    mean_iou: float
    even_center_x_px: float
    even_center_y_px: float
    odd_center_x_px: float
    odd_center_y_px: float
    split_center_distance_px: float
    geometry_ok: bool

    @property
    def center_px(self) -> np.ndarray:
        return np.array([self.center_x_px, self.center_y_px], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


def run_directories() -> dict[str, Path]:
    """Return the three frozen checkpoints by semantic label."""
    smoke: dict[int, Path] = {}
    for run_dir in sorted(SMOKE_RUNS.glob("phase_a_*")):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        smoke[int(cfg.get("heatmap_res", 64))] = run_dir
    assert 64 in smoke and 128 in smoke, f"missing smoke runs: {smoke}"
    assert TASK80_RUN.exists(), TASK80_RUN
    return {"task80": TASK80_RUN, "smoke64": smoke[64], "smoke128": smoke[128]}


def load_run(run_dir: str | Path) -> tuple[KeypointExtractor, dict]:
    """Reconstruct an extractor using config.json, never checkpoint defaults."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    extractor = KeypointExtractor(
        in_channels=3,
        num_keypoints=int(cfg["num_keypoints"]),
        base_channels=int(cfg["base_channels"]),
        temperature=float(cfg["temperature"]),
        padding_mode=cfg["padding_mode"],
        heatmap_res=int(cfg.get("heatmap_res", 64)),
    )
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    state = {
        key[len("extractor.") :]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("extractor.")
    }
    extractor.load_state_dict(state, strict=True)
    extractor.eval()
    return extractor, cfg


def frame_files() -> list[Path]:
    files = sorted(FRAMES_DIR.glob("img_*.png"))
    assert len(files) == 180, f"expected 180 frames, found {len(files)}"
    return files


def preprocess_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor - MEAN) / STD


def forward_sequence(
    extractor: KeypointExtractor,
    *,
    batch_size: int = 12,
    include_heatmaps: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    coords: list[torch.Tensor] = []
    heatmaps: list[torch.Tensor] = []
    files = frame_files()
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            batch = torch.cat([preprocess_image(p) for p in files[start : start + batch_size]])
            flat_coords, logits = extractor(batch)
            coords.append(flat_coords.view(-1, extractor.num_keypoints, 2).cpu())
            if include_heatmaps:
                heatmaps.append(logits.cpu())
    coord_array = torch.cat(coords).numpy().astype(np.float32)
    heatmap_array = (
        torch.cat(heatmaps).numpy().astype(np.float32) if include_heatmaps else None
    )
    return coord_array, heatmap_array


def trajectories(extractor: KeypointExtractor) -> np.ndarray:
    return forward_sequence(extractor, include_heatmaps=False)[0]


def load_masks() -> np.ndarray:
    files = sorted(MASKS_DIR.glob("mask_*.png"))
    assert len(files) == 180, f"expected 180 masks, found {len(files)}"
    masks = np.stack(
        [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in files]
    )
    values = set(np.unique(masks).tolist())
    assert values.issubset({0, 255}) and 255 in values, f"non-binary masks: {values}"
    return masks > 0


def to_px(coords_norm: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_norm, dtype=np.float64)
    return (coords + 1.0) * 0.5 * (IMG_SIZE - 1)


def to_norm(coords_px: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_px, dtype=np.float64)
    return coords / (IMG_SIZE - 1) * 2.0 - 1.0


def _mask_centroids(masks: np.ndarray) -> np.ndarray:
    centroids = []
    for mask in masks:
        rows, cols = np.nonzero(mask)
        assert len(rows) > 0, "empty mask"
        centroids.append([cols.mean(), rows.mean()])
    return np.asarray(centroids, dtype=np.float64)


def _sampling_grid(angle_deg: float, center_xy: Sequence[float]) -> torch.Tensor:
    """Grid for forward image rotation around an arbitrary pixel centre."""
    cx, cy = map(float, center_xy)
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rows, cols = torch.meshgrid(
        torch.arange(IMG_SIZE, dtype=torch.float32),
        torch.arange(IMG_SIZE, dtype=torch.float32),
        indexing="ij",
    )
    # grid_sample asks, for each output q, which input p to read. If the
    # forward point transform is q = c + R(angle)(p-c), use R(-angle).
    qx, qy = cols - cx, rows - cy
    px = cos_a * qx + sin_a * qy + cx
    py = -sin_a * qx + cos_a * qy + cy
    gx = px / (IMG_SIZE - 1) * 2.0 - 1.0
    gy = py / (IMG_SIZE - 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)


def warp_masks(
    masks: np.ndarray,
    angle_deg: float,
    center_xy: Sequence[float],
) -> np.ndarray:
    source = torch.from_numpy(np.asarray(masks, dtype=np.float32))[:, None]
    grid = _sampling_grid(angle_deg, center_xy).expand(source.shape[0], -1, -1, -1)
    warped = F.grid_sample(
        source,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped[:, 0].numpy() > 0.5


def mask_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_bool, b_bool = np.asarray(a, bool), np.asarray(b, bool)
    axes = tuple(range(1, a_bool.ndim))
    intersection = np.logical_and(a_bool, b_bool).sum(axis=axes)
    union = np.logical_or(a_bool, b_bool).sum(axis=axes)
    return intersection / np.maximum(union, 1)


def _mean_pair_iou(
    masks: np.ndarray,
    indices: Sequence[int],
    center_xy: Sequence[float],
    sign: int,
    hop: int = 1,
) -> float:
    idx = np.asarray(indices, dtype=int)
    targets = masks[(idx + hop) % len(masks)]
    warped = warp_masks(masks[idx], sign * 2.0 * hop, center_xy)
    return float(mask_iou(warped, targets).mean())


def _multihorizon_score(
    masks: np.ndarray,
    indices: Sequence[int],
    center_xy: Sequence[float],
    sign: int,
) -> float:
    """Equal-weight short/long-horizon score used only to estimate centre."""
    idx = np.asarray(indices, dtype=int)
    if len(idx) > 6:
        positions = np.linspace(0, len(idx) - 1, 6).round().astype(int)
        idx = idx[positions]
    return float(np.mean([
        _mean_pair_iou(masks, idx, center_xy, sign, hop=hop)
        for hop in (1, 3, 15, 45, 90)
    ]))


def _refine_center(
    masks: np.ndarray,
    indices: Sequence[int],
    initial_xy: Sequence[float],
    *,
    signs: Iterable[int] = (-1, 1),
) -> tuple[np.ndarray, int, float]:
    initial = np.asarray(initial_xy, dtype=np.float64)
    best: tuple[float, int, np.ndarray] | None = None
    # Coarse +/-10 px search, then a 1-px local refinement.
    for sign in signs:
        for dx in range(-10, 11, 5):
            for dy in range(-10, 11, 5):
                center = initial + [dx, dy]
                score = _multihorizon_score(masks, indices, center, sign)
                if best is None or score > best[0]:
                    best = (score, sign, center)
    assert best is not None
    coarse = best
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            center = coarse[2] + [dx, dy]
            score = _multihorizon_score(masks, indices, center, coarse[1])
            if score > best[0]:
                best = (score, coarse[1], center)
    return best[2], best[1], best[0]


def estimate_rotation_model(masks: np.ndarray | None = None) -> RotationModel:
    masks = load_masks() if masks is None else np.asarray(masks, bool)
    centroids = _mask_centroids(masks)
    initial = centroids.mean(axis=0)
    indices = list(range(30))
    center, sign, _ = _refine_center(masks, indices, initial)
    score = _mean_pair_iou(masks, indices, center, sign, hop=1)
    even = [index for index in indices if index % 2 == 0]
    odd = [index for index in indices if index % 2 == 1]
    even_center, _, _ = _refine_center(masks, even, center, signs=(sign,))
    odd_center, _, _ = _refine_center(masks, odd, center, signs=(sign,))
    split_distance = float(np.linalg.norm(even_center - odd_center))
    geometry_ok = bool(score >= 0.95 and split_distance <= 3.0)
    return RotationModel(
        center_x_px=float(center[0]),
        center_y_px=float(center[1]),
        sign=int(sign),
        mean_iou=float(score),
        even_center_x_px=float(even_center[0]),
        even_center_y_px=float(even_center[1]),
        odd_center_x_px=float(odd_center[0]),
        odd_center_y_px=float(odd_center[1]),
        split_center_distance_px=split_distance,
        geometry_ok=geometry_ok,
    )


def transport(
    coords_norm: np.ndarray,
    k_frames: int,
    rotation: RotationModel,
) -> np.ndarray:
    pixels = to_px(coords_norm)
    angle = math.radians(rotation.sign * 2.0 * k_frames)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    relative = pixels - rotation.center_px
    out = np.empty_like(relative)
    out[..., 0] = cos_a * relative[..., 0] - sin_a * relative[..., 1]
    out[..., 1] = sin_a * relative[..., 0] + cos_a * relative[..., 1]
    return to_norm(out + rotation.center_px)


def highpass_residual(coords: np.ndarray, window: int = 9) -> np.ndarray:
    """Frozen Block-0 proxy retained only for comparability with E7b."""
    coords = np.asarray(coords, dtype=np.float64)
    pad = window // 2
    padded = np.concatenate([coords[:pad][::-1], coords, coords[-pad:][::-1]], axis=0)
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.stack(
        [
            np.stack(
                [np.convolve(padded[:, keypoint, axis], kernel, mode="valid") for axis in range(2)],
                axis=-1,
            )
            for keypoint in range(coords.shape[1])
        ],
        axis=1,
    )
    return coords - smooth


def derotate(coords: np.ndarray, rotation: RotationModel) -> np.ndarray:
    result = []
    for frame_index, points in enumerate(coords):
        result.append(transport(points, -frame_index, rotation))
    return np.asarray(result)


def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return linear_sum_assignment(np.asarray(cost))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_keypoints(
    image: Image.Image,
    coords_px: np.ndarray,
    *,
    source_size: int = IMG_SIZE,
) -> Image.Image:
    """Match visualize.py's sweep markers on an arbitrary-sized RGB panel."""
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    scale_x = image.width / source_size
    scale_y = image.height / source_size
    marker_radius = max(4, round(6 * min(scale_x, scale_y)))
    label_font = _font(max(10, round(11 * min(scale_x, scale_y))))
    for keypoint, (x_coord, y_coord) in enumerate(np.asarray(coords_px)):
        color = COLORS[keypoint % len(COLORS)]
        x_coord, y_coord = x_coord * scale_x, y_coord * scale_y
        draw.ellipse(
            [
                x_coord - marker_radius,
                y_coord - marker_radius,
                x_coord + marker_radius,
                y_coord + marker_radius,
            ],
            fill=color,
            outline="white",
            width=2,
        )
        label = str(keypoint)
        label_x = int(x_coord - marker_radius * 0.7)
        label_y = int(y_coord - marker_radius - 15 * min(scale_x, scale_y))
        bbox = draw.textbbox((label_x, label_y), label, font=label_font)
        draw.rounded_rectangle(
            [bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2],
            radius=3,
            fill=(*ImageColor.getrgb(color), 190),
            outline=(30, 30, 30, 220),
            width=1,
        )
        draw.text((label_x, label_y), label, fill="white", font=label_font)
    return image.convert("RGB")


def overlay(frame_index: int, coords_px: np.ndarray, path: str | Path) -> Path:
    """Untouched RGB with the exact marker semantics used by sweep figures."""
    image = Image.open(frame_files()[frame_index]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image = draw_keypoints(image, coords_px)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def write_rotation_report(rotation: RotationModel, path: Path | None = None) -> Path:
    output_path = path or (OUTPUTS / "geometry_gate.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rotation.to_dict(), indent=2))
    return output_path
