"""RGB-only rigid-silhouette pose primitives for material-point transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np


IMAGE_SIZE = 512
CORNER_SIZE = 16
MINIMUM_SELECTED_IOU = 0.95


class GlobalSilhouetteError(ValueError):
    """Raised when RGB does not expose an unambiguous rigid silhouette."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GlobalSilhouetteError(message)


@dataclass(frozen=True)
class SilhouetteObservation:
    mask: np.ndarray
    background_rgb: np.ndarray
    otsu_threshold: float
    component_label: int
    component_area: int
    centroid_xy: np.ndarray
    second_moment: complex
    anisotropy: float


def _corner_pixels(rgb: np.ndarray) -> np.ndarray:
    size = CORNER_SIZE
    return np.concatenate(
        (
            rgb[:size, :size].reshape(-1, 3),
            rgb[:size, -size:].reshape(-1, 3),
            rgb[-size:, :size].reshape(-1, 3),
            rgb[-size:, -size:].reshape(-1, 3),
        ),
        axis=0,
    )


def extract_silhouette(rgb: np.ndarray) -> SilhouetteObservation:
    """Extract the largest non-border component using corner colour and Otsu."""

    image = np.asarray(rgb)
    require(image.ndim == 3 and image.shape[2] == 3, "RGB shape differs")
    require(image.dtype == np.uint8, "RGB must be uint8")
    height, width = image.shape[:2]
    require(height >= 2 * CORNER_SIZE and width >= 2 * CORNER_SIZE, "RGB is too small")

    background = np.median(_corner_pixels(image), axis=0).astype(np.float64)
    difference = np.max(
        np.abs(image.astype(np.float64) - background.reshape(1, 1, 3)), axis=2
    )
    distance = np.clip(np.rint(difference), 0, 255).astype(np.uint8)
    otsu_threshold, binary = cv2.threshold(
        distance, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S
    )
    candidates: list[tuple[int, int]] = []
    for label in range(1, int(count)):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if not touches_border:
            candidates.append((area, label))
    require(bool(candidates), "no non-border foreground component remains")
    candidates.sort(key=lambda row: (-row[0], row[1]))
    area, selected_label = candidates[0]
    mask = labels == selected_label
    yy, xx = np.nonzero(mask)
    require(xx.size == area and area > 0, "component area differs")
    centroid = np.asarray([np.mean(xx), np.mean(yy)], dtype=np.float64)
    dx = xx.astype(np.float64) - centroid[0]
    dy = yy.astype(np.float64) - centroid[1]
    sxx = float(np.sum(dx * dx, dtype=np.float64))
    syy = float(np.sum(dy * dy, dtype=np.float64))
    sxy = float(np.sum(dx * dy, dtype=np.float64))
    second_moment = complex(sxx - syy, 2.0 * sxy)
    anisotropy = float(abs(second_moment))
    require(anisotropy > 0.0 and math.isfinite(anisotropy), "silhouette is isotropic")
    return SilhouetteObservation(
        mask=np.ascontiguousarray(mask),
        background_rgb=background,
        otsu_threshold=float(otsu_threshold),
        component_label=int(selected_label),
        component_area=int(area),
        centroid_xy=centroid,
        second_moment=second_moment,
        anisotropy=anisotropy,
    )


def normalize_angle(angle_rad: float) -> float:
    value = (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if value == -math.pi and angle_rad > 0.0 else value


def rigid_matrix(
    angle_rad: float,
    source_centroid_xy: np.ndarray,
    target_centroid_xy: np.ndarray,
) -> np.ndarray:
    """Return an x/y source-to-target affine matrix in image coordinates."""

    source = np.asarray(source_centroid_xy, dtype=np.float64)
    target = np.asarray(target_centroid_xy, dtype=np.float64)
    require(source.shape == (2,) and target.shape == (2,), "centroid shape differs")
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target - rotation @ source
    return np.concatenate((rotation, translation[:, None]), axis=1)


def warp_mask(mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=np.float32)
    height, width = value.shape
    warped = cv2.warpAffine(
        value,
        np.asarray(matrix, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.ascontiguousarray(warped >= 0.5)


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=bool)
    right = np.asarray(second, dtype=bool)
    require(left.shape == right.shape, "mask shapes differ")
    union = int(np.count_nonzero(left | right))
    require(union > 0, "mask union is empty")
    return float(np.count_nonzero(left & right) / union)


def decode_pose(
    reference: SilhouetteObservation,
    target: SilhouetteObservation,
) -> dict[str, np.ndarray | float | int]:
    candidates = pose_candidates(reference, target)
    candidate_ious = candidates["candidate_iou"]
    require(candidate_ious[0] != candidate_ious[1], "pi orientation is ambiguous")
    selected = int(np.argmax(candidate_ious))
    return {
        **candidates,
        "selected_index": selected,
        "selected_angle_rad": float(candidates["candidate_angle_rad"][selected]),
        "selected_iou": float(candidate_ious[selected]),
        "ambiguity_gap_iou": float(abs(candidate_ious[0] - candidate_ious[1])),
        "matrix": candidates["candidate_matrix"][selected],
    }


def pose_candidates(
    reference: SilhouetteObservation,
    target: SilhouetteObservation,
) -> dict[str, np.ndarray]:
    phase = math.atan2(
        (target.second_moment * reference.second_moment.conjugate()).imag,
        (target.second_moment * reference.second_moment.conjugate()).real,
    )
    base = 0.5 * phase
    candidate_angles = np.asarray(
        [normalize_angle(base), normalize_angle(base + math.pi)], dtype=np.float64
    )
    candidate_matrices = np.stack(
        [
            rigid_matrix(angle, reference.centroid_xy, target.centroid_xy)
            for angle in candidate_angles
        ]
    )
    candidate_ious = np.asarray(
        [
            intersection_over_union(warp_mask(reference.mask, matrix), target.mask)
            for matrix in candidate_matrices
        ],
        dtype=np.float64,
    )
    return {
        "candidate_angle_rad": candidate_angles,
        "candidate_iou": candidate_ious,
        "candidate_matrix": candidate_matrices,
    }


def select_temporal_candidate(
    previous_unwrapped_angle_rad: float,
    candidate_angles_rad: np.ndarray,
) -> tuple[int, float, float]:
    candidates = np.asarray(candidate_angles_rad, dtype=np.float64)
    require(candidates.shape == (2,), "temporal candidate shape differs")
    deltas = np.asarray(
        [normalize_angle(float(value) - previous_unwrapped_angle_rad) for value in candidates],
        dtype=np.float64,
    )
    magnitudes = np.abs(deltas)
    require(magnitudes[0] != magnitudes[1], "temporal pi branch is tied")
    selected = int(np.argmin(magnitudes))
    require(magnitudes[selected] < 0.5 * math.pi, "temporal angular step is not below pi/2")
    unwrapped = float(previous_unwrapped_angle_rad + deltas[selected])
    return selected, unwrapped, float(deltas[selected])


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2, "point shape differs")
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
    )
    return homogeneous @ np.asarray(matrix, dtype=np.float64).T


def decode_sequence(
    rgbs: list[np.ndarray],
    initial_points_xy: np.ndarray,
    frame_order: Iterable[int],
) -> dict[str, np.ndarray]:
    frame_count = len(rgbs)
    order = [int(value) for value in frame_order]
    require(sorted(order) == list(range(frame_count)), "frame order is not a permutation")
    points = np.asarray(initial_points_xy, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2, "initial point shape differs")

    observations: list[SilhouetteObservation | None] = [None] * frame_count
    for frame in order:
        observations[frame] = extract_silhouette(rgbs[frame])
    require(all(value is not None for value in observations), "silhouette extraction is incomplete")
    reference = observations[0]
    assert reference is not None

    height, width = reference.mask.shape
    masks = np.empty((frame_count, height, width), dtype=bool)
    background = np.empty((frame_count, 3), dtype=np.float64)
    otsu = np.empty(frame_count, dtype=np.float64)
    label = np.empty(frame_count, dtype=np.int64)
    area = np.empty(frame_count, dtype=np.int64)
    centroid = np.empty((frame_count, 2), dtype=np.float64)
    moment = np.empty((frame_count, 2), dtype=np.float64)
    anisotropy = np.empty(frame_count, dtype=np.float64)
    candidate_angle = np.empty((frame_count, 2), dtype=np.float64)
    candidate_iou = np.empty((frame_count, 2), dtype=np.float64)
    selected_index = np.empty(frame_count, dtype=np.int64)
    selected_angle = np.empty(frame_count, dtype=np.float64)
    selected_iou = np.empty(frame_count, dtype=np.float64)
    ambiguity_gap = np.empty(frame_count, dtype=np.float64)
    matrix = np.empty((frame_count, 2, 3), dtype=np.float64)
    prediction = np.empty((frame_count, points.shape[0], 2), dtype=np.float64)

    for frame in order:
        observation = observations[frame]
        assert observation is not None
        pose = decode_pose(reference, observation)
        masks[frame] = observation.mask
        background[frame] = observation.background_rgb
        otsu[frame] = observation.otsu_threshold
        label[frame] = observation.component_label
        area[frame] = observation.component_area
        centroid[frame] = observation.centroid_xy
        moment[frame] = [observation.second_moment.real, observation.second_moment.imag]
        anisotropy[frame] = observation.anisotropy
        candidate_angle[frame] = pose["candidate_angle_rad"]
        candidate_iou[frame] = pose["candidate_iou"]
        selected_index[frame] = pose["selected_index"]
        selected_angle[frame] = pose["selected_angle_rad"]
        selected_iou[frame] = pose["selected_iou"]
        ambiguity_gap[frame] = pose["ambiguity_gap_iou"]
        matrix[frame] = pose["matrix"]
        prediction[frame] = transform_points(points, matrix[frame])

    return {
        "foreground_mask": masks,
        "background_rgb": background,
        "otsu_threshold": otsu,
        "component_label": label,
        "component_area": area,
        "centroid_xy": centroid,
        "second_moment_real_imag": moment,
        "anisotropy": anisotropy,
        "candidate_angle_rad": candidate_angle,
        "candidate_iou": candidate_iou,
        "selected_index": selected_index,
        "selected_angle_rad": selected_angle,
        "selected_iou": selected_iou,
        "ambiguity_gap_iou": ambiguity_gap,
        "matrix": matrix,
        "prediction_xy": prediction,
    }


def decode_temporally_unwrapped_sequence(
    rgbs: list[np.ndarray],
    initial_points_xy: np.ndarray,
    traversal: Iterable[int],
) -> dict[str, np.ndarray]:
    frame_count = len(rgbs)
    order = [int(value) for value in traversal]
    require(len(order) == frame_count and order[0] == 0, "traversal must start at frame zero")
    require(sorted(order) == list(range(frame_count)), "traversal is not a permutation")
    points = np.asarray(initial_points_xy, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2, "initial point shape differs")

    observations: list[SilhouetteObservation | None] = [None] * frame_count
    for frame in order:
        observations[frame] = extract_silhouette(rgbs[frame])
    require(all(value is not None for value in observations), "silhouette extraction is incomplete")
    reference = observations[0]
    assert reference is not None

    height, width = reference.mask.shape
    masks = np.empty((frame_count, height, width), dtype=bool)
    background = np.empty((frame_count, 3), dtype=np.float64)
    otsu = np.empty(frame_count, dtype=np.float64)
    label = np.empty(frame_count, dtype=np.int64)
    area = np.empty(frame_count, dtype=np.int64)
    centroid = np.empty((frame_count, 2), dtype=np.float64)
    moment = np.empty((frame_count, 2), dtype=np.float64)
    anisotropy = np.empty(frame_count, dtype=np.float64)
    candidate_angle = np.empty((frame_count, 2), dtype=np.float64)
    candidate_iou = np.empty((frame_count, 2), dtype=np.float64)
    candidate_matrix = np.empty((frame_count, 2, 2, 3), dtype=np.float64)

    for frame in order:
        observation = observations[frame]
        assert observation is not None
        candidates = pose_candidates(reference, observation)
        masks[frame] = observation.mask
        background[frame] = observation.background_rgb
        otsu[frame] = observation.otsu_threshold
        label[frame] = observation.component_label
        area[frame] = observation.component_area
        centroid[frame] = observation.centroid_xy
        moment[frame] = [observation.second_moment.real, observation.second_moment.imag]
        anisotropy[frame] = observation.anisotropy
        candidate_angle[frame] = candidates["candidate_angle_rad"]
        candidate_iou[frame] = candidates["candidate_iou"]
        candidate_matrix[frame] = candidates["candidate_matrix"]

    selected_index = np.empty(frame_count, dtype=np.int64)
    selected_angle = np.empty(frame_count, dtype=np.float64)
    step_delta = np.empty(frame_count, dtype=np.float64)
    selected_iou = np.empty(frame_count, dtype=np.float64)
    matrix = np.empty((frame_count, 2, 3), dtype=np.float64)
    prediction = np.empty((frame_count, points.shape[0], 2), dtype=np.float64)

    selected_index[0] = 0
    selected_angle[0] = 0.0
    step_delta[0] = 0.0
    selected_iou[0] = candidate_iou[0, 0]
    matrix[0] = rigid_matrix(0.0, reference.centroid_xy, reference.centroid_xy)
    prediction[0] = transform_points(points, matrix[0])
    previous_angle = 0.0
    for frame in order[1:]:
        selected, unwrapped, delta = select_temporal_candidate(
            previous_angle, candidate_angle[frame]
        )
        selected_index[frame] = selected
        selected_angle[frame] = unwrapped
        step_delta[frame] = delta
        selected_iou[frame] = candidate_iou[frame, selected]
        observation = observations[frame]
        assert observation is not None
        matrix[frame] = rigid_matrix(
            unwrapped, reference.centroid_xy, observation.centroid_xy
        )
        prediction[frame] = transform_points(points, matrix[frame])
        previous_angle = unwrapped

    closure_delta = float(normalize_angle(-previous_angle))
    require(abs(closure_delta) < 0.5 * math.pi, "cyclic angular closure is not below pi/2")
    return {
        "foreground_mask": masks,
        "background_rgb": background,
        "otsu_threshold": otsu,
        "component_label": label,
        "component_area": area,
        "centroid_xy": centroid,
        "second_moment_real_imag": moment,
        "anisotropy": anisotropy,
        "candidate_angle_rad": candidate_angle,
        "candidate_iou": candidate_iou,
        "candidate_matrix": candidate_matrix,
        "selected_index": selected_index,
        "selected_angle_unwrapped_rad": selected_angle,
        "selected_iou_diagnostic": selected_iou,
        "traversal_step_delta_rad": step_delta,
        "closure_delta_rad": np.asarray(closure_delta, dtype=np.float64),
        "matrix": matrix,
        "prediction_xy": prediction,
    }


__all__ = [
    "CORNER_SIZE",
    "GlobalSilhouetteError",
    "IMAGE_SIZE",
    "MINIMUM_SELECTED_IOU",
    "SilhouetteObservation",
    "decode_pose",
    "decode_sequence",
    "decode_temporally_unwrapped_sequence",
    "extract_silhouette",
    "intersection_over_union",
    "normalize_angle",
    "pose_candidates",
    "rigid_matrix",
    "select_temporal_candidate",
    "transform_points",
    "warp_mask",
]
