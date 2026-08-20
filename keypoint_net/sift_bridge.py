"""Frozen, stateless SIFT detector/descriptor bridge.

The point source in this module sees RGB only.  It never consumes masks,
physical angles, learned coordinates, operator output, or another frame's
prediction.  Template fitting is restricted to an explicit train-frame list;
prediction for each frame is an independent one-to-one descriptor assignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class SiftBridgeError(ValueError):
    """Raised when the frozen bridge contract cannot be satisfied."""


@dataclass(frozen=True)
class SiftBridgeConfig:
    """R1 values frozen before holdout/guard evaluation."""

    n_identities: int = 10
    seed_frame_index: int = 27
    seed_separation_px: float = 8.0
    lowe_ratio: float = 0.80
    nfeatures: int = 0
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6

    def validate(self) -> None:
        if self.n_identities <= 0:
            raise SiftBridgeError("n_identities must be positive")
        if self.seed_frame_index < 0:
            raise SiftBridgeError("seed_frame_index must be non-negative")
        if not np.isfinite(self.seed_separation_px) or self.seed_separation_px <= 0:
            raise SiftBridgeError("seed_separation_px must be finite and positive")
        if not np.isfinite(self.lowe_ratio) or not 0 < self.lowe_ratio < 1:
            raise SiftBridgeError("lowe_ratio must lie strictly between zero and one")


@dataclass(frozen=True)
class SiftDetections:
    """SIFT observations from one RGB image."""

    xy_px: np.ndarray
    descriptors: np.ndarray
    response: np.ndarray
    size: np.ndarray
    angle_deg: np.ndarray

    def validate(self) -> None:
        points = np.asarray(self.xy_px)
        descriptors = np.asarray(self.descriptors)
        if points.ndim != 2 or points.shape[1] != 2:
            raise SiftBridgeError("xy_px must have shape (detections, 2)")
        if descriptors.ndim != 2 or descriptors.shape != (points.shape[0], 128):
            raise SiftBridgeError("descriptors must have shape (detections, 128)")
        for name, value in (
            ("xy_px", points),
            ("descriptors", descriptors),
            ("response", self.response),
            ("size", self.size),
            ("angle_deg", self.angle_deg),
        ):
            array = np.asarray(value)
            if name not in {"xy_px", "descriptors"} and array.shape != (points.shape[0],):
                raise SiftBridgeError(f"{name} must have one value per detection")
            if not np.isfinite(array).all():
                raise SiftBridgeError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class Assignment:
    """One independent frame's assignment to a fixed identity set."""

    detection_index: np.ndarray
    accepted: np.ndarray
    distance: np.ndarray
    row_ratio: np.ndarray
    column_ratio: np.ndarray
    mutual_nearest: np.ndarray

    def validate(self, n_identities: int) -> None:
        for name, value in asdict(self).items():
            array = np.asarray(value)
            if array.shape != (n_identities,):
                raise SiftBridgeError(f"assignment field {name} has the wrong shape")
        used = self.detection_index[self.accepted]
        if np.unique(used).size != used.size:
            raise SiftBridgeError("accepted assignment is not one-to-one")


@dataclass(frozen=True)
class FrozenSiftBridge:
    """Train-only descriptor identities frozen for stateless inference."""

    config: SiftBridgeConfig
    train_frame_indices: tuple[int, ...]
    seed_candidate_indices: np.ndarray
    seed_xy_px: np.ndarray
    seed_response: np.ndarray
    train_coverage: np.ndarray
    train_median_ratio: np.ndarray
    descriptor_banks: tuple[np.ndarray, ...]

    def validate(self) -> None:
        self.config.validate()
        n = self.config.n_identities
        if len(self.train_frame_indices) == 0:
            raise SiftBridgeError("train_frame_indices is empty")
        if tuple(sorted(set(self.train_frame_indices))) != self.train_frame_indices:
            raise SiftBridgeError("train_frame_indices must be sorted and unique")
        for name, value, shape in (
            ("seed_candidate_indices", self.seed_candidate_indices, (n,)),
            ("seed_xy_px", self.seed_xy_px, (n, 2)),
            ("seed_response", self.seed_response, (n,)),
            ("train_coverage", self.train_coverage, (n,)),
            ("train_median_ratio", self.train_median_ratio, (n,)),
        ):
            array = np.asarray(value)
            if array.shape != shape:
                raise SiftBridgeError(f"{name} must have shape {shape}")
            if not np.isfinite(array).all():
                raise SiftBridgeError(f"{name} contains non-finite values")
        if len(self.descriptor_banks) != n:
            raise SiftBridgeError("there must be one descriptor bank per identity")
        for bank in self.descriptor_banks:
            array = np.asarray(bank)
            if array.ndim != 2 or array.shape[1] != 128 or array.shape[0] == 0:
                raise SiftBridgeError("each descriptor bank must be non-empty (rows, 128)")
            if not np.isfinite(array).all():
                raise SiftBridgeError("descriptor bank contains non-finite values")


def _empty_detections() -> SiftDetections:
    return SiftDetections(
        xy_px=np.empty((0, 2), dtype=np.float64),
        descriptors=np.empty((0, 128), dtype=np.float64),
        response=np.empty((0,), dtype=np.float64),
        size=np.empty((0,), dtype=np.float64),
        angle_deg=np.empty((0,), dtype=np.float64),
    )


def rootsift(descriptors: np.ndarray) -> np.ndarray:
    """Convert OpenCV SIFT descriptors to deterministic RootSIFT vectors."""
    raw = np.asarray(descriptors, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 128:
        raise SiftBridgeError("SIFT descriptors must have shape (rows, 128)")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise SiftBridgeError("SIFT descriptors must be finite and non-negative")
    if raw.shape[0] == 0:
        return raw.copy()
    denominator = np.sum(raw, axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise SiftBridgeError("SIFT descriptor has zero L1 norm")
    transformed = np.sqrt(raw / denominator)
    if not np.isfinite(transformed).all():
        raise SiftBridgeError("RootSIFT conversion produced a non-finite value")
    return transformed


def create_detector(config: SiftBridgeConfig) -> cv2.SIFT:
    """Create the exact frozen CPU SIFT detector."""
    config.validate()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    return cv2.SIFT_create(
        nfeatures=config.nfeatures,
        nOctaveLayers=config.n_octave_layers,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
        sigma=config.sigma,
    )


def detect_rgb(image_rgb: np.ndarray, detector: cv2.SIFT) -> SiftDetections:
    """Detect one frame without reading or retaining any temporal state."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise SiftBridgeError("image_rgb must be uint8 with shape (height, width, 3)")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, raw_descriptors = detector.detectAndCompute(gray, None)
    if not keypoints or raw_descriptors is None:
        return _empty_detections()
    detections = SiftDetections(
        xy_px=np.asarray([point.pt for point in keypoints], dtype=np.float64),
        descriptors=rootsift(raw_descriptors),
        response=np.asarray([point.response for point in keypoints], dtype=np.float64),
        size=np.asarray([point.size for point in keypoints], dtype=np.float64),
        angle_deg=np.asarray([point.angle for point in keypoints], dtype=np.float64),
    )
    detections.validate()
    return detections


def suppress_seed_neighbours(
    detections: SiftDetections,
    *,
    minimum_separation_px: float,
) -> np.ndarray:
    """Return deterministic seed rows after spatial duplicate suppression."""
    detections.validate()
    if not np.isfinite(minimum_separation_px) or minimum_separation_px <= 0:
        raise SiftBridgeError("minimum_separation_px must be finite and positive")
    if detections.xy_px.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    order = sorted(
        range(detections.xy_px.shape[0]),
        key=lambda index: (
            -float(detections.response[index]),
            float(detections.xy_px[index, 0]),
            float(detections.xy_px[index, 1]),
            float(detections.size[index]),
            float(detections.angle_deg[index]),
            index,
        ),
    )
    retained: list[int] = []
    for index in order:
        point = detections.xy_px[index]
        if all(
            float(np.linalg.norm(point - detections.xy_px[prior]))
            >= minimum_separation_px
            for prior in retained
        ):
            retained.append(index)
    return np.asarray(retained, dtype=np.int64)


def _bank_costs(
    descriptor_banks: Sequence[np.ndarray],
    detections: SiftDetections,
) -> np.ndarray:
    detections.validate()
    if len(descriptor_banks) == 0:
        raise SiftBridgeError("descriptor_banks is empty")
    costs = np.empty((len(descriptor_banks), detections.descriptors.shape[0]), dtype=np.float64)
    for identity, bank in enumerate(descriptor_banks):
        values = np.asarray(bank, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 128 or values.shape[0] == 0:
            raise SiftBridgeError("descriptor bank must be non-empty with width 128")
        distances = np.linalg.norm(
            values[:, None, :] - detections.descriptors[None, :, :], axis=-1
        )
        costs[identity] = np.min(distances, axis=0)
    return costs


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("inf")
    if denominator <= 0:
        # Two exact-zero candidates are maximally ambiguous, not a perfect
        # unique match.  Returning one makes the strict ratio gate reject it.
        return 1.0 if numerator <= 0 else float("inf")
    return float(numerator / denominator)


def assign_descriptor_banks(
    descriptor_banks: Sequence[np.ndarray],
    detections: SiftDetections,
    *,
    lowe_ratio: float,
) -> Assignment:
    """Assign identities one-to-one with mutual and two-sided ratio gates."""
    if not np.isfinite(lowe_ratio) or not 0 < lowe_ratio < 1:
        raise SiftBridgeError("lowe_ratio must lie strictly between zero and one")
    n_identities = len(descriptor_banks)
    n_detections = detections.xy_px.shape[0]
    detection_index = np.full(n_identities, -1, dtype=np.int64)
    accepted = np.zeros(n_identities, dtype=bool)
    distance = np.full(n_identities, np.inf, dtype=np.float64)
    row_ratio = np.full(n_identities, np.inf, dtype=np.float64)
    column_ratio = np.full(n_identities, np.inf, dtype=np.float64)
    mutual = np.zeros(n_identities, dtype=bool)
    if n_detections == 0:
        result = Assignment(
            detection_index=detection_index,
            accepted=accepted,
            distance=distance,
            row_ratio=row_ratio,
            column_ratio=column_ratio,
            mutual_nearest=mutual,
        )
        result.validate(n_identities)
        return result

    costs = _bank_costs(descriptor_banks, detections)
    row_indices, column_indices = linear_sum_assignment(costs)
    for identity, detection in zip(row_indices.tolist(), column_indices.tolist()):
        detection_index[identity] = detection
        assigned_cost = float(costs[identity, detection])
        distance[identity] = assigned_cost

        row_order = np.argsort(costs[identity], kind="stable")
        row_best = int(row_order[0])
        row_second = float(costs[identity, row_order[1]]) if n_detections > 1 else np.inf
        row_ratio[identity] = _safe_ratio(assigned_cost, row_second)

        column_order = np.argsort(costs[:, detection], kind="stable")
        column_best = int(column_order[0])
        column_second = (
            float(costs[column_order[1], detection]) if n_identities > 1 else np.inf
        )
        column_ratio[identity] = _safe_ratio(assigned_cost, column_second)

        mutual[identity] = row_best == detection and column_best == identity
        accepted[identity] = bool(
            mutual[identity]
            and row_ratio[identity] <= lowe_ratio
            and column_ratio[identity] <= lowe_ratio
        )

    result = Assignment(
        detection_index=detection_index,
        accepted=accepted,
        distance=distance,
        row_ratio=row_ratio,
        column_ratio=column_ratio,
        mutual_nearest=mutual,
    )
    result.validate(n_identities)
    return result


def fit_from_detections(
    detections_by_frame: Mapping[int, SiftDetections],
    train_frame_indices: Iterable[int],
    config: SiftBridgeConfig,
) -> FrozenSiftBridge:
    """Fit ten frozen identities using descriptor evidence from train only."""
    config.validate()
    train = tuple(sorted(set(int(index) for index in train_frame_indices)))
    if not train or any(index < 0 for index in train):
        raise SiftBridgeError("train_frame_indices must be non-empty and non-negative")
    if config.seed_frame_index not in train:
        raise SiftBridgeError("seed frame is not in the train partition")
    if set(detections_by_frame) != set(train):
        raise SiftBridgeError("detections_by_frame must contain exactly the train frames")
    for detections in detections_by_frame.values():
        detections.validate()

    seed = detections_by_frame[config.seed_frame_index]
    retained = suppress_seed_neighbours(
        seed, minimum_separation_px=config.seed_separation_px
    )
    if retained.size < config.n_identities:
        raise SiftBridgeError(
            f"only {retained.size} separated seed detections; need {config.n_identities}"
        )
    seed_banks = tuple(seed.descriptors[index : index + 1] for index in retained)
    assignments = {
        frame: assign_descriptor_banks(
            seed_banks, detections_by_frame[frame], lowe_ratio=config.lowe_ratio
        )
        for frame in train
    }

    coverage = np.asarray(
        [
            np.mean([assignments[frame].accepted[candidate] for frame in train])
            for candidate in range(retained.size)
        ],
        dtype=np.float64,
    )
    median_ratio = np.empty(retained.size, dtype=np.float64)
    for candidate in range(retained.size):
        ratios = [
            max(
                float(assignments[frame].row_ratio[candidate]),
                float(assignments[frame].column_ratio[candidate]),
            )
            for frame in train
            if assignments[frame].accepted[candidate]
        ]
        median_ratio[candidate] = float(np.median(ratios)) if ratios else 1.0

    ranking = sorted(
        range(retained.size),
        key=lambda candidate: (
            -float(coverage[candidate]),
            float(median_ratio[candidate]),
            -float(seed.response[retained[candidate]]),
            int(retained[candidate]),
        ),
    )
    selected = np.asarray(ranking[: config.n_identities], dtype=np.int64)
    selected_seed_rows = retained[selected]

    banks: list[np.ndarray] = []
    for candidate in selected.tolist():
        rows = []
        for frame in train:
            assignment = assignments[frame]
            if assignment.accepted[candidate]:
                detection = int(assignment.detection_index[candidate])
                rows.append(detections_by_frame[frame].descriptors[detection])
        if not rows:
            raise SiftBridgeError("selected identity has an empty train descriptor bank")
        banks.append(np.asarray(rows, dtype=np.float64))

    model = FrozenSiftBridge(
        config=config,
        train_frame_indices=train,
        seed_candidate_indices=selected_seed_rows.astype(np.int64),
        seed_xy_px=seed.xy_px[selected_seed_rows].astype(np.float64),
        seed_response=seed.response[selected_seed_rows].astype(np.float64),
        train_coverage=coverage[selected].astype(np.float64),
        train_median_ratio=median_ratio[selected].astype(np.float64),
        descriptor_banks=tuple(banks),
    )
    model.validate()
    return model


def predict_from_detections(
    model: FrozenSiftBridge,
    detections: SiftDetections,
) -> tuple[np.ndarray, Assignment]:
    """Predict one frame independently; rejected identities remain NaN."""
    model.validate()
    detections.validate()
    assignment = assign_descriptor_banks(
        model.descriptor_banks,
        detections,
        lowe_ratio=model.config.lowe_ratio,
    )
    coordinates = np.full((model.config.n_identities, 2), np.nan, dtype=np.float64)
    for identity in range(model.config.n_identities):
        if assignment.accepted[identity]:
            coordinates[identity] = detections.xy_px[
                int(assignment.detection_index[identity])
            ]
    return coordinates, assignment


def config_as_dict(config: SiftBridgeConfig) -> dict[str, int | float]:
    """Return a JSON-safe exact config representation."""
    config.validate()
    return asdict(config)
