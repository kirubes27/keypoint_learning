from __future__ import annotations

import numpy as np
import pytest

from keypoint_net.sift_bridge import (
    SiftBridgeConfig,
    SiftBridgeError,
    SiftDetections,
    assign_descriptor_banks,
    fit_from_detections,
    predict_from_detections,
    rootsift,
    suppress_seed_neighbours,
)
from keypoint_net.sift_bridge_calibration import build_lock


def detections(
    descriptors: np.ndarray,
    *,
    xy: np.ndarray | None = None,
    response: np.ndarray | None = None,
) -> SiftDetections:
    values = np.asarray(descriptors, dtype=np.float64)
    n = values.shape[0]
    if xy is None:
        xy = np.stack((np.arange(n) * 12.0 + 10.0, np.arange(n) * 3.0 + 20.0), axis=1)
    if response is None:
        response = np.linspace(1.0, 0.1, n)
    return SiftDetections(
        xy_px=np.asarray(xy, dtype=np.float64),
        descriptors=values,
        response=np.asarray(response, dtype=np.float64),
        size=np.full(n, 4.0, dtype=np.float64),
        angle_deg=np.zeros(n, dtype=np.float64),
    )


def distinct_descriptors(n: int) -> np.ndarray:
    values = np.zeros((n, 128), dtype=np.float64)
    values[np.arange(n), np.arange(n)] = 1.0
    return values


def test_rootsift_has_unit_l2_norm() -> None:
    raw = np.asarray([[1.0, 3.0] + [0.0] * 126, [4.0, 0.0] + [0.0] * 126])
    transformed = rootsift(raw)
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0)
    np.testing.assert_allclose(transformed[0, :2], [0.5, np.sqrt(0.75)])


def test_rootsift_rejects_zero_descriptor() -> None:
    with pytest.raises(SiftBridgeError, match="zero L1 norm"):
        rootsift(np.zeros((1, 128), dtype=np.float64))


def test_seed_suppression_is_response_first_and_deterministic() -> None:
    desc = distinct_descriptors(4)
    observed = detections(
        desc,
        xy=np.asarray([[10.0, 10.0], [12.0, 10.0], [30.0, 10.0], [50.0, 10.0]]),
        response=np.asarray([0.3, 0.9, 0.8, 0.7]),
    )
    first = suppress_seed_neighbours(observed, minimum_separation_px=8.0)
    second = suppress_seed_neighbours(observed, minimum_separation_px=8.0)
    np.testing.assert_array_equal(first, [1, 2, 3])
    np.testing.assert_array_equal(first, second)


def test_assignment_is_one_to_one_and_rejects_ambiguous_identity() -> None:
    base = distinct_descriptors(3)
    clear = detections(base.copy())
    accepted = assign_descriptor_banks(
        tuple(base[index : index + 1] for index in range(3)),
        clear,
        lowe_ratio=0.8,
    )
    assert accepted.accepted.tolist() == [True, True, True]
    assert len(set(accepted.detection_index.tolist())) == 3

    ambiguous_banks = (base[0:1], base[0:1], base[2:3])
    ambiguous = assign_descriptor_banks(ambiguous_banks, clear, lowe_ratio=0.8)
    assert not ambiguous.accepted[0]
    assert not ambiguous.accepted[1]
    assert ambiguous.accepted[2]


def test_empty_frame_remains_missing_without_filling() -> None:
    base = distinct_descriptors(10)
    empty = detections(np.empty((0, 128), dtype=np.float64))
    assignment = assign_descriptor_banks(
        tuple(base[index : index + 1] for index in range(10)),
        empty,
        lowe_ratio=0.8,
    )
    assert not np.any(assignment.accepted)
    assert np.all(assignment.detection_index == -1)


def test_fit_uses_exact_train_membership_and_prediction_is_stateless() -> None:
    base = distinct_descriptors(12)
    train_indices = (27, 28, 29)
    by_frame = {
        frame: detections(base.copy(), response=np.linspace(1.2, 0.1, 12))
        for frame in train_indices
    }
    config = SiftBridgeConfig(n_identities=10, seed_frame_index=27)
    model = fit_from_detections(by_frame, train_indices, config)
    assert model.seed_candidate_indices.tolist() == list(range(10))
    np.testing.assert_allclose(model.train_coverage, 1.0)

    target = detections(base.copy())
    first_coordinates, first_assignment = predict_from_detections(model, target)
    _ = predict_from_detections(model, detections(base[::-1].copy()))
    second_coordinates, second_assignment = predict_from_detections(model, target)
    np.testing.assert_allclose(first_coordinates, second_coordinates)
    np.testing.assert_array_equal(first_assignment.accepted, second_assignment.accepted)

    with pytest.raises(SiftBridgeError, match="exactly the train frames"):
        fit_from_detections({27: by_frame[27], 28: by_frame[28]}, train_indices, config)


def test_full_resolution_calibration_passes_clean_and_rejects_two_pixel_spike() -> None:
    parent = {
        "schema_version": "frozen_wobble_oracle_calibration.v1_2",
        "all_semantic_assertions_pass": True,
        "frozen_thresholds": {
            "activity": {
                "minimum_raw_orbit_rms_normalized": 1.0 / 63.0,
            },
            "grounding_and_distinctness": {
                "minimum_fixed_channel_pair_distance_normalized": 2.0 / 63.0,
                "minimum_image_border_distance_px": 4.055555555555555,
                "required_on_object_rate": 1.0,
            }
        },
    }
    lock = build_lock(parent, parent_sha256="a" * 64)
    assert lock["all_semantic_assertions_pass"]
    assert all(lock["semantic_assertions"].values())
    assert lock["thresholds"]["maximum_material_error_px"] > np.sqrt(2.0)
    assert lock["planted_two_pixel_spike"]["maximum_material_error_px"] > lock[
        "thresholds"
    ]["maximum_material_error_px"]
