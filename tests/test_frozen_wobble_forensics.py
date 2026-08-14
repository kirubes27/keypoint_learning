from __future__ import annotations

import numpy as np

from keypoint_net.frozen_wobble_forensics import (
    canonicalize,
    coordinate_dynamics,
    deterministic_spike_frames,
    endpoint_grid,
    geometry_smoke,
    heatmap_topology,
    phase_metrics,
    raw_from_canonical,
    spatial_expectation_float32,
    warp_image_by_standard_rotation,
)
from keypoint_net.frozen_wobble_oracle_calibration import build_calibration
from keypoint_net.run_frozen_wobble_forensics import (
    EXPECTED_EVALUATION_COMMIT,
    IMPLEMENTATION_SOURCE_RELATIVE_PATHS,
    _implementation_binding,
)


def _orbit(canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(canonical.shape[0], dtype=np.float64) * 2.0
    return raw_from_canonical(canonical, theta), theta


def test_geometry_smoke_locks_physical_sign_and_pivot() -> None:
    report = geometry_smoke()
    assert report["correct_sign_rms_normalized"] < 1e-14
    assert report["wrong_sign_rms_normalized"] > 0.1
    assert report["wrong_pivot_rms_normalized"] > 0.05


def test_forensic_implementation_binding_separates_ancestor_from_current_head() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    head, source_records = _implementation_binding(repo_root)
    assert len(head) == 40
    assert head != EXPECTED_EVALUATION_COMMIT
    assert tuple(source_records) == IMPLEMENTATION_SOURCE_RELATIVE_PATHS
    assert all(record["sha256"] for record in source_records.values())


def test_exact_material_points_become_constant_after_derotation() -> None:
    frames = 180
    anchors = np.asarray(((0.05, 0.02), (0.62, -0.17), (-0.4, 0.51)))
    canonical = np.broadcast_to(anchors, (frames,) + anchors.shape).copy()
    raw, theta = _orbit(canonical)
    recovered = canonicalize(raw, theta)
    dynamics = coordinate_dynamics(recovered)
    assert np.max(np.abs(recovered - canonical)) < 1e-14
    assert np.max(dynamics["adjacent_step"]) < 1e-14
    assert np.max(dynamics["within_residue_step"]) < 1e-14
    assert np.max(dynamics["second_difference"]) < 1e-14


def test_constant_mod3_offsets_are_explained_and_removed() -> None:
    frames = 180
    theta = np.arange(frames, dtype=np.float64) * 2.0
    canonical = np.broadcast_to(np.asarray([[[0.42, -0.13]]]), (frames, 1, 2)).copy()
    offsets = np.asarray(((0.025, 0.0), (-0.0125, 0.021650635), (-0.0125, -0.021650635)))
    canonical[:, 0] += offsets[np.arange(frames) % 3]
    report = phase_metrics(canonical, theta)["channels"][0]
    assert report["period_three_position_energy_fraction"] > 0.999999
    assert report["transition_class_step_energy_fraction"] > 0.999999
    assert report["position_energy_removed_by_constant_residue_correction"] > 0.999999
    assert report["adjacent_step_energy_removed_by_constant_residue_correction"] > 0.999999
    assert report["maximum_spike_excess_over_phase_bound"] <= 0.0


def test_isolated_spike_is_not_mislabelled_as_bounded_phase() -> None:
    frames = 180
    theta = np.arange(frames, dtype=np.float64) * 2.0
    canonical = np.broadcast_to(np.asarray([[[0.42, -0.13]]]), (frames, 1, 2)).copy()
    canonical[73, 0, 0] += 0.25
    report = phase_metrics(canonical, theta)["channels"][0]
    assert report["transition_class_step_energy_fraction"] < 0.1
    assert report["position_energy_removed_by_constant_residue_correction"] < 0.1
    assert report["maximum_spike_excess_over_phase_bound"] > 0.1
    selected = deterministic_spike_frames(canonical, top_k=1)
    assert selected[0][0]["centre_frame_index"] == 73


def _blank_logits(frames: int) -> np.ndarray:
    return np.full((frames, 1, 64, 64), -20.0, dtype=np.float64)


def test_soft_mass_shift_keeps_hard_basin_but_moves_centroid() -> None:
    logits = _blank_logits(3)
    logits[:, 0, 32, 32] = 10.0
    logits[:, 0, 32, 40] = np.asarray((2.0, 8.0, 9.5))
    result = heatmap_topology(logits)
    assert np.array_equal(result["hard_x_cell"][:, 0], np.asarray((32, 32, 32)))
    assert np.ptp(result["soft_coordinate"][:, 0, 0]) > 0.05
    assert result["separated_peaks"]["4"]["second_peak_x_cell"][-1, 0] == 40
    assert result["separated_peaks"]["4"]["logit_margin"][-1, 0] == 0.5


def test_hard_basin_hop_is_distinct_from_soft_mass_shift() -> None:
    logits = _blank_logits(3)
    logits[:, 0, 32, 32] = np.asarray((10.0, 10.0, 8.0))
    logits[:, 0, 32, 40] = np.asarray((8.0, 9.5, 10.0))
    result = heatmap_topology(logits)
    assert np.array_equal(result["hard_x_cell"][:, 0], np.asarray((32, 32, 40)))
    assert result["separated_peaks"]["4"]["second_peak_x_cell"][-1, 0] == 32


def test_one_sharp_moving_peak_moves_hard_and_soft_together() -> None:
    logits = _blank_logits(3)
    for frame, x_cell in enumerate((28, 32, 36)):
        logits[frame, 0, 30, x_cell] = 15.0
    result = heatmap_topology(logits)
    assert np.array_equal(result["hard_x_cell"][:, 0], np.asarray((28, 32, 36)))
    cell = float(endpoint_grid(64)[1] - endpoint_grid(64)[0])
    soft_x = result["soft_coordinate"][:, 0, 0]
    hard_x = result["hard_coordinate"][:, 0, 0]
    assert np.max(np.abs(soft_x - hard_x)) < 1e-9
    assert np.allclose(np.diff(soft_x), 4.0 * cell, atol=1e-9, rtol=0.0)


def test_complete_model_blind_oracle_calibration_passes() -> None:
    calibration = build_calibration()
    assert calibration["model_checkpoint_opened"] is False
    assert calibration["training_or_weight_update_performed"] is False
    assert calibration["all_semantic_assertions_pass"] is True
    assert all(calibration["semantic_assertions"].values())
    hard = calibration["frozen_thresholds"]["hard_r64_quantized"]
    soft = calibration["frozen_thresholds"]["soft_single_gaussian"]
    assert calibration["schema_version"] == "frozen_wobble_oracle_calibration.v1_2"
    assert hard["adjacent_step_max"] > soft["adjacent_step_max"]
    assert hard["second_difference_max"] > soft["second_difference_max"]
    for family in calibration["planted_detection_ladders"].values():
        amplitudes = [row["amplitude_grid_steps"] for row in family]
        assert amplitudes == [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    topology = calibration["frozen_thresholds"]["single_gaussian_heatmap_topology"]
    # The reported width is radial RMS, sqrt(E[dx^2 + dy^2]); for the widest
    # clean sigma=2 Gaussian this is sqrt(2)*sigma.
    assert topology["rms_width_cells_max"] <= 2.83
    assert topology["separated_peak_radius4_logit_margin_min"] > 1.0


def test_float32_spatial_expectation_matches_production_torch() -> None:
    import torch

    from keypoint_net.model import spatial_softmax

    generator = np.random.default_rng(7331)
    logits = generator.normal(size=(4, 3, 64, 64)).astype(np.float32)
    expected = (
        spatial_softmax(torch.from_numpy(logits), temperature=1.0)
        .detach()
        .cpu()
        .numpy()
    )
    actual = spatial_expectation_float32(logits)
    assert np.max(np.abs(actual - expected)) < 2e-6


def test_mask_rotation_audit_distinguishes_sign() -> None:
    mask = np.zeros((129, 129), dtype=np.float64)
    mask[24:98, 61:69] = 1.0
    mask[77:91, 25:93] = 1.0
    forward = warp_image_by_standard_rotation(mask, 34.0, order=0) > 0.5
    recovered = warp_image_by_standard_rotation(forward, -34.0, order=0) > 0.5
    wrong = warp_image_by_standard_rotation(forward, 34.0, order=0) > 0.5

    def iou(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.count_nonzero(left & right) / np.count_nonzero(left | right))

    assert iou(mask > 0.5, recovered) > 0.95
    assert iou(mask > 0.5, recovered) > iou(mask > 0.5, wrong) + 0.5
