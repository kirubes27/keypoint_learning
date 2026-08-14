"""Focused semantic tests for adjacent RGB material observability."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from keypoint_net.rgb_material_observability import (
    RGBObservabilityConfig,
    candidate_rank,
    decode_rgb_edge,
    local_candidate_mask,
    normalized_to_pixel,
    patch_inside,
    pixel_to_normalized,
    rgb_correlation_map,
    stable_top_two,
)
from keypoint_net.evaluate_rgb_material_observability import _scope_metrics, _source_state


def textured_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((512, 512, 3), dtype=np.float32)


def test_endpoint_coordinate_round_trip() -> None:
    normalized = np.asarray(((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0), (0.25, -0.75)))
    np.testing.assert_allclose(pixel_to_normalized(normalized_to_pixel(normalized)), normalized)
    np.testing.assert_array_equal(normalized_to_pixel(normalized[:3]), [[0.0, 0.0], [255.5, 255.5], [511.0, 511.0]])


def test_patch_geometry_is_explicit() -> None:
    assert patch_inside((17.0, 17.0), 35)
    assert not patch_inside((16.9, 17.0), 35)
    assert patch_inside((52.0, 52.0), 105)
    assert not patch_inside((51.9, 52.0), 105)


def test_translation_is_recovered_globally_and_locally() -> None:
    source = textured_image(1)
    matrix = np.float32([[1.0, 0.0, 6.0], [0.0, 1.0, -4.0]])
    target = cv2.warpAffine(source, matrix, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    result = decode_rgb_edge(source, target, (250.25, 260.75), 35)
    expected = np.asarray((256.25, 256.75))
    for scope in ("global", "local"):
        assert result[scope]["valid"]
        np.testing.assert_allclose(result[scope]["top1_coordinate_px"], expected, atol=0.8)


def test_global_and_local_can_disagree_on_a_distant_exact_duplicate() -> None:
    rng = np.random.default_rng(2)
    source = np.zeros((512, 512, 3), dtype=np.float32)
    target = np.zeros_like(source)
    patch = rng.random((35, 35, 3), dtype=np.float32)
    source[183:218, 183:218] = patch
    # A noisy local copy and an exact distant copy.
    target[185:220, 186:221] = np.clip(patch + 0.03 * rng.standard_normal(patch.shape), 0.0, 1.0)
    target[383:418, 383:418] = patch
    result = decode_rgb_edge(source, target, (200.0, 200.0), 35)
    np.testing.assert_allclose(result["global"]["top1_coordinate_px"], (400.0, 400.0))
    np.testing.assert_allclose(result["local"]["top1_coordinate_px"], (203.0, 202.0))


def test_top_two_ties_are_row_major_and_spatially_separated() -> None:
    scores = np.zeros((478, 478), dtype=np.float32)
    scores[10, 20] = 0.9
    scores[10, 21] = 0.9
    scores[80, 90] = 0.8
    top = stable_top_two(scores, 35, exclusion_radius_px=32.0)
    np.testing.assert_array_equal(top["top1_coordinate_px"], [37.0, 27.0])
    np.testing.assert_array_equal(top["top2_coordinate_px"], [107.0, 97.0])
    assert top["margin"] == pytest.approx(0.1)


def test_candidate_rank_uses_stable_tie_order() -> None:
    scores = np.zeros((478, 478), dtype=np.float32)
    scores[0, 0] = 0.5
    scores[0, 1] = 0.5
    rank = candidate_rank(scores, 35, (18.0, 17.0))
    assert rank["coordinate_px"] == [18.0, 17.0]
    assert rank["rank"] == 2


def test_local_mask_has_frozen_radius() -> None:
    mask = local_candidate_mask((478, 478), 35, (255.5, 255.5))
    yy, xx = np.where(mask)
    centres_x = xx + 17.0
    centres_y = yy + 17.0
    assert np.max(np.abs(centres_x - 255.5)) <= 32.0
    assert np.max(np.abs(centres_y - 255.5)) <= 32.0
    assert mask.sum() == 64 * 64


def test_uniform_query_is_invalid_not_a_false_perfect_match() -> None:
    image = np.zeros((512, 512, 3), dtype=np.float32)
    scores, evidence = rgb_correlation_map(image, image, (255.5, 255.5), 35)
    assert scores is None
    assert evidence["source_patch_inside_and_informative"] is False


def test_opencv_multichannel_score_matches_manual_per_channel_zncc() -> None:
    source = textured_image(12)
    target = textured_image(13)
    scores, evidence = rgb_correlation_map(source, target, (100.0, 100.0), 35)
    assert scores is not None and evidence["source_patch_inside_and_informative"]
    template = cv2.getRectSubPix(source, (35, 35), (100.0, 100.0))
    candidate = target[:35, :35]
    template_centered = template - template.mean(axis=(0, 1), keepdims=True)
    candidate_centered = candidate - candidate.mean(axis=(0, 1), keepdims=True)
    manual = np.sum(template_centered * candidate_centered) / (
        np.linalg.norm(template_centered) * np.linalg.norm(candidate_centered)
    )
    assert scores[0, 0] == pytest.approx(float(manual), abs=2e-7)


def test_config_rejects_scale_sweep() -> None:
    with pytest.raises(ValueError):
        RGBObservabilityConfig(patch_sizes=(35, 71)).validate()


def test_scope_contract_passes_only_complete_distinct_grounded_edges() -> None:
    frames, channels = 180, 10
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    source_px = np.empty((frames, channels, 2), dtype=np.float64)
    for channel in range(channels):
        source_px[:, channel, 0] = 60.0 + 42.0 * channel + 8.0 * np.cos(phase)
        source_px[:, channel, 1] = 220.0 + 8.0 * np.sin(phase)
    masks = np.ones((frames, 512, 512), dtype=bool)
    source = _source_state(source_px, masks)
    physical = source_px.copy()
    shape = (2, frames, channels)
    arrays = {
        "source_valid": np.ones(shape, dtype=bool),
        "global_valid": np.ones(shape, dtype=bool),
        "global_top1_coordinate_px": np.stack((physical, physical), axis=0),
        "global_margin": np.ones(shape, dtype=np.float64),
    }
    ranks = {
        "global_physical_candidate_rank": np.ones(shape, dtype=np.int64),
        "global_physical_candidate_valid": np.ones(shape, dtype=bool),
    }
    report, _ = _scope_metrics(
        arrays,
        ranks,
        source,
        physical,
        np.ones((frames, channels), dtype=bool),
        masks,
        np.roll(np.arange(frames), -1),
        0,
        "global",
    )
    assert report["strict_pass_count"] == 10
    arrays["global_valid"][0, 7, 3] = False
    failed, _ = _scope_metrics(
        arrays,
        ranks,
        source,
        physical,
        np.ones((frames, channels), dtype=bool),
        masks,
        np.roll(np.arange(frames), -1),
        0,
        "global",
    )
    assert failed["strict_pass_count"] == 9
