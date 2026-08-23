from __future__ import annotations

import unittest

import numpy as np

from keypoint_net.certified_witness_capability import (
    EXPECTED_WITNESSES,
    FEATURE_SIZE,
)
from keypoint_net.certified_witness_linear_offset_head import (
    FEATURE_CHANNELS,
    OFFSET_LIMIT_GRID,
    predict_linear_offset_head,
    solve_affine_offset,
)
from keypoint_net.certified_witness_local_readout import grid_to_pixel


class CertifiedWitnessLinearOffsetHeadTest(unittest.TestCase):
    def test_full_rank_planted_affine_problem_replays_exactly(self) -> None:
        design = np.vstack(
            [
                np.eye(FEATURE_CHANNELS + 1, dtype=np.float64),
                np.linspace(-1.0, 1.0, FEATURE_CHANNELS + 1)[None, :],
            ]
        )
        planted = np.stack(
            [
                np.linspace(-0.25, 0.25, FEATURE_CHANNELS + 1),
                np.linspace(0.40, -0.40, FEATURE_CHANNELS + 1),
            ],
            axis=-1,
        )
        labels = np.einsum("ij,jk->ik", design, planted, optimize=False)
        coefficient, report = solve_affine_offset(design, labels)
        np.testing.assert_allclose(coefficient, planted, atol=1e-12)
        np.testing.assert_allclose(
            np.einsum("ij,jk->ik", design, coefficient, optimize=False),
            labels,
            atol=1e-12,
        )
        self.assertEqual(report["design_rank"], FEATURE_CHANNELS + 1)

    def test_zero_head_decodes_hard_cell_centers(self) -> None:
        features = np.zeros(
            (1, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float64
        )
        logits = np.full(
            (1, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
            -10.0,
            dtype=np.float64,
        )
        logits[:, :, 20, 30] = 4.0
        coefficient = np.zeros(
            (EXPECTED_WITNESSES, FEATURE_CHANNELS + 1, 2), dtype=np.float64
        )
        arrays = predict_linear_offset_head(features, logits, coefficient)
        expected = np.broadcast_to(
            grid_to_pixel(30.0, 20.0), (1, EXPECTED_WITNESSES, 2)
        )
        np.testing.assert_allclose(arrays["prediction_px"], expected)

    def test_remote_alias_cannot_escape_bounded_local_support(self) -> None:
        features = np.zeros(
            (1, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float64
        )
        logits = np.full(
            (1, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
            -10.0,
            dtype=np.float64,
        )
        logits[:, :, 40, 41] = 10.0
        coefficient = np.zeros(
            (EXPECTED_WITNESSES, FEATURE_CHANNELS + 1, 2), dtype=np.float64
        )
        coefficient[:, 0, :] = 100.0
        arrays = predict_linear_offset_head(features, logits, coefficient)
        self.assertTrue(np.all(arrays["offset_clamp_applied"]))
        self.assertTrue(
            np.all(arrays["bounded_offset_grid"] == OFFSET_LIMIT_GRID)
        )
        self.assertTrue(np.all(arrays["prediction_grid_xy"][..., 0] == 42.5))
        self.assertTrue(np.all(arrays["prediction_grid_xy"][..., 1] == 41.5))

    def test_row_major_tie_and_image_clip_are_deterministic(self) -> None:
        features = np.zeros(
            (1, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float64
        )
        logits = np.zeros(
            (1, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float64
        )
        coefficient = np.zeros(
            (EXPECTED_WITNESSES, FEATURE_CHANNELS + 1, 2), dtype=np.float64
        )
        coefficient[:, 0, :] = -100.0
        arrays = predict_linear_offset_head(features, logits, coefficient)
        self.assertTrue(np.all(arrays["hard_cell_x"] == 0))
        self.assertTrue(np.all(arrays["hard_cell_y"] == 0))
        self.assertTrue(np.all(arrays["prediction_grid_xy"] == 0.0))
        self.assertTrue(np.all(arrays["image_clamp_applied"]))


if __name__ == "__main__":
    unittest.main()
