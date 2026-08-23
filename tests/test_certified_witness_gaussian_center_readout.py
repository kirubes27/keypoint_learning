from __future__ import annotations

import unittest

import numpy as np

from keypoint_net.certified_witness_capability import (
    EXPECTED_WITNESSES,
    FEATURE_SIZE,
    HALF_CELL_DIAGONAL_PX,
)
from keypoint_net.certified_witness_gaussian_center_readout import (
    SIGMA_GRID,
    gaussian_center_readout_arrays,
)
from keypoint_net.certified_witness_local_readout import (
    classify_localization_failures,
    grid_to_pixel,
    readout_arrays,
)


def _gaussian_logits(center_x: float, center_y: float) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(FEATURE_SIZE, dtype=np.float64),
        np.arange(FEATURE_SIZE, dtype=np.float64),
        indexing="ij",
    )
    field = -((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * SIGMA_GRID**2)
    return np.broadcast_to(
        field, (1, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE)
    ).copy()


class CertifiedWitnessGaussianCenterReadoutTest(unittest.TestCase):
    def test_recovers_interior_known_sigma_center(self) -> None:
        logits = _gaussian_logits(20.25, 31.60)
        arrays = gaussian_center_readout_arrays(logits)
        expected = np.broadcast_to(
            grid_to_pixel(20.25, 31.60), (1, EXPECTED_WITNESSES, 2)
        )
        np.testing.assert_allclose(arrays["prediction_px"], expected, atol=1e-10)
        self.assertFalse(np.any(arrays["clamp_applied"]))
        self.assertTrue(np.all(arrays["design_rank"] == 3))

    def test_recovers_centers_at_each_clipped_border(self) -> None:
        centers = ((0.15, 0.20), (62.80, 0.10), (0.20, 62.75), (62.85, 62.80))
        for center_x, center_y in centers:
            with self.subTest(center=(center_x, center_y)):
                arrays = gaussian_center_readout_arrays(
                    _gaussian_logits(center_x, center_y)
                )
                expected = np.broadcast_to(
                    grid_to_pixel(center_x, center_y), (1, EXPECTED_WITNESSES, 2)
                )
                np.testing.assert_allclose(arrays["prediction_px"], expected, atol=1e-10)
                self.assertFalse(np.any(arrays["clamp_applied"]))

    def test_remote_alias_cannot_escape_selected_patch(self) -> None:
        logits = _gaussian_logits(11.0, 10.0)
        logits[:, :, 40, 41] = 20.0
        target = np.broadcast_to(
            grid_to_pixel(11.0, 10.0), (1, EXPECTED_WITNESSES, 2)
        ).copy()
        arrays = gaussian_center_readout_arrays(logits)
        baseline = readout_arrays(logits, target)
        error = np.linalg.norm(arrays["prediction_px"] - target, axis=-1)
        category, _ = classify_localization_failures(
            baseline, error <= HALF_CELL_DIAGONAL_PX + 1e-12
        )
        self.assertTrue(np.all(category == 2))
        self.assertTrue(np.all(arrays["prediction_grid_xy"][..., 0] >= 40.0))
        self.assertTrue(np.all(arrays["prediction_grid_xy"][..., 0] <= 42.0))
        self.assertTrue(np.all(arrays["prediction_grid_xy"][..., 1] >= 39.0))
        self.assertTrue(np.all(arrays["prediction_grid_xy"][..., 1] <= 41.0))

    def test_row_major_tie_and_clamp_are_finite_and_bounded(self) -> None:
        logits = np.zeros(
            (1, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float64
        )
        arrays = gaussian_center_readout_arrays(logits)
        self.assertTrue(np.all(arrays["hard_cell_x"] == 0))
        self.assertTrue(np.all(arrays["hard_cell_y"] == 0))
        self.assertTrue(np.isfinite(arrays["prediction_px"]).all())
        self.assertTrue(np.all(arrays["prediction_grid_xy"] >= 0.0))
        self.assertTrue(np.all(arrays["prediction_grid_xy"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
