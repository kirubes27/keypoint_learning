from __future__ import annotations

import unittest

import numpy as np

from keypoint_net.certified_witness_capability import (
    EXPECTED_WITNESSES,
    FEATURE_SIZE,
    HALF_CELL_DIAGONAL_PX,
)
from keypoint_net.certified_witness_local_readout import (
    category_name,
    classify_localization_failures,
    grid_to_pixel,
    readout_arrays,
)


class CertifiedWitnessLocalConfirmationTest(unittest.TestCase):
    def _blank(self, frames: int = 1) -> np.ndarray:
        return np.full(
            (frames, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
            -80.0,
            dtype=np.float64,
        )

    def test_local_readout_uses_current_hard_peak_and_renormalizes(self) -> None:
        logits = self._blank()
        target_cell_x = 20.25
        target_cell_y = 31.60
        x0 = int(np.floor(target_cell_x))
        y0 = int(np.floor(target_cell_y))
        wx = target_cell_x - x0
        wy = target_cell_y - y0
        weights = {
            (y0, x0): (1.0 - wx) * (1.0 - wy),
            (y0, x0 + 1): wx * (1.0 - wy),
            (y0 + 1, x0): (1.0 - wx) * wy,
            (y0 + 1, x0 + 1): wx * wy,
        }
        for (y, x), weight in weights.items():
            logits[:, :, y, x] = np.log(weight)
        target = np.broadcast_to(
            grid_to_pixel(target_cell_x, target_cell_y),
            (1, EXPECTED_WITNESSES, 2),
        ).copy()
        arrays = readout_arrays(logits, target)
        np.testing.assert_allclose(arrays["local_3x3_prediction_px"], target, atol=1e-10)
        self.assertTrue(np.all(arrays["target_cell_inside_local_window"]))
        self.assertTrue(np.all(arrays["inside_window_probability_mass"] > 1.0 - 1e-12))

    def test_remote_alias_is_wrong_coarse_mode_even_when_target_is_second(self) -> None:
        logits = self._blank()
        logits[:, :, 40, 41] = 9.0
        logits[:, :, 10, 11] = 8.0
        target = np.broadcast_to(
            grid_to_pixel(11.0, 10.0),
            (1, EXPECTED_WITNESSES, 2),
        ).copy()
        arrays = readout_arrays(logits, target)
        local_error = np.linalg.norm(arrays["local_3x3_prediction_px"] - target, axis=-1)
        within = local_error <= HALF_CELL_DIAGONAL_PX + 1e-12
        category, counts = classify_localization_failures(arrays, within)
        self.assertTrue(np.all(arrays["target_nearest_cell_rank"] == 2))
        self.assertTrue(np.all(category == 2))
        self.assertEqual(
            counts["wrong_coarse_mode_target_top10"], EXPECTED_WITNESSES
        )
        self.assertEqual(category_name(2), "wrong_coarse_mode_target_top10")

    def test_border_category_has_priority_and_window_is_clipped(self) -> None:
        logits = self._blank()
        logits[:, :, 0, 0] = 4.0
        logits[:, :, 0, 1] = 3.0
        logits[:, :, 1, 0] = 2.0
        logits[:, :, 1, 1] = 1.0
        target = np.broadcast_to(
            grid_to_pixel(12.0, 12.0),
            (1, EXPECTED_WITNESSES, 2),
        ).copy()
        arrays = readout_arrays(logits, target)
        local_error = np.linalg.norm(arrays["local_3x3_prediction_px"] - target, axis=-1)
        category, counts = classify_localization_failures(
            arrays, local_error <= HALF_CELL_DIAGONAL_PX + 1e-12
        )
        self.assertTrue(np.all(arrays["hard_cell_x"] == 0))
        self.assertTrue(np.all(arrays["hard_cell_y"] == 0))
        self.assertTrue(np.all(category == 1))
        self.assertEqual(counts["border_window_truncation"], EXPECTED_WITNESSES)

    def test_row_major_tie_rule_is_deterministic(self) -> None:
        logits = np.zeros(
            (1, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float64
        )
        arrays = readout_arrays(logits)
        self.assertTrue(np.all(arrays["hard_cell_x"] == 0))
        self.assertTrue(np.all(arrays["hard_cell_y"] == 0))
        expected = np.broadcast_to(
            grid_to_pixel(0.5, 0.5), (1, EXPECTED_WITNESSES, 2)
        )
        np.testing.assert_allclose(arrays["local_3x3_prediction_px"], expected)


if __name__ == "__main__":
    unittest.main()
