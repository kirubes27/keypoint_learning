from __future__ import annotations

import unittest

import numpy as np

from keypoint_net.analyze_certified_witness_readouts import _grid_to_pixel, _readout_arrays
from keypoint_net.certified_witness_capability import EXPECTED_FRAMES, EXPECTED_WITNESSES, FEATURE_SIZE


class FrozenReadoutDiagnosticTests(unittest.TestCase):
    def test_grid_to_pixel_maps_native_endpoints(self) -> None:
        prediction = _grid_to_pixel(
            np.asarray([0.0, FEATURE_SIZE - 1.0]),
            np.asarray([0.0, FEATURE_SIZE - 1.0]),
        )
        np.testing.assert_allclose(prediction, np.asarray([[0.0, 0.0], [511.0, 511.0]]))

    def test_local_readout_recovers_fractional_mass_inside_fixed_window(self) -> None:
        logits = np.full(
            (EXPECTED_FRAMES, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
            -100.0,
            dtype=np.float32,
        )
        logits[..., 10, 10] = np.log(0.75)
        logits[..., 10, 11] = np.log(0.25)
        target = np.broadcast_to(
            _grid_to_pixel(np.asarray(10.25), np.asarray(10.0)),
            (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2),
        ).copy()
        arrays = _readout_arrays(logits, target)
        expected_hard = np.broadcast_to(
            _grid_to_pixel(np.asarray(10.0), np.asarray(10.0)),
            (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2),
        )
        np.testing.assert_allclose(arrays["hard_prediction_px"], expected_hard)
        np.testing.assert_allclose(arrays["local_3x3_prediction_px"], target, atol=1e-5)

    def test_target_cell_rank_is_one_for_unique_target_peak(self) -> None:
        logits = np.full(
            (EXPECTED_FRAMES, EXPECTED_WITNESSES, FEATURE_SIZE, FEATURE_SIZE),
            -100.0,
            dtype=np.float32,
        )
        logits[..., 20, 30] = 10.0
        target = np.broadcast_to(
            _grid_to_pixel(np.asarray(30.0), np.asarray(20.0)),
            (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2),
        ).copy()
        arrays = _readout_arrays(logits, target)
        self.assertTrue(np.array_equal(arrays["target_nearest_cell_rank"], np.ones((180, 10), dtype=np.int64)))
        self.assertTrue(np.all(arrays["top1_top2_probability_margin"] > 0.0))


if __name__ == "__main__":
    unittest.main()
