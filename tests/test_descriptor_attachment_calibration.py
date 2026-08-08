"""CPU tests for the outcome-blind descriptor weight-calibration arithmetic."""

from __future__ import annotations

import math
import unittest

import torch

from keypoint_net import descriptor_attachment_calibration as calibration


def _records(ratios):
    return [
        {
            "cell_id": cell_id,
            "base_gradient_norm": 1.0,
            "attachment_gradient_norm": float(ratio),
        }
        for cell_id, ratio in zip(calibration.CALIBRATION_CELL_IDS, ratios)
    ]


class CalibrationArithmeticTests(unittest.TestCase):
    def test_reproducible_lambda_and_both_caps(self):
        records = _records([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        first = calibration.choose_global_lambda(records)
        second = calibration.choose_global_lambda(records)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["median_gradient_ratio"], 3.5)
        self.assertAlmostEqual(first["maximum_gradient_ratio"], 6.0)
        expected_lambda = min(0.10 / 3.5, 0.50 / 6.0)
        self.assertAlmostEqual(first["lambda_attach"], expected_lambda)
        self.assertLessEqual(first["scaled_median_contribution"], 0.10 + 1e-15)
        self.assertLessEqual(first["scaled_maximum_contribution"], 0.50 + 1e-15)
        self.assertEqual(first["status"], "authorized")

    def test_max_cap_below_two_percent_stops(self):
        result = calibration.choose_global_lambda(
            _records([1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        )
        self.assertEqual(
            result["status"],
            "stop_below_minimum_median_contribution",
        )
        self.assertLess(result["scaled_median_contribution"], 0.02)
        self.assertAlmostEqual(result["scaled_maximum_contribution"], 0.50)

    def test_cell_order_count_zero_nan_and_inf_fail(self):
        with self.assertRaises(ValueError):
            calibration.choose_global_lambda(_records([1.0] * 5))
        reversed_records = list(reversed(_records([1.0] * 6)))
        with self.assertRaises(ValueError):
            calibration.choose_global_lambda(reversed_records)
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            records = _records([1.0] * 6)
            records[2]["attachment_gradient_norm"] = invalid
            with self.assertRaises(ValueError):
                calibration.choose_global_lambda(records)

    def test_sample_weighted_gradient_is_global_mean_not_mean_norm(self):
        vectors = [
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor([-1.0, 2.0], dtype=torch.float64),
        ]
        observed = calibration.sample_weighted_mean_gradient(vectors, [3, 1])
        expected = torch.tensor([0.5, 0.5], dtype=torch.float64)
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        self.assertAlmostEqual(calibration.strict_gradient_norm(observed), math.sqrt(0.5))

    def test_missing_and_nonfinite_gradients_fail(self):
        used = torch.nn.Parameter(torch.tensor(2.0))
        unused = torch.nn.Parameter(torch.tensor(3.0))
        with self.assertRaises(RuntimeError):
            calibration.gradient_vector(
                used.square(),
                [used, unused],
                retain_graph=False,
            )
        bad = torch.nn.Parameter(torch.tensor(float("nan")))
        with self.assertRaises(RuntimeError):
            calibration.gradient_vector(
                bad.square(),
                [bad],
                retain_graph=False,
            )
        for value in (
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([float("nan")], dtype=torch.float64),
            torch.tensor([float("inf")], dtype=torch.float64),
        ):
            with self.assertRaises(ValueError):
                calibration.strict_gradient_norm(value)

    def test_autograd_calibration_does_not_update_parameter_or_grad_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        before = parameter.detach().clone()
        vector = calibration.gradient_vector(
            parameter.square().mean(),
            [parameter],
            retain_graph=False,
        )
        torch.testing.assert_close(parameter.detach(), before, rtol=0.0, atol=0.0)
        self.assertIsNone(parameter.grad)
        self.assertEqual(vector.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
