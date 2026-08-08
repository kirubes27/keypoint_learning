"""Focused geometry and decision tests for the no-training descriptor audit."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


KEYPOINT_NET_ROOT = Path(__file__).resolve().parents[1] / "keypoint_net"
if str(KEYPOINT_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(KEYPOINT_NET_ROOT))

from keypoint_net import descriptor_material_alignment as audit


class GeometryTests(unittest.TestCase):
    def test_known_rotation_and_inverse_are_exact(self):
        points = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
        rotated = audit.apply_rotation(points, 90.0)
        np.testing.assert_allclose(
            rotated,
            np.asarray([[[0.0, 1.0], [-1.0, 0.0]]]),
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            audit.apply_rotation(rotated, -90.0), points, atol=1e-12, rtol=0.0
        )

    def test_one_cell_step_has_exact_length_and_projects(self):
        candidate = np.asarray([[[0.0, 0.0], [0.99, 0.0], [0.2, 0.3]]])
        descent = np.asarray([[[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]]])
        stepped, projected = audit.normalized_coordinate_step(candidate, descent)
        self.assertAlmostEqual(
            float(np.linalg.norm(stepped[0, 0] - candidate[0, 0])),
            audit.GRID_STEP_64,
            places=12,
        )
        self.assertTrue(projected[0, 1])
        self.assertEqual(stepped[0, 1, 0], 1.0)
        np.testing.assert_array_equal(stepped[0, 2], candidate[0, 2])

    def test_mask_sampling_matches_endpoint_grid_rule(self):
        masks = np.zeros((1, 3, 3), dtype=bool)
        masks[0, 0, 0] = True
        masks[0, 2, 2] = True
        points = np.asarray([[[-1.0, -1.0], [1.0, 1.0], [1.6, 0.0]]])
        inside, in_image = audit.sample_mask(points, masks)
        np.testing.assert_array_equal(inside, [[True, True, False]])
        np.testing.assert_array_equal(in_image, [[True, True, False]])


class MaterialDirectionTests(unittest.TestCase):
    def test_aligned_descent_reduces_oracle_error(self):
        anchor = np.asarray([[[0.8, 0.0]]])
        oracle = audit.apply_rotation(anchor, 6.0)
        candidate = oracle + np.asarray([[[0.10, 0.0]]])
        descent = oracle - candidate
        masks = np.ones((1, 8, 8), dtype=bool)
        metrics = audit.direction_metrics(
            anchor=anchor,
            candidate=candidate,
            descent=descent,
            angle_deg=6.0,
            candidate_masks=masks,
            candidate_bbox_diagonal=np.asarray([1.0]),
        )
        self.assertAlmostEqual(float(metrics["cosine"][0, 0]), 1.0, places=12)
        self.assertLess(float(metrics["error_delta_objdiag"][0, 0]), 0.0)

    def test_misaligned_descent_increases_oracle_error(self):
        anchor = np.asarray([[[0.8, 0.0]]])
        oracle = audit.apply_rotation(anchor, 6.0)
        candidate = oracle + np.asarray([[[0.10, 0.0]]])
        descent = candidate - oracle
        masks = np.ones((1, 8, 8), dtype=bool)
        metrics = audit.direction_metrics(
            anchor=anchor,
            candidate=candidate,
            descent=descent,
            angle_deg=6.0,
            candidate_masks=masks,
            candidate_bbox_diagonal=np.asarray([1.0]),
        )
        self.assertAlmostEqual(float(metrics["cosine"][0, 0]), -1.0, places=12)
        self.assertGreater(float(metrics["error_delta_objdiag"][0, 0]), 0.0)

    def test_row_summary_passes_only_all_three_semantic_checks(self):
        rows = []
        for direction in ("forward", "reverse"):
            for channel in (0, 1):
                rows.append({
                    "direction": direction,
                    "channel": channel,
                    "cosine_alignment": 0.5,
                    "error_before_objdiag": 0.2,
                    "error_after_objdiag": 0.1,
                    "error_delta_objdiag": -0.1,
                    "coordinate_projection_applied": False,
                    "on_object_before": True,
                    "on_object_after": True,
                    "motion_attenuation_ratio": 0.8,
                })
        summary = audit.summarize_rows(rows, [0, 1])
        self.assertTrue(summary["checkpoint_pass"])
        self.assertEqual(
            summary["one_cell_coordinate_step"]["active_on_object_channel_count_after"],
            2,
        )

        rows[0] = {**rows[0], "cosine_alignment": -10.0,
                   "error_delta_objdiag": 10.0,
                   "error_after_objdiag": 10.2}
        rows[1] = {**rows[1], "cosine_alignment": -10.0,
                   "error_delta_objdiag": 10.0,
                   "error_after_objdiag": 10.2}
        rows[2] = {**rows[2], "cosine_alignment": -10.0,
                   "error_delta_objdiag": 10.0,
                   "error_after_objdiag": 10.2}
        summary = audit.summarize_rows(rows, [0, 1])
        self.assertFalse(summary["checkpoint_pass"])

    def test_rotation_magnitude_matches_six_degree_chord(self):
        point = np.asarray([[[0.5, 0.0]]])
        moved = audit.apply_rotation(point, 6.0) - point
        expected = 2.0 * 0.5 * math.sin(math.radians(3.0))
        self.assertAlmostEqual(float(np.linalg.norm(moved)), expected, places=12)


if __name__ == "__main__":
    unittest.main()
