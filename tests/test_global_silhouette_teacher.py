from __future__ import annotations

import unittest
import math

import cv2
import numpy as np

from keypoint_net.global_silhouette_teacher import (
    GlobalSilhouetteError,
    decode_sequence,
    decode_temporally_unwrapped_sequence,
    extract_silhouette,
    rigid_matrix,
    select_temporal_candidate,
    transform_points,
)


class GlobalSilhouetteTeacherTest(unittest.TestCase):
    image_size = 128
    background = np.asarray([171, 171, 171], dtype=np.uint8)

    def _asymmetric_rgb(self) -> np.ndarray:
        image = np.broadcast_to(
            self.background, (self.image_size, self.image_size, 3)
        ).copy()
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        cv2.rectangle(mask, (58, 25), (69, 104), 255, thickness=-1)
        cv2.rectangle(mask, (34, 20), (93, 42), 255, thickness=-1)
        cv2.rectangle(mask, (27, 24), (40, 37), 255, thickness=-1)
        image[mask > 0] = np.asarray([65, 32, 18], dtype=np.uint8)
        return image

    def _warp(
        self, image: np.ndarray, angle_rad: float, translation_xy: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = extract_silhouette(image)
        target_centroid = observation.centroid_xy + np.asarray(
            translation_xy, dtype=np.float64
        )
        matrix = rigid_matrix(angle_rad, observation.centroid_xy, target_centroid)
        warped = cv2.warpAffine(
            image,
            matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(int(value) for value in self.background),
        )
        return warped, matrix

    def test_identity_is_exact(self) -> None:
        source = self._asymmetric_rgb()
        points = np.asarray([[63.0, 31.0], [63.0, 80.0], [37.0, 30.0]])
        decoded = decode_sequence([source], points, [0])
        np.testing.assert_array_equal(decoded["prediction_xy"][0], points)
        self.assertEqual(decoded["selected_index"][0], 0)
        self.assertEqual(decoded["selected_angle_rad"][0], 0.0)
        self.assertEqual(decoded["selected_iou"][0], 1.0)

    def test_positive_and_negative_rotation_sign_and_xy(self) -> None:
        source = self._asymmetric_rgb()
        points = np.asarray([[63.0, 31.0], [63.0, 80.0], [37.0, 30.0]])
        for angle_deg, translation in ((37.0, (3.0, -4.0)), (-41.0, (-2.0, 5.0))):
            target, planted_matrix = self._warp(
                source, np.deg2rad(angle_deg), translation
            )
            decoded = decode_sequence([source, target], points, [0, 1])
            expected = transform_points(points, planted_matrix)
            np.testing.assert_allclose(
                decoded["prediction_xy"][1], expected, atol=0.5, rtol=0.0
            )
            self.assertGreater(decoded["selected_iou"][1], 0.95)

    def test_pi_alternative_is_resolved_by_shape_overlap(self) -> None:
        source = self._asymmetric_rgb()
        points = np.asarray([[63.0, 31.0], [63.0, 80.0], [37.0, 30.0]])
        target, planted_matrix = self._warp(source, np.pi, (1.0, 20.0))
        decoded = decode_sequence([source, target], points, [0, 1])
        expected = transform_points(points, planted_matrix)
        np.testing.assert_allclose(decoded["prediction_xy"][1], expected, atol=0.5)
        self.assertGreater(decoded["ambiguity_gap_iou"][1], 0.2)

    def test_border_connected_foreground_fails_closed(self) -> None:
        image = np.broadcast_to(
            self.background, (self.image_size, self.image_size, 3)
        ).copy()
        image[30:90, :20] = np.asarray([30, 20, 10], dtype=np.uint8)
        with self.assertRaisesRegex(GlobalSilhouetteError, "no non-border"):
            extract_silhouette(image)

    def test_isotropic_foreground_fails_closed(self) -> None:
        image = np.broadcast_to(
            self.background, (self.image_size, self.image_size, 3)
        ).copy()
        image[48:80, 48:80] = np.asarray([30, 20, 10], dtype=np.uint8)
        with self.assertRaisesRegex(GlobalSilhouetteError, "isotropic"):
            extract_silhouette(image)

    def test_frame_order_is_exact(self) -> None:
        source = self._asymmetric_rgb()
        positive, _ = self._warp(source, np.deg2rad(23.0), (2.0, -1.0))
        negative, _ = self._warp(source, np.deg2rad(-31.0), (-3.0, 4.0))
        points = np.asarray([[63.0, 31.0], [63.0, 80.0], [37.0, 30.0]])
        canonical = decode_sequence([source, positive, negative], points, [0, 1, 2])
        reversed_order = decode_sequence(
            [source, positive, negative], points, [2, 1, 0]
        )
        self.assertEqual(set(canonical), set(reversed_order))
        for key in canonical:
            np.testing.assert_array_equal(canonical[key], reversed_order[key])

    def test_temporal_unwrap_crosses_pi_and_agrees_forward_reverse(self) -> None:
        source = self._asymmetric_rgb()
        centre = np.asarray([64.0, 64.0], dtype=np.float64)
        points = np.asarray([[63.0, 31.0], [63.0, 80.0], [37.0, 30.0]])
        rgbs: list[np.ndarray] = []
        expected: list[np.ndarray] = []
        for angle_deg in range(0, 360, 10):
            matrix = rigid_matrix(np.deg2rad(angle_deg), centre, centre)
            rgbs.append(
                cv2.warpAffine(
                    source,
                    matrix,
                    (self.image_size, self.image_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=tuple(int(value) for value in self.background),
                )
            )
            expected.append(transform_points(points, matrix))
        forward = decode_temporally_unwrapped_sequence(
            rgbs, points, range(len(rgbs))
        )
        reverse = decode_temporally_unwrapped_sequence(
            rgbs, points, [0, *range(len(rgbs) - 1, 0, -1)]
        )
        np.testing.assert_allclose(
            forward["prediction_xy"], np.stack(expected), atol=0.5, rtol=0.0
        )
        np.testing.assert_allclose(
            forward["matrix"], reverse["matrix"], atol=1.0e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            forward["prediction_xy"], reverse["prediction_xy"], atol=1.0e-12, rtol=0.0
        )
        self.assertAlmostEqual(
            abs(forward["selected_angle_unwrapped_rad"][18]), math.pi, places=12
        )
        self.assertLess(
            np.max(np.abs(forward["traversal_step_delta_rad"])), 0.5 * math.pi
        )
        self.assertLess(abs(float(forward["closure_delta_rad"])), 0.5 * math.pi)

    def test_temporal_pi_over_two_tie_fails_closed(self) -> None:
        with self.assertRaisesRegex(GlobalSilhouetteError, "branch is tied"):
            select_temporal_candidate(
                0.0, np.asarray([0.5 * math.pi, -0.5 * math.pi])
            )


if __name__ == "__main__":
    unittest.main()
