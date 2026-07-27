"""CPU-only contract tests for independent oracle-tolerance calibration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from keypoint_net.diagnostics.calibrate_representation_oracle_tolerances import (
    _metric_scales_from_manifest,
    build_calibration,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    REPO_ROOT
    / "docs"
    / "decisions"
    / "2026-07-26"
    / "representation_oracle_calibration"
    / "CALIBRATION_FIXTURE_MANIFEST.json"
)


class CalibrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(FIXTURES.read_text(encoding="utf-8"))

    def test_error_metric_scale_uses_exact_expected_magnitude(self) -> None:
        scales = _metric_scales_from_manifest(
            self.manifest["geometry_fixtures"]
        )
        self.assertEqual(1.0, scales["signed_angle_error_deg"])
        self.assertEqual(1.0, scales["composition_bias_error_l2"])
        self.assertEqual(6.0, scales["proper_rotation_angle_deg"])
        self.assertEqual(4.72, scales["composition_bias_element"])

    def test_all_calibrated_tolerances_respect_provisional_ceilings(self) -> None:
        result = build_calibration(
            FIXTURES,
            fixture_manifest_commit="c694af75e803a1ed2f4f3abf2eee141f9ff0d486",
            reference_source_commit="0" * 40,
        )
        for metric, registry in result["metric_registries"].items():
            self.assertTrue(registry["within_provisional_ceiling"], metric)
            self.assertLessEqual(
                registry["tolerance"],
                registry["provisional_ceiling"],
                metric,
            )


if __name__ == "__main__":
    unittest.main()
