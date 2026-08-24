from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "keypoint_net"))

from certified_witness_capability import CapabilityContractError  # noqa: E402
from evaluate_final_feature_query_decoder import require_frame_coverage  # noqa: E402


def test_manifest_superset_covers_requested_frames() -> None:
    require_frame_coverage({0, 1, 2, 27, 176}, np.asarray([0, 1, 2]), "fixture")


def test_manifest_missing_requested_frame_fails_closed() -> None:
    with pytest.raises(CapabilityContractError, match=r"omits requested frames: \[2\]"):
        require_frame_coverage({0, 1, 27}, np.asarray([0, 1, 2]), "fixture")
