from __future__ import annotations

from pathlib import Path

from keypoint_net.run_same_frame_consensus_calibration import (
    SCHEMA_VERSION,
    _content_hash,
    build_calibration,
)


def test_model_blind_consensus_calibration_is_finite_and_self_hashed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = build_calibration(repo_root)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["model_checkpoint_opened"] is False
    assert result["training_or_weight_update_performed"] is False
    assert result["planted_case_count"] == 36
    assert result["all_semantic_assertions_pass"] is True
    assert all(result["semantic_assertions"].values())
    assert result["content_hash_sha256"] == _content_hash(result)
    floor = result["interpolation_floor"]
    assert floor["maximum_hard_displacement_cells"] == 0.0
    assert floor["maximum_soft_displacement_cells"] < 0.01
    assert floor["maximum_rms_width_increase_cells"] > 0.0
