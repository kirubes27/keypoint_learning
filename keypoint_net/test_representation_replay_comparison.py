from __future__ import annotations

import copy
import json
from pathlib import Path

from keypoint_net import representation_replay_comparison as comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    REPO_ROOT
    / "docs/decisions/2026-07-26/representation_oracle_replay/"
    "REPLAY_REGISTRY_v1.json"
)


def _documents() -> tuple[dict, dict, dict]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    historical = {
        "operator_spectrum": {
            "shared_affine": {
                "A": [[1.0, -0.1], [0.1, 1.0]],
                "bias": [0.01, -0.02],
                "det_A": 1.01,
                "singular_values": [1.01, 1.0],
                "closest_rotation_angle_deg": 5.8,
            }
        },
        "cyclic_compositionality": {
            "errors": {
                str(k): {
                    "mean": 0.001 * k,
                    "min": 0.0001 * k,
                    "max": 0.002 * k,
                    "n_samples": 180,
                }
                for k in range(1, 61)
            }
        },
    }
    evaluator = {
        "operator": {
            "learned_A": [[1.0, -0.1], [0.1, 1.0]],
            "learned_b": [0.01, -0.02],
            "determinant_A": 1.01,
            "singular_values": [1.01, 1.0],
            "proper_rotation_angle_deg": 5.8,
        },
        "rollout": {
            "full_corpus_identity_normalized_auc": {
                "horizons": [
                    {
                        "k": k,
                        "model_mse": {
                            "mean": 0.001 * k,
                            "minimum": 0.0001 * k,
                            "maximum": 0.002 * k,
                            "n": 180,
                        },
                    }
                    for k in range(1, 60)
                ]
            },
            "closure": {
                "k": 60,
                "model_mse": {
                    "mean": 0.06,
                    "minimum": 0.006,
                    "maximum": 0.12,
                    "n": 180,
                },
            },
        },
    }
    return registry, historical, evaluator


def test_exact_definition_identical_metrics_pass_245_records() -> None:
    registry, historical, evaluator = _documents()
    result = comparison.compare_definition_identical_metrics(
        evaluator_result=evaluator,
        registry_document=registry,
        historical_rollout=historical,
        task_id=20,
    )
    assert result["frozen_record_count"] == 245
    assert result["scalar_component_count"] == 250
    assert result["failed_record_count"] == 0
    assert result["all_definition_identical_fields_passed"] is True


def test_numeric_mismatch_and_sample_count_mismatch_both_fail() -> None:
    registry, historical, evaluator = _documents()
    changed = copy.deepcopy(evaluator)
    changed["operator"]["determinant_A"] = 1.5
    changed["rollout"]["closure"]["model_mse"]["n"] = 179
    result = comparison.compare_definition_identical_metrics(
        evaluator_result=changed,
        registry_document=registry,
        historical_rollout=historical,
        task_id=55,
    )
    assert result["failed_record_count"] == 2
    assert result["all_definition_identical_fields_passed"] is False


def test_relative_rule_accepts_difference_above_absolute_tolerance() -> None:
    registry, historical, evaluator = _documents()
    changed = copy.deepcopy(evaluator)
    changed["operator"]["determinant_A"] = 1.0101
    result = comparison.compare_definition_identical_metrics(
        evaluator_result=changed,
        registry_document=registry,
        historical_rollout=historical,
        task_id=80,
    )
    assert result["all_definition_identical_fields_passed"] is True
