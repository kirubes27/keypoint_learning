from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from keypoint_net import eval_representation as evaluator


SOURCE_COMMIT = "a" * 40
CHECKPOINT_SHA256 = "b" * 64
RUNTIME_MANIFEST_SHA256 = "c" * 64
PREFLIGHT_SHA256 = "d" * 64
FIXTURE_ID = "saved_task20_seed42_collapsed_negative_control"


def _bundle() -> dict:
    return {
        "case_id": FIXTURE_ID,
        "case_kind": "checkpoint",
        "provenance": {
            "source_commit": SOURCE_COMMIT,
            "external_files": [
                {
                    "role": "checkpoint",
                    "absolute_path": "/not/touched/model.pt",
                    "file_sha256": CHECKPOINT_SHA256,
                    "size_bytes": 123,
                }
            ],
        },
    }


def _receipt(**updates: object) -> dict:
    receipt = {
        "checkpoint_evaluation_authorized": True,
        "source_commit": SOURCE_COMMIT,
        "task_id": 20,
        "fixture_id": FIXTURE_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_source_manifest_file_sha256": RUNTIME_MANIFEST_SHA256,
        "checkpoint_hash_preflight_file_sha256": PREFLIGHT_SHA256,
        "training_or_weight_update_authorized": False,
        "selection_use_authorized": False,
    }
    receipt.update(updates)
    return receipt


def _provenance_load_receipt() -> dict:
    checkpoint = _bundle()["provenance"]["external_files"][0]
    return {
        **checkpoint,
        "task_id": 20,
        "fixture_id": FIXTURE_ID,
        "source_commit": SOURCE_COMMIT,
        "same_open_file_descriptor_hash_and_load": True,
        "weights_only": True,
    }


def _trusted_module(
    receipt: object,
    *,
    provenance_receipt: object | None = None,
) -> SimpleNamespace:
    if provenance_receipt is None:
        provenance_receipt = _provenance_load_receipt()
    return SimpleNamespace(
        validate_checkpoint_evaluator_authorization=lambda bundle: receipt,
        consume_checkpoint_provenance_load_receipt=(
            lambda *, source_commit, checkpoint_record: provenance_receipt
        ),
    )


def test_missing_authorization_module_fails_closed() -> None:
    missing = ModuleNotFoundError(
        "authorization module absent",
        name=evaluator.CHECKPOINT_AUTHORIZATION_MODULE,
    )
    with mock.patch.object(
        evaluator.importlib,
        "import_module",
        side_effect=missing,
    ):
        with pytest.raises(
            evaluator.EvaluationContractError,
            match="authorization module is absent",
        ) as raised:
            evaluator._require_checkpoint_evaluator_authorization(_bundle())
    assert raised.value.code == "checkpoint_replay_blocked"


def test_caller_boolean_cannot_enable_checkpoint_evaluation() -> None:
    bundle = _bundle()
    bundle["checkpoint_evaluation_authorized"] = True
    missing = ModuleNotFoundError(
        "authorization module absent",
        name=evaluator.CHECKPOINT_AUTHORIZATION_MODULE,
    )
    with mock.patch.object(
        evaluator.importlib,
        "import_module",
        side_effect=missing,
    ):
        with pytest.raises(
            evaluator.EvaluationContractError,
            match="authorization module is absent",
        ):
            evaluator._require_checkpoint_evaluator_authorization(bundle)


def test_trusted_validator_receipt_is_cross_bound_without_file_access() -> None:
    bundle = _bundle()
    trusted = _trusted_module(_receipt())
    consumer = mock.Mock(return_value=_provenance_load_receipt())
    trusted.consume_checkpoint_provenance_load_receipt = consumer
    with mock.patch.object(
        evaluator.importlib,
        "import_module",
        return_value=trusted,
    ):
        normalized, checkpoint_load_receipt = (
            evaluator._require_checkpoint_evaluator_authorization(
                bundle
            )
        )
    assert normalized == _receipt()
    assert checkpoint_load_receipt == _provenance_load_receipt()
    consumer.assert_called_once_with(
        source_commit=SOURCE_COMMIT,
        checkpoint_record=bundle["provenance"]["external_files"][0],
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"checkpoint_evaluation_authorized": False}, "does not authorize"),
        ({"source_commit": "e" * 40}, "different source_commit"),
        ({"fixture_id": "wrong"}, "fixture_id differs"),
        ({"checkpoint_sha256": "e" * 64}, "different checkpoint bytes"),
        ({"training_or_weight_update_authorized": True}, "forbid training"),
        ({"selection_use_authorized": True}, "forbid selection"),
    ],
)
def test_malformed_or_cross_bound_receipt_is_rejected(
    updates: dict,
    message: str,
) -> None:
    trusted = _trusted_module(_receipt(**updates))
    with mock.patch.object(
        evaluator.importlib,
        "import_module",
        return_value=trusted,
    ):
        with pytest.raises(evaluator.EvaluationContractError, match=message):
            evaluator._require_checkpoint_evaluator_authorization(_bundle())


def test_validator_failure_is_checkpoint_replay_blocked() -> None:
    def fail(_bundle: dict) -> dict:
        raise RuntimeError("reviewed authorization is stale")

    trusted = SimpleNamespace(
        validate_checkpoint_evaluator_authorization=fail
    )
    with mock.patch.object(
        evaluator.importlib,
        "import_module",
        return_value=trusted,
    ):
        with pytest.raises(
            evaluator.EvaluationContractError,
            match="reviewed authorization is stale",
        ) as raised:
            evaluator._require_checkpoint_evaluator_authorization(_bundle())
    assert raised.value.code == "checkpoint_replay_blocked"


def test_gate_does_not_inspect_checkpoint_path() -> None:
    trusted = _trusted_module(_receipt())
    with mock.patch.object(
        evaluator.importlib,
        "import_module",
        return_value=trusted,
    ), mock.patch.object(
        evaluator,
        "_file_sha256",
        side_effect=AssertionError("checkpoint bytes were touched"),
    ):
        evaluator._require_checkpoint_evaluator_authorization(_bundle())
