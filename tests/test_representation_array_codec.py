from __future__ import annotations

import base64
import copy
import hashlib

import numpy as np
import pytest

from keypoint_net import eval_representation as evaluator
from keypoint_net import representation_array_codec as codec


def _reseal(record: dict) -> dict:
    value = copy.deepcopy(record)
    payload = base64.b64decode(value["payload_base64"])
    value["payload_nbytes"] = len(payload)
    value["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    value["content_sha256"] = codec.compact_array_content_sha256(value)
    return value


def test_float32_round_trip_is_bit_exact_and_deterministic() -> None:
    values = np.asarray(
        [
            [[0.0, -0.0], [1.25, -3.5]],
            [[np.float32("-inf"), 2.0], [1.0e-20, -1.0e20]],
        ],
        dtype=np.float32,
    )
    first = codec.encode_float32_array(values)
    second = codec.encode_float32_array(values.copy())
    assert first == second
    decoded = codec.decode_float32_array(first)
    np.testing.assert_array_equal(
        decoded.view(np.uint32),
        values.view(np.uint32),
    )
    assert codec.canonical_json_bytes(first) == codec.canonical_json_bytes(second)


def test_bool_packbits_round_trip_requires_zero_tail_bits() -> None:
    values = (np.arange(30).reshape(2, 3, 5) % 3) == 0
    record = codec.encode_bool_packbits_array(values)
    assert record["payload_nbytes"] == 4
    np.testing.assert_array_equal(
        codec.decode_bool_packbits_array(record),
        values,
    )

    mutated = copy.deepcopy(record)
    payload = bytearray(base64.b64decode(mutated["payload_base64"]))
    payload[-1] |= 0b1000_0000
    mutated["payload_base64"] = base64.b64encode(payload).decode("ascii")
    mutated = _reseal(mutated)
    with pytest.raises(codec.ArrayCodecError, match="unused tail bits"):
        codec.decode_bool_packbits_array(mutated)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("payload_sha256", "0" * 64, "payload hash mismatch"),
        ("payload_nbytes", 999, "payload length mismatch"),
        ("content_sha256", "0" * 64, "content hash mismatch"),
    ],
)
def test_mutated_self_hash_or_length_is_rejected(
    field: str,
    replacement: object,
    message: str,
) -> None:
    record = codec.encode_float32_array(np.ones((2, 3), dtype=np.float32))
    record[field] = replacement
    if field != "content_sha256":
        record["content_sha256"] = codec.compact_array_content_sha256(record)
    with pytest.raises(codec.ArrayCodecError, match=message):
        codec.decode_float32_array(record)


def test_unknown_fields_and_noncanonical_base64_are_rejected() -> None:
    record = codec.encode_float32_array(np.ones((2, 2), dtype=np.float32))
    extra = copy.deepcopy(record)
    extra["caller_authorized"] = True
    extra["content_sha256"] = codec.compact_array_content_sha256(extra)
    with pytest.raises(codec.ArrayCodecError, match="record keys differ"):
        codec.decode_float32_array(extra)

    noncanonical = copy.deepcopy(record)
    noncanonical["payload_base64"] += "\n"
    noncanonical["content_sha256"] = codec.compact_array_content_sha256(
        noncanonical
    )
    with pytest.raises(codec.ArrayCodecError, match="strict base64"):
        codec.decode_float32_array(noncanonical)


def test_evaluator_array_helpers_preserve_list_inputs_and_accept_compact_records() -> None:
    floats = np.asarray([[0.25, -0.5], [0.75, 1.0]], dtype=np.float32)
    compact_float = evaluator._finite_array(
        codec.encode_float32_array(floats),
        name="compact points",
        shape=(2, 2),
    )
    list_float = evaluator._finite_array(
        floats.tolist(),
        name="list points",
        shape=(2, 2),
    )
    np.testing.assert_array_equal(compact_float, list_float)

    booleans = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    compact_bool = evaluator._strict_bool_array(
        codec.encode_bool_packbits_array(booleans),
        name="compact visibility",
        shape=(2, 2),
    )
    list_bool = evaluator._strict_bool_array(
        booleans.tolist(),
        name="list visibility",
        shape=(2, 2),
    )
    np.testing.assert_array_equal(compact_bool, list_bool)


def test_compact_float32_logits_use_the_existing_spatial_expectation_path() -> None:
    logits = np.full((2, 3, 4, 4), -8.0, dtype=np.float32)
    logits[:, :, 1, 2] = 4.0
    estimator = {
        "temperature": 1.0,
        "logit_dtype": "float32",
        "softmax_dtype": "float32",
    }
    compact_logits, compact_points = evaluator._validated_logits_and_points(
        codec.encode_float32_array(logits),
        name="compact logits",
        expected_shape=logits.shape,
        estimator=estimator,
    )
    list_logits, list_points = evaluator._validated_logits_and_points(
        logits.tolist(),
        name="list logits",
        expected_shape=logits.shape,
        estimator=estimator,
    )
    np.testing.assert_array_equal(compact_logits, list_logits)
    np.testing.assert_array_equal(compact_points, list_points)


def test_compact_logits_cannot_claim_float64_storage() -> None:
    logits = np.zeros((2, 2, 2, 2), dtype=np.float32)
    estimator = {
        "temperature": 1.0,
        "logit_dtype": "float64",
        "softmax_dtype": "float64",
    }
    with pytest.raises(
        evaluator.EvaluationContractError,
        match="require estimator logit_dtype=float32",
    ):
        evaluator._validated_logits_and_points(
            codec.encode_float32_array(logits),
            name="compact logits",
            expected_shape=logits.shape,
            estimator=estimator,
        )
