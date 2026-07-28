"""Deterministic compact JSON records for large representation arrays.

The evaluator's original nested-list inputs remain valid.  This module adds two
lossless encodings for the arrays that dominate saved-checkpoint replay bundles:

* C-order little-endian float32 bytes encoded as canonical base64;
* C-order boolean values packed with ``numpy.packbits(bitorder="little")`` and
  encoded as canonical base64.

Every record binds both the payload bytes and the complete record metadata.  A
decoder rejects alternate base64 spellings, non-zero unused packbits, unknown
fields, shape/length disagreements, and either hash mismatch.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


__representation_import_sha256__ = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()

ARRAY_CODEC_SCHEMA_VERSION = "representation_compact_array.v1"
FLOAT32_ENCODING = "raw_float32_le_base64"
BOOL_PACKBITS_ENCODING = "packbits_bool_little_base64"

_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "encoding",
        "dtype",
        "shape",
        "order",
        "payload_base64",
        "payload_nbytes",
        "payload_sha256",
        "content_sha256",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArrayCodecError(ValueError):
    """Raised when a compact array record is not exact and self-consistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArrayCodecError(message)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole canonical JSON representation used by this codec."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArrayCodecError(
            f"compact array record is not strict canonical JSON: {exc}"
        ) from exc


def compact_array_content_sha256(record: Mapping[str, Any]) -> str:
    """Hash a record after omitting its top-level ``content_sha256`` field."""

    _require(isinstance(record, Mapping), "compact array record must be a mapping")
    payload = dict(record)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def is_compact_array_record(value: object) -> bool:
    """Return whether ``value`` declares this codec's exact schema version."""

    return (
        isinstance(value, Mapping)
        and value.get("schema_version") == ARRAY_CODEC_SCHEMA_VERSION
    )


def _shape(value: object) -> tuple[int, ...]:
    _require(isinstance(value, list) and bool(value), "shape must be a non-empty list")
    dimensions: list[int] = []
    for index, dimension in enumerate(value):
        _require(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension > 0,
            f"shape[{index}] must be a strictly positive integer",
        )
        dimensions.append(dimension)
    return tuple(dimensions)


def _element_count(shape: Sequence[int]) -> int:
    count = math.prod(shape)
    _require(count > 0, "compact array must contain at least one element")
    return int(count)


def _canonical_payload(value: object) -> bytes:
    _require(
        isinstance(value, str) and bool(value),
        "payload_base64 must be a non-empty string",
    )
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ArrayCodecError("payload_base64 must contain only ASCII") from exc
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ArrayCodecError("payload_base64 is not strict base64") from exc
    _require(
        base64.b64encode(payload) == encoded,
        "payload_base64 is not the canonical base64 spelling",
    )
    return payload


def _validate_record(
    record: Mapping[str, Any],
    *,
    expected_encoding: str,
    expected_dtype: str,
) -> tuple[tuple[int, ...], bytes]:
    _require(isinstance(record, Mapping), "compact array record must be a mapping")
    actual_keys = set(record)
    _require(
        actual_keys == set(_RECORD_KEYS),
        "compact array record keys differ: "
        f"missing={sorted(_RECORD_KEYS - actual_keys)}, "
        f"extra={sorted(actual_keys - _RECORD_KEYS)}",
    )
    _require(
        record["schema_version"] == ARRAY_CODEC_SCHEMA_VERSION,
        "compact array schema_version is unsupported",
    )
    _require(
        record["encoding"] == expected_encoding,
        f"compact array encoding must be {expected_encoding}",
    )
    _require(
        record["dtype"] == expected_dtype,
        f"compact array dtype must be {expected_dtype}",
    )
    _require(record["order"] == "C", "compact array order must be C")
    shape = _shape(record["shape"])
    payload_nbytes = record["payload_nbytes"]
    _require(
        isinstance(payload_nbytes, int)
        and not isinstance(payload_nbytes, bool)
        and payload_nbytes >= 0,
        "payload_nbytes must be a non-negative integer",
    )
    payload_sha256 = record["payload_sha256"]
    content_sha256 = record["content_sha256"]
    _require(
        isinstance(payload_sha256, str)
        and _SHA256_RE.fullmatch(payload_sha256) is not None,
        "payload_sha256 must be lowercase 64-hex",
    )
    _require(
        isinstance(content_sha256, str)
        and _SHA256_RE.fullmatch(content_sha256) is not None,
        "content_sha256 must be lowercase 64-hex",
    )
    _require(
        compact_array_content_sha256(record) == content_sha256,
        "compact array content hash mismatch",
    )
    payload = _canonical_payload(record["payload_base64"])
    _require(len(payload) == payload_nbytes, "compact array payload length mismatch")
    _require(
        hashlib.sha256(payload).hexdigest() == payload_sha256,
        "compact array payload hash mismatch",
    )
    return shape, payload


def _record(
    *,
    encoding: str,
    dtype: str,
    shape: Sequence[int],
    payload: bytes,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": ARRAY_CODEC_SCHEMA_VERSION,
        "encoding": encoding,
        "dtype": dtype,
        "shape": [int(item) for item in shape],
        "order": "C",
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "payload_nbytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    record["content_sha256"] = compact_array_content_sha256(record)
    return record


def encode_float32_array(value: object) -> dict[str, Any]:
    """Encode ``value`` as exact little-endian C-order float32 bytes."""

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ArrayCodecError("float32 value cannot be converted to an array") from exc
    _shape(list(array.shape))
    little_endian = np.ascontiguousarray(array.astype("<f4", copy=False))
    return _record(
        encoding=FLOAT32_ENCODING,
        dtype="float32",
        shape=little_endian.shape,
        payload=little_endian.tobytes(order="C"),
    )


def decode_float32_array(record: Mapping[str, Any]) -> np.ndarray:
    """Validate and decode one exact little-endian float32 record."""

    shape, payload = _validate_record(
        record,
        expected_encoding=FLOAT32_ENCODING,
        expected_dtype="float32",
    )
    expected_nbytes = _element_count(shape) * np.dtype("<f4").itemsize
    _require(
        len(payload) == expected_nbytes,
        "float32 payload length disagrees with shape",
    )
    array = np.frombuffer(payload, dtype="<f4").reshape(shape, order="C")
    # Return writable native-endian storage detached from the base64 payload.
    return array.astype(np.float32, copy=True)


def encode_bool_packbits_array(value: object) -> dict[str, Any]:
    """Encode a strict boolean array using little-bit-order packbits."""

    array = np.asarray(value)
    _require(array.dtype == np.dtype(np.bool_), "boolean encoder requires bool values")
    _shape(list(array.shape))
    flat = np.ascontiguousarray(array).reshape(-1)
    payload = np.packbits(flat, bitorder="little").tobytes(order="C")
    return _record(
        encoding=BOOL_PACKBITS_ENCODING,
        dtype="bool",
        shape=array.shape,
        payload=payload,
    )


def decode_bool_packbits_array(record: Mapping[str, Any]) -> np.ndarray:
    """Validate and decode one canonical little-bit-order boolean record."""

    shape, payload = _validate_record(
        record,
        expected_encoding=BOOL_PACKBITS_ENCODING,
        expected_dtype="bool",
    )
    count = _element_count(shape)
    expected_nbytes = (count + 7) // 8
    _require(
        len(payload) == expected_nbytes,
        "packbits payload length disagrees with shape",
    )
    remainder = count % 8
    if remainder:
        unused_mask = (~((1 << remainder) - 1)) & 0xFF
        _require(
            payload[-1] & unused_mask == 0,
            "packbits payload has non-zero unused tail bits",
        )
    packed = np.frombuffer(payload, dtype=np.uint8)
    flat = np.unpackbits(packed, bitorder="little", count=count)
    return flat.astype(np.bool_, copy=False).reshape(shape, order="C").copy()


__all__ = [
    "ARRAY_CODEC_SCHEMA_VERSION",
    "ArrayCodecError",
    "BOOL_PACKBITS_ENCODING",
    "FLOAT32_ENCODING",
    "canonical_json_bytes",
    "compact_array_content_sha256",
    "decode_bool_packbits_array",
    "decode_float32_array",
    "encode_bool_packbits_array",
    "encode_float32_array",
    "is_compact_array_record",
]
