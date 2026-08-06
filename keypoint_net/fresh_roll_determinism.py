"""Fail-closed CUDA warning policy for the fresh roll head experiment.

The policy is executable engineering configuration.  It preserves the frozen
models while allowing only the PyTorch-declared CUDA kernels approved for each
head package.  Unexpected nondeterminism is rejected after backward and before
the optimizer is permitted to update any weight.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Mapping


AMENDMENT_RELATIVE_PATH = Path(
    "keypoint_net/FRESH_ROLL_DETERMINISM_AMENDMENT.json"
)
POLICY_SCHEMA_VERSION = "fresh_roll_nondeterminism_policy.v1"
EVIDENCE_SCHEMA_VERSION = "fresh_roll_nondeterminism_evidence.v1"
MODE = "warn_only_fail_closed_allowlist"
COVERAGE_SCOPE = "pytorch_declared_nondeterminism_only"
REFLECTION_OPERATION = "reflection_pad2d_backward_cuda"
BILINEAR_OPERATION = "upsample_bilinear2d_backward_out_cuda"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OPERATION_RE = re.compile(
    r"\b([A-Za-z0-9_]+) does not have a deterministic implementation\b"
)


class FreshRollDeterminismError(RuntimeError):
    """The approved warning policy or its runtime evidence was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreshRollDeterminismError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"amendment contains duplicate key {key!r}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        _require(math.isfinite(value), "amendment contains a non-finite number")
        return value

    def reject_constant(token: str) -> None:
        raise FreshRollDeterminismError(
            f"amendment contains forbidden constant {token!r}"
        )

    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except FreshRollDeterminismError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FreshRollDeterminismError("amendment is not strict JSON") from exc
    _require(isinstance(document, Mapping), "amendment must be an object")
    return document


def load_amendment(repo_root: Path | str) -> tuple[Mapping[str, Any], dict[str, str]]:
    """Load, hash, and semantically validate the executable amendment."""

    root = Path(repo_root).resolve(strict=True)
    path = (root / AMENDMENT_RELATIVE_PATH).resolve(strict=True)
    _require(path.parent.parent == root, "amendment path escaped repository root")
    data = path.read_bytes()
    document = _strict_json(data)
    expected_keys = {
        "schema_version",
        "artifact_type",
        "mode",
        "allowed_operations_by_head_package",
        "required_cuda_positive_control",
        "repeatability_smoke",
        "unchanged_runtime_controls",
        "coverage_scope",
        "interpretation",
        "content_hash_sha256",
    }
    _require(set(document) == expected_keys, "amendment keys differ")
    claimed = document.get("content_hash_sha256")
    _require(
        isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed) is not None,
        "amendment content hash is invalid",
    )
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(_canonical_sha256(payload) == claimed, "amendment content hash mismatch")
    _require(
        document.get("schema_version") == "fresh_roll_determinism_amendment.v1"
        and document.get("artifact_type") == "fresh_roll_determinism_amendment"
        and document.get("mode") == MODE,
        "amendment identity differs",
    )
    _require(
        document.get("allowed_operations_by_head_package")
        == {
            "64": [REFLECTION_OPERATION],
            "128": [REFLECTION_OPERATION, BILINEAR_OPERATION],
        },
        "amendment head-specific allowlist differs",
    )
    _require(
        document.get("required_cuda_positive_control") == REFLECTION_OPERATION,
        "amendment positive control differs",
    )
    _require(
        document.get("repeatability_smoke")
        == {
            "replicas_per_cell": 2,
            "optimizer_steps_per_replica": 3,
            "cells": [
                "task55_clean__r64__seed42",
                "task55_clean__r128__seed42",
                "task80_assisted__r64__seed42",
                "task80_assisted__r128__seed42",
            ],
            "numeric_acceptance_threshold": None,
            "decision_use": "descriptive_only",
        },
        "amendment repeatability smoke differs",
    )
    _require(
        document.get("unchanged_runtime_controls")
        == {
            "cublas_workspace_config": ":4096:8",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "amendment runtime controls differ",
    )
    _require(document.get("coverage_scope") == COVERAGE_SCOPE,
             "amendment coverage scope differs")
    _require(
        document.get("interpretation")
        == {
            "architecture_or_padding_change": False,
            "scientific_metric_or_decision_rule_change": False,
            "head_128_may_expose_an_additional_allowed_kernel": True,
            "repeatability_differences_are_acceptance_criteria": False,
        },
        "amendment interpretation differs",
    )
    return document, {
        "repo_relative_path": str(AMENDMENT_RELATIVE_PATH),
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "content_hash_sha256": str(claimed),
    }


def amendment_record(repo_root: Path | str) -> dict[str, str]:
    return load_amendment(repo_root)[1]


def policy_for_heatmap_resolution(
    repo_root: Path | str,
    heatmap_res: int,
) -> dict[str, Any]:
    document, amendment = load_amendment(repo_root)
    _require(type(heatmap_res) is int and heatmap_res in (64, 128),
             "heatmap resolution is outside the approved policy")
    head_package = str(heatmap_res)
    allowed = document["allowed_operations_by_head_package"][head_package]
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "mode": MODE,
        "head_package": head_package,
        "allowed_operations": list(allowed),
        "required_cuda_positive_control": document[
            "required_cuda_positive_control"
        ],
        "coverage_scope": document["coverage_scope"],
        "amendment": amendment,
    }


def validate_policy(policy: Mapping[str, Any]) -> None:
    _require(isinstance(policy, Mapping), "nondeterminism policy is invalid")
    expected_keys = {
        "schema_version",
        "mode",
        "head_package",
        "allowed_operations",
        "required_cuda_positive_control",
        "coverage_scope",
        "amendment",
    }
    _require(set(policy) == expected_keys, "nondeterminism policy keys differ")
    head = policy.get("head_package")
    expected_allowed = {
        "64": [REFLECTION_OPERATION],
        "128": [REFLECTION_OPERATION, BILINEAR_OPERATION],
    }
    _require(head in expected_allowed, "nondeterminism policy head differs")
    _require(
        policy.get("schema_version") == POLICY_SCHEMA_VERSION
        and policy.get("mode") == MODE
        and policy.get("allowed_operations") == expected_allowed[head]
        and policy.get("required_cuda_positive_control") == REFLECTION_OPERATION
        and policy.get("coverage_scope") == COVERAGE_SCOPE,
        "nondeterminism policy semantics differ",
    )
    amendment = policy.get("amendment")
    _require(
        isinstance(amendment, Mapping)
        and set(amendment)
        == {"repo_relative_path", "file_sha256", "content_hash_sha256"}
        and amendment.get("repo_relative_path") == str(AMENDMENT_RELATIVE_PATH)
        and _SHA256_RE.fullmatch(str(amendment.get("file_sha256"))) is not None
        and _SHA256_RE.fullmatch(str(amendment.get("content_hash_sha256")))
        is not None,
        "nondeterminism amendment binding is invalid",
    )


def configure_torch_determinism(seed: int, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Enable warn-only deterministic mode and return its static evidence."""

    validate_policy(policy)
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace_config not in (None, ":4096:8"):
        raise FreshRollDeterminismError(
            "fresh runs require CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    warn_only_enabled = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    _require(callable(warn_only_enabled), "PyTorch cannot report warn-only state")
    record = {
        "seed": seed,
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": bool(warn_only_enabled()),
        "cudnn_deterministic": bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        ),
        "cudnn_benchmark": bool(
            getattr(torch.backends.cudnn, "benchmark", False)
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "nondeterminism_policy": copy.deepcopy(dict(policy)),
    }
    _require(
        record["torch_deterministic_algorithms"]
        and record["torch_deterministic_warn_only"]
        and record["cudnn_deterministic"]
        and not record["cudnn_benchmark"],
        "warn-only deterministic runtime controls were not enabled",
    )
    return record


def validate_determinism_record(
    record: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_policy: Mapping[str, Any],
) -> None:
    validate_policy(expected_policy)
    expected_keys = {
        "seed",
        "torch_deterministic_algorithms",
        "torch_deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cublas_workspace_config",
        "nondeterminism_policy",
        "data_loader_workers",
    }
    _require(isinstance(record, Mapping) and set(record) == expected_keys,
             "determinism record keys differ")
    _require(
        record.get("seed") == expected_seed
        and record.get("torch_deterministic_algorithms") is True
        and record.get("torch_deterministic_warn_only") is True
        and record.get("cudnn_deterministic") is True
        and record.get("cudnn_benchmark") is False
        and record.get("cublas_workspace_config") == ":4096:8"
        and record.get("nondeterminism_policy") == expected_policy
        and type(record.get("data_loader_workers")) is int
        and record["data_loader_workers"] >= 0,
        "determinism record differs from approved policy",
    )


def new_warning_evidence(policy: Mapping[str, Any]) -> dict[str, Any]:
    validate_policy(policy)
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "mode": MODE,
        "head_package": policy["head_package"],
        "allowed_operations": list(policy["allowed_operations"]),
        "required_cuda_positive_control": policy[
            "required_cuda_positive_control"
        ],
        "coverage_scope": policy["coverage_scope"],
        "backward_calls": 0,
        "captured_warning_count": 0,
        "non_nondeterminism_warning_count": 0,
        "observed_operation_counts": {
            operation: 0 for operation in policy["allowed_operations"]
        },
        "unexpected_operations": [],
        "unparseable_nondeterminism_warnings": [],
    }


def _is_nondeterminism_warning(message: str) -> bool:
    lowered = message.lower()
    return (
        "deterministic implementation" in lowered
        or "nondeterministic" in lowered
        or "non-deterministic" in lowered
    )


def backward_and_step(
    loss: Any,
    optimizer: Any,
    *,
    policy: Mapping[str, Any],
    evidence: dict[str, Any],
) -> None:
    """Backpropagate, reject unapproved warnings, then and only then step."""

    validate_policy(policy)
    _require(
        evidence.get("head_package") == policy["head_package"]
        and evidence.get("allowed_operations") == policy["allowed_operations"],
        "warning evidence belongs to a different policy",
    )
    optimizer.zero_grad()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        loss.backward()
    evidence["backward_calls"] += 1
    evidence["captured_warning_count"] += len(captured)
    for warning in captured:
        message = str(warning.message)
        match = _OPERATION_RE.search(message)
        if match is not None:
            operation = match.group(1)
            if operation not in policy["allowed_operations"]:
                evidence["unexpected_operations"].append(operation)
                raise FreshRollDeterminismError(
                    "unexpected nondeterministic operation before optimizer step: "
                    f"{operation}"
                )
            evidence["observed_operation_counts"][operation] += 1
        elif _is_nondeterminism_warning(message):
            evidence["unparseable_nondeterminism_warnings"].append(message)
            raise FreshRollDeterminismError(
                "unparseable nondeterminism warning before optimizer step"
            )
        else:
            evidence["non_nondeterminism_warning_count"] += 1
    optimizer.step()


def finalize_warning_evidence(
    evidence: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    device_type: str,
) -> dict[str, Any]:
    validate_policy(policy)
    expected_base_keys = {
        "schema_version",
        "mode",
        "head_package",
        "allowed_operations",
        "required_cuda_positive_control",
        "coverage_scope",
        "backward_calls",
        "captured_warning_count",
        "non_nondeterminism_warning_count",
        "observed_operation_counts",
        "unexpected_operations",
        "unparseable_nondeterminism_warnings",
    }
    _require(isinstance(evidence, Mapping) and set(evidence) == expected_base_keys,
             "warning evidence keys differ")
    _require(
        evidence.get("schema_version") == EVIDENCE_SCHEMA_VERSION
        and evidence.get("mode") == MODE
        and evidence.get("head_package") == policy["head_package"]
        and evidence.get("allowed_operations") == policy["allowed_operations"]
        and evidence.get("required_cuda_positive_control")
        == policy["required_cuda_positive_control"]
        and evidence.get("coverage_scope") == COVERAGE_SCOPE,
        "warning evidence policy binding differs",
    )
    for count_key in (
        "backward_calls",
        "captured_warning_count",
        "non_nondeterminism_warning_count",
    ):
        _require(type(evidence.get(count_key)) is int and evidence[count_key] >= 0,
                 f"warning evidence {count_key} is invalid")
    counts = evidence.get("observed_operation_counts")
    _require(
        isinstance(counts, Mapping)
        and list(counts) == policy["allowed_operations"]
        and all(type(value) is int and value >= 0 for value in counts.values()),
        "warning evidence operation counts differ",
    )
    _require(evidence.get("unexpected_operations") == [],
             "warning evidence contains unexpected operations")
    _require(evidence.get("unparseable_nondeterminism_warnings") == [],
             "warning evidence contains unparseable warnings")
    _require(device_type in ("cpu", "cuda", "mps"), "device type is invalid")
    positive_control = str(policy["required_cuda_positive_control"])
    observed = counts[positive_control] > 0
    if device_type == "cuda":
        _require(observed, "CUDA positive-control warning was not observed")
    result = copy.deepcopy(dict(evidence))
    result.update(
        {
            "device_type": device_type,
            "positive_control_required": device_type == "cuda",
            "positive_control_observed": observed,
            "passed": True,
        }
    )
    return result


def validate_final_warning_evidence(
    evidence: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    device_type: str,
) -> None:
    final_keys = {
        "device_type",
        "positive_control_required",
        "positive_control_observed",
        "passed",
    }
    _require(final_keys.issubset(evidence), "final warning evidence keys differ")
    base = {key: value for key, value in evidence.items() if key not in final_keys}
    expected = finalize_warning_evidence(
        base,
        policy=policy,
        device_type=device_type,
    )
    _require(dict(evidence) == expected, "final warning evidence differs")
