"""Fail-closed training authorization for the descriptor-attachment experiment.

The historical 64-versus-128 manifest remains immutable.  This boundary uses a
new experiment manifest and the CPU calibration receipt, both exact-file and
content-hash bound, so neither code nor execution depends on Markdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from .descriptor_attachment import LOSS_SPEC_SHA256, canonical_sha256
    from .descriptor_attachment_calibration import (
        CHECKPOINT_SELECTOR,
        SCHEMA_VERSION as WEIGHT_RECEIPT_SCHEMA_VERSION,
    )
except ImportError:  # Historical direct-script execution.
    from descriptor_attachment import LOSS_SPEC_SHA256, canonical_sha256  # type: ignore
    from descriptor_attachment_calibration import (  # type: ignore
        CHECKPOINT_SELECTOR,
        SCHEMA_VERSION as WEIGHT_RECEIPT_SCHEMA_VERSION,
    )


EXPERIMENT_SCHEMA_VERSION = "descriptor_attachment_experiment_manifest.v1"
EXPERIMENT_ARTIFACT_TYPE = "descriptor_attachment_experiment_manifest"
EXPECTED_BRANCH = "agent/representation-oracles-20260726"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

# Every argument that can alter scientific data, model, optimization, loss,
# selection, or evaluation semantics must appear exactly once in each cell.
BOUND_TRAINING_ARGUMENTS = frozenset({
    "data_root",
    "object",
    "pairs_index",
    "indexed_mode",
    "train_pairs_index",
    "val_pairs_index",
    "test_pairs_index",
    "dataset_binding_sha256",
    "img_size",
    "num_keypoints",
    "base_channels",
    "heatmap_res",
    "temperature",
    "padding_mode",
    "operator_type",
    "learn_inverse_operator",
    "lambda_smooth",
    "lambda_disp",
    "lambda_ent",
    "lambda_act",
    "lambda_loc",
    "lambda_inv",
    "lambda_cycle",
    "lambda_attach",
    "sigma",
    "loc_bg_threshold",
    "num_action_classes",
    "frame_skip",
    "yaw_step_deg",
    "center_crop",
    "epochs",
    "frozen_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "seed",
    "output_dir",
    "save_every",
    "log_every",
    "auto_eval",
    "auto_eval_checkpoint",
    "eval_frames_dir",
    "eval_max_k",
})


@dataclass(frozen=True)
class DescriptorAttachmentBinding:
    cell_id: str
    arm: str
    lambda_attach: float
    run_directory: str
    experiment_manifest_file_sha256: str
    experiment_manifest_content_sha256: str
    weight_receipt_file_sha256: str
    weight_receipt_content_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular_json(path_value: str | Path, *, name: str) -> tuple[dict[str, Any], str]:
    path = Path(path_value).expanduser().resolve(strict=True)
    mode = path.stat().st_mode
    _require(stat.S_ISREG(mode), f"{name} is not a regular file")
    data = path.read_bytes()
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    _require(isinstance(document, dict), f"{name} must be a JSON object")
    return document, hashlib.sha256(data).hexdigest()


def _validate_content_hash(document: Mapping[str, Any], *, name: str) -> str:
    claimed = document.get("content_hash_sha256")
    _require(
        isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed) is not None,
        f"{name} content hash is invalid",
    )
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(canonical_sha256(payload) == claimed, f"{name} content hash mismatch")
    return claimed


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_state(repo_root: Path) -> tuple[str, str]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    _require(_COMMIT_RE.fullmatch(commit) is not None, "Git HEAD is invalid")
    _require(branch == EXPECTED_BRANCH, f"wrong source branch {branch!r}")
    tracked = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    _require(not tracked, "tracked source is modified")
    return commit, branch


def validate_weight_receipt(
    repo_root: Path | str,
    receipt_path: str | Path,
    *,
    expected_file_sha256: str,
) -> tuple[Mapping[str, Any], str, str]:
    root = Path(repo_root).resolve(strict=True)
    document, file_hash = _read_regular_json(receipt_path, name="weight receipt")
    _require(
        _SHA256_RE.fullmatch(expected_file_sha256 or "") is not None,
        "weight receipt expected file hash is invalid",
    )
    _require(file_hash == expected_file_sha256, "weight receipt file hash mismatch")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "source",
        "loss_spec_sha256",
        "checkpoint_selector",
        "calibration_protocol",
        "dataset",
        "cell_norms",
        "median_gradient_ratio",
        "maximum_gradient_ratio",
        "lambda_attach",
        "scaled_median_contribution",
        "scaled_maximum_contribution",
        "caps",
        "runtime",
        "content_hash_sha256",
    }
    _require(set(document) == expected_keys, "weight receipt fields differ from schema")
    content_hash = _validate_content_hash(document, name="weight receipt")
    _require(
        document["schema_version"] == WEIGHT_RECEIPT_SCHEMA_VERSION
        and document["artifact_type"] == "descriptor_attachment_weight_receipt",
        "wrong weight receipt schema",
    )
    _require(document["status"] == "authorized", "weight receipt did not authorize lambda")
    _require(document["loss_spec_sha256"] == LOSS_SPEC_SHA256, "loss specification changed")
    _require(
        document["checkpoint_selector"] == CHECKPOINT_SELECTOR,
        "checkpoint selector changed",
    )
    lambda_attach = document["lambda_attach"]
    _require(
        type(lambda_attach) is float and math.isfinite(lambda_attach) and lambda_attach > 0.0,
        "weight receipt lambda_attach is invalid",
    )
    source = document["source"]
    _require(
        isinstance(source, dict)
        and set(source) == {"commit", "branch", "module_hashes"},
        "weight receipt source record is invalid",
    )
    _require(source["branch"] == EXPECTED_BRANCH, "weight receipt branch changed")
    module_hashes = source["module_hashes"]
    expected_modules = {
        "keypoint_net/descriptor_attachment.py",
        "keypoint_net/descriptor_attachment_calibration.py",
        "keypoint_net/model.py",
        "keypoint_net/dataset.py",
        "aggregate_sha256",
    }
    _require(
        isinstance(module_hashes, dict) and set(module_hashes) == expected_modules,
        "weight receipt module hash set changed",
    )
    current_module_hashes = {
        relative: _sha256_file(root / relative)
        for relative in sorted(expected_modules - {"aggregate_sha256"})
    }
    _require(
        module_hashes["aggregate_sha256"]
        == canonical_sha256(current_module_hashes),
        "weight receipt aggregate code hash is invalid",
    )
    for relative, current_hash in current_module_hashes.items():
        _require(module_hashes[relative] == current_hash, f"bound code changed: {relative}")
    return document, file_hash, content_hash


def bind_training_namespace(
    repo_root: Path | str,
    args: argparse.Namespace,
) -> DescriptorAttachmentBinding:
    """Bind one control/attachment cell to exact code, manifest, and weight."""

    root = Path(repo_root).resolve(strict=True)
    _require(isinstance(args.descriptor_cell_id, str), "descriptor_cell_id is required")
    _require(
        isinstance(args.descriptor_experiment_manifest, str),
        "descriptor experiment manifest is required",
    )
    _require(
        isinstance(args.descriptor_experiment_manifest_sha256, str),
        "descriptor experiment manifest hash is required",
    )
    _require(
        isinstance(args.descriptor_weight_receipt, str),
        "descriptor weight receipt is required",
    )
    _require(
        isinstance(args.descriptor_weight_receipt_sha256, str),
        "descriptor weight receipt hash is required",
    )
    _require(
        args.descriptor_loss_spec_sha256 == LOSS_SPEC_SHA256,
        "descriptor loss specification hash differs from code",
    )
    _require(args.fresh_cell_id is None, "descriptor and historical fresh-cell paths cannot mix")

    manifest, manifest_file_hash = _read_regular_json(
        args.descriptor_experiment_manifest,
        name="descriptor experiment manifest",
    )
    _require(
        _SHA256_RE.fullmatch(args.descriptor_experiment_manifest_sha256) is not None,
        "descriptor experiment manifest expected hash is invalid",
    )
    _require(
        manifest_file_hash == args.descriptor_experiment_manifest_sha256,
        "descriptor experiment manifest file hash mismatch",
    )
    expected_manifest_keys = {
        "schema_version",
        "artifact_type",
        "source_commit",
        "source_branch",
        "loss_spec_sha256",
        "checkpoint_selector",
        "weight_receipt",
        "cells",
        "content_hash_sha256",
    }
    _require(set(manifest) == expected_manifest_keys, "experiment manifest fields differ")
    manifest_content_hash = _validate_content_hash(
        manifest,
        name="descriptor experiment manifest",
    )
    _require(
        manifest["schema_version"] == EXPERIMENT_SCHEMA_VERSION
        and manifest["artifact_type"] == EXPERIMENT_ARTIFACT_TYPE,
        "wrong descriptor experiment manifest schema",
    )
    _require(manifest["loss_spec_sha256"] == LOSS_SPEC_SHA256, "manifest loss spec changed")
    _require(
        manifest["checkpoint_selector"] == CHECKPOINT_SELECTOR,
        "manifest checkpoint selector changed",
    )
    source_commit, source_branch = _source_state(root)
    _require(
        manifest["source_commit"] == source_commit
        and manifest["source_branch"] == source_branch,
        "experiment manifest source differs from exact checkout",
    )

    receipt_record = manifest["weight_receipt"]
    _require(
        isinstance(receipt_record, dict)
        and set(receipt_record)
        == {"absolute_path", "file_sha256", "content_hash_sha256"},
        "manifest weight receipt binding is invalid",
    )
    supplied_receipt_path = str(
        Path(args.descriptor_weight_receipt).expanduser().resolve(strict=True)
    )
    _require(
        receipt_record["absolute_path"] == supplied_receipt_path,
        "weight receipt path differs from manifest",
    )
    _require(
        receipt_record["file_sha256"] == args.descriptor_weight_receipt_sha256,
        "weight receipt CLI hash differs from manifest",
    )
    receipt, receipt_file_hash, receipt_content_hash = validate_weight_receipt(
        root,
        supplied_receipt_path,
        expected_file_sha256=args.descriptor_weight_receipt_sha256,
    )
    _require(
        receipt_record["content_hash_sha256"] == receipt_content_hash,
        "weight receipt content hash differs from manifest",
    )
    _require(receipt["source"]["commit"] == source_commit, "weight receipt source commit changed")

    cells = manifest["cells"]
    _require(isinstance(cells, list) and cells, "experiment manifest cells are empty")
    matches = [cell for cell in cells if cell.get("cell_id") == args.descriptor_cell_id]
    _require(len(matches) == 1, "descriptor cell ID is missing or duplicated")
    cell = matches[0]
    _require(
        isinstance(cell, dict)
        and set(cell) == {"cell_id", "arm", "training_arguments"},
        "descriptor cell fields differ from schema",
    )
    _require(cell["arm"] in {"control", "attachment"}, "descriptor cell arm is invalid")
    expected_arguments = cell["training_arguments"]
    _require(
        isinstance(expected_arguments, dict)
        and set(expected_arguments) == BOUND_TRAINING_ARGUMENTS,
        "descriptor cell does not bind every scientific training argument",
    )
    for name, expected in expected_arguments.items():
        actual = getattr(args, name, object())
        _require(
            type(actual) is type(expected) and actual == expected,
            f"descriptor argument --{name} differs from frozen cell",
        )
    lambda_attach = expected_arguments["lambda_attach"]
    if cell["arm"] == "control":
        _require(type(lambda_attach) is float and lambda_attach == 0.0, "control arm requires lambda_attach=0")
    else:
        _require(
            type(lambda_attach) is float
            and lambda_attach == receipt["lambda_attach"]
            and lambda_attach > 0.0,
            "attachment arm lambda differs from the weight receipt",
        )
    run_directory = (
        Path(expected_arguments["output_dir"]).expanduser().resolve()
        / args.descriptor_cell_id
    ).absolute()
    _require(not run_directory.exists(), "descriptor cell output already exists")
    return DescriptorAttachmentBinding(
        cell_id=args.descriptor_cell_id,
        arm=cell["arm"],
        lambda_attach=lambda_attach,
        run_directory=str(run_directory),
        experiment_manifest_file_sha256=manifest_file_hash,
        experiment_manifest_content_sha256=manifest_content_hash,
        weight_receipt_file_sha256=receipt_file_hash,
        weight_receipt_content_sha256=receipt_content_hash,
    )


def reject_unbound_descriptor_arguments(args: argparse.Namespace) -> None:
    """Keep all non-experiment entry paths exactly on the legacy zero arm."""

    _require(
        type(args.lambda_attach) is float and args.lambda_attach == 0.0,
        "lambda_attach requires an authorized descriptor_cell_id",
    )
    for name in (
        "descriptor_experiment_manifest",
        "descriptor_experiment_manifest_sha256",
        "descriptor_weight_receipt",
        "descriptor_weight_receipt_sha256",
        "descriptor_loss_spec_sha256",
    ):
        _require(getattr(args, name) is None, f"--{name} is unbound without descriptor_cell_id")


__all__ = [
    "BOUND_TRAINING_ARGUMENTS",
    "DescriptorAttachmentBinding",
    "EXPECTED_BRANCH",
    "EXPERIMENT_ARTIFACT_TYPE",
    "EXPERIMENT_SCHEMA_VERSION",
    "bind_training_namespace",
    "reject_unbound_descriptor_arguments",
    "validate_weight_receipt",
]
