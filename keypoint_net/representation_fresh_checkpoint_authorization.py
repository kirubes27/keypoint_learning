"""Compact fail-closed contract for fresh roll head-package runs.

The historical Task 20/55/80 authority is intentionally untouched.  This
module has one smaller job: bind a primary manifest cell to training arguments
and later authorize only a completed, hash-bound run of that exact cell.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from . import fresh_roll_determinism
except ImportError:  # train.py also supports direct script execution.
    import fresh_roll_determinism


EXPERIMENT_MANIFEST_RELATIVE_PATH = Path(
    "docs/decisions/2026-07-29/roll_head_package_training/"
    "EXPERIMENT_MANIFEST_v1.json"
)
EXPERIMENT_MANIFEST_FILE_SHA256 = (
    "a754174e8692fdc21fe2629dcc50d3ee83f00ff521366495a4280cff03594899"
)
EXPERIMENT_MANIFEST_CONTENT_SHA256 = (
    "2089c039f3dd7cab220e530834342188d282d8513a22fc395008c213a7913625"
)
RUN_RECEIPT_NAME = "COMPLETED_RUN_RECEIPT.json"
RUN_RECEIPT_SCHEMA_VERSION = "roll_head_package_completed_run_receipt.v2"
EXPECTED_BRANCH = "agent/representation-oracles-20260726"
REVIEWED_LOCK_COMMIT = "4736be95fe055d95f233e8a1ad8eda3ed528938d"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CELL_RE = re.compile(
    r"^(task55_clean|task80_assisted)__r(64|128)__seed(42|43|44)$"
)
RUN_SOURCE_PATHS = (
    "keypoint_net/FRESH_ROLL_DETERMINISM_AMENDMENT.json",
    "keypoint_net/fresh_roll_determinism.py",
    "keypoint_net/train.py",
    "keypoint_net/run_fresh_roll_cell.py",
    "keypoint_net/run_fresh_roll_primary_matrix.py",
    "keypoint_net/representation_fresh_checkpoint_authorization.py",
    "keypoint_net/representation_fresh_checkpoint_runtime.py",
    "keypoint_net/model.py",
    "keypoint_net/dataset.py",
    "keypoint_net/eval_representation.py",
    "keypoint_net/representation_evaluation_provenance.py",
    "keypoint_net/representation_array_codec.py",
    "keypoint_net/representation_corpus_inventory.py",
)


class FreshCheckpointAuthorizationError(ValueError):
    """The fresh cell, run, or checkpoint is not exactly authorized."""


@dataclass(frozen=True)
class FrozenFreshCell:
    cell_id: str
    recipe: str
    head_package: str
    seed: int
    source_commit: str
    manifest_file_sha256: str
    expected_config: Mapping[str, Any]
    train_pairs_path: str
    validation_pairs_path: str


@dataclass(frozen=True)
class FreshRunBinding:
    cell: FrozenFreshCell
    source_branch: str
    run_directory: str


@dataclass(frozen=True)
class FreshCheckpointCapability:
    cell_id: str
    source_commit: str
    checkpoint_absolute_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    config_absolute_path: str
    config_sha256: str
    config_size_bytes: int
    history_absolute_path: str
    history_sha256: str
    history_size_bytes: int
    receipt_absolute_path: str
    receipt_file_sha256: str
    receipt_size_bytes: int
    expected_embedded_config: Mapping[str, Any]
    _capability: object


_VERIFIED_FRESH_CAPABILITY = object()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreshCheckpointAuthorizationError(message)


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return data + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes, *, name: str) -> Mapping[str, Any] | list[Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        _require(math.isfinite(value), f"{name} contains a non-finite number")
        return value

    def reject_constant(token: str) -> None:
        raise FreshCheckpointAuthorizationError(
            f"{name} contains forbidden constant {token!r}"
        )

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except FreshCheckpointAuthorizationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FreshCheckpointAuthorizationError(f"{name} is not strict JSON") from exc
    _require(isinstance(value, (Mapping, list)), f"{name} has invalid top level")
    return value


def _read_regular(path: Path, *, name: str) -> tuple[bytes, int]:
    _require(path.is_absolute(), f"{name} path must be absolute")
    flags = os.O_RDONLY | (getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FreshCheckpointAuthorizationError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), metadata.st_size
    finally:
        os.close(descriptor)


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshCheckpointAuthorizationError(
            f"git command failed: {' '.join(arguments)}"
        ) from exc
    return result.stdout if binary else result.stdout.strip()


def _committed_bytes(root: Path, relative_path: str, commit: str) -> bytes:
    _require(
        relative_path and not Path(relative_path).is_absolute()
        and ".." not in Path(relative_path).parts,
        f"unsafe repository path {relative_path!r}",
    )
    committed = _git(root, "show", f"{commit}:{relative_path}", binary=True)
    working_path = (root / relative_path).absolute()
    working, _ = _read_regular(working_path, name=relative_path)
    _require(working == committed, f"working bytes differ from Git for {relative_path}")
    return working


def _content_hash_is_valid(document: Mapping[str, Any], *, name: str) -> None:
    claimed = document.get("content_hash_sha256")
    _require(isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed),
             f"{name} content hash is invalid")
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(canonical_sha256(payload) == claimed, f"{name} content hash mismatch")


def _source_state(root: Path, *, require_clean: bool) -> tuple[str, str]:
    commit = str(_git(root, "rev-parse", "HEAD"))
    branch = str(_git(root, "rev-parse", "--abbrev-ref", "HEAD"))
    _require(_COMMIT_RE.fullmatch(commit) is not None, "Git HEAD is invalid")
    _require(branch == EXPECTED_BRANCH, f"wrong source branch {branch!r}")
    _git(root, "merge-base", "--is-ancestor", REVIEWED_LOCK_COMMIT, commit)
    if require_clean:
        tracked = str(_git(root, "status", "--porcelain", "--untracked-files=no"))
        _require(not tracked, "tracked source is modified")
    return commit, branch


def validate_experiment_manifest(repo_root: Path | str) -> tuple[Mapping[str, Any], str]:
    """Validate the exact committed manifest and its reviewed bindings."""

    root = Path(repo_root).resolve(strict=True)
    commit, _ = _source_state(root, require_clean=False)
    data = _committed_bytes(root, str(EXPERIMENT_MANIFEST_RELATIVE_PATH), commit)
    file_hash = hashlib.sha256(data).hexdigest()
    _require(file_hash == EXPERIMENT_MANIFEST_FILE_SHA256,
             "experiment manifest file hash mismatch")
    manifest = _strict_json(data, name="experiment manifest")
    _require(isinstance(manifest, Mapping), "experiment manifest must be an object")
    _content_hash_is_valid(manifest, name="experiment manifest")
    _require(
        manifest.get("content_hash_sha256") == EXPERIMENT_MANIFEST_CONTENT_SHA256,
        "experiment manifest content binding differs",
    )
    _require(
        manifest.get("authorization_boundary") == {
            "training_authorized_by_manifest": False,
            "gpu_smoke_authorized_by_manifest": False,
            "full_matrix_authorized_by_manifest": False,
            "fresh_checkpoint_evaluation_requires_completed_run_receipt": True,
            "fixture_authorization_path_modified": False,
        },
        "experiment authorization boundary differs",
    )
    review = manifest.get("independent_review")
    _require(
        isinstance(review, Mapping)
        and review.get("verdict") == "PASS_WITH_NONBLOCKING_FINDINGS",
        "reviewed specification is not implementation-authorized",
    )
    for record_name, record in (
        ("reviewed specification", manifest.get("reviewed_specification")),
        ("independent review", review),
    ):
        _require(isinstance(record, Mapping), f"{record_name} binding is invalid")
        bound = _committed_bytes(root, str(record["repo_relative_path"]), commit)
        _require(hashlib.sha256(bound).hexdigest() == record["file_sha256"],
                 f"{record_name} hash mismatch")
    cells = manifest.get("cells")
    _require(isinstance(cells, list) and len(cells) == 12,
             "manifest must contain exactly 12 primary cells")
    observed = {str(cell.get("cell_id")) for cell in cells if isinstance(cell, Mapping)}
    expected = {
        f"{recipe}__r{head}__seed{seed}"
        for recipe in ("task55_clean", "task80_assisted")
        for head in (64, 128)
        for seed in (42, 43, 44)
    }
    _require(observed == expected, "manifest primary cells differ")
    return manifest, file_hash


def expected_cell_config(manifest: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    _require(_CELL_RE.fullmatch(cell_id) is not None,
             "cell_id is outside the frozen primary matrix")
    matches = [cell for cell in manifest["cells"] if cell["cell_id"] == cell_id]
    _require(len(matches) == 1, "cell_id is not unique")
    cell = matches[0]
    fixed = copy.deepcopy(dict(manifest["fixed_training"]))
    fixed["heatmap_res"] = manifest["head_packages"][cell["head_package"]][
        "heatmap_res"
    ]
    fixed.update(copy.deepcopy(dict(manifest["recipes"][cell["recipe"]])))
    return {
        "schema_version": "roll_head_package_fresh_cell.v1",
        "cell_id": cell_id,
        "recipe": cell["recipe"],
        "head_package": cell["head_package"],
        "seed": cell["seed"],
        "dataset": {
            "basename": manifest["dataset"]["basename"],
            "binding_sha256": manifest["dataset"]["binding_sha256"],
            "corpus_inventory": copy.deepcopy(
                dict(manifest["dataset"]["corpus_inventory"])
            ),
        },
        "object": copy.deepcopy(dict(manifest["object"])),
        "transform": copy.deepcopy(dict(manifest["transform"])),
        "split": {
            "train": copy.deepcopy(dict(manifest["split_bundle"]["train_pairs"])),
            "validation": copy.deepcopy(
                dict(manifest["split_bundle"]["validation_pairs"])
            ),
        },
        "training": fixed,
    }


def resolve_fresh_cell(repo_root: Path | str, cell_id: str) -> FrozenFreshCell:
    root = Path(repo_root).resolve(strict=True)
    manifest, manifest_hash = validate_experiment_manifest(root)
    commit, _ = _source_state(root, require_clean=False)
    config = expected_cell_config(manifest, cell_id)
    return FrozenFreshCell(
        cell_id=cell_id,
        recipe=str(config["recipe"]),
        head_package=str(config["head_package"]),
        seed=int(config["seed"]),
        source_commit=commit,
        manifest_file_sha256=manifest_hash,
        expected_config=config,
        train_pairs_path=str(
            (root / config["split"]["train"]["repo_relative_path"]).resolve(strict=True)
        ),
        validation_pairs_path=str(
            (root / config["split"]["validation"]["repo_relative_path"]).resolve(
                strict=True
            )
        ),
    )


def training_arguments(
    repo_root: Path | str,
    cell_id: str,
    *,
    data_root: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Return the only permitted train.py argument set for one primary cell."""

    cell = resolve_fresh_cell(repo_root, cell_id)
    training = cell.expected_config["training"]
    transform = cell.expected_config["transform"]
    yaw_step_deg = transform["signed_generator"] / transform["stride"]
    _require(math.isclose(yaw_step_deg, 2.0, rel_tol=0.0, abs_tol=0.0),
             "fresh roll generator no longer implies the pinned 2-degree step")
    return {
        "fresh_cell_id": cell_id,
        "data_root": str(Path(data_root).expanduser().resolve(strict=True)),
        "object": cell.expected_config["object"]["name"],
        "pairs_index": None,
        "indexed_mode": training["indexed_mode"],
        "train_pairs_index": cell.train_pairs_path,
        "val_pairs_index": cell.validation_pairs_path,
        "test_pairs_index": None,
        "dataset_binding_sha256": cell.expected_config["dataset"]["binding_sha256"],
        "img_size": training["img_size"],
        "num_keypoints": training["num_keypoints"],
        "base_channels": training["base_channels"],
        "heatmap_res": training["heatmap_res"],
        "temperature": training["temperature"],
        "padding_mode": training["padding_mode"],
        "operator_type": training["operator_type"],
        "learn_inverse_operator": training["learn_inverse_operator"],
        "lambda_smooth": training["lambda_smooth"],
        "lambda_disp": training["lambda_disp"],
        "lambda_ent": training["lambda_ent"],
        "lambda_act": training["lambda_act"],
        "lambda_loc": training["lambda_loc"],
        "lambda_inv": training["lambda_inv"],
        "lambda_cycle": training["lambda_cycle"],
        "sigma": training["sigma"],
        "loc_bg_threshold": training["loc_bg_threshold"],
        "num_action_classes": training["num_action_classes"],
        "frame_skip": transform["stride"],
        "yaw_step_deg": yaw_step_deg,
        "center_crop": training["center_crop"],
        "epochs": training["epochs"],
        "frozen_epochs": None,
        "batch_size": training["batch_size"],
        "lr": training["lr"],
        "weight_decay": training["weight_decay"],
        "seed": cell.seed,
        "output_dir": str(Path(output_root).expanduser().resolve()),
        "save_every": training["save_every"],
        "log_every": training["log_every"],
        "auto_eval": training["auto_eval"],
        "auto_eval_checkpoint": "best",
        "eval_frames_dir": None,
        "eval_max_k": 10,
    }


def bind_training_namespace(
    repo_root: Path | str,
    args: argparse.Namespace,
) -> FreshRunBinding:
    """Reject any direct train.py argument that differs from its cell."""

    root = Path(repo_root).resolve(strict=True)
    _require(isinstance(args.fresh_cell_id, str), "fresh_cell_id is required")
    expected = training_arguments(
        root,
        args.fresh_cell_id,
        data_root=args.data_root,
        output_root=args.output_dir,
    )
    for name, value in expected.items():
        _require(
            hasattr(args, name) and type(getattr(args, name)) is type(value)
            and getattr(args, name) == value,
            f"fresh argument --{name} differs from frozen cell",
        )
    cell = resolve_fresh_cell(root, args.fresh_cell_id)
    source_commit, branch = _source_state(root, require_clean=True)
    _require(source_commit == cell.source_commit, "cell/source commit changed")
    data_root = Path(args.data_root).resolve(strict=True)
    _require(data_root.name == cell.expected_config["dataset"]["basename"],
             "data_root basename differs from frozen corpus")
    run_directory = (Path(args.output_dir).resolve() / cell.cell_id).absolute()
    _require(not run_directory.exists(), "fresh cell output already exists")
    return FreshRunBinding(cell=cell, source_branch=branch,
                           run_directory=str(run_directory))


def _file_record(path: Path, *, name: str) -> dict[str, Any]:
    data, size = _read_regular(path.absolute(), name=name)
    return {
        "absolute_path": str(path.absolute()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": size,
    }


def committed_source_records(repo_root: Path, source_commit: str) -> list[dict[str, str]]:
    records = []
    for relative_path in RUN_SOURCE_PATHS:
        data = _committed_bytes(repo_root, relative_path, source_commit)
        records.append(
            {
                "repo_relative_path": relative_path,
                "file_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def write_completed_run_receipt(
    repo_root: Path | str,
    binding: FreshRunBinding,
    *,
    device: str,
    optimizer_step_count: int,
    determinism: Mapping[str, Any],
    nondeterminism_evidence: Mapping[str, Any],
    full_command: list[str],
    runtime_environment: Mapping[str, Any],
) -> Path:
    """Write the receipt once, after all three authoritative files exist."""

    root = Path(repo_root).resolve(strict=True)
    current_commit, current_branch = _source_state(root, require_clean=True)
    _require(current_commit == binding.cell.source_commit, "source commit changed during run")
    _require(current_branch == binding.source_branch, "source branch changed during run")
    run_directory = Path(binding.run_directory).resolve(strict=True)
    policy = fresh_roll_determinism.policy_for_heatmap_resolution(
        root,
        int(binding.cell.expected_config["training"]["heatmap_res"]),
    )
    amendment = fresh_roll_determinism.amendment_record(root)
    fresh_roll_determinism.validate_determinism_record(
        determinism,
        expected_seed=binding.cell.seed,
        expected_policy=policy,
    )
    fresh_roll_determinism.validate_final_warning_evidence(
        nondeterminism_evidence,
        policy=policy,
        device_type=device.split(":", 1)[0],
    )
    files = {
        "checkpoint": _file_record(run_directory / "best_model.pt", name="checkpoint"),
        "config": _file_record(run_directory / "config.json", name="config"),
        "history": _file_record(run_directory / "history.json", name="history"),
    }
    document: dict[str, Any] = {
        "schema_version": RUN_RECEIPT_SCHEMA_VERSION,
        "artifact_type": "roll_head_package_completed_run_receipt",
        "experiment_manifest": {
            "repo_relative_path": str(EXPERIMENT_MANIFEST_RELATIVE_PATH),
            "file_sha256": binding.cell.manifest_file_sha256,
            "content_hash_sha256": EXPERIMENT_MANIFEST_CONTENT_SHA256,
        },
        "determinism_amendment": amendment,
        "cell_id": binding.cell.cell_id,
        "source_commit": current_commit,
        "source_branch": current_branch,
        "run_directory": str(run_directory),
        "source_files": committed_source_records(root, current_commit),
        "files": files,
        "execution": {
            "training_completed": True,
            "training_mode": "development",
            "device": device,
            "optimizer_step_count": optimizer_step_count,
            "authoritative_checkpoint_name": "best_model.pt",
            "test_loader_constructed": False,
            "selection_use_authorized": True,
            "determinism": dict(determinism),
            "nondeterminism_evidence": copy.deepcopy(
                dict(nondeterminism_evidence)
            ),
            "full_command": list(full_command),
            "runtime_environment": dict(runtime_environment),
        },
        "expected_embedded_config": copy.deepcopy(dict(binding.cell.expected_config)),
    }
    document["content_hash_sha256"] = canonical_sha256(document)
    receipt_path = run_directory / RUN_RECEIPT_NAME
    data = canonical_json_bytes(document, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(receipt_path, flags, 0o644)
    except OSError as exc:
        raise FreshCheckpointAuthorizationError("cannot create completed-run receipt") from exc
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
    finally:
        os.close(descriptor)
    return receipt_path


def _validated_run_file(
    record: Mapping[str, Any], *, name: str, run_directory: Path
) -> tuple[str, int, str]:
    _require(set(record) == {"absolute_path", "sha256", "size_bytes"},
             f"{name} record keys differ")
    path = Path(str(record["absolute_path"]))
    _require(path.is_absolute() and path.parent == run_directory,
             f"{name} is outside run directory")
    data, size = _read_regular(path, name=name)
    digest = hashlib.sha256(data).hexdigest()
    _require(type(record["size_bytes"]) is int and record["size_bytes"] == size,
             f"{name} size mismatch")
    _require(record["sha256"] == digest, f"{name} hash mismatch")
    return str(path), size, digest


def authorize_completed_fresh_run(
    repo_root: Path | str,
    receipt_path: Path | str,
    *,
    expected_cell_id: str,
) -> FreshCheckpointCapability:
    root = Path(repo_root).resolve(strict=True)
    cell = resolve_fresh_cell(root, expected_cell_id)
    source_commit, source_branch = _source_state(root, require_clean=True)
    receipt = Path(receipt_path).absolute()
    receipt_data, receipt_size = _read_regular(receipt, name="completed-run receipt")
    document = _strict_json(receipt_data, name="completed-run receipt")
    _require(isinstance(document, Mapping), "completed-run receipt must be an object")
    _content_hash_is_valid(document, name="completed-run receipt")
    required = {
        "schema_version", "artifact_type", "content_hash_sha256",
        "experiment_manifest", "determinism_amendment", "cell_id",
        "source_commit", "source_branch",
        "run_directory", "source_files", "files", "execution",
        "expected_embedded_config",
    }
    _require(set(document) == required, "completed-run receipt keys differ")
    _require(document["schema_version"] == RUN_RECEIPT_SCHEMA_VERSION,
             "completed-run receipt schema differs")
    _require(document["cell_id"] == expected_cell_id, "receipt cell differs")
    _require(document["source_commit"] == source_commit == cell.source_commit,
             "receipt source commit differs")
    _require(document["source_branch"] == source_branch == EXPECTED_BRANCH,
             "receipt source branch differs")
    _require(document["experiment_manifest"] == {
        "repo_relative_path": str(EXPERIMENT_MANIFEST_RELATIVE_PATH),
        "file_sha256": cell.manifest_file_sha256,
        "content_hash_sha256": EXPERIMENT_MANIFEST_CONTENT_SHA256,
    }, "receipt experiment binding differs")
    amendment = fresh_roll_determinism.amendment_record(root)
    _require(document["determinism_amendment"] == amendment,
             "receipt determinism amendment differs")
    _require(document["source_files"] == committed_source_records(root, source_commit),
             "receipt source files differ")
    run_directory = Path(str(document["run_directory"])).resolve(strict=True)
    _require(receipt.parent == run_directory, "receipt is outside run directory")
    files = document["files"]
    _require(isinstance(files, Mapping) and set(files) == {"checkpoint", "config", "history"},
             "receipt file roles differ")
    checkpoint = _validated_run_file(files["checkpoint"], name="checkpoint",
                                     run_directory=run_directory)
    config = _validated_run_file(files["config"], name="config",
                                 run_directory=run_directory)
    history = _validated_run_file(files["history"], name="history",
                                  run_directory=run_directory)
    config_doc = _strict_json(Path(config[0]).read_bytes(), name="config")
    history_doc = _strict_json(Path(history[0]).read_bytes(), name="history")
    _require(isinstance(config_doc, Mapping), "config must be an object")
    _require(isinstance(history_doc, list) and history_doc,
             "history must be a non-empty list")
    _require(config_doc.get("fresh_run_contract") == cell.expected_config,
             "config fresh-run contract differs")
    _require(config_doc.get("source_commit") == source_commit,
             "config source commit differs")
    execution = document["execution"]
    _require(isinstance(execution, Mapping), "receipt execution is invalid")
    expected_execution_keys = {
        "training_completed", "training_mode", "device",
        "optimizer_step_count", "authoritative_checkpoint_name",
        "test_loader_constructed", "selection_use_authorized",
        "determinism", "nondeterminism_evidence", "full_command",
        "runtime_environment",
    }
    _require(set(execution) == expected_execution_keys,
             "receipt execution keys differ")
    _require(execution.get("training_completed") is True
             and execution.get("training_mode") == "development",
             "training is not complete development training")
    _require(execution.get("authoritative_checkpoint_name") == "best_model.pt"
             and Path(checkpoint[0]).name == "best_model.pt",
             "authoritative checkpoint differs")
    _require(execution.get("test_loader_constructed") is False,
             "test loader was constructed")
    _require(type(execution.get("optimizer_step_count")) is int
             and execution["optimizer_step_count"] > 0,
             "optimizer step count is invalid")
    full_command = execution.get("full_command")
    _require(isinstance(full_command, list) and full_command
             and all(isinstance(value, str) and value for value in full_command),
             "full command is invalid")
    runtime_environment = execution.get("runtime_environment")
    expected_environment_keys = {
        "python_implementation", "python_version", "pytorch_version",
        "torchvision_version", "numpy_version", "pytorch_cuda_version",
        "cudnn_version", "device_type", "gpu_name",
        "nvidia_driver_version", "driver_visible_cuda_version",
        "slurm_job_id", "slurm_job_script_sha256",
    }
    _require(isinstance(runtime_environment, Mapping)
             and set(runtime_environment) == expected_environment_keys,
             "runtime environment keys differ")
    for version_key in (
        "python_implementation", "python_version", "pytorch_version",
        "torchvision_version", "numpy_version", "device_type",
    ):
        _require(isinstance(runtime_environment.get(version_key), str)
                 and runtime_environment[version_key],
                 f"runtime environment {version_key} is invalid")
    receipt_device = execution.get("device")
    _require(runtime_environment["device_type"] == receipt_device,
             "runtime environment device differs")
    policy = fresh_roll_determinism.policy_for_heatmap_resolution(
        root,
        int(cell.expected_config["training"]["heatmap_res"]),
    )
    fresh_roll_determinism.validate_determinism_record(
        execution.get("determinism"),
        expected_seed=cell.seed,
        expected_policy=policy,
    )
    fresh_roll_determinism.validate_final_warning_evidence(
        execution.get("nondeterminism_evidence"),
        policy=policy,
        device_type=str(receipt_device).split(":", 1)[0],
    )
    if receipt_device == "cuda":
        for cuda_key in (
            "pytorch_cuda_version", "gpu_name", "nvidia_driver_version",
            "driver_visible_cuda_version",
        ):
            _require(isinstance(runtime_environment.get(cuda_key), str)
                     and runtime_environment[cuda_key],
                     f"CUDA environment {cuda_key} is invalid")
    slurm_job_id = runtime_environment.get("slurm_job_id")
    slurm_script_hash = runtime_environment.get("slurm_job_script_sha256")
    _require((slurm_job_id is None and slurm_script_hash is None)
             or (isinstance(slurm_job_id, str) and slurm_job_id
                 and isinstance(slurm_script_hash, str)
                 and _SHA256_RE.fullmatch(slurm_script_hash) is not None),
             "Slurm provenance is incomplete")
    _require(config_doc.get("full_command") == full_command,
             "config full command differs")
    _require(config_doc.get("runtime_environment") == runtime_environment,
             "config runtime environment differs")
    _require(config_doc.get("determinism") == execution["determinism"],
             "config determinism record differs")
    _require(config_doc.get("determinism_amendment") == amendment,
             "config determinism amendment differs")
    _require(document["expected_embedded_config"] == cell.expected_config,
             "receipt embedded config differs")
    return FreshCheckpointCapability(
        cell_id=expected_cell_id,
        source_commit=source_commit,
        checkpoint_absolute_path=checkpoint[0],
        checkpoint_size_bytes=checkpoint[1],
        checkpoint_sha256=checkpoint[2],
        config_absolute_path=config[0],
        config_size_bytes=config[1],
        config_sha256=config[2],
        history_absolute_path=history[0],
        history_size_bytes=history[1],
        history_sha256=history[2],
        receipt_absolute_path=str(receipt),
        receipt_file_sha256=hashlib.sha256(receipt_data).hexdigest(),
        receipt_size_bytes=receipt_size,
        expected_embedded_config=cell.expected_config,
        _capability=_VERIFIED_FRESH_CAPABILITY,
    )


def require_fresh_checkpoint_capability(
    value: FreshCheckpointCapability,
) -> FreshCheckpointCapability:
    _require(isinstance(value, FreshCheckpointCapability)
             and value._capability is _VERIFIED_FRESH_CAPABILITY,
             "fresh checkpoint capability is not authentic")
    return value


def validate_loaded_checkpoint(
    capability: FreshCheckpointCapability,
    loaded_checkpoint: Mapping[str, Any],
) -> None:
    checked = require_fresh_checkpoint_capability(capability)
    _require(isinstance(loaded_checkpoint, Mapping), "checkpoint is not a mapping")
    config = loaded_checkpoint.get("config")
    _require(isinstance(config, Mapping), "checkpoint lacks embedded config")
    _require(config.get("fresh_run_contract") == checked.expected_embedded_config,
             "checkpoint embedded fresh-run contract differs")
    _require(config.get("source_commit") == checked.source_commit,
             "checkpoint source commit differs")
    _require(config.get("cell_id") == checked.cell_id, "checkpoint cell differs")


def command_arguments(arguments: Mapping[str, Any]) -> list[str]:
    """Encode the internal exact Namespace as train.py CLI arguments."""

    result: list[str] = []
    boolean_flags = {"learn_inverse_operator", "auto_eval"}
    for name, value in arguments.items():
        if value is None:
            continue
        if name in boolean_flags:
            if value:
                result.append(f"--{name}")
            continue
        result.extend([f"--{name}", str(value)])
    return result


__all__ = [
    "EXPERIMENT_MANIFEST_RELATIVE_PATH",
    "FreshCheckpointAuthorizationError",
    "FreshCheckpointCapability",
    "FreshRunBinding",
    "FrozenFreshCell",
    "authorize_completed_fresh_run",
    "bind_training_namespace",
    "canonical_json_bytes",
    "canonical_sha256",
    "command_arguments",
    "expected_cell_config",
    "resolve_fresh_cell",
    "require_fresh_checkpoint_capability",
    "training_arguments",
    "validate_experiment_manifest",
    "validate_loaded_checkpoint",
    "write_completed_run_receipt",
]
