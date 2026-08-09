"""Fail-closed authorization for one approved held-out roll training run.

The module deliberately contains no baked-in recipe, epoch, object, or seed.
Those scientific choices must be written in a committed run manifest, reviewed,
and explicitly approved before ``train.py`` can enter fixed-final mode.
"""

from __future__ import annotations

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


MANIFEST_SCHEMA_VERSION = "heldout_roll_fixed_final_manifest.v1"
TRAINING_RECEIPT_SCHEMA_VERSION = "heldout_roll_fixed_final_training_receipt.v1"
EVIDENCE_RECEIPT_SCHEMA_VERSION = "heldout_roll_fixed_final_evidence_receipt.v1"
AUTHORITY_PREFIX = Path("docs/decisions/heldout_roll_fixed_final/v1")
MANIFEST_PREFIX = AUTHORITY_PREFIX / "manifests"
TRAINING_RECEIPT_NAME = "FIXED_FINAL_TRAINING_RECEIPT.json"
EVIDENCE_RECEIPT_NAME = "FIXED_FINAL_EVIDENCE_RECEIPT.json"
SOURCE_PATHS = (
    "keypoint_net/train.py",
    "keypoint_net/dataset.py",
    "keypoint_net/model.py",
    "keypoint_net/descriptor_attachment.py",
    "keypoint_net/fresh_roll_determinism.py",
    "keypoint_net/FRESH_ROLL_DETERMINISM_AMENDMENT.json",
    "keypoint_net/eval_representation.py",
    "keypoint_net/representation_array_codec.py",
    "keypoint_net/representation_corpus_inventory.py",
    "keypoint_net/representation_evaluation_provenance.py",
    "keypoint_net/representation_split_adapter.py",
    "keypoint_net/representation_split_bundle.py",
    "keypoint_net/representation_split_verifier.py",
    "keypoint_net/representation_splits.py",
    "keypoint_net/representation_fixed_final_authorization.py",
    "keypoint_net/representation_fixed_final_runtime.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,127}")
_CAPABILITY_TOKEN = object()

_TRAINING_ARGUMENT_KEYS = frozenset(
    {
        "img_size", "num_keypoints", "base_channels", "heatmap_res",
        "temperature", "padding_mode", "operator_type",
        "learn_inverse_operator", "lambda_smooth", "lambda_disp",
        "lambda_ent", "lambda_act", "lambda_loc", "lambda_inv",
        "lambda_cycle", "lambda_attach", "sigma", "loc_bg_threshold",
        "num_action_classes", "frame_skip", "yaw_step_deg", "center_crop",
        "epochs", "frozen_epochs", "batch_size", "lr", "weight_decay",
        "seed", "save_every",
    }
)


class FixedFinalAuthorizationError(ValueError):
    """The run manifest, source tree, or receipt is not exactly authorized."""


@dataclass(frozen=True)
class FixedFinalRunBinding:
    run_id: str
    recipe_id: str
    object_id: str
    object_role: str
    seed: int
    frozen_epochs: int
    source_commit: str
    source_branch: str
    run_directory: str
    manifest_repo_relative_path: str
    manifest_absolute_path: str
    manifest_file_sha256: str
    manifest_content_sha256: str
    train_pair_repo_relative_path: str
    train_pair_file_sha256: str
    train_pair_content_sha256: str
    test_pair_repo_relative_path: str
    test_pair_file_sha256: str
    test_pair_content_sha256: str
    dataset_binding_sha256: str
    corpus_inventory_repo_relative_path: str
    geometry_repo_relative_path: str
    implementation_lock_repo_relative_path: str
    decision_spec_repo_relative_path: str
    pro_review_repo_relative_path: str
    fable_review_repo_relative_path: str
    user_approval_repo_relative_path: str
    expected_training_arguments: Mapping[str, Any]
    _capability: object


@dataclass(frozen=True)
class FixedFinalCheckpointCapability:
    binding: FixedFinalRunBinding
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
    _capability: object


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixedFinalAuthorizationError(message)


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes, *, name: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
        raise FixedFinalAuthorizationError(
            f"{name} contains forbidden constant {token!r}"
        )

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except FixedFinalAuthorizationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FixedFinalAuthorizationError(f"{name} is not strict JSON") from exc
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _read_regular(path: Path, *, name: str) -> tuple[bytes, int]:
    _require(path.is_absolute(), f"{name} path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FixedFinalAuthorizationError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks), metadata.st_size
    finally:
        os.close(descriptor)


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {"GIT_CONFIG_NOSYSTEM": "1", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}
    )
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FixedFinalAuthorizationError(
            f"git command failed: {' '.join(arguments)}"
        ) from exc
    return completed.stdout if binary else completed.stdout.strip()


def _source_state(root: Path, *, require_clean: bool) -> tuple[str, str]:
    commit = str(_git(root, "rev-parse", "HEAD"))
    branch = str(_git(root, "rev-parse", "--abbrev-ref", "HEAD"))
    _require(_COMMIT_RE.fullmatch(commit) is not None, "Git HEAD is invalid")
    _require(branch != "HEAD", "fixed-final execution requires a named branch")
    if require_clean:
        tracked = str(_git(root, "status", "--porcelain", "--untracked-files=no"))
        _require(not tracked, "tracked source is modified")
    return commit, branch


def _repo_relative(root: Path, path: Path, *, name: str) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise FixedFinalAuthorizationError(f"{name} is outside the repository") from exc
    _require(".." not in Path(relative).parts, f"unsafe {name} path")
    return relative


def _committed_bytes(root: Path, relative_path: str, commit: str) -> bytes:
    _require(relative_path and not Path(relative_path).is_absolute(), "unsafe repository path")
    committed = _git(root, "show", f"{commit}:{relative_path}", binary=True)
    path = (root / relative_path).resolve(strict=True)
    current, _ = _read_regular(path, name=relative_path)
    _require(current == committed, f"working bytes differ from Git for {relative_path}")
    return current


def _content_hash(document: Mapping[str, Any], *, name: str) -> str:
    claimed = document.get("content_hash_sha256")
    _require(
        isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed) is not None,
        f"{name} content hash is invalid",
    )
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(canonical_sha256(payload) == claimed, f"{name} content hash mismatch")
    return claimed


def _validate_file_record(
    root: Path,
    commit: str,
    record: Any,
    *,
    name: str,
    prefix: Path | None = None,
) -> tuple[str, str]:
    _require(isinstance(record, Mapping), f"{name} record is invalid")
    _require(
        set(record) == {"repo_relative_path", "file_sha256"},
        f"{name} record keys differ",
    )
    relative = record["repo_relative_path"]
    digest = record["file_sha256"]
    _require(isinstance(relative, str) and relative, f"{name} path is empty")
    _require(
        isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
        f"{name} SHA-256 is invalid",
    )
    relative_path = Path(relative)
    _require(not relative_path.is_absolute() and ".." not in relative_path.parts,
             f"{name} path is unsafe")
    if prefix is not None:
        _require(
            relative_path.is_relative_to(prefix),
            f"{name} must be below {prefix.as_posix()}",
        )
    data = _committed_bytes(root, relative, commit)
    _require(hashlib.sha256(data).hexdigest() == digest, f"{name} hash mismatch")
    return relative, digest


def _validate_pair_record(
    root: Path,
    commit: str,
    record: Any,
    *,
    name: str,
    expected_split: str,
) -> tuple[str, str, str]:
    _require(isinstance(record, Mapping), f"{name} record is invalid")
    _require(
        set(record) == {
            "repo_relative_path", "file_sha256", "content_hash_sha256"
        },
        f"{name} record keys differ",
    )
    relative, digest = _validate_file_record(
        root,
        commit,
        {"repo_relative_path": record["repo_relative_path"],
         "file_sha256": record["file_sha256"]},
        name=name,
        prefix=Path("docs/decisions/2026-07-26/representation_oracle_splits/pairs"),
    )
    document = _strict_json(_committed_bytes(root, relative, commit), name=name)
    content_hash = _content_hash(document, name=name)
    _require(content_hash == record["content_hash_sha256"], f"{name} content binding differs")
    _require(document.get("split") == expected_split, f"{name} split differs")
    transform = document.get("transform")
    _require(
        isinstance(transform, Mapping)
        and transform.get("family") == "roll"
        and transform.get("physical_axis") == "world_z"
        and transform.get("direction") == "forward"
        and transform.get("stride") == 3
        and transform.get("signed_generator") == 6.0
        and transform.get("cyclic") is True,
        f"{name} is not the frozen +6 degree roll stratum",
    )
    return relative, digest, content_hash


def _validate_json_path_reference(
    root: Path,
    commit: str,
    record: Any,
    *,
    name: str,
    prefix: Path,
) -> tuple[str, Mapping[str, Any], str]:
    """Load one committed JSON authority record referenced by path only.

    Authority records bind the manifest hash, so embedding their file hashes in
    the manifest would create a circular hash dependency.  The user approval
    instead binds the exact decision/review/implementation-lock file hashes.
    """

    _require(
        isinstance(record, Mapping) and set(record) == {"repo_relative_path"},
        f"{name} path reference differs",
    )
    relative = record["repo_relative_path"]
    _require(isinstance(relative, str) and relative, f"{name} path is empty")
    path = Path(relative)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and path.is_relative_to(prefix)
        and path.suffix == ".json",
        f"{name} must be one JSON file below {prefix.as_posix()}",
    )
    raw = _committed_bytes(root, relative, commit)
    return relative, _strict_json(raw, name=name), hashlib.sha256(raw).hexdigest()


def _validate_implementation_lock(
    root: Path,
    commit: str,
    record: Any,
) -> tuple[str, str]:
    relative, digest = _validate_file_record(
        root,
        commit,
        record,
        name="implementation lock",
        prefix=AUTHORITY_PREFIX / "implementation_locks",
    )
    _require(Path(relative).suffix == ".json", "implementation lock must be JSON")
    document = _strict_json(
        _committed_bytes(root, relative, commit), name="implementation lock"
    )
    _require(
        set(document)
        == {
            "schema_version",
            "artifact_type",
            "content_hash_sha256",
            "source_files",
        },
        "implementation lock keys differ",
    )
    _require(
        document["schema_version"]
        == "heldout_roll_fixed_final_implementation_lock.v1"
        and document["artifact_type"]
        == "heldout_roll_fixed_final_implementation_lock",
        "implementation lock type differs",
    )
    _content_hash(document, name="implementation lock")
    expected_sources = [
        {
            "repo_relative_path": relative_path,
            "file_sha256": hashlib.sha256(
                _committed_bytes(root, relative_path, commit)
            ).hexdigest(),
        }
        for relative_path in SOURCE_PATHS
    ]
    _require(
        document["source_files"] == expected_sources,
        "current decision-critical source differs from approved implementation lock",
    )
    return relative, digest


def _validate_authority_records(
    root: Path,
    commit: str,
    document: Mapping[str, Any],
    *,
    manifest_content_sha256: str,
    implementation_lock_sha256: str,
) -> dict[str, str]:
    common = {
        "run_id": document["run_id"],
        "manifest_content_sha256": manifest_content_sha256,
    }
    decision_relative, decision, decision_sha256 = _validate_json_path_reference(
        root,
        commit,
        document["decision_spec"],
        name="decision specification",
        prefix=AUTHORITY_PREFIX / "decisions",
    )
    _require(
        set(decision)
        == {
            "schema_version",
            "artifact_type",
            "content_hash_sha256",
            "run_id",
            "manifest_content_sha256",
            "object",
            "recipe_id",
            "frozen_epochs",
        },
        "decision specification keys differ",
    )
    _require(
        decision["schema_version"]
        == "heldout_roll_fixed_final_decision_spec.v1"
        and decision["artifact_type"]
        == "heldout_roll_fixed_final_decision_spec",
        "decision specification type differs",
    )
    _content_hash(decision, name="decision specification")
    _require(
        {key: decision[key] for key in common} == common
        and decision["object"] == document["object"]
        and decision["recipe_id"] == document["recipe"]["id"]
        and decision["frozen_epochs"] == document["training_arguments"]["frozen_epochs"],
        "decision specification does not bind the exact run manifest",
    )

    review_records: dict[str, tuple[str, str, str]] = {}
    for role, reviewer in (("pro_review", "chatgpt_pro"), ("fable_review", "fable")):
        relative, review, file_sha256 = _validate_json_path_reference(
            root,
            commit,
            document[role],
            name=role.replace("_", " "),
            prefix=AUTHORITY_PREFIX / "reviews",
        )
        _require(
            set(review)
            == {
                "schema_version",
                "artifact_type",
                "content_hash_sha256",
                "reviewer",
                "verdict",
                "run_id",
                "manifest_content_sha256",
                "decision_spec_file_sha256",
                "implementation_lock_file_sha256",
                "raw_report",
            },
            f"{role.replace('_', ' ')} keys differ",
        )
        _require(
            review["schema_version"] == "heldout_roll_fixed_final_review.v1"
            and review["artifact_type"] == "heldout_roll_fixed_final_review"
            and review["reviewer"] == reviewer
            and review["verdict"]
            in {"PASS", "PASS_WITH_REQUIRED_FIXES", "BLOCK"},
            f"{role.replace('_', ' ')} type or verdict differs",
        )
        _content_hash(review, name=role.replace("_", " "))
        _require(
            {key: review[key] for key in common} == common
            and review["decision_spec_file_sha256"] == decision_sha256
            and review["implementation_lock_file_sha256"]
            == implementation_lock_sha256,
            f"{role.replace('_', ' ')} does not bind this run and implementation",
        )
        raw_relative, _ = _validate_file_record(
            root,
            commit,
            review["raw_report"],
            name=f"{role.replace('_', ' ')} raw report",
            prefix=AUTHORITY_PREFIX / "reports",
        )
        _require(
            raw_relative != relative,
            f"{role.replace('_', ' ')} raw report cannot be its review record",
        )
        review_records[role] = (relative, file_sha256, raw_relative)

    _require(
        review_records["pro_review"][2] != review_records["fable_review"][2],
        "Pro and Fable reviews must bind distinct raw reports",
    )

    approval_relative, approval, _ = _validate_json_path_reference(
        root,
        commit,
        document["user_approval"],
        name="user approval",
        prefix=AUTHORITY_PREFIX / "approvals",
    )
    _require(
        set(approval)
        == {
            "schema_version",
            "artifact_type",
            "content_hash_sha256",
            "affirmative_authorization",
            "authorization",
            "reviewer_findings_are_advisory",
            "run_id",
            "manifest_content_sha256",
            "object",
            "recipe_id",
            "frozen_epochs",
            "decision_spec_file_sha256",
            "pro_review_file_sha256",
            "fable_review_file_sha256",
            "implementation_lock_file_sha256",
        },
        "user approval keys differ",
    )
    _require(
        approval["schema_version"] == "heldout_roll_fixed_final_user_approval.v1"
        and approval["artifact_type"] == "heldout_roll_fixed_final_user_approval"
        and approval["affirmative_authorization"] is True
        and approval["authorization"] == "authorize_exact_fixed_final_run"
        and approval["reviewer_findings_are_advisory"] is True,
        "user approval is not an explicit affirmative authorization",
    )
    _content_hash(approval, name="user approval")
    _require(
        {key: approval[key] for key in common} == common
        and approval["object"] == document["object"]
        and approval["recipe_id"] == document["recipe"]["id"]
        and approval["frozen_epochs"]
        == document["training_arguments"]["frozen_epochs"]
        and approval["decision_spec_file_sha256"] == decision_sha256
        and approval["pro_review_file_sha256"] == review_records["pro_review"][1]
        and approval["fable_review_file_sha256"]
        == review_records["fable_review"][1]
        and approval["implementation_lock_file_sha256"]
        == implementation_lock_sha256,
        "user approval does not bind this run, reviews, and implementation",
    )
    paths = {
        "implementation_relative": document["implementation_lock"][
            "repo_relative_path"
        ],
        "decision_relative": decision_relative,
        "pro_relative": review_records["pro_review"][0],
        "fable_relative": review_records["fable_review"][0],
        "approval_relative": approval_relative,
    }
    _require(len(set(paths.values())) == len(paths), "authority record paths must be distinct")
    return paths


def validate_run_manifest(
    repo_root: Path | str,
    manifest_path: Path | str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve(strict=True)
    commit, branch = _source_state(root, require_clean=True)
    path = Path(manifest_path).expanduser().resolve(strict=True)
    relative = _repo_relative(root, path, name="fixed-final manifest")
    _require(
        Path(relative).is_relative_to(MANIFEST_PREFIX),
        f"fixed-final manifest must be below {MANIFEST_PREFIX.as_posix()}",
    )
    raw = _committed_bytes(root, relative, commit)
    document = _strict_json(raw, name="fixed-final manifest")
    required = {
        "schema_version", "artifact_type", "content_hash_sha256", "run_id",
        "object", "recipe", "transform", "dataset", "splits", "geometry",
        "training_arguments", "evidence_policy", "decision_spec",
        "pro_review", "fable_review", "user_approval", "implementation_lock",
    }
    _require(set(document) == required, "fixed-final manifest keys differ")
    _require(document["schema_version"] == MANIFEST_SCHEMA_VERSION,
             "fixed-final manifest schema differs")
    _require(document["artifact_type"] == "heldout_roll_fixed_final_manifest",
             "fixed-final manifest artifact type differs")
    content_hash = _content_hash(document, name="fixed-final manifest")
    run_id = document["run_id"]
    _require(isinstance(run_id, str) and _RUN_ID_RE.fullmatch(run_id) is not None,
             "fixed-final run_id is invalid")

    recipe = document["recipe"]
    _require(isinstance(recipe, Mapping) and set(recipe) == {"id", "source"}
             and recipe["id"] in {"task55_clean", "task80_assisted"},
             "fixed-final recipe must be task55_clean or task80_assisted")
    recipe_relative, _ = _validate_file_record(
        root,
        commit,
        recipe["source"],
        name="recipe specification",
    )
    _require(
        recipe_relative
        == "docs/decisions/2026-07-29/roll_head_package_training/"
        "EXPERIMENT_MANIFEST_v1.json",
        "fixed-final recipe must use the frozen roll head-package recipe source",
    )
    recipe_source = _strict_json(
        _committed_bytes(root, recipe_relative, commit), name="recipe specification"
    )
    recipe_arguments = recipe_source.get("recipes", {}).get(recipe["id"])
    _require(
        isinstance(recipe_arguments, Mapping),
        "recipe specification lacks the selected recipe",
    )

    object_record = document["object"]
    _require(isinstance(object_record, Mapping)
             and set(object_record) == {"id", "role"}, "object record differs")
    _require(object_record["role"] in {"confirmation", "final_test"},
             "fixed-final object role must be confirmation or final_test")
    _require(isinstance(object_record["id"], str) and object_record["id"],
             "fixed-final object id is empty")

    transform = document["transform"]
    _require(
        transform == {
            "family": "roll", "physical_axis": "world_z",
            "direction": "forward", "signed_generator": 6.0,
            "generator_units": "degrees", "stride": 3,
            "stride_units": "frames", "cyclic": True,
            "expected_2d_family": "planar_rotation_about_projected_center",
        },
        "fixed-final transform must be the frozen forward roll stratum",
    )
    dataset = document["dataset"]
    _require(isinstance(dataset, Mapping)
             and set(dataset) == {"basename", "binding_sha256", "corpus_inventory"},
             "dataset record differs")
    _require(isinstance(dataset["basename"], str) and dataset["basename"],
             "dataset basename is empty")
    _require(isinstance(dataset["binding_sha256"], str)
             and _SHA256_RE.fullmatch(dataset["binding_sha256"]) is not None,
             "dataset binding is invalid")
    inventory_relative, _ = _validate_file_record(
        root, commit, dataset["corpus_inventory"], name="corpus inventory",
        prefix=Path("docs/decisions/2026-07-26/representation_oracle_splits/inventories"),
    )
    inventory_document = _strict_json(
        _committed_bytes(root, inventory_relative, commit),
        name="corpus inventory",
    )
    _require(_content_hash(inventory_document, name="corpus inventory")
             == dataset["binding_sha256"],
             "corpus inventory content hash differs from dataset binding")
    _require(inventory_document.get("dataset_basename") == dataset["basename"],
             "corpus inventory basename differs")

    splits = document["splits"]
    _require(isinstance(splits, Mapping) and set(splits) == {"train", "test"},
             "split record differs")
    train_relative, train_file_sha256, train_content_sha256 = _validate_pair_record(
        root, commit, splits["train"], name="train pair index", expected_split="train"
    )
    test_relative, test_file_sha256, test_content_sha256 = _validate_pair_record(
        root, commit, splits["test"], name="test pair index", expected_split="test"
    )
    for split_name, relative in (("train", train_relative), ("test", test_relative)):
        pair_document = _strict_json(
            _committed_bytes(root, relative, commit), name=f"{split_name} pair index"
        )
        _require(pair_document.get("dataset_binding_sha256")
                 == dataset["binding_sha256"],
                 f"{split_name} pair index dataset binding differs")
        _require(pair_document.get("dataset_basename") == dataset["basename"],
                 f"{split_name} pair index dataset basename differs")
        role_lock = pair_document.get("object_roles")
        _require(isinstance(role_lock, Mapping)
                 and role_lock.get(object_record["id"]) == object_record["role"],
                 f"{split_name} pair index object role differs")
        pairs = pair_document.get("pairs")
        _require(isinstance(pairs, list)
                 and any(row.get("model_name") == object_record["id"]
                         for row in pairs if isinstance(row, Mapping)),
                 f"{split_name} pair index does not contain the approved object")
    geometry_relative, _ = _validate_file_record(
        root, commit, document["geometry"], name="geometry binding",
        prefix=Path("docs/decisions/2026-07-26/representation_oracle_geometry/bindings"),
    )

    training = document["training_arguments"]
    _require(isinstance(training, Mapping)
             and set(training) == set(_TRAINING_ARGUMENT_KEYS),
             "training argument profile differs")
    _require(type(training["seed"]) is int, "training seed must be an integer")
    _require(type(training["frozen_epochs"]) is int
             and training["frozen_epochs"] > 0,
             "frozen_epochs must be a positive integer")
    _require(training["epochs"] == training["frozen_epochs"],
             "epochs must equal frozen_epochs")
    _require(training["img_size"] == 512 and training["center_crop"] is None,
             "fixed-final evaluator currently requires uncropped 512x512 inputs")
    _require(
        training["num_keypoints"] == 10
        and training["base_channels"] == 32
        and training["heatmap_res"] == 64
        and training["operator_type"] == "shared_affine"
        and training["padding_mode"] == "reflect"
        and training["frame_skip"] == 3
        and training["yaw_step_deg"] == 2.0
        and training["lambda_attach"] == 0.0,
        "held-out roll requires the retained 64-head shared-affine, "
        "descriptor-free package",
    )
    _require(
        all(training.get(key) == value for key, value in recipe_arguments.items()),
        "training arguments differ from the selected committed recipe",
    )

    _require(
        document["evidence_policy"] == {
            "checkpoint": "final_model.pt",
            "validation_loader": False,
            "test_partition": "test",
            "test_content_open_phase_count": 1,
            "unique_test_image_open_count_per_frame": 1,
            "unique_test_mask_open_count_per_frame": 1,
            "evaluator_protocol": "generic",
            "full_corpus_diagnostic": False,
            "automatic_threshold_decision": False,
            "automatic_tuning_or_repair": False,
        },
        "fixed-final evidence policy differs",
    )
    implementation_relative, implementation_sha256 = _validate_implementation_lock(
        root, commit, document["implementation_lock"]
    )
    authorities = _validate_authority_records(
        root,
        commit,
        document,
        manifest_content_sha256=content_hash,
        implementation_lock_sha256=implementation_sha256,
    )
    _require(
        authorities["implementation_relative"] == implementation_relative,
        "implementation lock authority path differs",
    )
    return document, {
        "root": root, "commit": commit, "branch": branch,
        "relative": relative, "absolute": str(path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_hash,
        "train_relative": train_relative,
        "train_file_sha256": train_file_sha256,
        "train_content_sha256": train_content_sha256,
        "test_relative": test_relative,
        "test_file_sha256": test_file_sha256,
        "test_content_sha256": test_content_sha256,
        "dataset_binding_sha256": dataset["binding_sha256"],
        "inventory_relative": inventory_relative,
        "geometry_relative": geometry_relative,
        **authorities,
    }


def bind_training_namespace(
    repo_root: Path | str,
    args: Any,
) -> FixedFinalRunBinding:
    _require(getattr(args, "indexed_mode", None) == "fixed-final",
             "fixed-final manifest requires indexed_mode=fixed-final")
    manifest_path = getattr(args, "fixed_final_manifest", None)
    _require(manifest_path is not None, "fixed-final mode requires --fixed_final_manifest")
    document, checked = validate_run_manifest(repo_root, manifest_path)
    training = dict(document["training_arguments"])
    for key in _TRAINING_ARGUMENT_KEYS:
        _require(getattr(args, key, None) == training[key],
                 f"training argument --{key} differs from approved manifest")
    _require(getattr(args, "object", None) == document["object"]["id"],
             "training object differs from approved manifest")
    _require(getattr(args, "fresh_cell_id", None) is None
             and getattr(args, "descriptor_cell_id", None) is None,
             "fixed-final mode cannot reuse development or descriptor authorities")
    _require(Path(getattr(args, "data_root")).expanduser().resolve(strict=True).name
             == document["dataset"]["basename"],
             "live data root basename differs from approved manifest")
    _require(getattr(args, "dataset_binding_sha256", None)
             == document["dataset"]["binding_sha256"],
             "live dataset binding differs from approved manifest")
    root = checked["root"]
    expected_train = (root / checked["train_relative"]).resolve(strict=True)
    expected_test = (root / checked["test_relative"]).resolve(strict=True)
    _require(Path(getattr(args, "train_pairs_index")).expanduser().resolve(strict=True)
             == expected_train, "train pair path differs from approved manifest")
    _require(Path(getattr(args, "test_pairs_index")).expanduser().resolve(strict=True)
             == expected_test, "test pair path differs from approved manifest")
    run_directory = (Path(getattr(args, "output_dir")).expanduser().resolve()
                     / document["run_id"])
    return FixedFinalRunBinding(
        run_id=document["run_id"], recipe_id=document["recipe"]["id"],
        object_id=document["object"]["id"],
        object_role=document["object"]["role"], seed=training["seed"],
        frozen_epochs=training["frozen_epochs"], source_commit=checked["commit"],
        source_branch=checked["branch"], run_directory=str(run_directory),
        manifest_repo_relative_path=checked["relative"],
        manifest_absolute_path=checked["absolute"],
        manifest_file_sha256=checked["file_sha256"],
        manifest_content_sha256=checked["content_sha256"],
        train_pair_repo_relative_path=checked["train_relative"],
        train_pair_file_sha256=checked["train_file_sha256"],
        train_pair_content_sha256=checked["train_content_sha256"],
        test_pair_repo_relative_path=checked["test_relative"],
        test_pair_file_sha256=checked["test_file_sha256"],
        test_pair_content_sha256=checked["test_content_sha256"],
        dataset_binding_sha256=checked["dataset_binding_sha256"],
        corpus_inventory_repo_relative_path=checked["inventory_relative"],
        geometry_repo_relative_path=checked["geometry_relative"],
        implementation_lock_repo_relative_path=checked["implementation_relative"],
        decision_spec_repo_relative_path=checked["decision_relative"],
        pro_review_repo_relative_path=checked["pro_relative"],
        fable_review_repo_relative_path=checked["fable_relative"],
        user_approval_repo_relative_path=checked["approval_relative"],
        expected_training_arguments=training, _capability=_CAPABILITY_TOKEN,
    )


def require_run_binding(binding: FixedFinalRunBinding) -> FixedFinalRunBinding:
    _require(isinstance(binding, FixedFinalRunBinding)
             and binding._capability is _CAPABILITY_TOKEN,
             "fixed-final run binding is not authorized")
    return binding


def _file_record(path: Path, *, name: str) -> dict[str, Any]:
    raw, size = _read_regular(path.resolve(strict=True), name=name)
    return {"absolute_path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": size}


def committed_source_records(root: Path, commit: str) -> list[dict[str, str]]:
    return [
        {
            "repo_relative_path": relative,
            "sha256": hashlib.sha256(_committed_bytes(root, relative, commit)).hexdigest(),
        }
        for relative in SOURCE_PATHS
    ]


def write_training_receipt(
    repo_root: Path | str,
    binding: FixedFinalRunBinding,
) -> Path:
    checked = require_run_binding(binding)
    root = Path(repo_root).resolve(strict=True)
    commit, branch = _source_state(root, require_clean=True)
    _require((commit, branch) == (checked.source_commit, checked.source_branch),
             "source state changed after fixed-final authorization")
    run_dir = Path(checked.run_directory).resolve(strict=True)
    files = {
        "checkpoint": _file_record(run_dir / "final_model.pt", name="final checkpoint"),
        "config": _file_record(run_dir / "config.json", name="run config"),
        "history": _file_record(run_dir / "history.json", name="training history"),
    }
    receipt: dict[str, Any] = {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "artifact_type": "heldout_roll_fixed_final_training_receipt",
        "run_id": checked.run_id,
        "source_commit": commit,
        "source_branch": branch,
        "run_manifest": {
            "repo_relative_path": checked.manifest_repo_relative_path,
            "file_sha256": checked.manifest_file_sha256,
            "content_hash_sha256": checked.manifest_content_sha256,
        },
        "source_files": committed_source_records(root, commit),
        "run_directory": str(run_dir),
        "files": files,
        "training_contract": {
            "recipe_id": checked.recipe_id,
            "object_id": checked.object_id,
            "object_role": checked.object_role,
            "seed": checked.seed,
            "frozen_epochs": checked.frozen_epochs,
            "authoritative_checkpoint": "final_model.pt",
            "validation_loader": False,
        },
        "test_boundary": {
            "test_dataset_constructed": False,
            "test_content_open_phase_count": 0,
            "test_metric_computed": False,
        },
    }
    receipt["content_hash_sha256"] = canonical_sha256(receipt)
    path = run_dir / TRAINING_RECEIPT_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, canonical_json_bytes(receipt, newline=True))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def authorize_training_receipt(
    repo_root: Path | str,
    binding: FixedFinalRunBinding,
    receipt_path: Path | str,
) -> FixedFinalCheckpointCapability:
    checked = require_run_binding(binding)
    root = Path(repo_root).resolve(strict=True)
    commit, branch = _source_state(root, require_clean=True)
    _require((commit, branch) == (checked.source_commit, checked.source_branch),
             "source state differs from fixed-final binding")
    path = Path(receipt_path).resolve(strict=True)
    raw, size = _read_regular(path, name="fixed-final training receipt")
    document = _strict_json(raw, name="fixed-final training receipt")
    claimed = _content_hash(document, name="fixed-final training receipt")
    _require(document.get("schema_version") == TRAINING_RECEIPT_SCHEMA_VERSION,
             "training receipt schema differs")
    _require(document.get("run_id") == checked.run_id,
             "training receipt run_id differs")
    _require(document.get("source_commit") == commit
             and document.get("source_branch") == branch,
             "training receipt source differs")
    _require(document.get("source_files") == committed_source_records(root, commit),
             "training receipt source files differ")
    _require(document.get("training_contract") == {
        "recipe_id": checked.recipe_id,
        "object_id": checked.object_id,
        "object_role": checked.object_role,
        "seed": checked.seed,
        "frozen_epochs": checked.frozen_epochs,
        "authoritative_checkpoint": "final_model.pt",
        "validation_loader": False,
    }, "training receipt contract differs")
    _require(document.get("test_boundary") == {
        "test_dataset_constructed": False,
        "test_content_open_phase_count": 0,
        "test_metric_computed": False,
    }, "training receipt does not prove an unopened test boundary")
    run_dir = Path(checked.run_directory).resolve(strict=True)
    _require(path.parent == run_dir and document.get("run_directory") == str(run_dir),
             "training receipt is outside its run directory")
    manifest_record = document.get("run_manifest")
    _require(manifest_record == {
        "repo_relative_path": checked.manifest_repo_relative_path,
        "file_sha256": checked.manifest_file_sha256,
        "content_hash_sha256": checked.manifest_content_sha256,
    }, "training receipt manifest binding differs")
    files = document.get("files")
    _require(isinstance(files, Mapping) and set(files) == {"checkpoint", "config", "history"},
             "training receipt file roles differ")
    captured: dict[str, dict[str, Any]] = {}
    for role, filename in (("checkpoint", "final_model.pt"),
                           ("config", "config.json"), ("history", "history.json")):
        expected_path = run_dir / filename
        expected = _file_record(expected_path, name=role)
        _require(files[role] == expected, f"training receipt {role} binding differs")
        captured[role] = expected
    return FixedFinalCheckpointCapability(
        binding=checked,
        checkpoint_absolute_path=captured["checkpoint"]["absolute_path"],
        checkpoint_sha256=captured["checkpoint"]["sha256"],
        checkpoint_size_bytes=captured["checkpoint"]["size_bytes"],
        config_absolute_path=captured["config"]["absolute_path"],
        config_sha256=captured["config"]["sha256"],
        config_size_bytes=captured["config"]["size_bytes"],
        history_absolute_path=captured["history"]["absolute_path"],
        history_sha256=captured["history"]["sha256"],
        history_size_bytes=captured["history"]["size_bytes"],
        receipt_absolute_path=str(path), receipt_file_sha256=hashlib.sha256(raw).hexdigest(),
        receipt_size_bytes=size, _capability=_CAPABILITY_TOKEN,
    )


def require_checkpoint_capability(
    capability: FixedFinalCheckpointCapability,
) -> FixedFinalCheckpointCapability:
    _require(isinstance(capability, FixedFinalCheckpointCapability)
             and capability._capability is _CAPABILITY_TOKEN
             and capability.binding._capability is _CAPABILITY_TOKEN,
             "fixed-final checkpoint capability is not authorized")
    return capability


__all__ = [
    "EVIDENCE_RECEIPT_NAME", "FixedFinalAuthorizationError",
    "FixedFinalCheckpointCapability", "FixedFinalRunBinding",
    "TRAINING_RECEIPT_NAME", "authorize_training_receipt",
    "bind_training_namespace", "canonical_json_bytes", "canonical_sha256",
    "require_checkpoint_capability", "require_run_binding",
    "validate_run_manifest", "write_training_receipt",
]
