"""Reviewed CPU-only runtime for the registered Task 20/55/80 replays.

The public entry point performs no training and exposes no unsafe checkpoint
fallback.  Before opening a checkpoint it verifies the reviewed execution
authority, exact environment, committed manifests, live corpus bytes,
configuration, and metadata.  The checkpoint is then hashed and deserialized
through the same no-follow file descriptor with ``weights_only=True``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from keypoint_net import representation_checkpoint_authorization as authorization
from keypoint_net import representation_checkpoint_replay as replay_registry
from keypoint_net import representation_evaluation_provenance as provenance
from keypoint_net.diagnostics import build_checkpoint_replay_manifests as manifests


RUNTIME_RESULT_SCHEMA_VERSION = "representation_checkpoint_replay_result.v1"
RESULT_DIRECTORY_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_replay/results"
)
RESULT_FILENAMES = {
    20: "TASK20_CHECKPOINT_REPLAY_RESULT_v1.json",
    55: "TASK55_CHECKPOINT_REPLAY_RESULT_v1.json",
    80: "TASK80_CHECKPOINT_REPLAY_RESULT_v1.json",
}

_EXPECTED_ENVIRONMENT = {
    "python_implementation": "CPython",
    "python_version": "3.10.19",
    "packages": {
        "numpy": "2.2.6",
        "pillow": "12.1.0",
        "torch": "2.10.0",
    },
}
_EXPECTED_TASK_CONFIG = {
    20: {
        "lambda_smooth": 0.001,
        "lambda_disp": 0.0,
        "lambda_ent": 0.01,
        "lambda_act": 0.0,
        "lambda_loc": 0.0,
        "lambda_inv": 0.0,
        "lambda_cycle": 0.5,
        "learn_inverse_operator_effective": True,
        "expected_collapse": True,
        "candidate_eligibility": "forbidden",
    },
    55: {
        "lambda_smooth": 0.0,
        "lambda_disp": 0.1,
        "lambda_ent": 0.01,
        "lambda_act": 0.0,
        "lambda_loc": 0.0,
        "lambda_inv": 0.0,
        "lambda_cycle": 0.0,
        "learn_inverse_operator_effective": False,
        "expected_collapse": False,
        "candidate_eligibility": "replay_only_not_selection",
    },
    80: {
        "lambda_smooth": 0.001,
        "lambda_disp": 0.1,
        "lambda_ent": 0.01,
        "lambda_act": 0.0,
        "lambda_loc": 0.0,
        "lambda_inv": 0.5,
        "lambda_cycle": 0.5,
        "learn_inverse_operator_effective": True,
        "expected_collapse": False,
        "candidate_eligibility": "replay_only_not_selection",
    },
}
_GEOMETRY_MANIFEST_REPO_RELATIVE_PATH = (
    "docs/decisions/2026-07-26/representation_oracle_geometry/bindings/"
    "engineers_hammer_vray__roll__world_z__v1.json"
)
_SHA256_HEX = frozenset("0123456789abcdef")
_RUNTIME_IMPORT_SHA256 = hashlib.sha256(
    Path(__file__).absolute().read_bytes()
).hexdigest()
__representation_import_sha256__ = _RUNTIME_IMPORT_SHA256


class CheckpointRuntimeError(RuntimeError):
    """Raised when the official replay runtime cannot prove its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointRuntimeError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_HEX
    )


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointRuntimeError(
            f"value cannot be encoded as strict canonical JSON: {exc}"
        ) from exc
    return encoded + (b"\n" if newline else b"")


def _strict_json_bytes(data: bytes, *, name: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointRuntimeError(
                    f"{name} has duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise CheckpointRuntimeError(
            f"{name} has non-standard JSON constant {value!r}"
        )

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointRuntimeError(f"{name} is not strict UTF-8 JSON") from exc


def _safe_git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["LC_ALL"] = "C"
    return environment


def _git(repo_root: Path, arguments: Sequence[str], *, name: str) -> bytes:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repo_root),
                *arguments,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise CheckpointRuntimeError(
            f"{name}{': ' + detail if detail else ''}"
        ) from exc
    return result.stdout


def _git_head(repo_root: Path) -> str:
    value = _git(
        repo_root,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        name="cannot resolve runtime source commit",
    ).decode("ascii").strip()
    _require(
        len(value) == 40 and set(value) <= _SHA256_HEX,
        "runtime source commit must be full lowercase 40-hex",
    )
    return value


def _read_regular_file_once(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    name: str,
) -> bytes:
    """Read one exact regular file through one no-follow descriptor."""

    _require(path.is_absolute(), f"{name} path must be absolute")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointRuntimeError(f"{name} cannot be opened: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        _require(
            metadata.st_size == expected_size_bytes,
            f"{name} size differs from its binding",
        )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
            size += len(block)
    finally:
        os.close(descriptor)
    _require(size == expected_size_bytes, f"{name} changed while being read")
    _require(
        digest.hexdigest() == expected_sha256,
        f"{name} SHA-256 differs from its binding",
    )
    return b"".join(chunks)


def _read_relative_bound_file(
    root: Path,
    record: Mapping[str, Any],
    *,
    name: str,
) -> bytes:
    relative = record.get("relative_path")
    _require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts
        and "." not in Path(relative).parts,
        f"{name} has unsafe relative_path",
    )
    root = root.resolve(strict=True)
    cursor = root
    try:
        for part in Path(relative).parts[:-1]:
            cursor = cursor / part
            metadata = cursor.lstat()
            _require(
                stat.S_ISDIR(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode),
                f"{name} path contains a non-directory or symlink",
            )
    except OSError as exc:
        raise CheckpointRuntimeError(f"{name} parent cannot be inspected") from exc
    path = root / relative
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise CheckpointRuntimeError(f"{name} path escapes dataset root") from exc
    return _read_regular_file_once(
        path,
        expected_sha256=str(record["sha256"]),
        expected_size_bytes=int(record["size_bytes"]),
        name=name,
    )


def _validate_environment_before_checkpoint() -> dict[str, Any]:
    observed = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
            "pillow": importlib.metadata.version("pillow"),
            "torch": importlib.metadata.version("torch"),
        },
    }
    _require(
        observed == _EXPECTED_ENVIRONMENT,
        "runtime environment differs from the frozen phd environment: "
        f"observed={observed}",
    )
    torch = importlib.import_module("torch")
    torch.use_deterministic_algorithms(True)
    _require(
        bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic PyTorch algorithms could not be enabled",
    )
    observed["deterministic_algorithms_enabled"] = True
    return observed


def _committed_file_record(
    repo_root: Path,
    source_commit: str,
    role: str,
    relative_path: str,
) -> dict[str, Any]:
    path = repo_root / relative_path
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CheckpointRuntimeError(
            f"committed role {role} cannot be inspected"
        ) from exc
    _require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"committed role {role} is not a regular non-symlink file",
    )
    data = path.read_bytes()
    committed = _git(
        repo_root,
        ["show", f"{source_commit}:{relative_path}"],
        name=f"cannot read committed role {role}",
    )
    _require(data == committed, f"committed role {role} differs from source commit")
    return {
        "role": role,
        "repo_relative_path": relative_path,
        "file_sha256": hashlib.sha256(data).hexdigest(),
    }


def _external_file_record(
    *,
    role: str,
    absolute_path: str,
    file_sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    _require(Path(absolute_path).is_absolute(), f"external role {role} path is not absolute")
    _require(_is_sha256(file_sha256), f"external role {role} hash is invalid")
    _require(size_bytes > 0, f"external role {role} size is invalid")
    return {
        "role": role,
        "absolute_path": absolute_path,
        "file_sha256": file_sha256,
        "size_bytes": size_bytes,
    }


def _build_evaluation_provenance(
    *,
    repo_root: Path,
    source_commit: str,
    checkpoint_absolute_path: str,
    checkpoint_sha256: str,
    checkpoint_size_bytes: int,
    config_absolute_path: str,
    config_sha256: str,
    config_size_bytes: int,
    history_absolute_path: str,
    history_sha256: str,
    history_size_bytes: int,
) -> dict[str, Any]:
    required_committed, required_external = provenance.required_role_profile(
        "checkpoint",
        fit_from_pairs=False,
    )
    _require(
        required_external
        == frozenset({"checkpoint", "checkpoint_config", "checkpoint_metadata"}),
        "checkpoint external role profile changed",
    )
    role_paths: dict[str, str] = {}
    for role in required_committed:
        fixed = provenance.CORE_ROLE_PATHS.get(role) or provenance.FIXED_ROLE_PATHS.get(role)
        if fixed is not None:
            role_paths[role] = fixed
        elif role == "geometry_manifest":
            role_paths[role] = _GEOMETRY_MANIFEST_REPO_RELATIVE_PATH
        else:
            raise CheckpointRuntimeError(
                f"checkpoint committed role {role!r} lacks a fixed path"
            )
    committed_files = [
        _committed_file_record(
            repo_root,
            source_commit,
            role,
            role_paths[role],
        )
        for role in sorted(role_paths)
    ]
    external_files = [
        _external_file_record(
            role="checkpoint",
            absolute_path=checkpoint_absolute_path,
            file_sha256=checkpoint_sha256,
            size_bytes=checkpoint_size_bytes,
        ),
        _external_file_record(
            role="checkpoint_config",
            absolute_path=config_absolute_path,
            file_sha256=config_sha256,
            size_bytes=config_size_bytes,
        ),
        _external_file_record(
            role="checkpoint_metadata",
            absolute_path=history_absolute_path,
            file_sha256=history_sha256,
            size_bytes=history_size_bytes,
        ),
    ]
    return {
        "schema_version": provenance.PROVENANCE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "committed_files": committed_files,
        "external_files": external_files,
    }


def _load_preflight_receipt(
    *,
    repo_root: Path,
    source_commit: str,
) -> tuple[dict[str, Any], str]:
    relative_path = (
        authorization.CHECKPOINT_HASH_PREFLIGHT_RESULT_REPO_RELATIVE_PATH
    )
    record = _committed_file_record(
        repo_root,
        source_commit,
        "checkpoint_hash_preflight",
        relative_path,
    )
    data = (repo_root / relative_path).read_bytes()
    document = _strict_json_bytes(data, name="checkpoint hash preflight")
    _require(isinstance(document, dict), "checkpoint hash preflight must be an object")
    return document, str(record["file_sha256"])


def _load_bound_config_and_history(
    runtime_authorization: authorization.CheckpointRuntimeAuthorization,
) -> tuple[dict[str, Any], Any]:
    config_bytes = _read_regular_file_once(
        Path(runtime_authorization.config_absolute_path),
        expected_sha256=runtime_authorization.config_sha256,
        expected_size_bytes=runtime_authorization.config_size_bytes,
        name="checkpoint config",
    )
    history_bytes = _read_regular_file_once(
        Path(runtime_authorization.history_absolute_path),
        expected_sha256=runtime_authorization.history_sha256,
        expected_size_bytes=runtime_authorization.history_size_bytes,
        name="checkpoint history",
    )
    config = _strict_json_bytes(config_bytes, name="checkpoint config")
    history = _strict_json_bytes(history_bytes, name="checkpoint history")
    _require(isinstance(config, dict), "checkpoint config must be an object")
    _require(isinstance(history, (dict, list)), "checkpoint history must be JSON object/list")
    return config, history


def _validate_external_config(
    config: Mapping[str, Any],
    *,
    task_id: int,
    runtime_authorization: authorization.CheckpointRuntimeAuthorization,
) -> dict[str, Any]:
    task = _EXPECTED_TASK_CONFIG[task_id]
    expected_common = {
        "data_root": "./_tdw_world_z_roll_base_panel_512_v2",
        "object": "engineers_hammer_vray",
        "pairs_index": (
            "_tdw_world_z_roll_base_panel_512_v2/indices/"
            "pairs_skip3_cyclic.json"
        ),
        "img_size": 512,
        "num_keypoints": 10,
        "base_channels": 32,
        "temperature": 1.0,
        "padding_mode": "reflect",
        "operator_type": "shared_affine",
        "learn_inverse_operator": False,
        "num_action_classes": 2,
        "frame_skip": 3,
        "yaw_step_deg": 2.0,
        "center_crop": None,
        "seed": 42,
    }
    for key, expected in {**expected_common, **{
        name: task[name]
        for name in (
            "lambda_smooth",
            "lambda_disp",
            "lambda_ent",
            "lambda_act",
            "lambda_loc",
            "lambda_inv",
            "lambda_cycle",
            "learn_inverse_operator_effective",
        )
    }}.items():
        _require(
            type(config.get(key)) is type(expected) and config.get(key) == expected,
            f"Task {task_id} config field {key} differs from the frozen value",
        )
    derived_inverse = bool(
        config["learn_inverse_operator"]
        or float(config["lambda_inv"]) > 0.0
        or float(config["lambda_cycle"]) > 0.0
    )
    _require(
        derived_inverse is bool(config["learn_inverse_operator_effective"])
        and derived_inverse is runtime_authorization.expected_learn_inverse_operator,
        "effective inverse-operator reconstruction differs",
    )
    _require(
        float(config["lambda_act"]) == 0.0,
        "primary replay must not instantiate an action classifier",
    )
    return {
        "num_keypoints": 10,
        "in_channels": runtime_authorization.expected_in_channels,
        "base_channels": 32,
        "temperature": 1.0,
        "num_action_classes": runtime_authorization.expected_num_action_classes,
        "padding_mode": "reflect",
        "operator_type": "shared_affine",
        "learn_inverse_operator": derived_inverse,
        "heatmap_res": runtime_authorization.expected_heatmap_res,
    }


def _validate_checked_manifests_and_load_frames(
    *,
    repo_root: Path,
    dataset_root: Path,
) -> tuple[dict[str, Mapping[str, Any]], list[Any], Any, Any]:
    """Freshly rebuild/check manifests, then retain exact selected frame bytes."""

    documents = manifests.build_checkpoint_replay_manifest_documents()
    manifests.check_checked_in_manifests(documents)
    frame_manifest = documents["frame_mask"]
    metadata_manifest = documents["metadata"]

    np = importlib.import_module("numpy")
    pil_image = importlib.import_module("PIL.Image")
    images: list[Any] = []
    masks: list[Any] = []
    for record in frame_manifest["records"]:
        frame_index = int(record["frame_index"])
        image_bytes = _read_relative_bound_file(
            dataset_root,
            record["image"],
            name=f"hammer RGB frame {frame_index}",
        )
        mask_bytes = _read_relative_bound_file(
            dataset_root,
            record["mask"],
            name=f"hammer mask frame {frame_index}",
        )
        try:
            with pil_image.open(io.BytesIO(image_bytes)) as image_handle:
                rgb = np.asarray(
                    image_handle.convert("RGB"),
                    dtype=np.uint8,
                ).copy()
            with pil_image.open(io.BytesIO(mask_bytes)) as mask_handle:
                mask = np.asarray(
                    mask_handle.convert("L"),
                    dtype=np.uint8,
                ).copy() > 0
        except Exception as exc:
            raise CheckpointRuntimeError(
                f"cannot decode registered frame {frame_index}"
            ) from exc
        _require(
            rgb.shape == (512, 512, 3) and rgb.dtype == np.uint8,
            f"registered RGB frame {frame_index} has wrong geometry/dtype",
        )
        _require(
            mask.shape == (512, 512)
            and mask.dtype == np.dtype(np.bool_)
            and bool(np.any(mask)),
            f"registered mask frame {frame_index} is invalid",
        )
        images.append(rgb)
        masks.append(mask)
    _require(len(images) == len(masks) == 180, "live hammer corpus must have 180 frames")
    stacked_masks = np.stack(masks, axis=0)
    physical_states = [
        {"theta_deg": float(record["physical_state"]["theta_deg"])}
        for record in metadata_manifest["frame_records"]
    ]
    _require(
        physical_states
        == [{"theta_deg": float(2 * frame)} for frame in range(180)],
        "metadata manifest physical states differ from 0..358 degrees",
    )
    return documents, images, stacked_masks, physical_states


def _validate_import_time_source_digests(
    *,
    repo_root: Path,
    runtime_source_manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Bind every executed repository module's loaded bytes before checkpoint."""

    role_modules = {
        "authorization_source": "keypoint_net.representation_checkpoint_authorization",
        "runtime_source": "keypoint_net.representation_checkpoint_runtime",
        "replay_registry_source": "keypoint_net.representation_checkpoint_replay",
        "historical_comparison_source": "keypoint_net.representation_replay_comparison",
        "model_source": "keypoint_net.model",
        "evaluator_source": "keypoint_net.eval_representation",
        "provenance_source": "keypoint_net.representation_evaluation_provenance",
        "array_codec_source": "keypoint_net.representation_array_codec",
        "manifest_builder_source": (
            "keypoint_net.diagnostics.build_checkpoint_replay_manifests"
        ),
        "corpus_inventory_source": "keypoint_net.representation_corpus_inventory",
        "split_validator_source": "keypoint_net.representation_splits",
    }
    records = runtime_source_manifest.get("sources")
    _require(isinstance(records, list), "runtime source manifest sources are missing")
    by_role = {
        str(record.get("role")): record
        for record in records
        if isinstance(record, Mapping)
    }
    _require(
        set(by_role) == set(role_modules),
        "runtime source manifest/module role closure differs",
    )
    verified: dict[str, dict[str, str]] = {}
    for role, module_name in role_modules.items():
        module = importlib.import_module(module_name)
        loaded_digest = getattr(
            module,
            "__representation_import_sha256__",
            None,
        )
        record = by_role[role]
        expected_digest = record.get("file_sha256")
        expected_relative = record.get("repo_relative_path")
        _require(
            _is_sha256(loaded_digest)
            and loaded_digest == expected_digest,
            f"loaded source digest differs for {role}",
        )
        module_file = Path(str(module.__file__)).resolve(strict=True)
        expected_path = (repo_root / str(expected_relative)).resolve(strict=True)
        _require(
            module_file == expected_path,
            f"loaded module path differs for {role}",
        )
        verified[role] = {
            "module": module_name,
            "absolute_path": str(module_file),
            "import_time_sha256": str(loaded_digest),
        }
    return verified


def _expected_state_keys(model: Any) -> set[str]:
    return set(model.state_dict())


def _state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_checkpoint_payload_and_reconstruct(
    payload: Any,
    *,
    architecture: Mapping[str, Any],
    external_config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    torch = importlib.import_module("torch")
    model_module = importlib.import_module("keypoint_net.model")
    _require(isinstance(payload, Mapping), "checkpoint payload must be a mapping")
    _require(
        set(payload)
        == {"epoch", "model_state_dict", "optimizer_state_dict", "loss", "config"},
        "checkpoint top-level keys differ from the frozen legacy payload",
    )
    embedded = payload["config"]
    _require(isinstance(embedded, Mapping), "checkpoint embedded config must be a mapping")
    selected_embedded_expectations = {
        "num_keypoints": architecture["num_keypoints"],
        "base_channels": architecture["base_channels"],
        "temperature": architecture["temperature"],
        "padding_mode": architecture["padding_mode"],
        "operator_type": architecture["operator_type"],
        "learn_inverse_operator": architecture["learn_inverse_operator"],
        "frame_skip": 3,
        "yaw_step_deg": 2.0,
        "lambda_smooth": external_config["lambda_smooth"],
        "lambda_disp": external_config["lambda_disp"],
        "lambda_ent": external_config["lambda_ent"],
        "lambda_act": external_config["lambda_act"],
        "lambda_loc": external_config["lambda_loc"],
        "lambda_inv": external_config["lambda_inv"],
        "lambda_cycle": external_config["lambda_cycle"],
        "num_action_classes": 0,
        "sigma": external_config["sigma"],
        "loc_bg_threshold": external_config["loc_bg_threshold"],
        "img_size": 512,
        "seed": 42,
        "data_root": external_config["data_root"],
        "object": "engineers_hammer_vray",
        "pairs_index": external_config["pairs_index"],
    }
    for key, expected in selected_embedded_expectations.items():
        _require(
            key in embedded
            and type(embedded[key]) is type(expected)
            and embedded[key] == expected,
            f"checkpoint embedded config field {key} differs",
        )
    # The exact 2026-06-02 legacy checkpoint writer (Git commit
    # 952064e28f7661907382a203920628310c8ccbbe) did not copy
    # ``center_crop`` into ``ckpt_config``.  The separately hash-bound
    # run-level config does record center_crop=null, and the runtime performs
    # no crop.  Missing and explicit null are therefore the only equivalent
    # legacy encodings accepted here; any actual crop value remains fatal.
    _require(
        "center_crop" not in embedded or embedded["center_crop"] is None,
        "checkpoint embedded config field center_crop differs",
    )
    if "heatmap_res" in embedded:
        _require(embedded["heatmap_res"] == 64, "checkpoint embeds non-legacy heatmap resolution")

    model = model_module.PhaseAModel(**dict(architecture))
    _require(model.operator.__class__.__name__ == "SharedAffineOperator", "wrong operator class")
    _require(
        (model.inverse_operator is not None)
        is bool(architecture["learn_inverse_operator"]),
        "wrong inverse-operator module presence",
    )
    _require(model.action_classifier is None, "action classifier must be absent")
    state = payload["model_state_dict"]
    _require(isinstance(state, Mapping), "model_state_dict must be a mapping")
    expected_keys = _expected_state_keys(model)
    _require(set(state) == expected_keys, "checkpoint state_dict keys differ")
    _require(
        not any(str(key).startswith("extractor.head_upsample.") for key in state),
        "legacy replay checkpoint unexpectedly has a 128-resolution head",
    )
    for key, target in model.state_dict().items():
        source = state[key]
        _require(torch.is_tensor(source), f"state tensor {key} is not a tensor")
        _require(source.device.type == "cpu", f"state tensor {key} is not on CPU")
        _require(
            source.shape == target.shape and source.dtype == target.dtype,
            f"state tensor {key} shape/dtype differs",
        )
        _require(
            bool(torch.isfinite(source).all()),
            f"state tensor {key} contains non-finite values",
        )
    model.load_state_dict(state, strict=True)
    model.to(torch.device("cpu"))
    model.requires_grad_(False)
    model.eval()
    _require(model.training is False, "model did not enter eval mode")
    _require(
        all(parameter.requires_grad is False for parameter in model.parameters()),
        "model parameters still require gradients",
    )
    return model, {
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_loss": float(payload["loss"]),
        "state_key_count": len(expected_keys),
        "embedded_config_verified": True,
    }


def _safe_load_checkpoint_same_descriptor(
    runtime_authorization: authorization.CheckpointRuntimeAuthorization,
    *,
    architecture: Mapping[str, Any],
    external_config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Hash and load the checkpoint through the same no-follow descriptor."""

    authorization.require_checkpoint_runtime_authorization(runtime_authorization)
    torch = importlib.import_module("torch")
    path = Path(runtime_authorization.checkpoint_absolute_path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointRuntimeError(f"checkpoint cannot be opened: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "checkpoint is not a regular file")
        _require(
            metadata.st_size == runtime_authorization.checkpoint_size_bytes,
            "checkpoint size differs at load time",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = hashlib.sha256()
            size = 0
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
            _require(
                size == runtime_authorization.checkpoint_size_bytes
                and digest.hexdigest() == runtime_authorization.checkpoint_sha256,
                "checkpoint bytes differ at the load boundary",
            )
            handle.seek(0)
            try:
                payload = torch.load(
                    handle,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as exc:
                raise CheckpointRuntimeError(
                    "safe weights_only checkpoint deserialization failed"
                ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    model, payload_record = _validate_checkpoint_payload_and_reconstruct(
        payload,
        architecture=architecture,
        external_config=external_config,
    )
    payload_record.update(
        {
            "checkpoint_absolute_path": str(path),
            "checkpoint_size_bytes": size,
            "checkpoint_sha256": digest.hexdigest(),
            "load_device": "cpu",
            "weights_only": True,
            "same_open_file_descriptor_hash_and_load": True,
            "unsafe_fallback_used": False,
        }
    )
    return model, payload_record


def _preprocess_rgb_batch(images: Sequence[Any]) -> Any:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    array = np.stack(images, axis=0)
    _require(array.dtype == np.uint8, "RGB batch must be uint8")
    tensor = torch.from_numpy(array.copy()).permute(0, 3, 1, 2)
    tensor = tensor.to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return tensor.sub_(mean).div_(std)


def _run_inference(
    model: Any,
    images: Sequence[Any],
    *,
    batch_size: int,
) -> tuple[Any, Any, dict[str, Any]]:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    _require(batch_size > 0, "batch_size must be positive")
    before = _state_digest(model)
    points_batches = []
    logits_batches = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = _preprocess_rgb_batch(images[start : start + batch_size])
            _require(batch.device.type == "cpu", "inference batch left CPU")
            flattened, logits = model.extractor(batch)
            points = flattened.reshape(batch.shape[0], 10, 2)
            _require(
                tuple(points.shape[1:]) == (10, 2)
                and tuple(logits.shape[1:]) == (10, 64, 64),
                "runtime extractor output shape differs",
            )
            _require(
                points.dtype == torch.float32
                and logits.dtype == torch.float32
                and bool(torch.isfinite(points).all())
                and bool(torch.isfinite(logits).all()),
                "runtime extractor output dtype/finiteness differs",
            )
            points_batches.append(points.detach().cpu().numpy().copy())
            logits_batches.append(logits.detach().cpu().numpy().copy())
    after = _state_digest(model)
    _require(before == after, "model state changed during inference")
    points = np.concatenate(points_batches, axis=0)
    logits = np.concatenate(logits_batches, axis=0)
    _require(points.shape == (180, 10, 2), "runtime points shape differs")
    _require(logits.shape == (180, 10, 64, 64), "runtime logits shape differs")
    return points, logits, {
        "frame_count": 180,
        "channel_count": 10,
        "batch_size": batch_size,
        "device": "cpu",
        "model_training_flag": bool(model.training),
        "gradients_enabled_inside_inference": False,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_unchanged": True,
    }


def _bbox_diagonals(masks: Any) -> list[float]:
    np = importlib.import_module("numpy")
    values = []
    for frame_index, mask in enumerate(masks):
        rows, columns = np.nonzero(mask)
        _require(rows.size > 0, f"mask frame {frame_index} is empty")
        width = float(columns.max() - columns.min()) / 511.0
        height = float(rows.max() - rows.min()) / 511.0
        diagonal = math.hypot(width, height)
        _require(diagonal > 0.0, f"mask frame {frame_index} bbox is degenerate")
        values.append(diagonal)
    return values


def _pair_rows() -> list[dict[str, Any]]:
    return [
        {
            "pair_id": f"full_corpus:{source}:{(source + 3) % 180}",
            "source_frame": source,
            "target_frame": (source + 3) % 180,
            "direction": "forward",
            "stride": 3,
            "partition": "full_corpus",
        }
        for source in range(180)
    ]


def _evaluation_config() -> dict[str, Any]:
    return {
        "protocol": "full_primary_roll",
        "representation_thresholds": {
            "close_distance_objdiag": 0.06,
            "persistent_fraction": 0.50,
            "recurrent_fraction": 0.10,
            "transient_longest_fraction": 0.10,
            "clustered_median_objdiag": 0.12,
        },
        "motion_reference_magnitude_image01": 0.1,
        "motion_fraction_min": 0.1,
        "on_object_rate_min": 0.75,
        "minimum_eligible_channels": 2,
        "operator_composition_horizons": [1, 2, 10],
        "full_rollout_horizons": list(range(1, 60)),
        "role_scoped_holdout_frame_ids": list(range(24)),
        "holdout_rollout_horizons": list(range(1, 8)),
        "closure_horizon": 60,
    }


def _build_bundle(
    *,
    runtime_authorization: authorization.CheckpointRuntimeAuthorization,
    evaluation_provenance: Mapping[str, Any],
    points: Any,
    logits: Any,
    masks: Any,
    physical_states: Sequence[Mapping[str, Any]],
    model: Any,
) -> tuple[dict[str, Any], Any, Any]:
    evaluator = importlib.import_module("keypoint_net.eval_representation")
    codec = importlib.import_module("keypoint_net.representation_array_codec")
    A = model.operator.A.detach().cpu().numpy().astype("float64", copy=True)
    b = model.operator.bias.detach().cpu().numpy().astype("float64", copy=True)
    bundle = {
        "schema_version": evaluator.BUNDLE_SCHEMA_VERSION,
        "case_id": runtime_authorization.fixture_id,
        "case_kind": "checkpoint",
        "provenance": copy.deepcopy(dict(evaluation_provenance)),
        "evaluation_config": _evaluation_config(),
        "estimator_metadata": {
            "input_height": 512,
            "input_width": 512,
            "heatmap_height": 64,
            "heatmap_width": 64,
            "endpoint_grid": True,
            "temperature": 1.0,
            "logit_dtype": "float32",
            "softmax_dtype": "float32",
            "crop": None,
            "resize": [512, 512],
            "align_corners": None,
        },
        "transform": {
            "family": "roll",
            "physical_axis": "world_z",
            "direction": "forward",
            "signed_generator": 6.0,
            "generator_units": "degrees",
            "stride": 3,
            "stride_units": "frames",
            "cyclic": True,
            "expected_2d_family": "planar_rotation_about_projected_center",
            "projected_centre_xy": [0.0, 0.0],
        },
        "evaluation": {
            "object_id": "engineers_hammer_vray",
            "seed": 42,
            "partition": "full_corpus",
            "frame_ids": list(range(180)),
            "points": codec.encode_float32_array(points),
            "physical_states": [dict(record) for record in physical_states],
            "bbox_diagonal_image01": _bbox_diagonals(masks),
            "visibility": codec.encode_bool_packbits_array(
                importlib.import_module("numpy").ones((180, 10), dtype=bool)
            ),
            "masks": {
                "values": codec.encode_bool_packbits_array(masks),
                "geometry": {
                    "input_height": 512,
                    "input_width": 512,
                    "crop": None,
                    "resize": [512, 512],
                    "align_corners": None,
                },
            },
            "pair_rows": _pair_rows(),
            "logits": codec.encode_float32_array(logits),
        },
        "operator": {
            "mode": "supplied",
            "A": A.tolist(),
            "b": b.tolist(),
        },
    }
    bundle["bundle_content_sha256"] = evaluator.canonical_sha256(bundle)
    return bundle, evaluator, codec


def _write_result_exclusive(path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_absolute(), "result path must be absolute")
    repo_root = Path(__file__).resolve().parents[1]
    expected_directory = (repo_root / RESULT_DIRECTORY_REPO_RELATIVE_PATH).resolve()
    path_parent = path.parent.resolve()
    _require(path_parent == expected_directory, "result path must use the frozen replay result directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(result, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise CheckpointRuntimeError(
            f"refusing to overwrite existing replay result: {path}"
        ) from exc
    try:
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
    finally:
        os.close(descriptor)
    return {
        "absolute_path": str(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "write_mode": "exclusive_create_no_overwrite",
    }


def run_authorized_checkpoint_replay(
    *,
    task_id: int,
    output_path: str | os.PathLike[str],
    source_commit: str | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run one official registered replay and save a compact result."""

    _require(task_id in (20, 55, 80), "task_id must be one of 20, 55, 80")
    repo_root = Path(__file__).resolve().parents[1]
    resolved_output_path = Path(output_path).resolve()
    expected_output_path = (
        repo_root
        / RESULT_DIRECTORY_REPO_RELATIVE_PATH
        / RESULT_FILENAMES[task_id]
    ).resolve()
    _require(
        resolved_output_path == expected_output_path,
        f"Task {task_id} result must use the frozen path {expected_output_path}",
    )
    _require(
        not resolved_output_path.exists()
        and not resolved_output_path.is_symlink(),
        f"refusing to overwrite existing replay result: {resolved_output_path}",
    )
    source_commit = source_commit or _git_head(repo_root)
    _require(source_commit == _git_head(repo_root), "source_commit must equal current HEAD")
    environment = _validate_environment_before_checkpoint()

    registry_path = repo_root / replay_registry.REGISTRY_REPO_RELATIVE_PATH
    registry = replay_registry.load_and_validate_replay_registry(str(registry_path))
    registry_provenance = replay_registry.verify_committed_registry(
        registry,
        source_commit=source_commit,
    )
    preflight, _preflight_file_sha256 = _load_preflight_receipt(
        repo_root=repo_root,
        source_commit=source_commit,
    )

    # Bind the registered external paths into provenance before authorization.
    # This remains checkpoint-blind: no checkpoint path is opened or inspected.
    candidate = next(
        item for item in registry.checkpoint_candidates if item.task_id == task_id
    )
    binding = authorization._binding_for_candidate(candidate)
    evaluation_provenance = _build_evaluation_provenance(
        repo_root=repo_root,
        source_commit=source_commit,
        checkpoint_absolute_path=candidate.absolute_path,
        checkpoint_sha256=candidate.expected_sha256,
        checkpoint_size_bytes=candidate.expected_size_bytes,
        config_absolute_path=f"{binding['run_directory']}/config.json",
        config_sha256=str(binding["config_sha256"]),
        config_size_bytes=int(binding["config_size_bytes"]),
        history_absolute_path=f"{binding['run_directory']}/history.json",
        history_sha256=str(binding["history_sha256"]),
        history_size_bytes=int(binding["history_size_bytes"]),
    )
    runtime_authorization = authorization.authorize_checkpoint_runtime(
        registry=registry,
        provenance=registry_provenance,
        checkpoint_hash_preflight=preflight,
        task_id=task_id,
        runtime_source_commit=source_commit,
        evaluation_provenance=evaluation_provenance,
    )
    external_config, _history = _load_bound_config_and_history(
        runtime_authorization
    )
    architecture = _validate_external_config(
        external_config,
        task_id=task_id,
        runtime_authorization=runtime_authorization,
    )
    dataset_root = Path(runtime_authorization.dataset_root)
    manifest_documents, images, masks, physical_states = (
        _validate_checked_manifests_and_load_frames(
            repo_root=repo_root,
            dataset_root=dataset_root,
        )
    )
    loaded_source_digests = _validate_import_time_source_digests(
        repo_root=repo_root,
        runtime_source_manifest=manifest_documents["runtime_source"],
    )

    # All source, review, config, history, environment, and live-data checks
    # above occur before this sole real checkpoint open.
    model, checkpoint_record = _safe_load_checkpoint_same_descriptor(
        runtime_authorization,
        architecture=architecture,
        external_config=external_config,
    )
    authorization._register_loaded_checkpoint_after_same_descriptor_load(
        runtime_authorization=runtime_authorization,
        checkpoint_record=checkpoint_record,
    )
    points, logits, inference_record = _run_inference(
        model,
        images,
        batch_size=batch_size,
    )
    bundle, evaluator, _codec = _build_bundle(
        runtime_authorization=runtime_authorization,
        evaluation_provenance=evaluation_provenance,
        points=points,
        logits=logits,
        masks=masks,
        physical_states=physical_states,
        model=model,
    )
    authorization._mint_checkpoint_evaluator_context_after_inference(
        bundle,
        runtime_authorization=runtime_authorization,
        inference_record=inference_record,
    )
    evaluator_result = evaluator.evaluate_bundle(bundle)

    comparison = importlib.import_module(
        "keypoint_net.representation_replay_comparison"
    ).verify_registered_historical_outputs_and_compare(
        evaluator_result=evaluator_result,
        registry_path=str(registry_path),
        task_id=task_id,
    )
    expected = _EXPECTED_TASK_CONFIG[task_id]
    collapse = evaluator_result["collapse_evidence"][
        "structural_negative_control_collapse"
    ]
    classification_passed = collapse is expected["expected_collapse"]
    historical_passed = bool(
        comparison["all_definition_identical_fields_passed"]
    )
    gate_passed = classification_passed and historical_passed
    result: dict[str, Any] = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "artifact_type": "representation_checkpoint_replay_result",
        "task_id": task_id,
        "fixture_id": runtime_authorization.fixture_id,
        "fixture_role": runtime_authorization.role,
        "source_commit": source_commit,
        "candidate_eligibility": expected["candidate_eligibility"],
        "execution_boundary": {
            "checkpoint_loaded": True,
            "training_performed": False,
            "optimizer_constructed": False,
            "weight_update_performed": False,
            "device": "cpu",
            "weights_only": True,
            "selection_use_authorized": False,
        },
        "environment": environment,
        "architecture": dict(architecture),
        "checkpoint": checkpoint_record,
        "live_inputs": {
            "manifest_content_sha256": {
                key: document["content_hash_sha256"]
                for key, document in manifest_documents.items()
            },
            "frame_count": len(images),
            "mask_count": int(masks.shape[0]),
            "pair_count": 180,
            "dataset_modified": False,
        },
        "inference": inference_record,
        "loaded_source_digests": loaded_source_digests,
        "bundle_content_sha256": bundle["bundle_content_sha256"],
        "evaluator_result": evaluator_result,
        "historical_comparison": comparison,
        "gate": {
            "expected_structural_negative_control_collapse": expected[
                "expected_collapse"
            ],
            "observed_structural_negative_control_collapse": collapse,
            "classification_passed": classification_passed,
            "historical_245_record_comparison_passed": historical_passed,
            "passed": gate_passed,
            "next_task_authorized": (
                task_id == 20 and gate_passed
            ),
        },
    }
    payload = dict(result)
    result["content_hash_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    output_record = _write_result_exclusive(resolved_output_path, result)
    return {
        "result": result,
        "output": output_record,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one reviewed CPU-only saved-checkpoint replay"
    )
    parser.add_argument("--task", type=int, choices=(20, 55, 80), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--batch-size", type=int, default=8)
    arguments = parser.parse_args(argv)
    outcome = run_authorized_checkpoint_replay(
        task_id=arguments.task,
        output_path=arguments.output.resolve(),
        source_commit=arguments.source_commit,
        batch_size=arguments.batch_size,
    )
    summary = {
        "task_id": arguments.task,
        "gate_passed": outcome["result"]["gate"]["passed"],
        "output": outcome["output"],
        "checkpoint_loaded": True,
        "training_performed": False,
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
