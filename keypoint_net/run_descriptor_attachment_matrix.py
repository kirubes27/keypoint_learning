"""Prepare and execute the paired descriptor control/attachment GPU matrix.

One Slurm array task owns one recipe/seed pair and runs control followed by
attachment on the same assigned generic GPU.  Preparation and task execution
are non-overwriting.  This module never submits a job itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from keypoint_net import descriptor_attachment as attachment
from keypoint_net import descriptor_attachment_authorization as authorization
from keypoint_net import representation_fresh_checkpoint_authorization as fresh


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = authorization.EXPECTED_BRANCH
SCHEMA_VERSION = "descriptor_attachment_paired_matrix_launch.v1"
LOCK_NAME = "DESCRIPTOR_PAIRED_MATRIX_LAUNCH_LOCK.json"
SLURM_RELATIVE_PATH = Path("cluster/descriptor_attachment_paired_matrix.slurm")
PAIR_KEYS = tuple(
    (recipe, seed)
    for recipe in ("task55_clean", "task80_assisted")
    for seed in (42, 43, 44)
)
TASK_COUNT = len(PAIR_KEYS)
MAX_CONCURRENT_GPU_NODES = 6
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class DescriptorMatrixError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DescriptorMatrixError(message)


def _read_regular(path: Path, *, name: str) -> bytes:
    _require(path.is_absolute(), f"{name} path must be absolute")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise DescriptorMatrixError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        data = b"".join(chunks)
        _require(len(data) == metadata.st_size, f"{name} size changed while reading")
        return data
    finally:
        os.close(descriptor)


def _strict_json(data: bytes, *, name: str) -> Mapping[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in values:
            _require(key not in result, f"{name} duplicates key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise DescriptorMatrixError(f"{name} contains {token}")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=reject_constant)
    except DescriptorMatrixError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DescriptorMatrixError(f"{name} is not strict JSON") from exc
    _require(isinstance(value, Mapping), f"{name} is not an object")
    return value


def _file_record(path: Path, *, name: str) -> dict[str, Any]:
    absolute = path.expanduser().resolve(strict=True)
    data = _read_regular(absolute, name=name)
    return {"absolute_path": str(absolute), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _write_exclusive(path: Path, value: Mapping[str, Any], *, name: str) -> None:
    data = attachment.canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise DescriptorMatrixError(f"cannot create new {name}") from exc
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
    finally:
        os.close(descriptor)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _source_state(expected_commit: str) -> None:
    _require(_COMMIT_RE.fullmatch(expected_commit) is not None, "expected commit is invalid")
    _require(_git("rev-parse", "HEAD") == expected_commit, "source commit differs")
    _require(_git("branch", "--show-current") == EXPECTED_BRANCH, "source branch differs")
    _require(not _git("status", "--porcelain", "--untracked-files=no"),
             "tracked source is modified")


def _validate_content_hash(document: Mapping[str, Any], *, name: str) -> None:
    claimed = document.get("content_hash_sha256")
    _require(isinstance(claimed, str) and _SHA256_RE.fullmatch(claimed) is not None,
             f"{name} content hash is invalid")
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(attachment.canonical_sha256(payload) == claimed,
             f"{name} content hash differs")


def _cell_ids(recipe: str, seed: int) -> tuple[str, str]:
    return (
        f"{recipe}__control__seed{seed}",
        f"{recipe}__attachment__seed{seed}",
    )


def _manifest_cells(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = manifest.get("cells")
    _require(isinstance(cells, list) and len(cells) == 12,
             "descriptor manifest does not contain 12 cells")
    result: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        _require(isinstance(cell, Mapping), "descriptor manifest cell is invalid")
        cell_id = cell.get("cell_id")
        _require(isinstance(cell_id, str) and cell_id not in result,
                 "descriptor manifest cell IDs are invalid")
        result[cell_id] = cell
    expected = {cell_id for recipe, seed in PAIR_KEYS for cell_id in _cell_ids(recipe, seed)}
    _require(set(result) == expected, "descriptor manifest cell set differs")
    return result


def prepare_matrix(
    *, matrix_root: Path, expected_commit: str, experiment_manifest: Path,
    weight_receipt: Path, gpu_smoke_receipt: Path, slurm_script: Path,
    environment_lock: Path,
) -> Path:
    _source_state(expected_commit)
    script_path = slurm_script.resolve(strict=True)
    _require(script_path == (REPO_ROOT / SLURM_RELATIVE_PATH).resolve(strict=True),
             "paired matrix Slurm script path differs")
    manifest_record = _file_record(experiment_manifest, name="experiment manifest")
    receipt_record = _file_record(weight_receipt, name="weight receipt")
    smoke_record = _file_record(gpu_smoke_receipt, name="GPU smoke receipt")
    script_record = _file_record(script_path, name="paired matrix Slurm script")
    environment_record = _file_record(environment_lock, name="environment lock")
    manifest = _strict_json(_read_regular(Path(manifest_record["absolute_path"]),
                                         name="experiment manifest"),
                            name="experiment manifest")
    _validate_content_hash(manifest, name="experiment manifest")
    _require(manifest.get("source_commit") == expected_commit
             and manifest.get("source_branch") == EXPECTED_BRANCH,
             "experiment manifest source differs")
    cells = _manifest_cells(manifest)
    _require(manifest.get("weight_receipt", {}).get("absolute_path")
             == receipt_record["absolute_path"]
             and manifest.get("weight_receipt", {}).get("file_sha256")
             == receipt_record["sha256"],
             "matrix weight receipt differs from manifest")
    smoke = _strict_json(_read_regular(Path(smoke_record["absolute_path"]),
                                      name="GPU smoke receipt"),
                         name="GPU smoke receipt")
    _validate_content_hash(smoke, name="GPU smoke receipt")
    _require(smoke.get("schema_version") == "descriptor_attachment_gpu_smoke.v1"
             and smoke.get("status") == "pass"
             and smoke.get("scientific_result_authorized") is False
             and smoke.get("source_commit") == expected_commit,
             "GPU smoke did not authorize matrix execution")
    _require(smoke.get("experiment_manifest", {}).get("file_sha256")
             == manifest_record["sha256"]
             and smoke.get("weight_receipt", {}).get("file_sha256")
             == receipt_record["sha256"],
             "GPU smoke input binding differs")

    supplied_root = matrix_root.expanduser()
    parent = supplied_root.parent.resolve(strict=True)
    root = (parent / supplied_root.name).absolute()
    expected_runs = str(root / "runs")
    _require(all(cell["training_arguments"]["output_dir"] == expected_runs
                 for cell in cells.values()),
             "descriptor manifest output root differs from paired matrix root")
    try:
        os.mkdir(root, 0o755)
        os.mkdir(root / "runs", 0o755)
        os.mkdir(root / "evaluations", 0o755)
    except OSError as exc:
        raise DescriptorMatrixError("paired matrix root/layout must be new") from exc

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "descriptor_attachment_paired_matrix_launch_lock",
        "matrix_root": str(root),
        "source_commit": expected_commit,
        "source_branch": EXPECTED_BRANCH,
        "experiment_manifest": manifest_record,
        "weight_receipt": receipt_record,
        "gpu_smoke_receipt": smoke_record,
        "slurm_script": script_record,
        "environment_lock": environment_record,
        "pair_cell_ids_by_task_index": [list(_cell_ids(recipe, seed))
                                        for recipe, seed in PAIR_KEYS],
        "task_count": TASK_COUNT,
        "runs_per_task": 2,
        "gpu_count_per_task": 1,
        "max_concurrent_gpu_nodes": MAX_CONCURRENT_GPU_NODES,
        "resource_policy": {
            "generic_gpu_request": True,
            "gpu_model_constraint": None,
            "same_assigned_gpu_within_control_attachment_pair": True,
            "execution_order_within_pair": ["control", "attachment"],
        },
        "output_layout": {
            "runs_directory": str(root / "runs"),
            "evaluations_directory": str(root / "evaluations"),
            "evaluation_filename_template": "{cell_id}.json",
        },
        "submission": {"performed_by_prepare_command": False,
                       "resume_authorized": False, "overwrite_authorized": False},
    }
    document["content_hash_sha256"] = attachment.canonical_sha256(document)
    lock_path = root / LOCK_NAME
    _write_exclusive(lock_path, document, name="paired matrix launch lock")
    return lock_path


def validate_matrix(
    matrix_root: Path, *, expected_commit: str,
    slurm_script_sha256: str, environment_lock_sha256: str,
) -> Mapping[str, Any]:
    root = matrix_root.resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "matrix root is invalid")
    for child in (root / "runs", root / "evaluations"):
        _require(child.is_dir() and not child.is_symlink(), "matrix layout differs")
    lock = _strict_json(_read_regular(root / LOCK_NAME, name="paired matrix lock"),
                        name="paired matrix lock")
    _validate_content_hash(lock, name="paired matrix lock")
    required = {
        "schema_version", "artifact_type", "matrix_root", "source_commit",
        "source_branch", "experiment_manifest", "weight_receipt",
        "gpu_smoke_receipt", "slurm_script", "environment_lock",
        "pair_cell_ids_by_task_index", "task_count", "runs_per_task",
        "gpu_count_per_task", "max_concurrent_gpu_nodes", "resource_policy",
        "output_layout", "submission", "content_hash_sha256",
    }
    _require(set(lock) == required, "paired matrix lock fields differ")
    _require(lock.get("schema_version") == SCHEMA_VERSION
             and lock.get("artifact_type")
             == "descriptor_attachment_paired_matrix_launch_lock",
             "paired matrix lock identity differs")
    _require(lock.get("matrix_root") == str(root)
             and lock.get("source_commit") == expected_commit
             and lock.get("source_branch") == EXPECTED_BRANCH,
             "paired matrix lock source/root differs")
    _require(lock.get("pair_cell_ids_by_task_index")
             == [list(_cell_ids(recipe, seed)) for recipe, seed in PAIR_KEYS]
             and lock.get("task_count") == 6 and lock.get("runs_per_task") == 2
             and lock.get("gpu_count_per_task") == 1
             and lock.get("max_concurrent_gpu_nodes") == 6,
             "paired matrix task/resource shape differs")
    _require(lock.get("resource_policy") == {
        "generic_gpu_request": True, "gpu_model_constraint": None,
        "same_assigned_gpu_within_control_attachment_pair": True,
        "execution_order_within_pair": ["control", "attachment"],
    }, "paired matrix resource policy differs")
    _require(lock.get("submission") == {"performed_by_prepare_command": False,
                                        "resume_authorized": False,
                                        "overwrite_authorized": False},
             "paired matrix submission boundary differs")
    for role in ("experiment_manifest", "weight_receipt", "gpu_smoke_receipt",
                 "slurm_script", "environment_lock"):
        record = lock.get(role)
        _require(isinstance(record, Mapping)
                 and _file_record(Path(record["absolute_path"]), name=role) == record,
                 f"paired matrix {role} live binding differs")
    _require(lock["slurm_script"]["sha256"] == slurm_script_sha256
             and lock["environment_lock"]["sha256"] == environment_lock_sha256,
             "paired matrix submitted script/environment hash differs")
    return lock


def _train_command(
    cell: Mapping[str, Any], *, manifest_record: Mapping[str, Any],
    receipt_record: Mapping[str, Any],
) -> list[str]:
    values = dict(cell["training_arguments"])
    values.update({
        "descriptor_cell_id": cell["cell_id"],
        "descriptor_experiment_manifest": manifest_record["absolute_path"],
        "descriptor_experiment_manifest_sha256": manifest_record["sha256"],
        "descriptor_weight_receipt": receipt_record["absolute_path"],
        "descriptor_weight_receipt_sha256": receipt_record["sha256"],
        "descriptor_loss_spec_sha256": attachment.LOSS_SPEC_SHA256,
    })
    return [sys.executable, str((REPO_ROOT / "keypoint_net/train.py").resolve()),
            *fresh.command_arguments(values)]


def run_task(
    *, task_index: int, matrix_root: Path, expected_commit: str,
    slurm_script_sha256: str, environment_lock_sha256: str,
    print_commands: bool = False,
) -> None:
    _require(type(task_index) is int and 0 <= task_index < TASK_COUNT,
             "paired matrix task index is outside 0..5")
    lock = validate_matrix(
        matrix_root, expected_commit=expected_commit,
        slurm_script_sha256=slurm_script_sha256,
        environment_lock_sha256=environment_lock_sha256,
    )
    _source_state(expected_commit)
    manifest = _strict_json(
        _read_regular(Path(lock["experiment_manifest"]["absolute_path"]),
                      name="experiment manifest"), name="experiment manifest",
    )
    cells = _manifest_cells(manifest)
    pair = lock["pair_cell_ids_by_task_index"][task_index]
    for cell_id in pair:
        run_directory = matrix_root.resolve() / "runs" / cell_id
        result_path = matrix_root.resolve() / "evaluations" / f"{cell_id}.json"
        _require(not run_directory.exists(), f"descriptor run already exists: {cell_id}")
        _require(not result_path.exists(), f"descriptor evaluation already exists: {cell_id}")
        command = _train_command(
            cells[cell_id], manifest_record=lock["experiment_manifest"],
            receipt_record=lock["weight_receipt"],
        )
        if print_commands:
            print(shlex.join(command))
            continue
        _require(isinstance(os.environ.get("SLURM_JOB_ID"), str),
                 "paired matrix task requires SLURM_JOB_ID")
        environment = dict(os.environ)
        environment.update({
            "FRESH_ROLL_PRIMARY_MATRIX_LOCK": str(matrix_root.resolve() / LOCK_NAME),
            "FRESH_ROLL_PRIMARY_MATRIX_LOCK_SHA256": _file_record(
                matrix_root.resolve() / LOCK_NAME, name="paired matrix lock"
            )["sha256"],
            "FRESH_ROLL_ENV_LOCK": lock["environment_lock"]["absolute_path"],
            "FRESH_ROLL_ENV_LOCK_SHA256": lock["environment_lock"]["sha256"],
            "FRESH_ROLL_SLURM_SCRIPT": lock["slurm_script"]["absolute_path"],
            "FRESH_ROLL_SLURM_SCRIPT_SHA256": lock["slurm_script"]["sha256"],
        })
        print(f"descriptor_cell_id={cell_id} pair_task_index={task_index}", flush=True)
        subprocess.run(command, check=True, env=environment)
        receipt = run_directory / authorization.COMPLETED_RUN_RECEIPT_NAME
        authorization.authorize_completed_run(
            REPO_ROOT, receipt, expected_cell_id=cell_id
        )
        from keypoint_net import descriptor_attachment_runtime as runtime
        result = runtime.evaluate_authorized_descriptor_run(
            REPO_ROOT, receipt, cell_id=cell_id,
            data_root=Path(cells[cell_id]["training_arguments"]["data_root"]),
        )
        _write_exclusive(result_path, result, name=f"descriptor evaluation {cell_id}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--matrix-root", type=Path, required=True)
    prepare.add_argument("--expected-commit", required=True)
    prepare.add_argument("--experiment-manifest", type=Path, required=True)
    prepare.add_argument("--weight-receipt", type=Path, required=True)
    prepare.add_argument("--gpu-smoke-receipt", type=Path, required=True)
    prepare.add_argument("--slurm-script", type=Path, required=True)
    prepare.add_argument("--environment-lock", type=Path, required=True)
    task = commands.add_parser("task")
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--matrix-root", type=Path, required=True)
    task.add_argument("--expected-commit", required=True)
    task.add_argument("--slurm-script-sha256", required=True)
    task.add_argument("--environment-lock-sha256", required=True)
    task.add_argument("--print-commands", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_matrix(
            matrix_root=args.matrix_root, expected_commit=args.expected_commit,
            experiment_manifest=args.experiment_manifest,
            weight_receipt=args.weight_receipt,
            gpu_smoke_receipt=args.gpu_smoke_receipt,
            slurm_script=args.slurm_script, environment_lock=args.environment_lock,
        )
        print(path)
        return 0
    run_task(
        task_index=args.task_index, matrix_root=args.matrix_root,
        expected_commit=args.expected_commit,
        slurm_script_sha256=args.slurm_script_sha256,
        environment_lock_sha256=args.environment_lock_sha256,
        print_commands=args.print_commands,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
