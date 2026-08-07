"""Prepare, run, evaluate, and decide the frozen primary 64-versus-128 matrix.

This module intentionally does not submit Slurm jobs.  It exposes only the
frozen primary task index and filesystem/provenance bindings; all scientific
training arguments continue to come from the committed experiment manifest.
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
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keypoint_net import representation_fresh_checkpoint_authorization as fresh
from keypoint_net import roll_head_package_decision as package_decision
from keypoint_net import run_fresh_roll_cell


EXPECTED_BRANCH = "agent/representation-oracles-20260726"
PRIMARY_MATRIX_SCHEMA_VERSION = "roll_head_package_primary_matrix_launch.v1"
PRIMARY_MATRIX_LOCK_NAME = "PRIMARY_MATRIX_LAUNCH_LOCK.json"
PRIMARY_SLURM_RELATIVE_PATH = Path("cluster/fresh_roll_primary_matrix.slurm")
PRIMARY_CELL_IDS = tuple(
    f"{recipe}__r{head}__seed{seed}"
    for recipe in ("task55_clean", "task80_assisted")
    for head in (64, 128)
    for seed in (42, 43, 44)
)
PRIMARY_TASK_COUNT = len(PRIMARY_CELL_IDS)
MAX_CONCURRENT_GPU_NODES = 1
GPU_COUNT_PER_TASK = 1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PrimaryMatrixError(ValueError):
    """The primary matrix launch or output boundary is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrimaryMatrixError(message)


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    return fresh.canonical_json_bytes(value, newline=newline)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_regular(path: Path, *, name: str) -> bytes:
    _require(path.is_absolute(), f"{name} path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PrimaryMatrixError(f"cannot open {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, value: Mapping[str, Any], *, name: str) -> None:
    _require(path.is_absolute(), f"{name} path must be absolute")
    data = _canonical_json_bytes(value, newline=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        raise PrimaryMatrixError(f"cannot exclusively create {name}") from exc
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
    finally:
        os.close(descriptor)


def _strict_json(data: bytes, *, name: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise PrimaryMatrixError(f"{name} contains forbidden constant {token!r}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except PrimaryMatrixError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrimaryMatrixError(f"{name} is not strict JSON") from exc
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "--no-replace-objects", *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrimaryMatrixError("cannot establish source Git provenance") from exc


def _source_state(*, expected_commit: str, require_clean: bool) -> None:
    _require(_COMMIT_RE.fullmatch(expected_commit) is not None,
             "expected commit is invalid")
    actual_commit = _git("rev-parse", "HEAD")
    actual_branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    _require(actual_commit == expected_commit, "source commit differs")
    _require(actual_branch == EXPECTED_BRANCH, "source branch differs")
    if require_clean:
        tracked = _git("status", "--porcelain", "--untracked-files=no")
        _require(not tracked, "tracked source is modified")


def primary_cell_id(task_index: int) -> str:
    _require(type(task_index) is int, "task index must be an integer")
    _require(0 <= task_index < PRIMARY_TASK_COUNT, "task index is outside 0..11")
    return PRIMARY_CELL_IDS[task_index]


def _file_record(path: Path, *, name: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    data = _read_regular(resolved, name=name)
    return {
        "absolute_path": str(resolved),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def prepare_primary_matrix(
    *,
    matrix_root: Path | str,
    expected_commit: str,
    slurm_script: Path | str,
    environment_lock: Path | str,
) -> Path:
    """Create one immutable, non-submitting launch lock and empty layout."""

    _source_state(expected_commit=expected_commit, require_clean=True)
    expected_slurm = (REPO_ROOT / PRIMARY_SLURM_RELATIVE_PATH).resolve(strict=True)
    supplied_slurm = Path(slurm_script).expanduser().resolve(strict=True)
    _require(supplied_slurm == expected_slurm, "primary Slurm script path differs")
    script_record = _file_record(supplied_slurm, name="primary Slurm script")
    environment_record = _file_record(
        Path(environment_lock), name="environment lock"
    )

    supplied_root = Path(matrix_root).expanduser()
    _require(supplied_root.name not in {"", ".", ".."},
             "matrix root name is invalid")
    parent = supplied_root.parent.resolve(strict=True)
    root = (parent / supplied_root.name).absolute()
    try:
        os.mkdir(root, 0o755)
    except OSError as exc:
        raise PrimaryMatrixError("matrix root must be new") from exc
    try:
        os.mkdir(root / "runs", 0o755)
        os.mkdir(root / "evaluations", 0o755)
    except OSError as exc:
        raise PrimaryMatrixError("cannot create exact matrix output layout") from exc

    document: dict[str, Any] = {
        "schema_version": PRIMARY_MATRIX_SCHEMA_VERSION,
        "artifact_type": "roll_head_package_primary_matrix_launch_lock",
        "matrix_root": str(root),
        "source_commit": expected_commit,
        "source_branch": EXPECTED_BRANCH,
        "slurm_script": script_record,
        "environment_lock": environment_record,
        "primary_cell_ids_by_task_index": list(PRIMARY_CELL_IDS),
        "task_count": PRIMARY_TASK_COUNT,
        "gpu_count_per_task": GPU_COUNT_PER_TASK,
        "max_concurrent_gpu_nodes": MAX_CONCURRENT_GPU_NODES,
        "output_layout": {
            "runs_directory": str(root / "runs"),
            "evaluations_directory": str(root / "evaluations"),
            "evaluation_filename_template": "{cell_id}.json",
        },
        "submission": {
            "performed_by_prepare_command": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
        },
    }
    document["content_hash_sha256"] = _canonical_sha256(document)
    lock_path = root / PRIMARY_MATRIX_LOCK_NAME
    _write_exclusive(lock_path, document, name="primary matrix launch lock")
    return lock_path


def validate_primary_matrix(
    matrix_root: Path | str,
    *,
    expected_commit: str | None = None,
    slurm_script_sha256: str | None = None,
    environment_lock_sha256: str | None = None,
) -> Mapping[str, Any]:
    supplied_root = Path(matrix_root).expanduser().absolute()
    _require(not supplied_root.is_symlink(), "matrix root must not be a symlink")
    root = supplied_root.resolve(strict=True)
    _require(root.is_dir(), "matrix root is not a directory")
    for child_name in ("runs", "evaluations"):
        child = root / child_name
        _require(child.is_dir() and not child.is_symlink(),
                 f"matrix {child_name} directory differs")
    lock_path = root / PRIMARY_MATRIX_LOCK_NAME
    document = _strict_json(
        _read_regular(lock_path, name="primary matrix launch lock"),
        name="primary matrix launch lock",
    )
    required = {
        "schema_version", "artifact_type", "matrix_root", "source_commit",
        "source_branch", "slurm_script", "environment_lock",
        "primary_cell_ids_by_task_index", "task_count", "gpu_count_per_task",
        "max_concurrent_gpu_nodes", "output_layout", "submission",
        "content_hash_sha256",
    }
    _require(set(document) == required, "primary matrix launch lock keys differ")
    claimed_hash = document.get("content_hash_sha256")
    _require(isinstance(claimed_hash, str) and _SHA256_RE.fullmatch(claimed_hash),
             "primary matrix launch lock hash is invalid")
    payload = dict(document)
    payload.pop("content_hash_sha256")
    _require(_canonical_sha256(payload) == claimed_hash,
             "primary matrix launch lock hash differs")
    _require(document.get("schema_version") == PRIMARY_MATRIX_SCHEMA_VERSION,
             "primary matrix launch schema differs")
    _require(
        document.get("artifact_type")
        == "roll_head_package_primary_matrix_launch_lock",
        "primary matrix launch artifact type differs",
    )
    _require(document.get("matrix_root") == str(root), "matrix root binding differs")
    _require(document.get("source_branch") == EXPECTED_BRANCH,
             "matrix source branch differs")
    source_commit = document.get("source_commit")
    _require(isinstance(source_commit, str) and _COMMIT_RE.fullmatch(source_commit),
             "matrix source commit is invalid")
    _require(document.get("primary_cell_ids_by_task_index") == list(PRIMARY_CELL_IDS),
             "primary cell order differs")
    _require(
        document.get("task_count") == PRIMARY_TASK_COUNT
        and document.get("gpu_count_per_task") == GPU_COUNT_PER_TASK
        and document.get("max_concurrent_gpu_nodes") == MAX_CONCURRENT_GPU_NODES,
        "primary matrix resource shape differs",
    )
    _require(
        document.get("output_layout") == {
            "runs_directory": str(root / "runs"),
            "evaluations_directory": str(root / "evaluations"),
            "evaluation_filename_template": "{cell_id}.json",
        },
        "primary matrix output layout differs",
    )
    _require(
        document.get("submission") == {
            "performed_by_prepare_command": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
        },
        "primary matrix submission boundary differs",
    )
    for record_name in ("slurm_script", "environment_lock"):
        record = document.get(record_name)
        _require(
            isinstance(record, Mapping)
            and set(record) == {"absolute_path", "sha256", "size_bytes"}
            and isinstance(record.get("absolute_path"), str)
            and isinstance(record.get("sha256"), str)
            and _SHA256_RE.fullmatch(str(record.get("sha256"))) is not None
            and type(record.get("size_bytes")) is int
            and int(record.get("size_bytes")) > 0,
            f"matrix {record_name} record differs",
        )
    if expected_commit is not None:
        _require(source_commit == expected_commit, "matrix expected commit differs")
    if slurm_script_sha256 is not None:
        _require(_SHA256_RE.fullmatch(slurm_script_sha256) is not None,
                 "supplied Slurm script hash is invalid")
        _require(document["slurm_script"]["sha256"] == slurm_script_sha256,
                 "matrix Slurm script hash differs")
    if environment_lock_sha256 is not None:
        _require(_SHA256_RE.fullmatch(environment_lock_sha256) is not None,
                 "supplied environment-lock hash is invalid")
        _require(
            document["environment_lock"]["sha256"] == environment_lock_sha256,
            "matrix environment-lock hash differs",
        )
    return document


def build_task_commands(
    *,
    task_index: int,
    data_root: Path | str,
    matrix_root: Path | str,
) -> tuple[str, list[str], list[str]]:
    cell_id = primary_cell_id(task_index)
    root = Path(matrix_root).expanduser().resolve(strict=True)
    data = Path(data_root).expanduser().resolve(strict=True)
    train_command = run_fresh_roll_cell.build_command(
        cell_id=cell_id,
        data_root=str(data),
        output_root=str(root / "runs"),
    )
    receipt = root / "runs" / cell_id / fresh.RUN_RECEIPT_NAME
    result = root / "evaluations" / f"{cell_id}.json"
    evaluation_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "evaluate",
        "--receipt", str(receipt),
        "--cell-id", cell_id,
        "--data-root", str(data),
        "--output", str(result),
    ]
    return cell_id, train_command, evaluation_command


def evaluate_cell(
    *,
    receipt: Path | str,
    cell_id: str,
    data_root: Path | str,
    output: Path | str,
    evaluator_fn: Callable[..., dict[str, Any]] | None = None,
) -> Path:
    output_path = Path(output).expanduser().absolute()
    _require(output_path.parent.is_dir(), "evaluation output parent is missing")
    _require(not output_path.exists(), "evaluation output already exists")
    if evaluator_fn is None:
        from keypoint_net import representation_fresh_checkpoint_runtime as runtime

        evaluator_fn = runtime.evaluate_authorized_fresh_run
    result = evaluator_fn(
        REPO_ROOT,
        Path(receipt).expanduser().absolute(),
        cell_id=cell_id,
        data_root=Path(data_root).expanduser().resolve(strict=True),
    )
    _require(isinstance(result, Mapping), "fresh evaluator did not return an object")
    evidence = package_decision.extract_cell_evidence(result)
    _require(evidence.cell_id == cell_id, "evaluation result cell differs")
    _write_exclusive(output_path, result, name="fresh evaluation result")
    return output_path


def run_primary_task(
    *,
    task_index: int,
    data_root: Path | str,
    matrix_root: Path | str,
    expected_commit: str,
    slurm_script_sha256: str,
    environment_lock_sha256: str,
    print_commands: bool = False,
) -> tuple[Path, Path] | None:
    root = Path(matrix_root).expanduser().resolve(strict=True)
    lock = validate_primary_matrix(
        root,
        expected_commit=expected_commit,
        slurm_script_sha256=slurm_script_sha256,
        environment_lock_sha256=environment_lock_sha256,
    )
    lock_path = root / PRIMARY_MATRIX_LOCK_NAME
    live_lock_bytes = _read_regular(lock_path, name="primary matrix launch lock")
    _require(
        _strict_json(live_lock_bytes, name="primary matrix launch lock") == lock,
        "primary matrix launch lock changed after validation",
    )
    lock_record = {
        "absolute_path": str(lock_path),
        "sha256": hashlib.sha256(live_lock_bytes).hexdigest(),
        "size_bytes": len(live_lock_bytes),
    }
    environment_record = _file_record(
        Path(str(lock["environment_lock"]["absolute_path"])),
        name="environment lock",
    )
    script_record = _file_record(
        Path(str(lock["slurm_script"]["absolute_path"])),
        name="primary Slurm script",
    )
    _require(environment_record == lock["environment_lock"],
             "live environment lock differs from matrix lock")
    _require(script_record == lock["slurm_script"],
             "live Slurm script differs from matrix lock")
    _source_state(expected_commit=expected_commit, require_clean=True)
    cell_id, train_command, evaluation_command = build_task_commands(
        task_index=task_index, data_root=data_root, matrix_root=root
    )
    cell = fresh.resolve_fresh_cell(REPO_ROOT, cell_id)
    _require(cell.source_commit == expected_commit, "cell source commit differs")
    run_directory = root / "runs" / cell_id
    result_path = root / "evaluations" / f"{cell_id}.json"
    _require(not run_directory.exists(), "primary cell run directory already exists")
    _require(not result_path.exists(), "primary cell evaluation already exists")
    if print_commands:
        print(shlex.join(train_command))
        print(shlex.join(evaluation_command))
        return None
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    _require(isinstance(slurm_job_id, str) and slurm_job_id,
             "primary task requires SLURM_JOB_ID")
    training_environment = dict(os.environ)
    training_environment.update({
        "FRESH_ROLL_PRIMARY_MATRIX_LOCK": lock_record["absolute_path"],
        "FRESH_ROLL_PRIMARY_MATRIX_LOCK_SHA256": lock_record["sha256"],
        "FRESH_ROLL_ENV_LOCK": environment_record["absolute_path"],
        "FRESH_ROLL_ENV_LOCK_SHA256": environment_record["sha256"],
        "FRESH_ROLL_SLURM_SCRIPT": script_record["absolute_path"],
        "FRESH_ROLL_SLURM_SCRIPT_SHA256": script_record["sha256"],
    })
    print(f"primary_cell_id={cell_id} task_index={task_index}", flush=True)
    subprocess.run(train_command, check=True, env=training_environment)
    receipt = run_directory / fresh.RUN_RECEIPT_NAME
    _read_regular(receipt.absolute(), name="completed-run receipt")
    subprocess.run(evaluation_command, check=True)
    _read_regular(result_path.absolute(), name="fresh evaluation result")
    return receipt, result_path


def run_primary_decision(*, matrix_root: Path | str, output: Path | str) -> int:
    root = Path(matrix_root).expanduser().resolve(strict=True)
    lock = validate_primary_matrix(root)
    source_commit = str(lock["source_commit"])
    _source_state(expected_commit=source_commit, require_clean=True)
    arguments: list[str] = []
    for cell_id in PRIMARY_CELL_IDS:
        arguments.extend([
            "--result", str(root / "evaluations" / f"{cell_id}.json")
        ])
    arguments.extend([
        "--matrix-lock", str(root / PRIMARY_MATRIX_LOCK_NAME),
        "--expected-commit", source_commit,
        "--output", str(Path(output).expanduser().absolute()),
    ])
    return package_decision.main(arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="create a non-submitting matrix lock")
    prepare.add_argument("--matrix-root", type=Path, required=True)
    prepare.add_argument("--expected-commit", required=True)
    prepare.add_argument("--slurm-script", type=Path, required=True)
    prepare.add_argument("--environment-lock", type=Path, required=True)

    task = commands.add_parser("task", help="run one exact primary array task")
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--data-root", type=Path, required=True)
    task.add_argument("--matrix-root", type=Path, required=True)
    task.add_argument("--expected-commit", required=True)
    task.add_argument("--slurm-script-sha256", required=True)
    task.add_argument("--environment-lock-sha256", required=True)
    task.add_argument("--print-commands", action="store_true")

    evaluate = commands.add_parser("evaluate", help="evaluate one receipted cell")
    evaluate.add_argument("--receipt", type=Path, required=True)
    evaluate.add_argument("--cell-id", required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    decide = commands.add_parser("decide", help="apply the primary package rule")
    decide.add_argument("--matrix-root", type=Path, required=True)
    decide.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        lock = prepare_primary_matrix(
            matrix_root=args.matrix_root,
            expected_commit=args.expected_commit,
            slurm_script=args.slurm_script,
            environment_lock=args.environment_lock,
        )
        print(lock)
        return 0
    if args.command == "task":
        run_primary_task(
            task_index=args.task_index,
            data_root=args.data_root,
            matrix_root=args.matrix_root,
            expected_commit=args.expected_commit,
            slurm_script_sha256=args.slurm_script_sha256,
            environment_lock_sha256=args.environment_lock_sha256,
            print_commands=args.print_commands,
        )
        return 0
    if args.command == "evaluate":
        evaluate_cell(
            receipt=args.receipt,
            cell_id=args.cell_id,
            data_root=args.data_root,
            output=args.output,
        )
        return 0
    if args.command == "decide":
        return run_primary_decision(matrix_root=args.matrix_root, output=args.output)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
