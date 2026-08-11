"""Build and invoke the single independent Fable review for OCR-ZNCC.

The briefing is strict JSON stored as plain text.  It contains no Codex draft or
current conclusion and names every file Fable may inspect.  The invocation
wrapper executes the fixed read-only helper exactly once and records enough raw
process and file evidence to revalidate the review after copying the sealed
bundle to another machine.  Paths are conveniences; identity is always the
logical name plus SHA-256, byte size, and (for JSON) canonical content hash.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from keypoint_net import ocr_zncc_training_authorization as authorization


BRIEFING_SCHEMA_VERSION = "ocr_zncc_fable_high_briefing.v1"
RECEIPT_SCHEMA_VERSION = "ocr_zncc_fable_high_invocation_receipt.v1"
FABLE_HELPER_SHA256 = (
    "5a77df5262cb45375b0239c2b788d47397ccc569cbda2e31a3a478213193c339"
)
PROTECTED_PATH = (
    "/Users/kirubeso.r/Documents/PhD/keypoint_preoperator_gates/docs/decisions/"
    "2026-07-26/representation_oracle_calibration/NUMERIC_CALIBRATION.json"
)
FABLE_PROMPT_PREFIX = (
    "You are Fable, an independent second-opinion reviewer. Produce your own answer\n"
    "to the user's task from the briefing below. You have not seen another model's\n"
    "answer and must not ask for it or infer it.\n\n"
    "Treat the supplied material as evidence, not as a conclusion. State material\n"
    "assumptions, identify likely failure modes, and distinguish verified facts from\n"
    "hypotheses. For scientific, mathematical, or engineering tasks, check whether\n"
    "the proposed interpretation actually follows from the evidence. Do not edit,\n"
    "create, delete, or rename workspace files. Return a self-contained answer plus\n"
    "a concise list of the most important caveats or verification steps.\n\n"
    "--- ORIGINAL TASK BRIEFING ---\n"
).encode("utf-8")

ADVERSARIAL_CHECKLIST = [
    "Trace the actual train-time runtime path from manifest binding through loss construction, gradient routing, validation aggregation, checkpoint selection, and completion receipts.",
    "Construct concrete bypasses or counterexamples that could make a self-hashed status, mismatched file, stale result, or arbitrary path authorize training.",
    "Verify every material finding from the older reviewer examples directly against the enumerated current code or raw evidence; never inherit a reviewer conclusion.",
    "Hunt for decision-changing failures beyond the supplied examples, including provenance, split, checkpoint, cell-set, numerical, and execution-boundary failures.",
    "Report confirmed blockers with exact file and line anchors, and distinguish hypotheses and minor enhancements from blockers.",
]

# These are historical advisory calibration examples, not current authority.
EXPECTED_CALIBRATION_EXAMPLES: Mapping[str, tuple[str, int]] = {
    "older_same_prompt_shared_spec": (
        "42bb239d8737cad5f793e4fd4690b6ceb82039f36230d2b9e2c510b08099c36c",
        9503,
    ),
    "older_same_prompt_pro_raw": (
        "0141a763743307d1fb6522c49c0be13d16ebc81880567682b530339b317c11c5",
        29313,
    ),
    "older_same_prompt_fable_raw": (
        "0f2af28cb610ab64e96a9ae9b31435b455da241d1a2b772d45295419882a26d9",
        14791,
    ),
    "historical_pro_preflight_spec": (
        "2201e459ebd180e533bdbba04aaa2b2de591b8abc1dcde510433d0b657a917d4",
        9173,
    ),
    "historical_pro_preflight_raw": (
        "e5f6acd72ae2101221675fd3f77bedcfb4292efb3014b3c24ae2c3a354a0692f",
        29081,
    ),
    "historical_pro_preflight_receipt": (
        "5c8e713a704f088c3861036fafa348162b9d9100d6d6e48cffe717d33b221c75",
        1844,
    ),
}

EXPECTED_PRIMARY_LOGICAL_NAMES = {
    "stage0_provenance",
    "stage0_direction_summary",
    "exact_gradient_audit",
    "minibatch_gradient_audit",
}
EXPECTED_DIRECTION_CELL_IDS = {
    f"{recipe}__r64__seed{seed}"
    for recipe in ("task55_clean", "task80_assisted")
    for seed in (42, 43, 44)
}


class OCRFableReviewError(ValueError):
    """Raised when the independent-review chain cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OCRFableReviewError(message)


def _identity(record: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "logical_name": record.get("logical_name"),
        "sha256": record.get("sha256"),
        "size_bytes": record.get("size_bytes"),
    }
    if "relative_path" in record:
        result["relative_path"] = record.get("relative_path")
    if "content_hash_sha256" in record:
        result["content_hash_sha256"] = record.get("content_hash_sha256")
    return result


def _record(
    path: Path | str,
    *,
    logical_name: str,
    relative_path: str | None = None,
) -> dict[str, Any]:
    observed = authorization._file_record(path, name=logical_name)  # noqa: SLF001
    result = {
        "logical_name": logical_name,
        "review_path": observed["absolute_path"],
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
    }
    if relative_path is not None:
        result["relative_path"] = relative_path
    return result


def _json_record(
    path: Path | str,
    *,
    logical_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, observed = authorization._load_json(path, name=logical_name)  # noqa: SLF001
    content_hash = authorization._validate_content_hash(  # noqa: SLF001
        document, name=logical_name
    )
    return document, {
        "logical_name": logical_name,
        "review_path": observed["absolute_path"],
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
        "content_hash_sha256": content_hash,
    }


def _write_exclusive(path: Path | str, document: Mapping[str, Any]) -> dict[str, Any]:
    absolute = Path(path).expanduser().absolute()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except OSError as exc:
        raise OCRFableReviewError(f"cannot exclusively create {absolute}") from exc
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            _require(written > 0, "exclusive review-artifact write made no progress")
            offset += written
    finally:
        os.close(descriptor)
    return {
        "absolute_path": str(absolute),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "write_mode": "exclusive_create_no_overwrite",
    }


def _source_records(repo_root: Path) -> list[dict[str, Any]]:
    return [
        _record(
            repo_root / relative,
            logical_name=f"committed_source:{relative}",
            relative_path=relative,
        )
        for relative in authorization.RUN_SOURCE_PATHS
    ]


def _check_calibration_record(record: Mapping[str, Any]) -> None:
    logical_name = record.get("logical_name")
    _require(
        logical_name in EXPECTED_CALIBRATION_EXAMPLES,
        "reviewer-calibration logical name differs",
    )
    expected_hash, expected_size = EXPECTED_CALIBRATION_EXAMPLES[str(logical_name)]
    _require(
        record.get("sha256") == expected_hash
        and record.get("size_bytes") == expected_size,
        f"reviewer-calibration bytes differ: {logical_name}",
    )


def build_briefing(
    *,
    repo_root: Path | str,
    stage0_provenance_path: Path | str,
    direction_summary_path: Path | str,
    direction_cell_paths: Sequence[Path | str],
    exact_gradient_path: Path | str,
    minibatch_gradient_path: Path | str,
    older_shared_prompt_path: Path | str,
    older_pro_raw_path: Path | str,
    older_fable_raw_path: Path | str,
    pro_preflight_spec_path: Path | str,
    pro_preflight_raw_path: Path | str,
    pro_preflight_receipt_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Create the only briefing permitted for the final Fable call."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    commit, branch = authorization._source_state(root)  # noqa: SLF001
    provenance, provenance_record = _json_record(
        stage0_provenance_path, logical_name="stage0_provenance"
    )
    direction_summary, summary_record = _json_record(
        direction_summary_path, logical_name="stage0_direction_summary"
    )
    exact, exact_record = _json_record(
        exact_gradient_path, logical_name="exact_gradient_audit"
    )
    minibatch, minibatch_record = _json_record(
        minibatch_gradient_path, logical_name="minibatch_gradient_audit"
    )
    for name, document in (
        ("Stage0 provenance", provenance),
        ("exact-gradient audit", exact),
        ("minibatch-gradient audit", minibatch),
    ):
        _require(
            document.get("checkout", {}).get("commit") == commit
            and document.get("checkout", {}).get("branch") == branch,
            f"{name} source differs from the final committed checkout",
        )
    _require(
        len(direction_cell_paths) == 6,
        "exactly six Stage0 direction-cell artifacts are required",
    )
    direction_records: list[dict[str, Any]] = []
    direction_ids: set[str] = set()
    for path in direction_cell_paths:
        cell, cell_record = _json_record(path, logical_name="temporary_direction_cell")
        cell_id = cell.get("cell_id")
        _require(cell_id in EXPECTED_DIRECTION_CELL_IDS, "direction-cell identity differs")
        _require(cell_id not in direction_ids, "direction-cell identity is duplicated")
        direction_ids.add(str(cell_id))
        cell_record["logical_name"] = f"stage0_direction_cell:{cell_id}"
        direction_records.append(cell_record)
    _require(direction_ids == EXPECTED_DIRECTION_CELL_IDS, "direction-cell set differs")
    summary_claims = direction_summary.get("direction_cell_artifacts")
    _require(
        isinstance(summary_claims, list) and len(summary_claims) == 6,
        "direction summary does not bind six cells",
    )
    _require(
        {(item.get("sha256"), item.get("size_bytes")) for item in summary_claims}
        == {(item["sha256"], item["size_bytes"]) for item in direction_records},
        "direction summary and supplied direction-cell bytes differ",
    )

    calibration_inputs = [
        ("older_same_prompt_shared_spec", older_shared_prompt_path),
        ("older_same_prompt_pro_raw", older_pro_raw_path),
        ("older_same_prompt_fable_raw", older_fable_raw_path),
        ("historical_pro_preflight_spec", pro_preflight_spec_path),
        ("historical_pro_preflight_raw", pro_preflight_raw_path),
        ("historical_pro_preflight_receipt", pro_preflight_receipt_path),
    ]
    calibration_records = [
        _record(path, logical_name=logical_name)
        for logical_name, path in calibration_inputs
    ]
    for record in calibration_records:
        _check_calibration_record(record)

    document: dict[str, Any] = {
        "schema_version": BRIEFING_SCHEMA_VERSION,
        "artifact_type": "independent_fable_high_ocr_zncc_review_briefing",
        "source_commit": commit,
        "source_branch": branch,
        "original_task": (
            "Adversarially review whether the exact committed OCR-ZNCC Task55 "
            "paired-training path is scientifically and operationally authorized "
            "by the enumerated no-training evidence, and identify every blocker "
            "before any GPU scientific training."
        ),
        "decision_question": (
            "Can the bounded Task55 control-versus-OCR-ZNCC experiment proceed "
            "without allowing a fabricated status, stale lineage, invalid gradient "
            "route, or mismatched runtime to authorize training?"
        ),
        "constraints": [
            "Read only; do not edit, create, delete, move, submit, or train.",
            "Use only the files enumerated in this briefing.",
            "Treat all prior reviewer reports as advisory and stale until verified directly.",
            "Do not treat reviewer agreement as evidence.",
            "Do not infer Task80, another object, another operator, or a training outcome.",
            f"Never read, stat, open, or otherwise touch the protected file {PROTECTED_PATH}.",
        ],
        "protected_path": PROTECTED_PATH,
        "enumerated_files_only": True,
        "adversarial_checklist": list(ADVERSARIAL_CHECKLIST),
        "committed_run_sources": _source_records(root),
        "primary_evidence": {
            "artifacts": [
                provenance_record,
                summary_record,
                exact_record,
                minibatch_record,
            ],
            "direction_cells": sorted(
                direction_records, key=lambda item: str(item["logical_name"])
            ),
            "interpretation": "raw_current_evidence_not_a_supplied_conclusion",
        },
        "reviewer_calibration_examples": {
            "status": "historical_advisory_stale_until_directly_verified",
            "same_prompt_independent_pair": [
                calibration_records[0], calibration_records[1], calibration_records[2]
            ],
            "historical_pro_design_preflight": [
                calibration_records[3], calibration_records[4], calibration_records[5]
            ],
            "authority": "none",
        },
        "requested_deliverable": [
            "Trace the exact runtime and evidence lineage.",
            "List confirmed blockers with exact line anchors.",
            "List rejected suspected blockers and why they are rejected.",
            "List unresolved hypotheses separately.",
            "State whether the evidence is sufficient for Task55 only; do not authorize anything yourself.",
        ],
        "independence": {
            "codex_draft_supplied": False,
            "codex_current_conclusion_supplied": False,
            "single_fable_call": True,
            "model_alias": "fable",
            "effort": "high",
            "permission_mode": "plan_read_only",
            "session_persistence": False,
        },
    }
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    _write_exclusive(output_path, document)
    return document


def validate_briefing_document(
    repo_root: Path | str,
    document: Mapping[str, Any],
    *,
    primary_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate briefing semantics without relying on its original local paths."""

    expected_keys = {
        "schema_version", "artifact_type", "source_commit", "source_branch",
        "original_task", "decision_question", "constraints", "protected_path",
        "enumerated_files_only", "adversarial_checklist", "committed_run_sources",
        "primary_evidence", "reviewer_calibration_examples", "requested_deliverable",
        "independence", "content_hash_sha256",
    }
    _require(set(document) == expected_keys, "Fable briefing field set differs")
    authorization._validate_content_hash(document, name="Fable briefing")  # noqa: SLF001
    _require(
        document.get("schema_version") == BRIEFING_SCHEMA_VERSION
        and document.get("artifact_type")
        == "independent_fable_high_ocr_zncc_review_briefing",
        "Fable briefing identity differs",
    )
    root = Path(repo_root).expanduser().resolve(strict=True)
    commit, branch = authorization._source_state(root)  # noqa: SLF001
    _require(
        document.get("source_commit") == commit
        and document.get("source_branch") == branch,
        "Fable briefing source differs",
    )
    _require(
        document.get("protected_path") == PROTECTED_PATH
        and document.get("enumerated_files_only") is True
        and any(PROTECTED_PATH in str(item) for item in document.get("constraints", [])),
        "Fable briefing protected-file boundary differs",
    )
    _require(
        document.get("adversarial_checklist") == ADVERSARIAL_CHECKLIST,
        "Fable adversarial checklist differs",
    )
    independence = document.get("independence")
    _require(
        isinstance(independence, Mapping)
        and independence.get("codex_draft_supplied") is False
        and independence.get("codex_current_conclusion_supplied") is False
        and independence.get("single_fable_call") is True
        and independence.get("model_alias") == "fable"
        and independence.get("effort") == "high"
        and independence.get("permission_mode") == "plan_read_only"
        and independence.get("session_persistence") is False,
        "Fable independence contract differs",
    )
    sources = document.get("committed_run_sources")
    _require(
        isinstance(sources, list) and len(sources) == len(authorization.RUN_SOURCE_PATHS),
        "Fable committed-source set differs",
    )
    by_relative = {
        item.get("relative_path"): item for item in sources if isinstance(item, Mapping)
    }
    _require(
        set(by_relative) == set(authorization.RUN_SOURCE_PATHS),
        "Fable committed-source relative paths differ",
    )
    for relative in authorization.RUN_SOURCE_PATHS:
        live = _record(
            root / relative,
            logical_name=f"committed_source:{relative}",
            relative_path=relative,
        )
        _require(
            _identity(by_relative[relative]) == _identity(live),
            f"Fable reviewed source bytes differ: {relative}",
        )

    primary = document.get("primary_evidence")
    _require(isinstance(primary, Mapping), "Fable primary evidence is invalid")
    artifacts = primary.get("artifacts")
    direction = primary.get("direction_cells")
    _require(
        isinstance(artifacts, list) and isinstance(direction, list)
        and primary.get("interpretation")
        == "raw_current_evidence_not_a_supplied_conclusion",
        "Fable primary evidence structure differs",
    )
    observed_primary = {
        item.get("logical_name"): item
        for item in artifacts
        if isinstance(item, Mapping)
    }
    _require(
        set(observed_primary) == EXPECTED_PRIMARY_LOGICAL_NAMES,
        "Fable primary evidence artifact set differs",
    )
    observed_direction = {
        str(item.get("logical_name", "")).removeprefix("stage0_direction_cell:"): item
        for item in direction
        if isinstance(item, Mapping)
    }
    _require(
        set(observed_direction) == EXPECTED_DIRECTION_CELL_IDS,
        "Fable direction-cell evidence set differs",
    )
    if primary_identities is not None:
        combined = {**observed_primary, **{
            f"stage0_direction_cell:{key}": value
            for key, value in observed_direction.items()
        }}
        _require(set(combined) == set(primary_identities), "Fable/live evidence names differ")
        for name, live in primary_identities.items():
            _require(
                _identity(combined[name]) == _identity(live),
                f"Fable/live evidence bytes differ: {name}",
            )

    examples = document.get("reviewer_calibration_examples")
    _require(
        isinstance(examples, Mapping)
        and examples.get("status") == "historical_advisory_stale_until_directly_verified"
        and examples.get("authority") == "none",
        "Fable reviewer-calibration semantics differ",
    )
    calibration_records = [
        *examples.get("same_prompt_independent_pair", []),
        *examples.get("historical_pro_design_preflight", []),
    ]
    _require(len(calibration_records) == 6, "Fable calibration example count differs")
    _require(
        {item.get("logical_name") for item in calibration_records}
        == set(EXPECTED_CALIBRATION_EXAMPLES),
        "Fable calibration example set differs",
    )
    for record in calibration_records:
        _check_calibration_record(record)
    return {
        "source_commit": commit,
        "source_branch": branch,
        "primary_identities": {
            **{name: _identity(value) for name, value in observed_primary.items()},
            **{
                f"stage0_direction_cell:{name}": _identity(value)
                for name, value in observed_direction.items()
            },
        },
    }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _execution_record(path: Path | str, *, logical_name: str) -> dict[str, Any]:
    record = _record(path, logical_name=logical_name)
    record["execution_path"] = record.pop("review_path")
    return record


def _substantive_raw(path: Path) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 1000:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="strict").strip().lower()
    except UnicodeError:
        return False
    return text not in {
        "credit balance is too low", "not logged in", "authentication failed",
        "rate limit",
    }


def invoke_once(
    *,
    repo_root: Path | str,
    briefing_path: Path | str,
    helper_path: Path | str,
    raw_report_path: Path | str,
    receipt_path: Path | str,
    workdir: Path | str,
) -> dict[str, Any]:
    """Invoke the fixed helper once; always preserve an exclusive call receipt."""

    _require(not os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY must be absent")
    root = Path(repo_root).expanduser().resolve(strict=True)
    briefing_path = Path(briefing_path).expanduser().resolve(strict=True)
    helper_path = Path(helper_path).expanduser().resolve(strict=True)
    raw_path = Path(raw_report_path).expanduser().absolute()
    prompt_path = Path(f"{raw_path}.prompt.txt")
    receipt_path = Path(receipt_path).expanduser().absolute()
    workdir_path = Path(workdir).expanduser().resolve(strict=True)
    _require(workdir_path.is_dir(), "Fable workdir is not a directory")
    for path, name in (
        (raw_path, "raw report"), (prompt_path, "compiled prompt"),
        (receipt_path, "invocation receipt"),
    ):
        _require(not path.exists(), f"refusing to overwrite Fable {name}: {path}")

    briefing, _ = authorization._load_json(briefing_path, name="Fable briefing")  # noqa: SLF001
    validate_briefing_document(root, briefing)
    helper_record = _execution_record(helper_path, logical_name="fable_high_helper")
    _require(helper_record["sha256"] == FABLE_HELPER_SHA256, "Fable helper SHA-256 differs")
    briefing_record = _execution_record(briefing_path, logical_name="fable_briefing")
    briefing_record["content_hash_sha256"] = briefing["content_hash_sha256"]
    command = [
        str(helper_path), "--input", str(briefing_path), "--output", str(raw_path),
        "--workdir", str(workdir_path),
    ]
    started_utc = _utc_now()
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=workdir_path,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    ended_utc = _utc_now()

    prompt_record = (
        _execution_record(prompt_path, logical_name="fable_compiled_prompt")
        if prompt_path.is_file() and not prompt_path.is_symlink() else None
    )
    raw_record = (
        _execution_record(raw_path, logical_name="fable_raw_review")
        if raw_path.is_file() and not raw_path.is_symlink() else None
    )
    prompt_valid = bool(
        prompt_record is not None
        and prompt_path.read_bytes() == FABLE_PROMPT_PREFIX + briefing_path.read_bytes()
    )
    raw_valid = _substantive_raw(raw_path)
    success = completed.returncode == 0 and prompt_valid and raw_valid
    document: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_type": "single_call_fable_high_review_invocation_receipt",
        "status": "completed_validated" if success else "failed_no_authorization",
        "source_commit": briefing["source_commit"],
        "source_branch": briefing["source_branch"],
        "review_contract": {
            "helper_sha256": FABLE_HELPER_SHA256,
            "model_alias": "fable",
            "effort": "high",
            "permission_mode": "plan_read_only",
            "session_persistence": False,
            "single_call_no_retry": True,
            "api_key_present": False,
        },
        "execution": {
            "command": command,
            "workdir": str(workdir_path),
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "elapsed_seconds": elapsed,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "call_count": 1,
        },
        "artifacts": {
            "briefing": briefing_record,
            "compiled_prompt": prompt_record,
            "raw_review": raw_record,
            "helper": helper_record,
        },
        "validation": {
            "compiled_prompt_exact": prompt_valid,
            "raw_review_substantive": raw_valid,
        },
    }
    document["content_hash_sha256"] = authorization.canonical_sha256(document)
    _write_exclusive(receipt_path, document)
    _require(success, "single Fable call did not produce a valid substantive review")
    return document


def validate_review_bundle(
    *,
    repo_root: Path | str,
    receipt: Mapping[str, Any],
    briefing_path: Path | str,
    prompt_path: Path | str,
    raw_path: Path | str,
    helper_path: Path | str,
    primary_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Revalidate copied review bytes without requiring original Mac paths."""

    expected_keys = {
        "schema_version", "artifact_type", "status", "source_commit",
        "source_branch", "review_contract", "execution", "artifacts",
        "validation", "content_hash_sha256",
    }
    _require(set(receipt) == expected_keys, "Fable receipt field set differs")
    authorization._validate_content_hash(receipt, name="Fable receipt")  # noqa: SLF001
    _require(
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
        and receipt.get("artifact_type")
        == "single_call_fable_high_review_invocation_receipt"
        and receipt.get("status") == "completed_validated",
        "Fable receipt identity/status differs",
    )
    briefing, _ = authorization._load_json(briefing_path, name="Fable briefing")  # noqa: SLF001
    briefing_validation = validate_briefing_document(
        repo_root, briefing, primary_identities=primary_identities
    )
    live = {
        "briefing": _execution_record(briefing_path, logical_name="fable_briefing"),
        "compiled_prompt": _execution_record(prompt_path, logical_name="fable_compiled_prompt"),
        "raw_review": _execution_record(raw_path, logical_name="fable_raw_review"),
        "helper": _execution_record(helper_path, logical_name="fable_high_helper"),
    }
    live["briefing"]["content_hash_sha256"] = briefing["content_hash_sha256"]
    _require(live["helper"]["sha256"] == FABLE_HELPER_SHA256, "Fable helper differs")
    _require(
        Path(prompt_path).read_bytes()
        == FABLE_PROMPT_PREFIX + Path(briefing_path).read_bytes(),
        "Fable compiled prompt differs from briefing",
    )
    _require(_substantive_raw(Path(raw_path)), "Fable raw review is not substantive")
    recorded = receipt.get("artifacts")
    _require(isinstance(recorded, Mapping), "Fable receipt artifact records are invalid")
    for name, observed in live.items():
        claim = recorded.get(name)
        _require(isinstance(claim, Mapping), f"Fable receipt lacks {name}")
        _require(
            _identity(claim) == _identity(observed),
            f"Fable receipt/live bytes differ: {name}",
        )
    contract = receipt.get("review_contract")
    _require(
        contract == {
            "helper_sha256": FABLE_HELPER_SHA256,
            "model_alias": "fable",
            "effort": "high",
            "permission_mode": "plan_read_only",
            "session_persistence": False,
            "single_call_no_retry": True,
            "api_key_present": False,
        },
        "Fable review contract differs",
    )
    execution = receipt.get("execution")
    _require(
        isinstance(execution, Mapping)
        and execution.get("returncode") == 0
        and execution.get("call_count") == 1
        and isinstance(execution.get("stdout"), str)
        and isinstance(execution.get("stderr"), str)
        and isinstance(execution.get("elapsed_seconds"), (int, float))
        and math.isfinite(float(execution["elapsed_seconds"]))
        and float(execution["elapsed_seconds"]) >= 0.0,
        "Fable execution evidence differs",
    )
    _require(
        receipt.get("validation")
        == {"compiled_prompt_exact": True, "raw_review_substantive": True},
        "Fable receipt validation flags differ",
    )
    _require(
        receipt.get("source_commit") == briefing_validation["source_commit"]
        and receipt.get("source_branch") == briefing_validation["source_branch"],
        "Fable receipt source differs",
    )
    original_paths = recorded
    expected_command = [
        str(original_paths["helper"].get("execution_path")),
        "--input", str(original_paths["briefing"].get("execution_path")),
        "--output", str(original_paths["raw_review"].get("execution_path")),
        "--workdir", str(execution.get("workdir")),
    ]
    _require(execution.get("command") == expected_command, "Fable command differs")
    return {
        "review_valid": True,
        "source_commit": briefing_validation["source_commit"],
        "source_branch": briefing_validation["source_branch"],
        "briefing": _identity(live["briefing"]),
        "compiled_prompt": _identity(live["compiled_prompt"]),
        "raw_review": _identity(live["raw_review"]),
        "helper": _identity(live["helper"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-briefing")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--stage0-provenance", type=Path, required=True)
    build.add_argument("--direction-summary", type=Path, required=True)
    build.add_argument("--direction-cell", action="append", type=Path, required=True)
    build.add_argument("--exact-gradient", type=Path, required=True)
    build.add_argument("--minibatch-gradient", type=Path, required=True)
    build.add_argument("--older-shared-prompt", type=Path, required=True)
    build.add_argument("--older-pro-raw", type=Path, required=True)
    build.add_argument("--older-fable-raw", type=Path, required=True)
    build.add_argument("--pro-preflight-spec", type=Path, required=True)
    build.add_argument("--pro-preflight-raw", type=Path, required=True)
    build.add_argument("--pro-preflight-receipt", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    invoke = subparsers.add_parser("invoke-once")
    invoke.add_argument("--repo-root", type=Path, required=True)
    invoke.add_argument("--briefing", type=Path, required=True)
    invoke.add_argument("--helper", type=Path, required=True)
    invoke.add_argument("--raw-report", type=Path, required=True)
    invoke.add_argument("--receipt", type=Path, required=True)
    invoke.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-briefing":
        document = build_briefing(
            repo_root=args.repo_root,
            stage0_provenance_path=args.stage0_provenance,
            direction_summary_path=args.direction_summary,
            direction_cell_paths=args.direction_cell,
            exact_gradient_path=args.exact_gradient,
            minibatch_gradient_path=args.minibatch_gradient,
            older_shared_prompt_path=args.older_shared_prompt,
            older_pro_raw_path=args.older_pro_raw,
            older_fable_raw_path=args.older_fable_raw,
            pro_preflight_spec_path=args.pro_preflight_spec,
            pro_preflight_raw_path=args.pro_preflight_raw,
            pro_preflight_receipt_path=args.pro_preflight_receipt,
            output_path=args.output,
        )
        print(json.dumps({"status": "briefing_created", "content_hash_sha256": document["content_hash_sha256"]}, sort_keys=True))
        return 0
    document = invoke_once(
        repo_root=args.repo_root,
        briefing_path=args.briefing,
        helper_path=args.helper,
        raw_report_path=args.raw_report,
        receipt_path=args.receipt,
        workdir=args.workdir,
    )
    print(json.dumps({"status": document["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADVERSARIAL_CHECKLIST",
    "BRIEFING_SCHEMA_VERSION",
    "FABLE_HELPER_SHA256",
    "PROTECTED_PATH",
    "RECEIPT_SCHEMA_VERSION",
    "build_briefing",
    "invoke_once",
    "validate_briefing_document",
    "validate_review_bundle",
]
