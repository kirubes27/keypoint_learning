"""Build and safely emit the complete primary representation split bundle.

The production builder performs two independent content-inventory scans and two
split generations in memory, invokes the independent split verifier with the
second generation, and only then constructs the verifier report and top-level
manifest.  No dataset file is ever written.

The emitter is no-overwrite: its output directory must not already exist and
every file is opened in exclusive-creation mode.  A failed write can leave a
partial new directory for manual inspection; it is never automatically removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from keypoint_net.representation_corpus_inventory import (
    OBJECT_ORDER,
    build_all_corpus_inventories,
    validate_corpus_inventory,
)
from keypoint_net.representation_split_verifier import verify_primary_split_artifacts
from keypoint_net.representation_splits import (
    GENERATOR_CONFIG_SHA256,
    GENERATOR_NAME,
    OBJECT_ROLES,
    PRIMARY_GROUPS,
    PrimaryGroup,
    canonical_json_bytes,
    generate_primary_split_artifacts,
)


BUNDLE_SCHEMA_VERSION = "representation_split_bundle.v1"
MANIFEST_ARTIFACT_TYPE = "representation_split_manifest"
REPORT_ARTIFACT_TYPE = "representation_split_verification_report"
SPEC_RELPATH = "docs/decisions/2026-07-26/REPRESENTATION_ORACLE_EVALUATOR_SPLIT_SPEC_v1.md"
CONTAMINATION_LEDGER_RELPATH = (
    "docs/decisions/2026-07-26/OBJECT_ROLE_CONTAMINATION_LEDGER_2026-07-27.md"
)
GENERATOR_PROVENANCE_RELPATH = (
    "docs/decisions/2026-07-26/provenance/"
    "verified_generator_snapshot_2026-07-27/SNAPSHOT_MANIFEST.json"
)
GENERATOR_SOURCE_RELPATHS = (
    "keypoint_net/representation_corpus_inventory.py",
    "keypoint_net/representation_splits.py",
    "keypoint_net/representation_split_verifier.py",
    "keypoint_net/representation_split_bundle.py",
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class SplitBundleError(ValueError):
    """Raised when a final split bundle cannot be built or safely emitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitBundleError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(data: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitBundleError(f"{context}: invalid JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{context}: expected a JSON object")
    return value


def _content_hashed_document(document: Mapping[str, Any]) -> bytes:
    payload = dict(document)
    payload.pop("content_hash_sha256", None)
    payload["content_hash_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return canonical_json_bytes(payload, trailing_newline=True)


def _partition(group: PrimaryGroup) -> dict[str, list[int]]:
    if group.family == "roll":
        return {
            "train_frame_indices": list(range(27, 177)),
            "holdout_frame_indices": list(range(24)),
            "guard_frame_indices": list(range(24, 27)) + list(range(177, 180)),
        }
    if group.family in {"yaw", "pitch"}:
        return {
            "train_frame_indices": list(range(47)) + list(range(74, 121)),
            "holdout_frame_indices": list(range(53, 68)),
            "guard_frame_indices": list(range(47, 53)) + list(range(68, 74)),
        }
    if group.family == "scale":
        return {
            "train_frame_indices": list(range(49)) + list(range(72, 121)),
            "holdout_frame_indices": list(range(53, 68)),
            "guard_frame_indices": list(range(49, 53)) + list(range(68, 72)),
        }
    if group.physical_axis == "world_x":
        return {
            "train_frame_indices": [gy * 11 + gx for gy in range(7) for gx in range(11)],
            "holdout_frame_indices": [
                gy * 11 + gx for gy in range(9, 11) for gx in range(11)
            ],
            "guard_frame_indices": [
                gy * 11 + gx for gy in range(7, 9) for gx in range(11)
            ],
        }
    return {
        "train_frame_indices": [gy * 11 + gx for gy in range(11) for gx in range(7)],
        "holdout_frame_indices": [
            gy * 11 + gx for gy in range(11) for gx in range(9, 11)
        ],
        "guard_frame_indices": [
            gy * 11 + gx for gy in range(11) for gx in range(7, 9)
        ],
    }


def _source_edges(group: PrimaryGroup) -> list[tuple[int, int]]:
    if group.family == "roll":
        return [(src, (src + 3) % 180) for src in range(180)]
    if group.family in {"yaw", "pitch"}:
        if group.direction == "forward":
            return [(src, src + 6) for src in range(115)]
        return [(src, src - 6) for src in range(6, 121)]
    if group.family == "scale":
        if group.direction == "forward":
            return [(src, src + 4) for src in range(117)]
        return [(src, src - 4) for src in range(4, 121)]
    edges: list[tuple[int, int]] = []
    if group.physical_axis == "world_x":
        starts = range(8) if group.direction == "forward" else range(3, 11)
        for gy in range(11):
            for gx in starts:
                src = gy * 11 + gx
                edges.append((src, src + 3 if group.direction == "forward" else src - 3))
    else:
        starts = range(8) if group.direction == "forward" else range(3, 11)
        for gy in starts:
            for gx in range(11):
                src = gy * 11 + gx
                edges.append((src, src + 33 if group.direction == "forward" else src - 33))
    return edges


def _numeric_state_summary(states: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for state in states
            for key, value in state.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    fields: dict[str, Any] = {}
    for key in keys:
        values = [float(state[key]) for state in states if key in state]
        _require(all(math.isfinite(value) for value in values), f"non-finite state field {key}")
        fields[key] = {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "unique_count": len(set(values)),
        }
    return {
        "state_count": len(states),
        "state_sha256": hashlib.sha256(canonical_json_bytes(states)).hexdigest(),
        "numeric_fields": fields,
    }


def _group_summary(
    group: PrimaryGroup,
    inventory_document: Mapping[str, Any],
) -> dict[str, Any]:
    partition = _partition(group)
    train = set(partition["train_frame_indices"])
    holdout = set(partition["holdout_frame_indices"])
    guard = set(partition["guard_frame_indices"])
    dropped: list[dict[str, Any]] = []
    for model in OBJECT_ORDER:
        for src, dst in _source_edges(group):
            if (src in train and dst in train) or (src in holdout and dst in holdout):
                continue
            reason = "guard_endpoint" if src in guard or dst in guard else "cross_boundary"
            dropped.append(
                {
                    "model_name": model,
                    "src_frame_index": src,
                    "dst_frame_index": dst,
                    "reason": reason,
                }
            )

    frame_lookup = {
        (record["model_name"], record["frame_index"]): record
        for record in inventory_document["frame_records"]
        if record["source_partition"] == "train"
    }
    coverage: dict[str, Any] = {}
    for label, indices in (
        ("train", partition["train_frame_indices"]),
        ("holdout", partition["holdout_frame_indices"]),
        ("guard", partition["guard_frame_indices"]),
    ):
        by_object = {}
        for model in OBJECT_ORDER:
            states = [frame_lookup[(model, index)]["physical_state"] for index in indices]
            by_object[model] = _numeric_state_summary(states)
        coverage[label] = {
            "frame_indices": indices,
            "frame_count_per_object": len(indices),
            "by_object": by_object,
        }
    return {
        "group_id": f"{group.family}:{group.physical_axis}:{group.direction}:stride{group.stride}",
        "dataset_key": group.dataset_key,
        "transform": {
            "family": group.family,
            "physical_axis": group.physical_axis,
            "direction": group.direction,
            "stride": group.stride,
            "stride_units": group.stride_units,
            "cyclic": group.cyclic,
            "expected_2d_family": group.expected_2d_family,
        },
        "source_pair_index_relpath": group.source_pair_index_relpath,
        "frame_partition": partition,
        "dropped_pair_count": len(dropped),
        "dropped_pairs": dropped,
        "physical_state_coverage": coverage,
    }


def _repo_evidence(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = repo_root / Path(*PurePosixPath(relative_path).parts)
    _require(path.is_file(), f"missing required versioned evidence: {relative_path}")
    return {
        "relative_path": relative_path,
        "sha256": _sha256_file(path),
    }


def _hardened_git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {"GIT_CONFIG_NOSYSTEM": "1", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}
    )
    return environment


def _repo_evidence_at_commit(
    repo_root: Path,
    relative_path: str,
    commit: str,
) -> dict[str, Any]:
    """Bind frozen generator evidence to its historical Git blob.

    Validation must not require today's inventory scanner to be byte-identical
    to the scanner that created an already frozen, hash-bound split bundle.
    New bundle creation still uses ``verify_commit_contains_generator_sources``
    and therefore still requires the running generator to match its commit.
    """

    try:
        data = subprocess.run(
            ["git", "--no-replace-objects", "show", f"{commit}:{relative_path}"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_hardened_git_environment(),
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SplitBundleError(
            f"{relative_path}: absent from generator commit {commit}"
        ) from exc
    return {"relative_path": relative_path, "sha256": _sha256_bytes(data)}


def _scale_holdout_proof(
    scale_inventory: Mapping[str, Any],
    pair_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    holdout_files = [
        item
        for item in scale_inventory["files"]
        if PurePosixPath(item["relative_path"]).parts[:1] == ("eval_abs_holdout",)
    ]
    excluded_index = next(
        (
            item
            for item in scale_inventory["source_pair_indices"]
            if item["relative_path"] == "indices/pairs_eval_abs_holdout_ratio_r4.json"
        ),
        None,
    )
    _require(excluded_index is not None, "scale inventory lacks absolute-holdout pair index")
    holdout_frame_records = [
        record
        for record in scale_inventory["frame_records"]
        if record["source_partition"] == "eval_abs_holdout"
    ]
    holdout_frame_counts_by_object = {
        model: sum(
            record["model_name"] == model for record in holdout_frame_records
        )
        for model in OBJECT_ORDER
    }
    _require(
        holdout_frame_counts_by_object == {model: 31 for model in OBJECT_ORDER},
        "scale absolute holdout must contain exactly 31 frames for every object",
    )
    leaked_paths: list[str] = []
    for filename, raw in pair_artifacts.items():
        if not filename.startswith("scale__"):
            continue
        artifact = _decode(raw, filename)
        for pair in artifact["pairs"]:
            for key in (
                "src_image_relpath",
                "dst_image_relpath",
                "src_mask_relpath",
                "dst_mask_relpath",
            ):
                if "eval_abs_holdout" in PurePosixPath(pair[key]).parts:
                    leaked_paths.append(pair[key])
    _require(not leaked_paths, f"scale absolute holdout leaked into primary artifacts: {leaked_paths}")
    return {
        "excluded_path_prefix": "eval_abs_holdout/",
        "excluded_regular_file_count": len(holdout_files),
        "excluded_pair_index": excluded_index,
        "frame_count_per_object": 31,
        "object_count": len(OBJECT_ORDER),
        "excluded_frame_count": len(holdout_frame_records),
        "expected_excluded_frame_count": 31 * len(OBJECT_ORDER),
        "frame_count_proof": "31 frames/object * 6 objects = 186 frames",
        "excluded_frame_counts_by_object": holdout_frame_counts_by_object,
        "primary_pair_path_intersection_count": 0,
        "primary_pair_path_intersection": [],
    }


def _inventory_manifest_entry(
    dataset_key: str,
    raw: bytes,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "relative_path": f"inventories/CORPUS_INVENTORY__{dataset_key}.json",
        "file_sha256": _sha256_bytes(raw),
        "content_hash_sha256": document["content_hash_sha256"],
        "dataset_basename": document["dataset_basename"],
        "source_root_provenance": document["source_root_provenance"],
        "unversioned_source": document["unversioned_source"],
        "file_count": document["file_count"],
        "total_bytes": document["total_bytes"],
    }


def _pair_manifest_entry(
    filename: str,
    raw: bytes,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    source_sets = {
        model: sorted(
            {
                pair["src_frame_index"]
                for pair in document["pairs"]
                if pair["model_name"] == model
            }
        )
        for model in document["included_objects"]
    }
    target_sets = {
        model: sorted(
            {
                pair["dst_frame_index"]
                for pair in document["pairs"]
                if pair["model_name"] == model
            }
        )
        for model in document["included_objects"]
    }
    endpoint_sets = {
        model: sorted(set(source_sets[model]) | set(target_sets[model]))
        for model in document["included_objects"]
    }
    return {
        "relative_path": f"pairs/{filename}",
        "file_sha256": _sha256_bytes(raw),
        "content_hash_sha256": document["content_hash_sha256"],
        "dataset_basename": document["dataset_basename"],
        "split": document["split"],
        "transform": document["transform"],
        "included_objects": document["included_objects"],
        "pair_count": document["pair_count"],
        "pair_counts_by_object": document["pair_counts_by_object"],
        "source_frame_indices_by_object": source_sets,
        "target_frame_indices_by_object": target_sets,
        "endpoint_frame_indices_by_object": endpoint_sets,
        "source_frame_counts_by_object": {
            model: len(indices) for model, indices in source_sets.items()
        },
        "target_frame_counts_by_object": {
            model: len(indices) for model, indices in target_sets.items()
        },
        "endpoint_frame_counts_by_object": {
            model: len(indices) for model, indices in endpoint_sets.items()
        },
    }


def build_primary_split_bundle(
    dataset_roots: Mapping[str, str | Path],
    *,
    generator_commit: str,
) -> dict[str, bytes]:
    """Build a complete bundle bound to the exact committed generator sources."""

    _require(
        isinstance(generator_commit, str) and _COMMIT_RE.fullmatch(generator_commit) is not None,
        "generator_commit must be lowercase 40-hex",
    )
    repo_root = Path(__file__).resolve().parents[1]
    verify_commit_contains_generator_sources(repo_root, generator_commit)
    first_inventories = build_all_corpus_inventories(dataset_roots)
    second_inventories = build_all_corpus_inventories(dataset_roots)
    _require(
        first_inventories == second_inventories,
        "corpus inventories are not byte-identical across two complete scans",
    )
    validated = {
        key: validate_corpus_inventory(first_inventories[key], key, dataset_roots[key])
        for key in sorted(first_inventories)
    }
    first_artifacts = generate_primary_split_artifacts(
        dataset_roots,
        corpus_inventories=first_inventories,
        generator_commit=generator_commit,
    )
    second_artifacts = generate_primary_split_artifacts(
        dataset_roots,
        corpus_inventories=second_inventories,
        generator_commit=generator_commit,
    )
    report = verify_primary_split_artifacts(
        first_artifacts,
        dataset_roots,
        corpus_inventories=first_inventories,
        expected_generator_commit=generator_commit,
        regenerated_artifacts=second_artifacts,
    )
    _require(report["structurally_valid"] is True, "independent verifier structural failure")
    _require(report["validated_corpus_inventories"] is True, "inventory validation failure")
    _require(report["byte_identical_regeneration"] is True, "regeneration proof missing")
    _require(report["gate_pass"] is True, "independent split gate did not pass")

    pair_file_sha = {
        filename: _sha256_bytes(raw) for filename, raw in sorted(first_artifacts.items())
    }
    report_document = {
        **report,
        "corpus_inventory_content_hashes": {
            key: validated[key].content_hash_sha256 for key in sorted(validated)
        },
        "pair_artifact_file_sha256": pair_file_sha,
    }
    report_bytes = _content_hashed_document(report_document)
    report_with_hash = _decode(report_bytes, "split verification report")

    inventory_entries = {}
    for key in sorted(validated):
        document = validated[key].document
        inventory_entries[key] = _inventory_manifest_entry(
            key,
            first_inventories[key],
            document,
        )

    pair_entries = []
    for filename, raw in sorted(first_artifacts.items()):
        document = _decode(raw, filename)
        pair_entries.append(_pair_manifest_entry(filename, raw, document))

    manifest_document = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "specification": _repo_evidence(repo_root, SPEC_RELPATH),
        "object_role_lock": {
            "roles": OBJECT_ROLES,
            "contamination_ledger": _repo_evidence(repo_root, CONTAMINATION_LEDGER_RELPATH),
        },
        "render_source_provenance": _repo_evidence(repo_root, GENERATOR_PROVENANCE_RELPATH),
        "split_generator": {
            "name": GENERATOR_NAME,
            "commit": generator_commit,
            "config_sha256": GENERATOR_CONFIG_SHA256,
            "sources": [
                _repo_evidence(repo_root, relative_path)
                for relative_path in GENERATOR_SOURCE_RELPATHS
            ],
        },
        "corpus_inventories": inventory_entries,
        "pair_artifact_count": len(pair_entries),
        "pair_artifacts": pair_entries,
        "group_summaries": [
            _group_summary(group, validated[group.dataset_key].document)
            for group in PRIMARY_GROUPS
        ],
        "scale_absolute_holdout_nonmembership": _scale_holdout_proof(
            validated["scale"].document,
            first_artifacts,
        ),
        "regeneration": {
            "corpus_inventory_byte_identical": True,
            "pair_artifact_byte_identical": first_artifacts == second_artifacts,
        },
        "independent_verifier": {
            "relative_path": "SPLIT_VERIFIER_REPORT.json",
            "file_sha256": _sha256_bytes(report_bytes),
            "content_hash_sha256": report_with_hash["content_hash_sha256"],
            "structurally_valid": report["structurally_valid"],
            "validated_corpus_inventories": report["validated_corpus_inventories"],
            "byte_identical_regeneration": report["byte_identical_regeneration"],
            "gate_pass": report["gate_pass"],
        },
    }
    manifest_bytes = _content_hashed_document(manifest_document)

    bundle: dict[str, bytes] = {
        f"inventories/CORPUS_INVENTORY__{key}.json": first_inventories[key]
        for key in sorted(first_inventories)
    }
    bundle.update(
        {f"pairs/{filename}": raw for filename, raw in sorted(first_artifacts.items())}
    )
    bundle["SPLIT_VERIFIER_REPORT.json"] = report_bytes
    bundle["SPLIT_MANIFEST.json"] = manifest_bytes
    _require(len(bundle) == 40, f"expected 40 bundle files, got {len(bundle)}")
    sorted_bundle = dict(sorted(bundle.items()))
    validate_primary_split_bundle(
        sorted_bundle,
        verify_corpus_contents=False,
    )
    return sorted_bundle


def verify_commit_contains_generator_sources(repo_root: Path, commit: str) -> None:
    """Fail unless the named commit contains the exact running generator sources."""

    _require(_COMMIT_RE.fullmatch(commit) is not None, "commit must be lowercase 40-hex")
    try:
        subprocess.run(
            ["git", "--no-replace-objects", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_hardened_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SplitBundleError(f"generator commit does not exist: {commit}") from exc
    for relative_path in GENERATOR_SOURCE_RELPATHS:
        current = (repo_root / relative_path).read_bytes()
        try:
            committed = subprocess.run(
                ["git", "--no-replace-objects", "show", f"{commit}:{relative_path}"],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_hardened_git_environment(),
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SplitBundleError(
                f"{relative_path}: absent from generator commit {commit}"
            ) from exc
        _require(
            current == committed,
            f"{relative_path}: running source differs from generator commit {commit}",
        )


def _expected_bundle_names() -> set[str]:
    return {
        "SPLIT_MANIFEST.json",
        "SPLIT_VERIFIER_REPORT.json",
        *{f"inventories/CORPUS_INVENTORY__{key}.json" for key in validated_dataset_keys()},
        *{
            f"pairs/{group.artifact_stem}__{split}.json"
            for group in PRIMARY_GROUPS
            for split in ("train", "validation", "test")
        },
    }


def _validate_hashed_json(data: bytes, context: str) -> dict[str, Any]:
    document = _decode(data, context)
    _require(
        data == canonical_json_bytes(document, trailing_newline=True),
        f"{context}: bytes are not canonical JSON plus one newline",
    )
    claimed = document.get("content_hash_sha256")
    _require(
        isinstance(claimed, str) and re.fullmatch(r"[0-9a-f]{64}", claimed) is not None,
        f"{context}: invalid content hash",
    )
    payload = dict(document)
    del payload["content_hash_sha256"]
    _require(
        claimed == hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        f"{context}: content hash mismatch",
    )
    return document


def validate_primary_split_bundle(
    bundle: Mapping[str, bytes],
    *,
    verify_corpus_contents: bool = True,
) -> dict[str, Any]:
    """Validate all internal bundle hashes and the committed generator binding."""

    expected = _expected_bundle_names()
    _require(set(bundle) == expected, "bundle filename set is incomplete or contains extras")
    for relative_path, data in bundle.items():
        relative = PurePosixPath(relative_path)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe bundle relative path: {relative_path}",
        )
        _require(isinstance(data, bytes), f"{relative_path}: bundle value must be bytes")

    manifest = _validate_hashed_json(bundle["SPLIT_MANIFEST.json"], "SPLIT_MANIFEST.json")
    report = _validate_hashed_json(
        bundle["SPLIT_VERIFIER_REPORT.json"],
        "SPLIT_VERIFIER_REPORT.json",
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "artifact_type",
            "specification",
            "object_role_lock",
            "render_source_provenance",
            "split_generator",
            "corpus_inventories",
            "pair_artifact_count",
            "pair_artifacts",
            "group_summaries",
            "scale_absolute_holdout_nonmembership",
            "regeneration",
            "independent_verifier",
            "content_hash_sha256",
        },
        "manifest top-level schema mismatch",
    )
    _require(manifest.get("schema_version") == BUNDLE_SCHEMA_VERSION, "wrong manifest schema")
    _require(manifest.get("artifact_type") == MANIFEST_ARTIFACT_TYPE, "wrong manifest type")
    _require(
        report.get("artifact_type") == REPORT_ARTIFACT_TYPE,
        "wrong verifier report type",
    )
    for key in (
        "structurally_valid",
        "validated_corpus_inventories",
        "byte_identical_regeneration",
        "gate_pass",
    ):
        _require(report.get(key) is True, f"verifier report does not prove {key}")
        _require(
            manifest["independent_verifier"].get(key) is True,
            f"manifest does not bind verifier result {key}",
        )
    _require(
        manifest["regeneration"]
        == {
            "corpus_inventory_byte_identical": True,
            "pair_artifact_byte_identical": True,
        },
        "manifest regeneration proof is incomplete",
    )
    expected_verifier_entry = {
        "relative_path": "SPLIT_VERIFIER_REPORT.json",
        "file_sha256": _sha256_bytes(bundle["SPLIT_VERIFIER_REPORT.json"]),
        "content_hash_sha256": report["content_hash_sha256"],
        "structurally_valid": report["structurally_valid"],
        "validated_corpus_inventories": report["validated_corpus_inventories"],
        "byte_identical_regeneration": report["byte_identical_regeneration"],
        "gate_pass": report["gate_pass"],
    }
    _require(
        manifest["independent_verifier"] == expected_verifier_entry,
        "manifest verifier-report binding mismatch",
    )

    inventory_entries = manifest.get("corpus_inventories")
    _require(
        isinstance(inventory_entries, dict)
        and set(inventory_entries) == set(validated_dataset_keys()),
        "manifest inventory mapping mismatch",
    )
    inventory_bytes_by_key: dict[str, bytes] = {}
    inventory_documents_by_key: dict[str, dict[str, Any]] = {}
    for key, entry in inventory_entries.items():
        relative_path = f"inventories/CORPUS_INVENTORY__{key}.json"
        raw = bundle[relative_path]
        document = _validate_hashed_json(raw, relative_path)
        inventory_bytes_by_key[key] = raw
        inventory_documents_by_key[key] = document
        _require(
            entry == _inventory_manifest_entry(key, raw, document),
            f"{key}: manifest inventory summary disagrees with inventory artifact",
        )
        _require(
            report["corpus_inventory_content_hashes"][key]
            == document["content_hash_sha256"],
            f"{key}: verifier report inventory hash mismatch",
        )
        if verify_corpus_contents:
            validate_corpus_inventory(
                raw,
                key,
                document["source_root_provenance"],
            )

    pair_entries = manifest.get("pair_artifacts")
    _require(
        isinstance(pair_entries, list)
        and len(pair_entries) == 33
        and manifest.get("pair_artifact_count") == 33,
        "manifest pair artifact count mismatch",
    )
    pair_artifacts_by_filename: dict[str, bytes] = {}
    expected_pair_entries: list[dict[str, Any]] = []
    pair_paths = sorted(name for name in expected if name.startswith("pairs/"))
    for relative_path in pair_paths:
        raw = bundle[relative_path]
        document = _validate_hashed_json(raw, relative_path)
        source_filename = PurePosixPath(relative_path).name
        pair_artifacts_by_filename[source_filename] = raw
        expected_pair_entries.append(
            _pair_manifest_entry(source_filename, raw, document)
        )
        _require(
            report["pair_artifact_file_sha256"][source_filename] == _sha256_bytes(raw),
            f"{relative_path}: verifier report file hash mismatch",
        )
    _require(
        pair_entries == expected_pair_entries,
        "manifest pair summaries disagree with the pair artifacts",
    )
    _require(
        manifest["group_summaries"]
        == [
            _group_summary(group, inventory_documents_by_key[group.dataset_key])
            for group in PRIMARY_GROUPS
        ],
        "manifest group summaries disagree with the bound inventories",
    )
    _require(
        manifest["scale_absolute_holdout_nonmembership"]
        == _scale_holdout_proof(
            inventory_documents_by_key["scale"],
            pair_artifacts_by_filename,
        ),
        "manifest scale absolute-holdout proof mismatch",
    )

    if verify_corpus_contents:
        dataset_roots = {
            key: document["source_root_provenance"]
            for key, document in inventory_documents_by_key.items()
        }
        regenerated = generate_primary_split_artifacts(
            dataset_roots,
            corpus_inventories=inventory_bytes_by_key,
            generator_commit=manifest["split_generator"]["commit"],
        )
        _require(
            pair_artifacts_by_filename == regenerated,
            "bundle pair artifacts are not byte-identical to current regeneration",
        )
        current_report = verify_primary_split_artifacts(
            pair_artifacts_by_filename,
            dataset_roots,
            corpus_inventories=inventory_bytes_by_key,
            expected_generator_commit=manifest["split_generator"]["commit"],
            regenerated_artifacts=regenerated,
        )
        for key in (
            "structurally_valid",
            "validated_corpus_inventories",
            "artifact_count",
            "content_hashes",
            "pair_counts",
            "scale_abs_holdout_nonmembership",
            "byte_identical_regeneration",
            "gate_pass",
            "generator_config_sha256",
        ):
            _require(
                report.get(key) == current_report.get(key),
                f"bundled verifier report disagrees with current independent verification: {key}",
            )

    repo_root = Path(__file__).resolve().parents[1]
    generator_commit = manifest["split_generator"].get("commit")
    _require(
        isinstance(generator_commit, str)
        and _COMMIT_RE.fullmatch(generator_commit) is not None,
        "manifest split-generator commit must be lowercase 40-hex",
    )
    expected_split_generator = {
        "name": GENERATOR_NAME,
        "commit": generator_commit,
        "config_sha256": GENERATOR_CONFIG_SHA256,
        "sources": [
            _repo_evidence_at_commit(repo_root, relative_path, generator_commit)
            for relative_path in GENERATOR_SOURCE_RELPATHS
        ],
    }
    _require(
        manifest["split_generator"] == expected_split_generator,
        "manifest split-generator binding mismatch",
    )
    _require(
        manifest["specification"] == _repo_evidence(repo_root, SPEC_RELPATH),
        "manifest specification hash mismatch",
    )
    _require(
        manifest["object_role_lock"]
        == {
            "roles": OBJECT_ROLES,
            "contamination_ledger": _repo_evidence(
                repo_root,
                CONTAMINATION_LEDGER_RELPATH,
            ),
        },
        "manifest object-role lock mismatch",
    )
    _require(
        manifest["render_source_provenance"]
        == _repo_evidence(repo_root, GENERATOR_PROVENANCE_RELPATH),
        "manifest render-source provenance hash mismatch",
    )
    return manifest


def emit_primary_split_bundle(
    output_directory: str | Path,
    bundle: Mapping[str, bytes],
) -> list[str]:
    """Write a prevalidated bundle without overwriting any existing path."""

    output = Path(output_directory)
    _require(not output.exists(), f"refusing to overwrite existing output path: {output}")
    validate_primary_split_bundle(bundle)

    output.mkdir(parents=False, exist_ok=False)
    written: list[str] = []
    ordinary_paths = sorted(
        path
        for path in bundle
        if path not in {"SPLIT_MANIFEST.json", "SPLIT_VERIFIER_REPORT.json"}
    )
    write_order = ordinary_paths + ["SPLIT_VERIFIER_REPORT.json", "SPLIT_MANIFEST.json"]
    for relative_path in write_order:
        target = output / Path(*PurePosixPath(relative_path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(bundle[relative_path])
        written.append(relative_path)
    return written


def validated_dataset_keys() -> tuple[str, ...]:
    return tuple(sorted({group.dataset_key for group in PRIMARY_GROUPS}))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roll-root", required=True)
    parser.add_argument("--yaw-root", required=True)
    parser.add_argument("--pitch-root", required=True)
    parser.add_argument("--scale-root", required=True)
    parser.add_argument("--translation-root", required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    verify_commit_contains_generator_sources(repo_root, args.generator_commit)
    roots = {
        "roll": args.roll_root,
        "yaw": args.yaw_root,
        "pitch": args.pitch_root,
        "scale": args.scale_root,
        "translation": args.translation_root,
    }
    bundle = build_primary_split_bundle(roots, generator_commit=args.generator_commit)
    written = emit_primary_split_bundle(args.output_directory, bundle)
    print(
        json.dumps(
            {
                "gate_pass": True,
                "output_directory": str(Path(args.output_directory).resolve()),
                "file_count": len(written),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "SplitBundleError",
    "build_primary_split_bundle",
    "emit_primary_split_bundle",
    "main",
    "validate_primary_split_bundle",
    "verify_commit_contains_generator_sources",
]
