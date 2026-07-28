"""Focused fail-closed tests for checkpoint-blind replay manifests.

The integration tests read and hash source/dataset files only.  They never
inspect any checkpoint or run-directory artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from keypoint_net import representation_evaluation_provenance as provenance
from keypoint_net import representation_corpus_inventory as corpus_inventory
from keypoint_net import representation_splits as split_validator
from keypoint_net.diagnostics import build_checkpoint_replay_manifests as builder


REPO_ROOT = Path(__file__).resolve().parents[1]


def _rehash(document: dict) -> None:
    document["content_hash_sha256"] = builder.manifest_content_hash(document)


@pytest.mark.parametrize(
    "module",
    (builder, corpus_inventory, split_validator, provenance),
)
def test_manifest_validation_sources_capture_their_import_bytes(module) -> None:
    assert module.__representation_import_sha256__ == hashlib.sha256(
        Path(module.__file__).absolute().read_bytes()
    ).hexdigest()


def test_checkpoint_provenance_roles_have_exact_fixed_paths() -> None:
    expected = {
        "checkpoint_runtime_source_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "CHECKPOINT_RUNTIME_SOURCE_MANIFEST_v1.json"
        ),
        "checkpoint_hash_preflight": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "CHECKPOINT_HASH_PREFLIGHT_RESULT_2026-07-27.json"
        ),
        "replay_dataset_inventory": (
            "docs/decisions/2026-07-26/representation_oracle_splits/"
            "inventories/CORPUS_INVENTORY__roll.json"
        ),
        "replay_pair_index": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_PAIR_INDEX_MANIFEST_v1.json"
        ),
        "replay_frame_mask_inventory": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_FRAME_MASK_INVENTORY_v1.json"
        ),
        "replay_metadata_manifest": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "HAMMER_ROLL_METADATA_MANIFEST_v1.json"
        ),
        "checkpoint_execution_authorization": (
            "docs/decisions/2026-07-26/representation_oracle_replay/"
            "CHECKPOINT_EXECUTION_AUTHORIZATION_v1.json"
        ),
        "fable_execution_review": (
            "docs/decisions/2026-07-26/"
            "FABLE_5_HIGH_CHECKPOINT_REPLAY_RUNTIME_REVIEW_2026-07-28.md"
        ),
    }
    assert {
        role: provenance.FIXED_ROLE_PATHS[role]
        for role in expected
    } == expected
    committed, external = provenance.required_role_profile(
        "checkpoint",
        fit_from_pairs=False,
    )
    assert set(expected) <= committed
    assert external == frozenset(
        {"checkpoint", "checkpoint_config", "checkpoint_metadata"}
    )


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(
        builder.CheckpointReplayManifestError,
        match="duplicate JSON object key",
    ):
        builder.decode_strict_json(
            b'{"value":1,"value":2}',
            context="duplicate fixture",
        )
    with pytest.raises(
        builder.CheckpointReplayManifestError,
        match="non-standard JSON constant",
    ):
        builder.decode_strict_json(
            b'{"value":NaN}',
            context="nonfinite fixture",
        )


@pytest.mark.parametrize(
    "value",
    (
        "/absolute/path.json",
        "../escape.json",
        "folder/../escape.json",
        "folder\\windows.json",
        "C:drive.json",
        "./dot.json",
    ),
)
def test_unsafe_relative_paths_are_rejected(value: str) -> None:
    with pytest.raises(
        builder.CheckpointReplayManifestError,
        match="unsafe|forbidden",
    ):
        builder._safe_relative_path(value, context="unsafe fixture")


def test_manifest_content_hash_is_strict_and_top_level_only() -> None:
    document = {
        "schema_version": "fixture.v1",
        "nested": {"content_hash_sha256": "semantic nested value"},
    }
    digest = builder.manifest_content_hash(document)
    assert len(digest) == 64
    with_hash = dict(document)
    with_hash["content_hash_sha256"] = digest
    assert builder.manifest_content_hash(with_hash) == digest
    with_hash["nested"]["content_hash_sha256"] = "changed"
    assert builder.manifest_content_hash(with_hash) != digest


@pytest.fixture(scope="module")
def built_documents() -> dict:
    return builder.build_checkpoint_replay_manifest_documents()


def test_live_build_is_checkpoint_blind_and_has_exact_semantics(
    built_documents: dict,
) -> None:
    builder.validate_manifest_shapes(built_documents)
    encoded = builder.encoded_manifest_documents(built_documents)
    combined = b"\n".join(encoded.values())
    assert b"best_model.pt" not in combined
    assert b"cluster_downloads" not in combined
    assert b"history.json" not in combined

    pair = built_documents["pair_index"]
    assert pair["pair_index"] == {
        "relative_path": "indices/pairs_skip3_cyclic.json",
        "size_bytes": 640562,
        "sha256": (
            "87f51500a080fa964fced92f6382c5d421d74f707c926686d565cabaf9a39fc3"
        ),
    }
    assert pair["semantics"]["selected_object_pair_count"] == 180
    assert pair["semantics"]["target_frame_formula"] == (
        "(source_frame_index + 3) modulo 180"
    )

    frame_mask = built_documents["frame_mask"]
    assert len(frame_mask["records"]) == 180
    assert frame_mask["records"][0]["image"]["relative_path"].endswith(
        "img_0000.png"
    )
    assert frame_mask["records"][-1]["mask"]["relative_path"].endswith(
        "mask_0179.png"
    )

    metadata = built_documents["metadata"]
    assert len(metadata["frame_records"]) == 180
    assert metadata["frame_records"][0]["physical_state"]["theta_deg"] == 0.0
    assert metadata["frame_records"][-1]["physical_state"]["theta_deg"] == 358.0


def test_runtime_manifest_binds_every_executable_source(
    built_documents: dict,
) -> None:
    runtime = built_documents["runtime_source"]
    observed = [
        (record["role"], record["repo_relative_path"])
        for record in runtime["sources"]
    ]
    assert observed == [
        (
            "authorization_source",
            "keypoint_net/representation_checkpoint_authorization.py",
        ),
        (
            "runtime_source",
            "keypoint_net/representation_checkpoint_runtime.py",
        ),
        (
            "replay_registry_source",
            "keypoint_net/representation_checkpoint_replay.py",
        ),
        (
            "historical_comparison_source",
            "keypoint_net/representation_replay_comparison.py",
        ),
        ("model_source", "keypoint_net/model.py"),
        ("evaluator_source", "keypoint_net/eval_representation.py"),
        (
            "provenance_source",
            "keypoint_net/representation_evaluation_provenance.py",
        ),
        (
            "array_codec_source",
            "keypoint_net/representation_array_codec.py",
        ),
        (
            "manifest_builder_source",
            "keypoint_net/diagnostics/build_checkpoint_replay_manifests.py",
        ),
        (
            "corpus_inventory_source",
            "keypoint_net/representation_corpus_inventory.py",
        ),
        (
            "split_validator_source",
            "keypoint_net/representation_splits.py",
        ),
    ]
    assert observed == list(builder.RUNTIME_SOURCE_PATHS)
    assert all(
        record["import_time_digest_required"] is True
        for record in runtime["sources"]
    )
    assert runtime["execution_boundary"] == {
        "cpu_only": True,
        "weights_only": True,
        "same_open_file_descriptor_hash_and_load": True,
        "strict_state_dict": True,
        "model_eval": True,
        "inference_mode": True,
        "task_ids": [20, 55, 80],
        "training_or_weight_update_authorized": False,
        "selection_use_authorized": False,
    }


def test_rehashed_safe_frame_path_substitution_is_rejected(
    built_documents: dict,
) -> None:
    mutated = copy.deepcopy(built_documents)
    mutated["frame_mask"]["records"][0]["image"]["relative_path"] = (
        "train/engineers_hammer_vray/frames/a/img_0001.png"
    )
    _rehash(mutated["frame_mask"])
    with pytest.raises(
        builder.CheckpointReplayManifestError,
        match="wrong image path",
    ):
        builder.validate_manifest_shapes(mutated)


def test_rehashed_pair_semantic_substitution_is_rejected(
    built_documents: dict,
) -> None:
    mutated = copy.deepcopy(built_documents)
    mutated["pair_index"]["semantics"]["delta_theta_deg"] = -6.0
    _rehash(mutated["pair_index"])
    with pytest.raises(
        builder.CheckpointReplayManifestError,
        match="frozen semantics differ",
    ):
        builder.validate_manifest_shapes(mutated)


def test_unknown_manifest_field_is_rejected(
    built_documents: dict,
) -> None:
    mutated = copy.deepcopy(built_documents)
    mutated["metadata"]["unexpected"] = True
    _rehash(mutated["metadata"])
    with pytest.raises(
        builder.CheckpointReplayManifestError,
        match="keys differ",
    ):
        builder.validate_manifest_shapes(mutated)


def test_encoding_is_canonical_deterministic_and_checked_in(
    built_documents: dict,
) -> None:
    first = builder.encoded_manifest_documents(built_documents)
    second = builder.encoded_manifest_documents(
        json.loads(json.dumps(built_documents))
    )
    assert first == second
    assert all(value.endswith(b"\n") for value in first.values())
    assert all(
        value[:-1] == builder.canonical_json_bytes(
            builder.decode_strict_json(value, context=filename)
        )
        for filename, value in first.items()
    )
    checked = builder.check_checked_in_manifests(built_documents)
    assert set(checked) == set(builder.MANIFEST_FILENAMES.values())
