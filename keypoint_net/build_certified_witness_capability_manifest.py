"""Build the source-bound portable manifest for the supervised capability gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from certified_witness_capability import (
    EXPECTED_FRAMES,
    EXPECTED_WITNESSES,
    EXPECTED_WITNESS_IDS,
    file_record,
    pixel_to_normalized,
    require,
)


EXPECTED_SOURCE_HASHES = {
    "source_feature_manifest": "16e3af9e7afbe736370ca0cd0b0531fed0a622194741f478d7164e9ca61f60ec",
    "witness_arrays": "aa6844576546dd4cf90178cc783991381d362ca507613befe36d9e631767c5bd",
    "witness_summary": "693ae6b08f5bc46448c110575c4347322bda20ac51902669cee24df1bc00a017",
    "native_35_certificate": "75f4b0ccb3cec8f6d633578ba7cc7c07c355cdf1e968eeed6b4309c3e2647171",
    "frozen_feature_matrix_summary": "f71d0af14ee578f2d78a0c58a388ad7669efb1d6242133f12f1ec924e67c406d",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _bound_record(path: Path, expected_hash: str, name: str) -> dict[str, Any]:
    record = file_record(path)
    require(record["sha256"] == expected_hash, f"{name} SHA-256 differs")
    return record


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_clean(repo_root: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return status == ""


def build(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), "output directory already exists; use a fresh attempt")
    require(args.repo_root.is_dir(), "repository root missing")
    require(_git_clean(args.repo_root), "implementation worktree must be clean")

    sources = {
        "source_feature_manifest": _bound_record(
            args.source_feature_manifest,
            EXPECTED_SOURCE_HASHES["source_feature_manifest"],
            "source feature manifest",
        ),
        "witness_arrays": _bound_record(
            args.witness_arrays,
            EXPECTED_SOURCE_HASHES["witness_arrays"],
            "witness arrays",
        ),
        "witness_summary": _bound_record(
            args.witness_summary,
            EXPECTED_SOURCE_HASHES["witness_summary"],
            "witness summary",
        ),
        "native_35_certificate": _bound_record(
            args.native_35_certificate,
            EXPECTED_SOURCE_HASHES["native_35_certificate"],
            "native-35 certificate",
        ),
        "frozen_feature_matrix_summary": _bound_record(
            args.frozen_feature_matrix_summary,
            EXPECTED_SOURCE_HASHES["frozen_feature_matrix_summary"],
            "frozen feature matrix summary",
        ),
        "semantic_lock": file_record(args.semantic_lock),
    }
    source_feature_manifest = _load_json(args.source_feature_manifest)
    certificate = _load_json(args.native_35_certificate)
    frozen_summary = _load_json(args.frozen_feature_matrix_summary)
    require(
        tuple(certificate["certified_high_motion_witness"]["candidate_ids"])
        == EXPECTED_WITNESS_IDS,
        "certificate witness IDs differ",
    )
    require(certificate["certified_high_motion_witness"]["independent_check"]["passed"] is True, "certificate independent check failed")
    require(frozen_summary["decision"]["branch"] == "no_role_site_passes_exact_global_decoder_contract", "unexpected preceding decision branch")

    with np.load(args.witness_arrays) as arrays:
        frame_index = np.asarray(arrays["frame_index"], dtype=np.int64)
        candidate_id = np.asarray(arrays["candidate_id"], dtype=np.int64)
        selected_id = np.asarray(arrays["selected_candidate_id"], dtype=np.int64)
        require(tuple(selected_id.tolist()) == EXPECTED_WITNESS_IDS, "selected witness IDs differ")
        require(np.array_equal(frame_index, np.arange(EXPECTED_FRAMES)), "frame index differs")
        location = {int(value): index for index, value in enumerate(candidate_id)}
        columns = [location[value] for value in EXPECTED_WITNESS_IDS]
        # The source archive is edge-indexed: source_coordinate_px[f] belongs to
        # RGB frame f, while physical_target_coordinate_px[f] belongs to frame
        # f+1 modulo the cyclic orbit. Capability targets must align with the
        # input RGB frame, so use source_coordinate_px and prove the shift.
        target_px = np.asarray(arrays["source_coordinate_px"][:, columns], dtype=np.float64)
        next_frame_target_px = np.asarray(
            arrays["physical_target_coordinate_px"][:, columns], dtype=np.float64
        )
        physical_valid = np.asarray(arrays["physical_candidate_valid"][:, columns], dtype=bool)
    require(target_px.shape == (EXPECTED_FRAMES, EXPECTED_WITNESSES, 2), "target shape differs")
    require(bool(np.isfinite(target_px).all()), "target contains non-finite values")
    require(bool(physical_valid.all()), "a certified physical target is invalid")
    require(float(target_px.min()) >= 0.0 and float(target_px.max()) <= 511.0, "target is outside image")
    edge_cycle_alignment_error_px = float(
        np.max(np.abs(target_px - np.roll(next_frame_target_px, shift=1, axis=0)))
    )
    require(edge_cycle_alignment_error_px <= 1e-9, "source/next-frame edge alignment differs")

    frame_rows = source_feature_manifest["corpus"]["frames"]
    require(len(frame_rows) == EXPECTED_FRAMES, "source feature manifest frame count differs")
    object_root = args.data_object_root.resolve()
    frames: list[dict[str, Any]] = []
    masks = np.empty((EXPECTED_FRAMES, 512, 512), dtype=bool)
    for expected_frame, source_row in enumerate(frame_rows):
        require(int(source_row["frame_index"]) == expected_frame, "source frame order differs")
        image_relpath = Path(str(source_row["image_relpath"]))
        mask_relpath = Path("masks") / "a" / f"mask_{expected_frame:04d}.png"
        image_record = file_record(object_root / image_relpath)
        require(image_record["sha256"] == source_row["image_sha256"], f"frame {expected_frame} RGB SHA-256 differs")
        mask_record = file_record(object_root / mask_relpath)
        mask = np.asarray(Image.open(object_root / mask_relpath).convert("L")) > 0
        require(mask.shape == (512, 512), f"frame {expected_frame} mask shape differs")
        masks[expected_frame] = mask
        frames.append(
            {
                "frame_index": expected_frame,
                "image_relpath": str(image_relpath),
                "image_sha256": image_record["sha256"],
                "mask_relpath": str(mask_relpath),
                "mask_sha256": mask_record["sha256"],
            }
        )

    rounded = np.rint(target_px).astype(np.int64)
    target_on_object = masks[
        np.arange(EXPECTED_FRAMES)[:, None],
        rounded[..., 1],
        rounded[..., 0],
    ]
    require(bool(target_on_object.all()), "a certified target is off object at nearest pixel")
    pair_distance = np.linalg.norm(
        target_px[:, :, None, :] - target_px[:, None, :, :], axis=-1
    )
    pair_mask = np.triu(np.ones((EXPECTED_WITNESSES, EXPECTED_WITNESSES), dtype=bool), k=1)
    minimum_pair_distance = float(pair_distance[:, pair_mask].min())
    require(abs(minimum_pair_distance - 12.0) <= 1e-9, "minimum target separation differs")

    args.output_dir.mkdir(parents=True)
    tracks_path = args.output_dir / "CERTIFIED_WITNESS_TRACKS.npz"
    np.savez_compressed(
        tracks_path,
        frame_index=frame_index,
        witness_id=np.asarray(EXPECTED_WITNESS_IDS, dtype=np.int64),
        target_coordinate_px=target_px,
        target_coordinate_normalized=pixel_to_normalized(target_px),
        physical_valid=physical_valid,
        target_on_object=target_on_object,
    )
    tracks_record = file_record(tracks_path)

    implementation_files = (
        "keypoint_net/model.py",
        "keypoint_net/certified_witness_capability.py",
        "keypoint_net/build_certified_witness_capability_manifest.py",
        "keypoint_net/run_certified_witness_capability.py",
    )
    implementation = {
        relative: file_record(args.repo_root / relative) for relative in implementation_files
    }
    manifest = {
        "schema_version": "certified_witness_supervised_capability_manifest.v1",
        "artifact_type": "source_bound_portable_supervised_capability_manifest",
        "implementation_head": _git_head(args.repo_root),
        "implementation_sources": implementation,
        "sources": sources,
        "portable_tracks": tracks_record,
        "dataset": {
            "object_root_at_build_time": str(object_root),
            "frame_count": EXPECTED_FRAMES,
            "frames": frames,
            "portable_rebinding_rule": "data object root may differ only if every RGB and mask SHA-256 matches",
        },
        "witness": {
            "witness_ids": list(EXPECTED_WITNESS_IDS),
            "frame_count": EXPECTED_FRAMES,
            "minimum_physical_pair_distance_px": minimum_pair_distance,
            "edge_index_semantics": "source_coordinate_px[f] aligns with RGB frame f; physical_target_coordinate_px[f] aligns with RGB frame (f+1) mod 180",
            "source_equals_previous_edge_target_max_abs_error_px": edge_cycle_alignment_error_px,
            "all_targets_physical_valid": bool(physical_valid.all()),
            "all_targets_on_object_at_nearest_pixel": bool(target_on_object.all()),
        },
        "training_spec": {
            "architecture": {
                "class": "KeypointExtractor",
                "num_keypoints": EXPECTED_WITNESSES,
                "base_channels": 32,
                "temperature": 1.0,
                "padding_mode": "reflect",
                "heatmap_res": 64,
                "true_quarter_res": False,
            },
            "loss": "gaussian_target_distribution_cross_entropy_only",
            "sigma_input_px": 8.0,
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "train_frames": "all_180_exact_frames",
            "augmentation": "none",
            "maximum_updates": 5000,
            "evaluation_interval_updates": 100,
            "batch_size": 16,
            "seeds": [42, 43, 44],
        },
        "success_contract": {
            "cell_spacing_px": 511.0 / 63.0,
            "half_cell_diagonal_px": (511.0 / 63.0) / np.sqrt(2.0),
            "all_1800_within_half_cell_diagonal": True,
            "all_1800_on_object": True,
            "all_1800_closest_to_own_identity": True,
            "all_predicted_pairs_at_least_half_physical_pair_separation": True,
        },
        "statistical_scope": {
            "inference": "descriptive_only",
            "object_count": 1,
            "orbit_count": 1,
            "frame_values_independent": False,
            "replication_unit_if_matrix_runs": "optimization_seed",
            "planned_seed_count": 3,
        },
        "preservation_phase_authorized": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = {
        "manifest": file_record(manifest_path),
        "tracks": tracks_record,
        "implementation_head": manifest["implementation_head"],
        "semantic_controls_passed": True,
        "training_or_weight_update_performed": False,
    }
    (args.output_dir / "MANIFEST_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-feature-manifest", type=Path, required=True)
    parser.add_argument("--witness-arrays", type=Path, required=True)
    parser.add_argument("--witness-summary", type=Path, required=True)
    parser.add_argument("--native-35-certificate", type=Path, required=True)
    parser.add_argument("--frozen-feature-matrix-summary", type=Path, required=True)
    parser.add_argument("--semantic-lock", type=Path, required=True)
    parser.add_argument("--data-object-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
