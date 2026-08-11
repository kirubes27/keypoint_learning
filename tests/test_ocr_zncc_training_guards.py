"""Focused CPU guards for the bounded OCR-ZNCC training path.

These tests use in-memory authorization documents and do not create, delete,
or modify experiment evidence.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
KEYPOINT_NET_ROOT = REPO_ROOT / "keypoint_net"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(KEYPOINT_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(KEYPOINT_NET_ROOT))

from keypoint_net import ocr_zncc_training_authorization as authorization
from keypoint_net import ocr_zncc_training_decision as task55_decision
from keypoint_net import ocr_zncc_task80_training_authorization as task80_decision
from keypoint_net import run_ocr_zncc_paired_training as paired_training
from keypoint_net import train as train_module


_MANIFEST_FILE_SHA256 = "a" * 64
_DECISION_FILE_SHA256 = "b" * 64
_SOURCE_COMMIT = "1" * 40
_COEFFICIENT = 0.314


class _MarkerModel(nn.Module):
    """Return the batch marker needed by the synthetic loss function."""

    def forward(
        self,
        x_t: torch.Tensor,
        x_t1: torch.Tensor,
        *,
        return_descriptor_features: bool,
    ) -> dict[str, torch.Tensor]:
        del x_t1, return_descriptor_features
        return {"marker": x_t.mean()}


def _batch(sample_count: int, marker: float) -> dict[str, torch.Tensor]:
    return {
        "x_t": torch.full((sample_count, 1), marker),
        "x_t1": torch.zeros((sample_count, 1)),
        "action_label": torch.zeros(sample_count, dtype=torch.long),
    }


def _marker_losses(outputs: dict[str, torch.Tensor], **_: object) -> dict[str, torch.Tensor]:
    marker = outputs["marker"]
    zero = marker * 0.0
    return {
        "loss": marker,
        "base_loss": marker,
        "attachment_loss": zero,
        "l_pred": marker,
        "l_smooth": zero,
        "l_disp": zero,
        "l_ent": zero,
        "l_act": zero,
        "l_loc": zero,
        "l_inv": zero,
        "l_cycle": zero,
        "act_acc": zero,
    }


def _evaluate(loader: list[dict[str, torch.Tensor]]) -> dict:
    with mock.patch.object(train_module, "compute_losses", side_effect=_marker_losses):
        return train_module.evaluate(
            _MarkerModel(),
            loader,
            torch.device("cpu"),
            lambda_smooth=0.0,
            lambda_disp=0.1,
            lambda_ent=0.01,
            lambda_act=0.0,
            lambda_loc=0.0,
            lambda_inv=0.0,
            lambda_cycle=0.0,
            loc_bg_threshold=30.0,
            sigma=0.1,
            num_keypoints=10,
            lambda_attach=0.0,
            lambda_ocr_zncc=0.0,
            ocr_zncc_config=None,
        )


def _content_bound(document: dict) -> dict:
    result = copy.deepcopy(document)
    result.pop("content_hash_sha256", None)
    result["content_hash_sha256"] = authorization.canonical_sha256(result)
    return result


def _validation_row(epoch: int, *, arm: str) -> dict:
    accepted = 0 if arm == "control" else 84
    possible = 0 if arm == "control" else 210
    return {
        "epoch": epoch,
        "val_loss": 1.0,
        "val_train_loss": 1.0,
        "val_base_loss": 1.0,
        "val_attachment_loss": 0.0,
        "val_ocr_zncc_loss": 0.0,
        "val_ocr_zncc_accepted_match_count": accepted,
        "val_ocr_zncc_possible_match_count": possible,
        "val_ocr_zncc_accepted_match_coverage": (
            None if arm == "control" else accepted / possible
        ),
        "val_ocr_zncc_zero_accept_batch_count": 0,
        "val_pred": 1.0,
        "val_act": 0.0,
        "val_loc": 0.0,
        "val_inv": 0.0,
        "val_cycle": 0.0,
        "val_act_acc": 0.0,
        "val_aggregation": {
            "sample_count": 21,
            "batch_count": 2,
            "base_and_standard_metrics": "sample_weighted_complete_loader_mean",
            "ocr_zncc_loss": "accepted_match_weighted_complete_loader_mean",
        },
    }


def _history_binding(*, arm: str, epochs: int) -> authorization.OCRTrainingBinding:
    return authorization.OCRTrainingBinding(
        cell_id=f"task55_clean__{arm}__seed42",
        recipe="task55_clean",
        arm=arm,
        run_directory="/unused",
        manifest_absolute_path="/unused/manifest.json",
        manifest_file_sha256=_MANIFEST_FILE_SHA256,
        manifest_content_sha256="c" * 64,
        expected_training_arguments={"epochs": epochs, "log_every": 10},
    )


def _authorization_fixture(
    *,
    recipe: str = "task55_clean",
    arm: str = "control",
    seed: int = 42,
) -> tuple[argparse.Namespace, dict, dict]:
    cell_id = f"{recipe}__{arm}__seed{seed}"
    expected = {
        **authorization.FIXED_TRAINING_ARGUMENTS,
        **authorization.TASK_RECIPES[recipe],
        "data_root": "/private/tmp/ocr_zncc_fixture_data",
        "train_pairs_index": "TRAIN_INDEX",
        "val_pairs_index": "VALIDATION_INDEX",
        "seed": seed,
        "output_dir": (
            f"/private/tmp/ocr_zncc_authorization_{uuid.uuid4().hex}/runs"
        ),
        "lambda_ocr_zncc": 0.0 if arm == "control" else _COEFFICIENT,
    }
    if set(expected) != authorization.BOUND_ARGUMENTS:
        raise AssertionError("test fixture argument set differs from authorization contract")

    decision_schema, decision_status = {
        "task55_clean": (
            "ocr_zncc_task55_training_authorization.v1",
            "authorize_task55_matched_experiment",
        ),
        "task80_assisted": (
            "ocr_zncc_task80_training_authorization.v1",
            "authorize_task80_matched_experiment",
        ),
    }[recipe]
    decision = _content_bound(
        {
            "schema_version": decision_schema,
            "status": decision_status,
            "source_commit": _SOURCE_COMMIT,
            "source_branch": authorization.EXPECTED_BRANCH,
            "coefficient": _COEFFICIENT,
            "ocr_zncc_config": {
                "peak_margin_exclusion_radius_cells": 1,
            },
        }
    )
    manifest = _content_bound(
        {
            "schema_version": authorization.EXPERIMENT_SCHEMA_VERSION,
            "source_commit": _SOURCE_COMMIT,
            "source_branch": authorization.EXPECTED_BRANCH,
            "authorization_decision": {
                "absolute_path": "DECISION",
                "sha256": _DECISION_FILE_SHA256,
            },
            "ocr_zncc": {"coefficient": _COEFFICIENT},
            "cells": [
                {
                    "cell_id": cell_id,
                    "recipe": recipe,
                    "arm": arm,
                    "seed": seed,
                    "stage": (
                        "task55_primary"
                        if recipe == "task55_clean"
                        else "task80_conditional"
                    ),
                    "training_arguments": expected,
                }
            ],
        }
    )
    args = argparse.Namespace(
        **expected,
        ocr_cell_id=cell_id,
        ocr_experiment_manifest="MANIFEST",
        ocr_experiment_manifest_sha256=_MANIFEST_FILE_SHA256,
    )
    return args, manifest, decision


def _bind(
    args: argparse.Namespace,
    manifest: dict,
    decision: dict,
    *,
    reject_deep_task55: bool = False,
) -> authorization.OCRTrainingBinding:
    manifest = _content_bound(manifest)
    decision = _content_bound(decision)

    def load_json(path_value: str | Path, *, name: str):
        del name
        if str(path_value) == "MANIFEST":
            return manifest, {
                "absolute_path": "/private/tmp/in_memory_ocr_manifest.json",
                "sha256": _MANIFEST_FILE_SHA256,
                "size_bytes": 1,
            }
        if str(path_value) == "DECISION":
            return decision, {
                "absolute_path": "/private/tmp/in_memory_ocr_decision.json",
                "sha256": _DECISION_FILE_SHA256,
                "size_bytes": 1,
            }
        raise AssertionError(f"unexpected JSON path {path_value!r}")

    def file_record(path_value: str | Path, *, name: str):
        del name
        digest = {
            "TRAIN_INDEX": authorization.TRAIN_INDEX_SHA256,
            "VALIDATION_INDEX": authorization.VALIDATION_INDEX_SHA256,
        }.get(str(path_value))
        if digest is None:
            raise AssertionError(f"unexpected file path {path_value!r}")
        return {
            "absolute_path": f"/private/tmp/{path_value}",
            "sha256": digest,
            "size_bytes": 1,
        }

    with (
        mock.patch.object(
            authorization,
            "_source_state",
            return_value=(_SOURCE_COMMIT, authorization.EXPECTED_BRANCH),
        ),
        mock.patch.object(authorization, "_load_json", side_effect=load_json),
        mock.patch.object(authorization, "_file_record", side_effect=file_record),
        mock.patch.object(
            task55_decision,
            "validate_authorization_document",
            side_effect=(
                ValueError("fabricated whitelisted Task55 decision")
                if reject_deep_task55 else None
            ),
            return_value={
                "authorization_valid": True,
                "status": "authorize_task55_matched_experiment",
                "source_commit": _SOURCE_COMMIT,
                "source_branch": authorization.EXPECTED_BRANCH,
                "coefficient": _COEFFICIENT,
            },
        ),
        mock.patch.object(
            task80_decision,
            "validate_authorization_document",
            return_value={
                "authorization_valid": True,
                "status": "authorize_task80_matched_experiment",
                "source_commit": _SOURCE_COMMIT,
                "source_branch": authorization.EXPECTED_BRANCH,
                "coefficient": _COEFFICIENT,
            },
        ),
    ):
        return authorization.bind_training_namespace(REPO_ROOT, args)


class EvaluationAggregationTests(unittest.TestCase):
    def test_validation_metrics_weight_the_16_plus_5_split_by_sample(self) -> None:
        result = _evaluate([_batch(16, 1.0), _batch(5, 9.0)])
        expected = (16.0 * 1.0 + 5.0 * 9.0) / 21.0

        self.assertAlmostEqual(result["loss"], expected, places=7)
        self.assertAlmostEqual(result["base_loss"], expected, places=7)
        self.assertAlmostEqual(result["l_pred"], expected, places=7)
        self.assertNotAlmostEqual(result["loss"], (1.0 + 9.0) / 2.0, places=7)
        self.assertEqual(result["aggregation"]["sample_count"], 21)
        self.assertEqual(result["aggregation"]["batch_count"], 2)
        self.assertEqual(
            result["aggregation"]["base_and_standard_metrics"],
            "sample_weighted_complete_loader_mean",
        )

    def test_zero_ocr_weight_keeps_the_exact_base_control_path(self) -> None:
        base = torch.tensor(2.5, requires_grad=True)
        auxiliary = torch.tensor(9.0, requires_grad=True)
        combined = train_module.combine_with_base_loss(base, auxiliary, weight=0.0)
        self.assertIs(combined, base)

        with (
            mock.patch.object(
                train_module,
                "ocr_zncc_transport_loss",
                side_effect=AssertionError("zero-weight control executed matcher"),
            ),
            mock.patch.object(
                train_module,
                "combine_with_base_loss",
                side_effect=AssertionError("zero-weight control combined auxiliary"),
            ),
        ):
            result = _evaluate([_batch(4, 3.0)])
        self.assertEqual(result["loss"], 3.0)
        self.assertEqual(result["base_loss"], 3.0)
        self.assertEqual(result["ocr_zncc_accepted_match_count"], 0)
        self.assertEqual(result["ocr_zncc_possible_match_count"], 0)


class CompletedHistoryValidationTests(unittest.TestCase):
    def test_authentic_smoke_control_aggregation_accepts_none_ocr_coverage(self) -> None:
        rows = [_validation_row(1, arm="control")]
        observed = authorization._validate_completed_validation_history(
            rows,
            _history_binding(arm="control", epochs=1),
        )
        self.assertEqual(observed, rows)

    def test_authentic_scientific_ocr_epoch_sequence_and_aggregation_pass(self) -> None:
        epochs = [1, *range(10, 1001, 10)]
        rows = [_validation_row(epoch, arm="ocr_zncc") for epoch in epochs]
        observed = authorization._validate_completed_validation_history(
            rows,
            _history_binding(arm="ocr_zncc", epochs=1000),
        )
        self.assertEqual([row["epoch"] for row in observed], epochs)

    def test_missing_accepted_weighted_aggregation_label_fails_closed(self) -> None:
        row = _validation_row(1, arm="control")
        row["val_aggregation"].pop("ocr_zncc_loss")
        with self.assertRaisesRegex(ValueError, "validation aggregation differs"):
            authorization._validate_completed_validation_history(
                [row],
                _history_binding(arm="control", epochs=1),
            )


class AuthorizationIdentityTests(unittest.TestCase):
    def test_valid_in_memory_contract_binds(self) -> None:
        args, manifest, decision = _authorization_fixture()
        binding = _bind(args, manifest, decision)
        self.assertEqual(binding.cell_id, args.ocr_cell_id)
        self.assertEqual(binding.recipe, "task55_clean")
        self.assertEqual(binding.arm, "control")

    def test_recipe_arm_and_seed_mismatches_fail_closed(self) -> None:
        mutations = {
            "recipe": lambda cell: cell.update(recipe="task80_assisted"),
            "arm": lambda cell: cell.update(arm="ocr_zncc"),
            "seed": lambda cell: cell.update(seed=43),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                args, manifest, decision = _authorization_fixture()
                mutate(manifest["cells"][0])
                with self.assertRaisesRegex(
                    ValueError,
                    "OCR cell ID (?:recipe/arm|seed) differs from payload",
                ):
                    _bind(args, manifest, decision)

    def test_coefficient_mismatch_fails_closed(self) -> None:
        args, manifest, decision = _authorization_fixture(
            arm="ocr_zncc",
        )
        manifest["ocr_zncc"]["coefficient"] = _COEFFICIENT + 0.001
        with self.assertRaisesRegex(
            ValueError,
            "OCR manifest coefficient differs from authorization",
        ):
            _bind(args, manifest, decision)

    def test_arbitrary_decision_status_cannot_authorize_training(self) -> None:
        args, manifest, decision = _authorization_fixture()
        decision["status"] = "proceed_because_it_seems_good"
        with self.assertRaisesRegex(
            ValueError,
            "OCR authorization decision does not authorize task55_clean",
        ):
            _bind(args, manifest, decision)

    def test_whitelisted_status_cannot_bypass_deep_task55_validation(self) -> None:
        args, manifest, decision = _authorization_fixture()
        self.assertEqual(decision["status"], "authorize_task55_matched_experiment")
        with self.assertRaisesRegex(
            ValueError,
            "fabricated whitelisted Task55 decision",
        ):
            _bind(
                args,
                manifest,
                decision,
                reject_deep_task55=True,
            )


class PairWrapperRuntimeTests(unittest.TestCase):
    def test_scientific_pair_uses_one_wrapper_and_exact_arm_positions(self) -> None:
        recipe = "task55_clean"
        seed = 42
        output_root = Path(f"/private/tmp/ocr_pair_{uuid.uuid4().hex}")
        cells = [
            {
                "cell_id": f"{recipe}__{arm}__seed{seed}",
                "training_arguments": {"output_dir": str(output_root)},
            }
            for arm in ("control", "ocr_zncc")
        ]
        manifest = {
            "decision_lock": {"authorized_recipe": recipe},
            "cells": cells,
        }
        with (
            mock.patch.object(paired_training, "_load_manifest", return_value=manifest),
            mock.patch.object(paired_training, "_load_stage_lock", return_value={}),
            mock.patch.object(
                paired_training,
                "_cell_command",
                side_effect=lambda **kwargs: ["python", kwargs["cell"]["cell_id"]],
            ),
            mock.patch.object(paired_training.subprocess, "run") as run_process,
        ):
            paired_training.run_pair(
                manifest_path=Path("/private/tmp/manifest.json"),
                manifest_sha256="a" * 64,
                recipe=recipe,
                seed=seed,
                stage_lock_path=Path("/private/tmp/stage.json"),
                stage_lock_sha256="b" * 64,
                print_only=False,
            )
        self.assertEqual(run_process.call_count, 2)
        environments = [call.kwargs["env"] for call in run_process.call_args_list]
        self.assertEqual(
            [environment["OCR_PAIR_POSITION"] for environment in environments],
            ["1_control", "2_ocr_zncc"],
        )
        self.assertEqual(
            {environment["OCR_PAIR_ID"] for environment in environments},
            {f"{recipe}__seed{seed}"},
        )
        self.assertEqual(
            len({environment["OCR_PAIR_WRAPPER_PID"] for environment in environments}),
            1,
        )
        self.assertTrue(all(call.kwargs["check"] for call in run_process.call_args_list))

    def test_runtime_pair_binding_rejects_multi_gpu_visibility(self) -> None:
        provenance = {
            "stage_launch_lock": {"sha256": "1" * 64},
            "environment_lock": {"sha256": "2" * 64},
            "slurm_script": {"sha256": "3" * 64},
        }
        environment = {
            "SLURM_JOB_ID": "12345",
            "OCR_PAIR_ID": "task55_clean__seed42",
            "OCR_PAIR_POSITION": "1_control",
            "OCR_PAIR_WRAPPER_PID": str(os.getpid()),
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
        with mock.patch.dict(os.environ, environment, clear=False), self.assertRaisesRegex(
            RuntimeError,
            "single-GPU sequential pair-wrapper binding",
        ):
            train_module._runtime_environment(  # noqa: SLF001
                torch.device("cpu"),
                provenance_files=provenance,
                ocr=True,
            )


if __name__ == "__main__":
    unittest.main()
