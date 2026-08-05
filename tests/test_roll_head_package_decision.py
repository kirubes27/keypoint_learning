"""Focused synthetic tests for the frozen primary package decision."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from keypoint_net import roll_head_package_decision as decision


def _cell(
    recipe: str,
    head: str,
    seed: int,
    *,
    auc: float,
    drift: float,
    duplicates: int = 1,
    recurrent: int = 2,
    active: int = 8,
    angle: float = 0.2,
    eligible: bool = True,
) -> decision.CellEvidence:
    cell_id = f"{recipe}__r{head}__seed{seed}"
    return decision.CellEvidence(
        cell_id=cell_id,
        recipe=recipe,
        head_package=head,
        seed=seed,
        eligible=eligible,
        ineligibility_reasons=() if eligible else ("evaluator_critical_failure",),
        angle_error_deg=angle if eligible else None,
        validation_auc=auc if eligible else None,
        canonical_drift=drift if eligible else None,
        persistent_duplicate_count=duplicates if eligible else None,
        recurrent_duplicate_count=recurrent if eligible else None,
        active_on_object_count=active if eligible else None,
        source_commit="1" * 40,
        evaluation_config_sha256="4" * 64,
        result_content_sha256=("a" if head == "64" else "b") * 64,
    )


def _matrix(
    *,
    auc_wins: dict[str, set[int]] | None = None,
    drift_wins: dict[str, set[int]] | None = None,
) -> list[decision.CellEvidence]:
    auc_wins = auc_wins or {recipe: set(decision.PRIMARY_SEEDS) for recipe in decision.RECIPES}
    drift_wins = drift_wins or {recipe: set(decision.PRIMARY_SEEDS) for recipe in decision.RECIPES}
    cells = []
    for recipe in decision.RECIPES:
        for seed in decision.PRIMARY_SEEDS:
            cells.append(_cell(recipe, "64", seed, auc=1.0, drift=1.0))
            cells.append(_cell(
                recipe,
                "128",
                seed,
                auc=0.8 if seed in auc_wins[recipe] else 1.0,
                drift=0.8 if seed in drift_wins[recipe] else 1.0,
            ))
    return cells


def _evaluator_result(cell_id: str) -> dict:
    recipe, head_token, seed_token = cell_id.split("__")
    head = head_token.removeprefix("r")
    seed = int(seed_token.removeprefix("seed"))
    source_commit = "1" * 40
    result = {
        "schema_version": decision.RESULT_SCHEMA_VERSION,
        "case_id": cell_id,
        "case_kind": "checkpoint",
        "stratum": {
            "object_id": "engineers_hammer_vray",
            "seed": seed,
            "partition": "full_corpus",
            "transform_family": "roll",
            "direction": "forward",
            "stride": 3,
        },
        "transform": {
            "family": "roll",
            "physical_axis": "world_z",
            "direction": "forward",
            "stride": 3,
            "cyclic": True,
            "signed_generator": 6.0,
        },
        "evaluation_config": {
            "protocol": "full_primary_roll",
            "role_scoped_holdout_frame_ids": list(range(24)),
            "holdout_rollout_horizons": list(range(1, 8)),
            "full_rollout_horizons": list(range(1, 60)),
            "closure_horizon": 60,
        },
        "checkpoint_authorization": {
            "checkpoint_evaluation_authorized": True,
            "source_commit": source_commit,
            "cell_id": cell_id,
            "checkpoint_sha256": "2" * 64,
            "completed_run_receipt_sha256": "3" * 64,
            "training_or_weight_update_authorized": False,
            "selection_use_authorized": True,
        },
        "provenance": {
            "source_commit": source_commit,
            "committed_files": [],
            "external_files": [],
        },
        "critical_failures": [],
        "operator": {
            "wrong_locked_sign": False,
            "improper_or_reflection": False,
            "absolute_angle_error_deg": 0.2 if head == "64" else 0.1,
        },
        "collapse_evidence": {
            "structural_negative_control_status_v2_excluding_confirmed_flat_dead": (
                "available"
            ),
            "structural_negative_control_collapse_v2_excluding_confirmed_flat_dead": (
                False
            ),
        },
        "channel_health": {"active_on_object_count": 8},
        "trajectory_separation": {
            "eligible_channel_count": 8,
            "eligible_active_on_object": {
                "category_counts": {
                    "persistent_duplicate": 1,
                    "recurrent_close_pair": 2,
                }
            },
        },
        "rollout": {
            "role_scoped_holdout_identity_normalized_auc": {
                "value": 1.0 if head == "64" else 0.8,
            }
        },
        "canonical_drift": {
            "median_rms_objdiag": 1.0 if head == "64" else 0.8,
        },
    }
    result["evaluation_config_sha256"] = decision.canonical_sha256(
        result["evaluation_config"]
    )
    result["result_content_sha256"] = decision.canonical_sha256(result)
    return result


class RollHeadPackageDecisionTests(unittest.TestCase):
    def test_all_required_primary_conditions_adopt_128(self) -> None:
        result = decision.decide_primary_package(_matrix())
        self.assertEqual(result["decision"], "adopt_128")
        self.assertEqual(result["selected_head_package"], "128")
        self.assertFalse(result["extension_required"])

    def test_exactly_two_of_three_required_wins_triggers_only_extension(self) -> None:
        two = {recipe: {42, 43} for recipe in decision.RECIPES}
        result = decision.decide_primary_package(
            _matrix(auc_wins=two, drift_wins=two)
        )
        self.assertEqual(result["decision"], "run_extension_45_46")
        self.assertIsNone(result["selected_head_package"])
        self.assertEqual(result["extension_seeds"], [45, 46])

    def test_insufficient_required_axis_or_guardrail_retains_64(self) -> None:
        one = {recipe: {42} for recipe in decision.RECIPES}
        result = decision.decide_primary_package(_matrix(auc_wins=one))
        self.assertEqual(result["decision"], "retain_64")
        two = {recipe: {42, 43} for recipe in decision.RECIPES}
        cells = _matrix(auc_wins=two, drift_wins=two)
        for index, cell in enumerate(cells):
            if (
                cell.recipe == "task55_clean"
                and cell.head_package == "128"
                and cell.seed in {42, 43}
            ):
                cells[index] = _cell(
                    cell.recipe, cell.head_package, cell.seed,
                    auc=0.8, drift=0.8, duplicates=3, active=7,
                )
        self.assertEqual(
            decision.decide_primary_package(cells)["decision"], "retain_64"
        )

    def test_ineligible_cell_retains_64_but_incomplete_or_duplicate_matrix_rejects(self) -> None:
        cells = _matrix()
        failed = cells[0]
        cells[0] = _cell(
            failed.recipe, failed.head_package, failed.seed,
            auc=1.0, drift=1.0, eligible=False,
        )
        result = decision.decide_primary_package(cells)
        self.assertEqual(result["decision"], "retain_64")
        self.assertIn(failed.cell_id, result["ineligible_cells"])
        with self.assertRaisesRegex(decision.PackageDecisionError, "cell set differs"):
            decision.decide_primary_package(cells[:-1])
        with self.assertRaisesRegex(decision.PackageDecisionError, "duplicate cell"):
            decision.decide_primary_package([*cells, cells[-1]])

    def test_hashed_fresh_evaluator_result_extracts_and_tamper_rejects(self) -> None:
        result = _evaluator_result("task80_assisted__r128__seed44")
        cell = decision.extract_cell_evidence(result)
        self.assertTrue(cell.eligible)
        self.assertEqual((cell.recipe, cell.head_package, cell.seed),
                         ("task80_assisted", "128", 44))
        tampered = copy.deepcopy(result)
        tampered["operator"]["absolute_angle_error_deg"] = 9.0
        with self.assertRaisesRegex(decision.PackageDecisionError, "content hash differs"):
            decision.extract_cell_evidence(tampered)

    def test_cli_consumes_exactly_twelve_results_and_writes_exclusively(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="roll_head_decision_test_"))
        arguments: list[str] = []
        for recipe in decision.RECIPES:
            for head in decision.HEAD_PACKAGES:
                for seed in decision.PRIMARY_SEEDS:
                    cell_id = f"{recipe}__r{head}__seed{seed}"
                    path = root / f"{cell_id}.json"
                    path.write_text(json.dumps(_evaluator_result(cell_id)))
                    arguments.extend(("--result", str(path)))
        output = root / "PRIMARY_PACKAGE_DECISION.json"
        self.assertEqual(decision.main([*arguments, "--output", str(output)]), 0)
        written = json.loads(output.read_text())
        self.assertEqual(written["decision"], "adopt_128")
        self.assertEqual(len(written["input_files"]), 12)
        with self.assertRaisesRegex(
            decision.PackageDecisionError, "exclusively create"
        ):
            decision.main([*arguments, "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
