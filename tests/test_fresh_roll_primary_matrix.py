"""Focused tests for the frozen primary matrix orchestration.

The project deletion policy forbids automatic cleanup, so these tests leave
their small uniquely named fixtures below the operating-system temp root.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from keypoint_net import run_fresh_roll_primary_matrix as matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "_tdw_world_z_roll_base_panel_512_v2"
SLURM_SCRIPT = REPO_ROOT / "cluster/fresh_roll_primary_matrix.slurm"
SMOKE_SLURM_SCRIPT = REPO_ROOT / "cluster/fresh_roll_cuda_smoke.slurm"
COMMIT = "a" * 40


def _prepared_matrix() -> tuple[Path, dict]:
    parent = Path(tempfile.mkdtemp(prefix="fresh_primary_matrix_test_"))
    environment_lock = parent / "FRESH_ROLL_ENVIRONMENT.json"
    environment_lock.write_text("{}\n", encoding="utf-8")
    root = parent / "matrix"
    with mock.patch.object(matrix, "_source_state"):
        lock_path = matrix.prepare_primary_matrix(
            matrix_root=root,
            expected_commit=COMMIT,
            slurm_script=SLURM_SCRIPT,
            environment_lock=environment_lock,
        )
    return root, json.loads(lock_path.read_text(encoding="utf-8"))


class FreshRollPrimaryMatrixTests(unittest.TestCase):
    def test_task_indices_map_to_the_exact_twelve_primary_cells(self) -> None:
        expected = [
            f"{recipe}__r{head}__seed{seed}"
            for recipe in ("task55_clean", "task80_assisted")
            for head in (64, 128)
            for seed in (42, 43, 44)
        ]
        self.assertEqual(
            [matrix.primary_cell_id(index) for index in range(12)], expected
        )
        for invalid in (-1, 12, True):
            with self.assertRaises(matrix.PrimaryMatrixError):
                matrix.primary_cell_id(invalid)

    def test_prepare_is_non_submitting_exact_and_exclusive(self) -> None:
        root, lock = _prepared_matrix()
        self.assertEqual(lock["primary_cell_ids_by_task_index"], list(matrix.PRIMARY_CELL_IDS))
        self.assertEqual(lock["task_count"], 12)
        self.assertEqual(lock["gpu_count_per_task"], 1)
        self.assertEqual(lock["max_concurrent_gpu_nodes"], 1)
        self.assertFalse(lock["submission"]["performed_by_prepare_command"])
        self.assertEqual(list((root / "runs").iterdir()), [])
        self.assertEqual(list((root / "evaluations").iterdir()), [])
        with mock.patch.object(matrix, "_source_state"):
            with self.assertRaisesRegex(matrix.PrimaryMatrixError, "must be new"):
                matrix.prepare_primary_matrix(
                    matrix_root=root,
                    expected_commit=COMMIT,
                    slurm_script=SLURM_SCRIPT,
                    environment_lock=Path(lock["environment_lock"]["absolute_path"]),
                )

    def test_matrix_lock_rejects_a_wrong_script_hash(self) -> None:
        root, lock = _prepared_matrix()
        matrix.validate_primary_matrix(
            root,
            expected_commit=COMMIT,
            slurm_script_sha256=lock["slurm_script"]["sha256"],
            environment_lock_sha256=lock["environment_lock"]["sha256"],
        )
        with self.assertRaisesRegex(matrix.PrimaryMatrixError, "Slurm script hash"):
            matrix.validate_primary_matrix(
                root,
                expected_commit=COMMIT,
                slurm_script_sha256="0" * 64,
            )

    def test_task_commands_bind_cell_paths_without_public_scientific_overrides(self) -> None:
        root, _ = _prepared_matrix()
        cell_id, train_command, evaluation_command = matrix.build_task_commands(
            task_index=5,
            data_root=DATA_ROOT,
            matrix_root=root,
        )
        self.assertEqual(cell_id, "task55_clean__r128__seed44")
        self.assertIn("--fresh_cell_id", train_command)
        self.assertEqual(train_command[train_command.index("--fresh_cell_id") + 1], cell_id)
        self.assertEqual(train_command[train_command.index("--heatmap_res") + 1], "128")
        self.assertEqual(train_command[train_command.index("--seed") + 1], "44")
        self.assertEqual(
            evaluation_command[-1],
            str(root.resolve() / "evaluations" / f"{cell_id}.json"),
        )
        task_help = matrix._parser().parse_args([
            "task", "--task-index", "5", "--data-root", str(DATA_ROOT),
            "--matrix-root", str(root), "--expected-commit", COMMIT,
            "--slurm-script-sha256", "0" * 64,
            "--environment-lock-sha256", "1" * 64,
        ])
        self.assertFalse(hasattr(task_help, "heatmap_res"))
        self.assertFalse(hasattr(task_help, "seed"))
        for expected_cell in matrix.PRIMARY_CELL_IDS:
            (root / "evaluations" / f"{expected_cell}.json").write_text(
                "{}\n", encoding="utf-8"
            )
        decision_output = root / "PRIMARY_PACKAGE_DECISION.json"
        with (
            mock.patch.object(matrix, "_source_state"),
            mock.patch.object(matrix.package_decision, "main", return_value=0) as decide,
        ):
            self.assertEqual(
                matrix.run_primary_decision(
                    matrix_root=root, output=decision_output
                ),
                0,
            )
        decision_arguments = decide.call_args.args[0]
        self.assertEqual(decision_arguments.count("--result"), 12)
        self.assertEqual(decision_arguments[-2:], ["--output", str(decision_output)])

    def test_evaluation_writes_once_only_after_decision_schema_validation(self) -> None:
        parent = Path(tempfile.mkdtemp(prefix="fresh_primary_eval_test_"))
        output = parent / "cell.json"
        cell_id = "task55_clean__r64__seed42"
        result = {"case_id": cell_id, "result_content_sha256": "0" * 64}
        evidence = mock.Mock(cell_id=cell_id)
        with mock.patch.object(
            matrix.package_decision, "extract_cell_evidence", return_value=evidence
        ) as validation:
            matrix.evaluate_cell(
                receipt=parent / "receipt.json",
                cell_id=cell_id,
                data_root=DATA_ROOT,
                output=output,
                evaluator_fn=mock.Mock(return_value=result),
            )
        validation.assert_called_once_with(result)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
        with self.assertRaisesRegex(matrix.PrimaryMatrixError, "already exists"):
            matrix.evaluate_cell(
                receipt=parent / "receipt.json",
                cell_id=cell_id,
                data_root=DATA_ROOT,
                output=output,
                evaluator_fn=mock.Mock(return_value=result),
            )

    def test_slurm_array_requests_one_gpu_node_at_a_time_and_is_valid_bash(self) -> None:
        text = SLURM_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-11%1", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertNotIn("#SBATCH -A", text)
        self.assertNotIn("sbatch ", text)
        self.assertIn("srun python -B -u", text)
        for script in (SLURM_SCRIPT, SMOKE_SLURM_SCRIPT):
            script_text = script.read_text(encoding="utf-8")
            self.assertIn("module purge", script_text)
            self.assertNotRegex(script_text, r"module(?:\s+--ignore_cache)?\s+load")
            self.assertIn('source "$VENV_DIR/bin/activate"', script_text)
            subprocess.run(["bash", "-n", str(script)], check=True)
        self.assertEqual(len(hashlib.sha256(SLURM_SCRIPT.read_bytes()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
