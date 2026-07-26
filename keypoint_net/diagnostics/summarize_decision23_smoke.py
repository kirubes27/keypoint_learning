"""Fail-closed D1 smoke summarizer for the Decision 2.3 three-arm panel."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
KEYPOINT_ROOT = HERE.parent
sys.path.insert(0, str(KEYPOINT_ROOT))

from diagnostics.decision23_diagnostic_head import (  # noqa: E402
    ARMS,
    SMOKE_RECIPE,
    assert_json_finite,
    sha256_file,
    slurm_runfile_hashes,
    source_dependency_hashes,
    write_json,
)
from diagnostics.stage_a_supervised_control import read_history  # noqa: E402


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def summarize(output_root: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    rows: list[dict[str, Any]] = []
    commits: set[str] = set()
    source_hashes: set[str] = set()
    spec_hashes: set[str] = set()
    dependency_hashes: list[dict[str, str]] = []
    slurm_runfile_hash_sets: list[dict[str, str]] = []
    runtimes: list[dict[str, Any]] = []
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError("D1 summarization must run inside the smoke Slurm job")
    for arm in ARMS:
        run_dir = (
            output_root
            / "smoke"
            / "runs"
            / f"e2e_{arm}_standard64_k10_seed42"
        )
        required = {
            "config": run_dir / "config.json",
            "summary": run_dir / "training_summary.json",
            "history": run_dir / "history.csv",
            "audit": run_dir / "gradient_audit_history.json",
            "best": run_dir / "best_model.pt",
            "initial_validation": run_dir / "initial_validation_metrics.json",
            "best_validation": run_dir / "best_validation_metrics.json",
            "restore": run_dir / "smoke_checkpoint_restore.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"D1 smoke artifacts missing: {missing}")
        forbidden = [
            str(path)
            for path in (
                run_dir / "test_metrics.json",
                run_dir / "test_predictions.npz",
            )
            if path.exists()
        ]
        if forbidden:
            raise RuntimeError(f"D1 smoke accessed test: {forbidden}")
        config = json.loads(required["config"].read_text())
        summary = json.loads(required["summary"].read_text())
        initial_validation = json.loads(
            required["initial_validation"].read_text()
        )
        best_validation = json.loads(required["best_validation"].read_text())
        audit_payload = json.loads(required["audit"].read_text())
        restore = json.loads(required["restore"].read_text())
        history = read_history(required["history"])
        if config["arm"] != arm or int(config["seed"]) != 42:
            raise RuntimeError(f"D1 identity mismatch: {run_dir}")
        if config["run_scope"] != "smoke" or config["freeze_backbone"]:
            raise RuntimeError(f"D1 accepted a non-smoke run: {run_dir}")
        observed_recipe = {
            "batch_size": config["batch_size"],
            "lr": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "min_epochs": config["min_epochs"],
            "max_epochs": config["max_epochs"],
            "eval_every": config["eval_every"],
            "plateau_patience": config["plateau_patience_epochs"],
            "relative_improvement": config["relative_improvement"],
        }
        if observed_recipe != SMOKE_RECIPE:
            raise RuntimeError(f"D1 recipe mismatch: {observed_recipe}")
        if not str(config["device"]).startswith("cuda"):
            raise RuntimeError(f"D1 did not use CUDA: {run_dir}")
        if config.get("slurm", {}).get("slurm_job_id") != slurm_job_id:
            raise RuntimeError(f"D1 Slurm job binding mismatch: {run_dir}")
        if set(config["loaded_frame_indices"]) & set(
            config["test_frames_committed_not_evaluated"]
        ):
            raise RuntimeError(f"D1 loaded a test frame: {run_dir}")
        if not summary.get("split_access_assertion") or summary.get("test_evaluated"):
            raise RuntimeError(f"D1 split assertion failed: {run_dir}")
        if restore.get("status") != "pass":
            raise RuntimeError(f"D1 checkpoint restore failed: {run_dir}")
        if [int(row["epoch"]) for row in history] != [1, 2]:
            raise RuntimeError(f"D1 history must contain exactly epochs 1 and 2: {run_dir}")
        if int(summary.get("completed_epoch", -1)) != 2:
            raise RuntimeError(f"D1 did not complete exactly two epochs: {run_dir}")
        if not math.isclose(
            float(best_validation["selection_score_cells64"]),
            float(summary["best_validation_score_cells64"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"D1 selected validation metric mismatch: {run_dir}")
        records = audit_payload.get("records", [])
        if {int(record["epoch"]) for record in records} != {0, 1, 2}:
            raise RuntimeError(f"D1 audit epochs differ from 0/1/2: {run_dir}")
        if not _all_finite(
            {
                "summary": summary,
                "history": history,
                "initial_validation": initial_validation,
                "best_validation": best_validation,
                "audits": records,
                "restore": restore,
            }
        ):
            raise FloatingPointError(f"D1 contains non-finite values: {run_dir}")
        for record in records:
            audit = record["audit"]
            if audit["pooled_logit_gradient_l2_norm"] <= 0:
                raise RuntimeError(f"D1 zero logit gradient: {run_dir}")
            expected = (
                "encoder_first_conv_weight",
                "encoder_final_conv_weight",
                "heatmap_head_weight",
            )
            if any(audit["parameter_gradient_norms"][name] <= 0 for name in expected):
                raise RuntimeError(f"D1 zero upstream gradient: {run_dir}")
        commits.add(config["git_commit"])
        source_hashes.add(config["source_sha256"])
        spec_hashes.add(config["decision_spec_sha256"])
        dependency_hashes.append(config["source_dependencies_sha256"])
        slurm_runfile_hash_sets.append(config["slurm_runfiles_sha256"])
        runtimes.append(config["runtime"])
        rows.append(
            {
                "arm": arm,
                "seed": 42,
                "run_dir": str(run_dir.resolve()),
                "config_sha256": sha256_file(required["config"]),
                "checkpoint_sha256": sha256_file(required["best"]),
                "completed_epoch": int(summary["completed_epoch"]),
                "best_epoch": int(summary["best_epoch"]),
                "checkpoint_restore": "pass",
                "test_evaluated": False,
            }
        )
    if len(commits) != 1 or len(source_hashes) != 1 or len(spec_hashes) != 1:
        raise RuntimeError("D1 arms do not share one frozen commit/source/spec")
    if any(item != dependency_hashes[0] for item in dependency_hashes):
        raise RuntimeError("D1 arms do not share one dependency hash set")
    if dependency_hashes[0] != source_dependency_hashes():
        raise RuntimeError("D1 dependency hashes differ from the current source tree")
    if any(item != slurm_runfile_hash_sets[0] for item in slurm_runfile_hash_sets):
        raise RuntimeError("D1 arms do not share one Slurm-runfile hash set")
    if slurm_runfile_hash_sets[0] != slurm_runfile_hashes():
        raise RuntimeError("D1 Slurm runfile hashes differ from the current source tree")
    if any(item != runtimes[0] for item in runtimes):
        raise RuntimeError("D1 arms do not share one exact runtime/GPU identity")
    payload = {
        "schema_version": 1,
        "gate": "Decision_2_3_D1_smoke",
        "status": "pass",
        "scientific_result": False,
        "seed": 42,
        "slurm_job_id": slurm_job_id,
        "arms": rows,
        "git_commit": next(iter(commits)),
        "source_sha256": next(iter(source_hashes)),
        "decision_spec_sha256": next(iter(spec_hashes)),
        "source_dependencies_sha256": dependency_hashes[0],
        "slurm_runfiles_sha256": slurm_runfile_hash_sets[0],
        "runtime": runtimes[0],
        "budget": SMOKE_RECIPE,
        "test_policy": "no test frame loaded or evaluated",
        "meaning": (
            "wiring, CUDA, optimizer, validation, checkpoint, restore, and active "
            "gradient paths passed; no localization claim follows"
        ),
    }
    assert_json_finite(payload)
    write_json(output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args.output_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
