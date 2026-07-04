"""Semantic/integrity checks for all completed Day-2.5 outputs."""

import csv
import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    with (OUT / "day25_noise_ladder.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 22
    assert all(math.isfinite(float(row["ordering_fraction"])) for row in rows)
    by_cell = {(row["noise"], int(row["K"])): float(row["ordering_fraction"]) for row in rows}
    assert by_cell[("iid_homogeneous", 6)] >= 0.95
    assert by_cell[("iid_homogeneous", 10)] >= 0.95
    assert by_cell[("empirical", 6)] < 0.95
    assert by_cell[("empirical", 10)] < 0.95

    attribution = json.loads((OUT / "day25_noise_attribution.json").read_text())
    assert attribution["verdict"] == "interaction_or_unresolved"
    assert attribution["attributed_properties"] == []

    precheck = json.loads((OUT / "day25_similarity_lr_precheck.json").read_text())
    assert precheck["selected_learning_rate"] == 0.02
    assert all(row["stable"] for row in precheck["rows"])

    control = json.loads((OUT / "day25_filter_artifact_control.json").read_text())
    assert control["verdict"] == "network_residual_like"
    assert control["selected_residual_bank"].endswith("empirical_jitter_residuals.npz")
    for keypoints in (6, 10):
        cell = control["control_summary"][str(keypoints)]
        assert len(cell["individual_bank_fractions"]) == 5
        assert cell["closer_to"] == "iid_heterogeneous"
        assert math.isfinite(cell["across_bank_sample_std_ddof1"])

    with (OUT / "day25_similarity_results.csv").open() as handle:
        similarity_rows = list(csv.DictReader(handle))
    assert len(similarity_rows) == 10
    full = [row for row in similarity_rows if float(row["noise_scale"]) == 1.0]
    half = [row for row in similarity_rows if float(row["noise_scale"]) == 0.5]
    assert sum(row["success"] == "True" for row in full) == 3
    assert sum(row["success"] == "True" for row in half) == 5
    failed_full = [row for row in full if row["success"] == "False"]
    assert all(row["radius_le_0_8"] == "False" for row in failed_full)
    for condition in (
        "paired_separation_gt_2",
        "all_other_distances_gt_1",
        "healthy_displacement_lt_1",
        "heldout_loss_decreased",
        "finite",
    ):
        assert all(row[condition] == "True" for row in failed_full)
    verdict = json.loads((OUT / "day25_similarity_verdict.json").read_text())
    assert verdict["verdict_by_noise_scale"]["1.0"]["pass"] is False
    assert verdict["verdict_by_noise_scale"]["0.5"]["pass"] is True
    print("ALL COMPLETED DAY-2.5 OUTPUT CHECKS PASSED")


if __name__ == "__main__":
    main()
