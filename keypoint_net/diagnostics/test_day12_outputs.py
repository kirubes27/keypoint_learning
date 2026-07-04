"""Semantic integrity checks for completed Day-1/Day-2 artifacts."""

import csv
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent / "outputs"
MODELS = ("task80", "smoke64", "smoke128")


def rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    geometry = json.loads((OUT / "geometry_gate.json").read_text())
    assert geometry["geometry_ok"] is True
    assert geometry["mean_iou"] >= 0.95
    assert geometry["split_center_distance_px"] <= 3.0

    summaries = rows(OUT / "day1_summary.csv")
    assert {row["model"] for row in summaries} == set(MODELS)
    for model in MODELS:
        channel_rows = rows(OUT / f"day1_channels_{model}.csv")
        assert len(channel_rows) == 10
        assert {int(row["channel"]) for row in channel_rows} == set(range(10))
        for row in channel_rows:
            for key in ("on_mask_frac", "on_mask_frac_dilated", "hm_mass_in_mask"):
                assert 0.0 <= float(row[key]) <= 1.0, (model, key, row[key])
            assert float(row["eq_err_1_median_cells64"]) >= 0.0
            assert float(row["hm_entropy_nats"]) >= 0.0
        cache = np.load(OUT / f"day1_cache_{model}.npz")
        assert cache["coords"].shape == (180, 10, 2)
        assert cache["logits"].shape[:2] == (180, 10)
        assert np.isfinite(cache["coords"]).all() and np.isfinite(cache["logits"]).all()

    assert len(rows(OUT / "day2_switching.csv")) == 30
    assert len(rows(OUT / "day2_hard_soft.csv")) == 30
    aliasing = rows(OUT / "day2_aliasing.csv")
    assert len(aliasing) == 3 * 20 * 10
    assert len(rows(OUT / "day2_diagnostic_levers.csv")) == 3 * 5 * 10
    for row in aliasing:
        assert float(row["residual_median_cells64"]) >= 0.0
        assert float(row["control_median_cells64"]) >= 0.0

    summary = json.loads((OUT / "day2_summary.json").read_text())
    assert set(summary) == set(MODELS)
    print("ALL DAY-1/DAY-2 OUTPUT CHECKS PASSED")


if __name__ == "__main__":
    main()
