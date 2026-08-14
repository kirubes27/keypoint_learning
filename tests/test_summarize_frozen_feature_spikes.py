from __future__ import annotations

import numpy as np

from keypoint_net.summarize_frozen_feature_spikes import numeric_summary, summarize_grounded_rows


def test_numeric_summary_reports_descriptive_quantiles() -> None:
    row = numeric_summary((0.0, 1.0, 2.0, 3.0, 4.0))
    assert row["n"] == 5
    assert row["median"] == 2.0
    assert np.isclose(row["q10"], 0.4)
    assert np.isclose(row["q90"], 3.6)


def test_grounded_summary_excludes_off_object_edges() -> None:
    grounded = {
        "grounded_physical_edge": True,
        "physical_target_mask_percentile": 0.9,
        "detector_minus_physical_similarity": -0.2,
        "object_argmax_distance_to_physical_cells": 2.0,
        "detector_distance_to_physical_cells": 5.0,
        "object_best_minus_physical_similarity": 0.1,
        "object_top1_top2_margin": 0.01,
    }
    off_object = {
        **grounded,
        "grounded_physical_edge": False,
        "physical_target_mask_percentile": 0.0,
    }
    row = summarize_grounded_rows([grounded, off_object])
    assert row["selected_edge_count"] == 2
    assert row["grounded_physical_edge_count"] == 1
    assert row["grounded_physical_edge_fraction"] == 0.5
    assert row["grounded_metrics"]["physical_target_mask_percentile"]["median"] == 0.9
    assert row["grounded_metrics"]["physical_similarity_exceeds_detector_fraction"] == 1.0
    assert row["grounded_metrics"]["feature_argmax_closer_than_detector_fraction"] == 1.0
