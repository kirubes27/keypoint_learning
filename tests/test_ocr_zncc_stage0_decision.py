from __future__ import annotations

from keypoint_net.run_ocr_zncc_stage0 import (
    _calibration_receipt,
    _summarize_direction_rows,
)


def _rows(*, usable: int, total: int, cosine: float, error_delta: float):
    rows = []
    for index in range(total):
        is_usable = index < usable
        rows.append(
            {
                "channel": index % 2,
                "accepted_match": is_usable,
                "usable_direction": is_usable,
                "direction_cosine": cosine if is_usable else None,
                "one_cell_material_error_delta_objdiag": error_delta if is_usable else None,
                "on_object_before": True,
                "on_object_after": True,
            }
        )
    return rows


def test_direction_gate_uses_preregistered_coverage_and_signs() -> None:
    passed = _summarize_direction_rows(
        _rows(usable=6, total=10, cosine=0.2, error_delta=-0.01),
        [0, 1],
    )
    assert passed["cell_pass"] is True
    insufficient = _summarize_direction_rows(
        _rows(usable=4, total=10, cosine=0.2, error_delta=-0.01),
        [0, 1],
    )
    assert insufficient["cell_pass"] is False
    wrong_direction = _summarize_direction_rows(
        _rows(usable=6, total=10, cosine=-0.2, error_delta=0.01),
        [0, 1],
    )
    assert wrong_direction["cell_pass"] is False


def test_combined_calibration_chooses_one_shared_capped_coefficient() -> None:
    cells = []
    for recipe, ratios in (
        ("task55_clean", [0.2, 0.3, 0.4]),
        ("task80_assisted", [0.25, 0.35, 0.45]),
    ):
        for seed, ratio in zip((42, 43, 44), ratios):
            cells.append(
                {
                    "cell_id": f"{recipe}__r64__seed{seed}",
                    "recipe": recipe,
                    "seed": seed,
                    "auxiliary_to_base_gradient_ratio": ratio,
                }
            )
    receipt = _calibration_receipt(cells)
    assert abs(receipt["coefficient"] - (0.1 / 0.325)) < 1e-12
    assert abs(receipt["scaled_median_contribution_all_six_cells"] - 0.1) < 1e-12
    assert receipt["coefficient_rule"]["one_shared_coefficient_across_task55_and_task80"] is True


def test_combined_calibration_uses_point_five_only_as_cap() -> None:
    cells = []
    for recipe in ("task55_clean", "task80_assisted"):
        for seed in (42, 43, 44):
            cells.append(
                {
                    "cell_id": f"{recipe}__r64__seed{seed}",
                    "recipe": recipe,
                    "seed": seed,
                    "auxiliary_to_base_gradient_ratio": 0.01,
                }
            )
    receipt = _calibration_receipt(cells)
    assert receipt["coefficient"] == 0.5
    assert receipt["status"] == "calibrated_at_safety_cap"
