from keypoint_net.run_mature_equivariance_gradient_decomposition import (
    EXPECTED_ROLES,
    TASK80_WEIGHTS,
)


def test_mature_scope_is_exact_task80_seed42_best_pair() -> None:
    assert EXPECTED_ROLES == (
        "task80_assisted__control__seed42__best_model",
        "task80_assisted__ocr_zncc__seed42__best_model",
    )
    assert TASK80_WEIGHTS == {
        "lambda_smooth": 0.001,
        "lambda_disp": 0.1,
        "lambda_ent": 0.01,
        "lambda_inv": 0.5,
        "lambda_cycle": 0.5,
    }
