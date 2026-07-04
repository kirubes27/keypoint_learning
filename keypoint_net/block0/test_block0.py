"""
Standalone regression tests for Block 0 solver degeneracies (amendment 5).
No pytest in the phd env — run directly:
    /opt/anaconda3/envs/phd/bin/python test_block0.py
Every test asserts; the runner reports PASS/FAIL per test and exits nonzero
on any failure.
"""

import sys
import traceback

import torch

torch.set_default_dtype(torch.float64)

from block0 import (SOLVERS, FLOOR, LAM, subset_consistency, plain_fit_loss,
                    sample_G, apply_G, grid_err, _anchors, _distinct3,
                    make_world)


def test_recovery_exact_own_family():
    """Each solver recovers its own family's transform (lam -> 0)."""
    g = torch.Generator().manual_seed(1)
    for fam, fit in SOLVERS.items():
        A, t = sample_G(fam, g)
        P = torch.cat([_distinct3(g), _anchors(g, 5)], 0)
        Q = apply_G(A, t, P)
        Ah, th, _ = fit(P, Q, 1e-12)
        e = grid_err(Ah, th, A, t).item()
        assert e < 1e-8, f"{fam}: grid err {e:.2e}"


def test_finite_outputs_and_grads_at_collapse():
    """All solvers: finite A, t and finite input-gradients at full collapse."""
    for fam, fit in SOLVERS.items():
        P = torch.tensor([[0.3, 0.2]]).repeat(5, 1).requires_grad_(True)
        Q = P.detach() + 0.05
        A, t, diag = fit(P, Q, LAM)
        s = A.sum() + t.sum()
        grad, = torch.autograd.grad(s, P)
        assert torch.isfinite(A).all() and torch.isfinite(t).all(), fam
        assert torch.isfinite(grad).all(), f"{fam}: non-finite grad at collapse"
        assert diag["fallback"] > 0.0, f"{fam}: fallback not logged"


def test_finite_at_exact_zero():
    """Degenerate all-zeros input (worst case for known-centre solvers)."""
    for fam, fit in SOLVERS.items():
        P = torch.zeros(4, 2).requires_grad_(True)
        Q = torch.zeros(4, 2)
        A, t, _ = fit(P, Q, LAM)
        s = A.sum() + t.sum()
        grad, = torch.autograd.grad(s, P)
        assert torch.isfinite(A).all() and torch.isfinite(t).all(), fam
        assert torch.isfinite(grad).all(), fam


def test_coincident_two_point_similarity_subset():
    """Amendment 5: the exact case called out — coincident 2-pt similarity
    subset. Zero-variance denominator must be regularized; fallback = identity
    linear part + centroid translation; finite gradients."""
    P = torch.tensor([[0.3, 0.2], [0.3, 0.2]]).requires_grad_(True)
    Q = torch.tensor([[0.4, 0.15], [0.4, 0.15]])
    A, t, diag = SOLVERS["similarity"](P, Q, LAM)
    grad, = torch.autograd.grad(A.sum() + t.sum(), P)
    assert torch.isfinite(A).all() and torch.isfinite(grad).all()
    assert torch.allclose(A, torch.eye(2), atol=1e-9), "no identity fallback"
    assert torch.allclose(t, torch.tensor([0.1, -0.05]), atol=1e-9), \
        "translation fallback should be centroid displacement"
    assert diag["fallback"] > 0.99, "fallback activation not logged as ~1"


def test_identity_shrinkage_at_rank_deficiency():
    """Rank-deficient inputs shrink the linear part toward identity."""
    # zero-variance (single location) for centroid-based solvers
    P = torch.tensor([[0.25, -0.1]]).repeat(6, 1)
    Q = P + torch.tensor([0.07, 0.02])
    for fam in ["similarity", "affine"]:
        A, t, _ = SOLVERS[fam](P, Q, LAM)
        assert torch.allclose(A, torch.eye(2), atol=1e-9), fam
    # known-centre solvers at the centre itself
    Z = torch.zeros(3, 2)
    for fam in ["rotation", "rotscale"]:
        A, t, _ = SOLVERS[fam](Z, Z, LAM)
        assert torch.allclose(A, torch.eye(2), atol=1e-9), fam


def test_complement_guard():
    """Amendment 6: K <= floor must raise (empty complement undefined)."""
    g = torch.Generator().manual_seed(2)
    w = make_world("healthy", "affine", 3, g)
    try:
        subset_consistency(w["P0"], w["P1"], w["P2"], "affine")
    except ValueError:
        return
    raise AssertionError("K=3 affine subset loss did not raise")


def test_min_aggregation_blind_to_duplicates():
    """min-agg on exact duplicates collapses to healthy level; mean does not."""
    g = torch.Generator().manual_seed(3)
    wd = make_world("dup_exact", "affine", 6, g)
    sc = subset_consistency(wd["P0"], wd["P1"], wd["P2"], "affine")
    assert sc["min"] < sc["mean"] / 10, \
        f"min {sc['min']:.2e} not far below mean {sc['mean']:.2e}"


def test_softmin_between_min_and_mean():
    g = torch.Generator().manual_seed(4)
    w = make_world("dup_exact", "affine", 6, g)
    sc = subset_consistency(w["P0"], w["P1"], w["P2"], "affine")
    assert sc["min"] - 1e-12 <= sc["softmin"] <= sc["mean"] + 1e-12


def test_affine_underdetermined_below_floor():
    """2 points cannot identify an affine map; 3 non-collinear can."""
    g = torch.Generator().manual_seed(5)
    errs = {}
    for n in [2, 3]:
        acc = []
        for tr in range(10):
            gg = torch.Generator().manual_seed(50 + tr)
            A, t = sample_G("affine", gg)
            P = _distinct3(gg) if n == 3 else _anchors(gg, 2)
            Q = apply_G(A, t, P)
            Ah, th, _ = SOLVERS["affine"](P, Q, LAM)
            acc.append(grid_err(Ah, th, A, t).item())
        errs[n] = sum(acc) / len(acc)
    assert errs[3] < 1e-2, f"floor fit poor: {errs[3]:.2e}"
    assert errs[3] < 0.05 * errs[2], f"no floor gap: {errs}"


def test_gradients_through_subset_loss():
    """Gradient flows through the full subset loss on a near-duplicate world
    and is finite everywhere (autograd through the ridge solve)."""
    g = torch.Generator().manual_seed(6)
    w = make_world("dup_near", "affine", 6, g, eps=1e-3)
    P0 = w["P0"].clone().requires_grad_(True)
    P1 = w["P1"].clone().requires_grad_(True)
    P2 = w["P2"].clone().requires_grad_(True)
    loss = subset_consistency(P0, P1, P2, "affine")["mean"]
    loss.backward()
    for name, p in [("P0", P0), ("P1", P1), ("P2", P2)]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_rotation_regularized_zero_signal():
    """atan2 regularization: zero signal returns exactly theta=0 (identity)."""
    P = torch.zeros(3, 2)
    A, t, _ = SOLVERS["rotation"](P, P, LAM)
    assert torch.allclose(A, torch.eye(2), atol=1e-12)


TESTS = [
    test_recovery_exact_own_family,
    test_finite_outputs_and_grads_at_collapse,
    test_finite_at_exact_zero,
    test_coincident_two_point_similarity_subset,
    test_identity_shrinkage_at_rank_deficiency,
    test_complement_guard,
    test_min_aggregation_blind_to_duplicates,
    test_softmin_between_min_and_mean,
    test_affine_underdetermined_below_floor,
    test_gradients_through_subset_loss,
    test_rotation_regularized_zero_signal,
]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} tests passed")
    sys.exit(1 if failed else 0)
