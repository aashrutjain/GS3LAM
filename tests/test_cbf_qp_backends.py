"""Direct unit tests for the QP backends in src/cbf/qp_filter.py, checked
against hand-solved solutions.

This closes PROGRESS.md's "Not covered, deliberately" item: before the Clarabel
swap, `_solve_scipy_slsqp` had no isolated test at all -- it was exercised only
indirectly, through thousands of rollout steps in
test_cbf_cov_inflate_regression.py. That catches drift but says nothing about
whether either backend gets a *known* QP right.

Every test runs against both backends via the `solve_fn` parametrization, so
the two are held to one standard. That parametrization is also what makes it
impossible for the Clarabel path to be silently skipped: if the registry entry
were missing, or the reformulation returned SLSQP's answer by some accident of
plumbing, these would not both pass.

The QP under test (see the module docstring in qp_filter.py):

    min_u ||u - u_ref||^2  s.t.  w_i^T u >= -(k_alpha_i/2) h_i,  ||u|| <= a_max
"""

import numpy as np
import pytest

from src.cbf.qp_filter import _SOLVERS

A_MAX = 1.0

# Loose enough to accommodate an interior-point solver's convergence tolerance
# against SLSQP's active-set exactness; tight enough that a genuine formulation
# error (wrong sign, boxed norm bound, dropped row) fails.
TOL = 1e-5

# Where the *curved* SOC boundary is the binding constraint, Clarabel lands
# ~1e-5 off the exact radial projection while SLSQP hits it to float32
# precision -- an interior-point solver approaches a nonlinear active
# constraint from the interior and stops at its convergence tolerance. Still
# orders of magnitude tighter than any formulation error would produce.
SOC_TOL = 1e-4


@pytest.fixture(params=sorted(_SOLVERS), ids=sorted(_SOLVERS))
def solve_fn(request):
    return _SOLVERS[request.param]


def _f32(*arrays):
    return [np.asarray(a, dtype=np.float32) for a in arrays]


def test_inactive_constraint_returns_u_ref(solve_fn):
    """A constraint slack at u_ref shouldn't move the solution: the QP reduces
    to the unconstrained argmin, which is u_ref itself.

    Note the direction of h: the row is w^T u >= -(k/2) h, so h just below zero
    is the *slack* end of the range and large negative h is the demanding end.
    h=-0.01 here puts the row at u_x >= 0.005, which u_ref already satisfies.
    """
    u_ref, w, h, k_alpha = _f32([0.2, 0.1, 0.0], [[1.0, 0.0, 0.0]], [-0.01], [1.0])

    u, infeasible, diag = solve_fn(u_ref, w, h, k_alpha, A_MAX)

    assert not infeasible
    assert u.dtype == np.float32
    np.testing.assert_allclose(u, u_ref, atol=TOL)
    assert diag["n_active"] == 1


def test_single_binding_cbf_row_projects_onto_halfspace(solve_fn):
    """u_ref violates one CBF row, the norm bound is slack. The solution is the
    Euclidean projection of u_ref onto {u : w^T u >= -(k/2)h}:

        u = u_ref + ((rhs_violation) / ||w||^2) * w

    Here w = [1,0,0], h = -1, k_alpha = 1 => the row is u_x >= -0.5*(-1) = 0.5.
    u_ref has u_x = -0.3, so the projection slides it to exactly 0.5 and leaves
    the other two components untouched.
    """
    u_ref, w, h, k_alpha = _f32([-0.3, 0.25, -0.1], [[1.0, 0.0, 0.0]], [-1.0], [1.0])

    u, infeasible, _ = solve_fn(u_ref, w, h, k_alpha, A_MAX)

    assert not infeasible
    np.testing.assert_allclose(u, [0.5, 0.25, -0.1], atol=TOL)
    assert np.linalg.norm(u) <= A_MAX + TOL


def test_norm_bound_is_a_true_ball_not_a_box(solve_fn):
    """The actuator bound is ||u|| <= a_max, a second-order cone -- not a
    per-axis box. With no active CBF row and a u_ref well outside the ball, the
    answer is the radial projection a_max * u_ref/||u_ref||.

    This is the test that fails if the SOC is ever approximated as a box: the
    box would admit u = [1,1,1] (norm 1.732), and the radial projection of this
    particular u_ref has all three components strictly inside +-a_max.
    """
    u_ref = np.asarray([3.0, -4.0, 12.0], dtype=np.float32)  # norm 13, nicely exact
    # Row is u_z >= 0.005, slack at the projected point (u_z = 12/13).
    w, h, k_alpha = _f32([[0.0, 0.0, 1.0]], [-0.01], [1.0])

    u, infeasible, _ = solve_fn(u_ref, w, h, k_alpha, A_MAX)

    assert not infeasible
    np.testing.assert_allclose(u, A_MAX * u_ref / 13.0, atol=SOC_TOL)
    assert np.linalg.norm(u) == pytest.approx(A_MAX, abs=SOC_TOL)


def test_stacked_active_rows(solve_fn):
    """Two simultaneously-violated rows, as the active-set gate can produce when
    several splats are in range. u_ref = 0 violates both u_x >= 0.5 and
    u_y >= 0.5; since the rows are axis-aligned and independent, the minimum-norm
    point satisfying both is exactly [0.5, 0.5, 0] (norm 0.707, inside the ball).
    """
    u_ref = np.zeros(3, dtype=np.float32)
    w, h, k_alpha = _f32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [-1.0, -1.0], [1.0, 1.0])

    u, infeasible, diag = solve_fn(u_ref, w, h, k_alpha, A_MAX)

    assert not infeasible
    assert diag["n_active"] == 2
    np.testing.assert_allclose(u, [0.5, 0.5, 0.0], atol=TOL)


def test_genuinely_infeasible_qp_reports_infeasible(solve_fn):
    """Two directly opposed rows -- u_x >= 5 and -u_x >= 5 -- cannot both hold,
    and neither is reachable inside ||u|| <= 1 anyway. Both backends must report
    infeasible=True so CBFSafetyFilter.step routes to _max_braking(); the
    contract is the flag, not any particular returned u.
    """
    u_ref = np.zeros(3, dtype=np.float32)
    w, h, k_alpha = _f32([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [-10.0, -10.0], [1.0, 1.0])

    _, infeasible, diag = solve_fn(u_ref, w, h, k_alpha, A_MAX)

    assert infeasible
    assert diag["success"] is False


def test_backends_agree_on_randomized_instances():
    """Cross-check the two backends against each other over random feasible
    instances, rather than against a closed form. Guards the reformulation as a
    whole -- objective scaling, row signs, cone ordering -- in cases too messy
    to hand-solve.

    Seeded and non-parametrized: this compares backends, so it needs both.
    """
    rng = np.random.default_rng(0)
    slsqp, clarabel_fn = _SOLVERS["scipy_slsqp"], _SOLVERS["clarabel"]
    compared = 0

    for _ in range(200):
        n = int(rng.integers(1, 4))
        u_ref = rng.normal(0.0, 0.5, 3).astype(np.float32)
        w = rng.normal(0.0, 1.0, (n, 3)).astype(np.float32)
        # Small |h| keeps the halfspaces near the origin, so the intersection
        # with the a_max ball is nonempty and the instance is actually feasible.
        h = (-np.abs(rng.normal(0.0, 0.1, n))).astype(np.float32)
        k_alpha = np.ones(n, dtype=np.float32)

        u_s, infeas_s, _ = slsqp(u_ref, w, h, k_alpha, A_MAX)
        u_c, infeas_c, _ = clarabel_fn(u_ref, w, h, k_alpha, A_MAX)
        if infeas_s or infeas_c:
            continue

        np.testing.assert_allclose(u_c, u_s, atol=1e-4)
        compared += 1

    assert compared > 150, f"too few feasible instances ({compared}) to be a meaningful check"
