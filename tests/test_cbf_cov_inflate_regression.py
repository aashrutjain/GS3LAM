"""Digit-exact regression test for the COV_INFLATE disambiguation experiment
(PROGRESS.md "COV_INFLATE disambiguation experiment" + its independent
2026-07-12 reproduction, which matched the original numbers "to the displayed
precision" on this same numpy/scipy combination). It exists so a solver swap
has something concrete to fail if it silently changes behavior -- which is what
it was used for when the Clarabel backend landed (see PROGRESS.md).

Most pinned values below are backend-independent -- verified bit-identical
across both solvers. The two that aren't (narrow/k=1.0 severity and time_ratio)
are keyed on the backend in NARROW_K1_EXPECTED. Runs against whatever
CBFQPConfig defaults to; pass --solver=scipy_slsqp for the fallback backend.

Reconstructs the committed seed=42 narrow/wide corridor scenes from
scripts/gen_cbf_synthetic_scene.py (the .ply files themselves were never
committed, only the generator) and drives them through eval_cbf_modes.py's
own helpers (build_qp_cfg, synthesize_hazard_safety, PassthroughFilter) and
src.cbf.sim.rollout -- mirroring PROGRESS.md's independent-reproduction
method exactly rather than reimplementing the harness.

Slow: runs several thousand solver-backed rollout steps (the k_alpha_base=3.0
and 10.0 sweep points run the full max_steps=2000 without reaching the goal).
"""

import numpy as np
import pytest

from eval_cbf_modes import PassthroughFilter, build_qp_cfg, synthesize_hazard_safety
from scripts.gen_cbf_synthetic_scene import build_scene, write_ply
from src.cbf.collision_cone import effective_c
from src.cbf.metrics import collision_severity, mahalanobis_signed_distance, path_efficiency
from src.cbf.ply_io import ZeroSafetyPolicy, load_splat_field
from src.cbf.qp_filter import CBFSafetyFilter
from src.cbf.semantic_weighting import SemanticMode
from src.cbf.sim import rollout

START = np.array([-2.5, 0.0, 0.0], dtype=np.float32)
GOAL = np.array([2.5, 0.0, 0.0], dtype=np.float32)
HAZARD_CENTER = (START + GOAL) / 2.0
HAZARD_RADIUS = 0.5
HAZARD_SAFETY = 0.1

# Matches configs/cbf/room0_cbf.py's defaults, per PROGRESS.md's documented
# experiment config (k_alpha_base=1.0, a_max=1.0, robot_radius=0.16, dt=0.05,
# max_steps=2000, hazard radius=0.5/safety=0.1).
BASE_CFG_DICT = dict(
    a_max=1.0,
    k_alpha_base=1.0,
    robot_radius=0.16,
    pd=dict(kp=1.0, kd=2.0),
    sim=dict(dt=0.05, max_steps=2000, goal_tol=0.1),
    spatial_filter=dict(
        mode="radius",
        radius=None,
        max_candidates=200,
        lookahead_horizon=2.0,
        radius_margin=1.0,
        radius_cap=3.0,
        opacity_prune_thresh=0.1,
    ),
    zero_policy="warn_only",
    semantic=dict(alpha_f=lambda s: s),
    solver="scipy_slsqp",  # overridden per-run by the base_cfg fixture; see conftest.py --solver
)


@pytest.fixture(scope="module")
def base_cfg(solver_name):
    """BASE_CFG_DICT with the backend under test substituted in. Every rollout
    below goes through this rather than the module-level dict, so a
    --solver=clarabel run cannot silently leave some path on SLSQP.
    """
    return {**BASE_CFG_DICT, "solver": solver_name}


def _make_scene(tmp_path, collar_radius):
    rows = build_scene(collar_radius, seed=42)
    out_path = tmp_path / f"scene_{collar_radius}.ply"
    write_ply(rows, str(out_path))
    splats = load_splat_field(str(out_path), zero_policy=ZeroSafetyPolicy.WARN_ONLY)
    splats.safety_raw = synthesize_hazard_safety(
        splats.xyz, HAZARD_CENTER, HAZARD_RADIUS, HAZARD_SAFETY, background_safety=1.0
    )
    splats.ambiguous_zero_mask = None
    splats.zero_fraction = None
    return splats


def _rollout(filt, cfg_dict):
    return rollout(
        filt,
        p0=START,
        v0=np.zeros(3, dtype=np.float32),
        goal=GOAL,
        dt=cfg_dict["sim"]["dt"],
        a_max=cfg_dict["a_max"],
        kp=cfg_dict["pd"]["kp"],
        kd=cfg_dict["pd"]["kd"],
        max_steps=cfg_dict["sim"]["max_steps"],
        goal_tol=cfg_dict["sim"]["goal_tol"],
    )


def _run_cov_inflate(splats, geom_filt, base_cfg, k_alpha_base, cov_gamma):
    cfg_dict = {**base_cfg, "k_alpha_base": k_alpha_base}
    qp_cfg = build_qp_cfg(cfg_dict, SemanticMode.COV_INFLATE, ZeroSafetyPolicy.WARN_ONLY, cov_gamma)
    filt = CBFSafetyFilter(splats, qp_cfg)
    result = _rollout(filt, cfg_dict)

    c_m = effective_c(geom_filt.c_base, qp_cfg.robot_radius, geom_filt.s_min)
    min_dist = mahalanobis_signed_distance(result.trajectory_p, geom_filt.xyz, geom_filt.A, c_m)
    severity = collision_severity(min_dist)

    actual_time = result.time_to_goal if result.reached_goal else cfg_dict["sim"]["max_steps"] * cfg_dict["sim"]["dt"]
    return result, severity, actual_time


@pytest.fixture(scope="module")
def narrow_splats(tmp_path_factory):
    return _make_scene(tmp_path_factory.mktemp("narrow"), collar_radius=0.90)


@pytest.fixture(scope="module")
def wide_splats(tmp_path_factory):
    return _make_scene(tmp_path_factory.mktemp("wide"), collar_radius=1.30)


@pytest.fixture(scope="module")
def narrow_geom_filt(narrow_splats):
    cfg = build_qp_cfg(BASE_CFG_DICT, SemanticMode.NONE, ZeroSafetyPolicy.WARN_ONLY, cov_gamma=1.0)
    return CBFSafetyFilter(narrow_splats, cfg)


@pytest.fixture(scope="module")
def wide_geom_filt(wide_splats):
    cfg = build_qp_cfg(BASE_CFG_DICT, SemanticMode.NONE, ZeroSafetyPolicy.WARN_ONLY, cov_gamma=1.0)
    return CBFSafetyFilter(wide_splats, cfg)


@pytest.fixture(scope="module")
def oracle_time():
    # PassthroughFilter bypasses the QP entirely, so the denominator of every
    # time_ratio below is solver-invariant by construction -- BASE_CFG_DICT, not
    # base_cfg, on purpose.
    result = _rollout(PassthroughFilter(), BASE_CFG_DICT)
    assert result.reached_goal
    return result.time_to_goal


# The only two pinned values that differ between backends. Both are real
# measured numbers, but they are not equally trustworthy: the state-by-state
# re-solve in PROGRESS.md ("Clarabel QP backend added", 2026-07-18) showed SLSQP
# returning suboptimal, over-conservative controls on ~12% of the constrained
# steps while reporting success=True, which roughly doubles its traversal time.
# Clarabel's row is the correct one. SLSQP's is retained only so the fallback
# backend stays regression-guarded -- it is not the number to quote.
NARROW_K1_EXPECTED = {
    # severity, time_ratio
    "clarabel": (0.1528, 5.896),
    "scipy_slsqp": (0.1646, 11.91),
}


def test_step0_narrow_corridor_gamma1_pathological_slowdown(
    narrow_splats, narrow_geom_filt, base_cfg, oracle_time, solver_name
):
    """PROGRESS.md 'Step 0' / reproduced 'Step 0/1' table, narrow corridor,
    COV_INFLATE gamma=1.0, k_alpha_base=1.0 (config default), infeasible_count=0.

    severity/time_ratio are backend-dependent (see NARROW_K1_EXPECTED): 0.1528 /
    5.896 under clarabel, the default and the trusted values; 0.1646 / 11.91
    under scipy_slsqp, whose slower traversal is a solver artifact rather than a
    property of the scene. Everything else asserted here -- reached_goal and
    infeasible_count, plus every value in the other two tests -- was verified
    bit-identical across both backends.
    """
    expected_severity, expected_time_ratio = NARROW_K1_EXPECTED[solver_name]
    result, severity, actual_time = _run_cov_inflate(
        narrow_splats, narrow_geom_filt, base_cfg, k_alpha_base=1.0, cov_gamma=1.0
    )
    time_ratio = actual_time / oracle_time

    assert result.reached_goal
    assert result.infeasible_count == 0
    assert severity == pytest.approx(expected_severity, abs=5e-4)
    assert time_ratio == pytest.approx(expected_time_ratio, abs=5e-3)


def test_step2_wide_corridor_gamma1_time_ratio_collapses(wide_splats, wide_geom_filt, base_cfg, oracle_time):
    """PROGRESS.md 'Step 2': identical gamma/k_alpha_base, wider corridor --
    the narrow corridor's slowdown collapses to ~1.09x. This value is
    bit-identical under both backends, which is part of why the narrow
    corridor's backend-dependence is attributable to the solver rather than to
    the scene."""
    result, _, actual_time = _run_cov_inflate(
        wide_splats, wide_geom_filt, base_cfg, k_alpha_base=1.0, cov_gamma=1.0
    )
    time_ratio = actual_time / oracle_time

    assert result.reached_goal
    assert time_ratio == pytest.approx(1.09, abs=5e-3)


@pytest.mark.parametrize("k_alpha_base,expected_infeasible_count", [(3.0, 2), (10.0, 40)])
def test_step3_narrow_corridor_k_alpha_sweep_infeasible_count(
    narrow_splats, narrow_geom_filt, base_cfg, k_alpha_base, expected_infeasible_count
):
    """PROGRESS.md 'Step 3' k_alpha_base sweep (narrow corridor, gamma=1.0
    fixed): pushing the gain past the default drives h further negative
    pre-violation until the QP genuinely goes infeasible and falls back to
    max-braking -- infeasible_count 2 at k_alpha_base=3.0, 40 at 10.0,
    neither run reaching the goal within max_steps."""
    result, _, _ = _run_cov_inflate(
        narrow_splats, narrow_geom_filt, base_cfg, k_alpha_base=k_alpha_base, cov_gamma=1.0
    )

    assert not result.reached_goal
    assert result.infeasible_count == expected_infeasible_count
