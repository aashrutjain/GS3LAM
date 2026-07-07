"""CBF-QP safety filter: assembles and solves the per-step QP from
collision_cone.py's per-splat (w, h) pairs, under whichever semantic_weighting
strategy is configured.

QP (paper's formulation, k_alpha renamed from p_k -- see collision_cone.py):

    min_u  ||u - u_ref||^2_2
    s.t.   w_i(x)^T u >= -(k_alpha_i/2) * h_i(p,v)   for each active splat i
           ||u|| <= a_max

Solver: kept behind CBFQPConfig.solver as a string key into a small registry
(see _SOLVERS below) rather than a hard dependency -- this was an explicitly
DEFERRED decision (see the Stage 3 design plan): the actuator bound is a norm
(second-order-cone) constraint, not plain-linear, so a general QP solver
(OSQP/quadprog) needs it approximated as a per-axis box, while the paper's own
choice (Clarabel) handles SOCP natively. The default backend here
("scipy_slsqp") adds no new pinned dependency (scipy is already required by
spatial_filter.py) and is fine for this control-dimension (u in R^3, one
inequality row per active splat) -- swap in a Clarabel/OSQP backend later by
adding an entry to _SOLVERS; this is a stopgap, not a claim that scipy_slsqp
is the right production choice for a real-time TurtleBot4 loop.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .collision_cone import compute_collision_cones, effective_c
from .ellipsoid import build_A, build_sigma
from .interfaces import RobotState, SafetyFilterResult
from .ply_io import SplatField, ZeroSafetyPolicy
from .semantic_weighting import SemanticMode, alpha_gain_per_splat, inflate_covariance
from .spatial_filter import SpatialFilterConfig, build_kdtree, prune_low_opacity, select_candidates


@dataclass
class CBFQPConfig:
    a_max: float
    k_alpha_base: float
    robot_radius: float = 0.16  # TurtleBot4 footprint radius (m) -- CONFIRM actual value before real use
    chi2_conf: float = field(default_factory=lambda: float(chi2.ppf(0.99, df=3)))
    semantic_mode: SemanticMode = SemanticMode.NONE
    alpha_f: Any = field(default=lambda s: s)          # f(safety) for ALPHA_SCALE, identity default
    cov_inflate_gamma: float = 1.0                      # gamma for COV_INFLATE -- a value to sweep, not a settled constant
    zero_policy: ZeroSafetyPolicy = ZeroSafetyPolicy.WARN_ONLY
    spatial_filter: SpatialFilterConfig = field(default_factory=SpatialFilterConfig)
    solver: str = "scipy_slsqp"


def build_baseline_inputs(splats: SplatField, cfg: CBFQPConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-geometric path. MUST NOT read splats.safety_raw -- this function's
    output must be identical whether safety_raw is None or garbage; that
    invariant is what makes SemanticMode.NONE a genuine baseline and should be
    covered by a unit test before this is trusted.
    """
    Sigma, s_min = build_sigma(splats.log_scale_raw, splats.rot_raw)
    A = build_A(Sigma)
    k_alpha_uniform = np.full(splats.n, cfg.k_alpha_base, dtype=np.float32)
    return A, s_min, k_alpha_uniform


def build_semantic_inputs(splats: SplatField, cfg: CBFQPConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatches on cfg.semantic_mode. Always returns shape-uniform
    (A: (N,3,3), s_min: (N,), k_alpha: (N,)) regardless of mode, so
    collision_cone.py and the QP assembly below are identical code for all
    three modes.
    """
    if cfg.semantic_mode is SemanticMode.NONE:
        return build_baseline_inputs(splats, cfg)

    if splats.safety_raw is None:
        raise ValueError(
            f"semantic_mode={cfg.semantic_mode} requires a 'safety' column, but "
            f"{splats.source_path} has none. Load a safety_gsplat.ply, or use "
            "SemanticMode.NONE for a plain gsplat.ply."
        )

    Sigma, s_min = build_sigma(splats.log_scale_raw, splats.rot_raw)
    A = build_A(Sigma)

    if cfg.semantic_mode is SemanticMode.ALPHA_SCALE:
        k_alpha = alpha_gain_per_splat(cfg.k_alpha_base, splats.safety_raw, cfg.alpha_f)
        return A, s_min, k_alpha

    if cfg.semantic_mode is SemanticMode.COV_INFLATE:
        A_eff, s_min_eff = inflate_covariance(A, s_min, splats.safety_raw, cfg.cov_inflate_gamma)
        k_alpha_uniform = np.full(splats.n, cfg.k_alpha_base, dtype=np.float32)
        return A_eff, s_min_eff, k_alpha_uniform

    raise ValueError(f"Unhandled semantic_mode: {cfg.semantic_mode!r}")


def _max_braking(v: np.ndarray, a_max: float) -> np.ndarray:
    speed = float(np.linalg.norm(v))
    if speed < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (-a_max * v / speed).astype(np.float32)


def _clip_to_a_max(u: np.ndarray, a_max: float) -> np.ndarray:
    """Dtype-preserving: used both on float32 result arrays and, inside the
    float64-cast solver path, on the float64 x0 seed -- an earlier version
    hardcoded .astype(np.float32) here, which silently downcast the SLSQP
    initial guess back to float32 while every other solver array had been
    promoted to float64, crashing the Fortran backend ("expected elsize=8
    but got 4") on the very first step whose reference control exceeded
    a_max.
    """
    norm = float(np.linalg.norm(u))
    if norm <= a_max or norm < 1e-12:
        return u
    return (u * (a_max / norm)).astype(u.dtype)


def _solve_scipy_slsqp(
    u_ref: np.ndarray,
    w_active: np.ndarray,
    h_active: np.ndarray,
    k_alpha_active: np.ndarray,
    a_max: float,
) -> tuple[np.ndarray, bool, dict]:
    # SLSQP's Fortran backend requires float64 throughout (jac/constraint arrays must be
    # elsize=8); everything upstream is float32, so cast once here at the solver boundary.
    u_ref = u_ref.astype(np.float64)
    w_active = w_active.astype(np.float64)
    h_active = h_active.astype(np.float64)
    k_alpha_active = k_alpha_active.astype(np.float64)
    rhs = 0.5 * k_alpha_active * h_active  # (n_active,)

    def objective(u):
        return 0.5 * float(np.sum((u - u_ref) ** 2))

    def obj_grad(u):
        return u - u_ref

    def cbf_con(u):
        return w_active @ u + rhs

    def cbf_jac(u):
        return w_active

    def norm_con(u):
        return a_max - np.linalg.norm(u)

    def norm_jac(u):
        n = np.linalg.norm(u)
        if n < 1e-12:
            return np.zeros_like(u)
        return -u / n

    x0 = _clip_to_a_max(u_ref, a_max)
    result = minimize(
        objective,
        x0,
        jac=obj_grad,
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": cbf_con, "jac": cbf_jac},
            {"type": "ineq", "fun": norm_con, "jac": norm_jac},
        ],
        options={"maxiter": 100, "ftol": 1e-9},
    )
    diagnostics = {
        "solver": "scipy_slsqp",
        "success": bool(result.success),
        "message": str(result.message),
        "n_active": int(w_active.shape[0]),
    }
    return result.x.astype(np.float32), (not result.success), diagnostics


_SOLVERS = {
    "scipy_slsqp": _solve_scipy_slsqp,
}


class CBFSafetyFilter:
    """Implements the SafetyFilter protocol (see interfaces.py)."""

    def __init__(self, splats: SplatField, cfg: CBFQPConfig):
        self._cfg = cfg
        keep_mask = prune_low_opacity(splats.opacity, cfg.spatial_filter.opacity_prune_thresh)
        self._orig_idx = np.nonzero(keep_mask)[0]

        A_full, s_min_full, k_alpha_full = build_semantic_inputs(splats, cfg)
        self._xyz = splats.xyz[self._orig_idx]
        self._A = A_full[self._orig_idx]
        self._s_min = s_min_full[self._orig_idx]
        self._k_alpha = k_alpha_full[self._orig_idx]
        self._tree = build_kdtree(self._xyz)
        self._c_base = float(np.sqrt(cfg.chi2_conf))

        if self._orig_idx.size == 0:
            raise ValueError(
                f"opacity_prune_thresh={cfg.spatial_filter.opacity_prune_thresh} pruned every "
                "splat -- check the threshold or the loaded scene."
            )

    @property
    def xyz(self) -> np.ndarray:
        """Opacity-pruned splat means this filter was built against."""
        return self._xyz

    @property
    def A(self) -> np.ndarray:
        """Per-splat A=Sigma^-1 for the pruned set, under this filter's semantic_mode."""
        return self._A

    @property
    def s_min(self) -> np.ndarray:
        """Per-splat smallest true scale for the pruned set, under this filter's semantic_mode."""
        return self._s_min

    @property
    def c_base(self) -> float:
        """sqrt(chi2_conf), pre-Minkowski-inflation confidence radius."""
        return self._c_base

    @property
    def robot_radius(self) -> float:
        return self._cfg.robot_radius

    def step(self, state: RobotState, u_ref: np.ndarray) -> SafetyFilterResult:
        cand_local = select_candidates(state.p, state.v, self._tree, self._cfg.spatial_filter)

        if cand_local.size == 0:
            return SafetyFilterResult(
                u_safe=_clip_to_a_max(u_ref, self._cfg.a_max),
                active_splat_ids=np.array([], dtype=np.int64),
                min_h=float("inf"),
                infeasible=False,
                solver_diagnostics={"note": "no candidate splats in range"},
            )

        mu = self._xyz[cand_local]
        A = self._A[cand_local]
        s_min = self._s_min[cand_local]
        k_alpha = self._k_alpha[cand_local]
        c_m = effective_c(self._c_base, self._cfg.robot_radius, s_min)
        cones = compute_collision_cones(state.p, state.v, mu, A, c_m)

        active_mask = cones.cone_exists
        if not active_mask.any():
            return SafetyFilterResult(
                u_safe=_clip_to_a_max(u_ref, self._cfg.a_max),
                active_splat_ids=np.array([], dtype=np.int64),
                min_h=float(cones.h.min()),
                infeasible=False,
                solver_diagnostics={"note": "no active collision cones"},
            )

        solve_fn = _SOLVERS[self._cfg.solver]
        u_safe, infeasible, diagnostics = solve_fn(
            u_ref, cones.w[active_mask], cones.h[active_mask], k_alpha[active_mask], self._cfg.a_max
        )
        if infeasible:
            u_safe = _max_braking(state.v, self._cfg.a_max)

        active_ids = self._orig_idx[cand_local[active_mask]]
        return SafetyFilterResult(
            u_safe=u_safe,
            active_splat_ids=active_ids,
            min_h=float(cones.h[active_mask].min()),
            infeasible=infeasible,
            solver_diagnostics=diagnostics,
        )
