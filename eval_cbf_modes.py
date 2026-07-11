"""Stage 3 evaluation: does semantic weighting of the collision-cone CBF
actually help over the pure-geometric baseline? (ARCHITECTURE.md Sec 1,
"Execution" sub-question; Sec 2.3's open design question.)

Runs three SemanticMode configurations (NONE / ALPHA_SCALE / COV_INFLATE)
through an identical double-integrator rollout (src/cbf/sim.py) between the
same start/goal, and tabulates collision severity, near-miss frequency, and
path efficiency (src/cbf/metrics.py) against an oracle (CBF disabled) run.

Two-phase evaluation, per ARCHITECTURE.md Sec 2.3 / PROGRESS.md:
  - Phase A (--phase-a): overrides the loaded safety column with a synthetic
    hazard directly on the start->goal line, to sanity-check that the two
    weighting strategies actually do what the math says (ALPHA_SCALE brakes
    harder, longitudinal only; COV_INFLATE routes wider, lateral too) before
    trusting any real-scene numbers.
  - Phase B (default, no flag): uses the safety column as loaded. Currently
    blocked on a real research signal by the Stage 2 classifier bug in
    PROGRESS.md (today's per-splat class IDs, and therefore safety scores,
    are not semantically meaningful) -- this mode will still run, but the
    tabulated numbers should not be read as a real result until that bug is
    fixed upstream.

Usage:
    python eval_cbf_modes.py --config configs/cbf/room0_cbf.py \
        --ply-path <path/to/safety_gsplat.ply> --phase-a
"""

import argparse
import importlib.util

import numpy as np

from src.cbf.collision_cone import effective_c
from src.cbf.dynamics import DoubleIntegratorState
from src.cbf.interfaces import RobotState, SafetyFilterResult
from src.cbf.metrics import collision_severity, mahalanobis_signed_distance, near_miss_events, path_efficiency
from src.cbf.ply_io import SplatField, ZeroSafetyPolicy, load_splat_field
from src.cbf.qp_filter import CBFQPConfig, CBFSafetyFilter
from src.cbf.semantic_weighting import SemanticMode
from src.cbf.sim import rollout
from src.cbf.spatial_filter import SpatialFilterConfig


class PassthroughFilter:
    """CBF-disabled oracle: passes the (already a_max-clipped) reference
    control straight through. Used only to compute oracle_time for the
    path-efficiency denominator -- never a candidate mode in the comparison.
    """

    def step(self, state: RobotState, u_ref: np.ndarray) -> SafetyFilterResult:
        return SafetyFilterResult(
            u_safe=u_ref,
            active_splat_ids=np.array([], dtype=np.int64),
            min_h=float("inf"),
            infeasible=False,
            solver_diagnostics={"note": "passthrough oracle, CBF disabled"},
        )


def synthesize_hazard_safety(
    xyz: np.ndarray,
    hazard_center: np.ndarray,
    hazard_radius: float,
    hazard_safety: float = 0.1,
    background_safety: float = 1.0,
) -> np.ndarray:
    """Phase A helper: override safety with a single deliberate low-safety
    hazard, everything else near-safe. Isolates "does the weighting math do
    what it should" from "is the upstream semantic labeling correct."
    """
    dist = np.linalg.norm(xyz - hazard_center[None, :], axis=-1)
    safety = np.where(dist <= hazard_radius, hazard_safety, background_safety)
    return safety.astype(np.float32)


def load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("cbf_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def build_spatial_cfg(cfg_dict: dict) -> SpatialFilterConfig:
    sf = cfg_dict["spatial_filter"]
    return SpatialFilterConfig(
        mode=sf["mode"],
        radius=sf["radius"],
        max_candidates=sf["max_candidates"],
        lookahead_horizon=sf["lookahead_horizon"],
        radius_margin=sf["radius_margin"],
        radius_cap=sf["radius_cap"],
        opacity_prune_thresh=sf["opacity_prune_thresh"],
    )


def build_qp_cfg(cfg_dict: dict, semantic_mode: SemanticMode, zero_policy: ZeroSafetyPolicy, cov_gamma: float) -> CBFQPConfig:
    return CBFQPConfig(
        a_max=cfg_dict["a_max"],
        k_alpha_base=cfg_dict["k_alpha_base"],
        robot_radius=cfg_dict["robot_radius"],
        semantic_mode=semantic_mode,
        alpha_f=cfg_dict["semantic"]["alpha_f"],
        cov_inflate_gamma=cov_gamma,
        zero_policy=zero_policy,
        spatial_filter=build_spatial_cfg(cfg_dict),
        solver=cfg_dict.get("solver", "scipy_slsqp"),
    )


def default_start_goal(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Heuristic default along the scene's longest horizontal extent, offset
    inward from the bounding box. NOT guaranteed collision-free -- verified
    (and rejected with a clear error) by the caller before use. Pass --start/
    --goal explicitly once the real room0 scene geometry is known.
    """
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    center = (lo + hi) / 2.0
    extent = hi - lo
    axis = int(np.argmax(extent[:2]))  # longest of x/y, keep z at scene center
    offset = np.zeros(3, dtype=np.float32)
    offset[axis] = extent[axis] * 0.3
    start = (center - offset).astype(np.float32)
    goal = (center + offset).astype(np.float32)
    return start, goal


def verify_collision_free(point: np.ndarray, filt: CBFSafetyFilter) -> float:
    c_m = effective_c(filt.c_base, filt.robot_radius, filt.s_min)
    d = mahalanobis_signed_distance(point[None, :], filt.xyz, filt.A, c_m)
    return float(d[0])


def run_mode(
    label: str,
    splats: SplatField,
    cfg_dict: dict,
    semantic_mode: SemanticMode,
    zero_policy: ZeroSafetyPolicy,
    cov_gamma: float,
    start: np.ndarray,
    goal: np.ndarray,
    oracle_time: float,
    near_miss_thresh: float,
    geom_filt: CBFSafetyFilter,
) -> dict:
    # geom_filt is a single SemanticMode.NONE filter, built once by the caller and shared
    # across all three mode calls. It supplies the ellipsoid (.A/.s_min/.xyz/.c_base) for
    # collision_severity/near_miss_events below. `filt` (this mode's own filter, built with
    # this mode's semantic weighting) still drives the rollout -- routing decisions during
    # the sim should reflect what each mode actually sees. But for COV_INFLATE, filt.A is
    # A/scale (semantic_weighting.inflate_covariance), i.e. the ellipsoid the controller was
    # dodging, NOT the true splat boundary -- grading COV_INFLATE's trajectory against its own
    # inflated ellipsoid would make its severity/near-miss numbers artificially good relative
    # to NONE/ALPHA_SCALE (which are graded against the true boundary), regardless of whether
    # it actually kept the robot further from the real object. Evaluation must use the same
    # true geometry for all three modes; only the controller differs by mode. (Found and
    # fixed 2026-07-11 -- see PROGRESS.md.)
    qp_cfg = build_qp_cfg(cfg_dict, semantic_mode, zero_policy, cov_gamma)
    filt = CBFSafetyFilter(splats, qp_cfg)

    sim_cfg = cfg_dict["sim"]
    pd_cfg = cfg_dict["pd"]
    result = rollout(
        filt,
        p0=start,
        v0=np.zeros(3, dtype=np.float32),
        goal=goal,
        dt=sim_cfg["dt"],
        a_max=cfg_dict["a_max"],
        kp=pd_cfg["kp"],
        kd=pd_cfg["kd"],
        max_steps=sim_cfg["max_steps"],
        goal_tol=sim_cfg["goal_tol"],
    )

    c_m = effective_c(geom_filt.c_base, qp_cfg.robot_radius, geom_filt.s_min)
    min_dist_over_time = mahalanobis_signed_distance(result.trajectory_p, geom_filt.xyz, geom_filt.A, c_m)
    severity = collision_severity(min_dist_over_time)
    near_misses = near_miss_events(min_dist_over_time, near_miss_thresh)

    actual_time = result.time_to_goal if result.reached_goal else sim_cfg["max_steps"] * sim_cfg["dt"]
    straight_line_dist = float(np.linalg.norm(goal - start))
    efficiency = path_efficiency(result.trajectory_p, straight_line_dist, actual_time, oracle_time)

    return {
        "mode": label,
        "reached_goal": result.reached_goal,
        "infeasible_count": result.infeasible_count,
        "wall_clock_per_step_s": result.wall_clock_per_step,
        "collision_severity": severity,
        "near_miss_events": near_misses,
        **efficiency,
    }


def print_table(rows: list[dict]) -> None:
    cols = [
        "mode", "reached_goal", "collision_severity", "near_miss_events",
        "path_length_ratio", "time_ratio", "infeasible_count", "wall_clock_per_step_s",
    ]
    widths = {c: max(len(c), *(len(f"{r[c]}") for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(f"{r[c]}".ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cbf/room0_cbf.py")
    parser.add_argument("--ply-path", default=None, help="Override config's ply_path")
    parser.add_argument("--start", default=None, help="'x,y,z', default: heuristic from scene bbox")
    parser.add_argument("--goal", default=None, help="'x,y,z', default: heuristic from scene bbox")
    parser.add_argument("--phase-a", action="store_true", help="Override safety with a synthetic hazard")
    parser.add_argument("--hazard-radius", type=float, default=0.5, help="Phase A hazard radius (m)")
    parser.add_argument("--hazard-safety", type=float, default=0.1, help="Phase A hazard safety score")
    parser.add_argument("--cov-gamma", type=float, default=1.0, help="cov_inflate_gamma for COV_INFLATE")
    parser.add_argument("--near-miss-thresh", type=float, default=0.3, help="meters, for near_miss_events")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    ply_path = args.ply_path or cfg_dict["ply_path"]
    zero_policy = ZeroSafetyPolicy(cfg_dict.get("zero_policy", "warn_only"))
    splats = load_splat_field(ply_path, zero_policy=zero_policy)

    if args.start is not None and args.goal is not None:
        start = np.array([float(x) for x in args.start.split(",")], dtype=np.float32)
        goal = np.array([float(x) for x in args.goal.split(",")], dtype=np.float32)
    else:
        start, goal = default_start_goal(splats.xyz)
        print(
            f"[eval_cbf_modes] No --start/--goal given; using heuristic scene-bbox "
            f"midline start={start} goal={goal}. This is NOT guaranteed collision-free "
            "-- verifying against baseline geometry now."
        )

    if args.phase_a:
        hazard_center = (start + goal) / 2.0
        print(
            f"[eval_cbf_modes] Phase A: overriding safety column with a synthetic "
            f"hazard at {hazard_center} (radius={args.hazard_radius}, "
            f"safety={args.hazard_safety}), background=1.0."
        )
        splats.safety_raw = synthesize_hazard_safety(
            splats.xyz, hazard_center, args.hazard_radius, args.hazard_safety, background_safety=1.0
        )
        splats.ambiguous_zero_mask = None
        splats.zero_fraction = None
    else:
        print(
            "[eval_cbf_modes] Phase B (real safety column, no override). Per PROGRESS.md, "
            "today's safety scores are not yet semantically meaningful (Stage 2 classifier "
            "bug) -- treat these numbers as a pipeline smoke test, not a research result, "
            "until that bug is fixed."
        )

    # Verify start/goal against the pure-geometric baseline (never the semantic modes --
    # a bad start/goal should fail identically regardless of weighting).
    baseline_cfg = build_qp_cfg(cfg_dict, SemanticMode.NONE, zero_policy, args.cov_gamma)
    baseline_filt = CBFSafetyFilter(splats, baseline_cfg)
    for label, point in (("start", start), ("goal", goal)):
        d = verify_collision_free(point, baseline_filt)
        if d < 0:
            raise SystemExit(
                f"[eval_cbf_modes] {label}={point} is inside a splat's confidence ellipsoid "
                f"(signed distance={d:.3f} < 0). Pass an explicit --start/--goal outside "
                "obstacles for this scene."
            )

    oracle_result = rollout(
        PassthroughFilter(),
        p0=start,
        v0=np.zeros(3, dtype=np.float32),
        goal=goal,
        dt=cfg_dict["sim"]["dt"],
        a_max=cfg_dict["a_max"],
        kp=cfg_dict["pd"]["kp"],
        kd=cfg_dict["pd"]["kd"],
        max_steps=cfg_dict["sim"]["max_steps"],
        goal_tol=cfg_dict["sim"]["goal_tol"],
    )
    if not oracle_result.reached_goal:
        print(
            "[eval_cbf_modes] WARNING: oracle (CBF-disabled) run did not reach the goal "
            f"within max_steps={cfg_dict['sim']['max_steps']} -- time_ratio below will be "
            "unreliable. Check kp/kd/a_max/dt in the config."
        )
    oracle_time = oracle_result.time_to_goal or (cfg_dict["sim"]["max_steps"] * cfg_dict["sim"]["dt"])

    rows = []
    for label, mode in (("NONE", SemanticMode.NONE), ("ALPHA_SCALE", SemanticMode.ALPHA_SCALE), ("COV_INFLATE", SemanticMode.COV_INFLATE)):
        print(f"[eval_cbf_modes] Running mode={label}...")
        rows.append(
            run_mode(
                label, splats, cfg_dict, mode, zero_policy, args.cov_gamma,
                start, goal, oracle_time, args.near_miss_thresh, baseline_filt,
            )
        )

    print()
    print_table(rows)


if __name__ == "__main__":
    main()
