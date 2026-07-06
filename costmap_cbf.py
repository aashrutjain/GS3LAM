"""Stage 3 entry point: build a CBFSafetyFilter from a safety_gsplat.ply +
config and run it. This is the online single-step piece: given a robot state
and a reference control, return a CBF-safe control. A future ROS2/TurtleBot4
node replaces the CLI loop below with its own odometry subscription / publish
loop around the same CBFSafetyFilter.step() call (see src/cbf/interfaces.py).

For the offline 3-mode comparison experiment, see eval_cbf_modes.py instead.

Usage:
    python costmap_cbf.py --config configs/cbf/room0_cbf.py \
        --p 0,0,0 --v 0.5,0,0 --u-ref 0,0,0 --semantic-mode none
"""

import argparse
import importlib.util
import sys

import numpy as np

from src.cbf.interfaces import RobotState
from src.cbf.ply_io import ZeroSafetyPolicy, load_splat_field
from src.cbf.qp_filter import CBFQPConfig, CBFSafetyFilter
from src.cbf.semantic_weighting import SemanticMode
from src.cbf.spatial_filter import SpatialFilterConfig


def load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("cbf_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def build_filter_from_config(cfg_dict: dict, ply_path_override: str | None, semantic_mode_override: str | None) -> CBFSafetyFilter:
    ply_path = ply_path_override or cfg_dict["ply_path"]
    zero_policy = ZeroSafetyPolicy(cfg_dict.get("zero_policy", "warn_only"))
    splats = load_splat_field(ply_path, zero_policy=zero_policy)

    semantic_mode = SemanticMode(semantic_mode_override or SemanticMode.NONE.value)
    sf = cfg_dict["spatial_filter"]
    spatial_cfg = SpatialFilterConfig(
        mode=sf["mode"],
        radius=sf["radius"],
        max_candidates=sf["max_candidates"],
        lookahead_horizon=sf["lookahead_horizon"],
        radius_margin=sf["radius_margin"],
        radius_cap=sf["radius_cap"],
        opacity_prune_thresh=sf["opacity_prune_thresh"],
    )
    qp_cfg = CBFQPConfig(
        a_max=cfg_dict["a_max"],
        k_alpha_base=cfg_dict["k_alpha_base"],
        robot_radius=cfg_dict["robot_radius"],
        semantic_mode=semantic_mode,
        alpha_f=cfg_dict["semantic"]["alpha_f"],
        cov_inflate_gamma=cfg_dict["semantic"].get("cov_inflate_gamma", 1.0),
        zero_policy=zero_policy,
        spatial_filter=spatial_cfg,
        solver=cfg_dict.get("solver", "scipy_slsqp"),
    )
    return CBFSafetyFilter(splats, qp_cfg)


def _parse_vec3(s: str) -> np.ndarray:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {s!r}")
    return np.array(parts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cbf/room0_cbf.py")
    parser.add_argument("--ply-path", default=None, help="Override config's ply_path")
    parser.add_argument("--semantic-mode", default=None, choices=[m.value for m in SemanticMode])
    parser.add_argument("--p", default="0,0,0", help="Robot position, 'x,y,z'")
    parser.add_argument("--v", default="0,0,0", help="Robot velocity, 'vx,vy,vz'")
    parser.add_argument("--u-ref", default="0,0,0", help="Reference control, 'ax,ay,az'")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    filt = build_filter_from_config(cfg_dict, args.ply_path, args.semantic_mode)

    state = RobotState(p=_parse_vec3(args.p), v=_parse_vec3(args.v))
    u_ref = _parse_vec3(args.u_ref)
    result = filt.step(state, u_ref)

    print(f"u_safe = {result.u_safe}")
    print(f"active splats = {result.active_splat_ids.size}")
    print(f"min_h = {result.min_h}")
    print(f"infeasible = {result.infeasible}")
    print(f"solver_diagnostics = {result.solver_diagnostics}")


if __name__ == "__main__":
    sys.exit(main())
