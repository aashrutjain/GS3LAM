# Stage 3 CBF/costmap config for the Replica room0 scene. Mirrors the plain-dict
# style of configs/Replica/room0.py.
#
# ply_path below is a placeholder -- Stage 1/2 write into a timestamped run
# directory (see vlm_safety_score.py's PLY_PATH/OUTPUT_PLY_PATH), so this must
# be pointed at an actual run's safety_gsplat.ply before use. costmap_cbf.py /
# eval_cbf_modes.py also accept --ply-path on the CLI to override this.

scene_name = "room0"
run_name = "PLACEHOLDER"  # e.g. "260422-13:33:16" -- set to an actual Stage 1/2 run

ply_path = f"./logs/Replica/{scene_name}_seed1/{run_name}/safety_gsplat.ply"

config = dict(
    ply_path=ply_path,

    # Robot / actuator parameters. robot_radius is a TurtleBot4 footprint
    # placeholder (m) -- CONFIRM against the actual hardware spec before
    # trusting Minkowski-inflation results.
    robot_radius=0.16,
    a_max=1.0,  # m/s^2

    # CBF gain (k_alpha, renamed from the paper's p_k -- see collision_cone.py).
    # Larger = more permissive (weaker) barrier.
    k_alpha_base=1.0,

    # PD reference controller (dynamics.pd_reference_controller) -- a toy
    # tracker for the eval harness, not a real planner.
    pd=dict(kp=1.0, kd=2.0),

    # Sim rollout parameters.
    sim=dict(dt=0.05, max_steps=2000, goal_tol=0.1),

    # Spatial prefilter (spatial_filter.SpatialFilterConfig field names).
    # Placeholder radius/count values -- NOT measured against the real
    # trained room0 scene's bounding box, confirm before a real run.
    spatial_filter=dict(
        mode="radius",
        radius=None,  # None -> adaptive, see lookahead_horizon/radius_margin/radius_cap
        max_candidates=200,
        lookahead_horizon=2.0,
        radius_margin=1.0,
        radius_cap=3.0,
        opacity_prune_thresh=0.1,
    ),

    # Zero-safety handling (see ply_io.ZeroSafetyPolicy) -- WARN_ONLY is the
    # non-silent default; do not flip to TREAT_AS_NEUTRAL without reading
    # ply_io.py's module docstring first.
    zero_policy="warn_only",

    # Semantic weighting sweep (semantic_weighting.SemanticMode) -- NEITHER
    # validated per ARCHITECTURE.md Sec 2.3. alpha_f is the f(safety) map for
    # ALPHA_SCALE (identity default); cov_inflate_gamma is the gamma for
    # COV_INFLATE. Both are initial values for the Phase A/B sweep described
    # in the Stage 3 design plan, not settled research choices.
    semantic=dict(
        alpha_f=lambda s: s,
        cov_inflate_gamma_sweep=[0.5, 1.0, 2.0],
    ),

    # Omitted deliberately: defers to CBFQPConfig.solver's default ("clarabel" as of
    # 2026-07-18). Set explicitly to "scipy_slsqp" to pin the fallback backend.
)
