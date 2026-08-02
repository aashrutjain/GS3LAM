"""PRE-REGISTERED head-on sanity-check scene for Results section 5.1.

The criterion below is fixed from geometry BEFORE any rollout is run, and the
scene parameters are DERIVED from it -- they are not chosen by looking at any
output ratio. The scene is generated and evaluated exactly once. Whatever ratio
results is the finding; there is no outcome-driven adjustment of hazard size,
placement, or collar radius.

Why a new scene: the first reconstruction attempt (isolated hazard, r_phys~0.50m,
3cm offset, collar pushed out of range) was too mild -- both NONE and ALPHA_SCALE
clipped past the single hazard at full speed (time ratio 1.06 each, path ~1.00),
so it never exercised the deceleration / lateral-departure regime the original
(unpersisted) sanity scene was built to demonstrate.

CRITERION (geometric, fixed in advance):
  (1) BLOCKING: the hazard's Minkowski-inflated boundary must cover the direct
      start->goal path. Hazard isotropic scale s_h = 0.10 and robot radius
      rho = 0.16 give r_phys_hazard = c_base*s_h + rho. The path passes only
      cluster_offset_y = 0.03 m from the hazard center, and 0.03 << r_phys_hazard,
      so a straight run penetrates the inflated boundary by (r_phys_hazard - 0.03).
      (Hazard scale and offset are the validated disambiguation-generator
      defaults, not tuned here.)
  (2) NARROW CLEARANCE: the robot-center lateral corridor between the hazard's
      inflated boundary and the collar's inner inflated boundary,
        W_center = R_collar - c_base*(s_h + s_c) - 2*rho,
      is set to HALF the robot footprint radius: W_center = rho/2 = 0.08 m.
      A free band narrower than the robot's own radius makes a small full-speed
      sidestep geometrically impossible; clearing the hazard requires a real
      deceleration and/or a real lateral departure. R_collar is solved from this.

Both backends are run to confirm agreement (everything else is reported under
Clarabel).
"""
import sys
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, ".")
from scripts.gen_cbf_synthetic_scene import build_scene, write_ply, C_BASE, ROBOT_RADIUS
from src.cbf.ply_io import ZeroSafetyPolicy, load_splat_field
from src.cbf.qp_filter import CBFSafetyFilter
from src.cbf.semantic_weighting import SemanticMode
from src.cbf.sim import rollout
import eval_cbf_modes as E

# ---- derive scene parameters FROM the criterion (no outcome inspection) ----
rho = ROBOT_RADIUS            # 0.16 m, robot footprint radius (config robot_radius)
c_base = C_BASE               # sqrt(chi2.ppf(0.99, df=3))
s_h, s_c = 0.10, 0.04         # generator hazard / clutter isotropic scales
offset_y = 0.03               # generator default; breaks exact-symmetry degeneracy

r_phys_hazard = c_base * s_h + rho
r_phys_collar = c_base * s_c + rho
W_target = rho / 2.0                                   # 0.08 m  (the pre-registered clearance)
R_collar = W_target + c_base * (s_h + s_c) + 2.0 * rho # solved from W_center = R_collar - c_base(s_h+s_c) - 2rho

print("=== PRE-REGISTERED CRITERION -> DERIVED PARAMETERS ===")
print(f"c_base = sqrt(chi2.ppf(0.99,3))      = {c_base:.5f}")
print(f"robot radius rho                     = {rho:.3f} m")
print(f"hazard scale s_h / clutter s_c       = {s_h} / {s_c}")
print(f"r_phys_hazard = c_base*s_h + rho     = {r_phys_hazard:.4f} m  (inflated blocking radius)")
print(f"path offset from hazard center       = {offset_y:.3f} m  ({offset_y/r_phys_hazard:.2%} of r_phys_hazard)")
print(f"  -> straight-line penetration depth = {r_phys_hazard - offset_y:.4f} m  (BLOCKING condition met)")
print(f"pre-registered clearance W_center    = rho/2 = {W_target:.3f} m  ({W_target/rho:.0%} of robot radius)")
print(f"  -> DERIVED collar radius R_collar  = {R_collar:.4f} m")
print(f"  (check) W_center back-computed      = {R_collar - c_base*(s_h+s_c) - 2*rho:.4f} m")
print()

START = np.array([-2.5, 0.0, 0.0], dtype=np.float32)
GOAL = np.array([2.5, 0.0, 0.0], dtype=np.float32)
SCENE = "scenes/prereg_headon_seed42.ply"
write_ply(build_scene(R_collar, seed=42, cluster_offset_y=offset_y), SCENE)
print(f"wrote scene -> {SCENE}  (collar_radius={R_collar:.4f}, seed=42, offset={offset_y})\n")


def run(solver):
    cfg = E.load_config("configs/cbf/room0_cbf.py")
    cfg["solver"] = solver
    zp = ZeroSafetyPolicy(cfg.get("zero_policy", "warn_only"))
    sp = load_splat_field(SCENE, zero_policy=zp)
    hc = (START + GOAL) / 2.0
    sp.safety_raw = E.synthesize_hazard_safety(sp.xyz, hc, 0.5, 0.1, 1.0)
    sp.ambiguous_zero_mask = None
    sp.zero_fraction = None
    base = CBFSafetyFilter(sp, E.build_qp_cfg(cfg, SemanticMode.NONE, zp, 1.0))
    for lbl, pt in (("start", START), ("goal", GOAL)):
        if E.verify_collision_free(pt, base) < 0:
            raise SystemExit(f"{lbl} inside ellipsoid")
    orc = rollout(E.PassthroughFilter(), p0=START, v0=np.zeros(3, np.float32), goal=GOAL,
                  dt=cfg["sim"]["dt"], a_max=cfg["a_max"], kp=cfg["pd"]["kp"], kd=cfg["pd"]["kd"],
                  max_steps=cfg["sim"]["max_steps"], goal_tol=cfg["sim"]["goal_tol"])
    ot = orc.time_to_goal or (cfg["sim"]["max_steps"] * cfg["sim"]["dt"])
    out = {}
    for lbl, mode in (("NONE", SemanticMode.NONE), ("ALPHA_SCALE", SemanticMode.ALPHA_SCALE),
                      ("COV_INFLATE", SemanticMode.COV_INFLATE)):
        out[lbl] = E.run_mode(lbl, sp, cfg, mode, zp, 1.0, START, GOAL, ot, 0.3, base)
    return out


print(f"{'solver':>12} {'mode':>12} {'reached':>7} {'plen_ratio':>11} {'time_ratio':>11} "
      f"{'severity':>9} {'near_miss':>9} {'infeas':>6}")
print("-" * 84)
results = {}
for solver in ("clarabel", "scipy_slsqp"):
    results[solver] = run(solver)
    for mode in ("NONE", "ALPHA_SCALE", "COV_INFLATE"):
        r = results[solver][mode]
        print(f"{solver:>12} {mode:>12} {str(r['reached_goal']):>7} "
              f"{r['path_length_ratio']:>11.4f} {r['time_ratio']:>11.4f} "
              f"{r['collision_severity']:>9.4f} {r['near_miss_events']:>9} {r['infeasible_count']:>6}")
    print()

# backend-agreement check on the headline (NONE / ALPHA_SCALE)
print("=== backend agreement (clarabel vs scipy_slsqp) ===")
for mode in ("NONE", "ALPHA_SCALE", "COV_INFLATE"):
    c, s = results["clarabel"][mode], results["scipy_slsqp"][mode]
    dt = abs(c["time_ratio"] - s["time_ratio"])
    dp = abs(c["path_length_ratio"] - s["path_length_ratio"])
    print(f"  {mode:>12}: |dtime_ratio|={dt:.2e}  |dplen_ratio|={dp:.2e}  "
          f"reached_match={c['reached_goal']==s['reached_goal']}")
