"""EXPERIMENT B -- end-to-end hazard-gain sweep on the pre-registered head-on scene.

NONE and ALPHA_SCALE differ only in the effective k_alpha applied to the hazard splat
(k_alpha_base vs k_alpha_base*0.1, given hazard_safety=0.1). This sweeps intermediate
gains between those two endpoints, full rollout each time, and reports severity,
time_ratio and path_length_ratio -- the trajectory-level severity-monotonicity
counterfactual left unrun by the 2026-08-02 trace (which swept gain at ONE state only).

Because configs/cbf/room0_cbf.py sets alpha_f = identity and k_alpha_base = 1.0, the
hazard's effective gain under ALPHA_SCALE is exactly its safety value, so the sweep is
expressed as a hazard_safety sweep and needs no library change. Endpoints double as
consistency checks: gain 1.0 must reproduce NONE, gain 0.1 must reproduce ALPHA_SCALE.

Measurement only. Run:  python3 scripts/exp_b_gain_sweep.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import eval_cbf_modes as E
from src.cbf.dynamics import DoubleIntegratorState, pd_reference_controller
from src.cbf.interfaces import RobotState
from src.cbf.ply_io import ZeroSafetyPolicy, load_splat_field
from src.cbf.qp_filter import CBFSafetyFilter
from src.cbf.semantic_weighting import SemanticMode
from src.cbf.sim import rollout

SCENE = "scenes/prereg_headon_seed42.ply"
CONFIG = "configs/cbf/room0_cbf.py"
START = np.array([-2.5, 0.0, 0.0], dtype=np.float32)
GOAL = np.array([2.5, 0.0, 0.0], dtype=np.float32)
HAZARD_ID = 0
GAINS = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
# committed reference values, PROGRESS.md "Pre-registered head-on scene" (2026-08-02)
REF = {"NONE": (-0.1330, 1.1418, 1.0003), "ALPHA_SCALE": (-0.4251, 1.0597, 0.9999)}


def window(filt, cfg, sim, pd):
    """Hazard-active window + active-set composition for one mode's own rollout."""
    r = rollout(filt, p0=START, v0=np.zeros(3, np.float32), goal=GOAL, dt=sim["dt"],
                a_max=cfg["a_max"], kp=pd["kp"], kd=pd["kd"],
                max_steps=sim["max_steps"], goal_tol=sim["goal_tol"])
    act, con, collar = [], 0, 0
    for k in range(len(r.min_h_history)):
        st = DoubleIntegratorState(p=r.trajectory_p[k], v=r.trajectory_v[k])
        u_ref = pd_reference_controller(st, GOAL, pd["kp"], pd["kd"], cfg["a_max"])
        ids = np.asarray(filt.step(RobotState(p=st.p, v=st.v), u_ref).active_splat_ids)
        if ids.size:
            con += 1
            if HAZARD_ID in ids:
                act.append(k)
            if (ids != HAZARD_ID).any():
                collar += 1
    return (act[0], act[-1], len(act), con, collar) if act else (-1, -1, 0, con, collar)


def main():
    cfg = E.load_config(CONFIG)
    cfg["solver"] = "clarabel"
    zp = ZeroSafetyPolicy(cfg.get("zero_policy", "warn_only"))
    sp = load_splat_field(SCENE, zero_policy=zp)
    hc = (START + GOAL) / 2.0
    sim, pd = cfg["sim"], cfg["pd"]

    # shared true-geometry grading filter, built once (NONE never reads `safety`,
    # so it stays valid as safety_raw is rewritten across the sweep)
    sp.safety_raw = E.synthesize_hazard_safety(sp.xyz, hc, 0.5, 0.1, 1.0)
    sp.ambiguous_zero_mask = None
    sp.zero_fraction = None
    base = CBFSafetyFilter(sp, E.build_qp_cfg(cfg, SemanticMode.NONE, zp, 1.0))
    for lbl, pt in (("start", START), ("goal", GOAL)):
        if E.verify_collision_free(pt, base) < 0:
            raise SystemExit(f"{lbl} inside ellipsoid")

    orc = rollout(E.PassthroughFilter(), p0=START, v0=np.zeros(3, np.float32), goal=GOAL,
                  dt=sim["dt"], a_max=cfg["a_max"], kp=pd["kp"], kd=pd["kd"],
                  max_steps=sim["max_steps"], goal_tol=sim["goal_tol"])
    ot = orc.time_to_goal or (sim["max_steps"] * sim["dt"])

    print("=== EXPERIMENT B: end-to-end hazard-gain sweep ===")
    print(f"scene={SCENE}  config={CONFIG}  solver={cfg['solver']}  "
          f"k_alpha_base={cfg['k_alpha_base']}  alpha_f=identity")
    print(f"oracle_time={ot:.4f}s   effective hazard gain = k_alpha_base * hazard_safety\n")

    print(f"{'mode':>12} {'k_haz':>7} {'reached':>8} {'severity':>10} {'time_ratio':>11} "
          f"{'plen_ratio':>11} {'near_miss':>10} {'infeas':>7} {'haz_window':>12} {'collar_steps':>13}")
    print("-" * 118)
    rows = []

    # NONE reference row
    sp.safety_raw = E.synthesize_hazard_safety(sp.xyz, hc, 0.5, 0.1, 1.0)
    rN = E.run_mode("NONE", sp, cfg, SemanticMode.NONE, zp, 1.0, START, GOAL, ot, 0.3, base)
    wN = window(CBFSafetyFilter(sp, E.build_qp_cfg(cfg, SemanticMode.NONE, zp, 1.0)), cfg, sim, pd)
    print(f"{'NONE':>12} {1.0:>7.2f} {str(rN['reached_goal']):>8} {rN['collision_severity']:>10.4f} "
          f"{rN['time_ratio']:>11.4f} {rN['path_length_ratio']:>11.4f} {rN['near_miss_events']:>10} "
          f"{rN['infeasible_count']:>7} {f'{wN[0]}..{wN[1]} ({wN[2]})':>12} {wN[4]:>13}")
    print("-" * 118)

    for gain in GAINS:
        sp.safety_raw = E.synthesize_hazard_safety(sp.xyz, hc, 0.5, gain, 1.0)
        r = E.run_mode("ALPHA_SCALE", sp, cfg, SemanticMode.ALPHA_SCALE, zp, 1.0,
                       START, GOAL, ot, 0.3, base)
        w = window(CBFSafetyFilter(sp, E.build_qp_cfg(cfg, SemanticMode.ALPHA_SCALE, zp, 1.0)),
                   cfg, sim, pd)
        rows.append((gain, r, w))
        print(f"{'ALPHA_SCALE':>12} {gain:>7.2f} {str(r['reached_goal']):>8} "
              f"{r['collision_severity']:>10.4f} {r['time_ratio']:>11.4f} "
              f"{r['path_length_ratio']:>11.4f} {r['near_miss_events']:>10} "
              f"{r['infeasible_count']:>7} {f'{w[0]}..{w[1]} ({w[2]})':>12} {w[4]:>13}")

    # ---- endpoint consistency checks ----
    print("\n=== endpoint consistency checks vs committed 2026-08-02 values ===")
    ok = True
    for gain, ref_lbl in ((1.0, "NONE"), (0.1, "ALPHA_SCALE")):
        r = next(x[1] for x in rows if abs(x[0] - gain) < 1e-9)
        s, t, p = REF[ref_lbl]
        d = (abs(r["collision_severity"] - s), abs(r["time_ratio"] - t), abs(r["path_length_ratio"] - p))
        good = d[0] < 5e-4 and d[1] < 5e-4 and d[2] < 5e-4
        ok &= good
        print(f"  gain {gain:.2f} vs {ref_lbl:>12}: severity {r['collision_severity']:.4f} [{s}] "
              f"time_ratio {r['time_ratio']:.4f} [{t}] plen {r['path_length_ratio']:.4f} [{p}]  "
              f"-> {'PASS' if good else 'FAIL'}")
    print(f"  ALPHA_SCALE@1.0 identical to NONE row: "
          f"{abs(rows[-1][1]['collision_severity'] - rN['collision_severity']) < 1e-9}")
    print(f"  OVERALL {'PASS' if ok else 'FAIL'}")

    # ---- monotonicity ----
    print("\n=== monotonicity in gain (gain increasing 0.1 -> 1.0) ===")
    for key in ("collision_severity", "time_ratio", "path_length_ratio"):
        vals = [r[key] for _, r, _ in rows]
        inc = all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
        dec = all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
        kind = "monotonic increasing" if inc else "monotonic decreasing" if dec else "NON-MONOTONIC"
        print(f"  {key:>18}: {[round(v, 4) for v in vals]}  -> {kind}")
        if not (inc or dec):
            d = np.diff(vals)
            flips = [i for i in range(len(d) - 1) if d[i] * d[i + 1] < 0]
            print(f"  {'':>18}  sign flips after gain(s): {[GAINS[i + 1] for i in flips]}")
    print("\n  step-to-step severity deltas:")
    sev = [r["collision_severity"] for _, r, _ in rows]
    for i in range(len(sev) - 1):
        print(f"    {GAINS[i]:.2f} -> {GAINS[i+1]:.2f}: {sev[i+1] - sev[i]:+.4f}")


if __name__ == "__main__":
    main()
