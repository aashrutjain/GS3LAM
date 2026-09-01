"""EXPERIMENT A -- per-step matched-state QP re-solve across the hazard-active window.

Extends the single matched-state check at step 13 (PROGRESS.md, "Constraint-activation
trace at the NONE/ALPHA_SCALE divergence", 2026-08-02) to every step of the active
window. For each mode, at that mode's OWN recorded (p, v), the QP is re-solved with the
OTHER mode's gain. Nothing is forward-integrated: each step is an independent "what
would this state's correction have been under the other gain".

Measurement only -- imports committed helpers, changes no library math. Committed
(rather than left as a scratch driver) so these numbers can be re-derived; see the
PROGRESS.md entry for why that departs from the ephemeral-driver convention.

Run:  python3 scripts/exp_a_matched_state_window.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import eval_cbf_modes as E
from src.cbf.collision_cone import compute_collision_cones, effective_c
from src.cbf.dynamics import DoubleIntegratorState, pd_reference_controller
from src.cbf.interfaces import RobotState
from src.cbf.metrics import mahalanobis_signed_distance
from src.cbf.ply_io import ZeroSafetyPolicy, load_splat_field
from src.cbf.qp_filter import CBFSafetyFilter
from src.cbf.semantic_weighting import SemanticMode
from src.cbf.sim import rollout

SCENE = "scenes/prereg_headon_seed42.ply"
CONFIG = "configs/cbf/room0_cbf.py"
START = np.array([-2.5, 0.0, 0.0], dtype=np.float32)
GOAL = np.array([2.5, 0.0, 0.0], dtype=np.float32)
HAZARD_ID = 0          # scripts/gen_cbf_synthetic_scene.build_scene puts the hazard at index 0
START_STEP = 13        # the last state before the two trajectories diverge

np.set_printoptions(precision=4, suppress=True)


def build(cfg, sp, zp, mode):
    return CBFSafetyFilter(sp, E.build_qp_cfg(cfg, mode, zp, 1.0))


def main():
    cfg = E.load_config(CONFIG)
    cfg["solver"] = "clarabel"
    zp = ZeroSafetyPolicy(cfg.get("zero_policy", "warn_only"))
    sp = load_splat_field(SCENE, zero_policy=zp)
    hc = (START + GOAL) / 2.0
    sp.safety_raw = E.synthesize_hazard_safety(sp.xyz, hc, 0.5, 0.1, 1.0)
    sp.ambiguous_zero_mask = None
    sp.zero_fraction = None

    filts = {"NONE": build(cfg, sp, zp, SemanticMode.NONE),
             "ALPHA_SCALE": build(cfg, sp, zp, SemanticMode.ALPHA_SCALE)}
    other = {"NONE": "ALPHA_SCALE", "ALPHA_SCALE": "NONE"}
    geom = filts["NONE"]   # shared true geometry for grading, per eval_cbf_modes.run_mode

    sim, pd = cfg["sim"], cfg["pd"]
    print("=== EXPERIMENT A: per-step matched-state re-solve ===")
    print(f"scene={SCENE}  config={CONFIG}  solver={cfg['solver']}")
    print(f"splats={sp.n}  hazard id={HAZARD_ID} at {sp.xyz[HAZARD_ID]}  "
          f"safety: n(0.1)={int((sp.safety_raw < 0.5).sum())} n(1.0)={int((sp.safety_raw >= 0.5).sum())}")
    print(f"k_alpha[hazard]: NONE={cfg['k_alpha_base']:.5f}  "
          f"ALPHA_SCALE={cfg['k_alpha_base'] * float(sp.safety_raw[HAZARD_ID]):.5f}\n")

    # ---- roll out each mode to recover its actual trajectory ----
    traj = {}
    for lbl, f in filts.items():
        r = rollout(f, p0=START, v0=np.zeros(3, np.float32), goal=GOAL, dt=sim["dt"],
                    a_max=cfg["a_max"], kp=pd["kp"], kd=pd["kd"],
                    max_steps=sim["max_steps"], goal_tol=sim["goal_tol"])
        traj[lbl] = r
        c_m = effective_c(geom.c_base, cfg["robot_radius"], geom.s_min)
        d = mahalanobis_signed_distance(r.trajectory_p, geom.xyz, geom.A, c_m)
        print(f"{lbl:>12}: steps={len(r.min_h_history)} reached={r.reached_goal} "
              f"severity={d.min():.4f} (argmin step {int(d.argmin())}) infeas={r.infeasible_count}")
    print()

    # ---- recompute each mode's hazard-active window (do not assume 13..54 / 13..49) ----
    c_m_g = effective_c(geom.c_base, cfg["robot_radius"], geom.s_min)
    haz_A, haz_mu = geom.A[HAZARD_ID], geom.xyz[HAZARD_ID]
    haz_cm = c_m_g[HAZARD_ID]

    windows, dominance = {}, {}
    for lbl, r in traj.items():
        act, constrained, haz_dom = [], 0, 0
        for k in range(len(r.min_h_history)):
            st = DoubleIntegratorState(p=r.trajectory_p[k], v=r.trajectory_v[k])
            u_ref = pd_reference_controller(st, GOAL, pd["kp"], pd["kd"], cfg["a_max"])
            res = filts[lbl].step(RobotState(p=st.p, v=st.v), u_ref)
            ids = np.asarray(res.active_splat_ids)
            if ids.size:
                constrained += 1
                if HAZARD_ID in ids:
                    act.append(k)
                    # is the hazard the min-h (dominant) splat this step?
                    cone = compute_collision_cones(st.p, st.v, geom.xyz[ids], geom.A[ids], c_m_g[ids])
                    if ids[int(np.argmin(cone.h))] == HAZARD_ID:
                        haz_dom += 1
        windows[lbl] = act
        dominance[lbl] = (haz_dom, constrained)
        print(f"{lbl:>12}: hazard active steps {act[0]}..{act[-1]} ({len(act)} steps); "
              f"deactivates at step {act[-1] + 1}")
        print(f"{'':>12}  hazard-dominant {haz_dom}/{constrained} constrained steps "
              f"({100.0 * haz_dom / constrained:.1f}%)")
    print()

    # ---- per-step matched-state re-solve ----
    summary = {}
    for lbl, r in traj.items():
        oth = other[lbl]
        last = windows[lbl][-1]
        rows = []
        print(f"--- family: {lbl}'s own recorded states, re-solved under {oth}'s gain ---")
        print(f"{'step':>5} {'speed':>8} {'dist_haz':>9} {'min_h':>12} {'nact':>5} {'haz':>4} "
              f"{'|u_own-uref|':>13} {'|u_oth-uref|':>13} {'ratio':>8} {'reduc%':>8}")
        for k in range(START_STEP, last + 1):
            p, v = r.trajectory_p[k], r.trajectory_v[k]
            st = DoubleIntegratorState(p=p, v=v)
            u_ref = pd_reference_controller(st, GOAL, pd["kp"], pd["kd"], cfg["a_max"])
            rs = RobotState(p=p, v=v)
            r_own = filts[lbl].step(rs, u_ref)
            r_oth = filts[oth].step(rs, u_ref)
            c_own = float(np.linalg.norm(r_own.u_safe - u_ref))
            c_oth = float(np.linalg.norm(r_oth.u_safe - u_ref))
            # ratio expressed as (low gain)/(high gain) so it is comparable across families
            lo, hi = (c_own, c_oth) if lbl == "ALPHA_SCALE" else (c_oth, c_own)
            ratio = lo / hi if hi > 1e-12 else float("nan")
            ids = np.asarray(r_own.active_splat_ids)
            dh = float(np.sqrt(max((p - haz_mu) @ haz_A @ (p - haz_mu), 0.0)) - haz_cm)
            rows.append((k, ratio, lo, hi, c_own, c_oth))
            print(f"{k:>5} {np.linalg.norm(v):>8.4f} {dh:>9.4f} {r_own.min_h:>12.4f} "
                  f"{ids.size:>5} {str(HAZARD_ID in ids):>4} {c_own:>13.6f} {c_oth:>13.6f} "
                  f"{ratio:>8.4f} {100.0 * (1.0 - ratio):>8.2f}")
        ratios = np.array([x[1] for x in rows])
        summary[lbl] = (rows, ratios)
        print(f"  n={len(rows)}  reduction%: mean={100 * (1 - ratios.mean()):.2f} "
              f"median={100 * (1 - np.median(ratios)):.2f} "
              f"min={100 * (1 - ratios.max()):.2f} max={100 * (1 - ratios.min()):.2f}")
        print(f"  ratio: first(step {rows[0][0]})={ratios[0]:.4f}  last(step {rows[-1][0]})={ratios[-1]:.4f}  "
              f"std={ratios.std():.4f}\n")

    # ---- step-13 cross-check against the committed 2026-08-02 numbers ----
    print("=== step-13 cross-check vs committed trace (1.276847 NONE / 0.994977 ALPHA_SCALE / 0.7792) ===")
    p, v = traj["NONE"].trajectory_p[13], traj["NONE"].trajectory_v[13]
    st = DoubleIntegratorState(p=p, v=v)
    u_ref = pd_reference_controller(st, GOAL, pd["kp"], pd["kd"], cfg["a_max"])
    cn = float(np.linalg.norm(filts["NONE"].step(RobotState(p=p, v=v), u_ref).u_safe - u_ref))
    ca = float(np.linalg.norm(filts["ALPHA_SCALE"].step(RobotState(p=p, v=v), u_ref).u_safe - u_ref))
    same13 = np.allclose(traj["NONE"].trajectory_p[13], traj["ALPHA_SCALE"].trajectory_p[13]) and \
             np.allclose(traj["NONE"].trajectory_v[13], traj["ALPHA_SCALE"].trajectory_v[13])
    print(f"  states identical at step 13: {same13}")
    print(f"  |u-uref| NONE={cn:.6f} [1.276847]  ALPHA_SCALE={ca:.6f} [0.994977]  "
          f"ratio={ca / cn:.4f} [0.7792]")
    ok = abs(cn - 1.276847) < 1e-4 and abs(ca - 0.994977) < 1e-4
    print(f"  CROSS-CHECK {'PASS' if ok else 'FAIL'}\n")

    # ---- approximate sufficiency check (ignores state divergence by construction) ----
    print("=== approximate sufficiency check ===")
    dt = sim["dt"]
    for lbl in ("NONE", "ALPHA_SCALE"):
        rows, _ = summary[lbl]
        d = sum(hi - lo for _, _, lo, hi, _, _ in rows)
        print(f"  at {lbl:>12} states: sum(|u_highgain-uref| - |u_lowgain-uref|) over window "
              f"= {d:.4f}; x dt = {d * dt:.4f} m/s of speed difference implied")
    sN = float(np.linalg.norm(traj["NONE"].trajectory_v[windows["NONE"][-1]]))
    sA = float(np.linalg.norm(traj["ALPHA_SCALE"].trajectory_v[windows["ALPHA_SCALE"][-1]]))
    print(f"  actual speed at last hazard-active step: NONE={sN:.4f}  ALPHA_SCALE={sA:.4f}  "
          f"difference={sA - sN:.4f} m/s")
    print("  (upper bound: correction magnitudes are not all aligned with the direction of travel,")
    print("   and this deliberately ignores the trajectory divergence it is testing for.)")


if __name__ == "__main__":
    main()
