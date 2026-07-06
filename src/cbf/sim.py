"""Trajectory rollout harness for eval_cbf_modes.py.

Wires dynamics.py's double-integrator + PD tracker through a SafetyFilter
(interfaces.py) into a single rollout() call, so the same harness runs
identically for the baseline and both semantic-weighting modes -- only the
CBFQPConfig.semantic_mode passed to CBFSafetyFilter differs between runs.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .dynamics import DoubleIntegratorState, pd_reference_controller, step_dynamics
from .interfaces import RobotState, SafetyFilter


@dataclass
class RolloutResult:
    trajectory_p: np.ndarray          # (T+1, 3)
    trajectory_v: np.ndarray          # (T+1, 3)
    min_h_history: np.ndarray         # (T,)
    infeasible_count: int
    active_splat_history: list = field(default_factory=list)
    reached_goal: bool = False
    time_to_goal: float | None = None
    wall_clock_per_step: float = float("nan")


def rollout(
    filt: SafetyFilter,
    p0: np.ndarray,
    v0: np.ndarray,
    goal: np.ndarray,
    dt: float,
    a_max: float,
    kp: float,
    kd: float,
    max_steps: int,
    goal_tol: float = 0.1,
) -> RolloutResult:
    state = DoubleIntegratorState(p=np.asarray(p0, dtype=np.float32).copy(), v=np.asarray(v0, dtype=np.float32).copy())
    ps = [state.p.copy()]
    vs = [state.v.copy()]
    min_h_hist = []
    active_hist = []
    step_times = []
    infeasible_count = 0
    reached_goal = False
    time_to_goal = None

    for step in range(max_steps):
        u_ref = pd_reference_controller(state, goal, kp, kd, a_max)

        t0 = time.perf_counter()
        result = filt.step(RobotState(p=state.p, v=state.v), u_ref)
        step_times.append(time.perf_counter() - t0)

        if result.infeasible:
            infeasible_count += 1
        min_h_hist.append(result.min_h)
        active_hist.append(result.active_splat_ids)

        state = step_dynamics(state, result.u_safe, dt)
        ps.append(state.p.copy())
        vs.append(state.v.copy())

        if np.linalg.norm(state.p - goal) < goal_tol:
            reached_goal = True
            time_to_goal = (step + 1) * dt
            break

    return RolloutResult(
        trajectory_p=np.array(ps),
        trajectory_v=np.array(vs),
        min_h_history=np.array(min_h_hist),
        infeasible_count=infeasible_count,
        active_splat_history=active_hist,
        reached_goal=reached_goal,
        time_to_goal=time_to_goal,
        wall_clock_per_step=float(np.mean(step_times)) if step_times else float("nan"),
    )
