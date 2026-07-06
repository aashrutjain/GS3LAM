"""Double-integrator sim dynamics + a toy PD reference controller, used only
by the evaluation harness (sim.py / eval_cbf_modes.py). Matches the paper's
own dynamics model (state x=[p;v], control u=acceleration) -- not a real
planner, just a reference-generating stand-in until a real ROS2/TurtleBot4
loop exists (see interfaces.py for that seam).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DoubleIntegratorState:
    p: np.ndarray  # (3,)
    v: np.ndarray  # (3,)


def step_dynamics(state: DoubleIntegratorState, u: np.ndarray, dt: float) -> DoubleIntegratorState:
    """Semi-implicit Euler: v' = v + u*dt; p' = p + v'*dt."""
    v_new = state.v + u * dt
    p_new = state.p + v_new * dt
    return DoubleIntegratorState(p=p_new, v=v_new)


def pd_reference_controller(
    state: DoubleIntegratorState,
    goal: np.ndarray,
    kp: float,
    kd: float,
    a_max: float,
) -> np.ndarray:
    """u_ref = clip(kp*(goal - p) - kd*v, ||.|| <= a_max)."""
    u = kp * (goal - state.p) - kd * state.v
    norm = float(np.linalg.norm(u))
    if norm > a_max and norm > 1e-12:
        u = u * (a_max / norm)
    return u.astype(np.float32)
