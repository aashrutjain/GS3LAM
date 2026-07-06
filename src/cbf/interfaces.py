"""The seam a future ROS2/TurtleBot4 node plugs into.

Not wired to ROS2 in this pass. A node would subscribe to odometry to build a
RobotState, run its own outer planner/tracker to get u_ref, call
SafetyFilter.step(), and publish u_safe.

Known gap, not solved here: this seam is dimension-agnostic (R^3 position/
velocity/acceleration), but a TurtleBot4 is a planar differential-drive robot
commanded via (linear_vel, angular_vel), not raw 3D acceleration. Whatever
wraps CBFSafetyFilter in a real robot loop has to do that reduction.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class RobotState:
    p: np.ndarray            # (3,) world-frame position
    v: np.ndarray            # (3,) world-frame velocity
    t: float | None = None   # optional timestamp, for logging/diagnostics only


@dataclass
class SafetyFilterResult:
    u_safe: np.ndarray            # (3,) filtered acceleration command
    active_splat_ids: np.ndarray  # candidate-index ids constrained this step, for debugging/viz
    min_h: float                  # worst-case barrier value among active splats this step
    infeasible: bool              # QP infeasibility flag -- caller must handle (e.g. fall back to braking)
    solver_diagnostics: dict[str, Any] = field(default_factory=dict)


class SafetyFilter(Protocol):
    def step(self, state: RobotState, u_ref: np.ndarray) -> SafetyFilterResult: ...
