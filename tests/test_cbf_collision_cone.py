"""Regression tests for src/cbf/collision_cone.py's compute_collision_cones.

Ports two ad hoc checks described in PROGRESS.md's "Verification done", both
against a synthetic robot-heading-straight-at-an-obstacle scene. The exact
original scene (position/velocity/obstacle numbers behind the recorded
h = -119.456 figure) was never persisted to disk -- confirmed absent from this
repo and machine, the same situation scripts/gen_cbf_synthetic_scene.py's
docstring documents for the COV_INFLATE experiment's scene. Reconstructed here
instead, matching the "robot heading straight at a synthetic obstacle" setup:
an isotropic unit-scale splat (Sigma = A = I) with the robot's line-of-sight r
exactly parallel to its velocity v.

For an exactly head-on approach with A = I, the general h = beta*gamma -
delta**2 collapses algebraically: with beta = |v|^2, delta = r.v = |r||v|
(parallel, same sign), gamma = |r|^2 - c_m^2,

    h = |v|^2 * (|r|^2 - c_m^2) - (|r||v|)^2 = -|v|^2 * c_m^2

independent of the along-ray distance |r|. Solved backward for c_m (given
unit speed |v|=1) so this hand-computable closed form reproduces the exact
regression value pinned in PROGRESS.md, h = -119.456.
"""

import numpy as np

from src.cbf.collision_cone import compute_collision_cones

_C_M = np.sqrt(119.456)  # solved so that -|v|^2 * c_m^2 == -119.456 at |v|=1


def test_compute_collision_cones_head_on_h_matches_hand_computed_value():
    p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)          # unit speed, heading +x
    mu = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)        # obstacle straight ahead, on-axis
    A = np.eye(3, dtype=np.float64)[None, :, :]                # isotropic unit-scale splat
    c_m = np.array([_C_M], dtype=np.float64)

    cones = compute_collision_cones(p, v, mu, A, c_m)

    assert np.isclose(cones.h[0], -119.456, rtol=1e-9)
    assert cones.cone_exists[0]  # h <= 0 and delta >= 0 (approaching, not receding)


def test_compute_collision_cones_receding_trajectory_does_not_activate_cone():
    p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([-1.0, 0.0, 0.0], dtype=np.float64)          # moving away from the obstacle
    mu = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)
    A = np.eye(3, dtype=np.float64)[None, :, :]
    c_m = np.array([_C_M], dtype=np.float64)

    cones = compute_collision_cones(p, v, mu, A, c_m)

    assert cones.delta[0] < 0    # Eq 9b (r^T A v >= 0) fails: receding, not approaching
    assert not cones.cone_exists[0]
