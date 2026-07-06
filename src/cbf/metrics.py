"""Comparison metrics for eval_cbf_modes.py: collision severity, near-miss
frequency, and path efficiency, per the research question in ARCHITECTURE.md
Sec 1 (does semantic weighting actually improve outcomes over pure geometric
avoidance?).
"""

import numpy as np


def mahalanobis_signed_distance(trajectory_p: np.ndarray, mu: np.ndarray, A: np.ndarray, c_m: np.ndarray) -> np.ndarray:
    """d_i(t) = sqrt((p(t)-mu_i)^T A_i (p(t)-mu_i)) - c_m,i, minimized over i per sample.

    Returns (T,) array; negative values mean the trajectory sample is inside
    a splat's (Minkowski-inflated) confidence ellipsoid at that instant.

    trajectory_p: (T,3). mu, c_m: (N,3)/(N,). A: (N,3,3). Caller decides which
    splat subset to evaluate against (e.g. the same opacity-pruned set the
    safety filter used) -- this function is agnostic to that choice.
    """
    diffs = trajectory_p[:, None, :] - mu[None, :, :]        # (T, N, 3)
    md = np.sqrt(np.maximum(np.einsum("tni,nij,tnj->tn", diffs, A, diffs), 0.0))
    d = md - c_m[None, :]
    return d.min(axis=1)                                      # (T,)


def collision_severity(min_dist_over_time: np.ndarray) -> float:
    """Worst (most negative) signed distance over the whole trajectory."""
    return float(min_dist_over_time.min())


def near_miss_events(min_dist_over_time: np.ndarray, d_thresh: float) -> int:
    """Count distinct close-approach events where distance dips below
    d_thresh, de-duplicated via hysteresis: an event ends once distance rises
    back above d_thresh, so one slow pass by an object isn't counted many
    times as consecutive samples all fall below threshold.
    """
    below = min_dist_over_time < d_thresh
    if below.size == 0:
        return 0
    edges = np.diff(below.astype(np.int8))
    count = int(np.count_nonzero(edges == 1))
    if below[0]:
        count += 1
    return count


def path_efficiency(
    trajectory_p: np.ndarray,
    straight_line_dist: float,
    actual_time: float,
    oracle_time: float,
) -> dict:
    """path_length_ratio: actual arc length / straight-line start-goal distance.
    time_ratio: actual time-to-goal / oracle time-to-goal (CBF disabled, same
    controller, obstacle-free copy of the scene).
    """
    diffs = np.diff(trajectory_p, axis=0)
    arc_length = float(np.sum(np.linalg.norm(diffs, axis=-1)))
    return {
        "arc_length": arc_length,
        "straight_line_dist": straight_line_dist,
        "path_length_ratio": arc_length / straight_line_dist if straight_line_dist > 1e-9 else float("nan"),
        "actual_time": actual_time,
        "oracle_time": oracle_time,
        "time_ratio": actual_time / oracle_time if oracle_time > 1e-9 else float("nan"),
    }
