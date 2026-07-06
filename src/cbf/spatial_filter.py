"""Candidate-splat prefilter.

Scenes here run ~10^4-10^5 splats; putting every splat into the QP as a
constraint row per timestep is infeasible (the paper's own implementation
notes an "adaptive filter to only consider Gaussian splats within a certain
distance of the robot" for the same reason, on a ~170k-splat scene). This
module is that prefilter -- it only narrows the candidate set; the actual
Eq 9a/9b active-set gate happens downstream in qp_filter.py.

Uses scipy.spatial.cKDTree, rebuilt once per scene load (splats are static
within an episode) and queried per step. scipy is already a transitive
dependency in this environment; this module makes it a direct one -- see the
Stage 3 design's flagged dependency note before merging that requirement.

Default radius/count/threshold values below are placeholders, NOT measured
against the actual trained room0 scene's bounding box -- confirm before
trusting them for a real run (see build_kdtree's docstring).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SpatialFilterConfig:
    mode: Literal["radius", "knn"] = "radius"
    radius: float | None = None       # meters; if None, computed adaptively from speed
    k: int = 50                        # used only in "knn" mode
    max_candidates: int = 200          # hard cap regardless of mode, bounds QP size
    lookahead_horizon: float = 2.0     # seconds, used for adaptive radius
    radius_margin: float = 1.0         # meters, added to speed*horizon
    radius_cap: float = 3.0            # meters, hard ceiling on adaptive radius
    opacity_prune_thresh: float = 0.1  # drop near-transparent (likely floater) splats


def prune_low_opacity(opacity: np.ndarray, thresh: float) -> np.ndarray:
    """Boolean keep-mask; intended to run once per scene load, not per step."""
    return opacity >= thresh


def build_kdtree(xyz: np.ndarray) -> cKDTree:
    """Build once per scene load and reuse across all rollout steps.

    NOTE: the default SpatialFilterConfig radius/max_candidates values are
    placeholders. Before a real eval run, compute xyz.min(axis=0)/max(axis=0)
    on the actual safety_gsplat.ply being used and sanity-check these against
    the scene's true extent.
    """
    return cKDTree(xyz)


def select_candidates(
    p: np.ndarray,
    v: np.ndarray,
    tree: cKDTree,
    cfg: SpatialFilterConfig,
) -> np.ndarray:
    """Returns indices (into the array `tree` was built from) of candidate splats."""
    if cfg.mode == "radius":
        radius = cfg.radius
        if radius is None:
            speed = float(np.linalg.norm(v))
            radius = min(speed * cfg.lookahead_horizon + cfg.radius_margin, cfg.radius_cap)
        idx = np.asarray(tree.query_ball_point(p, radius), dtype=np.int64)
        if idx.size > cfg.max_candidates:
            dists = np.linalg.norm(tree.data[idx] - p[None, :], axis=-1)
            nearest = np.argsort(dists)[: cfg.max_candidates]
            idx = idx[nearest]
        return idx
    elif cfg.mode == "knn":
        k = min(cfg.k, cfg.max_candidates, tree.n)
        _, idx = tree.query(p, k=k)
        return np.atleast_1d(np.asarray(idx, dtype=np.int64))
    else:
        raise ValueError(f"Unknown SpatialFilterConfig.mode: {cfg.mode!r}")
