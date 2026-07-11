"""Synthetic scene generator for eval_cbf_modes.py --phase-a runs.

Writes a gsplat.ply matching the exact schema construct_list_of_attributes()
writes in src/utils/logger.py (x,y,z,nx,ny,nz,f_dc_0..2,opacity,scale_0..2,
rot_0..3) -- no 'safety' column, since --phase-a overrides SplatField.safety_raw
in-memory regardless of what's on disk (see synthesize_hazard_safety() in
eval_cbf_modes.py).

This replaces an earlier synthetic scene that was built ad hoc in a prior
session's sandbox and never persisted to disk -- discovered when trying to
re-run "the same scene" and finding it didn't exist anywhere on this machine.
Persisting the generator (not just the .ply) so the exact scene can be
regenerated deterministically (fixed seed) in any future session, and so the
"narrow" vs "wide" corridor variants are a documented one-parameter diff
(--collar-radius) rather than two independently hand-edited files.

Geometry: straight-line path along +x from start=(-2.5,0,0) to goal=(2.5,0,0).
One hazard splat at the origin. A double-ring "collar" of small clutter splats
encircling the corridor around the hazard (not just flanking in +/-y) so a
robot must detour in BOTH y and z to go around the hazard -- a single flanking
pair in y alone leaves z open as an unconstrained escape route, which would
silently defeat the corridor-width manipulation this scene exists to support.
300 background splats (240 far-field 'bulk' + 60 collar) + 1 hazard splat =
301 total, isotropic scale, identity quaternion (unnormalized-on-disk
convention per ply_io.py, stored here as the already-unit (1,0,0,0)).

All physical-radius math below follows collision_cone.py's actual formula
(confirmed against src/cbf/collision_cone.py, not assumed):
    r_phys(s) = c_base * s + robot_radius,   c_base = sqrt(chi2.ppf(0.99, df=3))
for an isotropic splat of raw scale s (Sigma = s^2 I) with a spherical robot
of the given robot_radius, BEFORE any semantic weighting. COV_INFLATE with
gamma inflates only the hazard (background/collar splats keep safety=1.0 under
--phase-a, i.e. gain=1, unaffected) via s_eff = s*sqrt(1+gamma*(1-safety)).
"""

import argparse

import numpy as np
from plyfile import PlyElement, PlyData
from scipy.stats import chi2

C_BASE = float(np.sqrt(chi2.ppf(0.99, df=3)))  # matches qp_filter.CBFQPConfig.chi2_conf default
ROBOT_RADIUS = 0.16  # matches configs/cbf/room0_cbf.py


def r_phys(scale: float, robot_radius: float = ROBOT_RADIUS) -> float:
    return C_BASE * scale + robot_radius


FIELDS = [
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]


def _splat_rows(xyz, log_scale, logit_opacity=4.0, rgb=(0.5, 0.5, 0.5), rot=(1.0, 0.0, 0.0, 0.0)):
    n = xyz.shape[0]
    normals = np.zeros((n, 3), dtype=np.float32)
    f_dc = np.tile(np.array(rgb, dtype=np.float32), (n, 1))
    opacity = np.full((n, 1), logit_opacity, dtype=np.float32)
    scale = np.tile(np.array([log_scale] * 3, dtype=np.float32), (n, 1))
    rotation = np.tile(np.array(rot, dtype=np.float32), (n, 1))
    return np.concatenate([xyz.astype(np.float32), normals, f_dc, opacity, scale, rotation], axis=1)


def build_scene(collar_radius: float, seed: int = 42, cluster_offset_y: float = 0.03) -> np.ndarray:
    """collar_radius: distance from the corridor axis (x-axis, through the
    hazard+collar cluster center) to the center of the clutter collar. Larger
    = wider corridor. Returns (301, 17) float32 array of raw attribute rows,
    ready to write in FIELDS order.

    cluster_offset_y: the whole hazard+collar cluster (kept concentric) is
    shifted by (0, cluster_offset_y, 0) off the robot's y=z=0 path centerline.
    Needed to break an exact-symmetry degenerate case found empirically: with
    the hazard sitting exactly on the centerline and the robot approaching
    dead-on-axis, the line-of-sight r and velocity v stay perfectly
    colinear, so the collision-cone vector w = gamma*Av - delta*Ar (see
    collision_cone.py) has zero lateral component throughout -- the one-step
    QP can only brake along the approach axis, never discover a lateral
    detour, regardless of how much clearance actually exists off-axis. A
    small deliberate offset gives r a nonzero perpendicular component from
    the start so the QP has a real lateral gradient to act on, matching any
    realistic (non-perfectly-centered) navigation scenario.
    """
    rng = np.random.default_rng(seed)

    hazard_scale = 0.10   # r_phys = 0.497 at baseline (gamma=0 / NONE / ALPHA_SCALE)
    clutter_scale = 0.04  # r_phys = 0.295, small clutter objects

    hazard_xyz = np.array([[0.0, cluster_offset_y, 0.0]], dtype=np.float32)
    hazard_rows = _splat_rows(hazard_xyz, np.log(hazard_scale))

    # Multi-ring collar spanning x in [-0.4, 0.4] (5 rings, 0.2m pitch,
    # overlapping since each ring's own x-half-width from clutter_scale's
    # r_phys=0.295 exceeds the 0.2m spacing). A single ring pair (tried
    # first) left an exploitable flank: past the ring's finite x-extent, the
    # hazard's own blocking radius (sqrt(r_phys_hazard^2 - x^2), zero at
    # x=+/-0.624 for gamma=1.0's inflated r_phys=0.624) was already small
    # enough that a shallow diagonal bypass existed just outside the ring's
    # edge -- COV_INFLATE threaded it at gamma=1.0 despite the donut's inner
    # radius being nominally "closed" on-axis. Ring x-extent must reach
    # roughly +/-0.6 to seal that flank for gamma up to 1.0. n_per_ring
    # chosen so adjacent splat footprints (2*r_phys(clutter_scale) chord)
    # overlap around the full circumference at collar_radius.
    n_per_ring = 30
    ring_rows = []
    for x_off in (-0.4, -0.2, 0.0, 0.2, 0.4):
        theta = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)
        theta = theta + rng.uniform(-0.02, 0.02, size=n_per_ring)  # de-alias, break exact symmetry
        y = collar_radius * np.cos(theta) + cluster_offset_y
        z = collar_radius * np.sin(theta)
        x = np.full(n_per_ring, x_off, dtype=np.float32) + rng.uniform(-0.03, 0.03, size=n_per_ring)
        xyz = np.stack([x, y, z], axis=-1)
        ring_rows.append(_splat_rows(xyz, np.log(clutter_scale)))
    ring_rows = np.concatenate(ring_rows, axis=0)  # (60, 17)

    # Far-field bulk background: outside spatial_filter's radius_cap=3.0 from
    # every point on the path (path x in [-2.5, 2.5]), so it never enters the
    # candidate set and cannot interfere with the corridor experiment.
    n_bulk = 300 - ring_rows.shape[0]
    bulk_x_sign = rng.choice([-1.0, 1.0], size=n_bulk)
    bulk_x = bulk_x_sign * rng.uniform(6.0, 10.0, size=n_bulk)
    bulk_y = rng.uniform(-3.0, 3.0, size=n_bulk)
    bulk_z = rng.uniform(-1.0, 1.0, size=n_bulk)
    bulk_xyz = np.stack([bulk_x, bulk_y, bulk_z], axis=-1)
    bulk_rows = _splat_rows(bulk_xyz, np.log(clutter_scale))

    return np.concatenate([hazard_rows, ring_rows, bulk_rows], axis=0)


def write_ply(rows: np.ndarray, out_path: str) -> None:
    dtype_full = [(f, "f4") for f in FIELDS]
    elements = np.empty(rows.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, rows))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collar-radius", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = build_scene(args.collar_radius, args.seed)
    write_ply(rows, args.out)
    print(f"[gen_cbf_synthetic_scene] wrote {rows.shape[0]} splats (collar_radius={args.collar_radius}) to {args.out}")
