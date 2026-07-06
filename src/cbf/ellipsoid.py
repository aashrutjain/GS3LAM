"""Per-splat covariance reconstruction from the raw PLY fields.

Mirrors the exact quaternion convention used by build_rotation() in
src/utils/gaussian_utils.py: q = (w, x, y, z), unnormalized on disk, and the
standard rotation-matrix formula. Reimplemented here in plain NumPy (rather
than importing build_rotation, which is CUDA/torch-hardcoded) so this module
can run without a GPU -- relevant for a lightweight offline eval CLI and,
eventually, a ROS2 node that shouldn't need to pull in torch just for this.

Sigma = R S S^T R^T, S = diag(exp(log_scale)), matching standard 3DGS
covariance reconstruction. A := Sigma^-1.

s_min (used for the Remark-3 Minkowski robot-radius inflation, see
collision_cone.py) does NOT need an SVD: R is orthogonal, so it doesn't change
the singular values of Sigma^(1/2) = S. Those singular values are exactly
diag(exp(log_scale)), so s_min is a plain per-splat min-reduce over the three
true scale components. Correct for both the isotropic case (Replica's
default, configs/Replica/room0.py: gaussian_distribution="isotropic" --
scale_0==scale_1==scale_2, splats are literal spheres) and the general
anisotropic case.
"""

import numpy as np


def normalize_quaternion(rot_raw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(N,4) wxyz, unnormalized -> unit quaternion. Guards near-zero norm."""
    norm = np.linalg.norm(rot_raw, axis=-1, keepdims=True)
    norm = np.clip(norm, eps, None)
    return rot_raw / norm


def quat_to_rotmat(q_unit: np.ndarray) -> np.ndarray:
    """(N,4) wxyz unit quaternion -> (N,3,3) rotation matrix.

    Matches build_rotation()'s formula in src/utils/gaussian_utils.py exactly
    (same real-first ordering, same coefficients).
    """
    r, x, y, z = q_unit[:, 0], q_unit[:, 1], q_unit[:, 2], q_unit[:, 3]
    n = q_unit.shape[0]
    rot = np.zeros((n, 3, 3), dtype=q_unit.dtype)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def build_sigma(log_scale_raw: np.ndarray, rot_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (Sigma, s_min): Sigma is (N,3,3) SPD covariance, s_min is (N,)."""
    scale = np.exp(log_scale_raw)  # (N, 3), true scale
    q_unit = normalize_quaternion(rot_raw)
    R = quat_to_rotmat(q_unit)  # (N, 3, 3)

    # S S^T is diagonal (S = diag(scale)); build Sigma without materializing S.
    scale_sq = scale ** 2  # (N, 3)
    Sigma = np.einsum("nij,nj,nkj->nik", R, scale_sq, R)  # R @ diag(scale_sq) @ R^T

    s_min = scale.min(axis=-1)  # (N,), no SVD needed -- see module docstring
    return Sigma.astype(np.float32), s_min.astype(np.float32)


def build_A(Sigma: np.ndarray) -> np.ndarray:
    """A = Sigma^-1, batched (N,3,3) SPD inverse."""
    return np.linalg.inv(Sigma).astype(np.float32)
