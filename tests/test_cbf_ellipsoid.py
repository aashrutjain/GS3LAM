"""Regression test for src/cbf/ellipsoid.py's isotropic-splat contract.

Ports the ad hoc check described in PROGRESS.md's "Verification done":
ellipsoid.build_sigma on an isotropic splat matches a hand-computed
scale^2 * I.
"""

import numpy as np

from src.cbf.ellipsoid import build_sigma


def test_build_sigma_isotropic_matches_hand_computed_scale_squared_identity():
    n = 4
    scale = 0.37
    log_scale_raw = np.full((n, 3), np.log(scale), dtype=np.float32)
    # Unnormalized quaternion (w,x,y,z) that represents identity rotation once
    # normalized -- exercises normalize_quaternion, matching ply_io.py's "left
    # raw on disk" convention rather than assuming a pre-normalized input.
    rot_raw = np.tile(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    Sigma, s_min = build_sigma(log_scale_raw, rot_raw)

    expected_sigma = (scale ** 2) * np.eye(3, dtype=np.float32)
    for i in range(n):
        np.testing.assert_allclose(Sigma[i], expected_sigma, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(s_min, np.full(n, scale, dtype=np.float32), rtol=1e-5)
