"""Regression tests for src/cbf/qp_filter.py.

Ports two ad hoc checks described in PROGRESS.md's "Verification done":

- The _clip_to_a_max dtype regression: an earlier version hardcoded
  `.astype(np.float32)` on its clipped branch, which silently downcast the
  SLSQP solver's float64 x0 seed back to float32 whenever the reference
  control exceeded a_max -- crashing SciPy's Fortran backend on the first
  real rollout step. Fixed to preserve the input dtype.
- The build_baseline_inputs safety-blindness contract: SemanticMode.NONE's
  baseline path must never read splats.safety_raw, so its output must be
  byte-identical whether safety_raw is real data or NaN.
"""

import numpy as np

from src.cbf.ply_io import SplatField
from src.cbf.qp_filter import CBFQPConfig, _clip_to_a_max, build_baseline_inputs


def test_clip_to_a_max_preserves_float64_dtype_when_clipped():
    a_max = 1.0
    u_ref = np.array([3.0, 4.0, 0.0], dtype=np.float64)  # norm=5 > a_max
    clipped = _clip_to_a_max(u_ref, a_max)

    assert clipped.dtype == np.float64
    assert np.isclose(np.linalg.norm(clipped), a_max)


def test_clip_to_a_max_preserves_float32_dtype_when_clipped():
    a_max = 1.0
    u_ref = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    clipped = _clip_to_a_max(u_ref, a_max)

    assert clipped.dtype == np.float32


def _make_splats(n, log_scale_raw, rot_raw, safety_raw):
    return SplatField(
        xyz=np.zeros((n, 3), dtype=np.float32),
        opacity=np.ones(n, dtype=np.float32),
        log_scale_raw=log_scale_raw,
        rot_raw=rot_raw,
        rgb=np.zeros((n, 3), dtype=np.float32),
        safety_raw=safety_raw,
        ambiguous_zero_mask=None,
        zero_fraction=None,
        source_path="<test>",
    )


def test_build_baseline_inputs_is_identical_whether_safety_raw_is_real_or_nan():
    n = 6
    rng = np.random.default_rng(0)
    log_scale_raw = rng.uniform(-2.0, -0.5, size=(n, 3)).astype(np.float32)
    rot_raw = rng.normal(size=(n, 4)).astype(np.float32)
    cfg = CBFQPConfig(a_max=1.0, k_alpha_base=2.0)

    splats_real = _make_splats(n, log_scale_raw, rot_raw, rng.uniform(0, 1, size=n).astype(np.float32))
    splats_nan = _make_splats(n, log_scale_raw, rot_raw, np.full(n, np.nan, dtype=np.float32))

    A_real, s_min_real, k_alpha_real = build_baseline_inputs(splats_real, cfg)
    A_nan, s_min_nan, k_alpha_nan = build_baseline_inputs(splats_nan, cfg)

    np.testing.assert_array_equal(A_real, A_nan)
    np.testing.assert_array_equal(s_min_real, s_min_nan)
    np.testing.assert_array_equal(k_alpha_real, k_alpha_nan)


def test_build_baseline_inputs_is_identical_whether_safety_raw_is_none_or_nan():
    n = 6
    rng = np.random.default_rng(1)
    log_scale_raw = rng.uniform(-2.0, -0.5, size=(n, 3)).astype(np.float32)
    rot_raw = rng.normal(size=(n, 4)).astype(np.float32)
    cfg = CBFQPConfig(a_max=1.0, k_alpha_base=2.0)

    splats_none = _make_splats(n, log_scale_raw, rot_raw, None)
    splats_nan = _make_splats(n, log_scale_raw, rot_raw, np.full(n, np.nan, dtype=np.float32))

    A_none, s_min_none, k_alpha_none = build_baseline_inputs(splats_none, cfg)
    A_nan, s_min_nan, k_alpha_nan = build_baseline_inputs(splats_nan, cfg)

    np.testing.assert_array_equal(A_none, A_nan)
    np.testing.assert_array_equal(s_min_none, s_min_nan)
    np.testing.assert_array_equal(k_alpha_none, k_alpha_nan)
