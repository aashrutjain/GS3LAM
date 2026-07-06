"""The unresolved research question this whole module exists to test:

how (if at all) should the per-splat VLM safety scalar modify the purely
geometric collision-cone CBF? Two candidates, per ARCHITECTURE.md Sec 2.3,
NEITHER validated -- both are implemented here as swappable/configurable
strategies over a shared NONE baseline that never reads `safety` at all.

- ALPHA_SCALE only changes the QP constraint's right-hand side (the class-K
  gain k_alpha becomes per-splat). It affects longitudinal braking
  aggressiveness only -- it does not change the shape of any confidence
  ellipsoid, so it cannot change lateral routing around an object.
- COV_INFLATE changes A (and s_min) directly, i.e. it changes the geometry
  the collision cone is built from -- this is the strategy that can route the
  robot laterally around a low-safety object, not just brake for it.

Open research parameters, NOT settled by this implementation (see
ARCHITECTURE.md Sec 2.3 and the Stage 3 design plan): the exact monotonic map
f(safety) for ALPHA_SCALE (default here: identity, f(s) = s) and the
cov_inflate_gamma constant for COV_INFLATE (no built-in default value here on
purpose -- callers must supply one; treat it as a value to sweep, e.g.
{0.5, 1.0, 2.0}, not a constant to hardcode).
"""

from collections.abc import Callable
from enum import Enum

import numpy as np


class SemanticMode(Enum):
    NONE = "none"                # pure geometric baseline; `safety` column never read
    ALPHA_SCALE = "alpha_scale"  # k_alpha,i = k_alpha_base * f(safety_i)
    COV_INFLATE = "cov_inflate"  # Sigma_eff,i = Sigma_i * (1 + gamma * (1 - safety_i))


def alpha_gain_per_splat(
    k_alpha_base: float,
    safety: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray] = lambda s: s,
    f_min: float = 1e-3,
) -> np.ndarray:
    """k_alpha,i = k_alpha_base * max(f(safety_i), f_min).

    f_min guards against k_alpha collapsing to exactly 0 (which would fully
    disable the CBF constraint for that splat) if `f` maps some safety value
    to 0 -- e.g. f(s)=s with safety_i=0.0 exactly. Given the zero-safety
    ambiguity documented in ply_io.py, a splat reading safety==0.0 is not
    necessarily a genuine confirmed hazard, so silently zeroing its gain
    (rather than just making it very conservative) is not obviously correct;
    f_min keeps the barrier active but maximally cautious in that case.
    """
    gain = f(np.asarray(safety, dtype=np.float32))
    gain = np.clip(gain, f_min, None)
    return k_alpha_base * gain


def inflate_covariance(
    A: np.ndarray,
    s_min: np.ndarray,
    safety: np.ndarray,
    cov_inflate_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar covariance inflation applied directly to A and s_min.

    Sigma_eff,i = Sigma_i * scale_i,  scale_i := 1 + gamma*(1 - safety_i)  (>= 1)

    Because inflation is a per-splat SCALAR multiply (not a general
    anisotropic reshape), this is applied directly to A and s_min without
    rebuilding/re-inverting a 3x3 Sigma per splat:

        A_eff_i     = A_i / scale_i          since (k * Sigma)^-1 = Sigma^-1 / k
        s_min_eff_i = s_min_i * sqrt(scale_i)  since scaling Sigma by k scales
                                                the singular values of
                                                Sigma^(1/2) = S by sqrt(k)

    Numerically identical to fully rebuilding Sigma_eff and re-inverting it.
    """
    safety = np.asarray(safety, dtype=np.float32)
    scale = 1.0 + cov_inflate_gamma * (1.0 - safety)  # (N,), >= 1 for safety in [0,1], gamma >= 0
    A_eff = A / scale[:, None, None]
    s_min_eff = s_min * np.sqrt(scale)
    return A_eff.astype(np.float32), s_min_eff.astype(np.float32)
