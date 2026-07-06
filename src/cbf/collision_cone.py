"""Analytic collision-cone CBF primitive.

Transcribed from Tscholl, Nakka & Gunter, "Perception-Integrated Safety
Critical Control via Analytic Collision Cone Barrier Functions on 3D Gaussian
Splatting" (arXiv:2509.14421). The paper's class-K gain constant is called
`p_k`; renamed to `k_alpha` throughout this codebase to avoid clashing with
robot position `p` -- a notational fix only, not a math change.

Setup: splat covariance Sigma in R^{3x3} (SPD), A := Sigma^-1. Confidence
ellipsoid E = {x | (x-mu)^T A (x-mu) <= c^2}, c^2 = chi2_{3,0.99}. Robot
position p, (instantaneous, assumed-constant) velocity v. Line-of-sight
r := mu - p. Ray x(t) = p + t*v, t >= 0.

Collision-cone existence test (paper Eq 9a/9b): the along-ray quadratic
phi(t) = (r - t*v)^T A (r - t*v) - c^2 = a*t^2 - 2*b*t + d, with a = v^T A v,
b = r^T A v, d = r^T A r - c^2, is minimized at t* = b/a. A collision exists
along the ray iff:
    (v^T A v)(r^T A r - c^2) - (r^T A v)^2 <= 0     (9a, discriminant)
    r^T A v >= 0                                     (9b, approaching not receding)

Barrier function (paper Eq 13), with beta := v^T A v, gamma := r^T A r - c^2,
delta := r^T A v:
    h(p, v) := beta*gamma - delta^2 >= 0
defines the per-splat safe set C_i = {(p,v) | h_i(p,v) >= 0}.

QP constraint vector (from the Lie-derivative simplification under
double-integrator dynamics, since v-dot's drift term is 0):
    w(x) := gamma * A v - delta * A r  in R^3
    CBF constraint: w(x)^T u >= -(k_alpha/2) * h(p,v)

Minkowski inflation for robot physical extent (paper Remark 3): for a
spherical robot of radius rho, replace c with c_m = c + rho/s_min, where
s_min is the splat's smallest true scale component (see ellipsoid.py). This
is per-splat since s_min varies per splat, and must be folded into gamma
(not just c) before it's used: gamma_i = r_i^T A_i r_i - c_m,i^2.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CollisionCone:
    beta: np.ndarray          # (N,) v^T A_i v
    gamma: np.ndarray         # (N,) r_i^T A_i r_i - c_m,i^2
    delta: np.ndarray         # (N,) r_i^T A_i v
    h: np.ndarray             # (N,) beta*gamma - delta^2
    w: np.ndarray             # (N,3) gamma_i * A_i v - delta_i * A_i r_i
    cone_exists: np.ndarray   # (N,) bool, Eq 9a & 9b -- diagnostic / optional active-set gate


def effective_c(c_base: float, robot_radius: float, s_min: np.ndarray) -> np.ndarray:
    """c_m,i = c_base + robot_radius / s_min_i (Remark 3, per-splat)."""
    return c_base + robot_radius / s_min


def compute_collision_cones(
    p: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    A: np.ndarray,
    c_m: np.ndarray,
) -> CollisionCone:
    r = mu - p[None, :]                              # (N, 3)
    Ar = np.einsum("nij,nj->ni", A, r)                # (N, 3)
    Av = np.einsum("nij,j->ni", A, v)                 # (N, 3)

    beta = np.einsum("j,nj->n", v, Av)                # (N,)
    delta = np.einsum("nj,nj->n", r, Av)              # (N,)
    gamma = np.einsum("nj,nj->n", r, Ar) - c_m ** 2    # (N,)

    h = beta * gamma - delta ** 2                     # (N,)
    w = gamma[:, None] * Av - delta[:, None] * Ar      # (N, 3)
    cone_exists = (h <= 0.0) & (delta >= 0.0)

    return CollisionCone(beta=beta, gamma=gamma, delta=delta, h=h, w=w, cone_exists=cone_exists)
