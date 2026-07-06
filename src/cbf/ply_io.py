"""Loader for gsplat.ply / safety_gsplat.ply.

Field order and dtypes are confirmed from construct_list_of_attributes() /
convert_npz_to_ply() in src/utils/logger.py, and the 'safety' column append in
vlm_safety_score.py's broadcast_scores_and_save() -- not assumed from generic
3DGS conventions:

    x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1,
    scale_2, rot_0, rot_1, rot_2, rot_3, [safety]

- nx,ny,nz are always zero (not real normals) -- not loaded.
- f_dc_0..2 are raw RGB, not SH coefficients. No f_rest_* fields exist.
- opacity is stored as a logit; sigmoid() is applied here to get true opacity.
- scale_0..2 are stored as log-scale; left RAW here (exp() happens in
  ellipsoid.py, which owns the Sigma/A reconstruction) so this module stays a
  thin I/O boundary.
- rot_0..3 is an unnormalized quaternion, (w,x,y,z) order; left RAW here for
  the same reason.
- 'safety' only exists in safety_gsplat.ply, appended via
  numpy.lib.recfunctions.append_fields with a np.zeros(...) default. Splats
  never queried by the VLM are therefore bit-identical to a splat genuinely
  rated 0.0 ("lethal hazard" per the VLM prompt convention) -- there is no
  sentinel in the file distinguishing the two. ambiguous_zero_mask below
  flags this; it is NOT a reliable "unscored" mask, only a "reads as exactly
  0.0" mask. Do not treat it as more certain than that.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from plyfile import PlyData


class ZeroSafetyPolicy(Enum):
    WARN_ONLY = "warn_only"              # default: keep raw values, log zero_fraction loudly
    TREAT_AS_NEUTRAL = "treat_as_neutral"  # explicit opt-in remap of exact-0.0 -> neutral_value
    TREAT_AS_HAZARD = "treat_as_hazard"    # explicit no-op; documents trusting the literal VLM convention


@dataclass
class SplatField:
    xyz: np.ndarray                       # (N, 3) float32, splat means (mu)
    opacity: np.ndarray                   # (N,) float32, TRUE opacity = sigmoid(logit_opacity)
    log_scale_raw: np.ndarray             # (N, 3) float32, as stored (log-scale, pre-exp)
    rot_raw: np.ndarray                   # (N, 4) float32, unnormalized quaternion (w,x,y,z)
    rgb: np.ndarray                       # (N, 3) float32, raw f_dc_0..2 -- carried through for viz only
    safety_raw: np.ndarray | None         # (N,) float32 if 'safety' column present, else None
    ambiguous_zero_mask: np.ndarray | None  # (N,) bool, safety_raw == 0.0 exactly; None if no safety column
    zero_fraction: float | None
    source_path: str

    @property
    def n(self) -> int:
        return self.xyz.shape[0]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_splat_field(
    ply_path: str,
    zero_policy: ZeroSafetyPolicy = ZeroSafetyPolicy.WARN_ONLY,
    neutral_value: float = 0.5,
) -> SplatField:
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0].data

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    rgb = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    opacity = _sigmoid(np.asarray(v["opacity"], dtype=np.float32))
    log_scale_raw = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    rot_raw = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)

    field_names = v.dtype.names
    safety_raw = None
    ambiguous_zero_mask = None
    zero_fraction = None

    if "safety" in field_names:
        safety_raw = np.asarray(v["safety"], dtype=np.float32).copy()
        ambiguous_zero_mask = safety_raw == 0.0
        zero_fraction = float(np.count_nonzero(ambiguous_zero_mask)) / max(len(safety_raw), 1)

        if zero_fraction > 0.5:
            print(
                f"[cbf.ply_io] WARNING: {zero_fraction:.1%} of splats in {ply_path} read "
                "safety==0.0. safety_gsplat.ply cannot distinguish 'VLM genuinely rated this "
                "lethal' from 'never queried, defaulted to 0.0' -- see vlm_safety_score.py's "
                "broadcast_scores_and_save(). Treat this fraction as diagnostic, not as a claim "
                "that this many splats are truly hazardous."
            )

        if zero_policy is ZeroSafetyPolicy.TREAT_AS_NEUTRAL:
            print(
                f"[cbf.ply_io] zero_policy=TREAT_AS_NEUTRAL: remapping {zero_fraction:.1%} of "
                f"splats (safety==0.0) to neutral_value={neutral_value}. This is an explicit "
                "experimental override, not a correction -- some of these may be genuine "
                "lethal ratings."
            )
            safety_raw = safety_raw.copy()
            safety_raw[ambiguous_zero_mask] = neutral_value
        elif zero_policy is ZeroSafetyPolicy.TREAT_AS_HAZARD:
            pass  # no-op: keep the literal VLM convention (0.0 = lethal)
        # WARN_ONLY: keep raw values, warning already logged above.

    return SplatField(
        xyz=xyz,
        opacity=opacity,
        log_scale_raw=log_scale_raw,
        rot_raw=rot_raw,
        rgb=rgb,
        safety_raw=safety_raw,
        ambiguous_zero_mask=ambiguous_zero_mask,
        zero_fraction=zero_fraction,
        source_path=ply_path,
    )
