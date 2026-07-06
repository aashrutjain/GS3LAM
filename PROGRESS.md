# Stage 3 Implementation Progress

Snapshot written mid-implementation (context budget ran low) so work can resume
cleanly. See `ARCHITECTURE.md` §2.3 for the design/math this implements, and the
original design plan (from the planning session) for the full rationale behind every
choice below.

## Done

All of these exist and are believed correct by construction/inspection, but **none
have been executed or unit-tested yet** — see "Not done" below.

- `src/cbf/__init__.py` — empty package marker.
- `src/cbf/interfaces.py` — `RobotState`, `SafetyFilterResult`, `SafetyFilter` protocol.
  The ROS2 seam. Not wired to ROS2.
- `src/cbf/ply_io.py` — `SplatField`, `ZeroSafetyPolicy`, `load_splat_field()`. Reads
  the exact 17(+1)-field schema confirmed from `src/utils/logger.py` /
  `vlm_safety_score.py`. Un-sigmoids opacity; leaves scale/rotation raw (ellipsoid.py's
  job). Surfaces `ambiguous_zero_mask`/`zero_fraction` for the unrecoverable
  zero-default ambiguity — never claims to reconstruct a true "unscored" mask.
- `src/cbf/ellipsoid.py` — `normalize_quaternion`, `quat_to_rotmat` (mirrors
  `src/utils/gaussian_utils.py`'s `build_rotation()` convention exactly, reimplemented
  in NumPy so this module doesn't need torch/CUDA), `build_sigma` (Σ=RSSᵀRᵀ, `s_min`
  via plain min-reduce, no SVD), `build_A`.
- `src/cbf/semantic_weighting.py` — `SemanticMode` enum (`NONE`/`ALPHA_SCALE`/
  `COV_INFLATE`), `alpha_gain_per_splat` (with an `f_min` floor so a splat's gain can't
  collapse to exactly 0), `inflate_covariance` (scalar-multiply shortcut on `A`/`s_min`,
  no need to rebuild/re-invert Σ).
- `src/cbf/collision_cone.py` — `CollisionCone` dataclass, `effective_c` (Minkowski
  inflation), `compute_collision_cones` (β/γ/δ/h/w(x) per Eq 9/13). One implementation
  shared by all three `SemanticMode`s.
- `src/cbf/spatial_filter.py` — `SpatialFilterConfig`, `prune_low_opacity`,
  `build_kdtree`, `select_candidates` (radius or knn mode, `scipy.spatial.cKDTree`).
  **Placeholder radius/max_candidates/opacity_prune_thresh defaults — not measured
  against the real trained `room0` scene's bounding box.**
- `src/cbf/qp_filter.py` — `CBFQPConfig`, `build_baseline_inputs` (contract: must
  never read `splats.safety_raw` — not yet covered by a unit test, see below),
  `build_semantic_inputs` (dispatches on mode, uniform output shape), `CBFSafetyFilter`
  (implements `SafetyFilter.step()`: spatial prefilter → collision cones → active-set
  gate → QP solve → fallback-to-max-braking on infeasibility). Default solver is
  `scipy.optimize.minimize(method="SLSQP")` behind a `_SOLVERS` registry — an explicit
  stopgap, not a claim that it's the right real-time choice; see the file's docstring.
- `src/cbf/dynamics.py` — `DoubleIntegratorState`, `step_dynamics` (semi-implicit
  Euler), `pd_reference_controller` (toy tracker, eval-only).
- `src/cbf/metrics.py` — `mahalanobis_signed_distance`, `collision_severity`,
  `near_miss_events` (hysteresis-deduplicated), `path_efficiency`.
- `src/cbf/sim.py` — `RolloutResult`, `rollout()`: wires dynamics + any `SafetyFilter` +
  per-step logging into one call, used identically across all three modes.
- `configs/cbf/room0_cbf.py` — plain-dict config mirroring `configs/Replica/room0.py`'s
  style. `ply_path` is a placeholder pointing at a run directory that doesn't exist yet
  (`run_name = "PLACEHOLDER"`) — must be set to a real Stage 1/2 output directory, or
  overridden via CLI, before anything can actually run.
- `costmap_cbf.py` — top-level CLI: loads a config + PLY, builds a `CBFSafetyFilter`,
  runs one `step()` with CLI-supplied `p`/`v`/`u_ref`, prints the result. This is the
  piece a future ROS2 node lifts the outer loop out of.
- `ARCHITECTURE.md` §2.3 and `CLAUDE.md` updated to reflect the above (module layout,
  exact math, confirmed data-contract gotchas, and the two Stage 2 bugs found below).

## Not done — pick up here

1. **`eval_cbf_modes.py`** — the actual 3-mode comparison harness (the experiment that
   answers the open research question). Design is fully specified in the plan/
   `ARCHITECTURE.md` §2.3: pick a verified collision-free start/goal on a real
   `room0` scene, run an oracle (CBF disabled) for the time-to-goal denominator, then
   run `NONE`/`ALPHA_SCALE`/`COV_INFLATE` through `src/cbf/sim.rollout()` and tabulate
   `collision_severity`/`near_miss_events`/`path_efficiency` for each, plus solver
   infeasibility counts and wall-clock/step. Needs a **Phase A synthetic-hazard sanity
   run first** (hand-edit one splat's `safety` to ~0.1 directly on the collision path,
   everything else ~1.0) before trusting any real-scene numbers — real-scene numbers
   are blocked on Stage 2 bug (1) below anyway.
2. **`scipy` is not yet added to `requirements.txt`/`environment.yml`.** `src/cbf`
   imports `scipy.spatial.cKDTree`, `scipy.optimize.minimize`, and `scipy.stats.chi2`.
   It's transitively present in the current env (confirmed importable) but not a
   direct pinned dependency — add it, and sanity-check it doesn't conflict with the
   pinned `cudatoolkit-dev=11.7.0` env (should be fine, pure-Python/no CUDA
   interaction, but confirm rather than assume).
3. **No code in `src/cbf/` has been run yet.** Nothing has been executed against a
   real `safety_gsplat.ply` — there's no confirmed-real PLY path to point at (Stage 1
   training run output isn't in this checkout). Before trusting any of the above:
   - Unit test `build_baseline_inputs`: assert identical output whether
     `splats.safety_raw` is `None` or garbage (the baseline-never-reads-safety
     contract).
   - Unit test `ellipsoid.build_sigma` against a hand-computed isotropic case (matches
     Replica's actual default) and one hand-picked anisotropic case.
   - Run `costmap_cbf.py` against a real `safety_gsplat.ply` once one exists, and watch
     for the `zero_fraction` warning from `ply_io.py` — expect it to be large today
     given Stage 2 bug (1).
4. **Measure the real `room0` scene's bounding box** and revisit the placeholder
   `spatial_filter` radius/max_candidates defaults in `configs/cbf/room0_cbf.py`
   accordingly.
5. **Confirm the actual TurtleBot4 footprint radius** (`configs/cbf/room0_cbf.py`'s
   `robot_radius=0.16` is a placeholder) before trusting Minkowski-inflation results.

## Two Stage 2 bugs found while building this (not fixed — out of scope for Stage 3)

Found while confirming the exact `safety_gsplat.ply` schema against the code that
writes it. Both are in `vlm_safety_score.py`, flagged per `CLAUDE.md`'s rule that
Stage 2 "should not need structural changes" — this isn't a structural change, it's a
correctness bug, worth a deliberate decision about whether/when to fix:

1. **Classifier weights never load.** `SemanticDecoder` in `vlm_safety_score.py:34-40`
   is `nn.Linear(16, 256)`, with state_dict keys `linear.weight`/`linear.bias`. The
   classifier actually trained and saved as `classifier.pth`
   (`src/Decoder.py:29-38`/`src/GS3LAM.py:103,492`) is `nn.Conv2d(16, 256,
   kernel_size=1)`, keys `conv.weight`/`conv.bias`. `load_state_dict(state_dict,
   strict=False)` (`vlm_safety_score.py:57`) silently loads nothing because no keys
   match — today's per-splat `class_ids` come from a randomly-initialized `nn.Linear`,
   not the trained decoder. **This means any `safety_gsplat.ply` produced today has
   semantically meaningless safety scores** — a real blocker for a Phase B (real-data)
   Stage 3 evaluation, though not for Phase A (synthetic sanity check) or for testing
   the CBF math itself.
2. **Missing import.** `vlm_safety_score.py:99` calls `glob.glob(...)` but never
   `import glob` — will raise `NameError` at runtime as currently written.

Recommended fix (not applied): change `SemanticDecoder` to match `src/Decoder.py`'s
architecture (or import/reuse that class directly), add `import glob`, and — while
touching this file — consider changing `broadcast_scores_and_save`'s
`np.zeros(num_points, ...)` default to `np.full(..., np.nan)` plus persisting
`class_ids` alongside `safety` in the output PLY, so a future loader *could*
distinguish "genuinely scored 0.0" from "never scored." That last change is a splat-schema
change and should be flagged/confirmed with Aashrut before doing it, per `CLAUDE.md`'s
rule on schema changes.
