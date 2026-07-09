# Stage 3 Implementation Progress

See `ARCHITECTURE.md` §2.3 for the design/math this implements, and the original
design plan (from the planning session) for the full rationale behind every choice
below. As of this update, the full `src/cbf/` library plus both entry scripts exist
and have been smoke-tested against a synthetic scene (see "Verification done" below)
— this is further along than the previous snapshot of this file, which was written
before `eval_cbf_modes.py` existed and before anything had been run.

## Done

All of the below exist. Everything except item (3) in "Not done" has now actually been
run at least once against a synthetic scene (see "Verification done").

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
- `eval_cbf_modes.py` — the 3-mode comparison harness. Verifies start/goal against the
  baseline geometry, runs a `PassthroughFilter` oracle (CBF disabled) for the
  time-to-goal denominator, then runs `NONE`/`ALPHA_SCALE`/`COV_INFLATE` through
  `src/cbf/sim.rollout()` and prints a metrics table. `--phase-a` overrides the loaded
  safety column with a single synthetic hazard on the start→goal line
  (`synthesize_hazard_safety()`) for the sanity check described in `ARCHITECTURE.md`
  §2.3, ahead of any real-scene run.
- `requirements.txt` / `environment.yml` — added `scipy` (used by
  `spatial_filter.py`'s `cKDTree`, `qp_filter.py`'s `scipy.optimize.minimize` and
  `scipy.stats.chi2`). Pure-Python, no CUDA interaction — should not conflict with the
  pinned `cudatoolkit-dev=11.7.0` env, but this has only been confirmed by import, not
  by installing into that actual pinned conda env (which isn't present on this
  machine — see "Verification done").
- `ARCHITECTURE.md` §2.3 and `CLAUDE.md` updated to reflect the above (module layout,
  exact math, confirmed data-contract gotchas, and the two Stage 2 bugs found below).

## Verification done

No real `safety_gsplat.ply` exists in this checkout (no Stage 1 training run has been
done here), and this machine has no conda/GPU env — only system Python with
numpy/scipy present and `plyfile` pip-installed for testing. So verification so far is
a **synthetic smoke test**, not a real-scene run:

- Built a synthetic `safety_gsplat.ply` matching the exact schema (300 background
  splats + 1 obstacle splat, isotropic scale, unnormalized identity quaternion).
- Unit-level checks, all passing: the loader reads the file correctly;
  `ellipsoid.build_sigma` on an isotropic splat matches a hand-computed `scale² · I`;
  `build_baseline_inputs` is confirmed to produce byte-identical output whether
  `safety_raw` is real data or `NaN` (the baseline-never-reads-safety contract);
  `compute_collision_cones`'s `h` value for a robot heading straight at the obstacle
  **exactly matches an independent hand-computed value** (`h = β·γ − δ² = -119.456`),
  confirming the Eq 9/13 math was transcribed correctly; a receding trajectory
  correctly does not activate the cone.
- Found and fixed a real bug: `qp_filter._clip_to_a_max()` hardcoded
  `.astype(np.float32)` on its clipped branch. The SLSQP solver path casts all its
  inputs to `float64` (SciPy's Fortran backend requires it), but `_clip_to_a_max` is
  also used to build the solver's `x0` seed from `u_ref` — so any step where the
  reference control exceeded `a_max` (i.e. almost every real step) silently downcast
  `x0` back to `float32`, crashing with `ValueError: failed to initialize
  intent(inout) array -- expected elsize=8 but got 4`. Fixed to preserve the input
  dtype (`u.dtype`) instead. This would have broken on literally the first real
  rollout — worth a regression test before this is considered done (see "Not done").
- Ran `eval_cbf_modes.py --phase-a` end-to-end on the synthetic scene. Qualitative
  result matched the design's theoretical prediction: `ALPHA_SCALE` reached the goal
  on essentially the same path as `NONE` (`path_length_ratio` 1.08 vs 1.09) but ~3x
  slower (`time_ratio` 3.27 vs 1.21) — i.e. purely longitudinal braking, no rerouting,
  exactly as predicted. `COV_INFLATE` behaved very differently (didn't reach the goal
  within `max_steps` at `gamma∈{0.3, 1.0}` in this scene), consistent with it changing
  the actual routing geometry rather than just the approach speed — but in this
  particular synthetic layout (background clutter placed close to the inflated
  hazard's footprint) the inflated ellipsoid seems to leave too little lateral
  clearance for the QP to route around within the step budget. Not yet determined
  whether that's "COV_INFLATE correctly refusing an unsafe corridor" or "the CBF's
  fixed `k_alpha_base` producing an overly-conservative asymptotic crawl regardless of
  obstacle geometry" (a step-by-step trace showed velocity monotonically decaying
  toward near-zero rather than the robot stopping or oscillating) — this needs a wider
  synthetic corridor and/or a `k_alpha_base` sweep to disambiguate, not more staring at
  this one scene. Flagging as a real open finding, not a known bug.

## Not done — pick up here

1. **Disambiguate the `COV_INFLATE` slow-convergence finding above.** Re-run
   `eval_cbf_modes.py --phase-a` with (a) background splats moved further from the
   hazard (wider corridor) and (b) a `k_alpha_base` sweep, to determine whether the
   observed near-halt is a legitimate "no safe corridor" outcome or an artifact of a
   too-small fixed gain interacting with multiple simultaneous active constraints.
2. **No real `safety_gsplat.ply` has been used.** Everything above ran against a
   hand-built synthetic PLY, not real Stage 1/2 output — there's no confirmed-real PLY
   path to point at in this checkout. Once one exists: run `costmap_cbf.py` against it
   and watch for the `zero_fraction` warning from `ply_io.py` (expect it to be large,
   given Stage 2 bug (1) below), then run `eval_cbf_modes.py` without `--phase-a` for
   the (currently not-yet-meaningful, per that same bug) Phase B numbers.
3. **No formal unit test suite** — the checks in "Verification done" were ad hoc
   scripts run manually, not committed as `pytest`/`unittest` cases. Worth turning
   `build_baseline_inputs`' safety-blindness check and `ellipsoid.build_sigma`'s
   isotropic hand-computation into a real `tests/` module, especially since the
   `_clip_to_a_max` dtype bug above shows the solver path wasn't exercised until now.
4. **`scipy` has not been confirmed inside the actual pinned `cudatoolkit-dev=11.7.0`
   conda env** — only confirmed importable/installable in this machine's plain system
   Python, which has no CUDA/conda setup at all. Low risk (pure-Python package) but
   still unconfirmed in the environment CLAUDE.md flags as deliberately fragile.
5. **Measure the real `room0` scene's bounding box** and revisit the placeholder
   `spatial_filter` radius/max_candidates defaults in `configs/cbf/room0_cbf.py`
   accordingly.
6. **Confirm the actual TurtleBot4 footprint radius** (`configs/cbf/room0_cbf.py`'s
   `robot_radius=0.16` is a placeholder) before trusting Minkowski-inflation results.

## Two Stage 2 bugs found while building this — now fixed

Found while confirming the exact `safety_gsplat.ply` schema against the code that
writes it. Both were in `vlm_safety_score.py`; fixed in a dedicated Stage-2-only session
per `CLAUDE.md`'s rule that Stage 2 "should not need structural changes" (this was a
correctness fix, not a structural one):

1. **Classifier weights never loaded — fixed.** `SemanticDecoder` in
   `vlm_safety_score.py` used to be a from-scratch `nn.Linear(16, 256)` with state_dict
   keys `linear.weight`/`linear.bias`. The classifier actually trained and saved as
   `classifier.pth` (`src/Decoder.py:29-38`/`src/GS3LAM.py:103,492`) is
   `nn.Conv2d(16, 256, kernel_size=1)`, keys `conv.weight`/`conv.bias`. Fixed by deleting
   the reimplementation and importing `src.Decoder.SemanticDecoder` directly, so the two
   can never drift apart again. Since the trained decoder is a 1x1 conv applied to
   per-pixel `(C,H,W)` feature maps in normal use (`src/Evaluater.py:174-175`,
   `src/Loss.py:91`) but Stage 2 has flat per-splat `(N,16)` vectors, the fix reshapes to
   `(N,16,1,1)` before the forward pass and squeezes the `(N,256,1,1)` logits back down
   — mathematically identical to the per-pixel case since a 1x1 conv does no spatial
   mixing. Also changed `load_state_dict(state_dict, strict=False)` to `strict=True`, so
   a key mismatch is now structurally impossible to pass silently.
2. **Missing import — fixed.** `vlm_safety_score.py` called `glob.glob(...)` without
   `import glob`; the import is now present.

**Verification is partial, not complete — flagging honestly rather than overclaiming:**
this fix session's sandbox has no torch installed at all and no conda/GPU environment
anywhere on the machine (same constraint noted in "Not done" item 4 below). Worse,
`src/Decoder.py:33`'s `SemanticDecoder.__init__` hardcodes `.cuda()` in its constructor,
so even a CPU-only torch install couldn't construct the real class to dynamically test
`load_state_dict` there — construction fails before key-matching is ever exercised. So:
- What's confirmed: by code inspection, `vlm_safety_score.py` now instantiates the
  *exact same class* with the *exact same constructor args* (`16, 256`) that
  `src/GS3LAM.py` used to create and save `classifier.pth` — both produce
  `conv.weight`/`conv.bias` keys by construction. The file parses cleanly
  (`python3 -m py_compile vlm_safety_score.py`).
- What's not confirmed: no real `classifier.pth` exists anywhere on this machine (see
  "real Stage 1 output search" below), so the corrected loader has never actually been
  run against real trained weights.
- A regression test exists at `tests/test_semantic_decoder_load.py` (constructs
  `src.Decoder.SemanticDecoder` twice, round-trips a state_dict with `strict=True`,
  asserts no missing/unexpected keys) but has **not been executed** — it needs a
  CUDA-capable environment because of the `.cuda()` issue above. Run it once GPU access
  is available; until then this is a TODO, not a silently-skipped step.

**New finding not in the original two bugs:** `src/Decoder.py:33` hardcodes `.cuda()` in
`SemanticDecoder.__init__`, unconditionally, regardless of the caller's intended device.
On the real dev machine (A2000 GPU, per `CLAUDE.md`) this is harmless, but it means the
module can never be constructed on a CPU-only machine — including this sandbox, which
blocked dynamic verification above. Not fixed here (out of the two-bug scope, and
`src/Decoder.py` wasn't touched, only imported from) — logged as a TODO for whoever next
has GPU access.

**Real Stage 1 output search (requested this session):** searched this repo (`logs/`
and `data/` don't exist in this checkout), the entire `/mnt/c/Users/jaina7/projects` tree,
the WSL home directory, and `~/miniconda3/envs` (empty — no conda envs exist on this
machine at all). **No real `gsplat.ply`, `params.npz`, `classifier.pth`, or
`safety_gsplat.ply` exists anywhere on this machine.** This is an explicit negative
result, not an assumption — there is currently no real Stage 1/2 output in existence to
run the corrected decoder against, so no live class-assignment sanity check was
possible.

Deliberately not done in this fix (per explicit instruction, flagged rather than
silently applied): changing `broadcast_scores_and_save`'s `np.zeros(num_points, ...)`
default to `np.full(..., np.nan)` plus persisting `class_ids` alongside `safety` in the
output PLY. That's a splat-schema change per `CLAUDE.md`'s rule and needs a deliberate
decision from Aashrut, not a silent fix alongside a bug fix.

## Third Stage 2 bug found (2026-07-09) — not yet fixed

Found while running the VLM safety-score consistency check (`GS3LAM_PAPER_SCOPE.md`,
"This Summer's Remaining Experimental Work" item 1) via a standalone script
(`vlm_consistency_check.py`), not while working on Stage 3 — logged as its own section
rather than folded into "Two Stage 2 bugs" above since it's a separate discovery session
and, unlike those two, has **not** been fixed in `vlm_safety_score.py` itself.

`vlm_safety_score.py:29,170` hardcodes `model='gemini-1.5-flash'` for
`query_vlm_safety()`. As of this session that model is fully deprecated — every call
404s with `models/gemini-1.5-flash is not found for API version v1beta, or is not
supported for generateContent`, confirmed against a live API key via
`client.models.list()`. The production Stage 2 script cannot currently run at all,
independent of any prompt- or score-quality question.

Not fixed here — `vlm_safety_score.py` was explicitly out of scope for this session.
For the standalone consistency-check script, `gemini-2.5-flash` was tried next and
found to be mid-deprecation itself (intermittent 404 "no longer available" partway
through a run); settled on the rolling alias `gemini-flash-latest`, which works but
needed two more adjustments beyond a bare model-name swap:
- `max_output_tokens` must be set explicitly (e.g. 1024) — left unset, the model's JSON
  answer was silently truncated mid-value (`{"safety_score": 0.1` with no closing
  brace) on a large fraction of calls, despite `finish_reason=STOP`.
- `thinking_config=types.ThinkingConfig(thinking_budget=0)` should be set — the model
  spends output-token budget on an internal "thinking" pass before the visible JSON
  answer; `gemini-1.5-flash` predates this behavior and never needed it.

Whoever repoints `vlm_safety_score.py` at a working model should carry both
adjustments, not just the model-name swap — see `vlm_consistency_check.py` for the
working config. Which model to standardize on (a rolling alias vs. a pinned dated
version, for reproducibility) is a decision for that session, not made here.
