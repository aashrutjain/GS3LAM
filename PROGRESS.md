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

1. ~~Disambiguate the `COV_INFLATE` slow-convergence finding above.~~ **Done
   (2026-07-11) — see "COV_INFLATE disambiguation experiment" section below.** Short
   answer: fixed-gain artifact (`k_alpha_base`), not a legitimate no-safe-corridor
   refusal, and not QP infeasibility at the default gain. This was run on a newly
   reconstructed synthetic scene (the original one referenced above was never persisted
   to disk — see that section for why), so treat the qualitative mechanism as the
   trustworthy result, not a literal continuation of the numbers quoted above.
2. **No real `safety_gsplat.ply` has been used.** Everything above ran against a
   hand-built synthetic PLY, not real Stage 1/2 output — there's no confirmed-real PLY
   path to point at in this checkout. Once one exists: run `costmap_cbf.py` against it
   and watch for the `zero_fraction` warning from `ply_io.py` (expect it to be large,
   given Stage 2 bug (1) below), then run `eval_cbf_modes.py` without `--phase-a` for
   the (currently not-yet-meaningful, per that same bug) Phase B numbers.
3. ~~**No formal unit test suite**~~ **Done (2026-07-17) — see "Real pytest test
   suite added" section below.** The checks below were ad hoc scripts run manually,
   not committed as `pytest`/`unittest` cases. Turned `build_baseline_inputs`'
   safety-blindness check and `ellipsoid.build_sigma`'s isotropic hand-computation
   into a real `tests/` module, especially since the `_clip_to_a_max` dtype bug above
   shows the solver path wasn't exercised until now.
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
  asserts no missing/unexpected keys). At the time this was written it had **not been
  executed** because it needs a CUDA-capable environment, per the `.cuda()` issue
  below. **Now executed (2026-07-18) — see "SemanticDecoder CPU-construction fix"
  below** — it was the hardcoded `.cuda()`, not the absence of GPU access per se, that
  blocked it; removing that let it run and pass on this CPU-only sandbox.

**New finding not in the original two bugs:** `src/Decoder.py:33` hardcodes `.cuda()` in
`SemanticDecoder.__init__`, unconditionally, regardless of the caller's intended device.
On the real dev machine (A2000 GPU, per `CLAUDE.md`) this is harmless, but it means the
module can never be constructed on a CPU-only machine — including this sandbox, which
blocked dynamic verification above. **Fixed (2026-07-18) — see "SemanticDecoder
CPU-construction fix" below.**

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

## Eval harness bug found and fixed (2026-07-11): COV_INFLATE was graded against its own inflated ellipsoid

Found during a diagnostic session on `eval_cbf_modes.py` before starting the
`COV_INFLATE` corridor-widening / `k_alpha_base` sweep work from "Not done" item 1
below — this session touched only the eval harness, not `src/cbf/`'s math.

**What was found:** `run_mode()` built each `SemanticMode`'s metrics
(`collision_severity`, `near_miss_events`) from that mode's own freshly-constructed
`CBFSafetyFilter.A`/`.s_min`. Traced through `qp_filter.build_semantic_inputs()` and
`semantic_weighting.inflate_covariance()`:
- `NONE` and `ALPHA_SCALE`: `filt.A` is the true geometric `Sigma^-1` — `ALPHA_SCALE`
  only modifies `k_alpha`, never touches `A`.
- `COV_INFLATE`: `filt.A = A / scale` (`scale = 1 + gamma*(1-safety) >= 1`), i.e. the
  **semantically-inflated** ellipsoid the controller was routing around, not the true
  splat boundary.

So `COV_INFLATE`'s severity/near-miss numbers were being measured against its own
artificially widened target, while `NONE`/`ALPHA_SCALE` were measured against the real
one. Any "`COV_INFLATE` has lower severity" result was therefore uninterpretable — it
could mean the mode kept the robot further from the real object, or just that it was
grading itself against an easier target. Also confirms an inconsistency already present
in the file: start/goal validity was already (correctly) checked against a baseline
`NONE` filter (`verify_collision_free`), but per-mode trajectory metrics weren't held to
the same standard.

**What was fixed:** `run_mode()` now takes a `geom_filt: CBFSafetyFilter` parameter — a
single `NONE`-mode filter built once in `main()` (reusing the existing `baseline_filt`
that was already built for the start/goal check) — and uses `geom_filt.c_base/.s_min/
.xyz/.A` for `collision_severity`/`near_miss_events` for all three modes. Each mode's
own filter (`filt`, built from that mode's `qp_cfg`) still drives its own `rollout()`
call, so routing/braking decisions during the sim are unchanged and still reflect what
each mode actually sees — only the evaluation metric now uses one shared true geometry.
A comment in `run_mode()` documents this split so it doesn't need re-deriving.

**Caveat for the "Not done" item 1 disambiguation work below:** the existing
"Verification done" section's `--phase-a` `COV_INFLATE` numbers (the `gamma∈{0.3, 1.0}`
"didn't reach goal within `max_steps`" / "velocity monotonically decaying" finding) were
produced under this buggy self-grading. That finding was about the *controller's*
behavior (did the QP reach the goal, what did velocity do), not the post-hoc severity
metric, so it isn't necessarily invalidated outright — but it has not been re-run since
the fix, and any severity/near-miss numbers alongside it should be re-derived, not
reused, before being cited in the disambiguation experiment.

## COV_INFLATE disambiguation experiment (2026-07-11)

> **CORRECTION (2026-07-18) — read before the numbers below.** The headline `11.91x`
> narrow-corridor slowdown recorded in this section is **inflated by a solver artifact
> worth roughly half the figure**. It was measured under the then-default `scipy_slsqp`
> backend, which a state-by-state re-solve later showed was returning suboptimal,
> over-conservative controls on ~12% of the constrained steps while reporting
> `success=True`. Under the now-default Clarabel backend the same scene and same
> configuration give **`5.896x`** (and `severity` `0.1528`, not `0.165`). Full analysis,
> including the objective-value comparison that establishes which solver is correct, is
> in "Clarabel QP backend added (2026-07-18)" below.
>
> **The qualitative conclusion of this section is unchanged and still stands:** a
> fixed-gain (`k_alpha_base`) artifact, not a legitimate no-safe-corridor refusal. Every
> supporting leg of that argument was re-verified **bit-identical** under both solvers —
> the low-gain traversals (`1.1791x` / `1.2463x` at `k_alpha_base` 0.1 / 0.3), the wide
> corridor collapsing to `1.09x`, and the high-gain infeasibility counts (`2` at 3.0,
> `40` at 10.0). Narrow-at-default is still ~5.4x slower than wide-at-default, so the
> corridor is still being navigated far too conservatively for reasons that are about the
> gain, not the geometry. The mechanism is real and still dominant; only its magnitude
> was overstated.
>
> The original numbers below are **left exactly as recorded** rather than edited in
> place — this is a visible correction, not a rewrite of what was measured at the time.

Follow-up to the eval-harness fix above, addressing "Not done" item 1: is the
`COV_INFLATE` near-halt finding a legitimate no-safe-corridor refusal, a `k_alpha_base`
fixed-gain artifact, or the QP hitting its infeasibility fallback? Only `eval_cbf_modes.py`
and a new scene-generation script were touched — `src/cbf/`'s math is unchanged.

**Scene reconstruction, not reuse.** The original synthetic scene referenced in
"Verification done" above was built ad hoc in a prior sandbox session and never written
to disk — confirmed absent by searching this machine. It could not literally be re-run.
Rebuilt one matching its documented properties (300 background splats + 1 hazard splat,
isotropic scale, identity quaternion) via a new, seeded, parameterized generator:
`scripts/gen_cbf_synthetic_scene.py` (`--collar-radius`, `--seed`, `--out`). Straight-line
path from `(-2.5,0,0)` to `(2.5,0,0)`; one hazard splat plus a multi-ring "collar" of
clutter splats encircling the corridor around it, so the robot must detour in both y and
z, not just dodge sideways in one axis; ~150 far-field bulk splats outside the spatial
filter's radius cap for scene bulk only, not part of the geometry under test.

Two real bugs in the *scene design itself* (not in `src/cbf/` or `eval_cbf_modes.py`)
were found and fixed before any of the numbers below were trustworthy — flagging both
since they'd silently invalidate a re-run that skipped them:
- **Exact-symmetry degeneracy.** With the hazard placed exactly on the robot's dead-on
  approach axis, line-of-sight `r` and velocity `v` stayed perfectly colinear, so
  `collision_cone.py`'s `w = gamma*Av - delta*Ar` had zero lateral component throughout —
  the one-step QP had no gradient information to ever discover a lateral detour and could
  only brake, regardless of how much clearance existed off-axis. Fixed by offsetting the
  hazard+collar cluster 0.03m off the path centerline (`cluster_offset_y` in the
  generator) — this is a property of *any* perfectly-centered synthetic test with this
  formulation, not specific to this scene, worth remembering for future synthetic scenes.
- **Exploitable collar flank.** A first collar design (2 rings, narrow x-extent) left a
  shallow diagonal bypass just past the ring's finite axial extent, where the hazard's own
  blocking radius had already tapered enough to slip through — `COV_INFLATE` threaded it
  even when the on-axis corridor was nominally fully closed by hand calculation. Fixed by
  widening the collar to 5 rings spanning the hazard's full blocking extent (see script
  docstring for the exact geometry argument).

Both scene files (`collar_radius=0.90` "narrow", `collar_radius=1.30` "wide") were
generated with `--seed 42` and are reproducible from the script; the `.ply` files
themselves are not committed (ephemeral scratch artifacts), only the generator is.

**Unrelated fix bundled into this same commit, undocumented at the time.** This
commit (`6301e28`) also corrected the GS3LAM citation's author list in
`ARCHITECTURE.md` and `GS3LAM_PAPER_SCOPE.md` — both had been transcribed with
fabricated authors (`ARCHITECTURE.md` read "Li, M., Liu, S., Zhou, H.";
`GS3LAM_PAPER_SCOPE.md` read "Li, Liu & Zhou") since the sessions that first wrote
those docs, corrected here to the real author list, "Li, Zhang, Wang, Shen"
(matching `README.md`'s bibtex, which had been correct all along). Neither the fix
nor its rationale was mentioned in this commit's message or in this file at the
time. Discovered via `git log -S` archaeology in a later session (2026-07-17), when
asked to confirm the fix had actually landed — it had. A real, correct fix; just an
undocumented one until now.

**Step 0 — clean baseline, narrow corridor** (baseline gap ≈10.8cm before any
inflation). All three modes reach the goal at both gammas. `collision_severity` /
`near_miss_events` are the (now-correctly-shared, per the eval-harness fix) true
geometric signed distance in Mahalanobis units, not meters — negative means the
trajectory entered a splat's true confidence ellipsoid.

| mode | gamma | reached_goal | severity | near_miss | time_ratio | infeasible_count |
|---|---|---|---|---|---|---|
| NONE | — | True | -0.064 | 1 | 1.06 | 0 |
| ALPHA_SCALE | — | True | -0.425 | 1 | 1.06 | 0 |
| COV_INFLATE | 0.3 | True | 0.046 | 1 | 1.41 | 0 |
| COV_INFLATE | 1.0 | True | 0.165 | 1 | **11.91** | 0 |

NONE/ALPHA_SCALE are gamma-invariant (expected — gamma only affects `COV_INFLATE`).
`COV_INFLATE` at `gamma=1.0` does *not* fail to reach the goal (unlike the original,
unpersisted scene's finding) — it reaches it nearly 12x slower than the oracle while
achieving the best (most positive) safety margin of the four rows, with a
near-identical path length to NONE (barely any lateral detour). Velocity trace:
73% of the 1596-step trajectory spent below 1cm/s.

**Step 1 — infeasibility check.** `infeasible_count=0` for every row above, including
the pathological `gamma=1.0` run. Traced the full rollout directly (bypassing the CLI):
`min_h` was negative on 96% of steps, yet the SLSQP solver reported `success=True` every
single step — the QP always found a feasible point, it just kept choosing near-zero net
acceleration. **This rules out the QP-infeasibility/fallback-to-max-braking hypothesis at
the default `k_alpha_base=1.0`** — the solver never once took the `_max_braking()` branch
in `qp_filter.py`. (It does start firing at higher gains — see Step 3.)

**Step 2 — wider corridor** (baseline gap ≈50.8cm, everything else identical, including
`k_alpha_base=1.0`):

| mode | gamma | reached_goal | severity | near_miss | time_ratio | infeasible_count |
|---|---|---|---|---|---|---|
| COV_INFLATE | 0.3 | True | 0.359 | 0 | 1.07 | 0 |
| COV_INFLATE | 1.0 | True | 1.190 | 0 | **1.09** | 0 |

Same gamma=1.0 inflation, same `k_alpha_base`, only the surrounding geometry changed —
and the 12x slowdown collapses to 1.09x (essentially indistinguishable from NONE/
ALPHA_SCALE's 1.06x) while achieving an even better safety margin. This is the cleanest
single piece of evidence: whatever caused the Step 0 slowdown depends heavily on how
tight the actual remaining clearance is, not on `COV_INFLATE`/gamma=1.0 in isolation.

**Step 3 — `k_alpha_base` sweep** (narrow corridor, `gamma=1.0` fixed, values spanning 2
orders of magnitude):

| k_alpha_base | reached_goal | severity | time_ratio | infeasible_count | frac(min_h<0) |
|---|---|---|---|---|---|
| 0.1 | True | -0.050 | 1.18 | 0 | 0.62 |
| 0.3 | True | 0.066 | 1.25 | 0 | — |
| 1.0 | True | 0.165 | 11.91 | 0 | 0.96 |
| 3.0 | **False** | 1.825 | (timeout) | 2 | — |
| 10.0 | **False** | 1.826 | (timeout) | 40 | 0.98 |

Non-monotonic and mechanistically clean: **lower** `k_alpha_base` (0.1, 0.3) makes the
barrier *stricter while still safe* (per the class-K constraint `w^Tu >= -(k_alpha/2)*h`,
a smaller gain means less permissive pre-violation braking), so the robot never lets `h`
drift deeply negative and threads the corridor briskly (~1.2x oracle time). The *default*
`k_alpha_base=1.0` is permissive enough pre-violation to let `h` go substantially
negative, and then the same fixed gain governs a comparatively slow proportional
recovery back toward `h=0` — this is the Step 0 crawl. Push the gain higher still (3.0,
10.0) and the now-more-aggressive pre-violation approach drives `h` even further
negative before the barrier engages; the demanded recovery acceleration this time
exceeds `a_max`, and the solver genuinely goes infeasible repeatedly (`infeasible_count`
40 at `k=10.0`), falling back to max-braking and never reaching the goal within the step
budget — i.e. hypothesis 3 (QP infeasibility) *does* occur, but only as a downstream
consequence of pushing the same fixed-gain mechanism further, not as an independent
explanation of the original default-gain finding.

**Conclusion.** Of the three hypotheses, the evidence points squarely at **hypothesis
2, the fixed-gain artifact**, as the explanation for the original slow-convergence
finding at the default `k_alpha_base=1.0`:
- Not hypothesis 1 (legitimate no-safe-corridor refusal) — the same corridor is
  trivially passable at `k_alpha_base` 0.1/0.3 (~1.2-1.25x oracle time) and at the wider
  corridor for any tested gamma (~1.1x). There is a safe corridor; the default gain just
  navigates it very conservatively once it has (permissively) let itself violate the
  nominal boundary.
- Not hypothesis 3 (QP infeasibility) at the gain actually used to produce the original
  finding — confirmed by direct trace, `infeasible_count=0` throughout. Infeasibility
  is real but only appears at gains well above default, as a *further* consequence of
  the same permissive/recover-later dynamic, not a separate mechanism.
- Hypothesis 2 is further supported by Step 2: identical `k_alpha_base` and gamma, only
  the geometry loosened, and the pathology vanished.

**Caveats, stated plainly.** This is four synthetic runs (plus a 5-point gain sweep) on
one hand-built, geometrically simple corridor with a spherical robot and a toy PD
tracker — not a claim about real `room0` geometry, real safety scores, or a tuned
`k_alpha_base`. The mechanism (permissive-then-slow-recovery under a fixed class-K gain)
is a property of the collision-cone CBF formulation itself and should generalize
qualitatively, but the *specific* threshold gains here (1.0 fine-but-slow, 3.0+
infeasible) are specific to this scene's geometry and are not to be read as tuning
recommendations for the real robot. `k_alpha_base` tuning against real geometry, and
whether a scheduled/adaptive gain (rather than one fixed constant) is warranted, remains
open — not decided by this experiment.

## COV_INFLATE disambiguation — independent reproduction (2026-07-12)

Re-ran Steps 0–3 of the disambiguation above, from scratch, under the (already-fixed)
eval harness, to turn "trust the mechanism, not the literal numbers" into a checked
result. The experiment above was worth reproducing precisely *because* it carried that
caveat: its numbers came from a scene that had to be reconstructed (the original was
never persisted), so an independent re-run tests whether the reconstruction + the fixed
harness actually reproduce, not just whether the story is self-consistent. "Not done"
item 1 stays struck-through — this section records that it was independently reproduced,
it does not reopen it.

**Method — no committed source changed.** Regenerated both seed-42 scenes from the
committed `scripts/gen_cbf_synthetic_scene.py` (`--collar-radius 0.90` narrow,
`1.30` wide, `--seed 42`) and drove Steps 0–3 from a scratch script that *imports* the
committed helpers (`eval_cbf_modes.build_qp_cfg` / `synthesize_hazard_safety` /
`PassthroughFilter` / `verify_collision_free`, `src.cbf.sim.rollout`) rather than
reimplementing them. Grading uses the exact same shared-`geom_filt` true-geometry path
as `run_mode()` (the eval-harness fix), replicated verbatim. Same config defaults as the
committed run (`k_alpha_base=1.0`, `a_max=1.0`, `robot_radius=0.16`, `dt=0.05`,
`max_steps=2000`, hazard `radius=0.5`/`safety=0.1`, `near_miss_thresh=0.3`), same
corridor path `(-2.5,0,0)→(2.5,0,0)`. Env: system Python, `numpy 2.2.6`, `scipy 1.15.2`
(the committed run's env was not recorded). The scratch scenes/driver are ephemeral
(not committed), same as the original run's `.ply` files.

**Result: reproduced to the displayed precision on every row.** Fresh numbers, with the
committed value in brackets where the table above quoted one:

*Step 0/1 — narrow, `k_alpha_base=1.0`* (severity = shared true-geometry signed distance,
Mahalanobis units):

| mode | gamma | reached | severity | near_miss | time_ratio | infeasible | frac(min_h<0) |
|---|---|---|---|---|---|---|---|
| NONE | — | True | -0.064 [-0.064] | 1 | 1.06 [1.06] | 0 [0] | 0.55 |
| ALPHA_SCALE | — | True | -0.425 [-0.425] | 1 | 1.06 [1.06] | 0 [0] | 0.57 |
| COV_INFLATE | 0.3 | True | +0.046 [0.046] | 1 | 1.41 [1.41] | 0 [0] | 0.68 |
| COV_INFLATE | 1.0 | True | +0.165 [0.165] | 1 | **11.91 [11.91]** | 0 [0] | 0.96 |

The `gamma=1.0` crawl reappears identically: 1596-step trajectory (matches committed),
**73% of steps below 1cm/s** (committed prose said 73%), `min_h<0` on **96%** of steps
(committed said "96%"). Step 1's discriminator holds exactly: `infeasible_count=0` on
every Step 0 row including the pathological one — the `_max_braking()` fallback never
fires at the default gain, so the SLSQP solver reports `success=True` every step while
still letting `h` sit negative and choosing near-zero net acceleration. Hypothesis 3 is
ruled out at the default gain, reproduced.

*Step 2 — wide corridor, identical `k_alpha_base=1.0`:*

| mode | gamma | reached | severity | near_miss | time_ratio | infeasible | frac(min_h<0) |
|---|---|---|---|---|---|---|---|
| COV_INFLATE | 0.3 | True | +0.359 [0.359] | 0 | 1.07 [1.07] | 0 [0] | 0.28 |
| COV_INFLATE | 1.0 | True | +1.190 [1.190] | 0 | **1.09 [1.09]** | 0 [0] | 0.40 |

Same gamma, same gain, only the geometry loosened — the 11.91× slowdown collapses to
1.09×, reproduced exactly. The lower `frac(min_h<0)` (0.40 vs 0.96) is the mechanism made
visible: in the wide corridor the controller barely lets `h` go negative, so there is
almost nothing for the fixed-gain recovery to crawl back from.

*Step 3 — narrow, COV_INFLATE, `gamma=1.0` fixed, `k_alpha_base` sweep:*

| k_alpha_base | reached | severity | time_ratio | infeasible | frac(min_h<0) |
|---|---|---|---|---|---|
| 0.1 | True | -0.050 [-0.050] | 1.18 [1.18] | 0 [0] | 0.62 [0.62] |
| 0.3 | True | +0.066 [0.066] | 1.25 [1.25] | 0 [0] | 0.63 |
| 1.0 | True | +0.165 [0.165] | 11.91 [11.91] | 0 [0] | 0.96 [0.96] |
| 3.0 | **False** | +1.825 [1.825] | (timeout) | 2 [2] | 0.99 |
| 10.0 | **False** | +1.826 [1.826] | (timeout) | 40 [40] | 0.98 [0.98] |

Non-monotonic time_ratio and the infeasibility onset at `k≥3.0` both reproduce exactly,
including `infeasible_count` 2 and 40.

**Cross-check against the stock CLI.** `python eval_cbf_modes.py --ply-path narrow.ply
--phase-a --cov-gamma 1.0 --start=-2.5,0,0 --goal=2.5,0,0` printed COV_INFLATE
`severity=0.16461`, `infeasible_count=0`, `time_ratio=11.91` (and NONE `-0.06385`, ALPHA
`-0.42509`) — identical to the driver's Step 0 `gamma=1.0` row. Confirms the scratch
driver reproduces the committed harness, not a subtly different computation.

**Verdict: confirms the committed conclusion (hypothesis 2, fixed-gain artifact).** The
reproduction is not merely qualitatively consistent — on this machine (`numpy 2.2.6` /
`scipy 1.15.2`) it matched the committed numbers to every displayed digit, on all four
steps and the CLI cross-check. So the earlier "trust the mechanism, not the numbers"
hedge can be tightened for *this* scene: both the mechanism and the literal numbers are
reproducible from the committed seed-42 generator. That the SLSQP path is bit-stable
across (at least these) scipy versions is a mild bonus finding, not something to lean on.

**Caveats unchanged.** This strengthens confidence in the *reproduction*, not the
*generality*. It is still one hand-built, geometrically simple corridor with a spherical
robot and a toy PD tracker; the specific threshold gains (1.0 fine-but-slow, 3.0+
infeasible) remain properties of this scene, not tuning recommendations. The open
questions the original section flagged — `k_alpha_base` against real `room0` geometry,
and whether a scheduled/adaptive gain beats one fixed constant — are untouched by a
reproduction and remain open. Real-scene numbers are still blocked (no real
`safety_gsplat.ply` exists on this machine).

## Third Stage 2 bug found (2026-07-09) — now fixed (2026-07-12)

Found while running the VLM safety-score consistency check (`GS3LAM_PAPER_SCOPE.md`,
"This Summer's Remaining Experimental Work" item 1) via a standalone script
(`vlm_consistency_check.py`), not while working on Stage 3 — logged as its own section
rather than folded into "Two Stage 2 bugs" above since it was a separate discovery
session. Fixed in `vlm_safety_score.py` in a dedicated Stage-2-only session (2026-07-12).

**The bug.** `vlm_safety_score.py:29,170` hardcoded `model='gemini-1.5-flash'` for
`query_vlm_safety()`. That model is fully deprecated — every call 404s with
`models/gemini-1.5-flash is not found for API version v1beta, or is not supported for
generateContent`, confirmed against a live API key via `client.models.list()`. The
production Stage 2 script could not run at all, independent of any prompt- or
score-quality question.

**Model decision (made this session): pinned to `gemini-3.5-flash`.** The
consistency-check script had settled on the rolling `gemini-flash-latest` alias (after
`gemini-2.5-flash` was found mid-deprecation), but that alias is deliberately *not*
carried into the production script: Google documents `*-latest` aliases as
experimental / not-for-production, and a rolling alias means a later re-run could
silently use a different model than the one reported in the write-up. `gemini-3.5-flash`
is the current stable GA release; pinning a dated/stable name is what makes the paper's
reported model reproducible. This is a research decision (model behind the safety
scores), recorded here rather than made silently.

**What was fixed in `vlm_safety_score.py`:**
- Model name pulled into a single `GEMINI_MODEL = "gemini-3.5-flash"` constant, referenced
  at the one real call site — so a future model swap is a one-line change, not a
  grep-and-replace across two hardcoded spots.
- Both config adjustments already proven necessary in `vlm_consistency_check.py` were
  carried over (not assumed unnecessary just because the model name changed):
  - `max_output_tokens=1024` — left unset, current "thinking" models silently truncate
    the JSON answer mid-value (`{"safety_score": 0.1` with no closing brace) despite
    `finish_reason=STOP`.
  - `thinking_config=types.ThinkingConfig(thinking_budget=0)` — disables the internal
    "thinking" pass that `gemini-1.5-flash` predated and never needed.
- Removed dead line 29 (`model = genai.GenerativeModel('gemini-1.5-flash')`): a leftover
  from the old `google.generativeai` SDK. `genai.GenerativeModel` does not exist in the
  new `google.genai` client the file already uses (`client = genai.Client(...)`), so the
  line raised `AttributeError` at import — the module was un-importable — and the `model`
  global it defined was never referenced anywhere.
- Closed the empty-key gap: `GEMINI_API_KEY = ""` (never loaded from anywhere) was
  replaced with `load_dotenv()` + `os.environ["GEMINI_API_KEY"]`, mirroring
  `vlm_consistency_check.py`, so the module-level `client` authenticates from `.env`.
  Found adjacent to the model bug; the script could not authenticate regardless of model.

**Verified against the real API — not just by inspection.** Ran 3 live calls through the
actual `vlm_safety_score.query_vlm_safety()` (imported from the real module, not a
reimplementation; the classifier-path-only `torch`/`src.Decoder` imports were stubbed
since they aren't installed here and the function never uses them). All 3 crops
(`assets/vlm_consistency/images/`) returned: **no 404, a valid parsed JSON
`safety_score` float, no truncation** — wall `1.000`, wooden chair `0.150`, fragile vase
`0.000` (semantically plausible, though that ordering was not the thing under test). This
is a smoke test — a handful of calls confirming the model/config path works end-to-end —
**not** a full-pipeline run: there is still no real `safety_gsplat.ply` on this machine,
so Stage 2 has not been run against real Stage 1 output, and score *quality/consistency*
on real hero-frame crops remains the open consistency-check question in
`GS3LAM_PAPER_SCOPE.md`, unaffected by this fix.

## Real pytest test suite added (2026-07-17)

Answers "Not done" item 3 above. Only `tests/`, a new root `conftest.py`, and this file
changed — `src/cbf/` itself is untouched (no behavior changes; solver choice in
`qp_filter.py` is explicitly next-session scope, not this one).

Added a root `conftest.py` (previously none existed) that inserts the repo root onto
`sys.path`, since there's no `pyproject.toml`/`setup.cfg` installing this project as a
package — without it, plain `pytest` (as opposed to `python -m pytest`) only puts
`tests/` itself on `sys.path`, and `from src.cbf... import ...` / `from eval_cbf_modes
import ...` would not resolve.

Four new test modules, porting the ad hoc "Verification done" checks above one-to-one,
plus one new addition:

- `tests/test_cbf_qp_filter.py` — the `_clip_to_a_max` dtype regression (asserts a
  `float64` input whose norm exceeds `a_max` is clipped without being downcast — the
  exact bug that crashed SLSQP before, now with a companion float32 case) and the
  `build_baseline_inputs` safety-blindness contract (byte-identical `(A, s_min,
  k_alpha)` whether `safety_raw` is real data, `NaN`, or `None`).
- `tests/test_cbf_ellipsoid.py` — `ellipsoid.build_sigma` on an isotropic splat matches
  hand-computed `scale² · I` (using an unnormalized identity-equivalent quaternion, to
  exercise `normalize_quaternion` rather than assume pre-normalized input).
- `tests/test_cbf_collision_cone.py` — `compute_collision_cones`'s head-on `h` value and
  the receding-trajectory non-activation check. **Caveat:** the original ad hoc scene's
  exact numbers were never persisted (same situation `scripts/gen_cbf_synthetic_scene.py`
  documents for the COV_INFLATE scene) — confirmed absent via repo-wide grep for
  `119.456`. Reconstructed a scene matching the "robot heading straight at a synthetic
  obstacle" description (isotropic unit-scale splat, `A = I`, velocity exactly parallel
  to line-of-sight) where the general `h = beta*gamma - delta²` collapses to `-|v|²·c_m²`
  independent of distance; solved `c_m` backward so this closed form reproduces the exact
  pinned value `h = -119.456`. This pins the *value* PROGRESS.md recorded, not the
  original scene's literal (lost) numbers — flagging the distinction rather than
  overclaiming a byte-for-byte port.
- `tests/test_cbf_cov_inflate_regression.py` — **new, did not exist before.** Pins the
  digit-exact COV_INFLATE disambiguation numbers (Step 0/2/3 tables, seed=42 narrow/wide
  corridor scenes) as regression assertions: Step 0 narrow corridor `gamma=1.0`
  (`severity≈0.165`, `time_ratio≈11.91`, `infeasible_count=0`), Step 2 wide corridor
  `time_ratio` collapsing to `≈1.09`, and Step 3's `infeasible_count` at
  `k_alpha_base=3.0` (`2`) and `10.0` (`40`). Reconstructs both scenes from the committed
  `scripts/gen_cbf_synthetic_scene.py --seed 42` generator and drives them through
  `eval_cbf_modes.py`'s own helpers (`build_qp_cfg`, `synthesize_hazard_safety`,
  `PassthroughFilter`) + `src.cbf.sim.rollout`, mirroring the independent-reproduction
  method already validated in the section above rather than reimplementing the harness.
  This is the test that will catch it if the upcoming Clarabel solver swap silently
  changes behavior — the whole point of writing it now, before that swap, not after.

**Run and confirmed passing, this session, on this machine** (`numpy 2.2.6`, `scipy
1.15.2`, `pytest 6.2.5` — same versions the independent COV_INFLATE reproduction used,
which is why the digit-exact assertions above hold here):

```
$ python3 -m pytest -v --ignore=tests/test_semantic_decoder_load.py
collected 11 items

tests/test_cbf_collision_cone.py::test_compute_collision_cones_head_on_h_matches_hand_computed_value PASSED
tests/test_cbf_collision_cone.py::test_compute_collision_cones_receding_trajectory_does_not_activate_cone PASSED
tests/test_cbf_cov_inflate_regression.py::test_step0_narrow_corridor_gamma1_pathological_slowdown PASSED
tests/test_cbf_cov_inflate_regression.py::test_step2_wide_corridor_gamma1_time_ratio_collapses PASSED
tests/test_cbf_cov_inflate_regression.py::test_step3_narrow_corridor_k_alpha_sweep_infeasible_count[3.0-2] PASSED
tests/test_cbf_cov_inflate_regression.py::test_step3_narrow_corridor_k_alpha_sweep_infeasible_count[10.0-40] PASSED
tests/test_cbf_ellipsoid.py::test_build_sigma_isotropic_matches_hand_computed_scale_squared_identity PASSED
tests/test_cbf_qp_filter.py::test_clip_to_a_max_preserves_float64_dtype_when_clipped PASSED
tests/test_cbf_qp_filter.py::test_clip_to_a_max_preserves_float32_dtype_when_clipped PASSED
tests/test_cbf_qp_filter.py::test_build_baseline_inputs_is_identical_whether_safety_raw_is_real_or_nan PASSED
tests/test_cbf_qp_filter.py::test_build_baseline_inputs_is_identical_whether_safety_raw_is_none_or_nan PASSED

11 passed in 78.56s
```

**Why `--ignore=tests/test_semantic_decoder_load.py` is needed:** that pre-existing test
(see "Two Stage 2 bugs found" above) fails at *collection* (`ModuleNotFoundError: No
module named 'torch'`) on this torch-less sandbox, which aborts the entire pytest session
before any test runs — not something introduced this session, and not fixed here (out of
scope; it's already documented above as blocked on a CUDA-capable environment). Plain
`pytest` (no `--ignore`) will hit this same collection error in this environment; use the
flag, or run this session's four `test_cbf_*` modules directly, until a GPU/torch
environment is available.

**Not covered, deliberately:** `_solve_scipy_slsqp` itself (the SLSQP call site) has no
direct unit test — it's exercised indirectly through every `test_cbf_cov_inflate_
regression.py` rollout (thousands of real solves, all passing), but there's no isolated
test asserting its output against a hand-solved QP. Worth adding once the Clarabel swap
happens, so both backends can be checked against the same known-good solution.
**Update (2026-07-18) — now covered.** The swap happened; see "Clarabel QP backend added"
below. `tests/test_cbf_qp_backends.py` asserts both backends against hand-solved QPs.
Writing it immediately caught a sign error in the *test's* own setup — both backends
failed identically, which is what identified the test rather than the code as wrong.
Worth recording, since an identical failure across two unrelated solvers was the first
direct evidence they implement the same constraint.

## Clarabel QP backend added (2026-07-18)

**Scope.** Solver mechanics only. `collision_cone.py`, `ellipsoid.py`, and
`semantic_weighting.py` are untouched — the geometry was already verified and did not
need to move. Changed: `src/cbf/qp_filter.py` (new backend + docstring),
`requirements.txt`/`environment.yml` (new pinned dep), root `conftest.py` (new
`--solver` option), `tests/test_cbf_cov_inflate_regression.py` (threads the selected
solver through; no assertion values touched), and one new test module.

**What changed.** `clarabel==0.11.1` is now a pinned dependency (pure Rust-compiled
manylinux wheel — installs in seconds, needs no CUDA and no compiler, which was the main
practical worry). `_solve_clarabel` is registered alongside `_solve_scipy_slsqp` in
`_SOLVERS` (`src/cbf/qp_filter.py:175`). **The SLSQP path is unchanged and still
selectable**. `CBFQPConfig.solver` defaulted to `"scipy_slsqp"` when this section was
first written; it was flipped to `"clarabel"` later the same day — see "Resolved"
under "Verdict" below.

**Reformulation.** Clarabel's standard conic form is
`min (1/2)x'Px + q'x s.t. Ax + s = b, s in K`, with `x = u`:

- *Objective.* `||u - u_ref||^2 = u'u - 2 u_ref'u + const`; the constant doesn't move the
  argmin, so `P = 2I`, `q = -2 u_ref`. Note this is 2x `_solve_scipy_slsqp`'s objective
  (it minimizes `0.5*||u - u_ref||^2`) — same argmin, different reported objective value.
  Nothing currently compares objective values across backends, but it would bite if
  anything did.
- *CBF rows.* Upstream asserts `w_i'u >= -(k_alpha_i/2) h_i`, i.e. `w @ u + rhs >= 0` with
  `rhs = 0.5*k_alpha*h`. A nonnegative cone gives `s >= 0`, so a row reads `A x <= b`;
  negating yields `A_cbf = -w`, `b_cbf = rhs`. One row per active splat, all stacked into
  a single `NonnegativeConeT(n_active)` — the upstream active mask has already done the
  stacking. Because the active set is gated on `h <= 0`, every `rhs` entry is `<= 0`.
- *Actuator bound.* `||u|| <= a_max` goes in as a genuine `SecondOrderConeT(4)` — not a
  linear row, not a per-axis box. `A_soc = [0; -I]`, `b_soc = [a_max, 0, 0, 0]`, so
  `s = b - A u = [a_max; u]`, i.e. `a_max >= ||u||`. This was the entire reason Clarabel
  was preferred over OSQP/quadprog in the first place.

**Status mapping.** Strict: only `SolverStatus.Solved` counts as success; every other
status returns `infeasible=True` and routes into the *existing* `_max_braking()` fallback
at `qp_filter.py:259-260`. No second fallback mechanism was built. The strictness was a
deliberate call with a flagged risk — that `AlmostSolved` (reduced-tolerance convergence)
might fire often and inflate `infeasible_count`. **It did not fire once.** Across every
rollout below, the only statuses observed were `Solved` and `PrimalInfeasible`, so the
strict-vs-lenient question turned out to be empirically moot on these scenes. Recording
that as a measured non-issue, not a settled principle — it could differ on real geometry.

**Test comparison.** Both backends, identical scenes, `--solver` selecting the backend:

| Scene / gain | metric | `scipy_slsqp` | `clarabel` | delta |
|---|---|---|---|---|
| narrow, k=1.0 | `reached_goal` | True | True | — |
| | `infeasible_count` | 0 | 0 | — |
| | `severity` | 0.164610 | 0.152777 | −7.19% |
| | **`time_ratio`** | **11.9104** | **5.8955** | **−50.50%** |
| | `path_length_ratio` | 1.0010 | 1.0015 | +0.05% |
| | `near_miss` | 1 | 1 | — |
| wide, k=1.0 | all metrics | — | — | **bit-identical** |
| narrow, k=3.0 | `reached_goal` / `infeasible_count` | False / 2 | False / 2 | — |
| narrow, k=10.0 | `reached_goal` / `infeasible_count` | False / 40 | False / 40 | — |
| narrow, k=0.1 | `time_ratio` | 1.1791 | 1.1791 | **bit-identical** |
| narrow, k=0.3 | `time_ratio` | 1.2463 | 1.2463 | **bit-identical** |

`reached_goal` and `infeasible_count` agree **exactly everywhere**, including the
`PrimalInfeasible` counts landing on precisely SLSQP's `2` and `40` — two unrelated
algorithms declaring infeasibility on the same steps is strong evidence the reformulation
is faithful. Every gain in the sweep is bit-identical except one. **The entire divergence
is the single default-gain narrow-corridor point, where `time_ratio` halves.**

**Which solver is right?** Answered directly rather than assumed: re-solving every
sub-problem of the narrow k=1.0 rollout with *both* backends and comparing objective
values and constraint satisfaction at identical states.

```
=== rollout driven by scipy_slsqp  (1478 both-feasible steps) ===
  objective ||u-u_ref||^2 : slsqp mean 1.024932 | clarabel mean 1.022364
  max objective excess of clarabel over slsqp : +3.765e-08
  max objective excess of slsqp over clarabel : +7.575e-02
  min CBF slack (w@u+rhs, must be >= 0): slsqp -9.303e-04 | clarabel -9.303e-04
  max ||u||  (bound 1.0)               : slsqp 0.999833 | clarabel 0.999818
```

Clarabel never exceeds SLSQP's objective by more than `3.8e-08` (float noise), while
SLSQP exceeds Clarabel's by up to `7.6e-02` on ~12% of steps. Both respect the actuator
bound; both carry the same small `-9.3e-04` worst-case CBF slack, so neither is buying
speed by violating constraints. **SLSQP was returning suboptimal — over-conservative —
controls on a minority of constrained steps while reporting `success=True`.** Driving the
same rollout with Clarabel and re-solving with SLSQP, the two agree to `1.6e-09`: SLSQP's
suboptimality is state-dependent, appearing only on the hard states its own trajectory
wanders into. That is a self-reinforcing loop, which is why a ~12%-of-steps defect
compounds into a 2x traversal time.

**What this does to the COV_INFLATE conclusion.** The qualitative finding — *fixed-gain
artifact, not a legitimate no-safe-corridor refusal* — **survives intact, and every leg of
its supporting argument is bit-identical under Clarabel**: low gains still traverse
easily (1.18x/1.25x), the wide corridor still collapses to 1.09x, infeasibility still
appears only at gains well above default with the same counts. Narrow-at-default is still
~5.4x slower than wide-at-default, so the corridor is still being navigated far too
conservatively for geometric reasons.

What changes is the *magnitude*: roughly half of the headline "11.91x pathological
slowdown" was SLSQP suboptimality, not the fixed-gain mechanism. The mechanism is real
and still dominant; the number overstated it. Stating that plainly because it is a
correction to a previously recorded result, not a footnote.

**Verdict at the time of the swap: default NOT changed, pinned values NOT updated.** The
pre-agreed rule was to flip `CBFQPConfig.solver` to `clarabel` only if the regression came
back clean, where a >~2% metric shift counts as not-clean. A 50% `time_ratio` move is far
outside that, so the swap was left selectable-but-not-default pending a decision. The
evidence said Clarabel is the correct solver and SLSQP the inaccurate one — but "the new
solver is right and the old recorded number was inflated" is a research call about a
recorded finding, not a coding call, so it was flagged rather than silently applied.

**Resolved (2026-07-18): the swap was accepted on the strength of the state-by-state
re-solve above.** `CBFQPConfig.solver` now defaults to `"clarabel"`
(`src/cbf/qp_filter.py:50`). `scipy_slsqp` is retained as a valid, still-tested fallback
backend — it is no longer the default, but it is not deprecated and its code path is
unchanged.

Because the two backends genuinely produce different (both real) numbers on exactly one
scenario, the two affected pins are now keyed on the backend in
`NARROW_K1_EXPECTED` (`tests/test_cbf_cov_inflate_regression.py`), so **both** backends
stay regression-guarded and both suites run green:

| assertion (narrow, k=1.0) | `clarabel` (default, trusted) | `scipy_slsqp` (fallback) |
|---|---|---|
| `severity` | `0.1528` | `0.1646` |
| `time_ratio` | `5.896` | `11.91` |

Every other pinned value is single-valued and untouched — confirmed bit-identical across
backends. The comment above that dict is explicit that SLSQP's row is retained to guard
the fallback path, **not** because it is the number to quote.

The root `conftest.py` `--solver` default is now read off `CBFQPConfig` rather than
hardcoded, so a plain no-flag run always exercises whatever the library actually defaults
to. A hardcoded literal there would have silently kept the default run on SLSQP after
this flip — and a run on the wrong backend looks identical to a run on the right one
until an assertion happens to disagree.

```
$ python3 -m pytest -q --ignore=tests/test_semantic_decoder_load.py
22 passed in 174.49s

$ python3 -m pytest -q --ignore=tests/test_semantic_decoder_load.py --solver=scipy_slsqp
22 passed in 186.04s
```

**Performance.** Per-step wall clock is a wash — 0.345ms (SLSQP) vs 0.370ms (Clarabel) on
the narrow scene, 0.137ms vs 0.133ms on the wide. Clarabel is marginally slower per solve
but needed 670 steps where SLSQP needed 1478, so the *rollout* is ~2x faster end to end.
Caveat: `wall_clock_per_step` times the whole `filt.step()` — KD-tree query and cone
computation included — not the solve alone, so none of these numbers isolate the solver.
No solver-only timer exists; adding one to the free-form `diagnostics` dict is the natural
place if per-solve cost ever matters.

**New tests.** `tests/test_cbf_qp_backends.py` — five hand-solved QPs (slack constraint,
single binding CBF row, SOC-binding radial projection, stacked active rows, genuinely
infeasible) run against both backends via parametrization, plus a randomized
200-instance cross-check between them. The parametrization is also the guard against the
failure mode where Clarabel is silently never exercised. One tolerance is deliberately
looser: where the *curved* SOC boundary is the binding constraint, Clarabel lands ~1e-5
off the exact radial projection while SLSQP hits it to float32 precision — an
interior-point method approaches a nonlinear active constraint from the interior and
stops at its convergence tolerance. Documented as `SOC_TOL` rather than quietly widening
the global tolerance.

Root `conftest.py` gained a `--solver` option, chosen over an env var so a run's backend
is visible in the command line and can't be left set by accident between runs. Its
default is read off `CBFQPConfig` rather than hardcoded, so a no-flag run always tests
the library's actual default.

Two harness call sites (`eval_cbf_modes.py`, `costmap_cbf.py`) previously read
`cfg_dict.get("solver", "scipy_slsqp")`; that literal fallback would have silently pinned
SLSQP for any config omitting the key even after the default moved, so both now defer to
`CBFQPConfig`'s own default instead. `configs/cbf/room0_cbf.py` likewise no longer pins
`solver` explicitly — flagging that as a real behavior change to the documented
`costmap_cbf.py`/`eval_cbf_modes.py` commands, which now run Clarabel.

```
$ python3 -m pytest -q --ignore=tests/test_semantic_decoder_load.py
22 passed in 116.94s

$ python3 -m pytest -q --ignore=tests/test_semantic_decoder_load.py --solver=clarabel
1 failed, 21 passed in 123.78s
```

The single failure is the narrow k=1.0 `severity` assertion analyzed above — the first of
that test's four assertions to trip; `time_ratio` never evaluated because `severity`
precedes it, which is why the 11.91→5.90 shift needed a separate run to surface. Both
runs on `numpy 2.2.6` / `scipy 1.15.2` / `clarabel 0.11.1` / `pytest 6.2.5`.

## Real-data VLM safety-score consistency check (2026-07-18)

Answers the Gap Tracking item "Real Replica-derived hero-frame version of the VLM
consistency check" — the 2026-07-09 run (`vlm_consistency_check.py`) used stock photos
because no Replica data existed in this checkout; this re-runs the identical
methodology against real Replica RGB frames. Standalone Stage 2 work, does not touch
`src/cbf/`.

**Data acquisition.** `huggingface.co/datasets/3David14/GS3LAM-Replica` (the dataset
this repo's README points to) turned out to be a single non-gated 12.77GB zip with no
per-scene file to selectively download — downloading it whole would have violated the
"small subset, not the full dataset" instruction. The underlying CDN supports HTTP
Range requests (confirmed live: `206 Partial Content`), so a small in-memory
`HTTPRangeFile` (stdlib `zipfile.ZipFile` fed a `requests`-backed seekable object) was
used to read only the archive's central directory (4 requests, ~8.4MB, for a full
64,051-entry listing) and then extract individual frames on demand (~400KB and 4
requests per frame). Total data pulled for this whole experiment, listing plus all
frames plus final crops: under 15MB — nowhere near the full archive. No
`huggingface_hub` install or auth was needed (dataset is public). This technique isn't
committed anywhere (it lived in a scratch script for this session) — worth promoting to
a small repo utility if real Replica data gets pulled again before Stage 1 training
access resumes in the fall.

**Scene and objects.** Used `room0` only — a furnished waiting-room/lounge scene, not
the original stock set's more generic object types, so the object list was adapted
rather than force-matched (per instruction). Ten frames spread across the ~2000-frame
sequence (`frame000000` … `frame001800`, step 200) were pulled and inspected directly
(by eye) to find a real safety gradient; a second scene was not needed since `room0`
alone yielded ten clearly distinct objects across safe/mid/hazard categories. Each
object was rough-cropped with a single eyeballed `PIL.Image.crop()` bounding box from
whichever sampled frame showed it best — no masks, no convex-hull background
suppression, matching the original methodology's rigor level:

| # | Object | Source frame | Category |
|---|---|---|---|
| 01 | Plain wall segment | frame000000 | safe |
| 02 | Grey carpet floor | frame000200 | safe |
| 03 | Heavy wood credenza/sideboard | frame000000 | safe |
| 04 | Sofa back cushion | frame000600 | safe |
| 05 | Round tufted ottoman | frame000000 | safe |
| 06 | Upholstered wingback accent chair | frame000800 | mid |
| 07 | Round wood side table (thin metal legs) | frame000000 | mid |
| 08 | Ceramic vase (dried-flower arrangement) | frame001000 | hazard |
| 09 | Real power cable snaking across the rug | frame001200 | hazard |
| 10 | Wall-mounted multi-pane glass panel | frame000000 | hazard |

Object 02 (carpet floor) was added beyond the original ten-object list deliberately: it
is the direct control the original finding's own explanation calls for — if the
prompt's "flat solid ground" anchor language is what made the *wall* ambiguous, a
literal floor patch should be the cleanest possible positive case to contrast it
against. Object 09 (cable) is a real environmental object (a power cord crossing the
rug, visible identically across multiple sampled frames, not a rendering artifact),
not a staged prop — closer to what a real Stage 2 hero-frame crop would look like than
the original studio cable photo.

**Model/config: `gemini-3.5-flash`, matching current production**
(`vlm_safety_score.py:34`), not the original consistency-check script's
`gemini-flash-latest` rolling alias — that alias predates the 2026-07-12 production
pinning decision. Everything else held identical to the original run for a clean
comparison: `PROMPT` reused verbatim, `temperature=0.2`, `max_output_tokens=1024`,
`thinking_config=ThinkingConfig(thinking_budget=0)`, 5 queries/object. A 1-call smoke
test confirmed the model/config path before the full run. New script:
`vlm_consistency_check_real.py` (sibling to the original, which is untouched — both
image sets and both result files now coexist for direct comparison). 50/50 calls
succeeded on the first attempt, no retries triggered.

**Results:**

| Object | Scores | Mean | StdDev |
|---|---|---|---|
| 01 wall (real) | 1.00 ×5 | 1.000 | 0.000 |
| 02 carpet floor (real) | 0.95 ×5 | 0.950 | 0.000 |
| 03 credenza, heavy wood | 0.00 ×5 | 0.000 | 0.000 |
| 04 sofa cushion | 0.15 ×5 | 0.150 | 0.000 |
| 05 ottoman | 0.10, 0.15, 0.10, 0.10, 0.10 | 0.110 | 0.022 |
| 06 accent chair | 0.10, 0.00, 0.10, 0.10, 0.10 | 0.080 | 0.045 |
| 07 wood side table | 0.00 ×5 | 0.000 | 0.000 |
| 08 ceramic vase | 0.10 ×5 | 0.100 | 0.000 |
| 09 cable | 0.15 ×5 | 0.150 | 0.000 |
| 10 glass panel | 0.00 ×5 | 0.000 | 0.000 |

Full raw data: `assets/vlm_consistency_real/results.json` (original stock-photo data
unchanged at `assets/vlm_consistency/results.json`).

**Comparison against the 2026-07-09 stock-photo run — does the finding replicate?**
Partially, and with an important reversal. The *shape* of the original finding
(overwhelming stability with a small minority of exceptions) does replicate: 8/10
objects were perfectly stable here too. But the *specific* finding does not — the wall,
the one unstable object in the original run (bimodal 0.0/1.0, mean 0.4, stdev 0.55),
was the single most confidently stable object in this run (1.000, stdev 0.000, tied for
the highest score alongside the floor). Instability instead showed up on two different,
previously-rock-solid categories (ottoman and accent chair — soft/mid-tier furniture,
the same category the original run's `chair_wooden_mid`/`pottedplant_mid` scored with
zero variance), and at an order of magnitude smaller scale: single-query jitter of
±0.05 within a tight cluster (stdev 0.02–0.04), not a full swing between the scale's two
endpoints. No object in this run showed wall-like bimodal instability.

**What this means.** The original wall-instability finding does not look like a general
property of "wall" as a semantic category under this prompt — a different real wall
photo, in situ with ambient room context rather than isolated on a studio background,
scored a clean, unanimous 1.0 across all 5 runs, alongside a literal floor patch (0.95,
also zero variance) that was added specifically as the "flat solid ground" positive
control the original hypothesis implied. That undercuts the original
"ground-anchored-language" explanation as a *general* mechanism — it may instead have
been specific to that one stock photo (its framing, lighting, or how "ground-like" that
particular wall read), not a property that reliably reproduces across different wall
images. Separately, the small jitter that did appear (ottoman, accent chair) is small
enough that it's hard to distinguish from ordinary run-to-run sampling noise at
temperature 0.2 rather than a second genuine bimodal-instability case — worth
re-testing at a higher temperature before treating it as a finding in its own right (the
original run's own temperature-0.2-suppresses-variance caveat still applies here
unchanged).

One more real-data-only observation, consistent with (not contradicting) the original
run: non-drivable solid furniture scored low regardless of how heavy/stable it actually
is (credenza and side table both 0.000, matching the original's `bookshelf_heavy_safe`
scoring 0.000) — the prompt's 1.0 anchor is "flat solid ground," not "stable object," so
solid furniture that isn't literally drivable ground reliably gets pushed toward 0
rather than scored as safely-avoidable-but-solid. This is a scale-calibration property
of the prompt, not new instability, and was already implicitly present in the original
run's bookshelf result — real data just reconfirms it on a second, independent heavy
object.

## SemanticDecoder CPU-construction fix (2026-07-18)

Fixes the hardcoded-`.cuda()` finding flagged above ("New finding not in the original
two bugs") — a device-flexibility fix, not a change to the classifier-loading logic,
which is untouched and remains separately verified as before.

**The change** (`src/Decoder.py`, `SemanticDecoder.__init__` only —
`SemanticDecoder_MLP` in the same file still hardcodes `.cuda()` on its own `fc4` layer,
untouched, out of scope for this fix): added an optional `device` parameter, default
`None`. When `None`, resolves to `"cuda" if torch.cuda.is_available() else "cpu"`;
`.cuda()` replaced with `.to(device)`. Every existing call site
(`src/GS3LAM.py:103`, `visualizer/{export_mesh,offline_recon,online_recon}.py`,
`vlm_safety_score.py`, `tests/test_semantic_decoder_load.py`) constructs with only the
two positional args, no `device` override, so all of them go through the auto-detect
path. On the real A2000 dev box `torch.cuda.is_available()` is expected to be `True`
(per `CLAUDE.md`'s pinned CUDA env), which resolves to `device="cuda"` — the same place
the old hardcoded `.cuda()` landed. This was **not** dynamically confirmed on that real
GPU box (none available this session, same constraint as everything else this summer)
— it's confirmed by inspection (no call site passes a conflicting `device`, and
`.to("cuda")` vs `.cuda()` are documented equivalent moves), not by an actual run on
the hardware. Flagging that distinction rather than rounding it up.

**Verification actually performed, live, not just by inspection:** this sandbox has no
GPU and previously had no torch install at all. A CPU-only torch wheel
(`torch==2.13.0+cpu`, ~175MB, `--index-url https://download.pytorch.org/whl/cpu`) was
installed into this plain-system Python — **not** the pinned `cudatoolkit-dev=11.7.0`
conda env `CLAUDE.md` protects, which was not touched. With that installed:
- `tests/test_semantic_decoder_load.py`, previously requiring `--ignore` because
  `SemanticDecoder.__init__`'s hardcoded `.cuda()` made it uncollectable/failing on any
  CPU-only machine, now **runs and passes directly**, no `--ignore` flag:
  `python3 -m pytest tests/test_semantic_decoder_load.py -v` → 1 passed. This
  constructs `SemanticDecoder(16, 256)` twice through the new default (auto-detect →
  `"cpu"` here), and does a real `state_dict()` round-trip with `strict=True`, asserting
  no missing/unexpected keys — genuine dynamic execution of the class, not a dry read.
- Full suite re-run for regressions: `python3 -m pytest -q` → **23 passed** (up from the
  prior 22 — this test is no longer excluded), nothing else broke.

**What this does not verify:** the real-hardware `cuda` branch itself (see above,
inspection only, not run), and — same caveat as the original classifier-loading fix —
there is still no real `classifier.pth` anywhere on any machine checked this summer, so
this remains untested against real trained weights regardless of device.

**One more thing worth a decision, not fixed silently:**
`tests/test_semantic_decoder_load.py`'s own docstring (lines 9-12) still says the test
was "NOT executed as of this writing... cannot be constructed here," which is now
stale — it just ran and passed. Left as-is since the instruction for this fix was
scoped to `src/Decoder.py` and this file; flagging rather than editing a file outside
that scope.

## Stage 2 fail-safe default corrected + top-5 cap noted (2026-07-20)

Two issues surfaced while documenting `vlm_safety_score.py` for the paper's Method
section. Both trace to the original first-pass commit (`b243f0a`, 2026-04-20) and had
survived every later commit untouched.

**Fixed — inverted fail-safe default (safety-score-scale decision, flagged per
CLAUDE.md Rules).** `vlm_safety_score.py`'s `__main__` wrapped `extract_canonical_view`
+ `query_vlm_safety` in a bare `except Exception` that set `safety_dictionary[obj_id] =
1.0` ("Default to safe if it fails"). Because `broadcast_scores_and_save` initializes
`safety_array = np.zeros(...)` and paints only queried objects, this produced an
inverted fail-safe:

- Unqueried splats defaulted to `0.0` (most conservative / "lethal") — the documented
  convention, surfaced by `ply_io.py`'s `ambiguous_zero_mask` / `ZeroSafetyPolicy`.
- A queried object whose call *failed* (projection error, API failure, or JSON parse
  error) defaulted to `1.0` (most permissive / "completely safe") — telling the CBF to
  drive over an object we had no valid judgment for.

An object important enough to be audited defaulted, on failure, to the most permissive
score, while an object never looked at defaulted to the most conservative one — the two
"no valid score" paths pointed in opposite directions, and the dangerous one applied to
exactly the objects the system had chosen to scrutinize. Changed the failure default
from `1.0` to `0.0` so "no valid score" is uniformly conservative. `f_min` in
`semantic_weighting.alpha_gain_per_splat` keeps the barrier active (not zeroed) at
safety 0.0, so this is consistent with the existing zero-safety handling. Rationale
recorded here rather than fixed silently, per CLAUDE.md's rule that safety-score-scale
changes are research decisions. Secondary, not changed: the bare `except Exception`
conflates geometry-projection failures with VLM/API failures; worth narrowing later,
but it's orthogonal to the default-value inversion.

**Noted, not changed — top-5 object cap is scaffolding.** The same `__main__` audits
only the five largest object classes (`np.argsort(-counts)[:5]`, comment "Filtering
noisy data..."). `git blame` confirms it is first-pass demo-driver code, lives only in
`__main__` (every library function is per-object with no cap), and contradicts the
"one query per object" method described in `ARCHITECTURE.md` §2.2 and the paper. Left
in place for now — it's not a recorded scope decision, and the fix (drop the cap, or
make it an explicit CLI arg) belongs with a real Stage 1/2 run, not this doc pass. The
Method draft describes the intended per-object behavior and excludes the cap.

## ARCHITECTURE.md §5: Gaussian World Model future-work bullet rewritten (2026-07-26)

Written 2026-07-26, committed 2026-08-02 — it sat uncommitted in the working tree for a
week and was held back from the 2026-07-20 doc commit pending a scope check, since
`CLAUDE.md` treats `ARCHITECTURE.md` as the design source of truth and forbids pulling
world-model components into current scope without being asked. Confirmed in scope as
*future work only*: §5 is explicitly the "not current scope" section, and the entry
states its own out-of-scope status. (Dating note: the working-tree copy under
`research/GS3LAM/` — a partial snapshot taken 2026-07-20 14:46 — contains the λ/γ and
AlphaAdj edits but *not* this one, which brackets the edit to after that snapshot; the
file mtime put it at 2026-07-26 22:22.)

**What it replaced.** A three-line bullet:

> - **Predictive/anticipatory safety** — a Gaussian World Model that forecasts future
>   scene states so the CBF can react before a hazard materializes, rather than scoring
>   only the current frame. GPU/VRAM cost is currently prohibitive on target hardware.

**What replaced it.** A ~17-line entry proposing a specific mechanism rather than a
generic pointer: GWM (Lu et al., ICCV '25) already generates future Gaussian splats via
a VAE plus an action-conditioned diffusion transformer, but purely geometrically with no
notion of hazard. The proposal is to add a safety channel to that generative process —
every `safety_gsplat.ply` this pipeline produces is already a (scene, VLM-judgment)
pair, i.e. exactly the supervision needed to fine-tune a world model's generated splats
to carry a predicted safety attribute. That is the same supervision pattern GS3LAM
itself uses to train reconstructive splats against DEVA's 2D labels, applied one level
up to a predictive model instead of a reconstructive one. The predicted,
safety-annotated future splats would feed the existing collision-cone CBF unchanged,
making the controller anticipatory rather than purely reactive.

**Why the change.** Two things. The old bullet named a direction without saying how it
would be built, which made it unactionable. And its stated blocker — "GPU/VRAM cost is
currently prohibitive on target hardware" — was the wrong obstacle: the binding
constraint on this direction is the compute and training data needed to fine-tune a
generative model at GWM's scale, not inference VRAM on the A2000. The rewrite says that
instead. No claim in it is validated; it is a recorded direction, not a result.

**Caveat, flagged not fixed: the citation is unverified in this checkout.** "Lu et al.,
ICCV '25", the VAE + diffusion-transformer architecture, and the action-conditioning
claim were all transcribed into `ARCHITECTURE.md` without being checked against the
actual paper here, and there is no offline copy in the repo to check against. This is
worth naming explicitly because this repo has already been burned by exactly this
failure mode once: the GS3LAM citation carried a fabricated author list ("Li, M., Liu,
S., Zhou, H.") across multiple sessions before being corrected in `6301e28` (see the
"Unrelated fix bundled into this same commit" note under the 2026-07-11 COV_INFLATE
entry). Verify the GWM authors, venue, and architecture description against the real
paper before any of this reaches the write-up.

## Pre-registered head-on scene: first real run, both backends (2026-08-02)

`scenes/prereg_headon.py` recreated and executed. This is the first time a
pre-registered head-on scene has actually been run in this checkout — the recap
on 2026-08-02 confirmed no `scenes/prereg_headon.py` and no `results_draft_v1`
existed anywhere here, so every earlier reference to this scene's outcome was to
an unpersisted run.

The scene's geometry is derived from a criterion fixed before the rollout (blocking
hazard + a robot-center lateral corridor set to half the robot footprint radius);
`R_collar` is solved from that criterion rather than chosen. Scene generated and
evaluated exactly once, no outcome-driven retuning.

Command: `python3 scenes/prereg_headon.py`
Env: system Python 3.10.12, `numpy 2.2.6`, `scipy 1.15.2`, `clarabel 0.11.1`.
The generated `scenes/prereg_headon_seed42.ply` is left untracked, per the existing
convention that generated scene `.ply` files are ephemeral (the script regenerates
it deterministically from `seed=42`).

Console output, verbatim:

```
=== PRE-REGISTERED CRITERION -> DERIVED PARAMETERS ===
c_base = sqrt(chi2.ppf(0.99,3))      = 3.36821
robot radius rho                     = 0.160 m
hazard scale s_h / clutter s_c       = 0.1 / 0.04
r_phys_hazard = c_base*s_h + rho     = 0.4968 m  (inflated blocking radius)
path offset from hazard center       = 0.030 m  (6.04% of r_phys_hazard)
  -> straight-line penetration depth = 0.4668 m  (BLOCKING condition met)
pre-registered clearance W_center    = rho/2 = 0.080 m  (50% of robot radius)
  -> DERIVED collar radius R_collar  = 0.8715 m
  (check) W_center back-computed      = 0.0800 m

wrote scene -> scenes/prereg_headon_seed42.ply  (collar_radius=0.8715, seed=42, offset=0.03)

      solver         mode reached  plen_ratio  time_ratio  severity near_miss infeas
------------------------------------------------------------------------------------
    clarabel         NONE    True      1.0003      1.1418   -0.1330         1      0
    clarabel  ALPHA_SCALE    True      0.9999      1.0597   -0.4251         1      0
    clarabel  COV_INFLATE   False      0.4416     14.9254    0.7573         0      0

 scipy_slsqp         NONE    True      1.0003      1.1418   -0.1330         1      0
 scipy_slsqp  ALPHA_SCALE    True      0.9999      1.0597   -0.4251         1      0
 scipy_slsqp  COV_INFLATE   False      0.4299     14.9254    0.8961         0      0

=== backend agreement (clarabel vs scipy_slsqp) ===
          NONE: |dtime_ratio|=0.00e+00  |dplen_ratio|=9.54e-08  reached_match=True
   ALPHA_SCALE: |dtime_ratio|=0.00e+00  |dplen_ratio|=0.00e+00  reached_match=True
   COV_INFLATE: |dtime_ratio|=0.00e+00  |dplen_ratio|=1.17e-02  reached_match=True
```

### What the numbers say (observations only — no claim rewritten yet)

**`ALPHA_SCALE` is faster than `NONE` here, and cuts deeper.** time_ratio 1.0597 vs
1.1418, severity -0.4251 vs -0.1330. Both reach the goal, both on an essentially
straight path (plen_ratio 0.9999 vs 1.0003 — no lateral departure by either), and both
log one near-miss. So on this scene the semantically-weighted mode gets there *sooner*
while penetrating the true confidence ellipsoid *further* than the pure-geometric
baseline. Severity is the shared true-geometry signed distance in Mahalanobis units, so
the two are graded against the same boundary.

This is the counterintuitive faster-but-worse-severity pattern, now reproduced on a
real, persisted, pre-registered scene rather than recalled from an unpersisted run.
Note it is *not* what the Step 0 narrow-corridor table (2026-07-11) showed: there,
`ALPHA_SCALE` and `NONE` had an identical time_ratio of 1.06. What carries over from
that table is `ALPHA_SCALE`'s severity alone (-0.425 there, -0.4251 here); the severity
*gap* does not -- Step 0's `NONE` severity was -0.064, giving a gap of 0.361 there
against 0.292 here. The time ratios, identical at Step 0, have separated here, with
`ALPHA_SCALE` the faster of the two.

> **Correction (2026-08-03) — this was an error in the original entry.** This paragraph
> first read "Here the severity gap is the same (-0.425 vs -0.133)". The two scenes'
> severity gaps are not the same, and -0.133 is *this* scene's own `NONE` severity, not
> Step 0's. Step 0's `NONE` severity was -0.064 (table at the `COV_INFLATE`
> disambiguation Step 0). Gaps: 0.361 at Step 0, 0.292 here. Only `ALPHA_SCALE`'s
> severity coincides across the two scenes; the equal-gap claim was wrong and the
> sentence above replaces it. No other number in this entry is affected.

**Bearing on `ARCHITECTURE.md:101.** That line currently reads "`ALPHA_SCALE` vs `NONE`
behaved exactly as predicted (same path, slower approach)". Against this run: "same
path" holds (plen ratios within 4e-4 of each other and of 1.0); "slower approach" does
not — `ALPHA_SCALE` is faster than `NONE`, which is the opposite of the design
prediction that a lower safety score should brake harder along the approach axis.
**Deliberately not edited in this session.** Whether that line gets kept, rewritten, or
replaced with an explicit "still needs investigation" note is a decision for the next
session, and it should be made after the constraint-activation trace-through, which was
also deliberately not attempted here.

**The pre-registered criterion did its job, partly.** Criterion (2) predicted that a
corridor narrower than the robot's own radius would force "a real deceleration and/or a
real lateral departure." `NONE` decelerated (1.14) without departing laterally.
`ALPHA_SCALE` did neither — it neither slowed relative to `NONE` nor routed around,
it went through. So the scene is materially harder than the earlier too-mild attempt
(both modes at 1.06 / ~1.00), and it does separate the modes, but not in the predicted
direction.

**`COV_INFLATE` fails on this scene.** `reached_goal=False` under both backends, with
plen_ratio 0.44 and time_ratio 14.93 — 14.93 is the max_steps ceiling
(2000 steps x 0.05 s = 100 s over a 6.70 s oracle), so it exhausted the step budget
partway along the corridor rather than converging slowly. `infeasible_count=0`, so the
QP never reported infeasibility or took the `_max_braking()` branch; consistent with
the 2026-07-11 finding that the crawl is a gain artifact, not a feasibility failure.
Its severity is the best of the three (+0.76 / +0.90) — it stays well clear, and never
arrives. Whether the `W_center = rho/2` corridor is simply too tight for gamma=1.0
inflation is untested; no gain or gamma sweep was run here.

**Backend agreement — clean on the two modes that matter, not clean on `COV_INFLATE`.**
`NONE` and `ALPHA_SCALE` agree to 0.00e+00 on time_ratio and 9.54e-08 / 0.00e+00 on
plen_ratio, so the headline comparison above is solver-independent. `COV_INFLATE`
disagrees materially: |dplen_ratio| = 1.17e-02, and severity differs 0.7573 (clarabel)
vs 0.8961 (scipy_slsqp) — roughly an 18% spread on a metric the other two modes
reproduce exactly. Both backends do agree it fails to reach the goal, and both hit the
same step ceiling. This is recorded, not chased; it is a divergence on a non-converging
run, so it is the least trustworthy row in the table and should not be quoted without
this caveat.

## Constraint-activation trace at the NONE/ALPHA_SCALE divergence (2026-08-02)

Follow-up to the pre-registered head-on run recorded above. Question under test, stated
as a hypothesis to be refuted rather than confirmed: *at the steps where ALPHA_SCALE's
severity is worst, is the hazard's own barrier the dominant active constraint, and does
the required corrective magnitude there scale with k_alpha (smaller gain -> smaller
required correction once h<0, not merely earlier activation while h>0)?*

Scene `scenes/prereg_headon_seed42.ply` (seed 42), config `configs/cbf/room0_cbf.py`,
`clarabel`. Trace driver was scratch and is not committed, per the ephemeral-driver
convention; it imports the committed helpers rather than reimplementing them, and every
number below was recomputed from the committed scene and config.

Safety field as actually assembled: 301 splats, exactly one at safety 0.1 (the hazard,
id 0, at `[0, 0.03, 0]`), 300 at 1.0. `k_alpha[hazard]` = 1.00000 under NONE, 0.10000
under ALPHA_SCALE; background 1.0 in both. `A` and `s_min` are byte-identical between
the two modes, as required for ALPHA_SCALE to be a pure gain change.

### Result: the hypothesis is REFUTED in its specific form

**At each mode's worst-severity step, the hazard barrier is not active at all.**

```
        NONE: hazard active steps 13..54 (42 steps); deactivates at step 55
              last active step 54: sev=-0.1186 speed=0.8493 h=-94.17 delta=+3.19
              deepest penetration step 55: dist=-0.1330 delta=-0.31 haz_active=False
 ALPHA_SCALE: hazard active steps 13..49 (37 steps); deactivates at step 50
              last active step 49: sev=-0.4164 speed=1.3692 h=-755.66 delta=+3.53
              deepest penetration step 50: dist=-0.4251 delta=-3.80 haz_active=False
```

The gate is `cone_exists = (h <= 0) & (delta >= 0)`. `delta = r^T A v` flips negative as
the robot passes the hazard, so the barrier releases *while the robot is still inside the
inflated ellipsoid* (dist -0.12 / -0.42 at release). Worst penetration then occurs on the
very next step, unconstrained. There is no dominant active constraint at the worst step to
scale with anything: ALPHA_SCALE's active set at step 50 is empty.

This is a property of the collision-cone formulation, not a bug: `h = beta*gamma - delta^2`
is a cone condition on the velocity ray, not a stay-outside-the-ellipsoid condition.

**Where the hazard IS dominant.** Over the active window it dominates overwhelmingly —
ALPHA_SCALE 37/37 constrained steps (100%), NONE 30/43 (69.8%), the remainder going to
collar splats 135/136. Dominant ids at the 20 worst-severity steps: `[0]` for
ALPHA_SCALE, `[0, 135, 136]` for NONE.

### The two sub-claims, tested separately

**"Not just earlier activation" — CONFIRMED, and structurally guaranteed.** The gate
contains no `k_alpha` term, and `A`/`s_min` are identical across modes, so the active set
at a given state is mode-independent by construction. Empirically both modes first
activate at step 13, from a bit-identical state.

**"Smaller gain -> smaller required correction" — CONFIRMED at a matched state, but
weakly and with a floor.** Trajectories are identical through step 13, so that state is a
clean matched-state comparison where `k_alpha` is the only variable:

```
        mode            h   k_haz  rhs=-k*h/2  |u_s-u_ref|                       u_safe
        NONE   -1039.0605   1.000    519.5303     1.276847    [-0.2304 -0.3411  0.    ]
 ALPHA_SCALE   -1039.0605   0.100     51.9530     0.994977    [ 0.0412 -0.2658  0.    ]
```

Sweeping only the hazard's gain at that same state (n_active=1, active id `[0]`,
h=-1039.0605):

```
    k_haz      rhs_haz  |u_safe-u_ref|  ratio_vs_k1   slack_haz
        1     519.5303        1.276847       1.0000    0.00e+00
      0.5     259.7651        1.120253       0.8774    0.00e+00
      0.2     103.9061        1.026296       0.8038    0.00e+00
      0.1      51.9530        0.994977       0.7792    3.81e-06
     0.05      25.9765        0.979318       0.7670   -5.72e-06
     0.02      10.3906        0.969922       0.7596    4.77e-06
     0.01       5.1953        0.966790       0.7572    2.38e-06
    0.001       0.5195        0.963972       0.7550    9.54e-07
```

`rhs` is exactly proportional to `k_alpha`, but the correction is strongly sublinear: a
1000x gain reduction buys only a 24.5% correction reduction. The floor is explained
exactly — as `k_alpha -> 0` the constraint tends to `w^T u >= 0`, and the distance from
`u_ref = [1,0,0]` to that halfspace is `-w.u_ref/||w|| = 0.963659`, against a measured
0.963972 at k=1e-3. `k_alpha` scales the constraint *offset*; it cannot rotate the
constraint *direction*, which is what sets the floor.

### What the approach phase actually looks like

```
 step | NONE speed   NONE dh  NONE rhs NONE act | ALPHA speed  ALPHA dh ALPHA rhs ALPHA act
   15 |     0.6503   17.1158    438.21     True |      0.6676   17.0999     49.66      True
   25 |     1.0916   12.7745    228.53     True |      1.0968   12.6961     43.15      True
   35 |     1.3739    6.6883    123.23     True |      1.4689    6.4499     40.48      True
   45 |     1.0666    1.7151     74.27     True |      1.4771    0.4734     38.47      True
   50 |     0.9398    0.3197     57.66     True |      1.3544   -0.4251     37.80     False
   55 |     0.8280   -0.1330     44.76    False |      1.2814    0.5948     22.54     False
```

(`dh` = signed distance to the hazard's inflated surface.) NONE begins shedding speed
around step 35 and arrives at the surface at 0.85 m/s; ALPHA_SCALE keeps accelerating to
~1.53 and arrives at 1.37 m/s. Essentially the entire severity gap is already present at
the moment the barrier releases (-0.1186 vs -0.4164, against final -0.1330 vs -0.4251),
so it accrues across the active window, not at any single binding step.

### Status

The specific hypothesis is refuted: there is no dominant active constraint at the
worst-severity steps to carry the mechanism. A plausible replacement account —
cumulative under-braking across a 37-step window, each step's correction reduced ~22% —
is *consistent* with the data but **not established**: nothing here demonstrates that
the per-step reduction integrates to the observed -0.29 gap, and ~~no counterfactual
(e.g. replaying NONE's control sequence under ALPHA_SCALE's gains, or sweeping
`k_alpha` end-to-end and checking severity monotonicity) has been run. Recorded as open.~~
**Partly done (2026-08-31) — see "Does the per-step correction reduction explain the
ALPHA_SCALE/NONE severity gap?" below.** The `k_alpha` end-to-end sweep was run (severity
is smooth and monotonic in gain, no threshold), and the matched-state re-solve was extended
across the whole window. Short answer: the per-step reduction is real but **not** the stable
~22% assumed here (it ranges 2.19%-30.29% / 5.41%-89.82%), and 83% of the severity gap
occurs at constant approach speed, so accumulated under-braking is not a sufficient
explanation. Replaying NONE's control *sequence* under ALPHA_SCALE's gains is still not run.
`ARCHITECTURE.md` states the numbers and flags the mechanism as unsettled; no mechanism
has been written into it.

## Does the per-step correction reduction explain the ALPHA_SCALE/NONE severity gap? (2026-08-31)

Follow-up to "Constraint-activation trace at the NONE/ALPHA_SCALE divergence" above,
testing the replacement account that section left open: *is a per-step corrective-magnitude
reduction, accumulated across the active window, a sufficient explanation for the 0.292
severity gap?* Two experiments — a per-step matched-state re-solve across the whole window
(A), and an end-to-end gain sweep between the two modes' effective gains (B). These are the
two counterfactuals `### Status` above named as unrun. No library math was touched:
`collision_cone.py`, `qp_filter.py` and `ellipsoid.py` are unchanged, as is
`scenes/prereg_headon_seed42.ply` (checksum-verified before and after).

**Departure from the ephemeral-driver convention, deliberately.** Both drivers are
committed, as `scripts/exp_a_matched_state_window.py` and `scripts/exp_b_gain_sweep.py`,
rather than discarded as scratch. The reason is directly above: the 2026-08-02 trace driver
and the earlier state-by-state re-solve driver were both discarded, and their numbers now
survive only as prose — which is exactly why the 30/43 figure corrected below could not be
re-checked without rewriting the driver from scratch. Scene generation stays ephemeral;
measurement drivers that produce recorded numbers no longer do.

Env: system Python 3.10.12, `numpy 2.2.6`, `scipy 1.15.2`, `clarabel 0.11.1`. Scene
`scenes/prereg_headon_seed42.ply` (seed 42), config `configs/cbf/room0_cbf.py`,
`clarabel` throughout. `k_alpha_base=1.0` and `alpha_f` is identity, so under
`ALPHA_SCALE` the hazard's effective gain *is* its safety value: 0.1 vs NONE's 1.0.

### Harness cross-checks (both pass, exactly)

Step 13 reproduces the committed matched-state numbers to all printed digits, and both
activation windows reproduce exactly:

```
  states identical at step 13: True
  |u-uref| NONE=1.276847 [1.276847]  ALPHA_SCALE=0.994977 [0.994977]  ratio=0.7792 [0.7792]
  CROSS-CHECK PASS

        NONE: hazard active steps 13..54 (42 steps); deactivates at step 55
 ALPHA_SCALE: hazard active steps 13..49 (37 steps); deactivates at step 50
```

Experiment B's endpoints reproduce the committed 2026-08-02 rollout table exactly —
`ALPHA_SCALE` at gain 1.00 is bit-identical to the `NONE` row, as it must be since every
gain then equals `k_alpha_base`:

```
  gain 1.00 vs         NONE: severity -0.1330 [-0.133] time_ratio 1.1418 [1.1418] plen 1.0003 [1.0003]  -> PASS
  gain 0.10 vs  ALPHA_SCALE: severity -0.4251 [-0.4251] time_ratio 1.0597 [1.0597] plen 0.9999 [0.9999]  -> PASS
  ALPHA_SCALE@1.0 identical to NONE row: True
```

### Experiment A — the per-step reduction is NOT stable across the window

For every step from 13 through each mode's deactivation, at that mode's own recorded
`(p, v)`, the QP was re-solved with the other mode's gain. Nothing is forward-integrated.
Ratio is always (low-gain correction)/(high-gain correction), so it is comparable across
both families.

At **NONE's** recorded states (re-solved under ALPHA_SCALE's gain):

```
 step    speed  dist_haz        min_h  nact  haz  |u_own-uref|  |u_oth-uref|    ratio   reduc%
   13   0.6500   17.7588   -1039.0605     1 True      1.276847      0.994977   0.7792    22.08
   14   0.6387   17.4397    -973.4023     1 True      1.028206      0.809295   0.7871    21.29
   15   0.6503   17.1158    -876.4141     1 True      0.659388      0.534116   0.8100    19.00
   16   0.6863   16.7754    -799.8535     1 True      0.515414      0.431122   0.8365    16.35
   17   0.7289   16.4152    -741.8066     1 True      0.459611      0.394721   0.8588    14.12
   18   0.7734   16.0342    -692.9336     1 True      0.434733      0.381523   0.8776    12.24
   19   0.8188   15.6319    -649.8418     1 True      0.424761      0.379519   0.8935    10.65
   20   0.8646   15.2083    -610.9316     1 True      0.423623      0.384229   0.9070     9.30
   21   0.9104   14.7634    -575.3262     1 True      0.428578      0.393682   0.9186     8.14
   22   0.9561   14.2972    -542.4805     1 True      0.438276      0.406949   0.9285     7.15
   23   1.0016   13.8102    -511.9922     1 True      0.452041      0.423613   0.9371     6.29
   24   1.0468   13.3025    -483.5938     1 True      0.469580      0.443545   0.9446     5.54
   25   1.0916   12.7745    -457.0508     1 True      0.490826      0.466785   0.9510     4.90
   26   1.1357   12.2269    -432.1875     1 True      0.515853      0.493490   0.9566     4.34
   27   1.1792   11.6601    -408.8750     1 True      0.544832      0.523884   0.9616     3.84
   28   1.2218   11.0751    -386.9727     1 True      0.577958      0.558202   0.9658     3.42
   29   1.2633   10.4726    -366.3828     1 True      0.615394      0.596637   0.9695     3.05
   30   1.3034    9.8540    -347.0039     2 True      1.230802      0.892114   0.7248    27.52
   31   1.3043    9.2383    -305.1035     1 True      0.677808      0.661367   0.9757     2.43
   32   1.3419    8.6088    -289.0840     1 True      0.724967      0.709081   0.9781     2.19
   33   1.3772    7.9673    -685.2500     2 True      0.787793      0.764612   0.9706     2.94
   34   1.4091    7.3167    -600.7500     3 True      1.533629      1.069123   0.6971    30.29
   35   1.3739    6.6883    -570.7500     3 True      1.518980      1.066848   0.7023    29.77
   36   1.3396    6.0824    -542.5000     3 True      1.504763      1.064633   0.7075    29.25
   37   1.3061    5.4990    -515.8750     3 True      1.490905      1.062443   0.7126    28.74
   38   1.2734    4.9387    -490.0000     3 True      1.477453      1.060287   0.7176    28.24
   39   1.2416    4.4018    -465.8750     3 True      1.464377      1.058172   0.7226    27.74
   40   1.2105    3.8888    -443.3125     3 True      1.451677      1.056087   0.7275    27.25
   41   1.1803    3.4006    -421.1875     3 True      1.439312      1.054014   0.7323    26.77
   42   1.1508    2.9379    -400.3125     3 True      1.427296      1.051963   0.7370    26.30
   43   1.1220    2.5019    -380.6250     3 True      1.415623      1.049936   0.7417    25.83
   44   1.0939    2.0939    -361.8125     3 True      1.404271      1.047919   0.7462    25.38
   45   1.0666    1.7151    -344.0938     3 True      1.393238      1.045915   0.7507    24.93
   46   1.0399    1.3673    -326.9062     3 True      1.382508      1.043909   0.7551    24.49
   47   1.0139    1.0522    -310.9062     3 True      1.372082      1.041906   0.7594    24.06
   48   0.9886    0.7714    -295.5312     3 True      1.361944      1.039894   0.7635    23.65
   49   0.9639    0.5267    -280.9375     3 True      1.352090      1.037866   0.7676    23.24
   50   0.9398    0.3197    -267.0938     3 True      1.342512      1.035816   0.7716    22.84
   51   0.9163    0.1513    -253.8594     3 True      1.333199      1.033731   0.7754    22.46
   52   0.8934    0.0223    -241.3594     3 True      1.324148      1.031601   0.7791    22.09
   53   0.8710   -0.0673    -229.4062     3 True      1.315348      1.029411   0.7826    21.74
   54   0.8493   -0.1186    -218.0781     3 True      1.306795      1.027145   0.7860    21.40
  n=42  reduction%: mean=17.70 median=21.91 min=2.19 max=30.29
  ratio: first(step 13)=0.7792  last(step 54)=0.7860  std=0.0960
```

At **ALPHA_SCALE's** recorded states (re-solved under NONE's gain):

```
 step    speed  dist_haz        min_h  nact  haz  |u_own-uref|  |u_oth-uref|    ratio   reduc%
   13   0.6500   17.7588   -1039.0605     1 True      0.994977      1.276847   0.7792    22.08
   14   0.6522   17.4329   -1025.1582     1 True      0.858499      1.097943   0.7819    21.81
   15   0.6676   17.0999    -993.2617     1 True      0.629401      0.793869   0.7928    20.72
   16   0.7005   16.7515    -962.5371     1 True      0.501103      0.618994   0.8095    19.05
   17   0.7410   16.3840    -941.6113     1 True      0.447145      0.541519   0.8257    17.43
   18   0.7841   15.9962    -926.0391     1 True      0.423460      0.503928   0.8403    15.97
   19   0.8283   15.5875    -913.4590     1 True      0.414588      0.485827   0.8534    14.66
   20   0.8732   15.1578    -902.7715     1 True      0.414458      0.479122   0.8650    13.50
   21   0.9182   14.7069    -893.3516     1 True      0.420320      0.480086   0.8755    12.45
   22   0.9632   14.2351    -884.8574     1 True      0.430833      0.486849   0.8849    11.51
   23   1.0081   13.7425    -877.0625     1 True      0.445343      0.498444   0.8935    10.65
   24   1.0526   13.2293    -869.8125     1 True      0.463575      0.514400   0.9012     9.88
   25   1.0968   12.6961    -863.0156     1 True      0.485485      0.534547   0.9082     9.18
   26   1.1404   12.1432    -856.5820     1 True      0.511170      0.558899   0.9146     8.54
   27   1.1834   11.5713    -850.4727     1 True      0.540819      0.587590   0.9204     7.96
   28   1.2254   10.9811    -844.6289     1 True      0.574651      0.620803   0.9257     7.43
   29   1.2664   10.3736    -839.0391     1 True      0.612843      0.658700   0.9304     6.96
   30   1.3059    9.7499    -833.6582     1 True      0.655424      0.701305   0.9346     6.54
   31   1.3438    9.1115    -828.4941     1 True      0.702123      0.748359   0.9382     6.18
   32   1.3795    8.4599    -823.5156     1 True      0.752167      0.799110   0.9413     5.87
   33   1.4126    7.7974    -818.7227     1 True      0.804066      0.852109   0.9436     5.64
   34   1.4426    7.1263    -814.0996     1 True      0.855487      0.905078   0.9452     5.48
   35   1.4689    6.4499    -809.6348     1 True      0.903376      0.955043   0.9459     5.41
   36   1.4908    5.7715    -805.3066     1 True      0.944454      0.998827   0.9456     5.44
   37   1.5080    5.0953    -801.1055     1 True      0.976025      1.033874   0.9440     5.60
   38   1.5202    4.4261    -797.0059     1 True      0.981586      1.043868   0.9403     5.97
   39   1.5271    3.7690    -792.9854     1 True      0.976395      1.044324   0.9350     6.50
   40   1.5292    3.1299    -789.0107     1 True      0.969955      1.045099   0.9281     7.19
   41   1.5267    2.5154    -785.0679     1 True      0.961740      1.046192   0.9193     8.07
   42   1.5201    1.9331    -781.1470     1 True      0.950943      1.047614   0.9077     9.23
   43   1.5095    1.3916    -777.2295     1 True      0.936234      1.049355   0.8922    10.78
   44   1.4952    0.9013    -773.3123     1 True      0.915233      1.051274   0.8706    12.94
   45   1.4771    0.4734    -769.3899     1 True      0.883240      1.052710   0.8390    16.10
   46   1.4553    0.1197    -765.4684     1 True      0.829805      1.050932   0.7896    21.04
   47   1.4294   -0.1491    -761.5934     1 True      0.728422      1.034354   0.7042    29.58
   48   1.3995   -0.3270    -757.9868     1 True      0.508319      0.952470   0.5337    46.63
   49   1.3692   -0.4164    -755.6561     1 True      0.068517      0.672817   0.1018    89.82
  n=37  reduction%: mean=14.59 median=9.88 min=5.41 max=89.82
  ratio: first(step 13)=0.7792  last(step 49)=0.1018  std=0.1498
```

**The ~22% at step 13 is not representative of the window.** It is close to the endpoints
of the NONE family (22.08% at step 13, 21.40% at step 54) but that is a coincidence of
where the window starts and stops, not a plateau: in between the reduction collapses to
**2.19%** (step 32) and rises to **30.29%** (step 34). The mean over the NONE family is
17.70%, over the ALPHA_SCALE family 14.59%. Neither family is flat, and the two do not
even have the same shape — the ALPHA_SCALE family decays to ~5.4% mid-window and then
runs away to **89.82%** at step 49, the last step before release.

**The discontinuities are active-set composition changes, not gain effects.** The jumps at
NONE's steps 30/33/34 line up exactly with `nact` going 1 -> 2 -> 3. Recording the active
sets over each rollout:

```
        NONE: constrained=43 hazard_active=42
              active-set compositions: (0,135,136) x21, (0,) x19, (0,136) x2, (135,136) x1
 ALPHA_SCALE: constrained=37 hazard_active=37
              active-set compositions: (0,) x37
```

NONE recruits the collar splats 135/136 on 23 of its 43 constrained steps; ALPHA_SCALE's
active set is the bare hazard `(0,)` on all 37. That is a structural difference in *which
constraints exist*, reached by trajectory divergence — not something a gain multiplier
does directly, and not visible in a single matched-state check.

> **Correction (2026-08-31) — one figure in the 2026-08-02 trace above does not reproduce.**
> That section reports hazard dominance as "NONE 30/43 (69.8%)". Re-measured here two
> independent ways — `argmin(h)` over the active set, and comparing the hazard's own `h`
> against the filter's reported `min_h` — it is **20/43 (46.5%)**. Both methods agree, and
> there is no opacity pruning on this scene (301 splats loaded, 301 after pruning), so the
> original-index mapping in `active_splat_ids` is exact. ALPHA_SCALE's 37/37 (100%)
> reproduces exactly. The original driver was not committed, so the original figure's exact
> definition cannot be recovered to determine whether this is a genuine error or a different
> definition of "dominant"; recorded as a discrepancy rather than asserted as a bug. The
> numbers in that section are left unedited. This is the concrete reason the drivers for
> *this* section are committed.

### Experiment B — end-to-end gain sweep, five intermediate gains

Full rollout at each hazard gain, same scene/start/goal/solver, one shared oracle
(`oracle_time=6.7000s`) and the shared true-geometry grading filter. `collar_steps` counts
steps whose active set contains a non-hazard splat; `haz_window` is the hazard-active
window.

```
        mode   k_haz  reached   severity  time_ratio  plen_ratio  near_miss  infeas   haz_window  collar_steps
        NONE    1.00     True    -0.1330      1.1418      1.0003          1       0  13..54 (42)            24
 ALPHA_SCALE    0.10     True    -0.4251      1.0597      0.9999          1       0  13..49 (37)             0
 ALPHA_SCALE    0.20     True    -0.3424      1.0597      1.0008          1       0  13..49 (37)             0
 ALPHA_SCALE    0.35     True    -0.2561      1.0597      1.0014          1       0  13..49 (37)             8
 ALPHA_SCALE    0.50     True    -0.1836      1.0672      1.0033          1       0  13..49 (37)            13
 ALPHA_SCALE    0.70     True    -0.1529      1.0821      1.0017          1       0  13..50 (38)            17
 ALPHA_SCALE    0.85     True    -0.1398      1.1045      1.0005          1       0  13..52 (40)            20
 ALPHA_SCALE    1.00     True    -0.1330      1.1418      1.0003          1       0  13..54 (42)            24
```

Monotonicity, gain increasing 0.1 -> 1.0:

| quantity | values | verdict |
|---|---|---|
| `collision_severity` | -0.4251, -0.3424, -0.2561, -0.1836, -0.1529, -0.1398, -0.1330 | **monotonic increasing** (strongly saturating) |
| `time_ratio` | 1.0597, 1.0597, 1.0597, 1.0672, 1.0821, 1.1045, 1.1418 | monotonic increasing, **flat over 0.1-0.35** |
| `path_length_ratio` | 0.9999, 1.0008, 1.0014, 1.0033, 1.0017, 1.0005, 1.0003 | non-monotonic, peak at 0.5 — but total spread 3.4e-3, negligible |

Step-to-step severity deltas: `+0.0827, +0.0863, +0.0725, +0.0307, +0.0132, +0.0067`.

So, unlike the earlier `COV_INFLATE` `k_alpha_base` sweep, there is **no threshold and no
non-monotonicity of consequence** here: no infeasibility at any gain (`infeas=0`
throughout), every run reaches the goal, and severity moves smoothly. "Lower gain -> faster
and worse severity" is a real, smooth, monotonic property of the gain on this scene. It is
however far from linear: the first 40% of the gain range carries 83% of the severity
change.

### The mechanism is not what the accumulated-under-braking account assumed

Because the deepest penetration occurs on the step *after* the barrier releases, severity is
essentially set by the state at release. Measuring that state across the sweep:

```
  k_haz  rel_step  speed@rel  depth@rel   severity  max_speed     p@rel x    p@rel y   maxdev_y
   0.10        49     1.3692    -0.4164    -0.4251     1.5292    -0.10249   -0.41349    0.44353
   0.20        49     1.3707    -0.3424    -0.3424     1.5310    -0.10277   -0.42103    0.45466
   0.35        49     1.3626    -0.2557    -0.2561     1.5304    -0.11300   -0.42743    0.46647
   0.50        49     1.3766    -0.1836    -0.1836     1.5071    -0.11859   -0.43238    0.47702
   0.70        50     1.2073    -0.1475    -0.1529     1.4950    -0.18987   -0.41945    0.47371
   0.85        52     1.0342    -0.1398    -0.1398     1.4622    -0.26234   -0.40492    0.46512
   1.00        54     0.8493    -0.1186    -0.1330     1.4091    -0.35903   -0.38459    0.46144
```

(hazard mean is `[0, 0.03, 0]`; `p@rel` is the position at step 49 for every row, so the
columns are directly comparable.)

**Over gains 0.10-0.50 the approach speed is flat and the severity still nearly halves.**
Release happens at step 49 in all four cases, at speed 1.3692 / 1.3707 / 1.3626 / 1.3766
and peak speed 1.5292 / 1.5310 / 1.5304 / 1.5071 — differences of order 1e-2 — while
severity goes -0.4251 -> -0.1836. That is **0.2415 of the total 0.292 gap, i.e. 83%,
traversed with no meaningful change in how fast the robot is going.** A cumulative
*under-braking* story predicts the opposite: lower gain -> less speed shed -> faster at the
surface -> deeper. The speed column does not move.

What does move is **lateral** displacement: `p@rel y` goes -0.41349 -> -0.43238 and peak
lateral deviation 0.44353 -> 0.47702 as gain rises over that range. The robot is pushed
further off the axis, so at the same step and the same speed it is shallower inside the
ellipsoid. Only from gain 0.70 upward does the longitudinal channel engage — release slips
to step 50/52/54, `p@rel x` falls back from -0.119 to -0.359, and speed at release drops to
1.2073 / 1.0342 / 0.8493. The `time_ratio` column shows the same split: dead flat at 1.0597
across 0.1-0.35, rising only above 0.5.

**Noted, not changed —** this sits awkwardly against `ARCHITECTURE.md` §2.3's description
of `ALPHA_SCALE` as affecting "longitudinal approach speed only; doesn't change lateral
routing around an object". The ellipsoid geometry is indeed untouched by `ALPHA_SCALE`, so
that statement is right about the mechanism it describes; but the constraint normal
`w = gamma*Av - delta*Ar` has a lateral component, so scaling the constraint offset does
change the realized lateral displacement, and on this scene that is the dominant channel
over most of the gain range. Whether to reword §2.3 is a research call about a recorded
finding, not a coding call, so it is flagged here rather than applied.

### Verdict

**Something else contributes meaningfully beyond simple accumulation.** Stated precisely,
because two distinct claims are at stake:

- *Is there a per-step correction reduction that accumulates?* Yes — it is real at every
  step of both windows, and it is never adverse.
- *Is it the stable ~22%-per-step effect the open account assumed?* **No.** It ranges
  2.19%-30.29% (NONE family) and 5.41%-89.82% (ALPHA_SCALE family), with the largest
  discontinuities driven by active-set composition changes rather than by the gain. The
  ~22% at step 13 is not a plateau; it is a coincidence of the window's endpoints.
- *Does accumulated braking account for the gap?* **No, not for most of it.** 83% of the
  0.292 gap occurs over a gain range where speed at release is constant to ~1%. The
  accumulated quantity that actually tracks severity there is lateral displacement, not
  shed speed.

The approximate sufficiency check makes the same point quantitatively. Summing the
per-step correction differences over the window and multiplying by `dt` gives an implied
speed difference of **0.4569 m/s** at NONE's states but **0.2175 m/s** at ALPHA_SCALE's
states, against an actual difference of **0.5200 m/s** at the last hazard-active step. A
genuine stable per-step effect would give consistent answers from either family; a
factor-of-two disagreement between them is the signature of the path dependence.
(This check is an upper bound in one direction — correction magnitudes are not all aligned
with the direction of travel — and by construction ignores the trajectory divergence it is
testing for. It is reported as corroboration, not as a measurement.)

**What is now established.** The gain dependence is real, smooth and monotonic in severity,
with no threshold or infeasibility anywhere in `[0.1, 1.0]` — that much of the informal
account survives, and Experiment B's endpoints reproduce the committed rollout exactly.
The per-step reduction is real but not stable, and its accumulation is not a sufficient
explanation on its own.

**What is not established.** No claim is made here that lateral displacement is *the*
mechanism in general — it is what the data shows on this one pre-registered scene, whose
geometry (a 0.08 m pre-registered lateral corridor, `W_center = rho/2`) was deliberately
constructed to make lateral clearance marginal, and could plausibly amplify exactly this
channel. A second scene with a different corridor width would be the natural check and has
not been run. Nor has the other counterfactual from the previous section — replaying NONE's
control *sequence* under ALPHA_SCALE's gains — which tests something different from the
matched-state re-solve done here. `ARCHITECTURE.md` is unchanged; no mechanism has been
written into it.


## Fourth Stage 2 bug found and fixed (2026-09-10) — hero-frame selection had exactly one candidate pose

Found while tracing the Stage 1/2 boundary for a future-work paragraph about backbone
swaps, then confirmed by a dedicated read-only investigation before anything was changed.
Three coupled defects on the same code path in `vlm_safety_score.py`, all in
`extract_canonical_view()`, plus the failure-reporting gap explicitly deferred in the
2026-07-20 entry above. Fixed together because fixing any one alone leaves the path
broken or, in the case of (2), newly broken.

**Bug 1 — `params['w2c']` is not the camera path.** `extract_canonical_view()` did
`params['w2c'].reshape(-1, 4, 4)` and looped over the result as a trajectory.
`src/GS3LAM.py:481` is the **only** line in that file that assigns the key, and it
assigns `first_frame_w2c` — a single (4,4). (Lines 125/336/340 look similar but write
`curr_data`/`iter_data`, transient dicts consumed by `get_loss`, never saved.) Worse,
because the dataset is built with `relative_pose=True` (`src/GS3LAM.py:63` →
`src/datasets/basedataset.py:156,228`, "setting first pose in a sequence to identity"),
that single matrix is the **identity**. NumPy reshapes 16 elements into `(1,4,4)`
without complaint, so the loop ran exactly once, and "hero-frame selection" selected
from a candidate set of size one: every object was scored from frame 0, or failed.

The real per-frame poses were always there. `cam_unnorm_rots` (1,4,N) and `cam_trans`
(1,3,N) are world-to-camera relative to frame 0, and they are genuinely persisted —
`src/utils/logger.py:14-21` (`params2cpu`) converts every key with no whitelist, and
`src/GaussianManager.py:54` explicitly excludes both from per-splat pruning, so they
survive the run at full length. This mattered for scoping the fix: it is "read the right
key", not "GS3LAM never saved it", so no Stage 1 change was needed.

Now reconstructed in a new `load_per_frame_w2c()`, using the **estimated** poses
(GS3LAM's own tracking output) rather than `gt_w2c_all_frames`, so Stage 2 does not
depend on ground truth a real robot will not have. The quaternion → rotation step reuses
`src/utils/gaussian_utils.build_rotation` rather than reimplementing the `(w,x,y,z)`
convention — reimplementing a convention that already exists in the repo is precisely
what produced the SemanticDecoder `state_dict` bug in "Two Stage 2 bugs" above.

**Bug 2 — the `keyframe_time_indices` indirection.** The old code did
`actual_image_idx = keyframe_indices[best_frame_idx]`. With the loop fixed,
`best_frame_idx` is a dataset `time_idx`, which indexes `frame*.jpg` 1:1 under the
committed config (`start=0`, `stride=1`, `configs/Replica/room0.py:50-52`). Keeping the
indirection alongside the Bug 1 fix would have introduced a *new* wrong-image bug, which
is why these were not applied separately. Now `image_paths[best_frame_idx]` directly,
with an up-front check that `len(image_paths) >= num_frames` so a strided or offset run
fails loudly instead of silently pairing every pose with the wrong image.

**Bug 3 — intrinsics at the wrong resolution.** `params['intrinsics']`
(`src/GS3LAM.py:480`) is saved at the *downsampled training* resolution:
`src/datasets/basedataset.py:295` applies `datautils.scale_intrinsics(...)` before the
dataset returns the frame. But this script loads the original `frame*.jpg` at native
size, so the projected convex hull was landing in roughly the upper-left quadrant of the
image it was masking — the VLM would have been shown the wrong pixels even after Bugs 1
and 2 were fixed. New `load_native_resolution_K()` scales K back up, touching only
fx/fy/cx/cy to mirror `datautils.scale_intrinsics:112-115` exactly.

Upscaling K was chosen over downscaling the images so the VLM sees a full-resolution
crop, which is the input quality the safety score is meant to reflect. The ratio is
derived at runtime — actual on-disk frame size (read from the JPEG header via PIL, no
decode) divided by `params['org_width']`/`org_height` — never hardcoded. Note the trap:
despite the name, `org_width`/`org_height` are the *desired*, i.e. downsampled,
dimensions (`src/GS3LAM.py:482-483` assigns them from
`dataset_config["desired_image_width"/"desired_image_height"]`). For the committed
Replica config the factor works out to exactly 2.0 × 2.0 (`configs/camera/replica.yaml`
1200×680 against `configs/Replica/room0.py` 600×340), confirmed from those files, but
nothing in the code depends on that number.

**Bug 4 — silent, indistinguishable failure.** When no frame won, `best_frame_idx`
stayed `-1` and fell through to `keyframe_indices[-1]` (NumPy negative indexing — the
*last* keyframe), then died on `best_2d_points` being `None` inside the bare
`except Exception`, emitting the same "projection/VLM error" line and the same 0.0 score
as a dead API key. That is exactly the conflation flagged as "worth narrowing later" in
the 2026-07-20 entry above, and it is why Bug 1 survived unnoticed. Now a typed
`HeroFrameSelectionError` with a specific message, caught separately in `__main__` and
reported as "HERO-FRAME SELECTION FAILED (not a VLM error)". The 0.0 conservative
fail-safe is unchanged on both paths.

**Verification — static only, as with every Stage 2 fix this summer.** There is still no
real `params.npz`, `gsplat.ply`, `classifier.pth`, or `safety_gsplat.ply` anywhere on
this machine (the negative-result search recorded under "Two Stage 2 bugs found" above
still holds). **None of this has been executed against real Stage 1 output, because none
exists.** What was actually done:

- Control flow traced by reading: every `w2c` reference in `src/GS3LAM.py` (20 hits,
  one assignment), every rebinding of `params` (`:135`, `:262`, `:362`, `:366`) and each
  callee's key handling, and `save_params` → `params2cpu` → `np.savez`.
- The reshape degeneracy was *executed*, not reasoned about: a (4,4) array
  `.reshape(-1,4,4)` returns `(1,4,4)`, one loop iteration, no error.
- The batched pose plumbing was checked against the reference per-frame construction at
  `src/GS3LAM.py:415-419` (transcribed as a test oracle, since `build_rotation`
  hardcodes `device='cuda'` and this box is CPU-only): **bit-identical, max abs
  difference 0.0**, bottom rows `[0,0,0,1]`, rotation blocks orthonormal, translation
  column equal to `cam_trans`. This tests the batching/indexing that was written here;
  the quaternion convention itself comes from reusing the real function.
- The K rescale was checked to be the exact inverse of `datautils.scale_intrinsics`:
  native (600, 600, 599.5, 339.5) → saved (300, 300, 299.75, 169.75) → rescaled back to
  (600, 600, 599.5, 339.5).
- `python3 -m py_compile vlm_safety_score.py` passes.

Not verified, and not claimable without real data: that a real run selects sensible hero
frames, that `keyframe_time_indices[0] == 0` in a real npz, the actual on-disk resolution
of the Replica jpgs, and whether the VLM's scores improve. The fix makes the code do what
`ARCHITECTURE.md` §2.2 and the Method draft already said it did; it does not demonstrate
that the output is good.

**New finding, flagged not fixed:** `src/utils/gaussian_utils.py:24` hardcodes
`device='cuda'` inside `build_rotation` — the same class of issue as the
`SemanticDecoder.__init__` `.cuda()` finding fixed on 2026-07-18. Consequence: the fixed
hero-frame path now requires CUDA, whereas the old (broken) path did not, since it never
called `build_rotation`. On the A2000 dev machine this is harmless, but it is a real
reduction in device flexibility. Not patched here because `gaussian_utils.py` is a
Stage 1 file and CLAUDE.md holds Stage 1 as not needing structural changes; the call site
raises `HeroFrameSelectionError` with the reason stated plainly rather than letting an
opaque device-mismatch error escape from inside `build_rotation`. The one-line fix
(`device=q.device`) is available if wanted — it would mirror the SemanticDecoder fix
exactly.

**Docs updated alongside the code**, so nothing is left asserting the old behavior:
`CLAUDE.md`'s data contract, `ARCHITECTURE.md` §2.1 outputs and §2.2 step 1, and
`GS3LAM_PAPER_SCOPE.md`'s Stage 2 paragraph. Deliberately **not** touched:
`method_draft_v2_1.pdf`, `introduction_draft_v1.md`, `abstract_conclusion_draft_v1.md` —
all three already describe the *intended* mechanism ("project into every recovered camera
pose along the Stage 1 trajectory"), which the fixed code now actually implements, so
they became accurate without edits rather than needing them.

**Explicitly out of scope, deferred:** the runtime-cost characterization. The loop goes
from 1 iteration to `num_frames` (~2000 for Replica room0) per object, which changes
Stage 2's cost profile from trivial to the dominant offline cost. Left alone by decision;
revisit when real timing matters.
