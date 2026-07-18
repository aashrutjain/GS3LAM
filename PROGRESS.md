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
