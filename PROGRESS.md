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
