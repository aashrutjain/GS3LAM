# 4. Experimental Protocol

This section specifies *how* the semantic weighting of the collision-cone CBF is
and will be evaluated — what is measured, on what scenes, against what baseline,
and with what metrics. It does not report outcomes; measured results appear in
Section 5. Two evaluation tiers are described. The first, a synthetic validation
of the Stage 3 controller, has been carried out and is reported here as completed
methodology. The second, a real-robot evaluation on a reconstructed room scene,
is a planned protocol that has not yet been run; it is described as a plan, and
the boundary between the two is kept explicit throughout.

The evaluation targets the *Execution* sub-question of Section 1: does weighting
the barrier with a per-object semantic safety score change navigation behavior —
collision severity, near-miss frequency, path efficiency — relative to pure
geometric avoidance, or does it merely reproduce the geometric solution? The
separate question of whether the VLM's per-object safety scalar is itself stable
(the *Translation* sub-question) is evaluated by an independent protocol in
Section 4.6 and must not be conflated with the CBF evaluation: the two answer
different sub-questions and share no scenes, metrics, or apparatus.

## 4.1 Platform and control setup

The target platform is a TurtleBot4, a 3 kg differential-drive robot; no
real-robot deployment has been performed, and every constant below that depends
on the physical robot or on a real scene is treated as a placeholder to be fixed
against hardware (Section 4.5). The robot is modeled as a sphere of radius
$\rho$ (configured at $\rho = 0.16$ m, a placeholder pending measurement of the
actual footprint), and its motion is integrated as a double integrator under a
proportional–derivative reference controller that tracks a straight line from
start to goal. At each step the reference control $\bar{u}$ is passed through the
CBF-QP safety filter, which returns the closest admissible control $u$ subject to
the per-splat barrier constraints and the actuator bound $\lVert u\rVert \le
a_{\max}$; the filtered control is then integrated forward. The confidence
ellipsoid of each splat is taken at the $\chi^2_{3,\,0.99}$ level, and the
robot's physical extent is folded into each splat's barrier by the per-splat
Minkowski inflation of the Method (Sec. 3).

Three controller configurations are compared, corresponding to the three
`SemanticMode` settings of the implementation and sharing one identical rollout
harness, start, goal, and reference controller:

- **NONE** — the pure-geometric baseline. The barrier is built from splat
  geometry alone and never reads the `safety` column. This is the control
  condition against which any semantic effect must be demonstrated.
- **ALPHA_SCALE** — the per-object safety scalar $s\in[0,1]$ modulates the
  barrier's class-K gain, $k_{\alpha,i} = k_{\alpha,\text{base}}\cdot f(s_i)$
  (identity $f$ by default, with a small floor so the gain of a splat cannot
  collapse to zero). This changes longitudinal approach aggressiveness only; it
  does not alter any confidence ellipsoid, and so by construction cannot change
  lateral routing.
- **COV_INFLATE** — the safety scalar inflates each splat's effective covariance,
  $\Sigma_{\text{eff},i} = \Sigma_i\,(1 + \lambda(1 - s_i))$, widening the
  geometry the collision cone is built from in proportion to hazard. This is the
  strategy that can route the robot laterally around a low-safety object rather
  than only braking for it. The inflation coefficient $\lambda$ (named
  `cov_inflate_gamma` in the implementation, and distinct from the collision-cone
  quantity $\gamma := r^\top A r - c^2$) is treated as a swept parameter, not a
  fixed constant.

Neither weighting strategy is asserted to be correct in advance; the comparison
against NONE, and against each other, is precisely the open design question the
evaluation exists to resolve.

## 4.2 Synthetic validation protocol (executed)

Synthetic validation is performed *first, and deliberately*, before any
real-scene evaluation. The reason is separability. Two distinct things could make
a semantic navigation result come out right or wrong: whether the weighting
mathematics does what its design predicts, and whether the upstream semantic
labeling of the real map is itself correct. These are independent failure modes,
and a real scene entangles them — a null result could mean the weighting does
nothing, or that the safety scores feeding it are wrong. A hand-built synthetic
scene, in which the safety field is assigned by construction rather than inferred,
answers the first question cleanly and in isolation, and does so without a GPU, a
trained map, or lab access. Real-scene evaluation (Section 4.5) then addresses the
second question once a real semantic map exists.

**Sanity-check scene.** The most basic check places a single low-safety hazard
directly on the straight-line path between start and goal, with all other geometry
assigned a near-safe background score, and confirms that each mode's one-step
response is what the barrier mathematics predicts — that ALPHA_SCALE brakes along
the approach axis without rerouting, and that COV_INFLATE alters the routing
geometry rather than only the speed. In the harness this is the "Phase A" path:
the loaded safety column is overridden in memory by a synthetic hazard field
(hazard placed at the midpoint of the start–goal line; hazard radius, hazard
score, and near-safe background supplied as parameters), so that the safety field
under test is known exactly and does not depend on any real labeling. An
independent, hand-computed check of the collision-cone barrier value for a robot
heading straight at a synthetic obstacle is used to confirm that the barrier
equations were transcribed correctly, and a receding trajectory is confirmed not
to activate the constraint.

**Corridor-width variants.** To probe lateral routing rather than a single
head-on encounter, a parameterized scene generator produces a corridor scene from
a fixed random seed: one hazard splat, a multi-ring "collar" of clutter splats
encircling the corridor around it (so that a detour must move in both lateral
axes, not merely sidestep along one), and a far-field bulk of background splats
placed beyond the spatial filter's radius so they contribute scene bulk without
entering the candidate set under test. Corridor width is controlled by a single
generator parameter (the collar radius), yielding a "narrow" and a "wide" variant
that are identical in every other respect — same seed, same hazard, same gain —
so that any difference between them is attributable to corridor tightness alone
and not to the weighting. The scene geometry was itself audited before use: a
perfectly centered hazard was found to create a degenerate exact-symmetry case in
which the one-step QP has no lateral gradient to act on, and was corrected by
offsetting the hazard–collar cluster slightly off the path centerline, matching
any realistic non-perfectly-centered approach.

**Three-mode comparison and oracle.** For a given scene, NONE, ALPHA_SCALE, and
COV_INFLATE are each driven through the identical rollout between the same start
and goal, and the metrics of Section 4.3 are tabulated for each. Start and goal
are first verified to lie outside every splat's confidence ellipsoid, using the
pure-geometric baseline geometry so that an invalid endpoint fails identically
regardless of weighting. A fourth, non-competing run disables the CBF entirely (a
pass-through of the reference control) and is used solely to obtain the
obstacle-free time-to-goal that serves as the denominator of the path-efficiency
time ratio; it is never scored as a candidate mode.

**Gain sweep.** Because the class-K gain $k_{\alpha,\text{base}}$ governs how
permissively the barrier is allowed to be approached before it engages, it is
swept across roughly two orders of magnitude on the narrow-corridor scene, with
the weighting mode and inflation coefficient held fixed. The purpose of the sweep
is to determine whether an observed behavior is a property of the scene geometry
or of the fixed gain — i.e., whether the same corridor is navigated differently at
different gains — so that gain-induced conservatism is not misattributed to the
semantic weighting or to a genuine absence of a safe corridor.

## 4.3 Metrics

Three metric families are computed from each rollout's trajectory. They are
implemented once and applied identically across all modes; the precise
definitions below match the implementation rather than a generic gloss.

**Collision severity.** At each timestep the signed distance from the robot
position to the nearest splat is computed as a Mahalanobis distance to that
splat's confidence ellipsoid minus the Minkowski-inflated confidence radius,
minimized over the splat set:
$d(t) = \min_i \big[\sqrt{(p(t)-\mu_i)^\top A_i (p(t)-\mu_i)} - c_{m,i}\big]$.
A negative value means the trajectory has entered a splat's inflated confidence
ellipsoid at that instant. Collision severity is the worst (most negative) such
value over the whole trajectory. The quantity is expressed in Mahalanobis
(ellipsoid-normalized) units, not meters — it measures penetration depth relative
to each object's own confidence geometry, which is the quantity the barrier is
defined on.

Crucially, severity and near-miss counts for *all three* modes are evaluated
against a single shared reference geometry — the true, un-inflated splat
boundaries of the pure-geometric (NONE) filter — rather than against whatever
geometry each mode's own controller perceived. This matters specifically for
COV_INFLATE, whose controller routes around an artificially widened ellipsoid:
scoring its trajectory against that same widened ellipsoid would flatter it
relative to NONE and ALPHA_SCALE, which see the true boundary, and the comparison
would be uninterpretable. Each mode's own controller still drives its own rollout
— routing and braking decisions reflect what that mode actually sees — but the
post-hoc metric is computed against one common true boundary, so that "closer to
the real object" means the same thing for every mode.

**Near-miss events.** A near-miss is a distinct close-approach event in which the
signed distance above dips below a threshold $d_{\text{thresh}}$. The count is
deduplicated by hysteresis: an event is registered when the distance first crosses
below the threshold, and is not re-counted until the distance has risen back above
the threshold and dipped below it again. Without this, a single slow pass by an
object would be counted many times over — once for every consecutive timestep that
happens to sit below the threshold — inflating the count of what is physically one
approach. The threshold is a configurable parameter of the evaluation (default
0.3), not a hard-coded constant, and is expressed in the same
Mahalanobis-normalized signed-distance units as the severity metric above.

**Path efficiency.** Two ratios are reported. The path-length ratio is the actual
trajectory arc length divided by the straight-line start-to-goal distance,
capturing how much the robot detoured. The time ratio is the actual time-to-goal
divided by the oracle time-to-goal from the CBF-disabled run of Section 4.2,
capturing how much the safety filter slowed the robot down relative to an
unconstrained straight-line traversal of the same start and goal.

## 4.4 Determinism and solver verification

The control pipeline is deterministic. Given a fixed scene, a fixed seed, and a
fixed configuration, each mode produces exactly one trajectory; there is no
averaging over random trials because there is no stochasticity in the pipeline
once the scene is fixed. The synthetic scenes are generated from a fixed seed, the
rollout is deterministic, and both QP backends are deterministic solvers. The
evaluation therefore characterizes what each weighting strategy does on a
*specified* scene and configuration, and comparisons across modes are exact rather
than distributional (see Section 4.7 for what this does and does not license).

The one place where determinism is leveraged for a genuine correctness argument is
solver verification. The QP is solvable by two independent backends — an
interior-point conic solver that handles the actuator-norm bound as a native
second-order cone, and a general-purpose sequential-least-squares solver retained
as a fallback. Because the two implement the same constraints by entirely
different algorithms, solver correctness is checked by a state-by-state re-solve:
every QP sub-problem encountered along a rollout is re-solved with both backends,
and their objective values and constraint satisfaction are compared at identical
states. Agreement of two unrelated algorithms on objective value and on the set of
states declared infeasible is strong evidence that the QP is being solved
correctly, and disagreement localizes exactly which states one backend solves
sub-optimally. This is a targeted claim about solver correctness on the scenes
tested — it is not, and is not presented as, evidence about navigation behavior
across scenes.

## 4.5 Real-data protocol (planned — not yet run)

The real-robot evaluation described here has not been performed. It is blocked on
a real Stage 1/2 output — a real semantic map with real per-object safety scores —
which does not yet exist on any machine used in this work, and on lab and GPU
access. It is stated as a plan.

Once a real semantic map of the reconstructed room scene (`room0`) exists, the
identical three-mode comparison and identical metrics of Sections 4.2–4.3 will be
applied, with one change: the `safety` column will be the real, VLM-derived
per-object scores rather than a synthetic hazard field. This is the "Phase B"
path of the harness, and it is precisely the step that tests the second of the two
separable questions from Section 4.2 — whether semantic weighting changes outcomes
on *real*, non-synthetic hazards. Several configuration constants that are
placeholders in the synthetic runs must be fixed against reality before this
evaluation is meaningful: the spatial-filter radius and candidate limits must be
set against the real scene's measured bounding box, and the robot-footprint radius
$\rho$ used in Minkowski inflation must be set to the TurtleBot4's actual measured
footprint. On real hardware, the same metrics would be computed against the real
scene geometry rather than a synthetic reference. Until a real map exists, this
protocol cannot produce a navigation result, and none is claimed.

## 4.6 VLM safety-score consistency protocol

This is a separate evaluation answering a separate sub-question — *Translation*,
not *Execution* (Section 1): can a free-form VLM judgment collapse into a single
$[0,1]$ per-object safety scalar without that scalar being unstable from query to
query? It shares no apparatus with the CBF evaluation above and is reported
separately; it is not a navigation experiment and produces no trajectory.

The protocol is a repeated-query consistency measurement. For each object in a
fixed object set, the production safety-auditor prompt is issued to the VLM
several times (five queries per object) at a fixed, non-zero sampling temperature,
with the generation configuration held identical to production (JSON-constrained
output, a fixed output-token budget, and the internal "thinking" pass disabled),
and the model pinned to a single dated release rather than a rolling alias so the
run is reproducible. Every knob other than the image is held constant across the
repeats, so that the only thing varying between a given object's queries is the
model's own sampling. The measured quantity is the within-object variation of the
returned scalar — the spread of an object's repeated scores — which tests whether
the score is a stable property of the object or a noisy draw.

The object set is drawn from real Replica `room0` RGB frames: objects spanning an
intended safety gradient (safe, mid, and hazard categories) were rough-cropped by
hand from sampled frames, together with a floor patch included deliberately as a
"flat solid ground" positive control. This is a proxy for, not an instance of, the
full Stage 2 hero-frame pipeline: these are eyeballed bounding-box crops without
convex-hull background suppression or classifier-decoded object masks, matching the
rigor level of the measurement rather than the full pipeline. The
fully-hero-frame version — background-suppressed crops decoded from a real
semantic map — remains future work, tied to the same real Stage 1/2 output that
Section 4.5 depends on. Because this protocol involves repeated sampling at a
non-zero temperature, it is the one part of the evaluation with a genuine
stochastic component, and its statistical character (per-object spread over
repeats) is distinct from the deterministic, single-trajectory character of the
CBF evaluation.

## 4.7 Scope and statistical character of the evaluation

The synthetic CBF evaluation is, at present, a single-scene study with controlled
parameter variations — corridor width and class-K gain — not a multi-scene
statistical study over many rooms or many hazard configurations. What it
establishes is deterministic and exact for the scenes it is run on: given a fixed
scene and configuration, each mode yields one trajectory, and differences between
modes and between parameter settings are read off directly rather than inferred
from a distribution over trials. This is a real strength for the specific claims it
supports — it makes the comparison exactly reproducible and isolates the
effect of a single varied parameter — but it is deliberately not oversold as
evidence that any observed behavior generalizes across scenes, robots, or real
semantic maps. Generalization across multiple room configurations is explicitly
future work.

Two things must not be collapsed together here. Whether a given behavior is a
scene-general property of the weighting strategy is a question this single-scene
protocol cannot settle, and no such generality is claimed. Whether the QP is
solved correctly is a different question, and one this protocol is designed to
settle rigorously for the scenes tested, via the two-backend state-by-state
re-solve of Section 4.4: agreement between the two backends would constitute strong
evidence for solver correctness precisely because it does not depend on scene
diversity — the check is an internal-consistency comparison between two independent
algorithms on the exact sub-problems the evaluation encountered. The two claims are reported at their true strengths: the
solver-correctness check is designed to be decisive for the tested scenes;
scene-general navigation benefit is a target of the planned real-data evaluation,
not an established result.

---

<!-- ============================================================= -->
<!-- DRAFTING NOTES — REMOVE BEFORE SUBMISSION                     -->
<!-- Flags requested: (a) protocol-vs-result judgment calls,      -->
<!-- (b) doc/code mismatches, (c) missing method_draft_v2_1.      -->
<!-- These are notes to the authors, not paper prose.             -->
<!-- ============================================================= -->

## Drafting notes (remove before submission)

### Missing input: `method_draft_v2_1`
The final Method draft could not be located. It is not in the `GS3LAM` repo in any
form (`.md`, `.tex`, `.docx`, `.txt`), and there are no saved Cowork artifacts on
the connected device. Notation here was therefore anchored to `ARCHITECTURE.md`
§2.3, which is defensible because `GS3LAM_PAPER_SCOPE.md` (line 267) instructs the
Method section to use "the exact math ... per `ARCHITECTURE.md` §2.3," so the
Method draft should already follow it. Symbols to reconcile against the actual
draft before this is final:
- The covariance-inflation coefficient is written $\lambda$ here (matching
  `ARCHITECTURE.md` §2.3 prose, which flags $\lambda$ as distinct from the
  collision-cone $\gamma$). The code and eval CLI call it `cov_inflate_gamma` /
  `--cov-gamma`. If the Method draft settled on $\gamma$ or another symbol, change
  it here to match.
- Class-K gain is $k_{\alpha}$ / $k_{\alpha,\text{base}}$ (matching the code's
  `k_alpha` rename of Tscholl's $p_k$). Confirm the draft uses the same.
- Mode names (NONE / ALPHA_SCALE / COV_INFLATE) are used as prose labels; confirm
  the draft's prose names for the two weighting strategies and align.

### Protocol-vs-result judgment calls (where the line was hard)
1. **Shared true-geometry grading (Section 4.3).** Describing *that* all modes are
   scored against one common NONE-geometry reference is protocol (a description of
   how the metric is computed). I deliberately did not state the *effect* this had
   on any mode's numbers (that would be a result). Borderline because the reason
   the code does this is a bug that was found and fixed — but the "how we measure"
   is protocol, the "what it changed" is Results.
2. **Solver verification (Sections 4.4/4.7).** I described the *method* of the
   two-backend state-by-state re-solve and framed it as evidence for solver
   correctness, per your instruction not to undersell it. I excluded every
   outcome (which backend was suboptimal, on what fraction of steps, the time-ratio
   effect) — those are Results. The claim "strong evidence for solver correctness"
   is a characterization of the method's logic, not a measured finding, so I judged
   it in-bounds for Protocol; flag if you'd rather it move to Results/Discussion.
3. **Design parameters vs finding numbers.** I included experiment-design numbers
   that a protocol needs (χ²₍₃,₀.₉₉₎ confidence level, 5 queries/object,
   temperature, seed-controlled scenes, narrow/wide corridor variants, a gain
   sweep across ~2 orders of magnitude, ρ=0.16, near-miss threshold 0.3). I
   excluded every *outcome* number (time ratios, severities, the ~5.9×/~11.9×
   slowdown, infeasibility counts, per-object variances). If you want the section
   even more number-free, the gain-sweep range and the placeholder constants are
   the first candidates to cut — but a protocol without its parameters stops being
   a protocol, so I kept them.
4. **Determinism as protocol.** Stating "the comparison is deterministic, not
   distributional" is a property of the method, so it's protocol. The *fact* that
   two solvers reproduced bit-identically is a result and was excluded.

### Doc/code mismatches found (report, not paper prose)
1. **Near-miss threshold units — real mismatch.** `eval_cbf_modes.py`'s
   `--near-miss-thresh` help string says `"meters, for near_miss_events"`, but the
   value is compared against `mahalanobis_signed_distance(...)`, which returns
   `md − c_m` in dimensionless Mahalanobis units (`PROGRESS.md` itself calls
   severity "Mahalanobis units, not meters"). So the threshold is *not* in meters.
   I described it in the section as being in Mahalanobis-normalized units and did
   not repeat the "meters" label. Worth fixing the help string in the code.
2. **Oracle description — minor wording mismatch.** `metrics.py`'s
   `path_efficiency` docstring calls the oracle an "obstacle-free copy of the
   scene," but the implementation (`PassthroughFilter` in `eval_cbf_modes.py`)
   doesn't remove obstacles — it disables the CBF and passes the reference control
   through on the same scene. Same timing effect (robot ignores obstacles), but the
   mechanism is "CBF disabled," not "obstacles removed." I described it as
   CBF-disabled to be accurate.
3. **Scene-generator docstring is stale — minor.** `scripts/gen_cbf_synthetic_scene.py`'s
   module docstring says "double-ring collar" and "300 background splats (240
   far-field 'bulk' + 60 collar)," but `build_scene()` builds 5 rings × 30 = 150
   collar + 150 bulk. `PROGRESS.md` correctly documents the 5-ring / ~150-bulk
   version; only the generator's own docstring wasn't updated. I described the
   collar generically ("multi-ring") to avoid propagating the stale count.
4. **Not a mismatch, but scope-relevant:** `vlm_safety_score.py`'s `__main__` has a
   top-5-object cap (`np.argsort(-counts)[:5]`) that contradicts the "one query per
   object" method (`PROGRESS.md`, 2026-07-20). It lives only in the demo `__main__`,
   not the library, and the Method draft is noted as describing the uncapped
   per-object behavior — so I described the consistency protocol as per-object
   (no cap), consistent with the intended method. Flag if the real run will hit
   the cap.
