# ARCHITECTURE.md

Deep technical reference for the Semantic Gaussian Field navigation project (RISL, RPI).
Read this before making any non-trivial change. For quick-start commands and environment
setup, see `CLAUDE.md`.

## 1. Research question

How can high-level, VLM-derived semantic safety reasoning be mathematically integrated
into real-time, low-level robot control, using continuous 3D Semantic Gaussian Fields
as the translation layer?

Three sub-questions structure the project:

- **Representation** — does a unified Semantic Gaussian Field give more stable, less
  noisy grounding for VLM safety queries than sequential 2D tracking pipelines?
- **Translation** — can a free-form VLM judgment collapse into a single [0,1] safety
  scalar per object without losing the fidelity of the VLM's underlying reasoning?
- **Execution** — does weighting a Control Barrier Function with semantic safety scores
  actually improve navigation outcomes (collision severity, near-miss frequency, path
  efficiency) over pure geometric avoidance, or does it just reproduce the geometric
  solution?

## 2. Pipeline overview

Three stages. The third is the current focus — mostly built and smoke-tested, see 2.3.

1. **Semantic Gaussian Splatting** (done) — GS3LAM produces a semantically labeled 3DGS map.
2. **VLM Safety Rating via Hero Frames** (done) — per-object safety scores grafted onto the splats.
3. **Costmap + CBF Tuning** (mostly built) — the safety-annotated map drives a real CBF on a TurtleBot4.

### 2.1 Stage 1 — GS3LAM

Three concurrent threads:

- **Semantic thread** — DEVA (a SAM2-like video-object segmentation model) produces a
  2D semantic map per frame.
- **Tracking thread** — gradient descent recovers a 4×4 world-to-camera matrix per
  frame (evaluated against Replica's ground-truth poses).
- **Mapping thread** — minimizes a combined photometric + geometric + semantic loss, so
  splat boundaries align with object boundaries directly during optimization. This is
  the key improvement over v1: no post-hoc ray casting required.

Outputs:
- `gsplat.ply` — geometry, color, opacity per splat.
- `params.npz` — `w2c` (camera path) and `obj_dc` (16-D semantic vector per splat).

Dataset: Replica (synthetic, zero motion blur), replacing TUM RGB-D `freiburg1_desk`
from v1. Note Replica still has holes in under-mapped regions — not perfect ground truth.

### 2.2 Stage 2 — VLM Safety Rating

1. **Hero frame selection** — for each semantic object, pick the camera pose that
   maximizes on-screen pixel area (`size = w2c · H · K`). Reduces VLM calls to one per
   object instead of one per keyframe.
2. **Convex hull + background suppression** — isolate the object in its hero frame so
   the VLM reasons about the object's material/structure, not incidental scene context.
3. **VLM query** — Gemini 1.5 Flash, `response_mime_type = "application/json"`, returns
   `{"safety_score": float}` in [0,1]. Prompt frames the VLM as a physical safety
   auditor for a 3kg wheeled robot (TurtleBot4): 1.0 = safe to drive near, 0.0 = lethal
   hazard / fragile / easily tipped / cables.
4. **Splat augmentation** — `numpy.lib.recfunctions` grafts the score onto `gsplat.ply`
   as a new `safety` column → `safety_gsplat.ply`.

VLM latency is 1–3s/call, but it's offline and per-object, not per-frame, so it doesn't
touch the control loop.

### 2.3 Stage 3 — Costmap + CBF (current focus, mostly built)

**Chosen geometric primitive:** Tscholl, Nakka & Gunter, "Perception-Integrated Safety
Critical Control via Analytic Collision Cone Barrier Functions on 3D Gaussian
Splatting" (arXiv:2509.14421, Sept 2025). Converts each splat into a closed-form
forward collision cone, yielding a first-order CBF in a QP — no high-order CBF
extensions needed. Validated on ~170k splats: 3x faster planning, lower trajectory
jerk than SAFER-Splat-style distance CBFs, same safety level.

**Open design question — this is the actual unresolved research contribution of Stage
3:** how to combine the geometric collision cone with the `safety` scalar from Stage 2.
Candidates, neither validated:

- *Multiplicative weight on the CBF's class-K function / α parameter.* Low safety
  score → more conservative braking. Affects longitudinal approach speed only; doesn't
  change lateral routing around an object.
- *Covariance inflation.* Scale the splat's effective covariance by
  `Σ_eff = Σ · (1 + γ(1 − safety))`, so low-safety objects cast a wider "semantic
  shadow" that the collision cone routes around laterally, not just brakes for. This
  is a hypothesis, not a published result — it needs to be derived carefully and
  validated against the actual collision-cone math before being treated as settled.

Do not treat either of these as decided. Resolving this is the point of Stage 3.

**Module status (see `PROGRESS.md` for exact state):** the geometric CBF-QP library
exists at `src/cbf/`, with both weighting strategies above implemented as swappable
`SemanticMode` options over a shared `SemanticMode.NONE` baseline that never reads the
`safety` column, plus `eval_cbf_modes.py`, the 3-mode comparison harness. Smoke-tested
against a synthetic scene (no real Stage 1/2 output exists in this checkout yet) — the
collision-cone math was confirmed to match an independent hand calculation, and a real
dtype bug in the QP solver path was found and fixed in the process (see `PROGRESS.md`).
`ALPHA_SCALE` vs `NONE` behaved exactly as predicted (same path, slower approach);
`COV_INFLATE`'s behavior in that synthetic scene is a real open finding, not yet
disambiguated from a possible gain-tuning artifact — see `PROGRESS.md`. Not yet built:
a pinned QP-solver dependency (currently a `scipy.optimize.SLSQP` stopgap behind an
abstract `CBFQPConfig.solver` string — see `src/cbf/qp_filter.py`'s module docstring),
a real (non-ad-hoc) unit test suite, and the ROS2/TurtleBot4 node itself (the seam for
it exists at `src/cbf/interfaces.py`).

Module layout:
```
src/cbf/
    ply_io.py             # loader: safety_gsplat.ply -> SplatField
    ellipsoid.py           # quaternion normalize, R, Sigma = R S S^T R^T, A = Sigma^-1, s_min
    semantic_weighting.py  # SemanticMode enum + alpha-scaling / covariance-inflation
    collision_cone.py       # per-splat beta/gamma/delta/h/w(x), Minkowski inflation
    spatial_filter.py       # radius/kNN candidate-splat prefilter (scipy cKDTree)
    qp_filter.py             # CBFQPConfig, CBFSafetyFilter, QP assembly + solve
    dynamics.py              # DoubleIntegratorState, step, PD reference controller
    interfaces.py            # RobotState, SafetyFilterResult, SafetyFilter protocol (ROS2 seam)
    metrics.py               # severity / near-miss / path-efficiency metric functions
    sim.py                   # trajectory rollout harness
configs/cbf/room0_cbf.py    # scene config (mirrors configs/Replica/room0.py style)
costmap_cbf.py               # top-level: online single-step filter demo (the ROS2-facing piece)
eval_cbf_modes.py            # top-level: 3-mode comparison harness
```

**Exact math implemented** (Tscholl, Nakka & Gunter, arXiv:2509.14421; the paper's
class-K gain constant `p_k` is renamed `k_alpha` throughout this codebase to avoid
clashing with robot position `p`):

- Confidence ellipsoid per splat: `A := Σ⁻¹`, `ℰ = {x | (x−μ)ᵀA(x−μ) ≤ c²}`,
  `c² = χ²_{3,0.99}`.
- Collision-cone existence (Eq 9a/9b): with `β:=vᵀAv, γ:=rᵀAr−c², δ:=rᵀAv`
  (`r := μ−p`), a collision exists iff `βγ − δ² ≤ 0` and `δ ≥ 0`.
- Barrier function (Eq 13): `h(p,v) := βγ − δ² ≥ 0` defines the per-splat safe set.
- QP: `min_u ‖u−ū‖² s.t. w(x)ᵀu ≥ −(k_alpha/2)h(p,v), ‖u‖ ≤ a_max`, where
  `w(x) := γ·Av − δ·Ar`.
- Minkowski inflation for robot radius `ρ` (Remark 3): `c_m = c + ρ/s_min`, and since
  `R` is orthogonal, `s_min` is just `min(exp(log_scale))` per splat — no SVD needed.

**Confirmed data contract detail** (from reading `src/utils/logger.py` and
`vlm_safety_score.py`, not assumed 3DGS convention): `opacity` is stored as a logit,
`scale_0..2` as log-scale, `rot_0..3` as an unnormalized `(w,x,y,z)` quaternion —
`src/cbf/ply_io.py` and `ellipsoid.py` un-transform these. Splats never queried by the
VLM default to `safety = 0.0` (bit-identical to a genuine "lethal" rating, and
unrecoverable from the PLY alone — no `class_ids` or queried-class-set is persisted).
`src/cbf/ply_io.py`'s `ambiguous_zero_mask` / `ZeroSafetyPolicy` surfaces this rather
than silently trusting it.

**Two real Stage 2 bugs found while building this — now fixed** (flagged per the Rules
section in `CLAUDE.md`, since Stage 2 was believed not to need changes; fixed in a
dedicated follow-up session scoped to exactly these two bugs, see `PROGRESS.md`):
1. `vlm_safety_score.py`'s `SemanticDecoder` used to be a from-scratch `nn.Linear(16,256)`
   with different state_dict keys than the actually-trained classifier
   (`src/Decoder.py`'s `nn.Conv2d(16,256,kernel_size=1)`), so `load_state_dict(...,
   strict=False)` silently loaded nothing. Fixed by importing `src.Decoder.SemanticDecoder`
   directly instead of reimplementing it, reshaping the flat `(N,16)` per-splat tensor to
   `(N,16,1,1)` to reuse the 1x1-conv architecture correctly (no spatial mixing, so this
   is mathematically identical to the per-pixel `(C,H,W)` case it's normally used in —
   `src/Evaluater.py`, `src/Loss.py`), and changing `strict=False` to `strict=True`.
2. `vlm_safety_score.py` called `glob.glob(...)` without `import glob` — fixed.

**Verification caveat:** no real `classifier.pth`/`gsplat.ply`/`params.npz` exists
anywhere on the machine the fix was made on (confirmed by an explicit search — see
`PROGRESS.md`), and that machine also has no torch/CUDA available, so the fix could only
be verified statically (same class + same constructor args as the code that saved
`classifier.pth`, guaranteeing key match by construction) plus a written-but-unexecuted
regression test (`tests/test_semantic_decoder_load.py`). A real dynamic check — and a
real Phase B evaluation of the open weighting question below — remain blocked on a real
Stage 1/2 run existing somewhere, not on this bug anymore. Separately, incidentally
found: `src/Decoder.py`'s `SemanticDecoder.__init__` hardcodes `.cuda()`, so it can only
ever be constructed on a CUDA-capable machine — harmless on the real A2000 dev box, but
worth knowing if anyone tries to run Stage 2 code on a CPU-only machine.

## 3. Design history — why v2 replaced v1

v1 (SplaTAM + SAM2 + post-hoc ray casting, on TUM `freiburg1_desk`) was abandoned:

- TUM `freiburg1_desk` (2011, first-gen Kinect) has heavy motion blur, depth holes, and
  a desk-level viewpoint — a poor match for TurtleBot-eye-level navigation.
- Ray casting 2D mask labels into the 3D map was computationally expensive, and errors
  in mask propagation compounded with errors in pose estimation, producing a noisy
  semantic field.

v2 (GS3LAM + Replica + hero frames) fixes both: semantics participate in the mapping
loss directly (no ray casting stage at all), and Replica isolates the pipeline from
sensor artifacts so it can be validated on the algorithm rather than the sensor.

## 4. Related work — where this project's novelty actually sits

Easy to conflate this project with adjacent 3DGS-safety work. It is not the same as
any of the following — keep this table current as new papers show up:

| Paper | What it does | How this project differs |
|---|---|---|
| SAFER-Splat (arXiv:2409.09868) | Distance-based CBF per splat, purely geometric, no semantic differentiation | No VLM anywhere; this project adds the semantic safety layer on top of a geometric CBF |
| Tscholl et al. (arXiv:2509.14421) | Collision-cone CBF, purely geometric — the word "semantic" doesn't appear in the paper | This is the Stage 3 geometric primitive we build on, not a competing approach |
| AlphaAdj | Frame-by-frame VLM risk scoring, 2D, reactive, bottlenecked by per-frame API latency | This project pre-computes a 3D safety field asynchronously via hero frames instead |
| GS3LAM | Semantic SLAM — geometry + class labels, no safety/hazard scoring at all | This project's Stage 1 front end; the safety layer is added in Stage 2 |

**The actual novel contribution:** a VLM-derived, per-object semantic safety scalar
grafted directly onto the 3D Gaussian representation as a first-class splat attribute,
via a hero-frame mechanism that avoids per-frame VLM cost. Nothing in the table above
does this. Keep this distinction sharp in any write-up — it's the thing that makes the
paper worth publishing, and it's untouched by the Tscholl paper or anything else
reviewed so far (SOTA literature pass, mid-2026).

## 5. Explicitly out of scope (future work, not current milestone)

Pulled from the original roadmap plus a SOTA literature pass. Don't implement any of
these unless asked directly — logged here so future sessions don't wander into them:

- **Real-time dynamic updates via OAK-D** — hybrid online/offline architecture where
  the robot halts and queries the VLM only when the geometric CBF can't route around a
  newly detected, unmapped object.
- **VLM latency removal via distillation** — train a small local model on
  Gemini-generated safety scores offline (EAMP/VISA-style architecture) so online
  inference doesn't need an API call at all.
- **Dynamic-scene SLAM backbone migration** — replacing GS3LAM with something that
  natively tracks per-object dynamic probability (e.g., DL-SLAM), so the map stops
  assuming a static environment.
- **Predictive/anticipatory safety** — a Gaussian World Model that forecasts future
  scene states so the CBF can react before a hazard materializes, rather than scoring
  only the current frame. GPU/VRAM cost is currently prohibitive on target hardware.

## 6. Key references

- Tscholl, D., Nakka, Y., Gunter, B. "Perception-Integrated Safety Critical Control via
  Analytic Collision Cone Barrier Functions on 3D Gaussian Splatting." arXiv:2509.14421, 2025.
- Chen, T. et al. "SAFER-Splat: A Control Barrier Function for Safe Navigation with
  Online Gaussian Splatting Maps." arXiv:2409.09868, 2024.
- Li, L., Zhang, L., Wang, Z., Shen, Y. "GS3LAM: Gaussian Semantic Splatting SLAM."
  Proc. 32nd ACM Int. Conf. Multimedia (MM '24), Melbourne, VIC, Australia, 2024,
  pp. 3019-3027.
- Straub, J. et al. "The Replica Dataset: A Digital Replica of Indoor Spaces."
  arXiv:1906.05797, 2019.
- Sermanet, P. et al. "Generating Robot Constitutions & Benchmarks for Semantic
  Safety." CoRL, 2025. (ASIMOV benchmark — candidate for validating VLM safety-score
  accuracy against ground truth, not yet integrated.)

Full literature review (Splatblox, GWM, GSFF-SLAM, EAMP, DL-SLAM, VISA, and others
surveyed mid-2026) lives outside this repo — ask Aashrut for the doc if deeper context
on any of these is needed.
