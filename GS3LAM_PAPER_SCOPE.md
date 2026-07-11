# GS³LAM Paper Draft — Claude Session Context

## Purpose of This Document

This document gives a Claude session full context to help write a research paper draft
on semantic safety-aware robot navigation. **Read this entirely before producing any
output, especially the status section immediately below.** A prior version of this
document incorrectly stated experiments were complete and misdescribed Stage 1; this
version corrects both. If a future session's understanding of the project's status
conflicts with this document, this document wins — it was written by cross-referencing
`ARCHITECTURE.md` and `PROGRESS.md` directly, not from a general template.

---

## Current Status (Read This First)

**No real experiments have been run yet.** Everything below is either (a) genuinely
finished writing-ready content with no experimental dependency, (b) a synthetic
sanity check, real but not a real-world result, or (c) blocked on GPU/lab access,
which resumes fall semester 2026. Do not conflate (b) or (c) with a completed result.

- No real `gsplat.ply`, `params.npz`, or `classifier.pth` has ever been confirmed to
  exist on any machine checked so far. Stage 1 (GS3LAM training) has apparently never
  been run to completion anywhere accessible this summer.
- Stage 2 had a real bug (a classifier architecture mismatch silently producing
  meaningless safety scores) that has been fixed in code, but never dynamically
  verified — no CUDA-capable machine has been available to run the regression test.
- Stage 3 (the CBF/costmap library) exists, is built, and its core math has been
  independently verified equation-for-equation against Tscholl et al. (arXiv:2509.14421).
  It has been run exactly once, against a hand-built synthetic scene (300 background
  splats + 1 obstacle). It has never touched real data or a real robot.
- One open research finding from that synthetic run is unresolved: whether the
  `COV_INFLATE` semantic-weighting mode's slow-convergence behavior is a legitimate
  "no safe corridor" result or an artifact of a fixed gain parameter.
- No TurtleBot4 deployment, real or simulated-on-real-data, has happened.

**What this means for the paper:** the writing can and should proceed now for every
section that describes the method rather than reports a result. The result-bearing
sections should be written as protocol/placeholder this summer and filled in with real
numbers in the fall. Two specific experiments below don't need a GPU and should be done
this summer, not deferred, since they directly strengthen the paper's empirical core
before lab access returns.

---

## Section-by-Section Feasibility This Summer

| Section | This summer | Notes |
|---|---|---|
| Related Work | Fully finished | No experimental dependency. Citation list already spot-verified against arXiv (see "Verified Related-Work Candidates" below). |
| Method | Fully finished | Describes the system as built, not results. Must use the corrected Stage 1 description below (GS3LAM/DEVA, not SAM2). |
| Introduction | Fully finished | Contributions must be stated as what's actually built/verified, not "demonstrated on TurtleBot4" (see Contributions below). |
| Experimental Protocol | Fully finished, as protocol | What will be measured and how, not what was found. Writing this precisely now is real work — it's what prevents figuring out what counts as a result after the fact. |
| Results — synthetic validation | Real, partial content | The Phase-A synthetic sanity check (collision-cone math verified, `ALPHA_SCALE` behaving as predicted) is legitimate content, framed explicitly as synthetic validation of the implementation, not a navigation result. |
| Results — VLM score reliability | Done (2026-07-09) | See item 1 under "This Summer's Remaining Experimental Work" below for the finding. Ran on stock-photo proxies, not real Replica hero-frames — see Gap Tracking. |
| Results — real-world / real-robot | Blocked until fall | Needs real Stage 1/2 output and lab/GPU access. Do not draft as if this exists. |
| Abstract, Discussion, Conclusion | Drafted now, revised once real numbers exist | Skeleton now; light revision pass in fall. |

---

## Project Overview

### What Was Built

A pipeline that grafts per-splat semantic safety scores, derived from a Vision-Language
Model, onto a 3D Gaussian Splatting map, and integrates those scores into a
Control-Barrier-Function-based safety filter for a TurtleBot4. The goal is navigation
behavior that responds to *what an object is*, not just that it occupies space.

### What Was NOT Built

- **GS³LAM** (the underlying semantic Gaussian Splatting SLAM system) is a foundation
  the pipeline builds on, not original work. Say "we build upon GS³LAM to..." — never
  describe GS³LAM itself as something this project created.
- **The Analytic Collision Cone CBF** (Tscholl, Nakka & Gunter, arXiv:2509.14421) is
  the geometric primitive Stage 3 is built on. It is purely geometric — no semantics,
  no VLM anywhere in that paper. This project's contribution is the semantic layer
  added on top of it, not the CBF formulation itself.
- **No real-world TurtleBot4 deployment has happened.** Do not describe or imply one.

---

## System Architecture (Method Section)

The pipeline has three stages. **Correction from a prior draft of this document: Stage
1 does not use SAM2 or post-hoc mask projection. That was an earlier, abandoned
approach (Roadmap v1) and must not appear in the paper as the current method.**

**Stage 1 — Semantic Gaussian Splatting (GS³LAM)**
Three concurrent threads: a semantic thread (DEVA, a SAM2-like video-object
segmentation model, producing a 2D semantic map per frame), a tracking thread
(recovers camera pose per frame), and a mapping thread that minimizes a combined
photometric + geometric + semantic loss, so splat boundaries align with object
boundaries *during* optimization. Semantics participate in the mapping loss directly —
there is no separate mask-projection or ray-casting step. This is a deliberate
improvement over an earlier abandoned pipeline (v1: SplaTAM + SAM2 + post-hoc ray
casting), which was dropped due to noisy sensor data and compounding pipeline errors.

**Stage 2 — VLM Safety Assessment via Hero Frames**
For each semantically labeled object, a single "hero frame" (the camera pose
maximizing the object's on-screen pixel area) is selected, background-suppressed, and
sent to a VLM (Gemini) with a safety-auditor prompt. The VLM returns a continuous
[0,1] safety scalar, grafted onto the object's splats. This is queried once per object,
not once per frame, avoiding the latency of frame-by-frame VLM approaches.

**Stage 3 — CBF Integration**
Per-splat safety scores modulate a Control Barrier Function built on Tscholl et al.'s
analytic collision-cone formulation. Two candidate weighting strategies are
implemented and swappable, **neither yet experimentally validated against the other**:
a multiplicative scaling of the CBF's class-K gain (affects approach speed only), and
a covariance-inflation approach (effectively widens the splat's geometric boundary in
proportion to its hazard score, intended to affect lateral routing, not just speed).

**Platform:** TurtleBot4 (targeted; no real deployment has occurred yet).

---

## Core Claim (The Paper's Thesis)

Standard robot navigation treats obstacles as geometry — anything solid is avoided
equally. This system treats obstacles as semantics — a fragile vase and a concrete
pillar at the same distance should generate different navigation constraints because a
VLM can distinguish what they are.

**The claim to prove — not yet established, this is the target, not a finding:**
semantically-weighted CBF constraints produce meaningfully different (and
arguably better-calibrated) navigation behavior compared to a geometry-only baseline.

---

## Paper Target & Timeline

- **Primary target:** ICRA 2027. As of this writing, the official CFP page states
  schedules will be "updated soon" — the deadline is not yet publicly posted. The last
  three years' ICRA deadlines have landed around mid-September, not mid-October —
  treat any specific date as an estimate to reverify closer to the time, not a fixed
  planning anchor.
- **Secondary target:** a robotics-relevant NeurIPS 2026 workshop (typical window
  August–September 2026) — check the specific workshop's own CFP once posted; there is
  no single universal NeurIPS workshop deadline.
- **Venue-fit consideration for the September conversation with Dr. Yel:** a full ICRA
  paper typically expects real-hardware validation. If real experiments don't
  materialize with enough runway before whichever deadline lands, a workshop or
  preliminary-results track may be the more honest and achievable target. This is
  worth deciding explicitly in September, not assumed now.
- **Draft goal this summer:** every writing-only section (Related Work, Method,
  Introduction, Experimental Protocol) finished; synthetic-validation results written
  up honestly as synthetic; VLM-consistency experiment run and written up if completed
  in time.
- **Review:** Dr. Esen Yel (advisor, RISL, RPI) reviews in September 2026.
- **Lab/GPU access:** resumes fall semester 2026.

---

## This Summer's Remaining Experimental Work (No GPU Needed)

These are real experiments, not writing tasks, and neither needs a GPU, TurtleBot, or
real Stage 1/2 output. Prioritize these alongside the writing, not after it — the first
one in particular determines how strong the paper's empirical foundation is.

1. **VLM safety-score consistency check — done (2026-07-09).** Ran the existing
   safety-auditor prompt (`vlm_safety_score.py`'s `query_vlm_safety()`, reused
   verbatim, no rewriting) against 10 stock object images spanning an intended safety
   gradient (wall, heavy bookshelf, cushion, yoga mat, wooden chair, potted plant,
   vase, glass, cable, knife — see `assets/vlm_consistency/images/`), 5 queries each
   at temperature=0.2 (matching production's exact call config), via a standalone
   script (`vlm_consistency_check.py`, does not touch `src/cbf/`). A second VLM was
   skipped — no `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` configured this session; doing so
   needs one of those keys plus the corresponding SDK.
   **Finding:** within-object variance was zero for 9 of 10 objects — identical score
   across all 5 runs each. The sole exception was the object intended as the most
   obviously-safe baseline (a plain wall), which flipped bimodally between exactly 0.0
   and 1.0 across runs (mean 0.4, std 0.55) rather than settling near a continuous
   in-between value. This is the *opposite* of the expected pattern: intuitively
   ambiguous mid-safety objects (wooden chair, potted plant) were perfectly stable,
   while the clearest-cut "safe" case was the only unstable one — plausibly because
   the prompt's anchor ("safe to drive on/flat solid ground") makes a vertical wall
   categorically ambiguous (it isn't literally drivable "ground") in a way a chair or
   plant photo isn't. Worth folding into Discussion/Limitations: low temperature
   (0.2, matching production) also mechanically suppresses variance, so this doesn't
   rule out instability at higher temperature — untested here since the goal was
   testing the existing config, not sweeping it.
   Separately: `vlm_safety_score.py`'s pinned `model='gemini-1.5-flash'` is fully
   deprecated and 404s on every call as of this session — a real, previously-unknown
   Stage 2 bug, independent of the finding above (see Gap Tracking).
   Full per-object raw data: `assets/vlm_consistency/results.json`.
2. **`COV_INFLATE` disambiguation.** Re-run the synthetic 3-mode comparison with (a) a
   wider corridor (background splats moved further from the hazard) and (b) a sweep
   over `k_alpha_base`, to determine whether the previously observed near-halt
   behavior is a legitimate "no safe corridor" result or a fixed-gain artifact. Also
   worth checking whether the QP is repeatedly hitting infeasibility and falling back
   to max-braking — Tscholl et al.'s own paper notes their filter can go infeasible
   near obstacles with insufficient control authority, which may be exactly what's
   happening here.

Both of these produce content that upgrades "Results" from purely-synthetic to
genuinely informative, without needing anything blocked by lab access.

---

## What the Draft Needs to Cover

### Abstract (~150 words)
Problem (geometry-only navigation is semantically blind), approach (VLM safety
scoring + CBF integration over 3DGS), and status — this summer, be precise that the
implementation and its geometric correctness are validated, and real-world navigation
results are a target, not yet a finding. Do not overstate.

### 1. Introduction
State contributions precisely:
1. A method for assigning per-splat semantic safety scores via VLM, using a
   hero-frame mechanism that avoids per-frame query cost.
2. Integration of those scores into an analytic collision-cone CBF (built on Tscholl
   et al.), with two candidate weighting formulations.
3. **Do not claim a TurtleBot4 real-world demonstration** — that hasn't happened.
   Frame this as the intended validation target instead.
Do NOT claim to have built GS³LAM.

### 2. Related Work
Four areas: 3DGS for robotics; semantic scene understanding / VLMs in robotics;
Control Barrier Functions for navigation; VLMs for robot perception. Tscholl et al. is
the closest related work — be explicit that the contribution is orthogonal (they
handle the geometric CBF formulation; this project handles semantic scoring on top of
it). See "Verified Related-Work Candidates" below for a pre-checked starting list.

### 3. Method
Follow the corrected architecture above. Include: how DEVA's 2D semantic signal
participates in GS³LAM's joint optimization loss, how hero frames are selected and
queried, how safety scores modulate the CBF (both candidate formulations, clearly
marked as unvalidated alternatives, not a decided design), and the exact math
(Eq 8/9/13/17 and the Minkowski inflation, per `ARCHITECTURE.md` §2.3).

### 4. Experiments (Protocol)
Platform (TurtleBot4, targeted), scene configuration, baseline (`SemanticMode.NONE`,
geometry-only), metrics (collision severity, near-miss frequency, path efficiency —
already implemented). Write this as a precise protocol. Be explicit about what has and
hasn't been run yet.

### 5. Results
- Synthetic validation: the collision-cone math verified against an independent
  calculation, `ALPHA_SCALE` behaving as theoretically predicted. Framed explicitly as
  implementation validation, not a navigation result.
- VLM consistency check, if completed by draft time.
- Real-world/real-robot results: explicitly marked as pending fall experiments, not
  included as findings.

### 6. Discussion
What the synthetic results show about the implementation's correctness. Limitations:
VLM latency, score calibration, the still-open `COV_INFLATE` question, scene-specific
generalization, and — honestly — that real-world validation is the paper's main
remaining gap. Flag this directly; it strengthens credibility, it doesn't weaken it.

### 7. Conclusion
Restate the contribution as what's built and verified so far, not as a demonstrated
navigation result. Point to fall's real-robot experiments as the next milestone.

---

## Gap Tracking (Experiments to Flag for Fall)

This is now the primary home for anything not achievable this summer. Keep it
current while writing — anything you *wish* existed but doesn't goes here, to become
the first conversation with Dr. Yel in September.

- Real Stage 1 (GS³LAM) training run on Replica, producing a real `gsplat.ply` /
  `classifier.pth` — nothing downstream can be real without this.
- Dynamic verification of the Stage 2 classifier fix (the regression test exists but
  has never been run — needs a CUDA-capable machine).
- Real Phase B evaluation: does semantic weighting change outcomes on real, non-
  synthetic hazards?
- Ablation: does VLM score granularity matter (binary vs. continuous)?
- Baseline comparison on a real scene: `NONE` vs. `ALPHA_SCALE` vs. `COV_INFLATE`.
- Generalization across multiple distinct room configurations.
- End-to-end VLM safety-scoring latency, measured for real.
- Real TurtleBot4 deployment.
- Measuring the real `room0` scene's bounding box (spatial-filter defaults are
  currently placeholders) and confirming the actual TurtleBot4 footprint radius
  (currently a placeholder, 0.16).
- Real Replica-derived hero-frame version of the VLM consistency check — this
  summer's run (2026-07-09) used stock-photo proxies since no local Replica data
  existed in this checkout, not real background-suppressed hero-frame crops.
- `vlm_safety_score.py`'s hardcoded `model='gemini-1.5-flash'` is fully deprecated
  and 404s on every call as of 2026-07-09 (discovered while running the consistency
  check above) — needs to be repointed at a currently-served model before any real
  Stage 2 run. Note also that whichever model replaces it may need
  `thinking_config=types.ThinkingConfig(thinking_budget=0)` and an explicit
  `max_output_tokens` set — `gemini-flash-latest` silently truncated JSON output
  without these, which `vlm_consistency_check.py` had to work around.

---

## Verified Related-Work Candidates

Spot-checked against arXiv directly (correct authors/dates confirmed), safe to build
citations from without re-verifying from scratch:

- Tscholl, Nakka & Gunter, "Analytic Collision Cone Barrier Functions on 3DGS,"
  arXiv:2509.14421 (2025) — the Stage 3 geometric primitive.
- Chen et al., "SAFER-Splat," arXiv:2409.09868 (2024).
- Li, Zhang, Wang & Shen, "GS3LAM: Gaussian Semantic Splatting SLAM," Proc. 32nd ACM
  Int. Conf. Multimedia (MM '24), Melbourne, VIC, Australia, 2024, pp. 3019-3027.
- Sermanet et al., "Generating Robot Constitutions & Benchmarks for Semantic Safety"
  (ASIMOV benchmark), CoRL (2025) — candidate for validating VLM safety-score accuracy
  against ground truth.
- Splatblox (Chopra et al., arXiv:2511.18525) — traversability/ESDF from 3DGS.
- GWM (Lu et al., ICCV 2025, arXiv:2508.17600) — Gaussian World Models, relevant to
  Future Work / predictive safety framing.
- GSFF-SLAM (Lu et al., arXiv:2504.19409) — decoupled semantic Gaussian SLAM.
- EAMP (Huang et al., arXiv:2606.25629), VISA (Xian et al.) — VLM distillation for
  latency reduction, relevant to Discussion/Future Work.

Full literature review lives outside this repo — ask Aashrut for the doc for anything
needing deeper context.

---

## Tone and Framing Guidance

- Precise and conservative — only claim what's actually verified or run.
- Distinguish clearly, every time: foundation (GS³LAM, DEVA, Gemini, Tscholl's CBF) vs.
  contribution (the semantic scoring + grafting + weighting layer on top).
- Cite heavily in Related Work.
- Never describe a synthetic sanity check as a "result" without the word "synthetic"
  attached. Never imply real-robot data exists.
- If a section would require inventing a number, a scene description, or a robot
  behavior that hasn't actually happened, stop and write it as a Gap Tracking item
  instead.

---

## Key People

- **Dr. Esen Yel** — research advisor, RISL, RPI. Reviews draft in September. Did
  postdoc at Stanford SISL. Target ICRA submission co-authored with her.
- **Aashrut Jain** — junior, RPI Computer Systems Engineering. Built the VLM + CBF
  pipeline on top of GS³LAM. TurtleBot4 is the target platform.

---

## What to Ask Claude to Do

Good tasks for this session:
- Draft any individual section given the corrected context above.
- Suggest additional related-work citations (verify any not already in the list above).
- Help formulate/typeset the CBF safety-weighting math (already derived — see
  `ARCHITECTURE.md` §2.3 — this is a presentation task, not a derivation task).
- Review and tighten drafted prose.
- Generate figure/diagram descriptions for the pipeline and the CBF formulation.
- Help design and analyze the VLM-consistency check.
- Maintain the Gap Tracking list as sections get drafted.

Do NOT ask Claude to:
- Invent experimental results, scenes, robot behaviors, or numbers.
- Describe any section's content as a completed result if it's synthetic-only or
  not yet run — mark it explicitly instead.
- Describe GS³LAM, DEVA, or Tscholl's CBF as this project's original contribution.
- Claim a TurtleBot4 real-world demonstration has occurred.
