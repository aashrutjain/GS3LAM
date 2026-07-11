# CLAUDE.md

This file is loaded into every Claude Code session in this repo. Keep it lean — it's
context budget, not documentation. Full technical detail, design rationale, and open
research questions live in `ARCHITECTURE.md`. Read that file before making any change
to pipeline logic, the CBF formulation, or the splat schema.

## What this project is

A semantic Gaussian Field pipeline for risk-aware robot navigation (RISL, RPI).
VLM-derived per-object safety scores are grafted onto a 3D Gaussian Splatting map,
with the eventual goal of driving a Control Barrier Function on a TurtleBot4.

**Current milestone:** Stage 3 — costmap + CBF integration, now mostly built and
smoke-tested on a synthetic scene (no real safety_gsplat.ply exists in this checkout
yet). The `src/cbf/` library, `costmap_cbf.py`, and `eval_cbf_modes.py` (the 3-mode
comparison harness) all exist and run; a pinned QP-solver choice and a real unit test
suite do not yet. See `PROGRESS.md` for exact status, including a real dtype bug found
and fixed in the solver path, and `ARCHITECTURE.md` §2.3 for the design. Stages 1
(GS3LAM semantic splatting) and 2 (hero-frame VLM safety scoring) are functional and
should not need structural changes — though see §2.3/`PROGRESS.md` for two real Stage
2 bugs found while building Stage 3 (not yet fixed).

## Environment

- Conda env pinned to `cudatoolkit-dev=11.7.0` for PyTorch/GS3LAM compatibility.
  Don't upgrade without checking this first — it was pinned deliberately to resolve
  a C++ compiler conflict between legacy PyTorch and the host toolchain.
- Beyond `requirements.txt`, the README also requires a separate install step:
  `pip install submodules/gaussian-semantic-rasterization`. Easy to miss during setup.
- Local dev GPU is an A2000, VRAM-constrained. GS3LAM training runs at ~80% GPU
  capacity for hours — don't kick off a full retrain casually, and don't remove the
  rasterizer downsampling patches "for cleanliness."
- Gemini API key lives in `.env` (gitignored). Never hardcode it, print it, or log it.

## Commands

<!-- TODO(Aashrut): fill in Stage 1/2 entry points -->
- Train GS3LAM on Replica: `TODO`
- Run hero-frame selection + VLM safety scoring + splat augmentation (writes
  `safety_gsplat.ply`): `python vlm_safety_score.py` (has two known bugs — see
  `PROGRESS.md`)
- Stage 3 online single-step filter demo: `python costmap_cbf.py --config
  configs/cbf/room0_cbf.py --ply-path <path/to/safety_gsplat.ply>`
- Stage 3 CBF/costmap 3-mode eval: `python eval_cbf_modes.py --config
  configs/cbf/room0_cbf.py --ply-path <path/to/safety_gsplat.ply> --phase-a` (drop
  `--phase-a` once Stage 2 bug in `PROGRESS.md` is fixed and scores are meaningful)

## Data contracts

Don't change these shapes without updating `ARCHITECTURE.md` §2 to match.

- `gsplat.ply` — 3D Gaussian splats: geometry, color, opacity (GS3LAM output)
- `params.npz` — `w2c` (camera poses per frame), `obj_dc` (16-D semantic vector per splat)
- `safety_gsplat.ply` — `gsplat.ply` + a grafted scalar `safety` column in [0, 1]

## Rules

- Research codebase — prioritize correctness and reproducibility over generality or
  premature abstraction.
- Don't touch or delete the Roadmap v1 code (SplaTAM + SAM2 + ray casting). It's kept
  intentionally for the documented v1→v2 comparison in the write-up.
- Any change to the safety-score scale, the splat schema, or the CBF formulation is a
  research decision, not a coding one. Flag it and explain the tradeoff — don't just
  implement it silently.
- New CBF work builds on the Analytic Collision Cone formulation (Tscholl et al.,
  arXiv:2509.14421) as the geometric primitive, not a SAFER-Splat-style distance CBF.
  See `ARCHITECTURE.md` §2.3 for the exact math and module design, and its open-question
  list for how the CBF combines with the semantic safety score — that combination is
  unresolved, not a settled design (both candidate strategies are implemented as
  swappable modes in `src/cbf/`, to be compared experimentally, not decided a priori).
- Don't pull in new SLAM backbones, VLM distillation, or world-model components without
  being asked. They're logged as future work in `ARCHITECTURE.md` §5, not current scope.
