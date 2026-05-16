# Changelog

## 2026-05-16 (later)

- Retargeted the project onto **JAX + MuJoCo Playground**.
  - Confirmed pixel obs supported via the MJWarp Batch Renderer.
  - Noted: no SAC trainer in Playground; will adapt Brax SAC.
  - Reproducibility: must set `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- Located original code repos:
  - RAD: https://github.com/MishaLaskin/rad (commit
    `18d079e677398c70ff2eefefcc81d5a99662103d` verified to run).
  - SISA: official code **not located** — IJCAI page has PDF only, not in
    RingBDStack GitHub org. Will reimplement strictly from the paper.
- Corrected `SISA_NOTES.md` against the IJCAI-23 abstract: SISA uses an
  **adaptive hierarchical encoding tree** with a **per-non-root-node
  aggregation function** and **conditional structural entropy**. The v1 SISA
  notes in this repo (and the v1 reimplementrad code) had this wrong — they
  described inverse dynamics + InfoNCE + DEC + flat 1D SE, which is *not*
  what the paper describes.
- Inserted M0.5 Playground smoke milestone, updated budget table for JAX.

## 2026-05-16

- Bootstrapped `rad-se` under `se-research-projects/`.
- Drafted `docs/PLAN.md`, `docs/LESSONS.md`, `docs/SISA_NOTES.md`,
  `docs/RISKS.md`, `REPRODUCE.md`.
- Consolidated lessons from:
  - `/workspace/rad-vastai-run/` (full 200k acrobot baseline on Vast.ai)
  - `/workspace/rad-vastai-run-2026-04-28T10:06:21+00:00/` (smoke + dep stack)
  - `/workspace/reimplementrad/` (one-file port + v1 SE attempt)
- Decision: do not inherit `reimplementrad`'s v1 SE intrinsic-reward code.
  v1 produced +1.7% reward at 2× wall time on a single seed and is not what
  SISA describes. Start SISA fresh from the clean RAD one-file baseline.
- No source code committed yet.
