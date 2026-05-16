# Changelog

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
