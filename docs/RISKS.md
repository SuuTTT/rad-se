# Risks

## Scientific

- **SISA may not actually beat RAD** at our compute budget on the tasks we
  pick. If so, we report the negative result and stop at M3 — do not invent
  a new method post-hoc.
- **Novelty overlap with SIDM / SI2E.** Per
  `/workspace/SEClust-paper/docs/related_work/RELATED_WORK_NOVELTY_AUDIT_STRUCTURAL_ENTROPY.md`,
  generic "SE for RL abstraction" claims are very high risk. Any
  optimization probe must be framed as *engineering* over SISA, not as a
  new method.
- **Seed variance dominates** at 3 seeds; a "win" of <5% may be noise.
  Require paired-by-seed Wilcoxon, not just means.

## Reproducibility

- RAD's official numbers are hard to reproduce exactly; our prior one-file
  port hit 861.58 vs paper ~840–870 on cartpole. Document the gap rather
  than fudge it.
- DMC pixel rendering is fragile across GPU generations. Lock onto RTX 3060
  / 4060 tier and document the exact apt+pip stack from
  `docs/LESSONS.md`.

## Cost

- Hard ceiling: $25 for M3. Estimate first via the scheduler `estimate`
  subcommand (see `rad-vastai-run/vastai_estimate.json`).
- One forgotten instance can blow the budget. Always pair launch with the
  monitor cleanup script.

## Code

- The reimplementrad SE-v1 path is **not** what SISA describes — do not
  start from it. Start from `rad_sac_dmc_pixel.py` (baseline only) and
  write SISA fresh.
- Do not introduce JAX unless we are also willing to maintain the JAX
  build on Vast.ai. PyTorch-first; JAX SE kernel can be ported by hand if
  needed.
