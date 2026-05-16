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
  PyTorch port hit 861.58 vs paper ~840–870 on cartpole. Document the gap
  rather than fudge it.
- **Playground != dm_control bit-exactly.** MJX uses MuJoCo physics but env
  specs, dt, and action repeat conventions can differ. A 5–10% gap from the
  PyTorch anchor on cartpole is acceptable; >15% needs investigation.
- **TF32 trap on Ampere/Ada.** Always set
  `JAX_DEFAULT_MATMUL_PRECISION=highest`; otherwise JAX uses TF32 and
  silently reduces precision, which destabilizes SAC critic updates.
- **SISA paper code is not public.** Reimplement strictly from the PDF; do
  not paraphrase blog posts. Cite formulas with paper line/equation numbers
  in code comments.
- DMC pixel rendering is still fragile; the MJWarp renderer is newer code
  than the PyTorch path, expect 1–2 days of integration friction.

## Cost

- Hard ceiling: $25 for M3. Estimate first via the scheduler `estimate`
  subcommand (see `rad-vastai-run/vastai_estimate.json`).
- One forgotten instance can blow the budget. Always pair launch with the
  monitor cleanup script.

## Code

- The reimplementrad SE-v1 path is **not** what SISA describes — do not
  start from it. Start fresh in JAX.
- We are choosing JAX. That commits us to maintaining a JAX + Playground +
  Brax SAC + (later) Flax encoder stack. If the M0.5 smoke does not pass
  cheaply, escalate before proceeding; do not fall back to PyTorch silently.
- No SAC trainer ships with Playground. Use Brax SAC as the reference and
  keep our SAC code as one file for diff-based review, following the
  PyTorch `reimplementrad` precedent.
