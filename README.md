# rad-se: Reproducing RAD + SISA, Then Optimizing

Combined reproduction and improvement track for two papers, retargeted onto
**JAX + MuJoCo Playground** for GPU-batched training with pixel observations:

1. **RAD** — Laskin, Lee, Stooke, Pinto, Abbeel, Srinivas. *Reinforcement Learning
   with Augmented Data.* NeurIPS 2020. PyTorch.
   Official: https://github.com/MishaLaskin/rad
2. **SISA** — Zeng, Peng, Li, Liu, He, Yu. *Hierarchical State Abstraction Based on
   Structural Information Principles.* IJCAI 2023, pp. 4549–4557.
   Official code: **not publicly located** (no link on IJCAI page; not in the
   RingBDStack GitHub org as of 2026-05-16). PDF only — see
   `docs/REFERENCES.md`.

The previous in-workspace attempt is [reimplementrad](../../reimplementrad/) which
ported RAD into a clean one-file PyTorch script and added a v1 "SE intrinsic
reward" head. That work showed a *marginal* +1.7% reward on cartpole-swingup
(861.58 → 876.01 @190k) at **~2× wall time** — not a credible SISA reproduction.
We are skeptical that v1 captured what SISA actually does. This project starts
again with the explicit goals below.

## Why JAX + MuJoCo Playground

- Playground is `dm_control` ported onto MuJoCo MJX (JAX) — same physics, GPU
  batched, ~100–1000× steps/s of the CPU pipeline RAD/SISA originally used.
- **Pixel observations are supported** via the MJWarp Batch Renderer
  (Playground README: *"Vision-based support available via the MJWarp Batch
  Renderer"*; there is also a `Vision Environments` tutorial colab).
- A single-GPU JAX run can replace the 3060 Vast.ai jobs of the prior repos and
  cut wall time per seed from ~5 h to well under an hour, which is what makes a
  paired-by-seed comparison grid affordable.
- SISA's encoding-tree partition + conditional SE losses are differentiable
  matrix ops — they JIT cleanly and are a natural fit for `jax.vmap` over the
  batch of envs.

## Goals

1. **Reproduce RAD on Playground** — port RAD SAC to JAX on top of MuJoCo
   Playground pixel envs (cartpole-balance/swingup, acrobot-swingup, cheetah-run)
   with seeds {23, 42, 7}. Anchor numbers against:
   - the original PyTorch RAD paper, and
   - the in-workspace one-file PyTorch port `reimplementrad`
     (cartpole-swingup eval@190k = **861.58**, our verified baseline).
2. **Reproduce SISA on Playground** — implement what the IJCAI-23 paper
   actually describes: **adaptive hierarchical state clustering** producing an
   **optimal encoding tree**, with a per-non-root-node aggregation function and
   **conditional structural entropy** loss, on top of the same shared encoder.
   Benchmarks: visual gridworld + the Playground continuous-control subset.
3. **Validate vs. ablation** — RAD, SISA(full), SISA(flat-1D-SE),
   SISA(no-encoding-tree). Same hardware/seed grid; sample efficiency and final
   return with 95% CIs, paired-by-seed Wilcoxon.
4. **Optimize** — at most two targeted improvements over SISA. Candidates:
   - reuse `glass-jax` differentiable 2D SE kernel for the encoding-tree loss,
   - target-K aware partitioning,
   - JIT-compiled graph construction.
   Ship only if it strictly Pareto-beats SISA at equal wall-time.

## Non-goals

- We do **not** try to invent a brand-new structural-entropy RL framework. SIDM
  and SI2E already exist; see
  `/workspace/SEClust-paper/docs/related_work/RELATED_WORK_NOVELTY_AUDIT_STRUCTURAL_ENTROPY.md`
  ("Very high" overlap risk for generic "SE for RL abstraction" claims).
- No multi-task transfer experiments in the first pass.
- No procgen / openai-gym variants of RAD.
- No port of the PyTorch baseline back to Vast.ai once the JAX baseline is
  green. PyTorch runs in `reimplementrad` stay frozen as historical anchors.

## Status

Bootstrapped. No code yet. See [docs/PLAN.md](docs/PLAN.md) for the milestone
plan and [docs/LESSONS.md](docs/LESSONS.md) for the consolidated lessons from
the three prior repos.

## Layout

| Path | Purpose |
| --- | --- |
| `docs/PLAN.md` | Milestones, experiment grid, success criteria. |
| `docs/LESSONS.md` | Concrete lessons from `rad-vastai-run`, `rad-vastai-run-2026-04-28T10:06:21+00:00`, `reimplementrad`. |
| `docs/SISA_NOTES.md` | What SISA actually does, vs. what the v1 attempt did. |
| `docs/RISKS.md` | Reproducibility / scope / cost risks and mitigations. |
| `configs/` | Hydra/YAML run configs per task / seed / method. |
| `scripts/` | Vast.ai launch + monitor scripts (reuse autosota-lite scheduler). |
| `src/rad_se/` | One-file RAD baseline and SISA agent (planned). |
| `experiments/` | Per-experiment env files + run scripts. |
| `references/` | Local copies of RAD/SISA papers + key SE references. |
