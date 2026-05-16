# rad-se: Reproducing RAD + SISA, Then Optimizing

Combined reproduction and improvement track for two papers, using the in-house
Structural Entropy (SE) toolkit:

1. **RAD** — Laskin, Lee, Stooke, Pinto, Abbeel, Srinivas. *Reinforcement Learning
   with Augmented Data.* NeurIPS 2020. https://github.com/MishaLaskin/rad
2. **SISA** — Zeng, Peng, Li, Liu, He, Yu. *Hierarchical State Abstraction Based on
   Structural Information Principles.* IJCAI 2023.

The previous in-workspace attempt is [reimplementrad](../../reimplementrad/) which
ported RAD into a clean one-file PyTorch script and added a v1 "SE intrinsic
reward" head. That work showed a *marginal* +1.7% reward on cartpole-swingup
(861.58 → 876.01 @190k) at **~2× wall time** — not a credible SISA reproduction.
We are skeptical that v1 captured what SISA actually does. This project starts
again with the explicit goals below.

## Goals

1. **Reproduce RAD** on at least 3 DMC pixel tasks (cartpole-swingup,
   acrobot-swingup, cheetah-run) with seeds {23, 42, 7} and match the paper's
   reported eval reward within stated noise.
2. **Reproduce SISA** as actually described in the IJCAI-23 paper: SAC + shared
   pixel encoder + SI pretrain (inverse / contrastive / smoothness) + SI
   finetune (clustering KL) + SI abstract loss (transition / action / reward
   graphs), with the staged schedule.
3. **Validate vs. ablation** — run RAD-only, SISA(full), and SISA(no-abstract)
   on the same hardware/seed grid; report sample efficiency and final return
   with 95% CIs.
4. **Optimize** — propose at most two targeted SI improvements over SISA
   (candidates: differentiable 2D SE over the abstract graphs using the
   `glass-jax` kernel, target-K aware partitioning, JIT-compiled graph
   construction). Only ship optimizations that beat SISA at the same wall-time
   budget.

## Non-goals

- We do **not** try to invent a brand-new structural-entropy RL framework. SIDM
  and SI2E already exist; see
  `/workspace/SEClust-paper/docs/related_work/RELATED_WORK_NOVELTY_AUDIT_STRUCTURAL_ENTROPY.md`
  ("Very high" overlap risk for generic "SE for RL abstraction" claims).
- No multi-task transfer experiments in the first pass.
- No procgen / openai-gym variants of RAD.

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
