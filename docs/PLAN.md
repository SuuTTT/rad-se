# rad-se Plan

## Milestones

### M0 — Reference materials (local)
- Pull RAD paper PDF + SISA paper PDF into `references/`.
- Pull MishaLaskin/rad at the commit verified to run on Vast.ai:
  `18d079e677398c70ff2eefefcc81d5a99662103d`.
- Copy `reimplementrad/implementations/rad_sac_dmc_pixel.py` as the
  starting point for `src/rad_se/baseline_rad.py` (re-license-clean, drop the
  v1 SE hash-graph reward — that is not SISA).

### M1 — RAD baseline reproduction
- Tasks: `cartpole swingup` (action_repeat=8), `acrobot swingup`
  (action_repeat=4), `cheetah run` (action_repeat=4).
- Seeds: {23, 42, 7}. Train steps: 200_000. Eval every 10k, 10 episodes.
- Hardware target: RTX 3060 / 4060 (cheap Vast.ai tiers). The acrobot run on
  3060 took 4.9h wall, ~$0.27.
- Acceptance: cartpole swingup eval@190k ≥ 800 (paper ≈ 840–870, our prior
  one-file port hit **861.58**).

### M2 — SISA faithful reimplementation
- Shared pixel encoder (RAD encoder), SAC actor + twin critic.
- **SI pretrain** stage: inverse-dynamics loss + contrastive smoothness +
  state-transition smoothness. Run every encoder update for a warm-up window.
- **SI finetune** stage: cluster-assignment KL between encoder embeddings and
  a target partition (target updated on a slow schedule).
- **SI abstract** stage: build batch-local partition-level transition / action
  / reward graphs; compute SE / cut-objective over them; gradient flows
  through the soft-assignment matrix `S`.
- Schedule: pretrain dense → finetune mid-training → abstract late, matching
  the SISA paper.
- Use the JAX 2D SE kernel in `/workspace/glass-jax/src/glass/objectives/structural_entropy.py`
  if we port the encoder to JAX; otherwise reimplement the same formula in
  PyTorch. Either way the formula is fixed: `H^2(G)` from Li & Pan 2016.

### M3 — Comparison grid
- Methods: `rad`, `sisa_full`, `sisa_no_abstract`, `sisa_no_pretrain`.
- Same 3 tasks × 3 seeds. Single Vast.ai launch batch per method/task. Total
  compute target ≤ $20 (4 methods × 3 tasks × 3 seeds × ~5h × ~$0.06/h ≈
  $10.8 plus overhead; gate at $25).
- Report: mean ± 95% CI at {100k, 150k, 200k}, sample efficiency = first step
  past 80% of asymptote, wall time, peak GPU mem.

### M4 — Optimization probe (only if M3 passes)
- Probe 1: replace SISA abstract SE with differentiable 2D SE from the
  glass-jax kernel and check if gradients are cleaner.
- Probe 2: JIT-compile graph construction; profile to confirm SI losses are
  the bottleneck before optimizing them.
- Ship only the probe that beats SISA at equal wall time on cartpole+acrobot.

## Hardware / cost budget

| Stage | $ budget | Wall | Notes |
| --- | --- | --- | --- |
| M1 RAD repro (9 runs) | $5 | ~45 GPU-h | RTX 3060 @ ~$0.055/h |
| M2 SISA dev (smoke) | $2 | ~10 GPU-h | small batches, short steps |
| M3 Comparison grid | $15 | ~135 GPU-h | hard ceiling $25 |
| M4 Optimization | $5 | ~30 GPU-h | only if M3 green |

Stop at any milestone where SISA(full) does not reproducibly beat RAD on
≥2/3 tasks — escalate to user before continuing.

## Success criteria

- RAD reproduction: ≥2/3 tasks within paper noise.
- SISA reproduction: SISA(full) statistically beats RAD (paired by seed) on
  ≥2/3 tasks at 200k steps, p<0.05.
- Optimization: strict Pareto improvement over SISA at equal wall time.

## Out-of-scope (defer)

- ProcGen, openai-gym Mujoco classic, robotic manipulation domains.
- DrQ-v2 / DreamerV3 comparisons.
- Hyperparameter sweeps beyond the SI loss weights and schedule.
