# rad-se Plan

## Stack decision

- **Framework: JAX.** Flax for nets (default), Optax for opt, Brax SAC as the
  RL anchor.
- **Envs: MuJoCo Playground.** `mujoco_playground.dm_control_suite` for
  cartpole / acrobot / cheetah, with pixel observations via the MJWarp Batch
  Renderer. Visual gridworld for SISA Goal-2 will be built as a tiny custom
  env on top of the same renderer when we get to M2.
- **PyTorch path is frozen.** `reimplementrad` stays as the anchor for the
  PyTorch baseline numbers; we do not touch it.

## Milestones

### M0 — Reference materials
- Pull RAD paper + SISA paper PDFs into `references/`.
- Pull MishaLaskin/rad at commit `18d079e677398c70ff2eefefcc81d5a99662103d`
  into `references/rad/` as read-only reference (do not import).
- Pull the in-workspace one-file PyTorch port as a side-by-side reference (do
  not import) so the JAX port stays diff-able.

### M0.5 — Playground smoke (cheap, must pass before anything else)
- `pip install playground` (or install from source per the README).
- Run the `Vision Environments` tutorial colab equivalent locally for one
  cartpole-swingup episode at 84×84 pixel obs. Capture timing.
- Export `JAX_DEFAULT_MATMUL_PRECISION=highest` on Ampere/Ada.
- Stand up a Brax SAC reference and run it for 1k steps on cartpole-swingup
  pixel obs to confirm SAC + pixel encoder wire correctly with the renderer.

### M1 — RAD baseline reproduction on Playground
- Tasks: `CartpoleSwingup`, `AcrobotSwingup`, `CheetahRun` from
  `mujoco_playground.dm_control_suite`, pixel obs at 84×84, frame_stack=3.
- Seeds: {23, 42, 7}. Train steps: 200_000 env-steps (action-repeat matched
  to the PyTorch baseline: cartpole 8, acrobot 4, cheetah 4).
- Hardware: 1 GPU (target 4090 / 5090 cloud tier; Playground is GPU-batched
  so per-run cost should be lower than the 3060 PyTorch runs).
- Acceptance for `cartpole swingup`: eval@190k ≥ 800 on ≥2 of 3 seeds. Anchor
  numbers:
  - paper RAD ~840–870.
  - PyTorch one-file port (reimplementrad): **861.58**.
  - Note: Playground != classic dm_control bit-exactly. A ~5–10% gap is
    acceptable, document the gap, do not fudge.

### M2 — SISA faithful reimplementation
- Adaptive hierarchical state clustering → encoding tree T over minibatches.
- Per-non-root-node aggregation function (formula pending PDF read).
- Conditional structural entropy loss summed over non-root nodes, weighted by
  node volume.
- Encoder is shared between SAC heads and the SISA losses; gradients from
  conditional SE flow into the encoder only (not into actor/critic heads).
- Smoke first (5k steps, K_leaf=8, depth=2), then full (200k).

### M3 — Comparison grid
- Methods: `rad`, `sisa_full`, `sisa_flat_1d` (ablate tree),
  `sisa_no_aggregation` (ablate aggregation function).
- Same 3 Playground tasks × 3 seeds. One JAX process per task; methods can
  share a vmap over seeds where possible (large GPU mem permitting).
- Compute target ≤ $20 total. Hard ceiling $30.
- Report: mean ± 95% CI at {100k, 150k, 200k}, sample efficiency = first
  step past 80% of asymptote, wall time, peak GPU mem, paired-by-seed
  Wilcoxon vs. RAD.

### M4 — Optimization probe (only if M3 passes)
- Probe 1: drop in `glass-jax` differentiable 2D SE kernel for the
  conditional SE loss; check gradient quality and wall time.
- Probe 2: target-K aware partitioning at each tree depth.
- Ship only the probe that strictly Pareto-beats SISA at equal wall time on
  ≥2/3 tasks.
## Hardware / cost budget

| Stage | $ budget | Wall | Notes |
| --- | --- | --- | --- |
| M0.5 Playground smoke | $1 | ~1 GPU-h | one cloud GPU hour |
| M1 RAD repro on JAX (9 runs) | $5 | ~9 GPU-h | 4090/5090; JAX batched |
| M2 SISA dev | $3 | ~6 GPU-h | smoke + iteration |
| M3 Comparison grid (36 runs) | $15 | ~36 GPU-h | hard ceiling $30 |
| M4 Optimization | $5 | ~10 GPU-h | only if M3 green |

Stop at any milestone where SISA(full) does not reproducibly beat RAD on
≥2/3 tasks — escalate to user before continuing.

The Vast.ai 3060 path documented in `LESSONS.md` is a **fallback** if cloud
GPU access for JAX fails. We do not plan to rerun PyTorch RAD on Vast.ai.

## Success criteria

- RAD reproduction: ≥2/3 tasks within paper noise.
- SISA reproduction: SISA(full) statistically beats RAD (paired by seed) on
  ≥2/3 tasks at 200k steps, p<0.05.
- Optimization: strict Pareto improvement over SISA at equal wall time.

## Out-of-scope (defer)

- ProcGen, openai-gym Mujoco classic, robotic manipulation domains.
- DrQ-v2 / DreamerV3 comparisons.
- Hyperparameter sweeps beyond the SI loss weights and schedule.
