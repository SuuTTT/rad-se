# SISA reimplementation notes

Paper: Zeng et al. *Hierarchical State Abstraction Based on Structural
Information Principles*, IJCAI-23, pp. 4549–4557.

Existing in-workspace summary (writeup, not code) we should cross-check:
`/workspace/se-research-projects/se-when-tracks-labels/paper/suuttt.github.io/content/projects/sidm-in-practice-distill-style-guide.md`
sections 4.1–4.4.

## SISA in one sentence

Model-free SAC control (on top of RAD/CURL) with a **shared pixel encoder
regularized by SI-driven latent-space losses on partition-level transition,
action, and reward graphs**, scheduled in three phases.

## Architecture

- Encoder `f_θ`: same RAD pixel encoder (4 conv layers + linear).
- Actor + twin critic: standard SAC heads on top of `f_θ`'s output.
- Soft partition head `S_φ : ℝ^d → Δ^{K-1}`: produces a soft assignment of
  each latent to one of K partitions.

## SI losses

For a minibatch `{(s_t, a_t, r_t, s_{t+1})}`:

1. **Pretrain — inverse dynamics**: predict `a_t` from
   `(f_θ(s_t), f_θ(s_{t+1}))`. Pulls actionable features into the encoder.
2. **Pretrain — contrastive smoothness**: positive pair
   `(f_θ(s_t), f_θ(s_{t+1}))`, negatives from the batch. InfoNCE-style.
3. **Finetune — KL clustering**: `KL( S_φ(f_θ(s)) || target_S(s) )`, where
   `target_S` is a slowly-updated target distribution sharpened from the
   running soft assignments (DEC-style).
4. **Abstract — graph SE**: build, per minibatch, three K×K graphs
   - transition: `T_ij = Σ_t S_φ(s_t)_i · S_φ(s_{t+1})_j` (source-normalized).
   - action: edge weight from action-similarity per pair.
   - reward: edge weight from positive reward per pair.

   Compute a structural-entropy-style objective on each graph (the paper uses
   normalized cut / 1D SE; we will use the differentiable 2D SE in
   `glass-jax/src/glass/objectives/structural_entropy.py` as a cleaner
   substitute, but only if the 1D version reproduces first).

## Schedule

- Steps 0 → T₁: dense pretrain (inverse + contrastive), every encoder update.
- Steps T₁ → T₂: pretrain + finetune (KL clustering).
- Steps T₂ → T_end: finetune + abstract (graph SE).

Defaults to start from (subject to paper-check): T₁ = 10k, T₂ = 50k,
T_end = 200k.

## Loss weighting

`L_total = L_SAC + λ_inv · L_inv + λ_con · L_con + λ_kl · L_kl
           + λ_abs · (L_trans + L_act + L_rew)`

Initial guesses (paper hyperparams may differ — verify):
`λ_inv = 1.0, λ_con = 0.5, λ_kl = 0.1, λ_abs = 0.05`.

## What we explicitly do NOT do

- No pixel-hash state graph (the v1 reimplementrad mistake).
- No reward shaping with SE.
- No multi-level encoding tree until M4 optimization probes.

## Implementation outline

- `src/rad_se/baseline_rad.py` — fork from `reimplementrad/implementations/rad_sac_dmc_pixel.py`.
- `src/rad_se/sisa.py` — adds `SoftPartitionHead`, `SILosses`, `SIScheduler`.
- `src/rad_se/graphs.py` — batch-local construction of T/A/R graphs from
  `(S, a, r)`.
- `src/rad_se/se.py` — 1D SE (paper-faithful) and 2D SE (optimization probe);
  match the formulas in `/workspace/glass-jax/src/glass/objectives/structural_entropy.py`.
