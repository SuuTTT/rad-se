# SISA reimplementation notes

Paper: Zeng et al. *Hierarchical State Abstraction Based on Structural
Information Principles*, IJCAI-23, pp. 4549–4557.

Existing in-workspace summary (writeup, not code) we should cross-check:
`/workspace/se-research-projects/se-when-tracks-labels/paper/suuttt.github.io/content/projects/sidm-in-practice-distill-style-guide.md`
sections 4.1–4.4.

## SISA in one sentence (corrected from the IJCAI-23 abstract)

Model-free SAC control with a shared pixel encoder, where an **unsupervised,
adaptive hierarchical state clustering** procedure produces an **optimal
encoding tree**, and a **conditional structural entropy** objective on each
**non-root tree node** regularizes the encoder while compensating for
sampling-induced information loss.

Key point our v1 attempt missed: SISA is **multi-level** (encoding tree with
non-root nodes), not a flat 1D SE on a single graph. The conditional SE term
is defined per non-root node, not over the whole graph.

Benchmarks reported by the paper:

- Visual gridworld domain.
- Six continuous-control benchmarks.
- Claimed gains: up to **+18.98 mean episode reward** and **+44.44% sample
  efficiency** vs. five SOTA state-abstraction baselines.

## Architecture

- Encoder `f_θ`: same RAD pixel encoder (4 conv layers + linear).
- Actor + twin critic: standard SAC heads on top of `f_θ`'s output.
- Soft partition head `S_φ : ℝ^d → Δ^{K-1}`: produces a soft assignment of
  each latent to one of K partitions.

## SI losses (best-effort reconstruction from the paper text)

The IJCAI abstract names two SISA mechanisms: an **aggregation function** on
non-root tree nodes, and a **conditional structural entropy** loss. The full
formulas live in the paper body — we will fill them in here once the PDF is
in `references/`. The reconstruction below is provisional and **must be
revised against the paper before coding M2**.

Provisional schedule:

1. Build batch-local soft assignments `S = softmax(g_φ(f_θ(s)))` of size
   `B × K_leaf`.
2. Run agglomerative grouping over the K_leaf prototypes to form an encoding
   tree T with non-root nodes `{α}`.
3. For each non-root node `α`, aggregate child latents via the SISA
   aggregation function (to be specified from the paper) and define a
   **conditional structural entropy** loss `H(α | parent(α))` on the partition
   graph induced by T.
4. Total encoder regularizer = sum over non-root α of `H(α | parent(α))`,
   weighted by node volume.

Notably absent from the abstract (so likely *not* central to SISA, despite
v1's framing): inverse dynamics, InfoNCE, DEC-style cluster KL, separate
transition / action / reward graphs. These were our guesses, not SISA. Treat
them as *optional extra losses* (the paper says SISA is "general" and can be
combined with existing representation-learning objectives) rather than core.

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
- No reward shaping with SE — SISA regularizes the encoder, not the reward.
- No skipping the multi-level encoding tree: per the abstract, the tree *is*
  the method. (The v1 attempt's "flat-tree-as-cheap-approximation" plan was
  wrong.)

## Implementation outline (JAX + Playground)

- `src/rad_se/envs.py` — thin Playground env wrapper exposing pixel obs via
  the MJWarp batch renderer; matches the cartpole/acrobot/cheetah names from
  the PyTorch baseline.
- `src/rad_se/encoder.py` — RAD pixel conv encoder in Flax / Equinox (to
  decide); reuses the layer config from
  `reimplementrad/implementations/rad_sac_dmc_pixel.py`.
- `src/rad_se/sac.py` — SAC actor / twin critic / target update / temperature.
  Anchor implementation = Brax SAC; we strip it to a single file like the
  PyTorch port.
- `src/rad_se/rad_aug.py` — random crop and flip augmentations on batched
  pixel tensors (pure-JAX).
- `src/rad_se/baseline_rad.py` — RAD training script (no SISA). This is the
  Goal-1 deliverable.
- `src/rad_se/encoding_tree.py` — adaptive hierarchical clustering producing
  the encoding tree T. (M2.)
- `src/rad_se/conditional_se.py` — conditional structural entropy loss per
  non-root node. (M2.)
- `src/rad_se/sisa.py` — composes encoder + SAC + encoding-tree losses.

Reuse policy: do NOT inherit `reimplementrad/implementations/rad_sac_dmc_pixel_se_v1.py`
(see `docs/LESSONS.md`). The flat 1D hashed-state code path is the wrong shape
for SISA.
