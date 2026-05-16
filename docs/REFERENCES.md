# References

Pinned commits and citations.

## Papers

- Laskin, Lee, Stooke, Pinto, Abbeel, Srinivas. *Reinforcement Learning with
  Augmented Data.* NeurIPS 2020. arXiv:2004.14990.
- Zeng, Peng, Li, Liu, He, Yu. *Hierarchical State Abstraction Based on
  Structural Information Principles.* IJCAI 2023, pp. 4549–4557.
- Li, Pan. *Structural Information and Dynamical Complexity of Networks.*
  IEEE TIT, 2016.

## Original code repos

- **RAD (official, PyTorch)** — https://github.com/MishaLaskin/rad
  Verified-working commit on Vast.ai (per `rad-vastai-run`):
  `18d079e677398c70ff2eefefcc81d5a99662103d`.
  Sibling repos by the same authors: ProcGen variant
  https://github.com/pokaxpoka/rad_procgen ; OpenAI Gym variant
  https://github.com/pokaxpoka/rad_openaigym (both out of scope for us).
- **SISA (official)** — **not publicly located.**
  IJCAI-23 page https://www.ijcai.org/proceedings/2023/506 (PDF only, no code
  link). Authors are with Beihang ACT Lab (RingBDStack on GitHub); their org
  https://github.com/RingBDStack lists 106 repos but no `SISA` as of
  2026-05-16. Follow-up: search Xianghua Zeng's personal page and the SI2E /
  SIDM citing papers for a code release; otherwise reimplement strictly from
  the paper.

## Target runtime

- **MuJoCo Playground** — https://github.com/google-deepmind/mujoco_playground
  - DM Control Suite ported onto MuJoCo MJX (JAX).
  - Pixel observations via the **MJWarp Batch Renderer**.
  - PPO trainers in `learning/train_jax_ppo.py`; **no SAC trainer shipped** —
    we will adapt Brax SAC or sbx.
  - Reproducibility note: set `JAX_DEFAULT_MATMUL_PRECISION=highest` on Ampere
    GPUs to avoid TF32 silently lowering precision.
- **MuJoCo MJX** — https://github.com/google-deepmind/mujoco/tree/main/mjx
- **Brax (SAC reference, JAX)** — https://github.com/google/brax
- **denisyarats/dmc2gym** — only as a cross-check against the PyTorch baseline.

## In-workspace prior work

- One-file PyTorch RAD port (canonical baseline number):
  `/workspace/reimplementrad/implementations/rad_sac_dmc_pixel.py`.
- JAX SE kernel (reuse for SISA encoding-tree losses):
  `/workspace/glass-jax/src/glass/objectives/structural_entropy.py`.

## Related prior in-workspace work

- `/workspace/SEClust-paper/docs/related_work/RELATED_WORK_NOVELTY_AUDIT_STRUCTURAL_ENTROPY.md`
  — novelty audit for SE-for-RL claims (SIDM, SI2E, SISA overlap).
- `/workspace/se-research-projects/se-rl-transition/` — sibling project on SE
  over RL transition matrices.
- `/workspace/se-research-projects/se-when-tracks-labels/` — sibling project
  with a SIDM writeup we should cross-check.
