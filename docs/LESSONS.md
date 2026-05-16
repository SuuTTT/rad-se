# Lessons from prior repos

Three prior repos inform this project. Concrete, actionable lessons only.

## Source repos

1. `/workspace/rad-vastai-run/` — first **full 200k** RAD baseline on Vast.ai.
   - Task: `acrobot swingup`, pixel obs, seed 23, batch 128, action_repeat 4.
   - Instance: RTX 3060, 12GB, Vietnam, $0.0549/h.
   - Wall: 17_536 s training; total 17_618 s. Cost ~$0.27.
   - Final evals (from `rad_run.log`): 170k=3.65, 180k=11.08, 190k=13.42. These
     are **DMC normalized returns**, not the 800+ of cartpole. Acrobot is
     harder; the RAD paper itself reports modest scores here. Check paper
     before treating as failure.

2. `/workspace/rad-vastai-run-2026-04-28T10:06:21+00:00/` — earlier smoke run.
   - 0.25h, 4 train steps. Verified the dependency stack works. Eval `ER:
     20.7950` on a 1-episode cartpole pixel smoke.
   - Catalogued every compat fix needed for the modern PyTorch image.

3. `/workspace/reimplementrad/` — clean one-file RAD port + a v1 SE attempt.
   - `implementations/rad_sac_dmc_pixel.py` reproduces RAD cartpole at
     **861.58 eval@190k** (W&B run `kq2uw5po`, 8059 s on RTX 5060 Ti). This
     is the canonical baseline we will fork.
   - `implementations/rad_sac_dmc_pixel_se_v1.py` adds an online pixel-hash
     transition-graph intrinsic reward. cartpole @190k: **876.01** vs 861.58.
     Wall time **16_692 s** vs 8_059 s. ⇒ **+1.7% reward at 2.07× compute.**
     This is *not* a credible win and is *not* SISA — it is hashed-state
     curiosity. See "What v1 got wrong" below.

## Environment / dependency lessons

These are all reproducible, observed failure→fix pairs:

- RAD's `conda_env.yml` is stale. On `pytorch/pytorch:2.5.1-cuda12.4` we
  must install: `numpy<2`, `gym==0.23.1`, `dm_control`, `mujoco`,
  `dmc2gym` (from GitHub), `termcolor`, `tabulate`, `imageio`,
  `imageio-ffmpeg`, `scikit-image`, `matplotlib`, `tb-nightly`, `patchelf`.
- `matplotlib` is imported by RAD's `data_augs.py` but missing from the
  env file. **Install it explicitly or training crashes at import.**
- Patch installed `dmc2gym/wrappers.py`: replace `np.int` with `int`
  (numpy ≥1.20 removed the alias). The exact one-liner is in
  `/workspace/rad-vastai-run/rad_remote_job.sh`.
- MuJoCo rendering on headless Vast.ai needs `MUJOCO_GL=osmesa`,
  `PYOPENGL_PLATFORM=osmesa`, and apt packages
  `libgl1 libgl1-mesa-dri libglfw3 libglvnd0 libglx-mesa0 libglx0
  libegl-mesa0 libegl1 libgles2 libosmesa6`.
- Always do a `dmc2gym.make(..., from_pixels=True)` **pixel preflight**
  before launching the 200k training. The remote job script we will reuse
  falls back to cartpole if acrobot pixels fail to render.
- On 50-series GPUs (RTX 5060/5090, sm_120), stock pytorch 2.5.1 is wrong;
  use pytorch nightly cu13x. The `reimplementrad` repo includes a working
  `rad_job_cmd_50series_cu132.sh`.

## Vast.ai scheduler lessons

- Use `/workspace/autosota-lite/plugins/autosota-lite/skills/autosota-vastai-scheduler/scripts/vastai_scheduler.py`
  with `--avoid-countries CN,US`, `--order dph`, on-demand. RTX 3060 in
  VN/SG was the cheapest reliable option.
- The bundled script had **unescaped f-string braces** in its generated
  on-start bash; fix locally before launch (rad-vastai-run patched it).
- The launch JSON redacts API keys — preserve that behavior.
- Always pass `--save-json` to capture instance details for the run log.
- Always pair launch with `monitor_vastai_instance.sh` to guarantee
  cleanup; verify post-cleanup with `vastai show instances --raw`.
- 200k RAD steps cost ~$0.27 per run on RTX 3060. Plan accordingly.

## What v1 SE-RAD got wrong (so we don't repeat it)

`reimplementrad/implementations/rad_sac_dmc_pixel_se_v1.py` claims SE
integration but is **not** SISA:

- It hashes pixel frames into discrete macro-states via blake2b. SISA
  uses **learned soft assignments** from the encoder, not pixel hashes.
- It computes only **1D structural entropy** `H_1(G)` on a state-only
  graph. SISA computes **partition-level SE-style objectives** on three
  graphs (transition / action / reward), with gradient flow through
  the soft-assignment matrix `S` into the encoder.
- It uses SE as an **intrinsic reward shaping term** added before replay
  insertion. SISA uses SE as a **representation regularizer** on the
  encoder gradient path, not as a reward.
- It has no pretrain / finetune / abstract staged curriculum.
- The reward shaping doubled wall time but the gain (1.7%) is within
  seed variance for a single seed. No CI was reported.

Implication for rad-se: do not reuse v1 SE code. Treat the one-file RAD
baseline as the only safe inheritance.

## Reproducibility lessons

- Seed everything: numpy, torch, env, replay sampler.
- Log to W&B with explicit `group=<method>__<task>__seed<n>` so the
  comparison grid post-processes cleanly.
- Always save `args.json`, `metrics.jsonl`, and `*_eval_scores.npy` so
  results survive instance teardown.
- Keep one file per method (RAD, SISA) for diff-based code review,
  following the `reimplementrad` precedent.
