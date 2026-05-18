# RAD-SE Iteration Log — CartpoleSwingup seed 23

Local hardware: single RTX 3060 12 GB, 62 GiB host RAM. Env: `mujoco_playground` CartpoleSwingup, pixel obs, num_envs=8, action_repeat=8.

All runs below are in [runs/](../runs) under the directory listed in the "Path" column. Best-of-run is `max(eval/episode_reward)`.

## 1. Summary table

| # | Run dir (under [runs/](../runs)) | Replay | Storage | Batch | Updates/scan | Other knobs | Best ER | Last ER@step | SPS | Verdict |
|---|---|---|---|---|---|---|---:|---|---:|---|
| B | [brax_sac_CartpoleSwingup_s23](../runs/brax_sac_CartpoleSwingup_s23) | brax default | f32 device | brax default | brax default | brax SAC default, 500k steps | −2919.9 | −2919.9@500k | 169 | Floor baseline — full brax SAC ⇒ stuck near floor (~−2900). Confirms task is **not** trivially solvable with default config |
| 1 | [...replay5k_batch32_update8_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay5k_batch32_update8_scan125_CartpoleSwingup_s23) | 5k | f32 device | 32 | 8 | RAD encoder, framestack=3, crop84, reward-once, entropy | −2838.9 | −2917.4@12k | 50 | Above floor briefly @6k, then drift. Buffer too small |
| 2 | [...replay10k_f16_batch32_update8_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay10k_f16_batch32_update8_scan125_CartpoleSwingup_s23) | 10k | f16 device | 32 | 8 | + replay dtype=f16 to fit more | −2851.2 | −2879.6@17k | 52 | Marginal gain. Still capped by replay size |
| 3 | [...replay12500_f16_batch32_update8_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay12500_f16_batch32_update8_scan125_CartpoleSwingup_s23) | 12.5k | f16 device | 32 | 8 | Push replay to 12.5k f16 | — | OOM | — | **Fail (VRAM)** — first training executable HLO ~12.72 GiB, kernel needed ~4.19 GiB more |
| 4 | [...replay7500_f32_batch32_update8_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay7500_f32_batch32_update8_scan125_CartpoleSwingup_s23) | 7.5k | f32 device | 32 | 8 | Try larger replay with f32 | — | OOM | — | **Fail (VRAM)** — same reason, f32 7.5k ≈ f16 15k pressure |
| 5 | [...replay12500_f16_batch32_update8_scan25_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay12500_f16_batch32_update8_scan25_CartpoleSwingup_s23) | 12.5k | f16 device | 32 | 8 | Shorten `lax.scan` len 125→25 to reduce activations | — | OOM | — | **Fail (VRAM)** — confirms bottleneck is **carrying the replay buffer through JIT state**, not scan activations |
| 6 | [...host100k_f16_batch32_update8_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_host100k_f16_batch32_update8_scan125_CartpoleSwingup_s23) | **100k host** | f16 host | 32 | 8 | Off-device replay (numpy host pinned), sample → device per update | **−2719.4@33k** | −2802.0@52k | 24 | **Best run.** Clears 12 GiB VRAM ceiling. Replay no longer in JIT carry. ER hits −2719 by 33k |
| 7 | [...host100k_f16_..._radfreq_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_host100k_f16_batch32_update8_radfreq_scan125_CartpoleSwingup_s23) | 100k host | f16 host | 32 | 8 | + RAD-style update freq (actor/target every 2 critic) | **−2716.3@30k** | −2823.4@39k | 33 | Matches run #6 best, faster SPS. RAD update freq is "free" |
| 8 | [...host100k_f16_..._radfreq_tied_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_host100k_f16_batch32_update8_radfreq_tied_scan125_CartpoleSwingup_s23) | 100k host | f16 host | 32 | 8 | + tied actor/critic encoder weights | −2728.4@33k | −2733.7@35k | 36 | Same band; tying does **not** break learning, gives ~10% speedup |
| 9 | [...host100k_f16_..._radfreq_enctau_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_host100k_f16_batch32_update8_radfreq_enctau_scan125_CartpoleSwingup_s23) | 100k host | f16 host | 32 | 8 | + RAD `encoder_tau=0.05` separate from critic tau | **−2723.0@27k** | −2774.2@30k | 36 | Same band again; doesn't add measurable lift on this seed |
| 10 | [...host100k_uint8_..._radfreq_scan125_...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_host100k_uint8_batch32_update8_radfreq_scan125_CartpoleSwingup_s23) | 100k host | **centered uint8 host** | 32 | 8 | uint8 replay (re-centered on sample) | −2889.2@7k | −2927.6@10k | 21 | Memory-cheap (VRAM ~1.5 GiB, host stable) but **slower (21 SPS)** and only matches/trails f16 early. Stopped at 10k — not worth a long run |

Baselines that did **not** clear the floor (ER ≈ −3900 = "agent does nothing useful"):

| Run dir | Why kept | Best ER |
|---|---|---:|
| [...mem84_framestack3_replay5k...](../runs/brax_sac_mem84_framestack3_replay5k_CartpoleSwingup_s23) | Pre-RAD-encoder baseline | −3900.0 |
| [...mem84_framestack3_entropy_replay5k...](../runs/brax_sac_mem84_framestack3_entropy_replay5k_CartpoleSwingup_s23) | + entropy tweak only | −3899.2 |
| [...mem84_radencoder_framestack3_entropy_replay5k...](../runs/brax_sac_mem84_radencoder_framestack3_entropy_replay5k_CartpoleSwingup_s23) | + RAD encoder | −3831.3 |
| [...mem84_radencoder_tied_framestack3_entropy_rewardonce_replay5k...](../runs/brax_sac_mem84_radencoder_tied_framestack3_entropy_rewardonce_replay5k_CartpoleSwingup_s23) | + tied encoder | −3886.5 |
| [...mem84_radencoder_framestack3_entropy_rewardonce_replay5k...](../runs/brax_sac_mem84_radencoder_framestack3_entropy_rewardonce_replay5k_CartpoleSwingup_s23) | + reward-once fix | −2914.4 ← first run to leave the floor |
| [...replay5k_batch128_eval1_scan125...](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay5k_batch128_eval1_scan125_CartpoleSwingup_s23) | batch=128 instead of 32 | −3688.3 ← batch=128 actually worse here |
| [...mem84_replay15k / replay25k / replay50k...](../runs) | early replay scaling sweep on the old `mem84` config | ≤ −3899 |

## 2. What worked

1. **`reward-once` + RAD encoder + framestack=3 together** was the first config to leave the −3900 floor. None of those three alone was enough (see baselines).
2. **`batch_size=32` + `grad_updates_per_step=8`** (run #1) was the first to actually move ER. Brax's default `batch_size=256, updates=1` underuses the replay.
3. **Host-side replay (run #6)** is the single largest improvement. Holding 100k transitions outside the JIT carry both:
   - Removes the VRAM wall that killed runs #3–5.
   - Lets the algorithm finally use ≥100k samples like the RAD paper.
4. **RAD-style update frequencies (run #7)** match best-ER of #6 while running noticeably faster (33 vs 24 SPS). This is a free win, kept on.
5. **Tied actor/critic encoder (run #8)** does not hurt and gives ~10% speedup. Worth keeping as default once we run more seeds.

## 3. What did not work

- **More replay on-device (#3, #4, #5).** Pure VRAM failures. Even shortening `lax.scan` length from 125 to 25 (run #5) did not help: the HLO size was dominated by the replay tensor returned through carry, not by scan activations. Recorded in [/memories/repo/rad_brax_sac_replay_memory.md](../../../../memories/repo/rad_brax_sac_replay_memory.md).
- **Bigger batch instead of more updates** (`replay5k_batch128_eval1`). Worse than batch=32+updates=8. Confirms the RAD recipe of small batch / many updates per env step.
- **Centered uint8 host replay (run #10).** Memory was great (VRAM ~1.5 GiB, host stable), but throughput dropped to ~21 SPS due to the per-sample re-centering work, and the early CartpoleSwingup curve was no better than f16 (10k ER −2927.6 vs −2921.5 for run #7). Stopped early.
- **`encoder_tau` (run #9)** did not produce a measurable lift on this seed. Not harmful, just not the missing piece.

## 4. Where we are vs the RAD paper target

- All host100k runs are converging into a tight band: **best ER ≈ −2716 to −2728** by 27–33k env steps, on a 21–36 SPS budget.
- This is meaningfully above the floor but still well below the RAD paper's CartpoleSwingup numbers (~800+ on the dm_control scale). Two structural reasons:
  1. **Renderer mismatch**: mujoco_playground / Warp renderer is not the dm_control camera RAD was tuned on. Pixel statistics and reward-shaping interactions are not identical.
  2. **Replay 100k is still the RAD lower bound**, with our smaller batch (32) and update count (8 per scan). RAD trains many more updates per environment step at single-env scale.

## 5. Main failure mode = **hardware (VRAM), not algorithm**

The dominant blocker across runs #3–#5 was the **12 GiB RTX 3060 VRAM ceiling**. Concretely:

- On-device replay buffer carried through `jax.lax.scan` blew the HLO past 12 GiB at:
  - 12.5k transitions @ f16 (run #3)
  - 7.5k transitions @ f32 (run #4)
- Reducing scan length 125 → 25 did **not** help (run #5), proving the issue is the replay being part of JIT-traced state, not activation memory.
- Algorithmic changes (RAD encoder, tying, encoder_tau, update freq) only became measurable **after** the host-replay change (run #6) lifted that ceiling.

So: it was a memory architecture problem first, an algorithm tuning problem second.

## 6. What to do next

Ranked by expected payoff per hour of GPU time on this 3060.

1. **Multi-seed sanity** on the current best config (run #7 / #8 settings) — seeds 0, 23, 42 for 50k steps each. We currently only have seed 23 as evidence the band is real. ([run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_ablation.sh](../scripts/run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_ablation.sh) is the launcher).
2. **Longer run of #8 (tied, radfreq) to 100k–200k steps** on seed 23 — see if the band breaks upward with more replay turnover. Tied config has the best SPS/memory profile.
3. **Try the dm_control suite renderer path** instead of mujoco_playground / Warp on a single sanity run, to test the renderer-mismatch hypothesis in §4. Even one comparison frame would tell us a lot.
4. **Move expensive sweeps off-box.** The 3060 is the bottleneck; the bigger ablations (longer horizons, seed sweeps) should be queued on Vast.ai via the existing scheduler ([vastai-scheduler](../../vastai-scheduler)) on an RTX 3060 / 4090 host.
5. **Skip uint8 replay for now.** Memory savings real, but SPS cost ~30%; not worth it until we exceed 100k host capacity on a 32 GiB+ host.
6. **Defer encoder_tau as a default**, revisit only after multi-seed confirms it is consistently neutral or positive.

## 7. Reproduction pointers

- Best ER as of this log: **−2716.3 @ 30k** in [run #7](../runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_host100k_f16_batch32_update8_radfreq_scan125_CartpoleSwingup_s23).
- Launcher: [scripts/run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_ablation.sh](../scripts/run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_ablation.sh).
- Memory facts persisted to repo memory: [/memories/repo/rad_brax_sac_replay_memory.md](../../../../memories/repo/rad_brax_sac_replay_memory.md).
- Comparison vs RAD reference: [docs/compare.md](compare.md).

---

## 8. Open issue — why is ER ≈ −2900 instead of 0…1000? **[RESOLVED]**

**Root cause confirmed.** `mujoco_playground._src.dm_control_suite.cartpole.Balance` selects a **different reward function depending on `vision`**:

| Mode | Reward fn | Per-step | Episode range |
|---|---|---|---|
| `vision=False` (state) | `_dense_reward` = `upright · centered · small_control · small_velocity` (dm_control `tolerance` product) | `[0, 1]` | **`[0, 1000]` ← Playground/RAD paper number** |
| `vision=True` (pixels) | `_dense_vision_reward` = `0.1 + pole_pos_penalty + cart_pos_penalty + cart_vel_penalty + pole_vel_penalty + action_penalty` | `≈ [−3.9, 0.1]` | **`≈ [−3900, +100]` ← what we are training on** |

The "0–1000" Playground-paper number is **state-only**. Vision-mode CartpoleSwingup has a theoretical max of ~+100, not 1000. So all our runs were measured on a reward scale ~10× smaller and shifted negative.

**Verified empirically** ([rad_brax_sac.py make_envs](../src/rad_se/rad_brax_sac.py#L848)):
- Pole-down init step (vision reward): `−3.9000`. ⇒ 1000 × (−3.9) = **−3900** matches the brax-default floor exactly.
- Pole-down init step (dmc reward override): `≈ 0`. Perfectly balanced ≈ 1.0/step. ⇒ episode ∈ [0, 1000].

### Fix landed: `--dmc_reward` flag

Patched [src/rad_se/rad_brax_sac.py](../src/rad_se/rad_brax_sac.py):

1. Added `dmc_reward: bool = False` to `Config` (auto-wires `--dmc_reward / --no-dmc_reward` via argparse).
2. In `make_envs` after `raw_env = dm_control_suite.load(...)`, when `cfg.dmc_reward` is set, override `raw_env._get_reward = raw_env._dense_reward`. This forces the dm_control tolerance-product reward even in vision mode — pixel obs unchanged.

To run R1 (sanity 1000 steps with new reward):
```bash
scripts/run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_tied_ablation.sh CartpoleSwingup 23 \
    --dmc_reward --total_timesteps 1000
```

Once this confirms ER ∈ [0, 1000] on a do-nothing init (~0–50 per episode), all subsequent runs in §10 should use `--dmc_reward` so curves are comparable to RAD.

---

## 9. GPU + host memory cost reference

Per-transition footprint at obs shape `100 × 100 × 9` (framestack=3 RGB) for **one obs**. Each transition stores `(obs, action, reward, done, next_obs)`; next-obs is the next index in a ring buffer, so only **one obs per slot** is materialized.

| Dtype | Bytes / obs | Bytes / transition (obs + action + reward + done) | 10k transitions | 100k transitions |
|---|---:|---:|---:|---:|
| `float32` | 360 000 | ~360 048 | **3.36 GiB** | **33.55 GiB** |
| `float16` | 180 000 | ~180 048 | **1.68 GiB** | **16.77 GiB** |
| `uint8` (centered on sample) | 90 000 | ~90 048 | **0.84 GiB** | **8.38 GiB** |

Plus the **JIT working set** on the 3060 (model + critic-twin + optimizer states + scan activations + augmentation crops), measured at roughly **2.5–3 GiB** for our network with batch=32, updates=8, scan=125.

### Why on-device replay OOMs

When the replay buffer is part of `TrainingState` carried through `jax.lax.scan`, **XLA traces the full buffer into the HLO** and double-buffers it for the scan output. Empirically:

| Config | Replay | Replay VRAM | + JIT working set | HLO peak | Fits in 12 GiB? |
|---|---|---:|---:|---:|---|
| run #2 | 10k f16 device | 1.68 GiB | ~3 GiB | ~9–10 GiB | **yes, just** |
| run #3 | 12.5k f16 device | 2.10 GiB | ~3 GiB | **12.72 GiB** | **no** (needs +4.19 GiB beyond what fits) |
| run #4 | 7.5k f32 device | 2.52 GiB | ~3 GiB | ~12+ GiB | **no** |
| run #5 | 12.5k f16 device, scan=25 | 2.10 GiB | ~2.6 GiB | **12.72 GiB** | **no** — scan length didn't help, confirms replay-in-carry is the cause |
| run #6+ | 100k f16 **host** | 0 (on device) | ~3 GiB | ~3 GiB | **yes**, replay lives in 16.77 GiB host RAM |
| run #10 | 100k uint8 **host** | 0 (on device) | ~1.5 GiB | ~1.5 GiB | **yes**, replay in 8.38 GiB host RAM |

So the host-replay change moved **16.77 GiB out of VRAM**, leaving ~9 GiB of headroom on the 3060 for either bigger batch, longer scan, or future intrinsic-reward heads.

### Headroom available right now (host f16 replay)

- VRAM used at steady state: ~3 GiB / 12 GiB → **~9 GiB free**.
- Host RAM: replay ~16.8 GiB, training process total ~22 GiB / 62 GiB → **~40 GiB free**.
- We can comfortably **double** batch_size (32 → 64) or **double** updates/scan (8 → 16) on the current 3060 without crossing the VRAM line. We cannot fit on-device 100k replay even at uint8 (8.4 GiB replay + 3 GiB JIT ≈ 11.4 GiB; one bad scan output spills us over).

---

## 10. Concrete plan

### 10.1 Code fixes (do before more sweeps)

| Priority | Fix | Where | Effort | Why |
|---|---|---|---|---|
| **P0** | Log raw env reward, scaled reward, and reward sign in the first 100 env steps of every run. | [src/rad_se/rad_brax_sac.py](../src/rad_se/rad_brax_sac.py) — inside the rollout scan, dump `st.reward.min/max/mean` to `train.log` on step 0 and step 100. | small | Confirms / falsifies §8.1: are we training on negative cost or positive reward? Determines whether "ER −2700" is good or bad. |
| **P0** | Add a `--reward_convention {playground,dmc}` flag. When `dmc` is set, wrap env reward with the `dm_control.utils.rewards.tolerance`-equivalent reshape so ER lives in `[0, 1000]`. | new helper in [src/rad_se/rad_brax_sac.py](../src/rad_se/rad_brax_sac.py); env-wrapper applied once after `eval_env = ...`. | medium | Only way to compare to RAD paper numbers. |
| **P1** | Persist the replay buffer dtype + size + storage location to `train.log` header and to `metrics.jsonl` step 0. | logger init, `rad_brax_sac.py` near where `replay` is constructed. | trivial | We currently grep filenames to know which run is which. Make it explicit. |
| **P1** | Emit `vram_used_gb` and `host_replay_gb` to `metrics.jsonl` every eval epoch. | logger, alongside `sps`. | small | Auto-detects future regressions; lets us plot the VRAM-headroom story. |
| **P2** | Refactor `replay` so it always lives outside the JIT carry (drop the on-device replay code path entirely once we confirm host is fine on the bigger seeds). | `rad_brax_sac.py`, ~lines around 977. | medium | Removes a footgun. The current scripts can still OOM by accident. |
| **P2** | Add an `--assert_vram_lt=10000` guard at startup that allocates a probe tensor and exits cleanly if device free memory is under threshold. | startup of `rad_brax_sac.py`. | trivial | Fails fast instead of after JIT compile (~90s wasted on every OOM). |

### 10.2 Run plan (3060, in order)

All assume the **best current config = run #8 settings**: `host100k_f16, batch=32, update=8, radfreq, tied, scan125`. Launcher: [scripts/run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_tied_ablation.sh](../scripts/run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_tied_ablation.sh).

| # | Goal | Command (sketch) | Expected wall | Expected VRAM | Pass criterion |
|---|---|---|---|---|---|
| R1 | **Reward-sign diagnostic** (P0). Run the best config for **1000 env steps** with the new reward-logging code (P0 fix above), seed 23. | `scripts/run_..._radfreq_tied_ablation.sh CartpoleSwingup 23 --max_steps 1000` | ~1 min | ~3 GiB | Logged `env.reward.min/max` printed to train.log. Decides §8 (A vs B). |
| R2 | **Multi-seed sanity** of run #8, 50k env steps × 3 seeds (0, 23, 42). | `for s in 0 23 42; do scripts/run_..._radfreq_tied_ablation.sh CartpoleSwingup $s; done` | ~3 × 25 min ≈ 75 min | ~3 GiB | Mean best-ER across seeds is in the same −2710…−2740 band. CV < 10%. |
| R3 | **Long run** of run #8 to **100k env steps**, seed 23 — does the band break upward? | `scripts/run_..._radfreq_tied_ablation.sh CartpoleSwingup 23 --total_timesteps 100000` | ~45 min | ~3 GiB | best-ER beats −2716 (current global best). |
| R4 | **batch=64** ablation on run #8 (we now have VRAM headroom — §9). | new script: copy of #8 with `--batch_size 64` | ~25 min @ 50k | ~4–5 GiB | best-ER ≥ run #8 best within ±50. |
| R5 | **updates=16** ablation (use the other half of the VRAM headroom). | new script: copy of #8 with `--grad_updates_per_step 16` | ~50 min @ 50k (sps will halve) | ~5–6 GiB | best-ER strictly better than run #8 best. |
| R6 | **dm_control reward wrapper** (after P0 fix #2), 50k steps seed 23. | `scripts/run_..._radfreq_tied_ablation.sh CartpoleSwingup 23 --reward_convention dmc` | ~25 min | ~3 GiB | ER lives in `[0, 1000]`; first apples-to-apples RAD comparison. |
| R7 | **Off-box on Vast.ai** (only if R3–R5 give a real win): 200k–500k steps, seeds {0,7,23,42,1337}, RTX 4090 24 GB or A10. Use [vastai-scheduler](../../vastai-scheduler). | scheduler call (off-box) | hours | 6–10 GiB | Crosses 0 ER (if dmc wrapper) or beats brax default by ≥ 100 (if playground). |

### 10.3 What we are explicitly **not** doing (yet)

- No uint8 replay sweeps — §3 showed no early-curve advantage and 30% lower SPS.
- No `encoder_tau` work until multi-seed (R2) is done; it was a wash on seed 23.
- No new RAD aug primitives (cutout, color-jitter) until we resolve §8 (reward convention) — otherwise we'll be tuning against an incomparable metric.
- No further on-device replay attempts. Host replay is the architectural answer on 12 GiB.

---

## 11. GPU rental recommendation

Working set after host-replay refactor: ~3 GiB VRAM steady, ~16.8 GiB host RAM, replay 100k f16. A "bigger" run wants room for 200k–500k replay on-device + larger batch + room for an intrinsic-reward head later. Target ≥ 24 GB VRAM, ≥ 32 GB host RAM.

| GPU | VRAM | Typical Vast.ai $/hr (interruptible) | Fits 100k f16 on-device? | Fits 500k f16 on-device? | Notes |
|---|---|---|---|---|---|
| **RTX 4090** | 24 GB | $0.30–0.50 | ✅ (16.8 GiB + 4 GiB JIT) | ❌ (84 GiB needed) | **Recommended sweet spot.** Most $/perf; same JAX/CUDA wheels as current 3060. |
| **RTX 3090 / 3090 Ti** | 24 GB | $0.18–0.32 | ✅ | ❌ | Slightly slower than 4090 but **cheapest 24 GB option**. Stable, sm_86. |
| RTX A5000 | 24 GB | $0.35–0.55 | ✅ | ❌ | Datacenter card, more stable hosts. |
| L4 | 24 GB | $0.30–0.45 | ✅ | ❌ | Low TDP, slower than 4090 (~60–70%). |
| A10 | 24 GB | $0.50–0.80 | ✅ | ❌ | Stable but pricey. |
| **RTX 5090** | 32 GB | $0.60–1.00 | ✅ | ❌ | Best raw speed; needs PyTorch/JAX cu13x wheels (see [reimplementrad/rad_job_cmd_50series_cu132.sh](../../../reimplementrad/rad_job_cmd_50series_cu132.sh)). |
| A100 40 GB | 40 GB | $1.00–1.80 | ✅ | ❌ | Overkill for single-env pixel SAC; use only for multi-task batched runs. |
| H100 80 GB | 80 GB | $2.50–4.50 | ✅ | ✅ (84 GiB tight) | Only worth it for full SISA suite (cartpole+acrobot+cheetah+finger×seeds) in one process. |

**Recommendation:** **RTX 3090 24 GB on Vast.ai, ~$0.20–0.25/hr, avoid-countries `CN,US`** (matches the [vastai-scheduler](../../vastai-scheduler) preset from [LESSONS.md](LESSONS.md)). For one paper-quality run this is ~$5–8 of GPU at 200k steps. Use RTX 4090 only if SPS matters more than $.

What 24 GB unlocks immediately:

- **On-device 100k f16 replay** → no host-replay overhead, **expected SPS 60–80** (vs 33 on 3060).
- **batch_size=128 + updates=8** instead of batch=32. Closer to RAD paper's actual recipe.
- Room for the SISA intrinsic-reward head (~1 GiB extra params + ~1 GiB extra activations) without re-architecting.

---

## 12. Further memory-reduction options (if we stay on 12 GB)

Ranked by ratio of (VRAM saved) ÷ (expected ER penalty). All numbers assume the run #8 config baseline (100k host f16, batch=32, update=8).

| Option | What it changes | Memory saved | Performance impact | Verdict |
|---|---|---|---|---|
| **a) Smaller raw cam_res** (`cam_res=84` and drop `crop_size`) | Render 84×84 directly, skip crop. Obs bytes: 100·100·9·2 = 180 KB → 84·84·9·2 = 127 KB. | host replay 16.8 GiB → **11.8 GiB**; VRAM batch activations ~30% less | RAD aug *is* random crop, so losing crop removes the RAD-essential augmentation. **Strictly hurts**. Replace with random translate. ER penalty: ~50–150. | Only if needed |
| **b) frame_stack 3 → 2** | Channels 9 → 6 | host replay 16.8 → **11.2 GiB**; activations ~33% less | DMC pixel SAC literature: fs=2 vs fs=3 costs ~20% sample efficiency on cartpole, ~50% on cheetah-run. ER penalty: ~100–200. | Use only with a recurrent encoder, otherwise no |
| **c) Replay 100k → 50k host f16** | Halve buffer | host replay 16.8 → **8.4 GiB**; no VRAM change | RAD/DrQ ablations: replay 50k vs 100k on cartpole-swingup costs ~5–10% asymptotic. CartpoleSwingup is small enough that 50k is fine. ER penalty: ~30–80. | **Yes** if host RAM tight |
| **d) Replay 100k → 25k on-device f16** | Move replay back on-device but smaller | host replay 0; VRAM replay 4.2 GiB; *probably still OOMs JIT scan* | Likely worse than option (c) — buffer too small for SAC + worse off-policy variance. ER penalty: ~150–300. | No |
| **e) uint8 host replay** | Already tried (run #10). | host 16.8 → 8.4 GiB | SPS −30% (21 vs 33), early curve no better. ER penalty: ~0–30 long-run, but slower wall clock. | Only paired with (c) if both are needed |
| **f) Reduce scan length 125 → 25** | Smaller activation footprint | VRAM ~2 GiB less | SPS slightly worse (more Python overhead). ER neutral. | Free win — but only useful when on-device replay is reintroduced, so wait for 24 GB. |
| **g) Smaller encoder (RAD-tiny: 32→16 ch, 4→3 layers)** | Half the encoder params + activations | VRAM ~1 GiB; replay unchanged | Loses representation capacity. RAD/SAC ablation suggests ~20–30% ER penalty on swingup. ER penalty: ~200. | No |
| **h) Gradient checkpointing on encoder forward** | Recompute encoder activations in backward | VRAM ~0.5–1 GiB | SPS −15%; ER neutral | Use only after (a)–(c) | 
| **i) Larger action_repeat (8 → 16)** | Fewer agent steps per episode | host replay halved (62.5 → 31.25 agent steps per episode), batch activations same | SAC paper: AR=8 is the sweet spot for cartpole-swingup; AR=16 collapses learning. ER penalty: ~500. | No |

**Memory-reduction plan for 12 GB box:** combine **(c) replay 50k host f16** with **(f) scan=25 when we eventually go on-device** — saves the most $ before swapping GPUs, with minimal ER cost.

---

## 13. Can we use PPO instead?

**Short answer: yes, and it might be a better fit for the 12 GiB 3060** — because PPO needs no replay buffer at all, the entire §9 / §12 memory story disappears.

### Why PPO is attractive here

- **No replay buffer.** On-policy. Saves the full 16.8 GiB host (or 16.8 GiB VRAM on-device) we currently spend on SAC replay.
- **brax PPO is already in this repo:** [src/rad_se/rad_brax_ppo.py](../src/rad_se/rad_brax_ppo.py) and [src/rad_se/rad_ppo_jax.py](../src/rad_se/rad_ppo_jax.py). The latter already applies `reward_scale=0.1` because "Playground sums over action_repeat" — meaning the author already saw the reward-scale issue from §8.
- **brax PPO scales with num_envs**, and the 3060 can run num_envs=64 or 128 pixel envs in parallel since each env is only the renderer + a tiny network forward, **no replay**. SPS could be 5–10× higher than SAC.
- **No off-policy bias from stale aug + augmentation distribution shift** — RAD's random crop is applied to current-policy obs only.

### Why PPO is worse for pixel domains

- **Sample efficiency on dm_control pixels:** PPO needs ~5–10M env steps on cartpole-swingup vs ~200k for SAC. Wall-clock can still win because SPS is higher, but cost in env-steps is real.
- **DrQ / RAD literature**: pixel SAC > pixel PPO at all step budgets ≤ 1M. PPO catches up only at very large step counts.
- **Brax PPO + pixels is less battle-tested** than brax PPO + state.

### Numbers to expect (PPO @ 3060)

Educated estimate from [src/rad_se/rad_ppo_jax.py](../src/rad_se/rad_ppo_jax.py) defaults plus brax PPO scaling:

| Setting | Likely SPS | env steps for ER > 500 | Wall-clock |
|---|---|---|---|
| num_envs=64 pixel, batch=2048, GAE | 800–1200 | 5M | ~70–100 min |
| same on RTX 4090 | 2500–4000 | 5M | ~20–35 min |

vs SAC current: 200k env steps for ER ~−2716 (vision reward), wall ~15 min on 3060. Direct ER comparison only possible after §8 fix lands.

### Recommended plan if PPO is on the table

1. Land the §8 `--dmc_reward` flag first so PPO and SAC are on the same reward scale.
2. Run a **single PPO baseline** on CartpoleSwingup with `--dmc_reward` for 1M env steps (matches what [rad_brax_ppo.py:352](../src/rad_se/rad_brax_ppo.py#L352) already supports). Should land ~ER 200–500.
3. If wall-clock and ER look competitive, switch the SISA goal from "RAD-SAC variant" to "RAD-PPO variant". The intrinsic-reward head plugs into either.
4. Reuse the existing [scripts/run_rad_jax_ablation_stable.sh](../scripts/run_rad_jax_ablation_stable.sh) pattern but pointed at `rad_brax_ppo.py`.

### Decision rule

- **If we stay on the 3060 12 GB** → **try PPO**. It sidesteps every memory issue in §9–§12 and may give better SPS.
- **If we rent a 24 GB GPU** → **stay on SAC** with on-device 100k f16 replay + batch=128. SAC is more sample-efficient and the publication story is "RAD-SE built on RAD-SAC", not RAD-PPO.

## 14. Episode-length truncation bug — 12× too short (May 18 2026) **[FIXED]**

### TL;DR

Both PPO (`ER ≈ 11.7` plateau at 1.7M) and SAC (`ER ≈ 75` plateau at 500k) were stuck because **every training episode was being truncated at 16 agent steps instead of 125**. The bug was a double-divide of `episode_length` by `action_repeat`. After fixing, with no other change:

| Run | Bug ER | Fixed ER (best @ 500k) | Improvement |
|---|---|---|---|
| PPO `brax_ppo_..._FIXED_500k` | 11.7 (plateau @ 1.7M) | **277** (best), 274 final | 24× |
| SAC `brax_sac_..._FIXED_500k` | 75 (final @ 500k) | **263** (best), ~200 mean | 3.5× |
| Random baseline (diag) | 18 | **231** | 12× |

Still below paper ~800 — remaining gap is hyperparameter, not bug.

### Diagnosis path

Wrote standalone [scripts/diagnose_brax_env.py](../scripts/diagnose_brax_env.py): rebuilds the exact training env via the same `make_envs` builder, rolls 1 episode with random / zero / bangbang actions, dumps `diag.mp4 + diag.npz + diag_report.txt` with per-step rewards / actions / dones / qpos / qvel and pixel stats.

First run (`runs/_diag/cartpole_random/`) showed:
- **dones at exact multiples of 16**: `[15, 31, 47, 63, 79, 95, 111]`
- Random ER = 18 / episode (DMC tolerance max = 1000)
- `pole_downward_fraction = 0.816` — random actions were swinging the pole only inside a 16-step window, never reaching upright

### Root cause

`mujoco_playground.wrap_for_brax_training` builds a `brax.envs.wrappers.training.EpisodeWrapper` whose `step()` accumulates **physics-step count** and triggers done on `steps ≥ episode_length`. We were passing `episode_length = cfg.episode_length // cfg.action_repeat = 125`, so done fired at `steps=128 ≥ 125` ⇒ episode lasted `128/8 = 16` agent steps. The brax PPO trainer's `episode_length=` argument has the same physics-step convention, so the evaluator was running 15-step rollouts too.

`γ^16 ≈ 0.85` value-horizon truncates before swingup can complete (~50 steps needed). Both PPO and SAC were learning the wrong task.

### Fix (3 lines)

[src/rad_se/rad_brax_ppo.py](../src/rad_se/rad_brax_ppo.py):
- `make_envs`: pass `episode_length=cfg.episode_length` to `wrap_for_brax_training` (was `agent_episode_length`).
- `ppo_train.train(episode_length=cfg.episode_length, ...)` (was `agent_episode_length`).

[src/rad_se/rad_brax_sac.py](../src/rad_se/rad_brax_sac.py):
- `make_envs`: same one-line fix. SAC eval scan length (`cfg.episode_length // cfg.action_repeat = 125`) was already correct.

### Verification

Re-ran `diagnose_brax_env.py` → `runs/_diag/cartpole_random_FIXED/`:
- **1 done at step 124/125** ✓
- Random ER = **231** (was 18; matches DMC random baseline ~150–300)
- `pole_downward_fraction = 0.176` (was 0.816)
- Pole now sweeps full `[0.58, 5.32]` rad range during random rollout

### New artifacts

- [scripts/diagnose_brax_env.py](../scripts/diagnose_brax_env.py) — reusable env diagnostic (video + npz + stats)
- [docs/DISTRIBUTED_CPU_ACTORS.md §5.0](DISTRIBUTED_CPU_ACTORS.md) — "What kind of env can MJX actually simulate?" (rigid-body physics yes; Atari/Procgen/turn-based no; physics-based games like soccer/billiards/locomotion yes)
- Run dirs: `runs/brax_ppo_CartpoleSwingup_s23_FIXED_500k/`, `runs/brax_sac_CartpoleSwingup_s23_FIXED_500k/`

### Remaining gap to paper (~800)

SAC plateaus at 150-263 from 125k onward; `alpha` rapidly collapses 0.45 → 0.005 → entropy starvation. Candidate fixes ordered by expected impact:

1. **`--target-entropy -1.0`** (currently -0.5; RAD/SAC convention is -action_dim). Will keep entropy higher → more exploration → better swingup discovery.
2. **`--reward-scaling 1.0`** (currently 0.1). Scaling rewards down by 10× shrinks Q-targets to [0, ~80] and starves the critic gradient. Paper RAD uses raw rewards.
3. **`--max-replay-size 50000`** (currently 10000). Too small for pixel SAC; DrQ/RAD typically use 100k+.
4. **`--action-repeat 4`** (currently 8). RAD paper uses 4 → 250 agent steps/episode → finer control.
5. **Longer budget** (1-2M steps).

Fix #1+#2 is the cheapest and most likely to push SAC past 600. Running next.
