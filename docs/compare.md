# Implementation Comparison: RAD-SE vs Original RAD vs Brax SAC

**Current implementation**: `src/rad_se/rad_brax_sac.py`  
**Reference A**: `reimplementrad/implementations/rad_sac_dmc_pixel.py` (JAX port of original RAD, PyTorch-faithful)  
**Reference B**: `brax/training/agents/sac/train.py` (brax official SAC)  
**Laskin et al. 2020**: "Reinforcement Learning with Augmented Data" (RAD paper)

---

## 1. Encoder Architecture

| Aspect | Original RAD (ref A) | Brax SAC (ref B) | RAD-SE (current) | Gap? |
|--------|---------------------|-----------------|-----------------|------|
| Layers | 4× Conv2d 3×3 | MLP (state obs) | 3× Conv2d 8×8/4×4/3×3 (DQN) | **YES** |
| Filters | 32 fixed each layer | — | 32→64→64 | **YES** |
| First stride | 2, rest stride-1 | — | 4, 2, 1 | **YES** |
| Feature projection | Linear(flat→50) + LayerNorm + tanh | RunningStatistics | Global avg pool → 64-dim | **YES** |
| Feature dim | 50 | obs_size | 64 | **YES** |
| Padding | valid (implicit) | — | VALID | Same |
| Image format | NCHW uint8 /255 | any | NHWC float32 | Different (JAX convention) |

**Gap summary**: RAD-SE uses a DQN-style 3-layer encoder (large kernels, increasing channels, global avg pool) rather than the RAD 4-layer 3×3 conv + feature projection. This changes the inductive bias significantly. The RAD encoder is shallower in kernel size but deeper in abstraction via the learned 50-d projection with LayerNorm+tanh.

**Recommended fix**: Replace `cnn_output_channels=(32,64,64)`, `cnn_kernel_size=(8,4,3)`, `cnn_stride=(4,2,1)`, `cnn_global_pool=True` with 4× 3×3 stride-2/1/1/1 conv layers, then a Linear(flat→50)+LayerNorm+tanh. This requires a custom `PixelEncoderRAD` Flax module.

---

## 2. Encoder Weight Tying (Actor–Critic)

| Aspect | Original RAD | Brax SAC | RAD-SE | Gap? |
|--------|-------------|---------|--------|------|
| Actor/critic share conv weights | **YES** — `copy_conv_weights_from` ties actor encoder to critic encoder | N/A (state obs) | **NO** — `VisionQModule` and `make_policy_network_vision` have fully independent CNNs | **YES** |
| Separate target encoder | YES — `critic_target.encoder` soft-updated with `encoder_tau=0.05` | YES — `target_q_params` | YES — `target_q_params` covers Q CNN | Partial |
| Actor encoder detach on critic update | YES — `detach_encoder=True` in actor loss | N/A | NO — brax `sac_losses` does not detach | **YES** |

**Gap summary**: RAD's key efficiency trick is that the actor's CNN is the critic's CNN (tied weights), so gradients only flow back through one CNN during the actor update. RAD-SE trains two independent CNNs in the actor and two independent CNNs (per Q-head) in the critic — 4 CNNs total. This likely hurts sample efficiency and is a clear deviation from RAD's design.

**Recommended fix**: Implement a shared encoder trunk (frozen for actor gradient updates via `jax.lax.stop_gradient` on actor CNN path) or use a single CNN whose params are referenced by both actor and critic.

---

## 3. Data Augmentation

| Aspect | Original RAD | Brax SAC | RAD-SE | Gap? |
|--------|-------------|---------|--------|------|
| Augmentation type | **Random crop** 100→84 (16-pixel margin) | None | **Random translate** (pad-4, then crop back to 100) | **YES** |
| Applied at | Sample time (CPU, numpy) | — | Sample time (JAX, on-device) | Same |
| Eval augmentation | Center crop | — | None (full 100×100) | **YES** |
| Image size after aug | 84×84 | — | 100×100 (unchanged) | **YES** |

**Gap summary**: The RAD paper's primary augmentation is random crop (implemented in all DMC experiments). Random translate is a valid RAD-compatible augmentation (it also appears in the RAD paper's Table 3), but the current implementation uses a 4-pixel padding vs. the RAD paper's 84×84 effective crop from 100×100 (16-pixel margin). Eval uses center crop in original RAD; RAD-SE evaluates on the full 100×100 frame without crop, which creates a train/eval distribution mismatch.

**Recommended fix**:
1. For strict RAD parity: switch to random crop 100→84 at sample time.
2. For current translate aug: ensure eval also uses translate (or center crop) to match training distribution.

---

## 4. Frame Stacking

| Aspect | Original RAD | Brax SAC | RAD-SE | Gap? |
|--------|-------------|---------|--------|------|
| Frames stacked | 3 (9 channels for RGB, 3 for grey) | — (state obs) | 1 (3 channels) | **YES** |
| Temporal context | YES | — | NO | **YES** |

**Gap summary**: The original RAD (and virtually all DMC pixel baselines) stacks 3 consecutive frames to provide the policy temporal information about velocity/direction. CartpoleSwingup requires velocity information to learn swingup. Without frame stacking, the policy sees only the current frame and cannot infer angular velocity or cart velocity from pixel obs alone.

**Recommended fix**: Add frame stacking (k=3) to the brax/playground wrapper. Each env.step() should concatenate the last 3 `pixels/view_0` frames along the channel axis: `(H, W, 3k)`.

---

## 5. SAC Hyperparameters

| Hyperparameter | Original RAD | Brax SAC default | RAD-SE (current) | Gap? |
|---------------|-------------|-----------------|------------------|------|
| Actor/critic LR | 1e-3 | 1e-4 | **3e-4** | Minor |
| Alpha LR | 1e-4 | — | **3e-4** | Minor |
| Adam beta1 | 0.9 | 0.9 | **default (0.9)** | Same |
| Initial temperature α | **0.1** (`init_temperature`) | — | **1.0** (log_alpha=0) | **YES** |
| Target entropy | `-prod(action_shape)` = -1 | `-action_size/2` | brax default | Minor |
| critic_tau | 0.01 | 0.005 | **0.005** (brax default) | Minor |
| encoder_tau | **0.05** (target encoder soft update) | — | Not separately tracked | **YES** |
| actor_update_freq | **2** (every 2 steps) | — | **1** (every step) | **YES** |
| critic_target_update_freq | **2** (every 2 steps) | — | **1** (every step) | **YES** |
| discount | 0.99 | 0.9 | **0.99** | Same |
| batch_size | 32 | 256 | **256** | Different |
| replay_capacity | **100,000** | num_timesteps | **10,000** | **CRITICAL** |
| grad_updates_per_step | 1 per env transition | configurable | **1 per 8-env vector step by default; `--grad-updates-per-step 8` ablation added** | **YES when `num_envs=8`** |
| reward_scaling | 1.0 | 1.0 | **0.1** | Minor deviation |
| init_steps / min_replay_size | 1000 | 0 | **200** (default) | Minor |

**Critical gaps**:

1. **Initial temperature 1.0 vs 0.1**: The RAD paper initialises `log_alpha = log(0.1)`. Starting at α=1.0 means maximum entropy exploration from the start, which can prevent the policy from committing to useful actions early in training.

2. **Replay capacity 10k vs 100k**: This is the primary cause of the observed plateau. 10,000 transitions at num_envs=8 is only 1,250 unique timesteps. The buffer overfits immediately. RAD uses 100,000 (10× more).

3. **actor_update_freq=1 vs 2**: RAD and SAC papers both recommend updating the actor half as often as the critic (since actor uses critic Q values). Updating every step wastes computation and can destabilise training.

4. **encoder_tau missing**: In original RAD, the target critic's encoder is soft-updated with tau=0.05 (slower than the Q-network tau=0.01). This is tracked separately. In RAD-SE, `target_q_params` is soft-updated uniformly with tau=0.005 — covering the CNN encoder too, but with the wrong tau value (10× slower than RAD's).

---

## 6. Replay Buffer Implementation

| Aspect | Original RAD | Brax SAC | RAD-SE | Gap? |
|--------|-------------|---------|--------|------|
| Storage | CPU numpy, uint8 | On-device JAX array, float32 | On-device JAX array, float32 | Matches brax |
| Capacity | 100,000 | max_replay_size | **10,000** | **CRITICAL** |
| Dtype | uint8 (pixels) + float32 (rest) | float32 | float32 | Memory inefficient but correct |
| Augmentation in sample | YES (random_crop) | NO | YES (random_translate) | Partial |
| Insertion | Sequential circular buffer | UniformSamplingQueue (JAX) | UniformSamplingQueue (JAX) | Matches brax |
| `not_done` / `discount` | Explicit `not_done=1-done` | `discount` field | `discount=1-done` | Same semantics |

**Gap**: RAD-SE's float32 pixel storage uses ~4× more memory than uint8. At 10k transitions × (100×100×3) × float32 ≈ 120 MB. If capacity is increased to 100k, this becomes 1.2 GB — still manageable on 12 GB VRAM but warrants uint8 compressed storage.

---

## 7. Training Loop

| Aspect | Original RAD | Brax SAC | RAD-SE | Gap? |
|--------|-------------|---------|--------|------|
| Vectorised envs | NO — single env Python loop | YES — `num_envs` parallel | YES — 8 parallel | Improvement |
| Loop implementation | Python `for step in range(T)` | `jax.lax.scan` (via `acting.generate_unroll`) | `jax.lax.scan` | Matches brax |
| Gradient updates | 1 per env transition | 1 per env step (default) | 1 per scan step by default; configurable | **YES when `num_envs>1`** |
| Episode reset | Explicit `done` check | Brax auto-reset | Brax auto-reset | Same |
| Eval | 10 sequential episodes, center crop | 128 eval envs, parallel | 16 eval envs, parallel | Improvement |
| Eval policy | Deterministic (compute_pi=False) | `deterministic=True` flag | `deterministic=True` | Same |
| Scan buffer | N/A | `lax.scan` retains O(scan_steps) activations | O(scan_steps) activations | Known cost |

**RAD-SE improvement over original**: 8 parallel environments + `jax.lax.scan` reduces Python overhead and enables XLA fusion. This is expected to be faster at convergence per wall-clock second, all else equal.

---

## 8. Environment / Physics Backend

| Aspect | Original RAD | Brax SAC | RAD-SE | Gap? |
|--------|-------------|---------|--------|------|
| Physics backend | MuJoCo (dm_control) | MuJoCo/brax (state obs) | **MuJoCo Warp** (mujoco_playground) | Different |
| Interface | dmc2gym (gym API) | brax Env API | brax Env API via mp_wrapper | Same as brax |
| Pixel rendering | dm_control camera | — | Warp GPU rasteriser | Different |
| Action repeat | 8 (cartpole) | configurable | 8 | Same |
| Episode length | 1000 env steps (dm_control default) | configurable | 500 / 8 = 62.5 agent steps (1000 steps / AR=8) | Same |
| Observation space | NCHW (gym convention) | state vector | NHWC dict `{'pixels/view_0': (H,W,C)}` | Different format |
| Resolution | 100×100 (pre-crop to 84) | — | 100×100 | Same pre-aug size |

**Note**: MuJoCo Warp renders differently from dm_control's MuJoCo. Pixel statistics (contrast, colour balance, background) may differ, potentially affecting aug effectiveness.

---

## 9. SAC Loss Formulation

| Aspect | Original RAD | Brax SAC (losses.py) | RAD-SE | Gap? |
|--------|-------------|---------------------|--------|------|
| Critic loss | MSE(Q1, target) + MSE(Q2, target) | MSE (same) | brax `sac_losses.make_losses` | Same |
| Actor loss | `α*log_π - min(Q1,Q2)` | Same | brax losses | Same |
| Alpha loss | `α*(−log_π − H_target)` | Same | brax losses | Same |
| Target Q | `r + γ*(min(Q1_t,Q2_t) − α*log_π)` | Same + reward_scaling | brax losses | Same |
| Double Q | YES — twin critics | YES | YES — VisionQModule n_critics=2 | Same |
| Target network update | Hard copy or soft (tau=0.01) | Soft (tau=0.005) | Soft (tau=0.005) | Minor |

**No functional gaps** in SAC loss formulation. The brax `sac_losses` module is a correct JAX implementation of the standard SAC algorithm (Haarnoja et al. 2018, rev. entropy objective).

---

## 10. Observation Normalisation

| Aspect | Original RAD | Brax SAC | RAD-SE | Gap? |
|--------|-------------|---------|--------|------|
| Pixel normalisation | /255 at sample time | RunningStatistics (state obs) | None applied (float32 from env) | **YES** |
| Expected range | [0, 1] for CNN | — | Unknown (likely [0,255] or [0,1]) | **YES** |

**Gap**: The original RAD divides uint8 pixels by 255 at sample time. RAD-SE should verify that `mujoco_playground` renders pixels in [0, 1] float range. If the env returns [0, 255] float32, the CNN inputs will be far out of distribution. Check: `print(env_state.obs['pixels/view_0'].max())` after env.reset().

---

## 11. brax SAC train.py Specific Features Not in RAD-SE

| Feature | Brax SAC | RAD-SE | Gap? |
|---------|---------|--------|------|
| `pmap` multi-device | YES | NO (single GPU) | Acceptable for development |
| Obs normalisation (state) | RunningStatistics | Dummy (correct for pixels) | No gap (pixels don't need it) |
| `min_replay_size` prefill | YES | YES (200) | Same |
| `grad_updates_per_step` config | YES | YES — default 1, ablation uses 8 | Important with vectorised envs |
| Checkpoint save/restore | YES | NO | Feature gap |
| `randomization_fn` domain rand | YES | NO | Feature gap |
| `progress_fn` callback | YES | Logger.log | Equivalent |

---

## 12. Cumulative Gap Table

| Priority | Gap | Root cause | Impact |
|----------|-----|-----------|--------|
| 🔴 P0 | `max_replay_size=10k` (should be ≥100k) | Config default | **ER plateau — primary cause** |
| 🔴 P0 | No frame stacking (k=3) | Missing wrapper | Cannot infer velocity from pixels |
| 🟠 P1 | Encoder architecture (DQN vs RAD 4×3×3) | CNN config | Wrong inductive bias for DMC |
| 🟠 P1 | Initial temperature α=1.0 (should be 0.1) | `log_alpha=0` init | Overly random early policy |
| 🟠 P1 | No tied actor/critic encoder weights | Architecture | Extra CNNs, gradient interference |
| 🟡 P2 | Augmentation: translate (pad-4) vs crop (100→84) | `data_augs` choice | Different aug strength/type |
| 🟡 P2 | No eval center crop / translate | Missing aug in eval | Train/eval distribution mismatch |
| 🟡 P2 | `actor_update_freq=1` (should be 2) | Missing update delay | Unnecessary actor updates |
| 🟡 P2 | `encoder_tau=0.005` (should be 0.05 for encoder) | Shared soft-update | Encoder target updated too slowly |
| 🟢 P3 | `reward_scaling=0.1` (RAD uses 1.0) | Config | Minor: helps stability on this env |
| 🟢 P3 | `batch_size=256` vs RAD's 32 | Config | Higher batch = more stable, but less on-policy |
| 🟢 P3 | `critic_tau=0.005` vs RAD's 0.01 | Config | Slower target update |
| 🟢 P3 | No checkpoint save/restore | Missing feature | Convenience only |

---

## 13. Recommended Action Plan

### Immediate (fix plateau) — P0
```python
# scripts/run_brax_sac.sh or config
max_replay_size=100000   # was 10000
total_timesteps=2000000  # was 500000 (need longer to fill larger buffer)
min_replay_size=1000     # was 200
```

### Frame stacking — P0
Implement a `FrameStackWrapper` around the mujoco_playground env that concatenates the last `k=3` pixel observations along the channel axis:
```python
# in rad_brax_sac.py: make_envs() or a wrapper
# stacked obs shape: (H, W, 3*k) = (100, 100, 9)
```

### Temperature init — P1
```python
# In training_state init
log_alpha=jnp.array(np.log(0.1))  # was jnp.zeros(())
```

### Encoder architecture — P1
Replace `brax_networks.make_policy_network_vision` CNN with a RAD-faithful encoder:
- 4× Conv2d 3×3 (first stride=2, rest stride=1), 32 filters each
- `flatten → Linear(32*out_dim*out_dim, 50) → LayerNorm → tanh` projection

### Augmentation eval fix — P2
Apply center-crop (or same translate aug) during `run_eval` to match training distribution.

### Update frequencies — P2
```python
actor_update_freq: int = 2    # currently 1
target_update_freq: int = 2   # currently 1
```
Requires splitting the single `sgd_step` into conditional actor/target updates.

---

## 14. What RAD-SE Does Correctly vs Original RAD

- ✅ SAC loss formulation (standard Haarnoja 2018 with entropy regularisation)
- ✅ On-device replay buffer (improvement over CPU numpy)
- ✅ Parallel environments (improvement over single-env)
- ✅ `jax.lax.scan` training loop (improvement)
- ✅ reward_scaling=0.1 (empirically helps stability on this env/scale)
- ✅ discounting=0.99 (matches RAD)
- ✅ Action repeat=8 (matches RAD cartpole setting)
- ✅ Resolution 100×100 pre-augmentation (matches RAD)
- ✅ Augmentation at sample time (not env time)
- ✅ Deterministic eval policy
- ✅ Warp warmup fix (CUDA stream capture safety)
- ✅ No obs normalisation on pixels (correct — RunningStatistics is for state obs)

---

## 15. Exact-Match JAX Version Added

**Implementation**: `src/rad_se/rad_jax.py`  
**Exact-match launcher**: `scripts/run_rad_jax_exact.sh`  
**Ablation launcher**: `scripts/run_rad_jax_ablation_stable.sh`
**Fast memory-aware launcher**: `scripts/run_brax_sac_mem_ablation.sh`
**Fast frame-stack launcher**: `scripts/run_brax_sac_framestack_ablation.sh`
**Fast frame-stack + entropy launcher**: `scripts/run_brax_sac_framestack_entropy_ablation.sh`
**Fast RAD encoder + entropy launcher**: `scripts/run_brax_sac_radencoder_entropy_ablation.sh`

The exact-match JAX version now targets the original RAD DMC pixel setup rather than the faster brax-loss hybrid:

| Item | Exact JAX setting | Original RAD target | Status |
|------|-------------------|---------------------|--------|
| Encoder | 4× 3×3 conv, first stride 2, rest stride 1 | Same | Matched |
| Encoder projection | Linear → 50, LayerNorm, tanh | Same | Matched |
| Actor/critic conv tying | critic encoder conv copied to actor encoder | Same | Matched |
| Target encoder | soft target update supported | Same | Matched |
| Augmentation | random crop 100→84 at replay sample | Same | Matched |
| Eval crop | center crop 100→84 | Same | Matched |
| Replay | CPU numpy uint8, capacity 100k | Same | Matched |
| Frame stack | RGB frame-stack k=3, NHWC 9 channels | RGB×3 stack | Matched semantically |
| Batch size | 32 | 32 | Matched |
| Initial α | 0.1 | 0.1 | Matched |
| Actor update freq | 2 | 2 | Matched |
| Target update freq | 2 | 2 | Matched |
| Critic tau | 0.01 | 0.01 | Matched |
| Encoder tau | 0.05 config retained | 0.05 | See note below |
| Reward scale | 1.0 | raw reward | Matched |
| Learning rates | actor/critic 1e-3, alpha 1e-4 | Same | Matched |

### Inevitable / intentional deviations from exact PyTorch RAD

1. **NHWC instead of NCHW**: JAX/Flax convolution layout is NHWC by default. This is a layout change only, not an algorithmic change.
2. **MuJoCo Playground + Warp renderer instead of dm_control/dmc2gym**: The physics task is the same DMC CartpoleSwingup family, but pixel rendering differs. This may affect visual statistics.
3. **Vectorised env collection (`num_envs=8`) instead of one Python gym env**: This is needed for practical GPU utilisation with Playground/Warp. The fast trainer now exposes `--grad-updates-per-step`; strict update-ratio parity under 8 vectorised envs uses `--grad-updates-per-step 8`.
4. **CPU uint8 replay buffer is kept deliberately**: A strict 100k replay with 100×100×9 stacked pixels would be expensive as on-device float32. CPU uint8 matches original RAD and avoids a GPU memory failure.
5. **Target encoder tau caveat**: The exact JAX implementation keeps `--encoder-tau 0.05` in the config. The current fused target updater still applies the critic tau uniformly to the whole target critic. If this run underperforms after replay/frame-stack fixes, the next code change should split encoder and Q-head target updates.
6. **Current exact trainer performance caveat**: A smoke run of `src/rad_se/rad_jax.py` entered training but remained CPU-bound after the initial 100 smoke steps on the local RTX 3060 host. It is valid as the exact reference path, but not yet the main long-run trainer.

### Hypotheses implemented as ablations

The exact run should be the reference. The following changes are intentionally isolated in `scripts/run_rad_jax_ablation_stable.sh` so they can be compared later:

| Hypothesis | Exact value | Ablation value | Rationale |
|------------|-------------|----------------|-----------|
| Reward scaling | 1.0 | 0.1 | Brax/playground rewards may be higher magnitude after action-repeat aggregation; scaling previously stabilised Q values. |
| Batch size | 32 | 128 | Larger batches may reduce Q noise with vectorised replay, at the cost of deviating from RAD. |
| Local memory run | 100×100 crop→84, CPU uint8 exact replay | 84×84 render, on-device float32 replay 15k | Fits the existing fast brax trainer on a 12 GB RTX 3060 while increasing replay 1.5× over the plateaued 10k run. A 50k attempt requested ~15.8 GiB at prefill; a 25k attempt reached prefill but OOMed during first eval/training executable overlap. |
| Frame-stack local run | frame_stack=3 | 84×84 render, frame_stack=3, replay 5k | Tests the velocity-observability hypothesis while keeping GPU memory similar to the replay15k/no-stack run. |
| Entropy local run | init_temperature=0.1, target_entropy=-1, alpha_lr=1e-4 | frame_stack=3 plus RAD entropy settings in the fast trainer | Tests whether brax's α=1.0 / target_entropy=-0.5 setting is causing the early entropy collapse seen in replay15k and frame-stack5k runs. |
| RAD encoder local run | 4× 3×3 conv, first stride 2, projection to 50 + LayerNorm + tanh | Fast trainer selectable with `--encoder-arch rad` | Tests the largest remaining architecture mismatch while still using local-memory replay. Actor/critic encoders are still independent, so weight tying remains a separate gap. |
| Update-ratio local run | 1 SGD update per collected env transition | Fast trainer selectable with `--grad-updates-per-step`; update8 launcher uses 8 updates per 8-env vector step | Tests whether the prior fast path was under-training by doing only 1 replay/SGD update after collecting 8 transitions. |
| Typed replay local run | CPU uint8 replay, 100k capacity | On-device typed tree replay with float16 pixel leaves and float32 action/reward leaves | Tests the replay capacity bottleneck without corrupting the MuJoCo Playground pixel normalization, which is float32 and not raw uint8. |

Do not mix ablations into the exact baseline unless the exact baseline demonstrably fails for an implementation-independent reason.

### Current launch priority

1. Run `scripts/run_brax_sac_mem_ablation.sh` on the local 12 GB GPU to test the replay-capacity hypothesis quickly.
2. If reward improves materially, move the exact JAX path or a uint8/on-device replay version to a larger GPU for strict matching.
3. If reward remains flat near -3000, implement the next exactness gap in the fast trainer: frame-stack + 100→84 random crop + RAD encoder.

### Local iteration notes

- `brax_sac_mem84_replay15k_CartpoleSwingup_s23`: fit in memory at ~8.8 GiB, but stayed flat at ER ≈ -3900 through 48k steps while α collapsed to 0.231. Stopped and replaced with the frame-stack ablation.
- `brax_sac_mem84_framestack3_replay5k_CartpoleSwingup_s23`: fit in memory at ~8.8 GiB with stacked obs `(84, 84, 9)`, but stayed flat through 40k steps (`ER=-3900.07`, `α=0.289`). Stopped and replaced with the RAD entropy ablation.
- `brax_sac_mem84_framestack3_entropy_replay5k_CartpoleSwingup_s23`: fit in memory at ~8.8 GiB and preserved the intended RAD entropy regime (`α=0.095` at 4k, `α=0.063` at 40k), but stayed flat (`ER=-3900.13` at 40k). Stopped and replaced with the RAD encoder ablation.
- `brax_sac_mem84_radencoder_framestack3_entropy_replay5k_CartpoleSwingup_s23`: launched with frame_stack=3, replay=5k, RAD entropy settings, and `--encoder-arch rad`. First checkpoint showed the first meaningful reward movement (`ER=-3831.31` at 4k, `α=0.095`, ~8.9 GiB), but regressed to `ER=-3900.11` by 12k. Stopped after discovering that the fast trainer was applying `reward_scaling` twice.
- Reward-scaling fix: `rad_brax_sac.py` now stores raw rewards in replay and applies `reward_scaling` only inside the SAC target, matching brax loss semantics. Prior fast runs with `--reward-scaling 0.1` effectively used 0.01.
- `brax_sac_mem84_radencoder_framestack3_entropy_rewardonce_replay5k_CartpoleSwingup_s23`: rerun of the strongest current variant after the reward-scaling fix. Checkpoints: `ER=-3830.45` at 4k, `-3896.58` at 8k, `-3236.18` at 12k, `-2992.75` at 16k, `-2947.38` at 20k, `-2939.43` at 24k, `-2914.43` at 28k, `-2952.28` at 32k, `-2930.66` at 36k, `-2971.22` at 40k. Memory ~8.9 GiB. Reward is no longer flat, but it appears to plateau near `-2.9k`; stopped at 40k to spend GPU time on the next closer-to-RAD ablation.
- `run_brax_sac_radencoder_tied_ablation.sh`: ablation with actor/critic RAD conv tying, actor/alpha update frequency 2, target update frequency 2, and `tau=0.01`. This keeps the local replay/render constraints but moves the fast path closer to original RAD update semantics. First attempt exposed a JAX donation alias bug from directly sharing actor/critic conv buffers; fixed by copying tied values without sharing leaves. Valid checkpoints: `ER=-3889.89` at 4k, `-3891.11` at 8k, `-3886.51` at 12k. Stopped at 12k because the untied reward-once run had already recovered by this point.
- `run_brax_sac_radcrop_ablation.sh`: ablation with 100px render/replay and 84px crops. Training samples use random 84px crops; action selection and eval use center crops. This targets the remaining original-RAD input pipeline gap while keeping the local 5k on-device replay constraint. The first `num_eval_envs=8` attempt reached replay prefill and training compile, but OOMed during the first eval (`RESOURCE_EXHAUSTED`, extra 4.56 GiB allocation after a 10.21 GiB training HLO). A `num_eval_envs=1` retry still OOMed at the same point. A `num_eval_envs=1`, `num_evals=2000` retry also reported the same 10.21 GiB training HLO and OOMed, falsifying the scan-window hypothesis. The trainer now materializes training metrics before launching eval; this changed the failure point to training metric materialization, confirming the training executable itself exceeds the 12 GiB local limit. Relaunched as workdir `brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay5k_batch128_eval1_scan125_CartpoleSwingup_s23` with `batch_size=128`, keeping the 100→84 crop path and replay capacity unchanged. Checkpoints: `ER=-3900.35` at 1k, `-3900.03` at 2k, `-3900.24` at 3k, `-3834.52` at 4k, `-3726.45` at 5k, `-3900.86` at 6k, `-3894.87` at 7k, `-3890.70` at 8k, `-3894.61` at 9k, `-3885.15` at 10k, `-3688.28` at 11k, `-3811.86` at 12k, `-3806.89` at 13k, `-3900.32` at 14k, `-3884.73` at 15k, `-3864.82` at 16k, `-3771.50` at 17k. Memory ~8.9 GiB. Stopped as negative/lagging: by 16k the 84px reward-once run was already at `-2992.75`, so the crop-input gap is not the current bottleneck under local batch/replay constraints.
- `run_brax_sac_radencoder_update8_ablation.sh`: ablation after the crop-input result. The fast trainer previously inserted 8 transitions per vectorized env step but performed only one replay/SGD update, so it was running about one-eighth of the original RAD update density. `rad_brax_sac.py` now honors `--grad-updates-per-step`; this launcher keeps the best 84px reward-once settings and uses `--grad-updates-per-step 8`. The first 4k-window launch was stopped before its first metric because the heavier update density made 4k-step logging too coarse. Relaunched as workdir `brax_sac_mem84_radencoder_framestack3_entropy_rewardonce_replay5k_update8_scan125_CartpoleSwingup_s23` with `num_evals=2000` / scan125 for 1k-step logging. Checkpoints: `ER=-3804.36` at 1k, `-3205.87` at 2k, `-2974.35` at 3k, `-2984.59` at 4k, `-2944.66` at 5k, `-2946.11` at 6k, `-2918.48` at 7k, `-3012.84` at 8k. Memory ~8.9 GiB, SPS ~9-12. Verdict: strongly positive for sample efficiency, but not enough to break the old `~ -2.9k` plateau. Stopped at 8k and moved to a closer original-RAD mix: 100→84 crop, original batch size 32, and update8.
- `run_brax_sac_radcrop_update8_batch32_ablation.sh`: closer original-RAD fast-path mix with 100px replay/render, random 84px train crops, center-cropped policy/eval observations, original batch size 32, and 8 SGD updates per 8-env vector step. Workdir `brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay5k_batch32_update8_scan125_CartpoleSwingup_s23`. Checkpoints: `ER=-3824.69` at 1k, `-3845.51` at 2k, `-3024.89` at 3k, `-2943.46` at 4k, `-2949.41` at 5k, `-2838.95` at 6k, `-2888.55` at 7k, `-2899.96` at 8k, `-2940.58` at 9k, `-2929.07` at 10k, `-2957.12` at 11k, `-2917.45` at 12k. Memory ~8.9 GiB, SPS ~50 after compile. Verdict: best transient local result and first clear break below the old `~ -2.9k` ceiling, but not a durable escape. Stopped at 12k and moved to replay-capacity/type, because the strongest remaining mismatch is still 5k on-device replay versus original 100k CPU replay.
- Reduced-precision replay implementation note: `rad_brax_sac.py` now supports a flat replay buffer with configurable `--replay-pixel-dtype` (`float32`, `float16`, or `bfloat16`). Brax's stock flattening buffer promotes the whole transition to one dtype, so the initial attempt to preserve mixed dtypes with a typed tree replay buffer passed smoke but OOMed in the full 10k run at the epoch boundary, trying to allocate another `3.50 GiB` replay-sized output. The implementation was switched to one flat reduced-precision storage array, matching Brax's aliasing-friendly update pattern; samples are cast back to float32 before augmentation/network use. A range check showed MuJoCo Playground pixels are normalized float32 values around `[-0.47, 0.28]`, so literal uint8 replay would need a separate inverse normalization rather than a naive cast. Smoke run `runs/smoke_flat_replay_f16` passed through 5k smoke steps with crop, frame-stack, RAD encoder, and float16 replay. Next launcher: `run_brax_sac_radcrop_update8_batch32_replay10k_f16_ablation.sh`, using the closest local RAD mix with `max_replay_size=10000` and `--replay-pixel-dtype float16`.
- `run_brax_sac_radcrop_update8_batch32_replay10k_f16_ablation.sh`: replay-capacity ablation using the closest local RAD mix but increasing replay from 5k float32-equivalent storage to 10k float16 flat storage. The first typed-tree replay attempt OOMed before metric 1; the flat replay retry fit at ~8.9 GiB. Checkpoints: `ER=-3895.15` at 1k, `-3837.91` at 2k, `-3152.07` at 3k, `-2989.53` at 4k, `-2902.89` at 5k, `-2921.93` at 6k, `-2867.51` at 7k, `-2879.86` at 8k, `-2869.99` at 9k, `-2880.30` at 10k, `-2882.64` at 11k, `-2878.72` at 12k, `-2851.21` at 13k, `-2890.99` at 14k, `-2914.81` at 15k, `-2907.30` at 16k, `-2879.62` at 17k. Verdict: positive and more durable than replay5k, with best sustained local result around `-2.85k`, but still plateaus well short of target RAD performance. Next ablation should separate replay capacity from float16 storage by trying the largest local float32 replay that fits.
- `run_brax_sac_radcrop_update8_batch32_replay7500_f32_ablation.sh`: replay precision/capacity ablation. Keeps the closest local RAD mix and flat replay implementation, but stores replay in float32 with capacity 7.5k. This tested whether replay10k's plateau was from remaining capacity limits or from float16 storage precision. Result: OOM before the first metric. Prefill HLO warned about `10.06 GiB`; the first training executable then reported `15.23 GiB` and failed while materializing `env_steps` with `RESOURCE_EXHAUSTED` on a `5.03 GiB` allocation. Verdict: on the RTX 3060, float32 replay above 5k is not a viable local path with the current crop+update8 executable. Next capacity push should stay reduced precision (`float16`/`bfloat16`) or move replay off-device/CPU.
- `run_brax_sac_radcrop_update8_batch32_replay12500_f16_ablation.sh`: reduced-precision replay capacity push. Same closest local RAD mix and flat replay, but increases float16 replay from 10k to 12.5k with the usual `scan125` epoch window. Result: OOM before the first metric. Prefill compiled with an `8.39 GiB` executable and survived, but the first training executable reported `12.72 GiB` after rematerialization and failed while materializing `env_steps` with `RESOURCE_EXHAUSTED` on a `4.19 GiB` allocation. Verdict: 12.5k float16 is over the local RTX 3060 limit with the current `scan125` update executable. Next step is to test whether shortening the compiled epoch window can fit the same replay capacity; if not, the capacity path needs off-device/CPU replay or a substantially more memory-efficient update executable.
- `run_brax_sac_radcrop_update8_batch32_replay12500_f16_scan25_ablation.sh`: memory-fit retry for 12.5k float16 replay with the compiled epoch shortened from `scan125` to `scan25` (`num_evals=10000`). Result: OOM before the first metric, identical to the `scan125` run. Prefill again used an `8.39 GiB` executable; first training again reported `12.72 GiB` and failed on a `4.19 GiB` allocation while materializing `env_steps`. Verdict: scan length is not the memory lever. The pressure comes from carrying/returning the full on-device replay buffer through the compiled training state. The next capacity experiment should move replay off-device/CPU or otherwise remove the multi-GB replay array from the jitted epoch carry.
- Host replay implementation note: `rad_brax_sac.py` now supports `--replay-backend host`, which stores the same flat reduced-precision transition array in NumPy CPU memory and samples device batches for jitted SGD. Env stepping and update chunks remain JIT compiled, but replay insertion/sampling happens in Python, so this trades throughput for removing the multi-GB replay array from the compiled epoch carry. Smoke run `runs/smoke_host_replay_f16` passed through prefill, two update/eval checkpoints, and logging with crop, frame-stack, RAD encoder, update2, and float16 host replay. This backend matches original RAD replay capacity in count when run with 100k, but still stores normalized MuJoCo Playground pixels as float16 rather than original uint8 DMC frames.
- `run_brax_sac_radcrop_update8_batch32_host100k_f16_ablation.sh`: closest local RAD mix with `max_replay_size=100000`, `--replay-backend host`, and `--replay-pixel-dtype float16`. Host replay allocated `33.53 GiB`; GPU use during training stayed around `2.6 GiB`, confirming that replay was removed from the jitted GPU carry. Throughput started around 35 SPS after compile, then slowed to ~24 SPS as the host replay filled. Checkpoints: `ER=-3889.04` at 1k, `-3032.40` at 3k, `-2873.81` at 7k, `-2869.96` at 10k, `-2829.03` at 23k, `-2790.36` at 27k, `-2757.92` at 30k, `-2724.36` at 31k, `-2729.50` at 32k, best `-2719.43` at 33k, `-2720.28` at 37k, then regression/plateau: `-2796.28` at 40k, `-2805.18` at 46k, `-2786.03` at 50k, `-2802.02` at 52k. Stopped after 52k. Verdict: strongly positive and the best local result so far, improving the prior replay10k float16 best by ~132 reward points (`-2851.21` -> `-2719.43`). Replay capacity was a real bottleneck, but host100k did not keep climbing after the low-30k window; next ablation should combine host replay with closer original RAD update cadence/tau or the remaining encoder-target exactness gap.
- `run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_ablation.sh`: follow-up on the winning host100k setup. Keeps 100k host float16 replay, crop, frame-stack, RAD encoder, batch32, and update8, then moves closer to original RAD update cadence with `actor_update_frequency=2`, `alpha_update_frequency=2`, `target_update_frequency=2`, and `tau=0.01`. This intentionally does not enable actor/critic encoder tying yet, so the effect of cadence/tau can be read separately. Checkpoints: `ER=-3884.15` at 1k, `-3058.22` at 3k, `-2921.52` at 10k, `-2904.32` at 20k, `-2862.50` at 23k, `-2853.28` at 26k, `-2774.98` at 27k, `-2743.62` at 28k, `-2726.66` at 29k, best `-2716.32` at 30k, then `-2736.35` at 31k, `-2729.82` at 32k, `-2738.30` at 33k, and `-2797.20` at 38k. Stopped after 38k. Verdict: tiny positive peak improvement over plain host100k (`-2719.43` -> `-2716.32`), but the curve is not more durable and the early/mid curve is worse until the late-20k jump. Keep `tau=0.01`/freq2 as plausible but not decisive; the next exactness target is separate critic/encoder target tau or actor/critic encoder-copy semantics.
- Separate encoder target tau implementation note: `rad_brax_sac.py` now supports `--encoder-tau`; when it is positive, target critic leaves under `PixelEncoderRAD_*` use that EMA coefficient while the rest of the critic target uses `--tau`. Default `encoder_tau=0.0` preserves the old all-leaves `tau` behavior. Smoke run `runs/smoke_host_replay_encoder_tau_f16` passed with host replay, RAD cadence, `tau=0.01`, and `encoder_tau=0.05`.
- `run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_enctau_ablation.sh`: target-encoder tau follow-up on the RAD-cadence host100k run. Keeps `tau=0.01`, actor/alpha/target freq2, and adds original RAD-style `encoder_tau=0.05` for target critic encoder leaves. Checkpoints: `ER=-3836.74` at 1k, `-2875.96` at 10k, `-2913.67` at 20k, late improvement to `-2798.72` at 25k, `-2748.41` at 26k, best `-2723.01` at 27k, then regression to `-2754.98` at 28k, `-2802.44` at 29k, and `-2774.17` at 30k. Stopped at 30k. Verdict: near miss but no new best; separate encoder target tau did not improve over RAD-cadence host100k (`-2716.32` at 30k) and appears less durable after its 27k peak. Keep `encoder_tau=0.05` as an exactness option, but do not treat it as the current winning local setting.
- `run_brax_sac_radcrop_update8_batch32_host100k_f16_radfreq_tied_ablation.sh`: actor/critic RAD conv-copy follow-up on the current-best RAD-cadence host100k run. Keeps 100k host float16 replay, crop, frame-stack, RAD encoder, batch32, update8, `tau=0.01`, and actor/alpha/target freq2, then enables `--tie-actor-critic-encoder`. Checkpoints: `ER=-3886.66` at 1k, `-2931.98` at 10k, `-2884.69` at 20k, `-2788.33` at 24k, regression to `-2901.19` at 26k, late recovery to `-2762.32` at 30k, best `-2728.40` at 33k, then `-2735.48` at 34k and `-2733.72` at 35k. Stopped after 35k. Verdict: competitive but no new best; conv-copy tying improves into the low `-2.7k` band, but under this implementation it remains worse than untied RAD-cadence host100k (`-2716.32`) and plain host100k (`-2719.43`). This suggests the remaining gap is not solved by actor conv copying alone; the next fidelity target should be true shared/detached encoder semantics or true uint8/inverse-normalized host replay, not the current copy-only tie.
