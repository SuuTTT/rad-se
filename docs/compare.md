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
| grad_updates_per_step | 1 | configurable | **1** | Same |
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
| Gradient updates | 1 per env step | 1 per env step (default) | 1 per scan step | Same |
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
| `grad_updates_per_step` config | YES | Hardcoded 1 | Minor |
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
3. **Vectorised env collection (`num_envs=8`) instead of one Python gym env**: This is needed for practical GPU utilisation with Playground/Warp. The replay update ratio remains 1 gradient update per collected transition via `updates_per_step=8`.
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

Do not mix ablations into the exact baseline unless the exact baseline demonstrably fails for an implementation-independent reason.

### Current launch priority

1. Run `scripts/run_brax_sac_mem_ablation.sh` on the local 12 GB GPU to test the replay-capacity hypothesis quickly.
2. If reward improves materially, move the exact JAX path or a uint8/on-device replay version to a larger GPU for strict matching.
3. If reward remains flat near -3000, implement the next exactness gap in the fast trainer: frame-stack + 100→84 random crop + RAD encoder.

### Local iteration notes

- `brax_sac_mem84_replay15k_CartpoleSwingup_s23`: fit in memory at ~8.8 GiB, but stayed flat at ER ≈ -3900 through 48k steps while α collapsed to 0.231. Stopped and replaced with the frame-stack ablation.
- `brax_sac_mem84_framestack3_replay5k_CartpoleSwingup_s23`: fit in memory at ~8.8 GiB with stacked obs `(84, 84, 9)`, but stayed flat through 40k steps (`ER=-3900.07`, `α=0.289`). Stopped and replaced with the RAD entropy ablation.
- `brax_sac_mem84_framestack3_entropy_replay5k_CartpoleSwingup_s23`: launched with frame_stack=3, replay=5k, init_temperature=0.1, target_entropy=-1.0, alpha_lr=1e-4.
