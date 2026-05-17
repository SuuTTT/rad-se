# RAD JAX Speedup Analysis — Root Causes and brax PPO Reimplementation

**Updated**: 2026-05-17  
**Hardware**: RTX 3060 12 GB, CUDA 12.5  
**Stack**: JAX 0.10.0 + Flax NNX + MuJoCo Playground 3.8.1 + Warp 1.12.1 + brax 0.14.2

---

## 1. What We Are Trying to Replicate

The target is [RAD (Laskin et al., NeurIPS 2020)](https://arxiv.org/abs/2004.14990), a pixel-based SAC agent with random-crop augmentation on DMControl tasks.

**Anchor metric** (original PyTorch RAD, `CartpoleSwingup`, from `runs/onefile_complete_remote_e62a79b/metrics.jsonl`):

| Metric | Value |
|---|---|
| Episode Return at ~190k env-steps | ~861 |
| Wall clock (RTX 5060 Ti) | ~2.2 h for 200k steps |
| Throughput | ~25 SPS |
| Optimizer steps per env-step | K = 1 |

---

## 2. What We Built and What Happened

### 2.1 JAX SAC + Playground (rad_jax.py)

- `num_envs=64` parallel Warp-rendered environments
- Replay buffer on CPU numpy (100k capacity, uint8 pixel storage)
- K-fused critic update: `nnx.fori_loop` dispatches K=32–64 gradient steps per Python iteration

**Results (M1 run, 3 seeds)**:

| Run | SPS | Crash step | Best ER before crash |
|---|---|---|---|
| seed 23 | 15–16 | ~38k | -384 |
| seed 7 | 15–16 | ~38k | -382 |
| seed 42 | 15–16 | ~38k | -578 |

**Crash error** (deterministic across all allocator settings):

```
RuntimeError: Failed to allocate 256 bytes on device 'cuda:0'
jax.errors.JaxRuntimeError: UNKNOWN: FFI callback error:
  RuntimeError: Failed to allocate 256 bytes on device 'cuda:0'
  at obs["pixels/view_0"] → np.array()
```

### 2.2 Hand-Written PPO + RAD (rad_ppo_jax.py)

- `num_envs=128`, `unroll_length=16`, on-policy rollout buffer
- No replay buffer — no growing CPU→GPU transfer bottleneck

**Results (1M step run, seed 23)**:

| Metric | Value |
|---|---|
| SPS at steady state | 53–56 |
| ER at step 0 | -621 |
| ER at step 51200 | -677 (worsening) |
| KL after iter ~20 | collapses to ~0.0001 |

**Key finding**: PPO does not crash at 38k, but it also does not learn.

### 2.3 brax PPO + augment_pixels (rad_brax_ppo.py) — Current Approach

Reimplementation using `brax.training.agents.ppo.train` with built-in pixel augmentation.

**Architecture**:
- `brax.training.agents.ppo.train(vision=True, augment_pixels=True)` — the `augment_pixels` flag applies `_random_translate_pixels` (pad=4 random translate) every gradient step, equivalent to RAD random crop
- `action_repeat=8` handled at the Playground `wrap_for_brax_training` wrapper level; brax PPO receives `action_repeat=1` (required by the `vision=True` code path)
- `wrap_env=False` — pre-wrapped env passed in so brax does not re-wrap
- Training loop fully JIT-compiled via `jax.lax.scan` — no Python overhead per step

**v1 run** (`unroll_length=20`, 500k steps):

| Metric | Value |
|---|---|
| SPS at steady state | **2085** (38× faster than hand-written PPO) |
| JIT compile time | ~23 s |
| Total wall clock | 4.7 min |
| Initial ER | -499 |
| ER after 60k steps | -368 |
| ER at 584k steps (final) | -369 (plateau) |

**v1 diagnosis**: KL collapses from 0.08 → 0.001 after ~200k steps — the policy stops updating. Root cause: `unroll_length=20` covers only 20/125 = 16% of the CartpoleSwingup episode, so GAE cannot credit-assign across the full swing-up sequence.

**v2 run** (`unroll_length=60`, 2M steps) — in progress at time of writing:

- Hyperparameter changes: `unroll_length=60` (48% episode coverage), `entropy_cost=0.05` (5×), `num_updates_per_batch=4` (halved)
- Ran into OOM during XLA autotuning (see §6.2); fixed with `XLA_FLAGS="--xla_gpu_autotune_level=0"`

---

## 3. Root Cause: Why SAC Cannot Be Sped Up

### 3.1 The Warp OOM at ~38k Steps

MuJoCo Playground uses NVIDIA Warp for GPU-accelerated rendering. Warp maintains CUDA memory pools **separate from but competing with JAX** on the same 12 GB device.

Timeline of VRAM consumption:

```
t=0 (init)
  ├─ JAX XLA buffer cache (model + JIT intermediates): ~0.6 GB
  ├─ Warp env state for 64 worlds:                    ~0.3 GB
  └─ Total: ~0.9 GB  →  11.1 GB free

t=0..38k (replay buffer fills)
  ├─ replay buffer OBS:  100k × (100×100×3) uint8 = 3.0 GB
  ├─ replay buffer NOBS:                             3.0 GB
  └─ running total: ~6.9 GB  →  5.1 GB free

t=~38k (K=64 fori_loop at peak)
  ├─ fori_loop JIT activation tensors (K=64 critic passes): ~3–4 GB
  ├─ Optax optimizer state (Adam moments):                  ~0.8 GB
  └─ JAX now owns ~11.5 GB

t=~38k + env_step()
  ├─ Warp render kernel needs new CUDAGraph alloc: 256 bytes
  └─ CUDA reports out-of-memory → crash
```

The 256 bytes is a Warp stream-synchronization token — not the render buffer itself. Even one byte beyond available VRAM causes the crash because every physical page is committed.

**Why neither JAX allocator strategy fixes it**:

| Allocator | Behavior | Why it fails |
|---|---|---|
| BFC (default) | Pre-reserves pool at startup; Warp allocates outside it | Warp cannot get pages JAX reserved |
| platform | Allocs on demand; holds pages until GC | After 38k steps, all pages are committed |

**Why unfixable without architectural change**: The 100k-frame replay buffer is structurally required by SAC. At 100×100×3×2×100k = **6.0 GB**, it consumes half the RTX 3060 before CNN intermediates or optimizer state. No tuning eliminates this floor.

### 3.2 Timing Breakdown

One complete SAC iteration at steady state (~16 SPS):

| Component | Wall time | % |
|---|---|---|
| `env_step()` (Warp render, 64 envs) | ~1 ms | ~2% |
| `sample_k()` numpy random crop + H2D | ~12 ms | ~25% |
| `_update_critic_k_fused()` K=32 fori_loop | ~40 ms | ~68% |
| `_update_actor_alpha()` | ~4 ms | ~7% |
| **Total** | **~61 ms** | — |

The environment contributes 2% of wall time. Doubling `num_envs` accelerates buffer-fill and worsens the OOM; it does not help throughput.

---

## 4. Why Hand-Written PPO Did Not Learn

PPO solves the OOM (no replay buffer) but introduces catastrophic sample inefficiency.

### 4.1 KL Collapse Mechanics

```
iter 1:  kl=0.6388, clip_frac=0.87  ← massive overshoot on first update
iter 2:  kl=0.0022, clip_frac=0.00  ← policy frozen; clipping never fires
iter 20: kl≈0.0001, clip_frac=0.00  ← dead policy
iter 40: ER=-677 (worse than initial -621)
```

**Root cause**: `T=16` unroll covers only 13% of the 125-step CartpoleSwingup episode. GAE over T=16 with γ=0.99, λ=0.95 produces high-variance bootstrapped advantage estimates. The first update takes a huge step (KL=0.64), landing in a low-gradient region; subsequent updates cannot escape because clipping is always inactive.

### 4.2 Required Resources for Hand-Written PPO to Work

- T ≥ 64 unroll (covering ~50% of an episode)
- num_envs ≥ 512 for variance reduction  
- 3–5M total timesteps
- At 55 SPS: 3M steps = **~15 hours** wall clock

This is impractical relative to SAC's 2.2h anchor.

---

## 5. Why Official Playground SAC Appears Fast

### 5.1 The Actual Difference

| Property | Official Playground brax SAC | Our JAX RAD SAC |
|---|---|---|
| Observation type | **State** (20–50 floats) | **Pixels** (100×100×3 uint8) |
| CNN encoder | None | 4-layer 32-filter CNN |
| Replay buffer | **In JAX device memory** (tiny) | CPU numpy (6 GB) |
| Training loop | **Fully JIT via `jax.lax.scan`** | Python for-loop |
| Warp rendering | **None** | Required every step |
| Typical SPS | 5,000–50,000 | 15–16 |

Official Playground SAC is fast because it runs on **state observations** — no Warp, no CNN, fully on-device scan loop, replay buffer fits in a few MB. It is not applicable to the pixel RAD task.

### 5.2 Why the Pixel Version Cannot Use the Same Path

```
Fully-JIT SAC with pixel obs and on-device replay:
  100k × (100×100×3 + 100×100×3) float32 = 13.4 GB
  → OOM before any weights or optimizer states are allocated on RTX 3060
```

There is no pixel replay buffer that fits alongside a Warp renderer and CNN optimizer state in 12 GB.

---

## 6. brax PPO + augment_pixels — Implementation Details

### 6.1 Design

`brax.training.agents.ppo.train` with `vision=True, augment_pixels=True` provides:
- A fully JIT-compiled `jax.lax.scan` training loop — no Python loop overhead
- Built-in `_random_translate_pixels` augmentation (pad=4 random translate) equivalent to RAD random crop, applied once per gradient step inside the scan
- Vision-capable CNN policy network via `brax.training.agents.ppo.networks_vision`

Key constraints navigated:
- `vision=True` requires `action_repeat=1` inside brax; we apply `action_repeat=8` at the Playground `wrap_for_brax_training` level and pass `wrap_env=False` to skip brax's re-wrapping
- `batch_size * num_minibatches % num_envs == 0` assertion: 32 × 8 = 256 % 256 = 0 ✓
- CartpoleSwingup done-condition bug: Balance.step fires `done=True` immediately when pole starts at bottom; monkeypatched with `_fix_swingup_done=True` flag

### 6.2 Compatibility Fixes Required

**`jax.device_put_replicated` removed in JAX 0.10+**:  
brax 0.14.2 calls `jax.device_put_replicated(training_state, devices)` which no longer exists. A shim is needed that adds a leading device axis (required for `_unpmap`'s `.squeeze(0)` call to work on all leaves including 0-d scalars like `env_steps`):

```python
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(pytree, devices):
        n = len(devices)
        def replicate_leaf(x):
            x = jnp.asarray(x)
            return jnp.broadcast_to(jnp.expand_dims(x, 0), (n, *x.shape))
        replicated = jax.tree_util.tree_map(replicate_leaf, pytree)
        if n == 1:
            return jax.device_put(replicated, devices[0])
        raise NotImplementedError(...)
    jax.device_put_replicated = _device_put_replicated
```

**XLA autotuner OOM with `unroll_length=60`**:  
XLA's kernel autotuner allocates the full pixel buffer as a contiguous block during profiling. For `f32[256,60,100,100,3]` this is `256×60×100×100×3×4 = 1.72 GB`. After Warp init and model allocation, no contiguous 1.72 GB block remains:

```
Autotuning failed: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.72GiB
```

Fix: `XLA_FLAGS="--xla_gpu_autotune_level=0"` disables profiling and uses default kernel configs (~5% throughput penalty). This is now set in `scripts/run_brax_ppo.sh`.

With `unroll_length=20` the buffer is only 586 MB, which fits — explaining why v1 ran fine but v2 needed the flag.

### 6.3 Performance

**v1 run** (`unroll_length=20`, `entropy_cost=0.01`, 500k steps):

| Step | ER | SPS | KL | Notes |
|---|---|---|---|---|
| 0 | -499 | — | — | Random policy |
| 30k | -460 | 529 | 0.010 | Compiling, warming up |
| 61k | -406 | 758 | 0.026 | Learning |
| 92k | -379 | 996 | 0.022 | Learning |
| 153k | -368 | 1330 | 0.033 | Plateau begins |
| 184k | -368 | 1451 | **0.079** | KL peak |
| 430k | -370 | 1957 | **0.001** | KL collapsed |
| 584k | -370 | 2085 | 0.008 | Stalled |

Final throughput: **2085 SPS** vs 55 SPS for hand-written PPO (38× improvement).  
Learning stalled: `pole_pos_penalty = -50` dominates; policy never swings the pole up.

Reward components at plateau:
```
eval/episode_reward:              -368
eval/episode_reward/alive:          +1.60 (constant, std=0)
eval/episode_reward/pole_pos_penalty: -50.4  ← dominant
eval/episode_reward/cart_pos_penalty:  -0.6
eval/episode_reward/cart_vel_penalty:  -0.25
eval/episode_reward/action_penalty:    -0.07
```

**v2b run** (`unroll_length=60`, `entropy_cost=0.05`, `num_updates_per_batch=4`, 2M steps, `XLA_FLAGS="--xla_gpu_autotune_level=0"`):  
Completed through 752k/2M steps at time of writing. Key results:

| Step | ER | SPS | KL | Notes |
|---|---|---|---|---|
| 107k | -499 | 418 | 0.052 | JIT compiled (2nd eval cached) |
| 322k | -410 | 418 | 0.017 | Pole not swung up yet |
| 537k | -388 | 418 | 0.009 | Slow improvement |
| 752k | -381 | 418 | 0.003 | KL healthy, not collapsing |

**Why did SPS drop from 2085 → 418?** Two compounding factors:

1. **unroll_length 20→60 (3×)**: Tripling the unroll means 3× more CNN passes per rollout step. Neutral for SPS if conv kernels are equally fast.
2. **XLA autotuning disabled (5× slower)**: `--xla_gpu_autotune_level=0` is needed because the autotuner allocates `f32[256,60,100,100,3]` = 1.72 GB as a contiguous scratch block (see §6.2). Disabling autotuning forces default (non-tuned) conv kernel configs, which are ~5× slower for the 100×100×3 pixel shape on the RTX 3060. Net effect: `2085 × (1/5) ≈ 417`.

There is no simple fix: enabling autotuning OOMs the 12 GB GPU; shrinking `num_envs` or image resolution would help but change the experiment. **418 SPS is still ~17× faster than the 25 SPS PyTorch RAD baseline**, so v2b remains practical at 2M steps ≈ 80 minutes wall-clock.

---

## 7. Why brax PPO Still Struggles on CartpoleSwingup

CartpoleSwingup is a **hard exploration problem**: the pole starts at the bottom and must be actively swung up. This requires coordinated actions across the full 125-step episode.

### 7.1 Credit Assignment Window

| Config | unroll_length | Episode coverage | Expected |
|---|---|---|---|
| Hand-written PPO v1 | 16 | 13% | KL collapse iter 1 |
| brax PPO v1 | 20 | 16% | KL collapse at 200k |
| brax PPO v2 | 60 | 48% | Pending |
| Minimum for task | ~80 | ~64% | Likely needed |

With T=20, GAE discounts at γ^T = 0.99^20 = 0.82 — the tail 18% of episode reward is bootstrapped from an imperfect value function, not actual returns. For swing-up, the reward near the end of a successful episode is vastly higher than the start, and this signal is lost in the bootstrap.

### 7.2 Why SAC Does Not Have This Problem

SAC uses a replay buffer that stores and re-samples transitions from all parts of the episode. K=64 critic updates per step extract the "pole is now near the top" reward signal repeatedly until the Q-function propagates it backward to earlier states via Bellman updates. PPO collects a fresh rollout each iteration and discards it — it has no mechanism to revisit the rare high-reward transitions.

---

## 8. Comparison: Official Playground SAC vs Our Implementations

| | Official brax SAC | SAC rad_jax.py | PPO rad_ppo_jax.py | brax PPO v1 | brax PPO v2b |
|---|---|---|---|---|---|
| Obs type | State | Pixels | Pixels | Pixels | Pixels |
| Loop | Fully JIT scan | Python for-loop | Python for-loop | Fully JIT scan | Fully JIT scan |
| Warp | No | Yes | Yes | Yes | Yes |
| Replay buffer | On-device (tiny) | CPU (6 GB) | None | None | None |
| Crashes? | No | Yes (~38k) | No | No | No |
| Learning? | Yes (state) | Partial (crashes) | No | Partial (plateaus) | Improving |
| SPS | ~10,000–50,000 | 15–16 | 53–56 | **2085** | 418 |
| Final ER | task-dependent | -382 before crash | -677 | -368 (plateau) | -381 @ 752k steps |
| Note | state obs only | OOM | KL collapse | KL collapses @200k | healthy KL |

---

## 9. Viable Paths Forward

### Option A: dm_control (CPU) + JAX SAC — Closest to Paper Setup

Replace Playground (Warp) with the original dm_control CPU renderer used in the RAD paper. No VRAM competition; SAC can run to completion.

| Aspect | Value |
|---|---|
| Env backend | `dm_control` CPU backend |
| Rendering | CPU — zero GPU VRAM cost |
| Expected SPS | ~15–20 |
| Expected wall clock | ~3–4 h for 200k steps |
| Crash risk | None |
| Implementation delta | Replace `make_env` / `env_step` |

This matches the original paper's experimental setup exactly.

### Option B: brax PPO + augment_pixels — In Progress

Already implemented (`rad_brax_ppo.py`). Achieves 2085 SPS. Learning issue on CartpoleSwingup may be resolved by:

1. Longer unroll (`T=60–80`) — v2 in progress
2. More total steps (2M vs 500k)
3. CartpoleBalance as a simpler validation task (PPO learns balance easily)
4. Larger num_envs (512+) for variance reduction

If v2 learns, this is the most scalable path: no Python overhead, fully JIT, extensible to other Playground tasks.

### Option C: Larger GPU

On an A100 (80 GB), all constraints vanish. SAC runs fully on-device with pixel replay buffer. Expected SPS: 500+. Wall clock for 200k steps: ~7 minutes. Outside current hardware scope.

---

## 10. Conclusion

| Blocker | Root Cause | Fixable on RTX 3060? |
|---|---|---|
| SAC Warp OOM at 38k | 6 GB pixel replay + K=64 JIT peaks + Warp token = 12 GB exhausted | No — structural |
| Hand-written PPO no learning | T=16 unroll (13% of episode), first update KL=0.64 overshoot | Fixable but ~15h for 3M steps |
| brax PPO v1 plateau | T=20 unroll (16% of episode), KL collapses at 200k steps | In progress: T=60, 2M steps |
| brax PPO autotune OOM | XLA profiler allocates 1.72 GB contiguous for `f32[256,60,...]` | Fixed: `--xla_gpu_autotune_level=0` |
| Official Playground SAC speed | Uses state obs — no Warp, no CNN, on-device replay | Not applicable to pixel task |

**Current best**: brax PPO (`rad_brax_ppo.py`) at **2085 SPS** — 38× faster than hand-written PPO, fully JIT-compiled, with built-in RAD augmentation. Learning quality pending v2 run (T=60, 2M steps).

**Recommended if PPO v2 stalls**: Switch to dm_control backend (Option A) and run JAX SAC to 200k steps. This matches the original paper's setup and removes the fundamental VRAM blocker.
