# GPU Memory & Configuration Guide for Pixel RL

This document explains how VRAM capacity affects training speed, replay buffer quality, and
learning performance in pixel-observation RL, and gives concrete recommended configs for
12 GB, 24 GB, and 48 GB+ cards.

All numbers are for **100×100×3 float32 observations** (CartpoleSwingup default).

---

## 1. Where VRAM Goes

### 1.1 Per-observation footprint

```
100 × 100 × 3 × 4 bytes = 120 KB per obs (float32)
One transition (obs + next_obs + action + reward) ≈ 240 KB
```

### 1.2 SAC: replay buffer dominates

`UniformSamplingQueue` stores the entire replay buffer as a flat float32 array on-device.
It lives in the `scan` carry, so XLA must allocate it for every training epoch.

| `max_replay_size` | VRAM (100×100×3) | Notes |
|---|---|---|
| 1 000 | **0.22 GB** | Too small; buffer cycles in <250 env-steps (4 envs) |
| 5 000 | **1.12 GB** | Minimum useful; ~1250 env-steps per cycle |
| 10 000 | **2.24 GB** | Default for 12 GB card |
| 50 000 | **11.18 GB** | Exceeds 12 GB when combined with other overhead |
| 100 000 | **22.35 GB** | Requires 24 GB+, matches RAD paper setting |

**Impact on learning:** A larger replay buffer provides more temporal diversity.
SAC's Bellman update propagates reward backward over many replay samples, so a bigger buffer
means slower value-function drift and more robust critic learning.
With only 10k transitions (2.24 GB), the buffer cycles ~every 2500 env-steps
and SAC relies on fast temporal credit assignment rather than long-term memory.

### 1.3 PPO: rollout buffer (transient)

The PPO rollout buffer is `num_envs × unroll_length × obs_shape` and lives inside the
`jax.lax.scan` body. XLA holds it only during the scan trace:

| `num_envs` | `unroll_length` | Rollout buffer | Notes |
|---|---|---|---|
| 256 | 20 | **0.57 GB** | Safe on 12 GB |
| 256 | 60 | **1.72 GB** | Needs `--xla_gpu_autotune_level=0` on 12 GB |
| 128 | 60 | **0.86 GB** | Safe on 12 GB with autotuner |
| 64  | 60 | **0.43 GB** | Conservative |

### 1.4 XLA autotuner scratch space

XLA's kernel autotuner allocates a contiguous copy of the largest tensor in the program to
profile kernel configs. For `f32[256,60,100,100,3]` this is **1.72 GB contiguous**.
After Warp engine init (~1–2 GB fragmented), no contiguous 1.72 GB block remains on a
12 GB card → OOM during compilation.

**Workaround:** `XLA_FLAGS="--xla_gpu_autotune_level=0"`
- Disables profiling, uses default kernel configs
- Costs ~5× throughput on conv-heavy pixel workloads on RTX 3060 (2085 SPS → 418 SPS)
- Not needed if rollout buffer fits in a contiguous block (small `num_envs` or `unroll`)

On 24 GB+ cards this flag is not needed.

### 1.5 Warp renderer overhead

The MuJoCo Warp renderer allocates GPU memory for physics state + rendering pipeline.
The rendered pixel buffer itself is negligible:

| `num_envs` | Pixel output | Warp state |
|---|---|---|
| 4 | 0.5 MB | ~400 MB (solver, BVH, mesh data) |
| 64 | 7.3 MB | ~400 MB (shared solver state) |
| 256 | 29.3 MB | ~400–800 MB |

Warp's base overhead (~400 MB) is approximately constant across env counts.
The solver's Cholesky factorization buffers scale with `num_envs`.

---

## 2. How Memory Limits Affect Speed (SPS)

The two main SPS bottlenecks are:

| Factor | Effect | Remedy |
|---|---|---|
| `num_envs` too low | GPU underutilized; compute waits for env steps | Increase `num_envs` |
| XLA autotune disabled | Default conv kernels ~5× slower for 100×100×3 | Smaller rollout buffer OR more VRAM |
| Replay buffer too large | OOM on first epoch compilation | Reduce `max_replay_size` |
| `batch_size` too small | Low GPU occupancy during gradient steps | Increase to 256–1024 |

**RTX 3060 (12 GB) measured throughput:**

| Config | SPS | Bottleneck |
|---|---|---|
| PPO, unroll=20, envs=256 (autotune on) | **2085** | Near-optimal |
| PPO, unroll=60, envs=256 (autotune off) | **418** | 5× slow convs |
| PPO, unroll=60, envs=128 (autotune on) | ~**900** (est.) | Half envs vs v1 |
| SAC, envs=4, replay=10k | ~**60–80** (est.) | Low env count |

---

## 3. Recommended Configurations

### 3.1 12 GB (RTX 3060, RTX 4070)

**Hard constraints:**
- `max_replay_size ≤ 10000` for SAC (2.24 GB; leaves room for model + optimizer states + Warp)
- Must set `XLA_FLAGS="--xla_gpu_autotune_level=0"` if `num_envs × unroll ≥ 100` or replay ≥ 5k
- `num_envs ≤ 256` for PPO (Warp physics stays under ~1 GB)

**SAC config:**
```bash
--num-envs 4 \
--max-replay-size 10000 \
--batch-size 256 \
--total-timesteps 500000 \
XLA_FLAGS="--xla_gpu_autotune_level=0"
```

**PPO config:**
```bash
--num-envs 256 \
--unroll-length 60 \
--batch-size 32 --num-minibatches 8 \
XLA_FLAGS="--xla_gpu_autotune_level=0"   # required for unroll=60
```

**Expected:** SAC ~60–80 SPS; PPO ~418 SPS (autotune off).

---

### 3.2 24 GB (RTX 4090, RTX 3090, A10)

No autotuner constraint. Replay buffer up to 50k (11.18 GB with 24 GB available after Warp + model).

**SAC config:**
```bash
--num-envs 8 \
--max-replay-size 50000 \
--batch-size 512 \
# No XLA_FLAGS needed
```

**PPO config:**
```bash
--num-envs 512 \
--unroll-length 80 \
--batch-size 64 --num-minibatches 8 \
# No XLA_FLAGS needed
```

**Expected:** SAC ~150–300 SPS with 8 envs; PPO ~6000–8000 SPS with 512 envs + full autotune.

---

### 3.3 48 GB+ (A100, H100, RTX 6000 Ada)

Full RAD paper settings possible: 100k replay buffer (22.35 GB), large `num_envs`.

**SAC config (matches RAD paper spirit):**
```bash
--num-envs 16 \
--max-replay-size 100000 \
--batch-size 512 \
```

**PPO config:**
```bash
--num-envs 1024 \
--unroll-length 80 \
--batch-size 128 --num-minibatches 8 \
```

**Expected:** SAC ~300–600 SPS; PPO ~15000–30000 SPS.

---

## 4. Memory Budget Worksheet

Use this to estimate before launching:

```
VRAM_budget = total_VRAM - warp_overhead - model_and_optimizer

warp_overhead ≈ 0.4 GB + 0.003 × num_envs GB  (empirical, CartpoleSwingup)
model_and_optimizer ≈ 0.3 GB  (CNN policy + twin-Q + 3 Adam states)
JIT_scratch ≈ 1.5 × max(rollout_buffer, replay_buffer)

--- SAC ---
replay_GB = max_replay_size × 60004 × 4 / 1e9
Total = warp + model + replay + JIT_scratch

--- PPO ---
rollout_GB = num_envs × unroll_length × 100 × 100 × 3 × 4 / 1e9
Total = warp + model + rollout + JIT_scratch
```

**12 GB example (SAC, 10k replay):**
```
warp     = 0.4 + 0.003 × 4 = 0.41 GB
model    = 0.30 GB
replay   = 2.24 GB
scratch  = 1.5 × 2.24 = 3.36 GB
Total  ≈  6.3 GB  ← fits comfortably
```

**12 GB example (SAC, 100k replay — the OOM we hit):**
```
replay   = 22.35 GB  ← already over budget alone
```

---

## 5. Learning Quality vs Memory Tradeoff

| Setting | Effect on learning | Effect on speed |
|---|---|---|
| Larger replay buffer | More data diversity, slower critic drift | Uses more VRAM, may need autotune off |
| More parallel envs (PPO) | Less-biased gradient estimates, better throughput | Uses more VRAM for rollout |
| More parallel envs (SAC) | Faster buffer fill but diminishing returns | Minor VRAM increase (env state only) |
| Larger batch size | More stable actor/critic gradients | Linear VRAM increase during backward |
| Higher resolution obs | Quadratic VRAM growth; larger CNN needed | Quadratic slowdown |

**Key insight for this project:** PPO's fundamental problem on CartpoleSwingup is not
memory — it's credit assignment. Even with 2M steps and `unroll=60`, the pole never swings
up. SAC is the right algorithm, but requires a memory-conscious config on 12 GB hardware.

---

## 6. Quick Reference

| VRAM | Algorithm | `num_envs` | `max_replay` / `unroll` | `XLA_FLAGS` | Est. SPS |
|---|---|---|---|---|---|
| 12 GB | SAC | 4 | 10 000 | `autotune_level=0` | 60–80 |
| 12 GB | PPO | 256 | unroll=60 | `autotune_level=0` | ~420 |
| 24 GB | SAC | 8 | 50 000 | none | 200–350 |
| 24 GB | PPO | 512 | unroll=80 | none | ~6 000 |
| 48 GB | SAC | 16 | 100 000 | none | 400–600 |
| 48 GB | PPO | 1024 | unroll=80 | none | ~20 000 |
