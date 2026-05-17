# RAD SAC Iteration Log — CartpoleSwingup

---

## Iteration 1 — 2026-05-17
**Status:** ❌ Killed at 400k (plateau + entropy collapse; seed 42)

### What Was Launched
```bash
bash scripts/run_brax_sac.sh CartpoleSwingup 42
```

### Config
| param | value |
|-------|-------|
| seed | 42 |
| num_envs | 8 |
| num_eval_envs | 16 |
| max_replay_size | 10 000 |
| min_replay_size | 1 000 |
| batch_size | 256 |
| total_timesteps | 500 000 |
| num_evals | 20 |
| reward_scaling | 0.1 |
| action_repeat | 8 |
| XLA autotune | **OFF** (`--xla_gpu_autotune_level=0`) |
| GPU | RTX 3060 12 GB, CUDA 12.5 |
| work_dir | runs/brax_sac_CartpoleSwingup_s42/ |

### Bugs Fixed Before This Run
1. Step count 8× too low — was dividing `total_timesteps` by `action_repeat` unnecessarily (wrapper already handles it)
2. `steps_per_eval` units mismatch — was computing in env_steps, using as scan_steps
3. Dead `evaluate()` JIT function — defined but never called, removed
4. `reward_scaling` 1.0 → 0.1 — SAC Q-values were 10× too large, unstable gradients

### Results
| step | SPS | ER | critic_loss | actor_loss | alpha | elapsed |
|------|-----|----|-------------|------------|-------|---------|
| 25 000 | 28 | -3900.29 | 0.004 | -2.16 | 0.446 | 14.7 min |
| 50 000 | 28 | -3900.17 | 0.003 | -1.94 | 0.218 | 29.1 min |
| 75 000 | 28 | -3664.07 | 0.004 |  0.06 | 0.109 | 43.6 min |
| 100 000 | 28 | -3478.06 | 0.003 |  2.49 | 0.055 | 58.1 min |
| 125 000 | 28 | -3103.83 | 0.003 |  5.19 | 0.028 | 72.5 min |
| 150 000 | 28 | -3082.21 | 0.001 |  7.83 | 0.014 | 87.0 min |
| 175 000 | 28 | -2955.80 | 0.001 | 10.16 | 0.008 | 101.4 min |
| 200 000 | 28 | -3023.13 | 0.001 | 12.18 | 0.004 | 115.9 min |
| 225 000 | 28 | -2926.89 | 0.001 | 14.06 | 0.002 | 130.4 min |
| 250 000 | 28 | -3008.70 | 0.001 | 15.71 | 0.001 | 144.8 min |
| 275 000 | 28 | -2894.89 | 0.002 | 17.14 | 0.001 | 159.3 min |
| 300 000 | 28 | -3019.91 | 0.002 | 18.52 | 0.003 | 173.8 min |
| 325 000 | 28 | -3140.00 | 0.005 | 20.05 | 0.006 | 188.2 min |
| 350 000 | 28 | -3000.84 | 0.002 | 21.22 | 0.004 | 202.7 min |
| 375 000 | 28 | -2990.92 | 0.002 | 22.15 | 0.002 | 217.2 min |
| 400 000 | 28 | -3136.13 | 0.002 | 22.86 | 0.002 | 231.7 min |

### Observations
- SPS **exactly 28** at every eval — fully compute-bound on CNN (autotune=OFF uses
  generic GEMM convolution, no cuDNN Winograd/FFT, no tensor core path)
- ER improves from -3900 → ~-2900 between steps 25k–175k, then plateaus
- alpha (temperature) decays fast: 0.446 → 0.001 by step 250k (entropy collapsed)
- actor_loss keeps rising (+1.5/eval) — actor is improving but entropy gone
- Total run time: ~4.9 hr for 500k steps

### Bottleneck Diagnosis
Autotune=OFF on CUDA 12.5 was unnecessary. CUDA ≥ 12.3 supports nested stream
capture → JAX autotune and Warp physics can coexist. The flag was added
conservatively for CUDA 12.1 (3090) and never revisited.
CNN autotune scratch peak (observed in iter 2): **8780 MiB** (73% of 12 GB) —
not the ~800 MB estimated. Still fits; 3162 MiB headroom. No OOM.

---

## Iteration 2 — 2026-05-17
**Status:** ✅ Running — seed=23, autotune_level=2, local RTX 3060

### What Changed
- **Removed `--xla_gpu_autotune_level=0`** from `run_brax_sac.sh`
- Enabled partial autotune: `--xla_gpu_autotune_level=2` (profiles top-5 candidates,
  ~30s startup vs ~2min for full level=4, but still gets most of the speedup)

### Launch Command
```bash
bash scripts/run_brax_sac.sh CartpoleSwingup 23
```

### Config Changes vs Iter 1
| param | iter 1 | iter 2 |
|-------|--------|--------|
| seed | 42 | **23** |
| XLA autotune | OFF (level=0) | **ON (level=2)** |
| everything else | same | same |

### Predicted vs Actual
| metric | iter 1 | iter 2 predicted | iter 2 actual |
|--------|--------|-----------------|---------------|
| SPS | 28 | 90–110 | **148→170** (+6×) |
| 500k steps | ~4.9 hr | ~1.4 hr | **49.1 min** |
| startup (autotune) | ~15 min | ~3 min | **2.8 min** |

### Results
| step | SPS | ER | critic | actor | alpha | elapsed |
|------|-----|----|--------|-------|-------|---------|
| 25 000  | 148 | -3900.41 | 0.004 | -2.21 | 0.446 | 2.8 min |
| 50 000  | 158 | -3900.05 | 0.003 | -1.94 | 0.218 | 5.2 min |
| 75 000  | 162 | -3762.05 | 0.003 |  0.10 | 0.109 | 7.7 min |
| 100 000 | 164 | -3179.90 | 0.003 |  2.51 | 0.055 | 10.1 min |
| 125 000 | 165 | -3037.76 | 0.002 |  5.17 | 0.029 | 12.6 min |
| 150 000 | 166 | -3025.18 | 0.001 |  7.79 | 0.015 | 15.0 min |
| 200 000 | 167 | -2989.74 | 0.001 | 12.33 | 0.004 | 19.9 min |
| 300 000 | 168 | -2997.91 | 0.001 | 18.86 | 0.003 | 29.6 min |
| 400 000 | 169 | -3108.09 | 0.002 | 22.57 | 0.003 | 39.3 min |
| 500 000 | 169 | -2919.91 | 0.001 | 25.18 | 0.002 | 49.1 min |

**Final: [done] 500000 steps in 2945.2s (170 SPS)**

### Observations
- SPS ramps 148 → 170 over first 50k steps as XLA JIT caches warm up, then stable
- ER plateau is **identical to iter 1**: both seeds saturate ~-2900 to -3100 after step 125k
- Alpha decays fast again: 0.446 → 0.002 by step 250k (entropy collapse, same as iter 1)
- Actor loss keeps rising (+1.6/eval) with no ER improvement — classic overestimated Q-values
- Compute bottleneck is fixed. RL bottleneck is now the algorithm / task / hyperparams.

### Root Cause of ER Plateau
1. **replay_size=10k too small** — 10k × 8 envs = 1250 unique env-steps; highly correlated
   batches → poor gradient diversity
2. **entropy collapse** — alpha → 0 too fast; policy stops exploring at ~200k steps
3. **CartpoleSwingup is sparse-ish** — most episodes fail (ER ≈ episode_length × -1 ≈ -1000 × ~3);
   hard to learn cartpole swing from pixels in 10k replay with 8 envs

### Next Steps
- Increase `max_replay_size` 10k → 100k (now affordable: 49 min per 500k steps)
- Increase `total_timesteps` 500k → 2M (only ~3.3 hr at 170 SPS)
- Consider higher `target_entropy` or initial `alpha` to slow entropy decay

---
