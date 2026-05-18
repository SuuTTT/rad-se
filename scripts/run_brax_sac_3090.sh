#!/usr/bin/env bash
# SAC training optimized for 24 GB RTX 3090.
#
# Key differences vs 12 GB (run_brax_sac.sh):
#   max_replay_size : 10_000 → 50_000   (+11 GB, richer experience diversity)
#   num_envs        : 8      → 16        (2× more parallel collection)
#   XLA autotune    : OFF    → ON        (room for 4 GB scratch → 5× faster CNN)
#   num_eval_envs   : 16     → 32
#
# VRAM budget (RTX 3090, 24 GB):
#   replay buffer   : 50000 × 60004 × 4B = ~11.2 GB
#   model+optimizer :                     ~0.6  GB
#   Warp (16 envs)  :                     ~0.5  GB
#   XLA scratch     :                     ~4.0  GB
#   -------------------------------------------
#   Total estimate  :                    ~16.3  GB  (7.7 GB headroom)
#
# Expected throughput: ~150-250 SPS (autotune ON = 5× faster kernels vs 3060)
# Expected runtime   : 500k steps / 200 SPS ≈ 40-60 min
set -e

export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Autotune: enable on 3090 (24 GB, CUDA driver 12.2).
# The f60f76d (nworld=16) module loading during stream capture is fixed in
# rad_brax_sac.py via jax.disable_jit() + enable_graph_capture_module_load=False
# during the warmup phase. Autotune=2 is safe because all modules are loaded
# before the first JIT call that starts CUDA graph capture.
export XLA_FLAGS="--xla_gpu_autotune_level=2"
export PYTHONPATH=~/rad-se/src

ENVNAME="${1:-CartpoleSwingup}"
SEED="${2:-23}"
WORKDIR=~/rad-se/runs/sac_3090_${ENVNAME}_s${SEED}

mkdir -p "$WORKDIR"

echo "=== SAC 3090 run: $ENVNAME seed=$SEED ===" | tee "$WORKDIR/train.log"
echo "VRAM: RTX 3090 24 GB | replay=50k | num_envs=16 | autotune=ON" | tee -a "$WORKDIR/train.log"

python3.12 -u ~/rad-se/src/rad_se/rad_brax_sac.py \
    --env "$ENVNAME" \
    --seed "$SEED" \
    --num-envs 16 \
    --max-replay-size 50000 \
    --min-replay-size 2000 \
    --batch-size 256 \
    --total-timesteps 500000 \
    --num-evals 20 \
    --num-eval-envs 16 \
    --episode-length 1000 \
    --action-repeat 8 \
    --learning-rate 3e-4 \
    --discounting 0.99 \
    --tau 0.005 \
    --reward-scaling 0.1 \
    --augment-pixels \
    --dmc-reward \
    --work-dir "$WORKDIR" \
    2>&1 | tee -a "$WORKDIR/train.log"
