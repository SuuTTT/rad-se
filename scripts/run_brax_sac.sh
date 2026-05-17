#!/usr/bin/env bash
# Run brax SAC + RAD pixel-translation (random crop) on CartpoleSwingup.
# Uses standalone training loop with brax SAC losses + UniformSamplingQueue.
# Brax's built-in sac.train raises NotImplementedError for dict/pixel obs.
set -e

export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Partial autotune (level=2): profiles top-5 cuDNN candidates per conv shape.
# Gives most of the speedup (~4-5x vs level=0) with only ~30s startup overhead.
# Safe on CUDA ≥ 12.3: nested stream capture means Warp physics and XLA autotune
# can coexist without CUDA error 900.
export XLA_FLAGS="--xla_gpu_autotune_level=2"
export PYTHONPATH=src

ENVNAME="${1:-CartpoleSwingup}"
SEED="${2:-23}"
WORKDIR="runs/brax_sac_${ENVNAME}_s${SEED}"

mkdir -p "$WORKDIR"

python3 -u src/rad_se/rad_brax_sac.py \
    --env "$ENVNAME" \
    --seed "$SEED" \
    --num-envs 8 \
    --max-replay-size 10000 \
    --min-replay-size 1000 \
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
    --work-dir "$WORKDIR" \
    2>&1 | tee "$WORKDIR/train.log"
