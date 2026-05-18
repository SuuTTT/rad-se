#!/usr/bin/env bash
# Run brax PPO + augment_pixels (RAD) on CartpoleSwingup.
# Uses the official brax.training.agents.ppo.train (fully JIT-compiled scan).
set -e

export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Disable XLA autotuning: the profiler allocates O(unroll_length * pixels) contiguous
# blocks which OOMs on 12GB when unroll_length > 20.  Default kernel configs are ~5%
# slower but prevent the crash.
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export PYTHONPATH=src

ENVNAME="${1:-CartpoleSwingup}"
SEED="${2:-23}"
WORKDIR="runs/brax_ppo_${ENVNAME}_s${SEED}"

mkdir -p "$WORKDIR"

python3 -u src/rad_se/rad_brax_ppo.py \
    --env "$ENVNAME" \
    --seed "$SEED" \
    --num-envs 256 \
    --unroll-length 20 \
    --batch-size 32 \
    --num-minibatches 8 \
    --num-updates-per-batch 8 \
    --total-timesteps 500000 \
    --num-evals 20 \
    --num-eval-envs 32 \
    --action-repeat 8 \
    --episode-length 1000 \
    --discounting 0.99 \
    --learning-rate 3e-4 \
    --entropy-cost 0.01 \
    --clipping-epsilon 0.2 \
    --max-grad-norm 1.0 \
    --reward-scaling 0.1 \
    --dmc-reward \
    --work-dir "$WORKDIR" \
    2>&1 \
    | grep -v "Failed to track device allocation" \
    | tee "${WORKDIR}.log"
