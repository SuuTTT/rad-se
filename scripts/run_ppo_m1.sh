#!/usr/bin/env bash
# PPO+RAD on CartpoleSwingup pixel — target ER comparable/surpassing RAD-SAC
# (PyTorch anchor: ~861 at 190k env-steps).  We run with a much larger
# env-step budget because PPO is sample-inefficient but wall-clock fast.
set -e
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONPATH=src
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

ENVNAME=CartpoleSwingup
SEED=23
WORKDIR="runs/ppo_${ENVNAME}_s${SEED}"

python3 -u src/rad_se/rad_ppo_jax.py \
    --env "$ENVNAME" \
    --seed "$SEED" \
    --num-envs 128 \
    --unroll-length 16 \
    --num-minibatches 8 \
    --update-epochs 4 \
    --total-timesteps 1000000 \
    --eval-freq 50000 \
    --work-dir "$WORKDIR" \
    2>&1 \
    | grep --line-buffered -v "Failed to track device allocation" \
    | tee "${WORKDIR}.log"
