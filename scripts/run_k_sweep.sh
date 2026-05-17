#!/usr/bin/env bash
# K (updates_per_step) sweep — seed=23, CartpoleSwingup, 30k env-steps.
# Compares SPS + early-learning trend across K ∈ {8, 16, 32}.
# Uses the fused critic loop (commit 1fb8097) + platform allocator.
set -e
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONPATH=src
# platform allocator: direct cudaMalloc/Free, coexists with Warp pool.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

ENVNAME=CartpoleSwingup
SEED=23
N=64
STEPS=30000

for K in 8 16 32; do
    WORKDIR="runs/k_sweep_K${K}"
    echo "============================================"
    echo "  K=${K}  seed=${SEED}  workdir=${WORKDIR}"
    echo "============================================"
    python3 -u src/rad_se/rad_jax.py \
        --env     "$ENVNAME" \
        --seed    "$SEED" \
        --num-envs "$N" \
        --updates-per-step "$K" \
        --total-timesteps "$STEPS" \
        --work-dir "$WORKDIR" \
        2>&1 \
        | grep --line-buffered -v "Failed to track device allocation" \
        | tee "${WORKDIR}.log"
    echo "  K=${K} finished."
done

echo "K sweep done."
echo "Summary:"
for K in 8 16 32; do
    echo "--- K=${K} ---"
    grep -E "SPS|ER:" "runs/k_sweep_K${K}.log" | tail -20
done
