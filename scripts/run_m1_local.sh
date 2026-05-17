#!/usr/bin/env bash
# M1 local GPU run — 3 seeds, CartpoleSwingup, 200k env-steps, N=64
set -e
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONPATH=src
# Use platform (cudaMalloc) allocator instead of JAX's BFC cache.
# BFC keeps growing within its cap and eventually denies Warp's transient
# 256-byte allocs (Warp + JAX share the same 12 GB VRAM pool on RTX 3060).
# platform allocator does direct cudaMalloc/Free per JAX op — slightly slower
# per-op but plays cleanly with Warp's pool.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

ENVNAME=CartpoleSwingup
N=64

for SEED in 23 42 7; do
    WORKDIR="runs/m1_${ENVNAME}_s${SEED}"
    echo "============================================"
    echo "  Starting seed=${SEED}  workdir=${WORKDIR}"
    echo "============================================"
    python3 -u src/rad_se/rad_jax.py \
        --env     "$ENVNAME" \
        --seed    "$SEED" \
        --num-envs "$N" \
        --total-timesteps 200000 \
        --work-dir "$WORKDIR" \
        2>&1 \
        | grep --line-buffered -v "Failed to track device allocation" \
        | tee "${WORKDIR}.log"
    echo "  Seed ${SEED} finished."
done

echo "All M1 seeds done."
