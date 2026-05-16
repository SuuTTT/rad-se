#!/usr/bin/env bash
# M1 local GPU run — 3 seeds, CartpoleSwingup, 200k env-steps, N=64
set -e
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONPATH=src
# Disable JAX's aggressive VRAM pre-allocation so Warp mempool has headroom
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Hard-cap JAX allocator at 50% of VRAM (6 GB on RTX 3060 12 GB).
# Prevents XLA's BFC cache from crowding out Warp's physics/render pool.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

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
        2>&1 | tee "${WORKDIR}.log"
    echo "  Seed ${SEED} finished."
done

echo "All M1 seeds done."
