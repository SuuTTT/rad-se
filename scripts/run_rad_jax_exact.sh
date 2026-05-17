#!/usr/bin/env bash
# RAD-exact JAX/Flax SAC on MuJoCo Playground pixels.
# Mirrors the original RAD DMC pixel setup as closely as the Playground backend allows.
set -euo pipefail

export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYTHONPATH=src

ENVNAME="${1:-CartpoleSwingup}"
SEED="${2:-23}"
WORKDIR="runs/rad_jax_exact_${ENVNAME}_s${SEED}"

mkdir -p "$WORKDIR"

python3 -u src/rad_se/rad_jax.py \
  --env "$ENVNAME" \
  --seed "$SEED" \
  --action-repeat 8 \
  --cam-res 100 \
  --image-size 84 \
  --frame-stack 3 \
  --num-envs 8 \
  --updates-per-step 8 \
  --total-timesteps 1000000 \
  --replay-capacity 100000 \
  --init-steps 1000 \
  --batch-size 32 \
  --eval-freq 10000 \
  --num-eval-episodes 10 \
  --discount 0.99 \
  --init-temperature 0.1 \
  --reward-scale 1.0 \
  --actor-lr 1e-3 \
  --critic-lr 1e-3 \
  --alpha-lr 1e-4 \
  --critic-tau 0.01 \
  --encoder-tau 0.05 \
  --actor-update-freq 2 \
  --critic-target-update-freq 2 \
  --encoder-feature-dim 50 \
  --num-layers 4 \
  --num-filters 32 \
  --hidden-dim 1024 \
  --log-std-min -10 \
  --log-std-max 2 \
  --log-interval 5000 \
  --work-dir "$WORKDIR" \
  2>&1 | tee "$WORKDIR/train.log"
