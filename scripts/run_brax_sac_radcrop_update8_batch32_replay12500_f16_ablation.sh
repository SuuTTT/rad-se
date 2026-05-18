#!/usr/bin/env bash
# Reduced-precision replay capacity push: closest local RAD mix with 12.5k
# float16 flat replay. Tests whether the replay10k improvement continues.
set -euo pipefail

export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export XLA_FLAGS="--xla_gpu_autotune_level=2"
export TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONPATH=src

ENVNAME="${1:-CartpoleSwingup}"
SEED="${2:-23}"
WORKDIR="runs/brax_sac_mem100_crop84_radencoder_framestack3_entropy_rewardonce_replay12500_f16_batch32_update8_scan125_${ENVNAME}_s${SEED}"

mkdir -p "$WORKDIR"

python3 -u src/rad_se/rad_brax_sac.py \
  --env "$ENVNAME" \
  --seed "$SEED" \
  --num-envs 8 \
  --max-replay-size 12500 \
  --min-replay-size 1000 \
  --batch-size 32 \
  --grad-updates-per-step 8 \
  --replay-pixel-dtype float16 \
  --total-timesteps 2000000 \
  --num-evals 2000 \
  --num-eval-envs 8 \
  --episode-length 1000 \
  --action-repeat 8 \
  --cam-res 100 \
  --crop-size 84 \
  --frame-stack 3 \
  --encoder-arch rad \
  --rad-feature-dim 50 \
  --learning-rate 3e-4 \
  --alpha-learning-rate 1e-4 \
  --init-temperature 0.1 \
  --target-entropy -1.0 \
  --discounting 0.99 \
  --tau 0.005 \
  --reward-scaling 0.1 \
  --augment-pixels \
  --work-dir "$WORKDIR" \
  2>&1 | tee "$WORKDIR/train.log"
