#!/usr/bin/env bash
# Fast frame-stack + RAD entropy ablation for local 12GB GPUs.
# Tests whether brax's default target_entropy=-0.5/action_dim and alpha=1.0
# caused early alpha collapse compared with original RAD's alpha=0.1, H=-action_dim.
set -euo pipefail

export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export XLA_FLAGS="--xla_gpu_autotune_level=2"
export TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONPATH=src

ENVNAME="${1:-CartpoleSwingup}"
SEED="${2:-23}"
WORKDIR="runs/brax_sac_mem84_framestack3_entropy_replay5k_${ENVNAME}_s${SEED}"

mkdir -p "$WORKDIR"

python3 -u src/rad_se/rad_brax_sac.py \
  --env "$ENVNAME" \
  --seed "$SEED" \
  --num-envs 8 \
  --max-replay-size 5000 \
  --min-replay-size 1000 \
  --batch-size 256 \
  --total-timesteps 2000000 \
  --num-evals 500 \
  --num-eval-envs 8 \
  --episode-length 1000 \
  --action-repeat 8 \
  --cam-res 84 \
  --frame-stack 3 \
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
