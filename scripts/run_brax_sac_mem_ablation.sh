#!/usr/bin/env bash
# Fast memory-aware SAC ablation for local 12GB GPUs.
# This is not the exact RAD baseline: it keeps the current brax-loss/DQN-CNN path,
# but increases replay capacity by rendering at the final 84x84 resolution.
# Local RTX 3060 12GB notes:
#   replay=50k OOMed at prefill (~15.8GiB executable)
#   replay=25k OOMed during first eval/training overlap (~11.9GiB executable)
set -euo pipefail

export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export XLA_FLAGS="--xla_gpu_autotune_level=2"
export TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONPATH=src

ENVNAME="${1:-CartpoleSwingup}"
SEED="${2:-23}"
WORKDIR="runs/brax_sac_mem84_replay15k_${ENVNAME}_s${SEED}"

mkdir -p "$WORKDIR"

python3 -u src/rad_se/rad_brax_sac.py \
  --env "$ENVNAME" \
  --seed "$SEED" \
  --num-envs 8 \
  --max-replay-size 15000 \
  --min-replay-size 1000 \
  --batch-size 256 \
  --total-timesteps 2000000 \
  --num-evals 500 \
  --num-eval-envs 8 \
  --episode-length 1000 \
  --action-repeat 8 \
  --cam-res 84 \
  --learning-rate 3e-4 \
  --discounting 0.99 \
  --tau 0.005 \
  --reward-scaling 0.1 \
  --augment-pixels \
  --work-dir "$WORKDIR" \
  2>&1 | tee "$WORKDIR/train.log"
