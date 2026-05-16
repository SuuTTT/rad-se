#!/usr/bin/env bash
# Launch a full RAD JAX training run.
#
# Usage:
#   ./scripts/run_rad_jax.sh [env] [seed] [extra args…]
#
# Examples:
#   ./scripts/run_rad_jax.sh CartpoleSwingup  23
#   ./scripts/run_rad_jax.sh AcrobotSwingup   23 --action-repeat 4
#   ./scripts/run_rad_jax.sh CheetahRun       23 --action-repeat 4
#   ./scripts/run_rad_jax.sh CartpoleSwingup  23 --track   # enables W&B logging
#
# Requirements (install once):
#   pip install "jax[cuda12]" flax>=0.9.0 optax mujoco_playground wandb orbax-checkpoint
#   export JAX_DEFAULT_MATMUL_PRECISION=highest    # critical on Ampere/Ada/Hopper
#
set -euo pipefail

ENV="${1:-CartpoleSwingup}"
SEED="${2:-23}"
shift 2 || true  # remaining args passed through

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="runs/rad_jax__${ENV}__s${SEED}__${TIMESTAMP}"

echo "=== RAD JAX training ==="
echo "  env      : $ENV"
echo "  seed     : $SEED"
echo "  work_dir : $WORK_DIR"
echo "  extra    : $@"
echo ""

export JAX_DEFAULT_MATMUL_PRECISION=highest
# Uncomment to restrict to one GPU:
# export CUDA_VISIBLE_DEVICES=0

python src/rad_se/rad_jax.py \
    --env "$ENV" \
    --seed "$SEED" \
    --work-dir "$WORK_DIR" \
    "$@"

echo "=== Training complete: $WORK_DIR ==="
