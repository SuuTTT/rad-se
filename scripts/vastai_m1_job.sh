#!/usr/bin/env bash
# M1: CartpoleSwingup × seeds {23, 42, 7}  on a Vast.ai GPU instance.
#
# Intended to be passed as --job-cmd to vastai_scheduler.py launch.
# The scheduler wraps this in its own onstart script that handles
# instance destroy on exit.
#
# Environment must provide CUDA 12.x (JAX cuda12 wheel).
# Tested with: RTX 3060 / 4090, CUDA 12.x, Ubuntu 22.04, Python 3.11
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export MUJOCO_GL=egl
export JAX_DEFAULT_MATMUL_PRECISION=highest
export CUDA_VISIBLE_DEVICES=0

echo "=== M1 RAD-JAX CartpoleSwingup × 3 seeds ==="
echo "HOST=$(hostname)  DATE=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── 1. Install deps ─────────────────────────────────────────────────────────
pip install --quiet --upgrade pip
pip install --quiet "jax[cuda12]" flax optax orbax-checkpoint wandb
pip install --quiet \
    "git+https://github.com/google-deepmind/mujoco_playground.git" \
    --ignore-installed blinker
# mujoco-mjx 3.8.x installs warp-lang latest; pin back to 1.12.1
pip install --quiet "warp-lang==1.12.1"

# ── 2. Clone repo ────────────────────────────────────────────────────────────
cd /workspace
rm -rf rad-se
git clone --depth 1 https://github.com/SuuTTT/rad-se.git rad-se
cd rad-se

# ── 3. Sanity: unit tests (CPU, ~30 s) ───────────────────────────────────────
echo "--- unit tests ---"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORM_NAME=cpu PYTHONPATH=src \
    python3 -m pytest src/rad_se/test_rad_jax.py -q --tb=short
echo "--- unit tests passed ---"

# ── 4. GPU smoke (CartpoleSwingup, 500 steps) ─────────────────────────────────
echo "--- GPU smoke ---"
PYTHONPATH=src python3 src/rad_se/rad_jax.py \
    --env CartpoleSwingup --seed 23 --smoke \
    --work-dir runs/smoke_gpu_s23
echo "--- GPU smoke passed ---"

# ── 5. Full runs ─────────────────────────────────────────────────────────────
for SEED in 23 42 7; do
    echo "=== CartpoleSwingup seed=$SEED ==="
    PYTHONPATH=src python3 src/rad_se/rad_jax.py \
        --env CartpoleSwingup \
        --seed "$SEED" \
        --work-dir "runs/rad_jax__CartpoleSwingup__s${SEED}"
    echo "=== DONE seed=$SEED ==="
done

echo "=== M1 ALL RUNS COMPLETE ==="
# Print final eval rewards from each run's logs
for SEED in 23 42 7; do
    LOGFILE="runs/rad_jax__CartpoleSwingup__s${SEED}/log.txt"
    if [[ -f "$LOGFILE" ]]; then
        echo "--- seed=$SEED last 5 eval lines ---"
        grep "eval" "$LOGFILE" | tail -5 || true
    fi
done
