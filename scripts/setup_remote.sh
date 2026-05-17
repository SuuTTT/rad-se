#!/usr/bin/env bash
# Install all dependencies on a fresh vast.ai node (CUDA 12.x, Ubuntu 22.04)
# Usage: bash setup_remote.sh
set -e

echo "=== [0/5] Installing Python 3.12 ==="
export DEBIAN_FRONTEND=noninteractive
apt-get install -y software-properties-common 2>&1 | tail -2
add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -2
apt-get update -q
apt-get install -y python3.12 python3.12-dev python3.12-venv 2>&1 | tail -3
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
python3.12 --version

echo "=== [1/5] Installing JAX 0.10.0 with CUDA 12 support ==="
python3.12 -m pip install --quiet \
    "jax==0.10.0" \
    "jaxlib==0.10.0" \
    "jax-cuda12-pjrt==0.10.0" \
    "jax-cuda12-plugin[with_cuda]==0.10.0"

echo "=== [2/5] Installing Brax + JAX ML stack ==="
python3.12 -m pip install --quiet \
    "brax==0.14.2" \
    "flax==0.12.7" \
    "optax==0.2.8" \
    "orbax-checkpoint>=0.6.0" \
    "chex>=0.1.87"

echo "=== [3/5] Installing MuJoCo + Playground ==="
python3.12 -m pip install --quiet \
    "mujoco==3.8.0" \
    "mujoco-mjx==3.8.1" \
    "playground==0.2.0"

echo "=== [4/5] Installing Warp (GPU physics renderer) ==="
python3.12 -m pip install --quiet "warp-lang==1.12.1"

echo "=== [5/5] Installing rad-se package ==="
cd ~/rad-se
python3.12 -m pip install --quiet -e . --no-deps

echo ""
echo "=== Verifying installs ==="
python3.12 -c "
import jax
print(f'jax: {jax.__version__}')
print(f'devices: {jax.devices()}')
import brax; print(f'brax: {brax.__version__}')
import mujoco_playground; print('mujoco_playground: OK')
import warp; print(f'warp: {warp.__version__}')
"
echo "=== Setup complete ==="
