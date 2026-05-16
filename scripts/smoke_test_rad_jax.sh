#!/usr/bin/env bash
# Smoke-test rad_jax.py without a full env run.
# Runs the unit tests in test_rad_jax.py (CPU, ~10 seconds).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== rad_jax smoke test (CPU unit tests) ==="
cd "$REPO_ROOT"

export JAX_PLATFORM_NAME=cpu
export JAX_DEFAULT_MATMUL_PRECISION=highest

python src/rad_se/test_rad_jax.py
echo "=== Smoke test PASSED ==="
