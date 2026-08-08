#!/usr/bin/env bash
# Step 1 environment setup for the 3dCT / MERLIN temporal-progression project.
# Creates a Python virtual environment and installs MERLIN + helpers.
#
# Usage:
#   bash setup_env.sh
#   source .venv/bin/activate
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PY="${PYTHON:-python3}"

echo "==> Using python: $($PY --version)"

# 1) Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment at .venv"
  "$PY" -m venv .venv
else
  echo "==> Reusing existing .venv"
fi

# 2) Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) Upgrade pip tooling
echo "==> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

# 4) Install requirements
echo "==> Installing requirements (this downloads MERLIN + torch, may take a while)"
python -m pip install -r requirements.txt

# 5) Sanity: report torch + CUDA availability
echo "==> Verifying install"
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
try:
    import merlin  # noqa: F401
    print("merlin: import OK")
except Exception as e:
    print(f"merlin import FAILED: {e}")
PY

echo ""
echo "==> Done. Activate with:  source .venv/bin/activate"
echo "==> Then run:            python scripts/01_run_demo.py"
