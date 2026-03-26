#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Flamatanyone — Install Script
#  Requires: Python 3.10+
# ─────────────────────────────────────────────────────────────

set -e

VENV_DIR="$(dirname "$0")/.venv"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         Flamatanyone — Install       ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Check Python ─────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "❌  python3 not found. Install Python 3.10+ from https://www.python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓  Python $PYTHON_VERSION found"

# ── 2. Create venv if needed ────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "✓  Virtual env already exists, skipping creation."
else
    echo "→  Creating virtual env..."
    python3 -m venv "$VENV_DIR"
fi

# ── 3. Activate venv ────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── 4. Install PyTorch ──────────────────────────────────────
if [[ "$(uname)" == "Darwin" ]]; then
    echo "→  macOS detected — installing PyTorch (MPS)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    echo "→  Linux detected — installing PyTorch (CUDA 12.1)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# ── 5. Install matanyone2 package ───────────────────────────
echo "→  Installing matanyone2 package..."
pip install -e . --no-deps

# ── 6. Install Python dependencies ──────────────────────────
echo "→  Installing Python dependencies..."
pip install -r hugging_face/requirements.txt

# ── 7. Download model weights ───────────────────────────────
echo "→  Downloading model weights..."
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import os

models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
os.makedirs(models_dir, exist_ok=True)

files = [
    ("hkchengrex/MatAnyone", "matanyone.pth"),
]
for repo, filename in files:
    dest = os.path.join(models_dir, filename)
    if os.path.exists(dest):
        print(f"  ✓ {filename} already downloaded")
    else:
        print(f"  ↓ Downloading {filename}...")
        hf_hub_download(repo_id=repo, filename=filename, local_dir=models_dir)
        print(f"  ✓ {filename} done")
EOF

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   ✅  Installation complete!         ║"
echo "║   Run:  bash run.sh                  ║"
echo "╚══════════════════════════════════════╝"
echo ""
