#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Flamatanyone — Install Script
#  Tested on macOS (Apple Silicon) with conda/miniconda
# ─────────────────────────────────────────────────────────────

set -e

CONDA_ENV="flamatanyone"
PYTHON_VERSION="3.11"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         Flamatanyone — Install       ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Check conda ──────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "❌  conda not found. Install Miniconda first:"
    echo "    https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ── 2. Create env if needed ─────────────────────────────────
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "✓  Conda env '$CONDA_ENV' already exists, skipping creation."
else
    echo "→  Creating conda env '$CONDA_ENV' (Python $PYTHON_VERSION)..."
    conda create -y -n "$CONDA_ENV" python="$PYTHON_VERSION"
fi

# ── 3. Activate env ─────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# ── 4. Install PyTorch (MPS for Apple Silicon, CUDA for Linux) ──
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

# ── 6. Install Python dependencies ─────────────────────────
echo "→  Installing Python dependencies..."
pip install -r hugging_face/requirements.txt

# ── 7. Download model weights ───────────────────────────────
echo "→  Downloading model weights..."
python - <<'EOF'
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
