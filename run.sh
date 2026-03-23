#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Flamatanyone — Launch Script
# ─────────────────────────────────────────────────────────────

CONDA_ENV="flamatanyone"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" 2>/dev/null || true

cd "$(dirname "$0")/hugging_face"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║    Flamatanyone — Starting server    ║"
echo "║    http://localhost:$PORT            ║"
echo "╚══════════════════════════════════════╝"
echo ""

uvicorn custom_server:app --host "$HOST" --port "$PORT" --reload
