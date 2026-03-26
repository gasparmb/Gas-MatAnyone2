#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Flamatanyone — Launch Script
# ─────────────────────────────────────────────────────────────

VENV_DIR="$(dirname "$0")/.venv"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌  Virtual env not found. Run: bash install.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"

cd "$(dirname "$0")/hugging_face"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║    Flamatanyone — Starting server    ║"
echo "║    http://localhost:$PORT            ║"
echo "╚══════════════════════════════════════╝"
echo ""

uvicorn custom_server:app --host "$HOST" --port "$PORT" --reload
