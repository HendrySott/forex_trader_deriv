#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_DIR/trader"
PYTHON_BIN="${PYTHON_BIN:-python3}"

create_env() {
    echo "==> Creating virtual environment: trader"
    if [ ! -d "$ENV_DIR" ]; then
        "$PYTHON_BIN" -m venv "$ENV_DIR"
    elif [ ! -f "$ENV_DIR/bin/activate" ] || [ ! -x "$ENV_DIR/bin/python" ]; then
        echo "==> Existing trader folder is incomplete. Rebuilding it..."
        "$PYTHON_BIN" -m venv --clear "$ENV_DIR"
    else
        echo "==> Environment already exists: trader"
    fi

    if [ ! -f "$ENV_DIR/bin/activate" ]; then
        echo "❌ Failed to create trader/bin/activate."
        echo "Install the Python venv package for your system, then run this script again."
        echo "On Ubuntu/Debian, try: sudo apt install python3-venv"
        exit 1
    fi
}

activate_env() {
    echo "==> Activating environment"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
}

install_libs() {
    echo "==> Upgrading pip tools"
    python -m pip install --upgrade pip setuptools wheel

    echo "==> Installing project libraries"
    python -m pip install -r "$PROJECT_DIR/requirements.txt"
}

create_env
activate_env
install_libs

echo
echo "✅ Trader environment is ready."
echo
echo "To start the bot:"
echo "  source trader/bin/activate"
echo "  python main.py"
