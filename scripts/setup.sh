#!/usr/bin/env bash
# Standard-library setup: Python's built-in venv plus pip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Could not find $PYTHON_BIN. Install Python 3.11 and try again." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  "numpy>=1.26,<3" "pandas>=2.2,<3" "PyYAML>=6,<7" "rich>=13.7,<15" \
  "scikit-learn>=1.4,<2" "scipy>=1.11,<2" "torch>=2.2,<3" "typer>=0.12,<1"

echo "MIRAGE-M1 environment ready."
echo "Run: bash scripts/run_mirage.sh --help"
