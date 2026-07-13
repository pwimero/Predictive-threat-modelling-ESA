#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "No project .venv found. Run: bash scripts/setup.sh" >&2
  exit 1
fi
cd "$ROOT"
exec "$PYTHON_BIN" -m mirage "$@"
