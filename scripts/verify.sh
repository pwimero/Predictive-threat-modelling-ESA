#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
FULL=0

if [[ "${1:-}" == "--full" ]]; then
  FULL=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/verify.sh [--full]" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "No project .venv found. Run: bash scripts/setup.sh" >&2
  exit 1
fi

cd "$ROOT"
"$PYTHON_BIN" -m compileall -q mirage
"$PYTHON_BIN" -c "import mirage.cli"
"$PYTHON_BIN" -m unittest discover -s tests -v
if [[ $FULL -eq 1 ]]; then
  bash scripts/run_mirage.sh forecast
  bash scripts/run_mirage.sh audit
fi

echo "MIRAGE-M1 verification passed."
