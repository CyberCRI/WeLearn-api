#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_BIN="$ROOT_DIR/.venv/bin"
PYTHON_BIN="$VENV_BIN/python"
FLAKE8_BIN="$VENV_BIN/flake8"
ISORT_BIN="$VENV_BIN/isort"
BLACK_BIN="$VENV_BIN/black"
PYTEST_BIN="$VENV_BIN/pytest"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv Python at $PYTHON_BIN" >&2
  exit 1
fi

OUT_DIR="${1:-$ROOT_DIR/.artifacts/python-3.14}"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="$OUT_DIR/phase0-baseline-$STAMP.md"
TEST_LOG="$OUT_DIR/phase0-pytest-$STAMP.log"

BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse --short HEAD)"
PY_VERSION="$($PYTHON_BIN --version 2>&1)"

{
  echo "# Phase 0 Baseline"
  echo
  echo "- Timestamp (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Branch: $BRANCH"
  echo "- Commit: $COMMIT"
  echo "- Python: $PY_VERSION"
  echo
  echo "## Lint"
} > "$OUT_FILE"

if /usr/bin/time -f 'flake8_elapsed_sec=%e' "$FLAKE8_BIN" src >> "$OUT_FILE" 2>&1; then
  echo "- flake8: pass" >> "$OUT_FILE"
else
  echo "- flake8: fail" >> "$OUT_FILE"
fi

if /usr/bin/time -f 'isort_check_elapsed_sec=%e' "$ISORT_BIN" --check-only src >> "$OUT_FILE" 2>&1; then
  echo "- isort --check-only: pass" >> "$OUT_FILE"
else
  echo "- isort --check-only: fail" >> "$OUT_FILE"
fi

if /usr/bin/time -f 'black_check_elapsed_sec=%e' "$BLACK_BIN" --check src >> "$OUT_FILE" 2>&1; then
  echo "- black --check: pass" >> "$OUT_FILE"
else
  echo "- black --check: fail" >> "$OUT_FILE"
fi

{
  echo
  echo "## Tests"
} >> "$OUT_FILE"

set +e
/usr/bin/time -f 'pytest_elapsed_sec=%e' "$PYTEST_BIN" -q --cov=src --cov-report=term-missing --cov-fail-under=82 --cov-report=html > "$TEST_LOG" 2>&1
TEST_EXIT="$?"
set -e

echo "- pytest exit code: $TEST_EXIT" >> "$OUT_FILE"
echo "- pytest log: $TEST_LOG" >> "$OUT_FILE"
echo '```text' >> "$OUT_FILE"
tail -n 80 "$TEST_LOG" >> "$OUT_FILE"
echo '```' >> "$OUT_FILE"

{
  echo
  echo "## Startup Proxy"
} >> "$OUT_FILE"

"$PYTHON_BIN" - <<'PY' >> "$OUT_FILE"
import statistics
import subprocess
import time

runs = []
for i in range(3):
    start = time.perf_counter()
    subprocess.run(
        ["./.venv/bin/python", "-c", "import src.main; _=src.main.app"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=45,
    )
    elapsed = time.perf_counter() - start
    runs.append(elapsed)
    print(f"- startup_run_{i + 1}_sec={elapsed:.4f}")

print(f"- startup_avg_sec={statistics.mean(runs):.4f}")
PY

{
  echo
  echo "## Embedding Proxy"
} >> "$OUT_FILE"

"$PYTHON_BIN" - <<'PY' >> "$OUT_FILE"
from pathlib import Path

MODEL_NAME = "granite-embedding-107m-multilingual"
model_path = Path("../models/embedding") / MODEL_NAME
print(f"- embedding_model_path={model_path.resolve()}")
print(f"- embedding_model_exists={model_path.exists()}")
if not model_path.exists():
    print("- embedding_latency_sec=SKIPPED_MODEL_PATH_MISSING")
PY

echo "Baseline report written to: $OUT_FILE"
