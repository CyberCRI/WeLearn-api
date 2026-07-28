# Phase 0 Baseline (Python 3.12)

- Timestamp (UTC): 2026-07-28T07:31:51Z
- Branch: upgrade-python-version
- Commit: 13b9fa1
- Interpreter: Python 3.12.3 (`.venv`)

## Scope Implemented

- Quality gates baseline: lint and tests.
- Startup proxy baseline: fresh process import of `src.main`.
- Embedding proxy baseline: local embedding model artifact availability check.

## Baseline Results

### Lint (equivalent to `make lint`)

- `flake8 src`: pass, `flake8_elapsed_sec=0.74`
- `isort --check-only src`: pass, `isort_check_elapsed_sec=0.50`
- `black --check src`: pass, `black_check_elapsed_sec=0.47`

### Tests (equivalent to `make test`)

- `pytest -q --cov=src --cov-report=term-missing --cov-fail-under=82 --cov-report=html`
- Result: `167 passed, 8 warnings`
- Coverage: `83.26%` (threshold `82%`)
- Runtime: `pytest_elapsed_sec=61.56`

### Startup Proxy Latency

- Method: run a fresh Python subprocess that imports `src.main` and reads `app`.
- Run 1: `startup_run_1_sec=15.0763`
- Run 2: `startup_run_2_sec=13.2675`
- Run 3: `startup_run_3_sec=15.1226`
- Average: `startup_avg_sec=14.4888`

### Embedding Proxy Latency

- Model expected at `../models/embedding/granite-embedding-107m-multilingual`
- Availability: `embedding_model_exists=True`
- Metric status: `latency not recorded by script when model exists`.

## Observations / Risks Logged During Phase 0

- `make lint` and `make test` fail in this local runner due to missing global `flake8` and `pytest` in PATH.
- The same checks pass via `.venv/bin/*`, which is now captured in the automation script.

## Artifacts

- Baseline automation script: `migration/python-3.14/phase0_collect_baseline.sh`
- Latest baseline report: `.artifacts/python-3.14/phase0-baseline-20260728T073151Z.md`
- Latest pytest log: `.artifacts/python-3.14/phase0-pytest-20260728T073151Z.log`
