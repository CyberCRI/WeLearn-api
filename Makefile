.PHONY: run-dev run-poetry generate-baml generate-baml-poetry lint lint-check-poetry test test-poetry openapi openapi-poetry ci-poetry install-hooks

generate-baml:
	baml-cli generate --from ./src/app/baml_src

generate-baml-poetry:
	poetry run baml-cli generate --from ./src/app/baml_src

# regenerate openapi.json, the contract WeLearn-client is generated from
openapi-poetry: generate-baml-poetry
	poetry run python -m src.export_openapi

openapi: generate-baml
	python -m src.export_openapi

run-poetry: generate-baml-poetry
	poetry run uvicorn src.main:app --reload

run-dev: generate-baml
	uvicorn src.main:app --reload

lint:
	flake8 src
	isort src
	black src
	lint-imports

# same checks as `lint`, without rewriting files
lint-check-poetry:
	poetry run flake8 src
	poetry run isort --check-only --diff src
	poetry run black --check --diff src
	poetry run lint-imports

test-poetry: generate-baml-poetry
	poetry run pytest -s -v --cov=src --cov-report=term-missing --cov-fail-under=82 --cov-report=html

test: generate-baml
	pytest -s -v --cov=src --cov-report=term-missing --cov-fail-under=82 --cov-report=html

# what the lint-and-test CI job runs; used by the pre-push hook
ci-poetry: lint-check-poetry test-poetry

install-hooks:
	git config core.hooksPath .githooks
