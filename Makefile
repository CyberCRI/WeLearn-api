.PHONY: run-dev run-poetry generate-baml generate-baml-poetry lint test test-poetry

generate-baml:
	baml-cli generate --from ./src/app/baml_src

generate-baml-poetry:
	poetry run baml-cli generate --from ./src/app/baml_src

run-poetry: generate-baml-poetry
	poetry run uvicorn src.main:app --reload

run-dev: generate-baml
	uvicorn src.main:app --reload

lint:
	flake8 src
	isort src
	black src

test-poetry: generate-baml-poetry
	poetry run pytest -s -v --cov=src --cov-report=term-missing --cov-fail-under=82 --cov-report=html

test: generate-baml
	pytest -s -v --cov=src --cov-report=term-missing --cov-fail-under=82 --cov-report=html
