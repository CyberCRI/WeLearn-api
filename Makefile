.PHONY: run-dev

run-poetry:
	poetry run baml-cli generate --from ./src/app/baml_src
	poetry run uvicorn src.main:app --reload

run-dev:
	baml-cli generate --from ./src/app/baml_src
	uvicorn src.main:app --reload

lint:
	flake8 src
	isort src
	black src

test-poetry:
	poetry run pytest -s -v --cov=src --cov-report=term-missing --cov-fail-under=82 --cov-report=html

test:
	pytest -s -v --cov=src --cov-report=term-missing --cov-fail-under=82 --cov-report=html
