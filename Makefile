.PHONY: run-dev

run-poetry:
	poetry run uvicorn src.main:app --reload

run-dev:
	uvicorn src.main:app --reload

lint:
	flake8 src
	isort src
	black src

test-poetry:
	poetry run pytest -s -v --cov=src --cov-report=term-missing --cov-fail-under=85 --cov-report=html

test:
	pytest -s -v --cov=src --cov-report=term-missing --cov-fail-under=83 --cov-report=html
