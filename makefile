.PHONY: install run test lint format typecheck coverage coverage-html quality clean diff-head

install:
	uv sync

run:
	uv run python -m project.Execution.run_daily --symbol XAUUSD

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/

coverage:
	uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80

coverage-html:
	uv run coverage html

clean:
	uv run coverage erase

quality:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/
	uv run pytest
	uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80

diff-head:
	powershell -NoProfile -Command "git --no-pager diff HEAD | Set-Content diff.txt -Encoding utf8"