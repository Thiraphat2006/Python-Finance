---
name: python-quality-gate
description: Enforce Python code quality workflow using uv, Ruff, mypy, pytest and coverage. Use whenever creating, modifying, refactoring, debugging Python code or implementing new features.
---

# Python Quality Gate

Use this workflow for every Python change, including new features, bug fixes,
refactors, tests, and configuration changes that affect Python code.

## Project conventions

- Inspect the repository before editing and read the existing architecture,
  tests, `pyproject.toml`, `uv.lock`, and `makefile`.
- Use `uv` for dependency management and `uv run` for Python tools.
- Treat `pyproject.toml` as the primary tool configuration.
- Keep `uv.lock` synchronized with dependency changes.
- Do not use `pip` to manage project dependencies.
- Reuse the existing environment; do not create a new virtual environment
  unless the project setup requires it.
- The source directory is `src/` and tests are under `Test/`.
- Do not change business logic or weaken existing tests merely to satisfy a
  gate.

## Required workflow

1. Explore the project structure and relevant implementation/tests.
2. Check `pyproject.toml`, dependency groups, `uv.lock`, and `makefile`.
3. Implement the requested change and add or update focused tests for normal,
   edge, and error cases as appropriate.
4. Run all gates in this order:

   ```powershell
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src/
   uv run pytest
   uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80
   ```

5. Read each failure, fix its root cause, and rerun the failed gate.
6. Rerun the complete quality gate after fixes.
7. Do not report completion while any required gate is failing.

## Gate rules

### Ruff

- `ruff check .` must pass.
- `ruff format --check .` must pass.
- Fix reported code and formatting issues rather than disabling rules.
- Only change Ruff configuration when there is a documented, repository-wide
  reason.

### mypy

- Run `mypy src/`.
- Add parameter and return annotations to new or changed functions.
- Prefer precise types and avoid unnecessary `Any`.
- Do not use `type: ignore` or loosen configuration to hide a real error.

### pytest

- Run the complete test suite with `uv run pytest`.
- Add assertions covering changed behavior, including relevant normal, edge,
  and error cases.
- Mock external services when appropriate.
- Never delete or weaken existing tests to make the suite pass.

### Coverage

- Use branch coverage and enforce a minimum total coverage of 80%:

  ```powershell
  uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80
  ```

- Add meaningful tests for uncovered logic; do not exclude code that should be
  tested or add assertion-free tests just to increase the percentage.
- If the source layout changes, update the command and coverage configuration
  to match the real source directory.

## Dependencies and configuration

Before adding a tool, inspect existing dependencies. If a required tool is
missing, add it with uv without duplicating or removing active dependencies:

```powershell
uv add --dev ruff mypy pytest pytest-cov
uv lock
```

Only add packages that are actually missing. Verify that `uv.lock` remains
consistent after dependency changes.

## Makefile

Keep the existing targets and provide these targets when applicable:

```make
.PHONY: install run test lint format typecheck coverage quality clean

install:
	uv sync

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

quality:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/
	uv run pytest
	uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80
```

Adapt paths only when the repository layout requires it. Preserve unrelated
existing targets.

## Failure handling and reporting

- Read the complete tool error before changing code.
- Distinguish failures that existed before the change from regressions caused
  by the change, and report both accurately.
- Never hide failures with broad ignores, disabled Ruff rules, reduced
  coverage thresholds, removed tests, or success-shaped fallbacks.
- If a gate cannot be fixed, state the exact failing command and reason.

Every completion report must use this table and must reflect commands actually
run:

| Quality Gate | Status |
|---|---|
| Ruff Check | PASS / FAIL |
| Ruff Format | PASS / FAIL |
| mypy | PASS / FAIL |
| pytest | PASS / FAIL |
| Coverage | PASS / FAIL |

Also report the passing and failing test counts, coverage percentage, number of
changed files, a concise change summary, and remaining problems. Do not claim
PASS for a gate that was not run or that returned a failure.
