# Handoff

## Completed

- Added `.github/workflows/quality.yml`.
- Configured Windows quality checks to mirror the `Makefile` commands.
- Added Ubuntu tracked-file and Python filename hygiene checks.
- Restricted workflow permissions to `contents: read`.
- Removed the duplicate standalone test step; coverage now runs the test suite.
- Allowed dunder Python filenames in the hygiene check.

## Validation

- `make quality` passed locally.
- `uv lock --check` passed.
- Workflow YAML parsed successfully with `uv run --with pyyaml`.

## Open question

- The remote GitHub Actions result can only be confirmed after the workflow is pushed.
