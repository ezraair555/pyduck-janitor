# Contributing to pyduck-janitor

Thanks for contributing.

## Development Setup

```bash
git clone https://github.com/yourusername/pyduck-janitor.git
cd pyduck-janitor
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Quick Checks Before PR

```bash
python3 -m pytest -q
```

Optional local checks:

```bash
black pyduck_janitor tests
flake8 pyduck_janitor tests
mypy pyduck_janitor
```

## Style and Architecture Rules

- Keep the lazy DuckDB-first execution model.
- Prefer SQL expressions over pandas materialization.
- Only use materialization for explicitly hybrid operations (for example `also`, `join_apply`, callable text/column transforms).
- Use shared SQL helpers for safety and consistency:
  - `_quote_id`
  - `_sql_literal`
  - `_register_relation`
  - `_ensure_columns_exist`
  - `_validate_sql_fragment`
- Add type annotations for all new functions and method signatures.
- Add NumPy-style docstrings for public methods and non-trivial internal helpers.

## Testing Guidelines

When adding or changing behavior:

- Add at least one happy-path test.
- Add failure-path tests for invalid inputs.
- Add edge-case tests where relevant (empty DataFrame, single-row DataFrame, all-null columns, duplicate column names).
- Keep tests deterministic (set seeds where randomness is involved).

## Pull Request Checklist

- Explain what changed and why.
- Include before/after behavior for bug fixes.
- Update docs (`README.md` and/or docstrings) for user-facing changes.
- Update `CHANGELOG.md`.
- Ensure `python3 -m pytest -q` passes.

## Reporting Issues

Include:

- Minimal reproducible example
- Python and DuckDB versions
- Expected behavior vs actual behavior
- Full stack trace (if available)

## Code of Conduct

This project follows [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
