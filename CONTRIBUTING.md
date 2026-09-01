# Contributing to pyduck-janitor

Thanks for contributing.

## Development Setup

```bash
git clone https://github.com/ezraair555/pyduck-janitor.git
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

## CI Contract (added 2026-09-01)

Three jobs run on every PR:

- **`lint`** — `ruff check` + `ruff format --check` on `pyduck_janitor/`, `scripts/`, `tests/`. Run locally with `ruff check . && ruff format --check .`. This job is **blocking** (failures prevent merge) — pre-existing style issues were cleared on 2026-09-01.
- **`test`** — pytest on Python 3.10, 3.11, 3.12 with DuckDB and pandas pinned to known-good versions. Stops on first failure with full traceback; on failure the log is uploaded as an artifact.
- **`review-agent`** — Runs `scripts/ci_review.py` and posts a comment on the PR with:
  1. **Module map** — every public symbol with its one-line docstring
  2. **API surface diff** — symbols added/removed vs. the PR base
  3. **Docstring coverage** — % of public symbols with full `Parameters` + `Returns` sections

The bot comment updates in place on every push; no need to dismiss or delete it.

### Running the review locally

```bash
python3 scripts/ci_review.py --dry-run --base-ref main
```

This prints the same markdown that CI would post. Use it to verify the diff and coverage before pushing.

### Docstring convention

Public symbols must have a NumPy-style docstring with `Parameters` and `Returns` sections. The `review-agent` job surfaces incomplete coverage in its PR comment; if a PR drops coverage, fix it before merging.

### Adding a new public function

1. Define it in the appropriate `cleaning_ops*.py` or `duck_janitor.py` module.
2. Re-export from `pyduck_janitor/__init__.py` if users should access it as `pyduck_janitor.foo`.
3. Add the name to `__all__` in `pyduck_janitor/__init__.py` so the `review-agent` job picks it up.
4. Write a full `Parameters` / `Returns` docstring.

The `review-agent` job will surface missing exports and missing docstrings automatically.
