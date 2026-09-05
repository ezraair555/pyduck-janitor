# pyduck-janitor Agent Guide

## Purpose

`pyduck-janitor` is a DuckDB-backed, lazy, method-chaining data preparation
library. It combines pyjanitor-style cleaning verbs with DuckDB SQL, temporal
joins, database metrics, graph adapters, and snapshot comparison.

## Start here

1. Read `README.md` for installation and workflows.
2. Read `docs/agent-guide.md` for the API map and capability boundaries.
3. Use `docs/api/functions.md` for exact signatures and runnable examples.
4. Inspect `pyduck_janitor/duck_janitor.py` for relation verbs and
   `pyduck_janitor/extensions.py` for optional DuckDB extensions.

## Design rules

- Preserve lazy DuckDB execution; do not collect to pandas unless the public
  method explicitly returns a DataFrame.
- Return a `DuckJanitor` relation from transformation and analysis methods.
- Keep optional extensions lazy and out of package-import paths.
- Quote identifiers with `DuckJanitor._quote()` and validate user-facing
  identifiers before interpolating them into SQL.
- Keep source-database drivers out of core dependencies.
- Add a focused regression test and a runnable documentation example for each
  public method.
- Do not make HR-specific assumptions in generic relation verbs.

## Validation

```bash
pytest -q
python3 scripts/generate_api_docs.py --check
ruff check pyduck_janitor scripts tests
ruff format --check pyduck_janitor scripts tests
git diff --check
```

Optional DuckDB community extensions require a matching DuckDB minor version;
test them separately from the core matrix.
