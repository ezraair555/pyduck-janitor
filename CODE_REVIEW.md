# Code Review (pyduck-janitor)

## Scope
- `pyduck_janitor/duck_janitor.py`
- `pyduck_janitor/cleaning_ops.py`
- `pyduck_janitor/cleaning_ops_extended.py`
- `pyduck_janitor/cleaning_ops_final.py`
- `pyduck_janitor/__init__.py`
- `tests/test_duck_janitor.py`
- `tests/test_cleaning_ops.py`
- `tests/property_cleaning_ops.py`

## Findings (ordered by severity)

### High
1. **Raw SQL fragment injection surface in user-provided predicates/expressions**
- Locations:
  - `cleaning_ops.py`: `filter_on`, `filter_column`, `select_rows(criteria)`, `transform_column(func=str)`
  - `cleaning_ops_extended.py`: `case_when` conditions, `change_type(dtype)`
  - `cleaning_ops_final.py`: `process_text(func=str)`
- Risk: multi-statement fragments (`;`) and destructive statements (`DROP`, `DELETE`, etc.) could be embedded in string inputs.
- Fix implemented: added centralized `_validate_sql_fragment` guard and enforced it for SQL-fragment APIs.

2. **Missing column validation across multiple operations**
- Locations: multiple functions in all three cleaning modules.
- Risk: opaque DuckDB binder/parser errors, inconsistent user experience, and error handling gaps.
- Fix implemented: added centralized `_ensure_columns_exist` and applied to column-referencing functions.

### Medium
3. **Input validation gaps for edge cases**
- Locations:
  - `head(n)` accepted negatives.
  - `groupby_topk(k)` accepted non-positive values.
  - `jitter(scale)` accepted negatives.
  - `bin_numeric(bins)` allowed invalid bin specs.
  - `find_replace(value_pairs)` allowed empty mapping.
  - `alias(alias='')` allowed empty alias names.
- Fix implemented: explicit validation with actionable errors.

4. **Hybrid function robustness gap on cross-connection joins**
- Location: `cleaning_ops_final.py::join_apply`.
- Risk: `other` relation from a different DuckDB connection could fail registration.
- Fix implemented: fallback materialization for right-side relation when direct register fails.

5. **Callable filter mask shape not validated**
- Location: `cleaning_ops.py::filter_column`.
- Risk: invalid mask shape can crash with unclear pandas errors.
- Fix implemented: explicit row-count validation for callable masks.

### Low
6. **Dead code / unused artifacts**
- `cleaning_ops_extended.py`: removed unused `escaped_col` variable and unused `re` import.
- `cleaning_ops_final.py`: removed unused `re` import.
- `__init__.py`: removed duplicate `truncate_datetime` import line.

7. **Type hint completeness and consistency**
- `DuckJanitor.__init__` now has `-> None`.
- `from_csv` and `mutate` now type `**kwargs: Any`.

8. **Doc quality inconsistency**
- Many function docstrings are short one-liners without constraints/error behavior.
- Improvement implemented in this pass is focused on code-level safeguards + test coverage; additional deep API-doc expansion remains a future incremental task.

## Performance Notes
- The package mostly preserves lazy SQL execution.
- Known N+1 query patterns remain in:
  - `remove_empty` (per-column COUNT)
  - `drop_constant_columns` (per-column DISTINCT count)
  - `drop_duplicate_columns` (pairwise comparisons)
- These are acceptable for current correctness-focused pass but are candidates for future batched-query optimization.

## Security Notes
- SQL fragment APIs now reject:
  - multi-statements (`;`)
  - SQL comments (`--`, `/* */`)
  - destructive/DDL/DML keywords (`drop`, `delete`, `update`, `insert`, etc.)
- This hardening applies while preserving legitimate expression/predicate usage.

## Test Coverage Expansion
- Added a new test module with broad validation/error/edge coverage, including:
  - SQL-fragment rejection paths
  - missing-column paths
  - empty/single-row/all-null/duplicate-column edge cases
  - hybrid method behavior (`join_apply`, `process_text`, `mutate`)

## Residual Risks
- `DuckJanitor.sql()` intentionally accepts raw SQL (power-user surface).
- Some heavy functions still use materialization by design (hybrid operations).
- Additional function-by-function docstring expansion can further improve maintainability.
