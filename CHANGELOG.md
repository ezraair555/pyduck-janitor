# Changelog

All notable changes to this project are documented in this file.

## 0.1.3 - 2026-08-28

### Added
- Added `CODE_REVIEW.md` with a full security/quality audit and remediation notes.
- Added `REVIEW_MINIMAX.md` with an independent review of the hardening pass.
- Added extensive validation and edge-case tests in `tests/test_validation_and_edges.py`.
- Added centralized validation helpers in `cleaning_ops.py`:
  - `_ensure_columns_exist`
  - `_validate_sql_fragment`
  - `_strip_sql_literals`

### Changed
- Hardened SQL-fragment entry points to reject multi-statement, commented, and destructive SQL fragments.
- SQL-fragment validator now strips string literals and quoted identifiers before pattern matching, preventing false positives on values like `'a--b'` or column names like `drop`.
- Destructive keyword matching now uses statement-form patterns (e.g. `DROP TABLE`, `DELETE FROM`) instead of bare keyword matching, allowing column names that match reserved words.
- Added input validation for missing columns and invalid argument values across core, extended, and final cleaning operations.
- Improved `DuckJanitor.sql()` self-table replacement to use word-boundary replacement.
- Added stricter typing updates (`DuckJanitor.__init__ -> None`, typed kwargs in selected methods).
- Improved hybrid robustness for `join_apply` when joining relations from different DuckDB connections.
- `filter_string` now catches `re.error` and raises a clean `ValueError` with an actionable message.

### Fixed
- Removed dead/unused code in cleaning modules.
- Removed duplicate import entry for `truncate_datetime` in `pyduck_janitor/__init__.py`.
- Fixed `_validate_sql_fragment` false positives (H1/H2 from minimax review) that blocked legitimate string literals and column names.
- Fixed `filter_string` to raise `ValueError` instead of raw `re.error` for invalid regex patterns (H3).

## 0.1.2
- Production-ready stabilization, pure SQL rewrites, and test expansion.

## 0.1.1
- Connection handling and crash fixes.

## 0.1.0
- Initial release.
