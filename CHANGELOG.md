# Changelog

All notable changes to this project are documented in this file.

## 0.1.3 - 2026-08-28

### Added
- Added `CODE_REVIEW.md` with a full security/quality audit and remediation notes.
- Added extensive validation and edge-case tests in `tests/test_validation_and_edges.py`.
- Added centralized validation helpers in `cleaning_ops.py`:
  - `_ensure_columns_exist`
  - `_validate_sql_fragment`

### Changed
- Hardened SQL-fragment entry points to reject multi-statement, commented, and destructive SQL fragments.
- Added input validation for missing columns and invalid argument values across core, extended, and final cleaning operations.
- Improved `DuckJanitor.sql()` self-table replacement to use word-boundary replacement.
- Added stricter typing updates (`DuckJanitor.__init__ -> None`, typed kwargs in selected methods).
- Improved hybrid robustness for `join_apply` when joining relations from different DuckDB connections.

### Fixed
- Removed dead/unused code in cleaning modules.
- Removed duplicate import entry for `truncate_datetime` in `pyduck_janitor/__init__.py`.

## 0.1.2
- Production-ready stabilization, pure SQL rewrites, and test expansion.

## 0.1.1
- Connection handling and crash fixes.

## 0.1.0
- Initial release.
