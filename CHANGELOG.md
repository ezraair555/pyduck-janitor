# Changelog

All notable changes to this project are documented in this file.

## 0.2.3 - 2026-09-04

### Added
- Added database-native metric APIs: `metrics()`, `profile()`, and
  `metric_cube()` with rollups, cubes, grouping sets, and total-row metadata.
- Added `rate_metrics()`, `cohort_metrics()`, `freshness()`, and `reconcile()`
  for reusable rates, retention, source monitoring, and relation comparison.
- Added `metric_from_database()` to aggregate source SQL before transferring
  results from DB-API connections.
- Added generic ONA and temporal helpers including `validate_keys()`,
  `deduplicate()`, `filter_noise()`, `hierarchy_edges()`, `time_slice()`,
  `event_window()`, `change_detection()`, and `network_evolution()`.

### Documentation and testing
- Added generated API documentation, examples, and regression coverage for
  database metrics, temporal analysis, and generic network workflows.
- Verified the release candidate with 343 tests and 130 documentation examples.

## 0.2.2 - 2026-09-04

### Added
- Added first-class `asof_join()` with backward, forward, nearest, key, and
  tolerance matching for point-in-time analysis.
- Added `window_mutate()` for composable partitioned, ordered, and framed
  window expressions.
- Added `recursive_cte()` for hierarchy traversal, path enumeration, and
  reachability analysis.
- Added optional Onager graph analytics through the `[graph]` extra, including
  `graph_analyze()` and the flexible `graph_algorithm()` escape hatch.
- Added optional `duck_diff` snapshot comparison through the `[diff]` extra,
  including `diff()`, `diff_summary()`, and `schema_diff()`.
- Added DuckDB Community Extension loading for Onager and duck_diff with
  explicit, opt-in installation and DuckDB 1.5.x compatibility guidance.

### Documentation and testing
- Added API documentation, examples, extension loading guidance, and
  regression coverage for temporal, recursive, graph, and snapshot analysis.
- Verified the release candidate with the full test suite and 115 generated
  documentation examples.

## 0.2.1 - 2026-09-04

### Added
- Added `DuckJanitor.from_database()` for loading query results from open
  DB-API 2.0 connections, including Vertica and Microsoft SQL Server drivers.
- Added parameter binding, optional pandas query arguments, and chunked-read
  support while preserving the caller's connection lifecycle.
- Added regression coverage and documentation for external database loading.

## 0.2.0 - 2026-09-03

### Added
- **Text & similarity verb families** backed by three lazy-loaded DuckDB
  extensions (`icu`, `fts`, `vss`) — new module `pyduck_janitor/text_ops.py`:
  - `text_normalize` — lowercase, accent strip (Python `unicodedata` NFD +
    combining-mark filter), whitespace collapse, optional Unicode form.
  - `build_fts_index` / `drop_fts_index` / `search_text` / `keyword_filter` —
    BM25-ranked full-text search with stemmer + stopword options.
- **Embedding model management + vector search** — new module
  `pyduck_janitor/embeddings.py`:
  - `embed_install` — three source modes: bundled companion wheel,
    HuggingFace (`hf:org/model[@rev]`, honors `HF_TOKEN`), or local path.
    Idempotent; never auto-downloads.
  - `embed_list_installed` / `embed_remove` — cache inspection + cleanup
    (`~/.cache/pyduck-janitor/embeddings/`, override via `PYDUCK_EMBED_CACHE`).
  - `embed_column` — sentence-transformers embeddings as a FLOAT[N] column.
  - `build_vector_index` / `vector_search` — HNSW index (cosine/ip/l2sq)
    + kNN search with optional distance threshold.
  - `fuzzy_dedupe` — near-duplicate detection via embedding cosine
    distance with union-find grouping.
- **Extension management** — new module `pyduck_janitor/extensions.py`:
  per-connection lazy `INSTALL`/`LOAD` with idempotent caching and typed
  `ExtensionNotAvailable` errors that name the pip extra to install.
  Set `PYDUCK_SKIP_EXTENSIONS=1` in locked-down environments.
- **Typed error surface**: `ExtensionNotAvailable`, `EmbeddingsNotAvailable`
  (both carry actionable install hints; no silent network calls).
- **Companion wheel** `pyduck-janitor-embeddings` (separate package):
  bundles `all-MiniLM-L6-v2` weights so `embed_install()` works offline
  after `pip install pyduck-janitor[embeddings]`.
- New extras: `[vss]` (sentence-transformers + hub), `[embeddings]`
  (companion wheel + vss deps), `[text]` (no-op placeholder).
- 33 new tests (`tests/test_text_ops.py`, `tests/test_embeddings.py`);
  full suite 317 passing.
- New docs: `docs/text_ops.md` covering all verb families, install
  modes, and a decision table.

- **100% pyjanitor API parity** — 94/94 of the functions documented on the pyjanitor API reference are now covered, verified by a fresh scan of pyjanitor's live docs. Parity analysis: `REVIEW_PYJANITOR_PARITY.md`.
- ~35 newly added chainable methods on `DuckJanitor` (all ports of pyjanitor functions — same names except where noted as aliases):
  - Date conversions: `convert_unix_date`, `convert_excel_date`, `convert_matlab_date`, `excel_time_to_numeric`, `sas_numeric_to_date`, `to_datetime` (float-safe via `TO_TIMESTAMP`), plus `convert_to_date`/`convert_to_datetime` aliases.
  - Structural verbs: `move`, `reorder_columns`, `get_columns`, `get_index_labels`, `row_to_names`, `collapse_levels`, `explode_index`, `change_index_dtype`.
  - Reshape/aggregation: `expand`, `expand_grid`, `summarise`, `pivot_longer_spec` (UNPIVOT), `pivot_wider_spec` (PIVOT), `join_agg`, `get_join_indices`.
  - Data quality / encoding: `rle_id`, `factorize_columns`, `update_where`, `unionize_dataframe_categories`, `scale_mad`, `round_to_fraction`.
  - Row/column utilities: `shuffle`, `toset`, `take_first`, `sort_naturally`, `sort_column_value_order`, `filter_date`, `cartesian_product`, `then`.
  - pyjanitor naming aliases: `rename_columns`, `truncate_datetime_dataframe`, `fill_direction`, `filter_column_isin`, `add_columns`, `assign`, `ungroup`.
- **Select DSL** in `select_columns`: comma-strings (`"a, b, c"`), shell-globs (`"value*"`), regex (`"re:^v_"`); plus a thin `select()` alias matching pyjanitor's placement of `select` under the select family.
- **pyjanitor helper surface**: `DropLabel` (functional select-DSL exclusion sentinel), `patterns` (regex helper), `describe_class()` (DESCRIBE-backed column types).
- 91 new tests in `tests/test_pyjanitor_aliases.py` — full suite grows from 193 to 284 passing.

### Fixed
- README now references only example files that exist (`anova_example.py`, `two_sample_test.py`, `confidence_intervals.py`, `proportion_test.py` were phantom references from an earlier draft and are removed).
- Supported-functions section updated from "51 functions" to the full 109-method surface.

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
- Added consistent `target_column` validation across `min_max_scale`, `transform_column`, `truncate_datetime`, `pivot_wider`, and `pivot_longer` (M3).
- Improved `DuckJanitor.sql()` self-table replacement to use word-boundary replacement.
- Added stricter typing updates (`DuckJanitor.__init__ -> None`, typed kwargs in selected methods).
- Improved hybrid robustness for `join_apply` when joining relations from different DuckDB connections.
- `filter_string` now catches `re.error` and raises a clean `ValueError` with an actionable message.
- `transform_columns` now accepts any sequence (not just `list`) for `target_columns` (L4).
- Fixed README "Supported Functions" counts to match actual module contents (L2).
- Updated CONTRIBUTING.md git clone URL from placeholder to actual repo (L7).

### Removed
- Removed duplicate import entry for `truncate_datetime` in `pyduck_janitor/__init__.py`.
- Removed unused `re` import from `cleaning_ops_extended.py` and `cleaning_ops_final.py`.
- Removed unused `escaped_col` variable in `pivot_longer`.

### Fixed
- Fixed `_validate_sql_fragment` false positives (H1/H2 from minimax review) that blocked legitimate string literals and column names.
- Fixed `filter_string` to raise `ValueError` instead of raw `re.error` for invalid regex patterns (H3).
- Fixed `add_column` to fall back to string literal when SQL validation rejects the value (M4).

## 0.1.2
- Production-ready stabilization, pure SQL rewrites, and test expansion.

## 0.1.1
- Connection handling and crash fixes.

## 0.1.0
- Initial release.
