# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## 0.4.0 - 2026-10-29

The release that turns pyduck-janitor from a pandas-backed cleaning library
into a DuckDB-native one. pandas is no longer the only way in or out.

### Added — Native output and write-back
Previously the only exit from a pipeline was `collect()`, which called
`relation.df()` and pulled the entire result into pandas — defeating the
out-of-core promise of the package. Now results land directly in DuckDB
formats and destinations:
- `to_parquet()`, `to_csv()`, `to_json()` (array or newline-delimited),
  `to_table()`, `to_view()` — all via DuckDB's `COPY ... TO` /
  `CREATE ... AS`, never through pandas.
- `write_relation()` — the generic `COPY ... TO` escape hatch with
  format, compression, delimiter, and header options.
- `to_arrow()` (zero-copy Arrow table) and `to_polars()`.
- `io_ops.py` is the new module that owns this surface.

### Added — Sessions, persistence, and catalog
- `DuckJanitor.connect(path)` opens a file-backed DuckDB database and
  returns a session object that owns its connection.
- `attach()`, `detach()`, `tables()`, `databases()`, `begin()`, `commit()`,
  `rollback()`, `close()`, and `__enter__`/`__exit__` for `with` blocks.
- Every loader (`from_pandas`, `from_parquet`, `from_csv`, `from_json`,
  `from_excel`, `from_database`) now accepts `connection=`, so data can be
  loaded into a session's database rather than only the shared connection.

### Added — Inspection and diagnostics
- `.columns`, `.dtypes`, `.shape`, `.count()`, `.relation`, `.connection` —
  cheap metadata without materialising the frame.
- `show()` renders a DuckDB-style box table; `glimpse()` aliases `head()`.
- `_repr_html_` gives Jupyter a rich schema + preview instead of a bare
  one-line repr.
- `explain(analyze=False, format='text')` now returns readable plan text.
  The previous implementation returned the raw `str(fetchall())` tuple
  repr, which was unusable.

### Added — Sampling and Python UDFs (`sampling.py`)
- `sample(n=..., frac=..., seed=..., method=...)` via DuckDB's `USING SAMPLE`,
  replacing the full `ORDER BY random()` sort that `shuffle()` required.
- `sample_rows()` for a deterministic N-row pick without a full sort.
- `approx_distinct()`, `approx_quantile()`.
- `register_python_function()` and `apply_python_function()` register a
  Python callable as a DuckDB scalar UDF, so user logic runs *inside*
  DuckDB instead of forcing a pandas round-trip.

### Added — Nested and typed data (`nested_ops.py`)
- `to_struct()` / `from_struct()`, `to_list()` / `from_list()`, `unnest()`.
- `list_transform()`, `list_filter()`, `list_transforms()` — DuckDB lambdas
  over LIST columns.
- `json_extract()` — JSON pointer extraction from a JSON column.

### Fixed — Correctness and safety
- **`add_column()` and `filter_column()` no longer silently degrade a user
  error into a wrong result.** Previously any exception during SQL
  interpretation was caught and the value was stored as a string literal
  instead, so a typo'd column name became data with no warning. A fragment
  that references a known column now raises a `ValueError` naming the
  available columns; only text that clearly is not an expression still
  falls back to a literal.
- **`sql()` no longer rewrites the word `self` inside string literals.**
  The old regex substitution corrupted queries such as
  `SELECT 'self' FROM self`. Substitution is now literal-, identifier-, and
  comment-aware (`_sqlutil.substitute_self`).
- **`sql()` supports parameter binding** (`$name` placeholders) so user
  values never become part of the statement text.
- **Temporary relation names are process-unique.** Names were built from
  `id(relation)`, and CPython reuses the address of a garbage-collected
  object, so a stale registration could collide with a new relation. All
  names now come from a monotonic counter (`_sqlutil.temp_name`).
- **`asof_join()` uses DuckDB's native `ASOF LEFT JOIN`** for the backward
  and forward directions instead of a 190-line rank-and-filter emulation.
  The right side is de-duplicated first so the surviving row matches the
  previous tie-breaking exactly, and tolerance still nulls the right-hand
  columns while preserving the left row.
- **`deduplicate()` uses `QUALIFY`** rather than a wrapping CTE.
- The shared connection is now created under a lock, so concurrent first
  use cannot create two connections.

### Changed
- `profile()` now reports mean, standard deviation, the requested
  quantiles, and the most-frequent values, in addition to null and
  distinct statistics.
- The `duckdb` pin is relaxed to `>=1.5.2,<1.6` to match what the test
  matrix actually proves.
- `ruff` `target-version` corrected to `py310` to match `requires-python`.
- The `pytest` `pythonpath = ["src"]` setting was removed — there is no
  `src/` directory.
- OptBinning's HiGHS `UserWarning` filter is scoped so warnings from other
  libraries still surface.

### Test and CI
- Measured coverage is now **81%** (was 77%), with `fail_under = 80`
  enforced in `pyproject.toml` and a new `coverage` CI job.
- Test suite grew from **364 to 561 tests**.
- `bins_opt.py` — the 0.3.0 headline feature — went from **4% coverage and
  zero tests to 85% coverage** with a dedicated suite.
- The orphaned Hypothesis suite (`tests/property_cleaning_ops.py`, gitignored
  and misnamed so it never ran) is revived as
  `tests/test_property_cleaning_ops.py` and skips cleanly when Hypothesis
  is absent.
- The public-package contract check now covers every 0.4.0 verb.

## 0.4.1 - 2026-10-29

Continues the DuckDB-native work from 0.4.0: the remaining verbs that were
still module-level functions, the set operations, and the remote-file
support the README had been promising but the package never implemented.

### Added — Chainable text and vector verbs
`text_normalize`, `build_fts_index`, `drop_fts_index`, `search_text`,
`keyword_filter`, `embed_column`, `build_vector_index`, `vector_search`, and
`fuzzy_dedupe` were module-level functions only (`text_normalize(dj, ...)`),
which broke the chaining contract every other verb follows. Each is now
also a method on `DuckJanitor`.

- `vector_search()` and `search_text()` accept `return_relation=True` so
  their results can be chained instead of being forced into a DataFrame.
- `fuzzy_dedupe()` now stays on the caller's connection rather than
  silently rebinding the result to the shared in-memory connection.

### Added — Set operations (new `set_ops.py`)
- `union()`, `intersect()`, `except_()` (alias `except_rows`),
  `concat_rows()`, and `select_except()`, each also available as a
  `DuckJanitor` method.
- `union(..., by_name=True)` uses DuckDB's `UNION BY NAME`, so frames with
  different column orders combine correctly and unmatched columns become
  NULL. `UNION ALL BY NAME` is supported by DuckDB 1.5.2 and is used when
  `all=True`.
- DuckDB 1.5.2 has no `INTERSECT BY NAME` / `EXCEPT BY NAME`, so those two
  are emulated by padding both sides to the union of the column sets with
  typed NULLs. The deviation is documented in the module and in each
  function's Notes.
- `select_except()` uses DuckDB's `COLUMNS(* EXCLUDE (...))` and
  `COLUMNS(* REPLACE (...))` star expressions.

### Added — Remote and S3 support (new `remote.py`)
The README advertised `from_parquet('s3://bucket/data.parquet')`, but
nothing in the package loaded DuckDB's `httpfs` extension or configured
credentials. That claim is now true.

- `load_httpfs()`, `configure_s3()`, `create_s3_secret()`, `read_remote()`,
  `list_remote()`, and `is_remote_path()`.
- `httpfs` is registered in `extensions.EXTENSIONS` so the existing loader
  (and its idempotency cache and error mapping) is reused. It loads
  offline from the standard DuckDB wheel, so nothing is downloaded
  implicitly.
- Credentials travel as **bind parameters**, never interpolated into SQL
  text, and are never printed or included in an exception message.
  `configure_s3()` prefers DuckDB's temporary secret manager and falls
  back to `SET s3_*` statements.
- `create_s3_secret()` can persist a secret on request; the default is
  in-memory only, so credentials do not hit disk.

### Fixed
- Calling a verb without a connection raised
  `AttributeError: 'NoneType' object has no attribute 'query'`. A shared
  `_require_conn()` helper now raises a `ValueError` that names the cause
  and points at the method form on `DuckJanitor`.

## 0.3.0 - 2026-09-25

### Added
- **Supervised binning** (`optimize_bins_to_target`, new module
  `pyduck_janitor/bins_opt.py`, chainable on `DuckJanitor`): learn optimal
  cut points for one or more numeric columns from a binary, multiclass, or
  continuous target. Two engines — OptBinning (IV/WoE-optimal, binary
  default) and DecisionTreeDiscretiser (all target types) — with honest
  train/test holdout scoring (`auc`/`f1`/`gini`, `f1_macro`, `r2`/`rmse`),
  text interval labels, `Missing`-bin preservation, and row-capped learning
  (`sample_n`) for large tables. Optional install: `pip install
  "pyduck-janitor[binning]"`. Guide: `docs/supervised_binning.md`;
  runnable demo: `examples/optimize_bins_example.py`.
- **dplyr-style explicit join verbs** (`pyduck_janitor.joins` and methods on
  `DuckJanitor`): `inner_join`, `left_join`, `right_join`, `full_join`,
  `semi_join`, `anti_join`. R/dplyr parity, DuckDB-native SQL.
  - Supports shared-key (`on='dept_id'`) and asymmetric-key
    (`left_on=..., right_on=...`) calling shapes.
  - Configurable `suffixes` for non-key column collisions (default `_x`/`_y`).
  - `semi_join` and `anti_join` have no pandas equivalent — they keep left
    columns only and filter rows via `WHERE EXISTS` / `WHERE NOT EXISTS`.
  - Module-level functions and methods are interchangeable.
  - 20 new tests in `tests/test_dplyr_joins.py`.
  - Auto-generated API entries in `docs/api/functions.md` (6 new sections,
    plus TOC entries).

## 0.2.7 - 2026-09-04

### Fixed
- Fixed Onager graph analytics for string and alpha-numeric node IDs. The
  adapter now maps non-integer endpoints to BIGINT surrogates and restores the
  original labels in result `node_id` values.
- Added regression coverage and a DuckDB 1.5.5 extension smoke check for
  alpha-numeric graph identifiers.

## 0.2.6 - 2026-09-04

### Changed
- Standardized the supported runtime on Python 3.10+ and DuckDB
  `>=1.5.5,<1.6` so core functionality and optional extensions share one
  tested compatibility family.
- Updated CI, package metadata, contributor guidance, and agent guidance to
  use DuckDB 1.5.5 consistently.

## 0.2.5 - 2026-09-04

### Fixed
- Fixed the CI Python 3.9 matrix to use DuckDB 1.4.3, because DuckDB 1.5.x
  requires Python 3.10 or newer.
- Preserved the advertised Python 3.9 core-package support while keeping the
  modern DuckDB 1.5.x extension tests on supported Python versions.

## 0.2.4 - 2026-09-04

### Added
- Added an agent-oriented repository guide in `AGENTS.md`, `llms.txt`, and
  `docs/agent-guide.md` covering the API map, workflows, and boundaries.
- Added a CI agent contract that verifies generated API examples, public
  methods, and structured review output as an artifact.
- Added Python 3.9 to the CI matrix and fixed Python 3.9-incompatible f-string
  expressions in embedding and text helpers.

### Documentation and testing
- Linked agent resources from the README and documented the expanded CI
  contract for contributors.

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
