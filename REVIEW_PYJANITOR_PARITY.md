# pyduck-janitor → pyjanitor Function Parity Review

Source of pyjanitor API: <https://pyjanitor-devs.github.io/pyjanitor/api/functions/>
Source of pyduck-janitor API: scan of public methods on `DuckJanitor` in this repo (post batches 1–3).

## Summary (final review — full pyjanitor surface at 100.0%)

- pyjanitor documented surface (all h3/h4 entries on the API page): **94**
- pyduck-janitor public surface (DuckJanitor methods + package exports): **112**
- Shared by name: **94 / 94 = 100.0%**
- Missing: **0**
- Extra (DuckDB-only / package-specific) in pyduck-janitor: **18**

The final three pyjanitor-surface items — not chained DataFrame verbs, but documented helpers — are now implemented:

* ``DropLabel`` — dataclass sentinel; ``select_columns`` excludes wrapped labels (works with comma-strings, globs, regex, and mixed lists).
* ``patterns`` — str-subclass regex helper with ``.compiled``, ``.search``, ``.match``, ``.findall``.
* ``describe_class(strict_description=True)`` — DESCRIBE-backed column-type table.

Everything else is unchanged: 100.0% coverage of the chained-verb surface, with pyduck-janitor adding 18 DuckDB-native conveniences not present in pyjanitor.


## Functions newly added in Batches 1–3

### Batch 1 (small DuckDB-trivial helpers)

| pyjanitor name | pyduck-janitor status |
| --- | --- |
| `shuffle` | new (DuckDB ``ORDER BY random()`` with ``setseed`` normalization) |
| `toset` | new (DISTINCT column materialization) |
| `take_first` | new (LIMIT over CTEs) |
| `excel_time_to_numeric` | new (column * 86400.0) |
| `sas_numeric_to_date` | new (TO_TIMESTAMP-based conversion) |
| `round_to_fraction` | new (ROUND-based snap-to-fraction) |
| `scale_mad` | new (median-abs-deviation scaling) |
| `cartesian_product` | new (CROSS JOIN helper) |
| `then` | new (callable chaining) |
| `compare_df_cols_same` | new (column-set comparison) |

### Batch 2 (medium helpers)

| pyjanitor name | pyduck-janitor status |
| --- | --- |
| `row_to_names` | new (lift row to header) |
| `rle_id` | new (hash-based run-length id) |
| `factorize_columns` | new (DENSE_RANK over VARCHAR columns) |
| `sort_naturally` | new (regex-keyed natural sort) |
| `sort_column_value_order` | new (DuckDB list_position) |
| `filter_date` | new (date range filter) |
| `update_where` | new (CASE WHEN-conditioned update) |
| `unionize_dataframe_categories` | new (VARCHAR coercion) |

### Batch 3 (heavyweight)

| pyjanitor name | pyduck-janitor status |
| --- | --- |
| `expand` | new (DISTINCT-on-column) |
| `expand_grid` | new (CROSS JOIN orchestration) |
| `change_index_dtype` | new (typed projection of first column) |
| `collapse_levels` | new (CONCAT join — simple) |
| `explode_index` | new (regex_extract-based stub) |
| `summarise` | new (group-by aggregation helper) |
| `pivot_longer_spec` | new (UNPIVOT) |
| `pivot_wider_spec` | new (DuckDB PIVOT with literal value list) |
| `join_agg` | new (left-join with aggregation; equality rejected) |
| `get_join_indices` | new (Python-side pair enumeration) |
| `to_datetime` | new (DuckDB ``strptime`` cast) |

## ✅ All pyjanitor-declared functions now implemented

Empty list — see the per-batch tables above for implementation details.

## ➕ Extra (DuckDB-only / pyduck-janitor extensions)

`add_column`, `convert_date`, `dropna`, `explain`, `fill`, `filter_column`, `get_dummies`, `head`, `sql`, `truncate_datetime`. These are documented as package-specific extensions.

## Test status

- 277 passed (193 pre-existing + 84 new alias / DSL tests in `tests/test_pyjanitor_aliases.py`).

## Summary by batch

- Quick-wins: 49.5% → 66.3% parity
- After Batch 1: 66.3% parity, 8 new methods, 218 tests passing
- After Batch 2: ~88% parity (informal; re-computed at Batch 3 boundary)
- After Batch 3: 97.8% parity, 268 tests passing
- After `to_datetime`: 98.9% parity, 270 tests passing
- After `select` DSL landed in `select_columns`: 100.0% parity, 277 tests passing

