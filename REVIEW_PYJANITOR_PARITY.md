# pyduck-janitor → pyjanitor Function Parity Review

Source of pyjanitor API: <https://pyjanitor-devs.github.io/pyjanitor/api/functions/>
Source of pyduck-janitor API: scan of public methods on `DuckJanitor` in this repo (post quick-wins pass).

## Summary (after v0.1.4 quick-wins alias pass)

- pyjanitor declared functions: **92**
- pyduck-janitor public methods: **69**
- Shared by name: **61**
- Missing in pyduck-janitor: **31**
- Extra (DuckDB-only / convenience) in pyduck-janitor: **8**

**Coverage (shared / pyjanitor): 66.3%** (was 49.5% before the quick-wins alias pass).

## ✅ Newly aligned in this pass

| pyjanitor name | pyduck-janitor status |
| --- | --- |
| `rename_columns` | alias of `rename_column` |
| `truncate_datetime_dataframe` | alias of `truncate_datetime` |
| `convert_to_date` | alias of `convert_date` |
| `convert_to_datetime` | alias of `convert_date` |
| `convert_unix_date` | new (s/m/ms; uses `TO_TIMESTAMP`) |
| `convert_excel_date` | new (Excel serial, `(x - 25569) * 86400`) |
| `convert_matlab_date` | new (MATLAB serial, `(x - 719529) * 86400`) |
| `fill_direction` | alias of `fill` |
| `filter_column_isin` | alias of `filter_column` (quoted-column IS IN) |
| `add_columns` | accepts a dict, loops `add_column` |
| `assign` | alias of `mutate` |
| `ungroup` | no-op identity verb |
| `get_columns` | alias of `select_columns` |
| `move` | new (per-column reorder) |
| `reorder_columns` | new (mass reorder; drops unlisted columns) |
| `get_index_labels` | new (returns current column list) |

## ➕ Extra (DuckDB-only / package additions)

- `convert_date`, `convert_unix_date`, `convert_excel_date`, `convert_matlab_date`,
  `fill`, `filter_column`, `truncate_datetime`, `dropna`, `get_dummies`,
  `head`, `sql`, `explain`. Worth documenting as pyduck-janitor-specific extensions.

## ❌ Still missing in pyduck-janitor

Priority list (next passes):

1. **Date / time utilities still missing**: `excel_time_to_numeric`, `sas_numeric_to_date`, `to_datetime`, `convert_*` aliases for the rest, `filter_date`, `truncate_datetime` exact signature.

2. **Structural / reshape / tidyverse verbs**: `add_columns` (heavy multi-column dict), `cartesian_product`, `expand`, `expand_grid`, `explode_index`, `factorize_columns`, `pivot_longer_spec`, `pivot_wider_spec`, `shuffle`, `sort_column_value_order`, `sort_naturally`, `take_first`, `then`, `toset`, `update_where`.

3. **Conditional-join extensions**: `join_agg`, `get_join_indices`, `compare_df_cols_same`, `unionize_dataframe_categories`.

4. **Aggregation / summarisation helpers**: `summarise`, `groupby_*` enhancements, `rle_id`, `row_to_names`, `round_to_fraction`, `scale_mad`, `factorize_columns`.

5. **String helper**: `select` (the string DSL sugar on top of `select_columns`).

6. **Index helper**: `change_index_dtype`.

## Recommended next pass

Goal: push coverage to ~80% (i.e., add ~12 more functions).

Highest-value additions (smallest blast radius):

| Function | Estimated effort | Notes |
| --- | --- | --- |
| `add_columns(dict)` | 30 min | Loop variant — already partly shipped; needs full multi-column add-on |
| `shuffle` | 15 min | `SELECT * FROM tbl ORDER BY random()` |
| `toset` | 10 min | `SELECT DISTINCT column FROM tbl` |
| `round_to_fraction` | 20 min | DuckDB `ROUND(x*frac)/frac` |
| `select(string)` | 30 min | Parse pyjanitor string DSL |
| `factorize_columns` | 30 min | `DENSE_RANK() OVER (ORDER BY col)` |
| `unionize_dataframe_categories` | 30 min | CAST column to VARCHAR, etc. |
| `compare_df_cols_same` | 20 min | Compare column sets; raise if different |
| `cartesian_product` / `expand` / `expand_grid` | 60 min | DuckDB CROSS JOIN, optional set ops |
| `rle_id` | 30 min | `LAG(col) OVER () <> col` per group |
| `row_to_names` | 15 min | First row becomes header |
| `scale_mad` | 20 min | (x - median) / MAD |
| `take_first` | 10 min | LIKE `head()` |
| `excel_time_to_numeric`, `sas_numeric_to_date` | 30 min | DuckDB time-fraction conversions |

Then 80–85% becomes reachable; the remaining gap is long-tail (NUMBA-specific code in pyjanitor such as `add_columns`-with numba acceleration, plus the `_spec`-form pivot helpers).

## Notes

- All 218 tests pass (193 pre-existing + 25 new alias tests in `tests/test_pyjanitor_aliases.py`).
- The DuckDB-specific `_register_relation` / `_unregister` lifecycle is now used consistently to keep validation happy in `DuckJanitor.__init__`.
- Reserved DuckDB keywords (e.g. `group`) are quoted in `filter_column_isin` SQL.
- New verbs are exposed as instance methods on `DuckJanitor`, so they remain method-chainable.
