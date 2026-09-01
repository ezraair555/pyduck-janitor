# pyduck-janitor

<p align="center">
  <img src="docs/duck_janitor_logo.jpg" alt="pyduck-janitor logo" width="200" style="border-radius: 10%;"/>
</p>

**DuckDB-backed pyjanitor for high-performance data cleaning on large datasets**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/pyduck-janitor.svg)](https://badge.fury.io/py/pyduck-janitor)

The objective of this package is to perform data cleaning using an expressive grammar that coheres with the tidyverse design framework, but powered by [DuckDB](https://duckdb.org/) for high-performance execution on large datasets. The package is centered around data cleaning verbs, supplemented with many utilities for data transformation and manipulation.

## Overview

pyduck-janitor provides a method-chaining API for data cleaning operations that mirrors [pyjanitor](https://pyjanitor-devs.github.io/pyjanitor/), but uses DuckDB as the backend for:

- **Speed**: DuckDB's vectorized execution engine accelerates data cleaning operations
- **Scalability**: Process datasets larger than memory by working directly with Parquet, CSV, or other file formats
- **Lazy evaluation**: Build complex cleaning pipelines that execute efficiently
- **Drop-in replacement**: Use familiar pyjanitor syntax with automatic DuckDB optimization

### The Main Verbs

The core functionality of pyduck-janitor is organized around several main groups of verbs:

1. **`clean_names()`** - Standardize column names to a consistent format
2. **`filter_on()`, `filter_string()`** - Filter rows based on conditions or string patterns
3. **`select_columns()`, `select_rows()`** - Select specific columns or rows
4. **`add_column()`, `remove_columns()`, `rename_column()`** - Modify columns
5. **`dropna()`, `remove_empty()`** - Handle missing data
6. **`coalesce()`, `fill()`, `fill_empty()`** - Impute missing values
7. **`encode_categorical()`, `get_dummies()`** - Encode categorical variables
8. **`transform_column()`, `transform_columns()`** - Transform column values
9. **`case_when()`, `find_replace()`** - Conditional transformations
10. **`pivot_wider()`, `pivot_longer()`** - Reshape data
11. **`groupby_agg()`, `groupby_topk()`** - Grouped operations

## Installation

```bash
pip install pyduck-janitor
```

For the development version:

```bash
cd /home/lucas/.openclaw/workspace/pyduck-janitor
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from pyduck_janitor import DuckJanitor

# Load your data
df = DuckJanitor.from_pandas(pd.DataFrame({
    'SalesMonth': ['Jan', 'Feb', 'Mar', 'April'],
    'Company1': [150.0, 200.0, 300.0, 400.0],
    'Company2': [180.0, 250.0, None, 500.0],
    'Company3': [400.0, 500.0, 600.0, 675.0]
}))

# Build a cleaning pipeline
result = (
    df
    .clean_names()
    .remove_columns(['company1'])
    .dropna(subset=['company2', 'company3'])
    .rename_column('company2', 'amazon')
    .add_column('google', [450.0, 550.0, 800.0])
    .collect()
)

print(result)
```

### Select DSL (pyjanitor-compatible)

```python
from pyduck_janitor import DuckJanitor, DropLabel

dj = DuckJanitor.from_pandas(pd.DataFrame({
    'sales_month': ['Jan', 'Feb'],
    'company2': [180.0, 250.0],
    'company3': [400.0, 500.0],
    'notes': ['x', 'y'],
}))

dj.select_columns('sales_month, company2')              # comma-string
dj.select_columns('company*')                           # glob expansion
dj.select_columns('re:^company')                        # regex
dj.select_columns(['company*', DropLabel('company3')])  # exclude one column
```

## Supported Functions

pyduck-janitor implements the **complete pyjanitor documented API** (94/94 functions — verified against the pyjanitor API reference) plus 18 DuckDB-specific extensions, exposed as **109 chainable methods** on `DuckJanitor`.

### Core cleaning verbs (`cleaning_ops.py`)
- `clean_names()` - Clean column names
- `remove_columns()` - Remove columns
- `add_column()` / `add_columns()` - Add a new column (single or dict form)
- `rename_column()` / `rename_columns()` - Rename a column
- `dropna()` - Drop rows with NA values
- `remove_empty()` - Remove empty rows/columns
- `filter_column()` / `filter_column_isin()` - Filter by column condition / IS IN list
- `filter_on()` - Filter with SQL-like criteria
- `filter_string()` - Filter by substring
- `coalesce()` - Merge columns
- `encode_categorical()` - Encode as categorical
- `get_dummies()` - One-hot encode
- `select_columns()` / `select()` - Select columns (supports comma-strings, globs, `re:` regex, and `DropLabel`)
- `select_rows()` - Select rows
- `transform_column()` / `transform_columns()` - Transform one or many columns

### Extended verbs (`cleaning_ops_extended.py`)
- `bin_numeric()` - Bin numeric column
- `change_type()` - Change column type
- `concatenate_columns()` - Join columns
- `deconcatenate_column()` - Split column
- `drop_constant_columns()` - Remove constant columns
- `fill()` / `fill_direction()` - Fill missing values (forward/backward)
- `fill_empty()` - Fill empty strings
- `flag_nulls()` - Flag null values
- `limit_column_characters()` - Truncate column names
- `min_max_scale()` - Scale to [0,1]
- `groupby_agg()` - Group and aggregate
- `groupby_topk()` - Top k per group
- `case_when()` - Conditional logic
- `currency_column_to_numeric()` - Parse currency
- `convert_date()` / `convert_to_date()` / `convert_to_datetime()` - Convert to date/datetime
- `convert_unix_date()`, `convert_excel_date()`, `convert_matlab_date()` - Numeric date conversions
- `truncate_datetime()` / `truncate_datetime_dataframe()` - Truncate datetime
- `pivot_wider()` / `pivot_longer()` - Reshape wide/long

### Hybrid verbs (`cleaning_ops_final.py`)
- `conditional_join()` - Join with condition
- `get_dupes()` - Find duplicate rows
- `dropnotnull()` - Drop non-null values
- `expand_column()` - Expand delimited column
- `impute()` - Impute missing values
- `jitter()` - Add noise to values
- `label_encode()` - Encode as integers
- `find_replace()` - Replace values
- `count_cumulative_unique()` - Count unique values
- `complete()` - Complete missing combinations
- `also()` - Apply multiple operations
- `alias()` - Create column aliases
- `mutate()` / `assign()` / `ungroup()` - Add/modify columns, tidyverse-style verbs
- `drop_duplicate_columns()` - Remove duplicate columns
- `compare_df_cols()` / `compare_df_cols_same()` - Compare column contents/shape
- `join_apply()` - Apply function to joined data
- `process_text()` - Text processing

### pyjanitor parity methods (v0.2.0)

These were added to reach 100% coverage of pyjanitor's documented API:

| pyjanitor name | pyduck-janitor method | Notes |
| --- | --- | --- |
| `rename_columns` | alias of `rename_column` | plural form |
| `truncate_datetime_dataframe` | alias of `truncate_datetime` | |
| `convert_to_date` / `convert_to_datetime` | aliases of `convert_date` | |
| `convert_unix_date` | new | `TO_TIMESTAMP`, seconds/millis/micros |
| `convert_excel_date` | new | Excel serial dates |
| `convert_matlab_date` | new | MATLAB datenums |
| `excel_time_to_numeric` | new | Excel time fraction → seconds |
| `sas_numeric_to_date` | new | SAS origin 1960-01-01 |
| `to_datetime` | new | DuckDB `strptime` cast |
| `fill_direction` | alias of `fill` | |
| `filter_column_isin` | new | quoted-column `IS IN` filter |
| `filter_date` | new | start/end date range filter |
| `add_columns` | new | dict of `{name: values}` |
| `assign` / `ungroup` | aliases of `mutate` / no-op | tidyverse naming |
| `get_columns` / `get_index_labels` | helpers | column introspection |
| `move` / `reorder_columns` | new | column placement verbs |
| `row_to_names` | new | promote a row to headers |
| `rle_id` | new | run-length ids via hash + window |
| `factorize_columns` | new | `DENSE_RANK` integer encoding |
| `sort_naturally` | new | human (non-lexicographic) sort |
| `sort_column_value_order` | new | explicit value ordering |
| `update_where` | new | conditional column update |
| `unionize_dataframe_categories` | new | cross-relation VARCHAR alignment |
| `shuffle` | new | `ORDER BY random()` |
| `toset` | new | distinct values as a list |
| `take_first` | new | first N rows |
| `round_to_fraction` | new | snap to 1/denominator |
| `scale_mad` | new | median-abs-deviation scaling |
| `cartesian_product` | new | cross join helper |
| `then` | new | chain callables |
| `expand` / `expand_grid` | new | distinct expansion / cross join grid |
| `change_index_dtype` | new | typed projection of a column |
| `collapse_levels` | new | concat-join helper |
| `explode_index` | new | regex-extract a parsed column |
| `summarise` | new | group-by aggregation helper |
| `pivot_longer_spec` / `pivot_wider_spec` | new | UNPIVOT / PIVOT spec forms |
| `join_agg` / `get_join_indices` | new | aggregated/non-equi join helpers |
| `select` DSL | built into `select_columns` | comma-strings, globs, `re:` regex, `DropLabel` |
| `DropLabel` | exported class | select-DSL exclusion sentinel |
| `patterns` | exported function | regex helper with `.compiled` |
| `describe_class` | method | column-type table (DESCRIBE-backed) |

## Supported Data Sources

pyduck-janitor can work with data from:

- **In-memory pandas DataFrames** - Via `from_pandas()`
- **Parquet files** - Local or remote (S3, HTTP)
- **CSV files** - Local or remote
- **JSON files** - Local or remote
- **DuckDB databases** - Existing `.duckdb` files
- **SQL queries** - Custom SQL as input

## Key Features

### Lazy Evaluation

Operations build a query plan without immediate execution. Use `.collect()` to execute:

```python
result = df.clean_names().remove_empty().dropna().collect()
```

### Out-of-Core Processing

Work with datasets larger than RAM:

```python
df = DuckJanitor.from_parquet('large_dataset.parquet')
result = df.clean_names().remove_empty().collect()
```

### Method Chaining

All methods return DuckJanitor objects, enabling fluent pipelines:

```python
result = (
    df
    .clean_names()
    .filter_on('age > 18')
    .groupby_agg('gender', {'income': 'mean'})
    .collect()
)
```

### SQL Interoperability

Mix janitor methods with custom SQL:

```python
result = df.sql('SELECT * FROM self WHERE age > 18').collect()
```

### SQL Fragment Safety (0.1.3)

For methods that accept SQL fragments (for example `filter_on`, `filter_column`, `select_rows(criteria=...)`, `transform_column(func=...)`, `case_when`, and `change_type`), `pyduck-janitor` now rejects:

- Multi-statement fragments containing `;`
- SQL comments (`--`, `/* ... */`)
- Destructive DDL/DML keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, etc.)

This keeps expression-based APIs usable while reducing accidental or unsafe query fragments.

## API Comparison

### Traditional pandas + pyjanitor

```python
import pandas as pd
import janitor

df = pd.read_csv('large_file.csv')
df = (
    df
    .clean_names()
    .remove_empty()
    .dropna(subset=['col1', 'col2'])
)
```

### pyduck-janitor (faster, scalable)

```python
from pyduck_janitor import DuckJanitor

df = DuckJanitor.from_csv('large_file.csv')
df = (
    df
    .clean_names()
    .remove_empty()
    .dropna(subset=['col1', 'col2'])
)
result = df.collect()  # Explicit execution
```

## Examples

See the `examples/` directory for complete workflows:

- `examples/basic_cleaning.py` - Basic data cleaning pipeline
- `examples/large_dataset.py` - Out-of-core processing with Parquet
- `examples/sql_interop.py` - Mixing janitor methods with SQL
- `examples/comparison.py` - Performance comparison with pandas + pyjanitor

## Architecture

pyduck-janitor works by:

1. **Wrapping DuckDB relations** - Data is stored in DuckDB tables
2. **Translating janitor methods** - Each method converts to DuckDB SQL
3. **Lazy evaluation** - Operations build a query plan
4. **Optimized execution** - DuckDB executes the entire pipeline efficiently
5. **Pandas compatibility** - Results can be converted to pandas DataFrames

### Hybrid Pattern

For operations that can't be pure SQL:

1. **Materialize** - Convert DuckDB relation to pandas DataFrame
2. **Apply** - Execute Python function
3. **Re-wrap** - Create new DuckJanitor instance

## Performance

pyduck-janitor provides significant speedups for:

- Large datasets (>1M rows)
- Complex cleaning pipelines
- Operations on disk-based data
- Column-wise transformations

Benchmark results vary by workload, but expect 2-10x speedups on typical data cleaning tasks.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.


## Changelog

### 0.2.0 — 100% pyjanitor API parity

- **Full pyjanitor surface**: pyduck-janitor now covers **94/94 (100%)** of the
  functions documented on the pyjanitor API reference, verified by a fresh
  scan of pyjanitor's live docs.
- **~35 new chainable methods** on `DuckJanitor`, including:
  - Date conversions: `convert_unix_date`, `convert_excel_date`,
    `convert_matlab_date`, `excel_time_to_numeric`, `sas_numeric_to_date`,
    `to_datetime` (all with float-safe numeric parsing via `TO_TIMESTAMP`),
    plus `convert_to_date` / `convert_to_datetime` aliases.
  - Structural verbs: `move`, `reorder_columns`, `get_columns`,
    `get_index_labels`, `row_to_names`, `collapse_levels`, `explode_index`,
    `change_index_dtype`.
  - Reshape/aggregation: `expand`, `expand_grid`, `summarise`,
    `pivot_longer_spec` (UNPIVOT), `pivot_wider_spec` (PIVOT with a literal
    value list), `join_agg`, `get_join_indices`.
  - Data quality / encoding: `rle_id`, `factorize_columns`, `update_where`,
    `unionize_dataframe_categories`, `scale_mad`, `round_to_fraction`.
  - Row/column utilities: `shuffle`, `toset`, `take_first`,
    `sort_naturally`, `sort_column_value_order`, `filter_date`,
    `cartesian_product`, `then`.
  - pyjanitor naming aliases: `rename_columns`, `truncate_datetime_dataframe`,
    `fill_direction`, `filter_column_isin`, `add_columns`, `assign`, `ungroup`.
- **Select DSL**: `select_columns` now accepts comma-separated strings
  (`"a, b, c"`), shell-globs (`"value*"`), and regex (`"re:^v_"`), matching
  pyjanitor's `select` helper where it lives (under `select_columns`).
  A thin `select()` alias is exposed; non-column kwargs raise
  `NotImplementedError` (pyjanitor itself deprecates them).
- **pyjanitor helper surface**: `DropLabel` (select-DSL exclusion sentinel,
  functional in mixed lists), `patterns` (regex helper), and
  `describe_class()` (DESCRIBE-backed column-type table).
- **Test suite**: grew from 193 to **284 passing tests**
  (+91 alias/DSL/parity tests in `tests/test_pyjanitor_aliases.py`).
- **Docs**: fixed stale example references and expanded the supported
  functions section; parity analysis in
  [`REVIEW_PYJANITOR_PARITY.md`](REVIEW_PYJANITOR_PARITY.md).

### 0.1.3 — Validation hardening, audit documentation, and test expansion

- Added a full package audit report in `CODE_REVIEW.md`.
- Added stronger validation for SQL-fragment inputs and missing-column errors across cleaning modules.
- Added expanded edge-case and error-path tests in `tests/test_validation_and_edges.py`.
- Improved `join_apply` cross-connection handling and `DuckJanitor.sql()` identifier replacement behavior.
- Bumped package version to `0.1.3`.

### 0.1.2 — Production-ready stabilization, pure SQL rewrites, and test expansion

- **Pure SQL Rewrites**: Rewrote `alias`, `complete`, and `drop_duplicate_columns` in 100% pure, out-of-core SQL to avoid in-memory materialization to Pandas.
- **API Expositions**: Exposed previously hidden hybrid and final functions (`drop_duplicate_columns`, `compare_df_cols`, `join_apply`, `process_text`, and `get_dupes`) directly as wrapper methods on `DuckJanitor`.
- **Bug Fixes**:
  - Fixed syntax parser errors in `fill` by introducing physical row number index CTEs instead of nesting window functions.
  - Added safe string literal quoting fallback to `add_column` and `filter_column` when passing raw string scalars.
  - Resolved name-collision bugs in `clean_names` and `coalesce`.
  - Implemented group-by partitioning support in `impute` using SQL window functions.
  - Ensured operations like `fill_empty`, `currency_column_to_numeric`, and `convert_date` gracefully return NULLs instead of crashing.
- **Metadata Update**: Updated package version to `0.1.2` and author information.
- **Unit Test Suite**: Added 54 new test cases covering all edge cases, raising code coverage from **44% to 93%** (with 100% coverage on `duck_janitor.py`).
- **Logo Sticker**: Added a custom package logo sticker of a duck dressed as a janitor.

### 0.1.1 — Connection handling and crash fixes

- Fixed cross-connection crashes across `cleaning_ops.py`,
  `cleaning_ops_extended.py`, and `cleaning_ops_final.py` by registering
  relations on the caller's DuckDB connection instead of creating new
  in-memory connections or relying on `FROM relation` replacement scans.
- `DuckJanitor.__init__` now validates that the relation and connection
  belong to the same DuckDB connection.
- `from_parquet`, `from_csv`, and `from_sql` now return real DuckDB
  relations without round-tripping through pandas.
- Fixed `remove_empty` to actually remove all-empty rows (in addition to
  all-empty columns).
- Fixed `dropna(how='all')` boolean condition.
- Fixed `case_when`, `currency_column_to_numeric`, `convert_date`
  `relation.database` AttributeError crashes.
- Fixed `impute()` `SELECT , COALESCE(...)` syntax error.
- Fixed `conditional_join` to use a single shared connection with an
  operator allow-list.
- Replaced invalid `ROW() OVER ()` in `select_rows` with
  `ROW_NUMBER() OVER ()`.
- Added safer handling for identical-value columns in `min_max_scale`.
- Added 10 regression tests. Full suite: 40 passing.

### 0.1.0 — Initial release

- DuckDB-backed pyjanitor-style cleaning API with 51 functions.
- Lazy SQL evaluation for simple operations; hybrid SQL/Python for
  complex operations.

For a complete release history, see [CHANGELOG.md](CHANGELOG.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [pyjanitor](https://pyjanitor-devs.github.io/pyjanitor/) - Original data cleaning API
- [DuckDB](https://duckdb.org/) - High-performance analytical database
- [infer](https://infer.netlify.app/) - Inspiration for the tidy grammar approach
- [duckplyr](https://duckplyr.tidyverse.org/) - Inspiration for DuckDB-backed tidyverse
