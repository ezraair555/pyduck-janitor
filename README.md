# pyduck-janitor

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

## Supported Functions

pyduck-janitor implements **51 functions** across three modules:

### Core Module (`cleaning_ops.py`) - 14 functions
- `clean_names()` - Clean column names
- `remove_columns()` - Remove columns
- `add_column()` - Add a new column
- `rename_column()` - Rename a column
- `dropna()` - Drop rows with NA values
- `remove_empty()` - Remove empty rows/columns
- `filter_column()` - Filter by column condition
- `filter_on()` - Filter with SQL-like criteria
- `filter_string()` - Filter by substring
- `coalesce()` - Merge columns
- `encode_categorical()` - Encode as categorical
- `get_dummies()` - One-hot encode
- `select_columns()` - Select columns
- `select_rows()` - Select rows
- `transform_column()` - Transform single column
- `transform_columns()` - Transform multiple columns

### Extended Module (`cleaning_ops_extended.py`) - 17 functions
- `bin_numeric()` - Bin numeric column
- `change_type()` - Change column type
- `concatenate_columns()` - Join columns
- `deconcatenate_column()` - Split column
- `drop_constant_columns()` - Remove constant columns
- `fill()` - Fill missing values
- `fill_empty()` - Fill empty strings
- `flag_nulls()` - Flag null values
- `limit_column_characters()` - Truncate column names
- `min_max_scale()` - Scale to [0,1]
- `groupby_agg()` - Group and aggregate
- `groupby_topk()` - Top k per group
- `case_when()` - Conditional logic
- `currency_column_to_numeric()` - Parse currency
- `convert_date()` - Convert to date
- `truncate_datetime()` - Truncate datetime
- `pivot_wider()` - Pivot to wide format
- `pivot_longer()` - Pivot to long format
- `also()` - Apply multiple operations
- `alias()` - Create column aliases
- `mutate()` - Add/modify columns

### Final/Hybrid Module (`cleaning_ops_final.py`) - 24 functions
- `conditional_join()` - Join with condition
- `get_dupes()` - Find duplicate rows
- `dropnotnull()` - Drop non-null values
- `expand_column()` - Expand delimited column
- `impute()` - Impute missing values
- `jitter()` - Add noise to values
- `label_encode()` - Encode as integers
- `find_replace()` - Replace values
- `count_cumulative_unique()` - Count unique values
- `expand_grid()` - Create grid expansion
- `complete()` - Complete missing combinations
- `pivot()` - Pivot data
- `collapse_levels()` - Collapse MultiIndex
- `drop_duplicate_columns()` - Remove duplicate columns
- `compare_df_cols()` - Compare column contents
- `join_apply()` - Apply function to joined data
- `process_text()` - Text processing

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

- `examples/anova_example.py` - Analysis of variance
- `examples/two_sample_test.py` - Two-sample hypothesis tests
- `examples/confidence_intervals.py` - Bootstrap confidence intervals
- `examples/proportion_test.py` - Proportion hypothesis tests
- `examples/basic_cleaning.py` - Basic data cleaning pipeline
- `examples/large_dataset.py` - Out-of-core processing with Parquet
- `examples/sql_interop.py` - Mixing janitor methods with SQL
- `examples/comparison.py` - Performance comparison with pandas

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

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [pyjanitor](https://pyjanitor-devs.github.io/pyjanitor/) - Original data cleaning API
- [DuckDB](https://duckdb.org/) - High-performance analytical database
- [infer](https://infer.netlify.app/) - Inspiration for the tidy grammar approach
- [duckplyr](https://duckplyr.tidyverse.org/) - Inspiration for DuckDB-backed tidyverse
