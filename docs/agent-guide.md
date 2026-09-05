# Agent Guide

## What this package does

`pyduck-janitor` wraps DuckDB relations in a lazy, chainable
`DuckJanitor` object. It is intended for data cleaning, analytical SQL,
temporal analysis, database metrics, graph analysis, and snapshot comparison.
Methods normally return another `DuckJanitor`; call `.collect()` only at the
boundary where a pandas DataFrame is needed.

## Capability map

| Need | Start with | Notes |
| --- | --- | --- |
| Load files/dataframes | `from_pandas`, `from_csv`, `from_parquet`, `from_excel` | Uses DuckDB readers where possible |
| Load an external database | `from_database` | Accepts an open DB-API connection; materializes the result |
| Push aggregation to source DB | `metric_from_database` | Wraps portable aggregate SQL around source SQL |
| Clean and reshape | `clean_names`, `filter_on`, `select_columns`, `groupby_agg` | Core pyjanitor-style verbs |
| Named database metrics | `metrics`, `profile`, `metric_cube` | Aggregates remain in DuckDB |
| Rates and cohorts | `rate_metrics`, `cohort_metrics` | Safe ratios and retention periods |
| Data monitoring | `freshness`, `validate_keys`, `reconcile` | Quality and source-health checks |
| Temporal enrichment | `asof_join`, `time_slice`, `event_window` | Point-in-time analysis |
| Ordered analytics | `window_mutate`, `change_detection` | Windows, deltas, and field changes |
| Hierarchies | `recursive_cte`, `hierarchy_edges` | Traversal and normalized edges |
| Graph evolution | `network_evolution`, `graph_analyze` | Onager is optional |
| Snapshot comparison | `diff`, `diff_summary`, `schema_diff` | duck_diff is optional |
| Escape hatch | `sql` | Use `self` as the current relation name |

## Typical database workflow

```python
from pyduck_janitor import DuckJanitor

data = DuckJanitor.from_database(connection, query)
quality = data.profile()
summary = data.metrics(
    {
        "entities": ("employee_id", "count_distinct"),
        "total_value": ("value", "sum"),
    },
    group_by="department",
)
result = summary.collect()
```

For a large external source, aggregate before transfer:

```python
summary = DuckJanitor.metric_from_database(
    connection,
    "SELECT department, employee_id, value FROM source_table",
    {"entities": ("employee_id", "count_distinct"), "total_value": ("value", "sum")},
    group_by="department",
)
```

## Optional extensions

Install Python extras when needed:

```bash
python -m pip install "pyduck-janitor[graph,diff]"
```

The extra does not silently download native binaries. Use
`auto_install=True` only when runtime network access and extension download
are explicitly permitted. Onager and `duck_diff` require a matching DuckDB
minor version; the supported extension CI target is DuckDB 1.5.5.

## Agent safety and compatibility

- Prefer public methods and the generated API reference over private relation
  internals.
- Preserve caller-owned database connections; `from_database()` does not close
  them.
- Do not assume SQL placeholder syntax across Vertica, SQL Server, SQLite, and
  other DB-API drivers.
- Treat raw SQL expressions and `where` clauses as trusted application input.
- Avoid collecting large relations during intermediate steps.
- Use `python3 scripts/generate_api_docs.py --check` after changing public API
  docstrings or examples.
