"""
Additional cleaning operations for DuckJanitor - Phase 2 Complete.

This module adds the final SQL-based transformations including:
- Conditional joins
- Duplicate detection
- Statistic-based imputation
- Text processing
- PIVOT operations
- Hybrid layer (materialize → Python → re-wrap)
"""

import duckdb
import pandas as pd
from typing import Optional, Union, List, Any, Dict, Callable

from .duck_janitor import DuckJanitor
from .cleaning_ops import (
    _quote_id,
    _sql_literal,
    _register_relation,
    _ensure_columns_exist,
    _validate_sql_fragment,
)


# ========== Hybrid Layer (Materialize → Python → Re-wrap) ==========

def drop_duplicate_columns(relation: duckdb.DuckDBPyRelation,
                           conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Remove columns that are exact duplicates of other columns.
    """
    table_name = _register_relation(conn, relation)
    cols = relation.columns
    duplicate_cols = set()

    for i in range(len(cols)):
        col1 = cols[i]
        if col1 in duplicate_cols:
            continue
        for j in range(i + 1, len(cols)):
            col2 = cols[j]
            if col2 in duplicate_cols:
                continue
            # Compare columns for differences
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {_quote_id(col1)} IS DISTINCT FROM {_quote_id(col2)} LIMIT 1"
            diff_count = conn.execute(query).fetchone()[0]
            if diff_count == 0:
                duplicate_cols.add(col2)

    keep_cols = [c for c in cols if c not in duplicate_cols]
    if not keep_cols:
        raise ValueError("Cannot remove all columns")

    select_parts = [_quote_id(c) for c in keep_cols]
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    return conn.query(query)


def compare_df_cols(dj1: 'DuckJanitor', dj2: 'DuckJanitor',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    """
    Compare columns between two DuckJanitor instances.
    """
    cols1 = [(col, str(dtype)) for col, dtype in zip(dj1._relation.columns, dj1._relation.dtypes)]
    cols2 = [(col, str(dtype)) for col, dtype in zip(dj2._relation.columns, dj2._relation.dtypes)]

    set1 = set(cols1)
    set2 = set(cols2)

    comparison = {
        'only_in_dj1': list(set1 - set2),
        'only_in_dj2': list(set2 - set1),
        'in_both_same': list(set1 & set2),
    }

    return pd.DataFrame([comparison])


def join_apply(self: 'DuckJanitor', other: 'DuckJanitor', on: Union[str, List[str]],
               func: Callable, new_column_name: str,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Perform join then apply Python function to each row.
    """
    if isinstance(on, str):
        on = [on]
    if not on:
        raise ValueError("on must contain at least one join column")
    _ensure_columns_exist(self._relation.columns, on)
    _ensure_columns_exist(other._relation.columns, on)
    if not callable(func):
        raise ValueError("func must be callable")
    if not isinstance(new_column_name, str) or not new_column_name.strip():
        raise ValueError("new_column_name must be a non-empty string")

    temp_self = f"_self_{id(self._relation)}"
    temp_other = f"_other_{id(other._relation)}"
    self._connection.register(temp_self, self._relation)
    try:
        self._connection.register(temp_other, other._relation)
    except Exception:
        self._connection.register(temp_other, other._relation.df())

    join_conditions = ' AND '.join(f'self.{_quote_id(col)} = other.{_quote_id(col)}' for col in on)

    join_query = f"""
        SELECT * FROM {temp_self} self
        INNER JOIN {temp_other} other
        ON {join_conditions}
    """

    joined = self._connection.execute(join_query)

    df = joined.df()
    df[new_column_name] = df.apply(func, axis=1)

    return DuckJanitor.from_pandas(df)


def process_text(self: 'DuckJanitor', column: str, func: Union[Callable, str],
                 new_column_name: str,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Apply text processing function to a column.
    """
    _ensure_columns_exist(self._relation.columns, [column])
    if not isinstance(new_column_name, str) or not new_column_name.strip():
        raise ValueError("new_column_name must be a non-empty string")
    if isinstance(func, str):
        _validate_sql_fragment(func, context="Text-processing SQL expression")
        return self.add_column(new_column_name, func)
    elif callable(func):
        df = self.collect()
        df[new_column_name] = df[column].apply(func)
        return DuckJanitor.from_pandas(df)
    else:
        raise ValueError("func must be a string (SQL) or callable")


# ========== Final Phase 2 SQL Functions ==========

_VALID_CONDITIONAL_OPS = frozenset({'<', '<=', '>', '>=', '=', '==', '!=', '<>'})


def conditional_join(relation: duckdb.DuckDBPyRelation, other_relation: duckdb.DuckDBPyRelation,
                     on: List[tuple], how: str = 'inner',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Perform conditional (non-equi) joins between two relations using a single shared connection.
    If the relations belong to different connections, the right relation is materialized and
    re-registered on the provided connection.
    """
    if conn is None:
        raise ValueError(
            "A DuckDB connection is required for conditional_join. "
            "Pass the connection that owns the left relation."
        )

    valid_join_types = {"inner", "left", "right", "full", "cross"}
    if how.lower() not in valid_join_types:
        raise ValueError(f"how must be one of {sorted(valid_join_types)}")
    if not on and how.lower() != "cross":
        raise ValueError("on must contain at least one (left_col, right_col, op) tuple.")

    conditions = []
    for left_col, right_col, op in on:
        _ensure_columns_exist(relation.columns, [left_col])
        _ensure_columns_exist(other_relation.columns, [right_col])
        if op not in _VALID_CONDITIONAL_OPS:
            raise ValueError(f"Invalid operator: {op!r}. Use one of {_VALID_CONDITIONAL_OPS}")
        conditions.append(
            f'self.{_quote_id(left_col)} {op} other.{_quote_id(right_col)}'
        )

    where_clause = ' AND '.join(conditions) if conditions else "TRUE"

    temp_self = f"_self_{id(relation)}"
    temp_other = f"_other_{id(other_relation)}"
    conn.register(temp_self, relation)
    try:
        conn.register(temp_other, other_relation)
    except Exception:
        # Relations come from different connections: materialize the right side.
        conn.register(temp_other, other_relation.df())

    query = f"""
        SELECT * FROM {temp_self} self
        {how.upper()} JOIN {temp_other} other
        ON {where_clause}
    """

    return conn.query(query)


def get_dupes(relation: duckdb.DuckDBPyRelation,
              columns: Optional[Union[str, List[str]]] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Return duplicate rows.
    """
    table_name = _register_relation(conn, relation)

    if columns is None:
        columns = relation.columns
    elif isinstance(columns, str):
        columns = [columns]
    if not columns:
        raise ValueError("columns must contain at least one column")
    _ensure_columns_exist(relation.columns, columns)

    partition_cols = ', '.join(_quote_id(c) for c in columns)

    query = f"""
        SELECT * EXCLUDE (_dup_count) FROM (
            SELECT *,
                   COUNT(*) OVER (PARTITION BY {partition_cols}) AS _dup_count
            FROM {table_name}
        ) WHERE _dup_count > 1
    """

    return conn.query(query)


def dropnotnull(relation: duckdb.DuckDBPyRelation,
                subset: Optional[Union[str, List[str]]] = None,
                how: str = 'any',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Remove rows where values are NOT null (keep nulls).
    Inverse of dropna().
    """
    table_name = _register_relation(conn, relation)

    if subset is None:
        subset = relation.columns
    elif isinstance(subset, str):
        subset = [subset]
    if not subset:
        raise ValueError("subset must contain at least one column")
    _ensure_columns_exist(relation.columns, subset)

    if how == 'any':
        conditions = [f'{_quote_id(col)} IS NULL' for col in subset]
        where_clause = ' OR '.join(conditions)
    elif how == 'all':
        conditions = [f'{_quote_id(col)} IS NULL' for col in subset]
        where_clause = ' AND '.join(conditions)
    else:
        raise ValueError("how must be 'any' or 'all'")

    return conn.query(f"SELECT * FROM {table_name} WHERE {where_clause}")


def expand_column(relation: duckdb.DuckDBPyRelation, column: str,
                  sep: str = '|', prefix: Optional[str] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Expand a delimited column into dummy variables.
    """
    table_name = _register_relation(conn, relation)
    _ensure_columns_exist(relation.columns, [column])

    if prefix is None:
        prefix = column

    col = _quote_id(column)
    sep_lit = _sql_literal(sep)

    query = f"""
        SELECT DISTINCT UNNEST(str_split({col}, {sep_lit})) AS value
        FROM {table_name}
        WHERE {col} IS NOT NULL
    """

    unique_vals = [row[0] for row in conn.execute(query).fetchall()]

    dummy_exprs = []
    for val in unique_vals:
        dummy_name = f"{prefix}_{val}".replace(' ', '_').replace('-', '_')
        dummy_expr = (
            f"CASE WHEN list_contains(str_split({col}, {sep_lit}), {_sql_literal(val)}) "
            f"THEN 1 ELSE 0 END AS {_quote_id(dummy_name)}"
        )
        dummy_exprs.append(dummy_expr)

    old_columns = relation.columns
    select_parts = [_quote_id(c) for c in old_columns if c != column] + dummy_exprs

    return conn.query(f"SELECT {', '.join(select_parts)} FROM {table_name}")


def impute(relation: duckdb.DuckDBPyRelation, column: str,
           value: Optional[Any] = None, statistic: str = 'mean',
           group_by: Optional[Union[str, List[str]]] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Impute missing values using a specified value or statistic.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    _ensure_columns_exist(old_columns, [column])

    col = _quote_id(column)

    if value is not None:
        fill_expr = f"COALESCE({col}, {_sql_literal(value)}) AS {col}"
    else:
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            _ensure_columns_exist(old_columns, group_by)
            partition = f"PARTITION BY {', '.join(_quote_id(g) for g in group_by)}"
        else:
            partition = ""

        if statistic == 'mean':
            fill_expr = f"COALESCE({col}, AVG({col}) OVER ({partition})) AS {col}"
        elif statistic == 'median':
            fill_expr = f"COALESCE({col}, MEDIAN({col}) OVER ({partition})) AS {col}"
        elif statistic == 'mode':
            fill_expr = f"COALESCE({col}, MODE({col}) OVER ({partition})) AS {col}"
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    select_parts = []
    for c in old_columns:
        if c == column:
            select_parts.append(fill_expr)
        else:
            select_parts.append(_quote_id(c))

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def jitter(relation: duckdb.DuckDBPyRelation, column: str,
           target_column: str, scale: float = 0.01,
           seed: Optional[int] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Add random noise (jitter) to a numeric column.
    """
    table_name = _register_relation(conn, relation)
    _ensure_columns_exist(relation.columns, [column])
    if scale < 0:
        raise ValueError("scale must be >= 0")

    if seed is not None:
        normalized = (seed % 1000) / 1000.0 if seed != 0 else 0.0
        conn.execute(f"SELECT setseed({normalized})")

    col = _quote_id(column)
    tgt = _quote_id(target_column)

    jitter_expr = (
        f"{col} + ((random() - 0.5) * 2 * {scale} * "
        f"(MAX({col}) OVER () - MIN({col}) OVER ())) AS {tgt}"
    )

    return conn.query(f"SELECT *, {jitter_expr} FROM {table_name}")


def label_encode(relation: duckdb.DuckDBPyRelation, columns: Union[str, List[str]],
                 suffix: str = '_encoded',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Encode categorical columns with numerical labels.
    """
    table_name = _register_relation(conn, relation)

    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        raise ValueError("columns must contain at least one column")
    _ensure_columns_exist(relation.columns, columns)

    select_parts = [_quote_id(c) for c in relation.columns]

    for col in columns:
        encoded_col = f"{col}{suffix}"
        encode_expr = (
            f"DENSE_RANK() OVER (ORDER BY {_quote_id(col)}) - 1 AS {_quote_id(encoded_col)}"
        )
        select_parts.append(encode_expr)

    return conn.query(f"SELECT {', '.join(select_parts)} FROM {table_name}")


def find_replace(relation: duckdb.DuckDBPyRelation, column: str,
                 value_pairs: Dict[Any, Any],
                 target_column: Optional[str] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Find and replace values in a column.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    _ensure_columns_exist(old_columns, [column])
    if not value_pairs:
        raise ValueError("value_pairs cannot be empty")

    if target_column is None:
        target_column = column

    col = _quote_id(column)
    tgt = _quote_id(target_column)

    case_parts = [col]
    for old_val, new_val in value_pairs.items():
        case_parts.append(f"WHEN {_sql_literal(old_val)} THEN {_sql_literal(new_val)}")

    case_expr = f"CASE {' '.join(case_parts)} END AS {tgt}"

    select_parts = [_quote_id(c) for c in old_columns if c != column]
    select_parts.append(case_expr)

    return conn.query(f"SELECT {', '.join(select_parts)} FROM {table_name}")


def count_cumulative_unique(relation: duckdb.DuckDBPyRelation, column: str,
                            dest_column: str = 'cumulative_unique',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Return a column with the cumulative count of unique values.
    """
    table_name = _register_relation(conn, relation)
    _ensure_columns_exist(relation.columns, [column])
    col = _quote_id(column)
    dest = _quote_id(dest_column)

    query = f"""
        SELECT *,
               ROW_NUMBER() OVER (ORDER BY {col}) -
               ROW_NUMBER() OVER (PARTITION BY {col} ORDER BY {col}) + 1
               AS {dest}
        FROM {table_name}
    """

    return conn.query(query)


def complete(relation: duckdb.DuckDBPyRelation, columns: Union[str, List[str]],
             fill_value: Any = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Expand relation to include all possible combinations of specified columns.
    """
    table_name = _register_relation(conn, relation)

    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        raise ValueError("columns must contain at least one column")
    _ensure_columns_exist(relation.columns, columns)

    # Build the Cartesian product of unique values for each specified column
    grid_parts = []
    for col in columns:
        col_quoted = _quote_id(col)
        grid_parts.append(f"(SELECT DISTINCT {col_quoted} FROM {table_name} WHERE {col_quoted} IS NOT NULL)")

    grid_query = " CROSS JOIN ".join(grid_parts)

    join_conditions = " AND ".join(f"grid.{_quote_id(col)} = orig.{_quote_id(col)}" for col in columns)

    other_cols = [c for c in relation.columns if c not in columns]
    if other_cols:
        if fill_value is not None:
            other_selects = ", ".join(f"COALESCE(orig.{_quote_id(c)}, {_sql_literal(fill_value)}) AS {_quote_id(c)}" for c in other_cols)
        else:
            other_selects = ", ".join(f"orig.{_quote_id(c)} AS {_quote_id(c)}" for c in other_cols)
        select_list = ", ".join(f"grid.{_quote_id(col)}" for col in columns) + ", " + other_selects
    else:
        select_list = ", ".join(f"grid.{_quote_id(col)}" for col in columns)

    query = f"""
        SELECT {select_list}
        FROM ({grid_query}) grid
        LEFT JOIN {table_name} orig
        ON {join_conditions}
    """

    return conn.query(query)


# ========== DuckJanitor Method Wrappers (Hybrid Layer) ==========

def also(self: 'DuckJanitor', func: Callable,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Apply a Python function with side effects.
    """
    df = self.collect()
    result = func(df)

    if isinstance(result, pd.DataFrame):
        return DuckJanitor.from_pandas(result)
    return DuckJanitor.from_pandas(df)


def alias(relation: duckdb.DuckDBPyRelation, alias: Union[str, Callable],
          conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Rename all columns using a string or callable.
    """
    df_cols = relation.columns

    if isinstance(alias, str):
        if not alias.strip():
            raise ValueError("alias must be a non-empty string")
        # Rename all columns to the same base name with unique suffixes if there's more than one
        new_columns = []
        for i, col in enumerate(df_cols):
            if len(df_cols) == 1:
                new_columns.append(alias)
            else:
                new_columns.append(f"{alias}_{i}")
    elif callable(alias):
        new_columns = [alias(col) for col in df_cols]
        if not all(isinstance(name, str) and name.strip() for name in new_columns):
            raise ValueError("Callable alias must return non-empty string names for all columns.")
    else:
        raise ValueError("alias must be a string or callable")

    # Rename them using SELECT AS
    select_parts = [f'{_quote_id(old)} AS {_quote_id(new)}' for old, new in zip(df_cols, new_columns)]
    table_name = _register_relation(conn, relation)
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    return conn.query(query)


def mutate(self: 'DuckJanitor',
                 conn: Optional[duckdb.DuckDBPyConnection] = None, **kwargs) -> 'DuckJanitor':
    """
    Create or modify columns using a dictionary.
    """
    result = self
    for col_name, value in kwargs.items():
        result = result.add_column(col_name, value)
    return result
