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
import re

from .duck_janitor import DuckJanitor
from .cleaning_ops import _quote_id, _sql_literal, _register_relation


# ========== Hybrid Layer (Materialize → Python → Re-wrap) ==========

def drop_duplicate_columns(relation: duckdb.DuckDBPyRelation,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Remove columns that are exact duplicates of other columns.
    """
    df = relation.df()

    unique_cols = []
    seen_hashes = set()

    for col in df.columns:
        col_hash = hash(tuple(df[col].fillna('__NA__').astype(str)))
        if col_hash not in seen_hashes:
            seen_hashes.add(col_hash)
            unique_cols.append(col)

    return DuckJanitor.from_pandas(df[unique_cols])


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

    temp_self = f"_self_{id(self._relation)}"
    temp_other = f"_other_{id(other._relation)}"
    self._connection.register(temp_self, self._relation)
    self._connection.register(temp_other, other._relation)

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
    if isinstance(func, str):
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

    conditions = []
    for left_col, right_col, op in on:
        if op not in _VALID_CONDITIONAL_OPS:
            raise ValueError(f"Invalid operator: {op!r}. Use one of {_VALID_CONDITIONAL_OPS}")
        conditions.append(
            f'self.{_quote_id(left_col)} {op} other.{_quote_id(right_col)}'
        )

    where_clause = ' AND '.join(conditions)

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

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    col = _quote_id(column)

    if value is not None:
        fill_expr = f"COALESCE({col}, {_sql_literal(value)}) AS {col}"
    else:
        if statistic == 'mean':
            stat_subquery = f"SELECT AVG({col}) AS val FROM {table_name}"
        elif statistic == 'median':
            stat_subquery = (
                f"SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}) AS val "
                f"FROM {table_name}"
            )
        elif statistic == 'mode':
            stat_subquery = (
                f"SELECT {col} AS val FROM {table_name} "
                f"GROUP BY {col} ORDER BY COUNT(*) DESC LIMIT 1"
            )
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        fill_expr = f"COALESCE({col}, ({stat_subquery})) AS {col}"

    other_cols = [_quote_id(c) for c in old_columns if c != column]
    select_parts = other_cols + [fill_expr]
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

    if seed is not None:
        conn.execute(f"SET seed = {seed}")

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

    select_parts = [_quote_id(c) for c in relation.columns]

    for col in columns:
        if col in relation.columns:
            encoded_col = f"{col}{suffix}"
            encode_expr = (
                f"DENSE_RANK() OVER (ORDER BY {_quote_id(col)}) - 1 AS {_quote_id(encoded_col)}"
            )
            select_parts.append(encode_expr)

    return conn.query(f"SELECT {', '.join(select_parts)} FROM {table_name}")


def find_replace(relation: duckdb.DuckDBPyRelation, column: str,
                 value_pairs: Dict[str, str],
                 target_column: Optional[str] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Find and replace values in a column.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

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
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Expand DataFrame to include all possible combinations of key columns.
    """
    df = relation.df()

    if isinstance(columns, str):
        columns = [columns]

    import itertools

    unique_values = {}
    for col in columns:
        unique_values[col] = df[col].dropna().unique().tolist()

    combinations = list(itertools.product(*[unique_values[col] for col in columns]))

    if combinations:
        full_df = pd.DataFrame(combinations, columns=columns)
        result = full_df.merge(df, on=columns, how='left')
        if fill_value is not None:
            result = result.fillna(fill_value)
        return DuckJanitor.from_pandas(result)
    else:
        return DuckJanitor.from_pandas(df)


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


def alias(self: 'DuckJanitor', alias: Union[str, Callable],
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Rename all columns using a string or callable.
    """
    df = self.collect()

    if isinstance(alias, str):
        new_columns = [alias] * len(df.columns)
    elif callable(alias):
        new_columns = [alias(col) for col in df.columns]
    else:
        raise ValueError("alias must be a string or callable")

    df.columns = new_columns
    return DuckJanitor.from_pandas(df)


def mutate(self: 'DuckJanitor',
                 conn: Optional[duckdb.DuckDBPyConnection] = None, **kwargs) -> 'DuckJanitor':
    """
    Create or modify columns using a dictionary.
    """
    result = self
    for col_name, value in kwargs.items():
        result = result.add_column(col_name, value)
    return result
