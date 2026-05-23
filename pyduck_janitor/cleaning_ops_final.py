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


# ========== Hybrid Layer (Materialize → Python → Re-wrap) ==========

def drop_duplicate_columns(relation: duckdb.DuckDBPyRelation,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Remove columns that are exact duplicates of other columns.
    
    Note: Requires materialization to compare column contents.
    
    Returns
    -------
    DuckJanitor
        Instance with duplicate columns removed.
    """
    # Materialize to compare column contents
    df = relation.df()
    
    # Find duplicate columns by content hash
    unique_cols = []
    seen_hashes = set()
    
    for col in df.columns:
        # Create hash of column values (handle nulls)
        col_hash = hash(tuple(df[col].fillna('__NA__').astype(str)))
        if col_hash not in seen_hashes:
            seen_hashes.add(col_hash)
            unique_cols.append(col)
    
    # Keep only unique columns
    return DuckJanitor.from_pandas(df[unique_cols])


def compare_df_cols(dj1: 'DuckJanitor', dj2: 'DuckJanitor',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> pd.DataFrame:
    """
    Compare columns between two DuckJanitor instances.
    
    Parameters
    ----------
    dj1 : DuckJanitor
        First instance.
    dj2 : DuckJanitor
        Second instance.
        
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame.
    """
    # Get column info from both
    cols1 = [(col, str(dtype)) for col, dtype in dj1._relation.dtypes.items()]
    cols2 = [(col, str(dtype)) for col, dtype in dj2._relation.dtypes.items()]
    
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
    
    Note: Materializes the joined data to apply Python function.
    
    Parameters
    ----------
    self : DuckJanitor
        Left relation.
    other : DuckJanitor
        Right relation.
    on : str or list of str
        Join key(s).
    func : callable
        Function to apply to each row.
    new_column_name : str
        Name of the new column.
        
    Returns
    -------
    DuckJanitor
        Result with new column.
    """
    if isinstance(on, str):
        on = [on]
    
    # Register relations
    temp_self = f"_self_{id(self._relation)}"
    temp_other = f"_other_{id(other._relation)}"
    self._connection.register(temp_self, self._relation)
    self._connection.register(temp_other, other._relation)
    
    # Build join condition
    join_conditions = ' AND '.join(f'self."{col}" = other."{col}"' for col in on)
    
    join_query = f"""
        SELECT * FROM {temp_self} self
        INNER JOIN {temp_other} other
        ON {join_conditions}
    """
    
    joined = self._connection.execute(join_query)
    
    # Materialize and apply function
    df = joined.df()
    df[new_column_name] = df.apply(func, axis=1)
    
    return DuckJanitor.from_pandas(df)


def process_text(self: 'DuckJanitor', column: str, func: Union[Callable, str],
                 new_column_name: str,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Apply text processing function to a column.
    
    If func is a string, it's treated as a SQL expression.
    If func is callable, data is materialized.
    
    Parameters
    ----------
    self : DuckJanitor
        The instance.
    column : str
        Column to process.
    func : callable or str
        Function or SQL expression.
    new_column_name : str
        Name of new column.
        
    Returns
    -------
    DuckJanitor
        Result with processed column.
    """
    if isinstance(func, str):
        # SQL expression
        return self.add_column(new_column_name, func)
    elif callable(func):
        # Materialize and apply Python function
        df = self.collect()
        df[new_column_name] = df[column].apply(func)
        return DuckJanitor.from_pandas(df)
    else:
        raise ValueError("func must be a string (SQL) or callable")


# ========== Final Phase 2 SQL Functions ==========

def conditional_join(relation: duckdb.DuckDBPyRelation, other_relation: duckdb.DuckDBPyRelation,
                     on: List[tuple], how: str = 'inner',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Perform conditional (non-equi) joins between two relations.
    """
    conn = conn or duckdb.connect()
    
    conditions = []
    for left_col, right_col, op in on:
        conditions.append(f'self."{left_col}" {op} other."{right_col}"')
    
    where_clause = ' AND '.join(conditions)
    
    temp_self = f"_self_{id(relation)}"
    temp_other = f"_other_{id(other_relation)}"
    conn.register(temp_self, relation)
    conn.register(temp_other, other_relation)
    
    query = f"""
        SELECT * FROM {temp_self} self
        {how.upper()} JOIN {temp_other} other
        ON {where_clause}
    """
    
    return conn.execute(query)


def get_dupes(relation: duckdb.DuckDBPyRelation,
              columns: Optional[Union[str, List[str]]] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Return duplicate rows.
    """
    conn = conn or duckdb.connect()
    
    if columns is None:
        columns = relation.columns
    elif isinstance(columns, str):
        columns = [columns]
    
    partition_cols = ', '.join(f'"{col}"' for col in columns)
    
    query = f"""
        SELECT * FROM (
            SELECT *,
                   COUNT(*) OVER (PARTITION BY {partition_cols}) as _dup_count
            FROM relation
        ) WHERE _dup_count > 1
    """
    
    return conn.execute(query)


def dropnotnull(relation: duckdb.DuckDBPyRelation,
                subset: Optional[Union[str, List[str]]] = None,
                how: str = 'any',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Remove rows where values are NOT null (keep nulls).
    Inverse of dropna().
    """
    conn = conn or duckdb.connect()
    
    if subset is None:
        subset = relation.columns
    elif isinstance(subset, str):
        subset = [subset]
    
    if how == 'any':
        conditions = [f'"{col}" IS NULL' for col in subset]
        where_clause = ' OR '.join(conditions)
    elif how == 'all':
        conditions = [f'"{col}" IS NULL' for col in subset]
        where_clause = ' AND '.join(conditions)
    else:
        raise ValueError("how must be 'any' or 'all'")
    
    return conn.execute(f"SELECT * FROM relation WHERE {where_clause}")


def expand_column(relation: duckdb.DuckDBPyRelation, column: str,
                  sep: str = '|', prefix: Optional[str] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Expand a delimited column into dummy variables.
    """
    conn = conn or duckdb.connect()
    
    if prefix is None:
        prefix = column
    
    query = f"""
        SELECT DISTINCT UNNEST(str_split("{column}", '{sep}')) as value
        FROM relation
        WHERE "{column}" IS NOT NULL
    """
    
    unique_vals = [row[0] for row in conn.execute(query).fetchall()]
    
    dummy_exprs = []
    for val in unique_vals:
        dummy_name = f"{prefix}_{val}".replace(' ', '_').replace('-', '_')
        dummy_expr = f"""
            CASE WHEN list_contains(str_split("{column}", '{sep}'), '{val}')
            THEN 1 ELSE 0 END AS "{dummy_name}"
        """
        dummy_exprs.append(dummy_expr)
    
    old_columns = relation.columns
    select_parts = [f'"{col}"' for col in old_columns if col != column] + dummy_exprs
    
    return conn.execute(f"SELECT {', '.join(select_parts)} FROM relation")


def impute(relation: duckdb.DuckDBPyRelation, column: str,
           value: Optional[Any] = None, statistic: str = 'mean',
           group_by: Optional[Union[str, List[str]]] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Impute missing values using a specified value or statistic.
    """
    conn = conn or duckdb.connect()
    old_columns = relation.columns
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    if value is not None:
        fill_expr = f"COALESCE(\"{column}\", {value}) AS \"{column}\""
    else:
        if statistic == 'mean':
            stat_subquery = f"SELECT AVG(\"{column}\") as val FROM relation"
        elif statistic == 'median':
            stat_subquery = f"SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY \"{column}\") as val FROM relation"
        elif statistic == 'mode':
            stat_subquery = f"SELECT \"{column}\" as val FROM relation GROUP BY \"{column}\" ORDER BY COUNT(*) DESC LIMIT 1"
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        fill_expr = f"""
            COALESCE(\"{column}\", (
                {stat_subquery}
            )) AS \"{column}\"
        """
    
    other_cols = [f'"{col}"' for col in old_columns if col != column]
    return conn.execute(f"SELECT {', '.join(other_cols)}, {fill_expr.strip()} FROM relation")


def jitter(relation: duckdb.DuckDBPyRelation, column: str,
           target_column: str, scale: float = 0.01,
           seed: Optional[int] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Add random noise (jitter) to a numeric column.
    """
    conn = conn or duckdb.connect()
    
    if seed is not None:
        conn.execute(f"SET seed = {seed}")
    
    jitter_expr = f"""
        "{column}" + (
            (random() - 0.5) * 2 * {scale} *
            (MAX("{column}") OVER () - MIN("{column}") OVER ())
        )
        AS "{target_column}"
    """
    
    return conn.execute(f"SELECT *, {jitter_expr} FROM relation")


def label_encode(relation: duckdb.DuckDBPyRelation, columns: Union[str, List[str]],
                 suffix: str = '_encoded',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Encode categorical columns with numerical labels.
    """
    conn = conn or duckdb.connect()
    
    if isinstance(columns, str):
        columns = [columns]
    
    select_parts = [f'"{col}"' for col in relation.columns]
    
    for col in columns:
        if col in relation.columns:
            encoded_col = f"{col}{suffix}"
            encode_expr = f"DENSE_RANK() OVER (ORDER BY \"{col}\") - 1 AS \"{encoded_col}\""
            select_parts.append(encode_expr)
    
    return conn.execute(f"SELECT {', '.join(select_parts)} FROM relation")


def find_replace(relation: duckdb.DuckDBPyRelation, column: str,
                 value_pairs: Dict[str, str],
                 target_column: Optional[str] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Find and replace values in a column.
    """
    conn = conn or duckdb.connect()
    old_columns = relation.columns
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    if target_column is None:
        target_column = column
    
    case_parts = [f'"{column}"']
    for old_val, new_val in value_pairs.items():
        if isinstance(new_val, str):
            case_parts.append(f"WHEN '{old_val}' THEN '{new_val}'")
        else:
            case_parts.append(f"WHEN '{old_val}' THEN {new_val}")
    
    case_expr = f"CASE {' '.join(case_parts)} END AS \"{target_column}\""
    
    select_parts = [f'"{col}"' for col in old_columns if col != column]
    select_parts.append(case_expr)
    
    return conn.execute(f"SELECT {', '.join(select_parts)} FROM relation")


def count_cumulative_unique(relation: duckdb.DuckDBPyRelation, column: str,
                            dest_column: str = 'cumulative_unique',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Return a column with the cumulative count of unique values.
    """
    conn = conn or duckdb.connect()
    
    query = f"""
        SELECT *,
               ROW_NUMBER() OVER (ORDER BY \"{column}\") -
               ROW_NUMBER() OVER (PARTITION BY \"{column}\" ORDER BY \"{column}\") + 1
               AS \"{dest_column}\"
        FROM relation
    """
    
    return conn.execute(query)


def complete(relation: duckdb.DuckDBPyRelation, columns: Union[str, List[str]],
             fill_value: Any = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
    """
    Expand DataFrame to include all possible combinations of key columns.
    Uses hybrid approach for simplicity.
    """
    df = relation.df()
    
    if isinstance(columns, str):
        columns = [columns]
    
    import itertools
    
    # Get unique values for each column
    unique_values = {}
    for col in columns:
        unique_values[col] = df[col].dropna().unique().tolist()
    
    # Create all combinations
    combinations = list(itertools.product(*[unique_values[col] for col in columns]))
    
    if combinations:
        full_df = pd.DataFrame(combinations, columns=columns)
        
        # Merge with original data
        result = full_df.merge(df, on=columns, how='left')
        
        # Fill missing values
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
    
    Note: This breaks lazy evaluation and materializes the data.
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
    
    Note: Series operation - materializes to pandas, applies function.
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
    
    Note: Redundant with add_column() - implemented as convenience.
    """
    result = self
    for col_name, value in kwargs.items():
        if isinstance(value, str):
            result = result.add_column(col_name, value)
        else:
            result = result.add_column(col_name, value)
    return result
