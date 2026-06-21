"""
Additional cleaning operations for DuckJanitor - Phase 1 High Priority Functions.

This module extends cleaning_ops.py with more SQL-based transformations.
"""

import duckdb
from typing import Optional, Union, List, Any, Dict
import re

from .cleaning_ops import _quote_id, _sql_literal, _register_relation


def bin_numeric(relation: duckdb.DuckDBPyRelation, column: str,
                target_column: str, bins: Union[int, List[float]] = 5,
                strategy: str = 'quantile', conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Bin a numeric column into discrete intervals.
    """
    table_name = _register_relation(conn, relation)
    col = _quote_id(column)
    tgt = _quote_id(target_column)

    if isinstance(bins, int):
        if strategy == 'quantile':
            bin_expr = f"NTILE({bins}) OVER (ORDER BY {col}) AS {tgt}"
        elif strategy == 'uniform':
            bin_expr = (
                f"WIDTH_BUCKET({col}, MIN({col}) OVER (), MAX({col}) OVER (), {bins}) AS {tgt}"
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        query = f"SELECT *, {bin_expr} FROM {table_name}"
    else:
        edges = sorted(bins)
        case_parts = []
        for i in range(len(edges) - 1):
            if i == 0:
                case_parts.append(
                    f"WHEN {col} <= {_sql_literal(edges[i + 1])} "
                    f"THEN {_sql_literal(f'({edges[i]}, {edges[i + 1]}]')}"
                )
            elif i == len(edges) - 2:
                case_parts.append(
                    f"WHEN {col} > {_sql_literal(edges[i])} "
                    f"THEN {_sql_literal(f'({edges[i]}, {edges[i + 1]}]')}"
                )
            else:
                case_parts.append(
                    f"WHEN {col} > {_sql_literal(edges[i])} AND {col} <= {_sql_literal(edges[i + 1])} "
                    f"THEN {_sql_literal(f'({edges[i]}, {edges[i + 1]}]')}"
                )

        case_expr = f"CASE {' '.join(case_parts)} END AS {tgt}"
        query = f"SELECT *, {case_expr} FROM {table_name}"

    return conn.query(query)


def change_type(relation: duckdb.DuckDBPyRelation, column: str,
                dtype: str,
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Change the data type of a column.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    col = _quote_id(column)
    select_parts = []
    for c in old_columns:
        if c == column:
            select_parts.append(f'CAST({col} AS {dtype.upper()}) AS {col}')
        else:
            select_parts.append(_quote_id(c))

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def concatenate_columns(relation: duckdb.DuckDBPyRelation, columns: List[str],
                        sep: str = '_', target_column: str = 'concatenated',
                        conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Concatenate multiple columns into a single column.
    """
    table_name = _register_relation(conn, relation)

    concat_parts = []
    for i, col in enumerate(columns):
        if i > 0:
            concat_parts.append(_sql_literal(sep))
        concat_parts.append(f'COALESCE(CAST({_quote_id(col)} AS VARCHAR), \'\')')

    concat_expr = ' || '.join(concat_parts) + f' AS {_quote_id(target_column)}'

    query = f"SELECT *, {concat_expr} FROM {table_name}"

    return conn.query(query)


def deconcatenate_column(relation: duckdb.DuckDBPyRelation, column: str,
                         sep: str, target_columns: List[str],
                         conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Split a column into multiple columns based on a delimiter.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    col = _quote_id(column)
    select_parts = [_quote_id(c) for c in old_columns if c != column]

    for i, target in enumerate(target_columns):
        select_parts.append(
            f"str_split({col}, {_sql_literal(sep)})[{i + 1}] AS {_quote_id(target)}"
        )

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def drop_constant_columns(relation: duckdb.DuckDBPyRelation,
                          conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Remove columns that have only one unique value.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    non_constant_cols = []
    for col in old_columns:
        query = f"SELECT COUNT(DISTINCT {_quote_id(col)}) FROM {table_name}"
        unique_count = conn.execute(query).fetchone()[0]
        if unique_count > 1:
            non_constant_cols.append(col)

    if not non_constant_cols:
        raise ValueError("All columns are constant")

    select_parts = [_quote_id(col) for col in non_constant_cols]
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def fill(relation: duckdb.DuckDBPyRelation, column: str,
         value: Optional[Any] = None, direction: str = 'forward',
         group_by: Optional[Union[str, List[str]]] = None,
         conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Fill missing values in a column.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    col = _quote_id(column)

    if direction == 'value':
        if value is None:
            raise ValueError("value must be provided for direction='value'")
        fill_expr = f"COALESCE({col}, {_sql_literal(value)}) AS {col}"
    elif direction in ['forward', 'backward']:
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            partition = f"PARTITION BY {', '.join(_quote_id(g) for g in group_by)}"
        else:
            partition = ""

        if direction == 'forward':
            fill_expr = (
                f"COALESCE({col}, "
                f"LAST_VALUE({col} IGNORE NULLS) OVER ("
                f"{partition} ORDER BY ROW_NUMBER() OVER () "
                f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS {col}"
            )
        else:
            fill_expr = (
                f"COALESCE({col}, "
                f"FIRST_VALUE({col} IGNORE NULLS) OVER ("
                f"{partition} ORDER BY ROW_NUMBER() OVER () "
                f"ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)) AS {col}"
            )
    else:
        raise ValueError(f"Unknown direction: {direction}")

    other_cols = [_quote_id(c) for c in old_columns if c != column]
    query = f"SELECT {', '.join(other_cols)}, {fill_expr} FROM {table_name}"

    return conn.query(query)


def fill_empty(relation: duckdb.DuckDBPyRelation, column: str,
               value: str = '',
               conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Fill empty strings in a column with a specified value.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    col = _quote_id(column)
    select_parts = []
    for c in old_columns:
        if c == column:
            select_parts.append(
                f"COALESCE(NULLIF({col}, ''), {_sql_literal(value)}) AS {col}"
            )
        else:
            select_parts.append(_quote_id(c))

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def flag_nulls(relation: duckdb.DuckDBPyRelation, columns: Optional[Union[str, List[str]]] = None,
               prefix: str = 'is_null_', present_value: Any = 1,
               absent_value: Any = 0,
               conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Flag null values in specified columns with binary indicators.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if columns is None:
        columns = old_columns
    elif isinstance(columns, str):
        columns = [columns]

    select_parts = [_quote_id(c) for c in old_columns]

    for col in columns:
        flag_name = f"{prefix}{col}"
        flag_expr = (
            f"CASE WHEN {_quote_id(col)} IS NULL THEN {_sql_literal(present_value)} "
            f"ELSE {_sql_literal(absent_value)} END AS {_quote_id(flag_name)}"
        )
        select_parts.append(flag_expr)

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def limit_column_characters(relation: duckdb.DuckDBPyRelation, column: str,
                            max_chars: int, suffix: str = '...',
                            conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Limit the number of characters in a string column.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    col = _quote_id(column)
    safe_suffix = _sql_literal(suffix)
    substr_len = max(max_chars - len(suffix), 0)
    select_parts = []
    for c in old_columns:
        if c == column:
            select_parts.append(
                f"CASE WHEN LENGTH({col}) > {max_chars} "
                f"THEN substr({col}, 1, {substr_len}) || {safe_suffix} "
                f"ELSE {col} END AS {col}"
            )
        else:
            select_parts.append(_quote_id(c))

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def min_max_scale(relation: duckdb.DuckDBPyRelation, column: str,
                  target_column: str, min_val: float = 0,
                  max_val: float = 1,
                  conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Apply Min-Max scaling to a numeric column.
    """
    table_name = _register_relation(conn, relation)
    col = _quote_id(column)
    tgt = _quote_id(target_column)

    scale_expr = (
        f"CASE "
        f"WHEN MAX({col}) OVER () - MIN({col}) OVER () = 0 THEN {_sql_literal(min_val)} "
        f"ELSE ({col} - MIN({col}) OVER ()) * ({max_val - min_val}) / "
        f"(MAX({col}) OVER () - MIN({col}) OVER ()) + {min_val} END AS {tgt}"
    )

    query = f"SELECT *, {scale_expr} FROM {table_name}"

    return conn.query(query)


def groupby_agg(relation: duckdb.DuckDBPyRelation, by: Union[str, List[str]],
                aggregations: Dict[str, Union[str, Dict]],
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Perform groupby aggregation.
    """
    table_name = _register_relation(conn, relation)

    if isinstance(by, str):
        by = [by]

    agg_parts = []
    for col, agg in aggregations.items():
        if isinstance(agg, str):
            agg_parts.append(f"{agg.upper()}({_quote_id(col)}) AS {_quote_id(f'{col}_{agg}')}")
        elif isinstance(agg, dict):
            for new_name, func in agg.items():
                agg_parts.append(f"{func.upper()}({_quote_id(col)}) AS {_quote_id(new_name)}")

    group_cols = ', '.join(_quote_id(g) for g in by)
    query = f"SELECT {group_cols}, {', '.join(agg_parts)} FROM {table_name} GROUP BY {group_cols}"

    return conn.query(query)


def groupby_topk(relation: duckdb.DuckDBPyRelation, by: Union[str, List[str]],
                 column: str, k: int, ascending: bool = False,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Get top k rows within each group based on a column.
    """
    table_name = _register_relation(conn, relation)

    if isinstance(by, str):
        by = [by]

    order = 'ASC' if ascending else 'DESC'
    partition = ', '.join(_quote_id(g) for g in by)

    rank_expr = (
        f"ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY {_quote_id(column)} {order}) "
        f"AS _row_num"
    )

    subquery = f"SELECT *, {rank_expr} FROM {table_name}"
    query = f"SELECT * EXCLUDE (_row_num) FROM ({subquery}) WHERE _row_num <= {k}"

    return conn.query(query)


def case_when(relation: duckdb.DuckDBPyRelation, conditions: List[tuple],
              target_column: str, default: Optional[Any] = None,
              conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Create a column based on multiple conditions (SQL CASE WHEN).
    """
    table_name = _register_relation(conn, relation)

    case_parts = []
    for condition, value in conditions:
        if isinstance(condition, str):
            case_parts.append(f"WHEN {condition} THEN {_sql_literal(value)}")
        elif callable(condition):
            raise ValueError("Callable conditions require materialization. Use SQL strings instead.")

    if default is not None:
        case_parts.append(f"ELSE {_sql_literal(default)}")

    case_expr = f"CASE {' '.join(case_parts)} END AS {_quote_id(target_column)}"

    query = f"SELECT *, {case_expr} FROM {table_name}"

    return conn.query(query)


def currency_column_to_numeric(relation: duckdb.DuckDBPyRelation, column: str,
                               target_column: Optional[str] = None,
                               conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Convert a currency column to numeric by removing currency symbols and commas.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    if target_column is None:
        target_column = column

    col = _quote_id(column)
    tgt = _quote_id(target_column)

    clean_expr = (
        f"CAST(regexp_replace(CAST({col} AS VARCHAR), '[^0-9.-]', '', 'g') AS DOUBLE) AS {tgt}"
    )

    select_parts = [_quote_id(c) for c in old_columns if c != column]
    select_parts.append(clean_expr)

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def convert_date(relation: duckdb.DuckDBPyRelation, column: str,
                 target_column: Optional[str] = None,
                 date_format: Optional[str] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Convert a column to date type.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    if target_column is None:
        target_column = column

    col = _quote_id(column)
    tgt = _quote_id(target_column)

    if date_format:
        convert_expr = f"strptime({col}, {_sql_literal(date_format)}) AS {tgt}"
    else:
        convert_expr = f"CAST({col} AS DATE) AS {tgt}"

    select_parts = [_quote_id(c) for c in old_columns if c != column]
    select_parts.append(convert_expr)

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def pivot_wider(relation: duckdb.DuckDBPyRelation,
                id_cols: Union[str, List[str]],
                name_col: str,
                value_col: str,
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Pivot data from long to wide format.
    """
    table_name = _register_relation(conn, relation)

    if isinstance(id_cols, str):
        id_cols = [id_cols]

    name_query = f"SELECT DISTINCT {_quote_id(name_col)} FROM {table_name}"
    name_values = [row[0] for row in conn.execute(name_query).fetchall()]

    pivot_cols = []
    for val in name_values:
        pivot_cols.append(
            f"MAX(CASE WHEN {_quote_id(name_col)} = {_sql_literal(val)} "
            f"THEN {_quote_id(value_col)} END) AS {_quote_id(str(val))}"
        )

    id_cols_str = ', '.join(_quote_id(c) for c in id_cols)
    pivot_str = ', '.join(pivot_cols)

    query = f"SELECT {id_cols_str}, {pivot_str} FROM {table_name} GROUP BY {id_cols_str}"

    return conn.query(query)


def pivot_longer(relation: duckdb.DuckDBPyRelation,
                 cols: Union[str, List[str]],
                 names_to: str = 'variable', values_to: str = 'value',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Pivot data from wide to long format.
    """
    table_name = _register_relation(conn, relation)

    if isinstance(cols, str):
        cols = [cols]

    old_columns = relation.columns
    id_cols = [col for col in old_columns if col not in cols]

    parts = []
    for col in cols:
        escaped_col = col.replace("'", "''")
        if id_cols:
            id_select = ', '.join(_quote_id(c) for c in id_cols)
            parts.append(
                f"SELECT {id_select}, {_sql_literal(col)} AS {_quote_id(names_to)}, "
                f"{_quote_id(col)} AS {_quote_id(values_to)} FROM {table_name}"
            )
        else:
            parts.append(
                f"SELECT {_sql_literal(col)} AS {_quote_id(names_to)}, "
                f"{_quote_id(col)} AS {_quote_id(values_to)} FROM {table_name}"
            )

    query = ' UNION ALL '.join(parts)

    return conn.query(query)


def truncate_datetime(relation: duckdb.DuckDBPyRelation, column: str,
                      unit: str = 'day',
                      target_column: Optional[str] = None,
                      conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Truncate a datetime column to a specified unit.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if target_column is None:
        target_column = column

    unit_map = {
        'year': 'year',
        'month': 'month',
        'day': 'day',
        'hour': 'hour',
        'minute': 'minute',
        'second': 'second',
    }

    if unit not in unit_map:
        raise ValueError(f"Unknown unit: {unit}. Use: {', '.join(unit_map.keys())}")

    col = _quote_id(column)
    tgt = _quote_id(target_column)
    truncate_expr = f"CAST(DATE_TRUNC({_sql_literal(unit_map[unit])}, CAST({col} AS TIMESTAMP)) AS DATE)"

    if target_column == column:
        select_parts = [_quote_id(c) for c in old_columns if c != column]
        select_parts.append(f'{truncate_expr} AS {col}')
    else:
        select_parts = [_quote_id(c) for c in old_columns]
        select_parts.append(f'{truncate_expr} AS {tgt}')

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)
