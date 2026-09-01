"""
Additional cleaning operations for DuckJanitor - Phase 1 High Priority Functions.

This module extends cleaning_ops.py with more SQL-based transformations.
"""

from typing import Any, Optional, Union

import duckdb

from .cleaning_ops import (
    _ensure_columns_exist,
    _quote_id,
    _register_relation,
    _sql_literal,
    _validate_sql_fragment,
)


def bin_numeric(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    target_column: str,
    bins: Union[int, list[float]] = 5,
    strategy: str = "quantile",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Bin a numeric column into discrete intervals.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Numeric column to bin.
    target_column : str
        Name of the output bin column.
    bins : int or list of float, default 5
        Number of bins (int) or explicit edges (list).
    strategy : str, default 'quantile'
        'quantile' or 'uniform' binning when bins is int.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with bin column added.
    """
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("target_column must be a non-empty string")
    table_name = _register_relation(conn, relation)
    _ensure_columns_exist(relation.columns, [column])
    col = _quote_id(column)
    tgt = _quote_id(target_column)

    if isinstance(bins, int):
        if bins < 1:
            raise ValueError("bins must be >= 1 when provided as an integer")
        if strategy == "quantile":
            bin_expr = f"NTILE({bins}) OVER (ORDER BY {col}) AS {tgt}"
        elif strategy == "uniform":
            bin_expr = (
                f"CASE "
                f"WHEN {col} = MAX({col}) OVER () THEN {bins} "
                f"WHEN MAX({col}) OVER () - MIN({col}) OVER () = 0 THEN 1 "
                f"ELSE FLOOR(({col} - MIN({col}) OVER ()) / (MAX({col}) OVER () - MIN({col}) OVER ()) * {bins})::INT + 1 "
                f"END AS {tgt}"
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        query = f"SELECT *, {bin_expr} FROM {table_name}"
    else:
        edges = sorted(bins)
        if len(edges) < 2:
            raise ValueError("bins must contain at least two edge values")
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


def change_type(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    dtype: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Change the data type of a column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to cast.
    dtype : str
        Target DuckDB type (e.g. 'VARCHAR', 'BIGINT').
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with column cast to dtype.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    _ensure_columns_exist(old_columns, [column])
    _validate_sql_fragment(dtype, context="dtype")

    col = _quote_id(column)
    select_parts = []
    for c in old_columns:
        if c == column:
            select_parts.append(f"CAST({col} AS {dtype.upper()}) AS {col}")
        else:
            select_parts.append(_quote_id(c))

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def concatenate_columns(
    relation: duckdb.DuckDBPyRelation,
    columns: list[str],
    sep: str = "_",
    target_column: str = "concatenated",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Concatenate multiple columns into a single column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : list of str
        Columns to concatenate.
    sep : str, default '_'
        Separator inserted between values.
    target_column : str, default 'concatenated'
        Name of the output column.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with concatenated column added.
    """
    table_name = _register_relation(conn, relation)
    if not columns:
        raise ValueError("columns must contain at least one column")
    _ensure_columns_exist(relation.columns, columns)

    concat_parts = []
    for i, col in enumerate(columns):
        if i > 0:
            concat_parts.append(_sql_literal(sep))
        concat_parts.append(f"COALESCE(CAST({_quote_id(col)} AS VARCHAR), '')")

    concat_expr = " || ".join(concat_parts) + f" AS {_quote_id(target_column)}"

    query = f"SELECT *, {concat_expr} FROM {table_name}"

    return conn.query(query)


def deconcatenate_column(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    sep: str,
    target_columns: list[str],
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Split a column into multiple columns based on a delimiter.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to split.
    sep : str
        Delimiter used to split the column.
    target_columns : list of str
        Names of the output columns.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with split columns replacing the source column.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    _ensure_columns_exist(old_columns, [column])
    if not target_columns:
        raise ValueError("target_columns must contain at least one column")

    col = _quote_id(column)
    select_parts = [_quote_id(c) for c in old_columns if c != column]

    for i, target in enumerate(target_columns):
        select_parts.append(
            f"str_split({col}, {_sql_literal(sep)})[{i + 1}] AS {_quote_id(target)}"
        )

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def drop_constant_columns(
    relation: duckdb.DuckDBPyRelation, conn: Optional[duckdb.DuckDBPyConnection] = None
) -> duckdb.DuckDBPyRelation:
    """
    Remove columns that have only one unique value.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with constant columns removed.
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


def fill(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    value: Optional[Any] = None,
    direction: str = "forward",
    group_by: Optional[Union[str, list[str]]] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Fill missing values in a column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column whose nulls are filled.
    value : scalar, optional
        Literal value used when direction='value'.
    direction : str, default 'forward'
        'value', 'forward', or 'backward'.
    group_by : str or list of str, optional
        Grouping columns for forward/backward fill.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with nulls in column filled.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    _ensure_columns_exist(old_columns, [column])

    col = _quote_id(column)

    if direction == "value":
        if value is None:
            raise ValueError("value must be provided for direction='value'")
        fill_expr = f"COALESCE({col}, {_sql_literal(value)}) AS {col}"
    elif direction in ["forward", "backward"]:
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            partition = f"PARTITION BY {', '.join(_quote_id(g) for g in group_by)}"
        else:
            partition = ""

        # Use a CTE to generate row index for ordering to avoid parser error on nested window functions
        numbered_table = f"temp_numbered_{id(relation)}"

        if direction == "forward":
            fill_expr = (
                f"COALESCE({col}, "
                f"LAST_VALUE({col} IGNORE NULLS) OVER ("
                f"{partition} ORDER BY _row_idx "
                f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS {col}"
            )
        else:
            fill_expr = (
                f"COALESCE({col}, "
                f"FIRST_VALUE({col} IGNORE NULLS) OVER ("
                f"{partition} ORDER BY _row_idx "
                f"ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)) AS {col}"
            )

        with_clause = f"WITH {numbered_table} AS (SELECT *, ROW_NUMBER() OVER () AS _row_idx FROM {table_name})"
        other_cols = [_quote_id(c) for c in old_columns if c != column]
        select_list = ", ".join(other_cols) + f", {fill_expr}" if other_cols else fill_expr
        query = f"{with_clause} SELECT {select_list} FROM {numbered_table}"
        return conn.query(query)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    other_cols = [_quote_id(c) for c in old_columns if c != column]
    select_list = ", ".join(other_cols) + f", {fill_expr}" if other_cols else fill_expr
    query = f"SELECT {select_list} FROM {table_name}"

    return conn.query(query)


def fill_empty(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    value: str = "",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Fill empty strings in a column with a specified value.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        String column to fill empty values in.
    value : str, default ''
        Replacement value for empty strings.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with empty strings replaced.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")

    # Check data type of the column. Empty strings only apply to string types.
    col_idx = old_columns.index(column)
    col_type = str(relation.dtypes[col_idx]).upper()
    string_types = {"VARCHAR", "TEXT", "CHAR", "STRING", "BLOB"}
    if not any(st in col_type for st in string_types):
        return relation

    col = _quote_id(column)
    select_parts = []
    for c in old_columns:
        if c == column:
            select_parts.append(f"COALESCE(NULLIF({col}, ''), {_sql_literal(value)}) AS {col}")
        else:
            select_parts.append(_quote_id(c))

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def flag_nulls(
    relation: duckdb.DuckDBPyRelation,
    columns: Optional[Union[str, list[str]]] = None,
    prefix: str = "is_null_",
    present_value: Any = 1,
    absent_value: Any = 0,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Flag null values in specified columns with binary indicators.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str, optional
        Columns to flag; defaults to all columns.
    prefix : str, default 'is_null_'
        Prefix for generated flag column names.
    present_value : scalar, default 1
        Value when the source is null.
    absent_value : scalar, default 0
        Value when the source is not null.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with null-flag columns added.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if columns is None:
        columns = old_columns
    elif isinstance(columns, str):
        columns = [columns]
    _ensure_columns_exist(old_columns, columns)

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


def limit_column_characters(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    max_chars: int,
    suffix: str = "...",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Limit the number of characters in a string column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        String column to truncate.
    max_chars : int
        Maximum character length (including suffix).
    suffix : str, default '...'
        Appended to truncated values.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with column truncated to max_chars.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    if max_chars < 0:
        raise ValueError("max_chars must be >= 0")

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


def min_max_scale(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    target_column: str,
    min_val: float = 0,
    max_val: float = 1,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Apply Min-Max scaling to a numeric column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Numeric column to scale.
    target_column : str
        Name of the output column.
    min_val : float, default 0
        Target minimum.
    max_val : float, default 1
        Target maximum.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with scaled column added.
    """
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("target_column must be a non-empty string")
    table_name = _register_relation(conn, relation)
    _ensure_columns_exist(relation.columns, [column])
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


def groupby_agg(
    relation: duckdb.DuckDBPyRelation,
    by: Union[str, list[str]],
    aggregations: dict[str, Union[str, dict]],
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Perform groupby aggregation.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    by : str or list of str
        Grouping column(s).
    aggregations : dict
        Mapping of column name to aggregation SQL string, or to a dict of
        alias-to-function mappings.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Aggregated relation with one row per group.
    """
    table_name = _register_relation(conn, relation)

    if isinstance(by, str):
        by = [by]
    if not by:
        raise ValueError("by must contain at least one grouping column")
    _ensure_columns_exist(relation.columns, by)
    if not aggregations:
        raise ValueError("aggregations cannot be empty")

    agg_parts = []
    for col, agg in aggregations.items():
        _ensure_columns_exist(relation.columns, [col])
        if isinstance(agg, str):
            _validate_sql_fragment(agg, context=f"Aggregation function for column '{col}'")
            agg_parts.append(f"{agg.upper()}({_quote_id(col)}) AS {_quote_id(f'{col}_{agg}')}")
        elif isinstance(agg, dict):
            for new_name, func in agg.items():
                _validate_sql_fragment(func, context=f"Aggregation function for column '{col}'")
                agg_parts.append(f"{func.upper()}({_quote_id(col)}) AS {_quote_id(new_name)}")

    group_cols = ", ".join(_quote_id(g) for g in by)
    query = f"SELECT {group_cols}, {', '.join(agg_parts)} FROM {table_name} GROUP BY {group_cols}"

    return conn.query(query)


def groupby_topk(
    relation: duckdb.DuckDBPyRelation,
    by: Union[str, list[str]],
    column: str,
    k: int,
    ascending: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Get top k rows within each group based on a column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    by : str or list of str
        Grouping column(s).
    column : str
        Column used to rank rows within each group.
    k : int
        Number of rows to keep per group.
    ascending : bool, default False
        Sort direction within each group.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation containing the top k rows per group.
    """
    table_name = _register_relation(conn, relation)

    if isinstance(by, str):
        by = [by]
    if not by:
        raise ValueError("by must contain at least one grouping column")
    _ensure_columns_exist(relation.columns, by + [column])
    if k < 1:
        raise ValueError("k must be >= 1")

    order = "ASC" if ascending else "DESC"
    partition = ", ".join(_quote_id(g) for g in by)

    rank_expr = (
        f"ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY {_quote_id(column)} {order}) "
        f"AS _row_num"
    )

    subquery = f"SELECT *, {rank_expr} FROM {table_name}"
    query = f"SELECT * EXCLUDE (_row_num) FROM ({subquery}) WHERE _row_num <= {k}"

    return conn.query(query)


def case_when(
    relation: duckdb.DuckDBPyRelation,
    conditions: list[tuple],
    target_column: str,
    default: Optional[Any] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Create a column based on multiple conditions (SQL CASE WHEN).

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    conditions : list of tuple
        Pairs of (SQL_condition_string, value).
    target_column : str
        Name of the output column.
    default : scalar, optional
        Value used when no condition matches.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with the new conditional column added.
    """
    table_name = _register_relation(conn, relation)

    case_parts = []
    if not conditions:
        raise ValueError("conditions must contain at least one condition/value pair")
    for condition, value in conditions:
        if isinstance(condition, str):
            _validate_sql_fragment(condition, context="CASE WHEN condition")
            case_parts.append(f"WHEN {condition} THEN {_sql_literal(value)}")
        elif callable(condition):
            raise ValueError(
                "Callable conditions require materialization. Use SQL strings instead."
            )

    if default is not None:
        case_parts.append(f"ELSE {_sql_literal(default)}")

    case_expr = f"CASE {' '.join(case_parts)} END AS {_quote_id(target_column)}"

    query = f"SELECT *, {case_expr} FROM {table_name}"

    return conn.query(query)


def currency_column_to_numeric(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    target_column: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Convert a currency column to numeric by removing currency symbols and commas.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Currency string column to convert.
    target_column : str, optional
        Name of the output column; defaults to overwriting column.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with currency column converted to numeric.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    _ensure_columns_exist(old_columns, [column])

    if target_column is None:
        target_column = column

    col = _quote_id(column)
    tgt = _quote_id(target_column)

    clean_expr = f"TRY_CAST(NULLIF(regexp_replace(CAST({col} AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE) AS {tgt}"

    select_parts = [_quote_id(c) for c in old_columns if c != column]
    select_parts.append(clean_expr)

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def convert_date(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    target_column: Optional[str] = None,
    date_format: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Convert a column to date type.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to convert.
    target_column : str, optional
        Name of the output column; defaults to overwriting column.
    date_format : str, optional
        Source date format string passed to try_strptime.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with column converted to DATE.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns

    _ensure_columns_exist(old_columns, [column])

    if target_column is None:
        target_column = column

    col = _quote_id(column)
    tgt = _quote_id(target_column)

    if date_format:
        convert_expr = f"try_strptime({col}, {_sql_literal(date_format)}) AS {tgt}"
    else:
        convert_expr = f"TRY_CAST({col} AS DATE) AS {tgt}"

    select_parts = [_quote_id(c) for c in old_columns if c != column]
    select_parts.append(convert_expr)

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def pivot_wider(
    relation: duckdb.DuckDBPyRelation,
    id_cols: Union[str, list[str]],
    name_col: str,
    value_col: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Pivot data from long to wide format.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    id_cols : str or list of str
        Identifier column(s) that define rows in the output.
    name_col : str
        Column whose distinct values become new columns.
    value_col : str
        Column whose values populate the pivoted columns.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Wide-format relation with one row per id_cols combination.
    """
    if not isinstance(name_col, str) or not name_col.strip():
        raise ValueError("name_col must be a non-empty string")
    if not isinstance(value_col, str) or not value_col.strip():
        raise ValueError("value_col must be a non-empty string")
    table_name = _register_relation(conn, relation)

    if isinstance(id_cols, str):
        id_cols = [id_cols]
    _ensure_columns_exist(relation.columns, id_cols + [name_col, value_col])

    name_query = f"SELECT DISTINCT {_quote_id(name_col)} FROM {table_name}"
    name_values = [row[0] for row in conn.execute(name_query).fetchall() if row[0] is not None]
    if not name_values:
        raise ValueError("name_col contains no non-null values to pivot.")

    pivot_cols = []
    for val in name_values:
        pivot_cols.append(
            f"MAX(CASE WHEN {_quote_id(name_col)} = {_sql_literal(val)} "
            f"THEN {_quote_id(value_col)} END) AS {_quote_id(str(val))}"
        )

    id_cols_str = ", ".join(_quote_id(c) for c in id_cols)
    pivot_str = ", ".join(pivot_cols)

    query = f"SELECT {id_cols_str}, {pivot_str} FROM {table_name} GROUP BY {id_cols_str}"

    return conn.query(query)


def pivot_longer(
    relation: duckdb.DuckDBPyRelation,
    cols: Union[str, list[str]],
    names_to: str = "variable",
    values_to: str = "value",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Pivot data from wide to long format.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    cols : str or list of str
        Columns to unpivot into rows.
    names_to : str, default 'variable'
        Name of the output column holding original column names.
    values_to : str, default 'value'
        Name of the output column holding cell values.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Long-format relation with one row per (id, column) pair.
    """
    if not isinstance(names_to, str) or not names_to.strip():
        raise ValueError("names_to must be a non-empty string")
    if not isinstance(values_to, str) or not values_to.strip():
        raise ValueError("values_to must be a non-empty string")
    table_name = _register_relation(conn, relation)

    if isinstance(cols, str):
        cols = [cols]
    if not cols:
        raise ValueError("cols must contain at least one column")
    _ensure_columns_exist(relation.columns, cols)

    old_columns = relation.columns
    id_cols = [col for col in old_columns if col not in cols]

    parts = []
    for col in cols:
        if id_cols:
            id_select = ", ".join(_quote_id(c) for c in id_cols)
            parts.append(
                f"SELECT {id_select}, {_sql_literal(col)} AS {_quote_id(names_to)}, "
                f"{_quote_id(col)} AS {_quote_id(values_to)} FROM {table_name}"
            )
        else:
            parts.append(
                f"SELECT {_sql_literal(col)} AS {_quote_id(names_to)}, "
                f"{_quote_id(col)} AS {_quote_id(values_to)} FROM {table_name}"
            )

    query = " UNION ALL ".join(parts)

    return conn.query(query)


def truncate_datetime(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    unit: str = "day",
    target_column: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Truncate a datetime column to a specified unit.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Datetime column to truncate.
    unit : str, default 'day'
        One of 'year', 'month', 'day', 'hour', 'minute', 'second'.
    target_column : str, optional
        Name of the output column; defaults to overwriting column.
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with column truncated to the requested unit.
    """
    if target_column is not None and (
        not isinstance(target_column, str) or not target_column.strip()
    ):
        raise ValueError("target_column must be a non-empty string or None")
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns
    _ensure_columns_exist(old_columns, [column])

    if target_column is None:
        target_column = column

    unit_map = {
        "year": "year",
        "month": "month",
        "day": "day",
        "hour": "hour",
        "minute": "minute",
        "second": "second",
    }

    if unit not in unit_map:
        raise ValueError(f"Unknown unit: {unit}. Use: {', '.join(unit_map.keys())}")

    col = _quote_id(column)
    tgt = _quote_id(target_column)
    truncate_expr = (
        f"CAST(DATE_TRUNC({_sql_literal(unit_map[unit])}, CAST({col} AS TIMESTAMP)) AS DATE)"
    )

    if target_column == column:
        select_parts = [_quote_id(c) for c in old_columns if c != column]
        select_parts.append(f"{truncate_expr} AS {col}")
    else:
        select_parts = [_quote_id(c) for c in old_columns]
        select_parts.append(f"{truncate_expr} AS {tgt}")

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)
