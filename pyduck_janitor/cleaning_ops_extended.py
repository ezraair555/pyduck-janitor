"""
Additional cleaning operations for DuckJanitor - Phase 1 High Priority Functions.

This module extends cleaning_ops.py with more SQL-based transformations.
"""

import duckdb
from typing import Optional, Union, List, Any, Dict
import re


def bin_numeric(relation: duckdb.DuckDBPyRelation, column: str,
                target_column: str, bins: Union[int, List[float]] = 5,
                strategy: str = 'quantile', conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Bin a numeric column into discrete intervals.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to bin.
    target_column : str
        Name of the new binned column.
    bins : int or list of float
        Number of bins or bin edges.
    strategy : str
        Binning strategy ('quantile', 'uniform', or 'edges').
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with binned column added.
    """
    conn = conn or duckdb.connect()
    
    if isinstance(bins, int):
        if strategy == 'quantile':
            # Use NTILE for quantile binning
            bin_expr = f"NTILE({bins}) OVER (ORDER BY \"{column}\") AS \"{target_column}\""
        elif strategy == 'uniform':
            # Uniform binning using WIDTH_BUCKET
            bin_expr = f"WIDTH_BUCKET(\"{column}\", MIN(\"{column}\") OVER (), MAX(\"{column}\") OVER (), {bins}) AS \"{target_column}\""
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        query = f"SELECT *, {bin_expr} FROM relation"
    else:
        # Custom bin edges - use CASE WHEN
        edges = sorted(bins)
        case_parts = []
        for i in range(len(edges) - 1):
            if i == 0:
                case_parts.append(f"WHEN \"{column}\" <= {edges[i+1]} THEN '({edges[i]}, {edges[i+1]}]'")
            elif i == len(edges) - 2:
                case_parts.append(f"WHEN \"{column}\" > {edges[i]} THEN '({edges[i]}, {edges[i+1]}]'")
            else:
                case_parts.append(f"WHEN \"{column}\" > {edges[i]} AND \"{column}\" <= {edges[i+1]} THEN '({edges[i]}, {edges[i+1]}]'")
        
        case_expr = f"CASE {' '.join(case_parts)} END AS \"{target_column}\""
        query = f"SELECT *, {case_expr} FROM relation"
    
    return conn.execute(query)


def change_type(relation: duckdb.DuckDBPyRelation, column: str,
                dtype: str,
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Change the data type of a column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to change.
    dtype : str
        Target data type ('INTEGER', 'FLOAT', 'VARCHAR', 'BOOLEAN', 'DATE', etc.).
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with column type changed.
    """
    conn = conn or duckdb.connect()
    old_columns = relation.columns
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    select_parts = []
    for col in old_columns:
        if col == column:
            select_parts.append(f'CAST("{col}" AS {dtype.upper()}) AS "{col}"')
        else:
            select_parts.append(f'"{col}"')
    
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    
    return conn.execute(query)


def concatenate_columns(relation: duckdb.DuckDBPyRelation, columns: List[str],
                        sep: str = '_', target_column: str = 'concatenated',
                        conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Concatenate multiple columns into a single column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : list of str
        Columns to concatenate.
    sep : str
        Separator string.
    target_column : str
        Name of the new column.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with concatenated column added.
    """
    conn = conn or duckdb.connect()
    
    # Build concatenation expression using || operator
    concat_parts = []
    for i, col in enumerate(columns):
        if i > 0:
            concat_parts.append(f"'{sep}'")
        concat_parts.append(f'COALESCE(CAST("{col}" AS VARCHAR), \'\')')
    
    concat_expr = ' || '.join(concat_parts) + f' AS "{target_column}"'
    
    query = f"SELECT *, {concat_expr} FROM relation"
    
    return conn.execute(query)


def deconcatenate_column(relation: duckdb.DuckDBPyRelation, column: str,
                         sep: str, target_columns: List[str],
                         conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Split a column into multiple columns based on a delimiter.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to split.
    sep : str
        Separator string.
    target_columns : list of str
        Names for the new columns.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with split columns added.
    """
    conn = conn or duckdb.connect()
    old_columns = relation.columns
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    # DuckDB's str_split returns an array
    # Extract elements using array indexing
    select_parts = [f'"{col}"' for col in old_columns if col != column]
    
    for i, target in enumerate(target_columns):
        select_parts.append(f"str_split(\"{column}\", '{sep}')[{i}] AS \"{target}\"")
    
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    
    return conn.execute(query)


def drop_constant_columns(relation: duckdb.DuckDBPyRelation,
                          conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Remove columns that have only one unique value.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with constant columns removed.
    """
    conn = conn or duckdb.connect()
    old_columns = relation.columns
    
    # Find non-constant columns
    non_constant_cols = []
    for col in old_columns:
        query = f"SELECT COUNT(DISTINCT \"{col}\") FROM relation"
        unique_count = conn.execute(query).fetchone()[0]
        if unique_count > 1:
            non_constant_cols.append(col)
    
    if not non_constant_cols:
        raise ValueError("All columns are constant")
    
    select_parts = [f'"{col}"' for col in non_constant_cols]
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    
    return conn.execute(query)


def fill(relation: duckdb.DuckDBPyRelation, column: str,
         value: Optional[Any] = None, direction: str = 'forward',
         group_by: Optional[Union[str, List[str]]] = None,
         conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Fill missing values in a column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to fill.
    value : scalar, optional
        Fixed value to fill with. If None, uses forward/backward fill.
    direction : str
        Fill direction ('forward', 'backward', or 'value').
    group_by : str or list of str, optional
        Column(s) to group by when filling.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with filled column.
    """
    conn = conn or duckdb.connect()
    old_columns = relation.columns
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    if direction == 'value':
        if value is None:
            raise ValueError("value must be provided for direction='value'")
        
        if isinstance(value, str):
            value = f"'{value}'"
        
        fill_expr = f"COALESCE(\"{column}\", {value}) AS \"{column}\""
    elif direction in ['forward', 'backward']:
        # Use window functions for forward/backward fill
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            partition = f"PARTITION BY {', '.join(f'\"{g}\"' for g in group_by)}"
        else:
            partition = ""
        
        if direction == 'forward':
            # LAST_VALUE with IGNORE NULLS
            fill_expr = f"""
                COALESCE("{column}", 
                    LAST_VALUE("{column}" IGNORE NULLS) OVER (
                        {partition}
                        ORDER BY ROW()
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    )
                ) AS "{column}"
            """
        else:  # backward
            fill_expr = f"""
                COALESCE("{column}",
                    FIRST_VALUE("{column}" IGNORE NULLS) OVER (
                        {partition}
                        ORDER BY ROW()
                        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                    )
                ) AS "{column}"
            """
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    # Build SELECT with filled column
    other_cols = [f'"{col}"' for col in old_columns if col != column]
    query = f"SELECT {', '.join(other_cols)}, {fill_expr.strip()} FROM relation"
    
    return conn.execute(query)


def flag_nulls(relation: duckdb.DuckDBPyRelation, columns: Optional[Union[str, List[str]]] = None,
               prefix: str = 'is_null_', present_value: Any = 1,
               absent_value: Any = 0,
               conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Flag null values in specified columns with binary indicators.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str, optional
        Columns to flag. If None, flags all columns.
    prefix : str
        Prefix for flag column names.
    present_value : scalar
        Value when null is present.
    absent_value : scalar
        Value when no null.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with flag columns added.
    """
    conn = conn or duckdb.connect()
    old_columns = relation.columns
    
    if columns is None:
        columns = old_columns
    elif isinstance(columns, str):
        columns = [columns]
    
    # Keep all original columns
    select_parts = [f'"{col}"' for col in old_columns]
    
    # Add flag columns
    for col in columns:
        flag_name = f"{prefix}{col}"
        flag_expr = f"CASE WHEN \"{col}\" IS NULL THEN {present_value} ELSE {absent_value} END AS \"{flag_name}\""
        select_parts.append(flag_expr)
    
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    
    return conn.execute(query)


def limit_column_characters(relation: duckdb.DuckDBPyRelation, column: str,
                            max_chars: int, suffix: str = '...',
                            conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Limit the number of characters in a string column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to truncate.
    max_chars : int
        Maximum number of characters.
    suffix : str
        Suffix to add when truncated.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with truncated column.
    """
    conn = conn or duckdb.connect()
    old_columns = relation.columns
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    select_parts = []
    for col in old_columns:
        if col == column:
            # Use substr and concatenate with suffix if needed
            select_parts.append(f"""
                CASE 
                    WHEN LENGTH("{col}") > {max_chars} 
                    THEN substr("{col}", 1, {max_chars - len(suffix)}) || '{suffix}'
                    ELSE "{col}"
                END AS "{col}"
            """)
        else:
            select_parts.append(f'"{col}"')
    
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    
    return conn.execute(query)


def min_max_scale(relation: duckdb.DuckDBPyRelation, column: str,
                  target_column: str, min_val: float = 0,
                  max_val: float = 1,
                  conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Apply Min-Max scaling to a numeric column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to scale.
    target_column : str
        Name of the new scaled column.
    min_val : float
        Target minimum value.
    max_val : float
        Target maximum value.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with scaled column added.
    """
    conn = conn or duckdb.connect()
    
    # Min-max scaling formula: (x - min) / (max - min) * (target_max - target_min) + target_min
    scale_expr = f"""
        ("{column}" - MIN("{column}") OVER ()) * ({max_val - min_val}) / 
        NULLIF(MAX("{column}") OVER () - MIN("{column}") OVER (), 0) + {min_val}
        AS "{target_column}"
    """
    
    query = f"SELECT *, {scale_expr.strip()} FROM relation"
    
    return conn.execute(query)


def groupby_agg(relation: duckdb.DuckDBPyRelation, by: Union[str, List[str]],
                aggregations: Dict[str, Union[str, Dict]],
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Perform groupby aggregation.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    by : str or list of str
        Column(s) to group by.
    aggregations : dict
        Aggregations to apply. Format: {column: 'agg_func'} or {column: {'new_name': 'agg_func'}}.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with aggregated results.
    """
    conn = conn or duckdb.connect()
    
    if isinstance(by, str):
        by = [by]
    
    # Build aggregation expressions
    agg_parts = []
    for col, agg in aggregations.items():
        if isinstance(agg, str):
            # Simple aggregation: {column: 'sum'}
            agg_parts.append(f"{agg.upper()}(\"{col}\") AS \"{col}_{agg}\"")
        elif isinstance(agg, dict):
            # Named aggregation: {column: {'result': 'sum'}}
            for new_name, func in agg.items():
                agg_parts.append(f"{func.upper()}(\"{col}\") AS \"{new_name}\"")
    
    group_cols = ', '.join(f'"{g}"' for g in by)
    query = f"SELECT {group_cols}, {', '.join(agg_parts)} FROM relation GROUP BY {group_cols}"
    
    return conn.execute(query)


def groupby_topk(relation: duckdb.DuckDBPyRelation, by: Union[str, List[str]],
                 column: str, k: int, ascending: bool = False,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Get top k rows within each group based on a column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    by : str or list of str
        Column(s) to group by.
    column : str
        Column to rank by.
    k : int
        Number of top rows to return per group.
    ascending : bool
        Sort order for ranking.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with top k rows per group.
    """
    conn = conn or duckdb.connect()
    
    if isinstance(by, str):
        by = [by]
    
    # Use ROW_NUMBER() window function
    order = 'ASC' if ascending else 'DESC'
    partition = ', '.join(f'"{g}"' for g in by)
    
    rank_expr = f"ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY \"{column}\" {order}) AS _row_num"
    
    # Wrap in subquery to filter by rank
    subquery = f"SELECT *, {rank_expr} FROM relation"
    query = f"SELECT * EXCLUDE (_row_num) FROM ({subquery}) WHERE _row_num <= {k}"
    
    return conn.execute(query)


def case_when(relation: duckdb.DuckDBPyRelation, conditions: List[tuple],
              target_column: str, default: Optional[Any] = None,
              conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Create a column based on multiple conditions (SQL CASE WHEN).
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    conditions : list of tuples
        List of (condition, value) pairs. Condition should be SQL string.
    target_column : str
        Name of the new column.
    default : scalar, optional
        Default value if no conditions match.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with new column added.
    """
    conn = relation.database
    
    # Build CASE WHEN expression
    case_parts = []
    for condition, value in conditions:
        if isinstance(condition, str):
            # SQL condition string
            if isinstance(value, str):
                value = f"'{value}'"
            case_parts.append(f"WHEN {condition} THEN {value}")
        elif callable(condition):
            # Callable - need to materialize and evaluate
            raise ValueError("Callable conditions require materialization. Use SQL strings instead.")
    
    if default is not None:
        if isinstance(default, str):
            default = f"'{default}'"
        case_parts.append(f"ELSE {default}")
    
    case_expr = f"CASE {' '.join(case_parts)} END AS \"{target_column}\""
    
    query = f"SELECT *, {case_expr} FROM relation"
    
    return conn.execute(query)


def currency_column_to_numeric(relation: duckdb.DuckDBPyRelation, column: str,
                               target_column: Optional[str] = None,
                               conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Convert a currency column to numeric by removing currency symbols and commas.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Currency column to convert.
    target_column : str, optional
        Name of the new column. If None, replaces in place.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with numeric column.
    """
    conn = relation.database
    old_columns = relation.columns
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    if target_column is None:
        target_column = column
    
    # Remove currency symbols and commas, then cast to numeric
    clean_expr = f"""
        CAST(regexp_replace(CAST("{column}" AS VARCHAR), '[^0-9.-]', '', 'g') AS DOUBLE)
        AS "{target_column}"
    """
    
    # Keep all columns, replace/add the target
    select_parts = [f'"{col}"' for col in old_columns if col != column]
    select_parts.append(clean_expr)
    
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    
    return conn.execute(query)


def convert_date(relation: duckdb.DuckDBPyRelation, column: str,
                 target_column: Optional[str] = None,
                 date_format: Optional[str] = None,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Convert a column to date type.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to convert.
    target_column : str, optional
        Name of new column. If None, replaces in place.
    date_format : str, optional
        Date format string (e.g., '%Y-%m-%d'). If None, auto-detect.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with date column.
    """
    conn = relation.database
    old_columns = relation.columns
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    if target_column is None:
        target_column = column
    
    if date_format:
        # Use strptime for custom format
        convert_expr = f"strptime(\"{column}\", '{date_format}') AS \"{target_column}\""
    else:
        # Try automatic conversion
        convert_expr = f"CAST(\"{column}\" AS DATE) AS \"{target_column}\""
    
    # Keep all columns, replace/add the target
    select_parts = [f'"{col}"' for col in old_columns if col != column]
    select_parts.append(convert_expr)
    
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    
    return conn.execute(query)


def pivot_wider(relation: duckdb.DuckDBPyRelation,
                id_cols: Union[str, List[str]],
                name_col: str,
                value_col: str,
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Pivot data from long to wide format (Wider = spread keys across columns).
    
    This is equivalent to DuckDB's PIVOT operation with dynamic column names.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation (long format).
    id_cols : str or list of str
        Column(s) that identify unique rows in the wide format.
    name_col : str
        Column containing the new column names.
    value_col : str
        Column containing the values to populate the new columns.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation in wide format.
    """
    conn = conn or duckdb.connect()
    
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    
    # Get unique values from name_col to create pivot columns
    table_name = f'_temp_{id(relation)}'
    conn.register(table_name, relation)
    
    # First, get the distinct values for the name column
    name_query = f"SELECT DISTINCT \"{name_col}\" FROM {table_name}"
    name_values = [row[0] for row in conn.execute(name_query).fetchall()]
    
    # Build the pivot query
    pivot_cols = []
    for val in name_values:
        # Escape the value for SQL
        escaped_val = str(val).replace("'", "''")
        pivot_cols.append(f"MAX(CASE WHEN \"{name_col}\" = '{escaped_val}' THEN \"{value_col}\" END) AS \"{val}\"")
    
    id_cols_str = ', '.join(f'\"{col}\"' for col in id_cols)
    pivot_str = ', '.join(pivot_cols)
    
    query = f"""
        SELECT {id_cols_str}, {pivot_str}
        FROM {table_name}
        GROUP BY {id_cols_str}
    """
    
    return conn.query(query)


def pivot_longer(relation: duckdb.DuckDBPyRelation,
                 cols: Union[str, List[str]],
                 names_to: str = 'variable',
                 values_to: str = 'value',
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Pivot data from wide to long format (Longer = gather columns into rows).
    
    This transforms wide data by gathering specified columns into key-value pairs.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation (wide format).
    cols : str or list of str
        Column(s) to pivot to long format.
    names_to : str
        Name of the new column containing variable names.
    values_to : str
        Name of the new column containing values.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation in long format.
    """
    conn = conn or duckdb.connect()
    
    if isinstance(cols, str):
        cols = [cols]
    
    # Get all columns from the relation
    old_columns = relation.columns
    id_cols = [col for col in old_columns if col not in cols]
    
    # Build UNION ALL query for each column
    parts = []
    for col in cols:
        # Escape quotes in column names
        escaped_col = col.replace('"', '""')
        if id_cols:
            id_select = ', '.join(f'\"{c}\"' for c in id_cols)
            parts.append(f"SELECT {id_select}, '{escaped_col}' AS \"{names_to}\", \"{col}\" AS \"{values_to}\" FROM relation")
        else:
            parts.append(f"SELECT '{escaped_col}' AS \"{names_to}\", \"{col}\" AS \"{values_to}\" FROM relation")
    
    query = ' UNION ALL '.join(parts)
    
    return conn.query(query)


def truncate_datetime(relation: duckdb.DuckDBPyRelation, column: str,
                      unit: str = 'day',
                      target_column: Optional[str] = None,
                      conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Truncate a datetime column to a specified unit.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to truncate.
    unit : str
        Unit to truncate to ('year', 'month', 'day', 'hour', 'minute', 'second').
    target_column : str, optional
        Name of the new column. If None, replaces the original.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with truncated datetime column.
    """
    conn = conn or duckdb.connect()
    
    old_columns = relation.columns
    table_name = f'_temp_{id(relation)}'
    conn.register(table_name, relation)
    
    if target_column is None:
        target_column = column
    
    # DuckDB date_trunc function
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
    
    truncate_expr = f"CAST(DATE_TRUNC('{unit_map[unit]}', CAST(\"{column}\" AS TIMESTAMP)) AS DATE)"
    
    if target_column == column:
        select_parts = [f'\"{c}\"' for c in old_columns if c != column]
        select_parts.append(f'{truncate_expr} AS \"{column}\"')
    else:
        select_parts = [f'\"{c}\"' for c in old_columns]
        select_parts.append(f'{truncate_expr} AS \"{target_column}\"')
    
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    
    return conn.query(query)


def fill_empty(relation: duckdb.DuckDBPyRelation, column: str,
               value: str = '',
               conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Fill empty strings in a column with a specified value.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to fill.
    value : str
        Value to fill empty strings with.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. If None, creates a new one.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with filled column.
    """
    conn = conn or duckdb.connect()
    
    old_columns = relation.columns
    table_name = f'_temp_{id(relation)}'
    conn.register(table_name, relation)
    
    if column not in old_columns:
        raise ValueError(f"Column '{column}' not found")
    
    select_parts = []
    for col in old_columns:
        if col == column:
            select_parts.append(f"COALESCE(NULLIF(\"{col}\", ''), '{value}') AS \"{col}\"")
        else:
            select_parts.append(f'\"{col}\"')
    
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    
    return conn.query(query)
