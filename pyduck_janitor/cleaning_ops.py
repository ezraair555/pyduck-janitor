"""
Cleaning operations for DuckJanitor.

This module provides the actual implementation of data cleaning operations
using DuckDB SQL expressions.
"""

import duckdb
from typing import Optional, Union, List, Any, Callable
import re


def _quote_id(name: str) -> str:
    """Return a safely quoted SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    """Return a safely quoted SQL literal."""
    if value is None:
        return 'NULL'
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _register_relation(conn: Optional[duckdb.DuckDBPyConnection],
                       relation: duckdb.DuckDBPyRelation) -> str:
    """Register *relation* on *conn* and return the temporary SQL name."""
    if conn is None:
        raise ValueError(
            "A DuckDB connection is required. Pass the connection that owns the relation."
        )
    table_name = f'_temp_{id(relation)}'
    conn.register(table_name, relation)
    return table_name


def clean_names(relation: duckdb.DuckDBPyRelation, strip_underscores: bool = True,
                case_type: str = 'lower', remove_special: bool = True,
                snakecase: bool = True,
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Clean column names.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    strip_underscores : bool
        Remove leading/trailing underscores.
    case_type : str
        Case conversion ('lower', 'upper', 'original').
    remove_special : bool
        Remove special characters.
    snakecase : bool
        Convert to snake_case.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with cleaned column names.
    """
    old_columns = relation.columns

    new_columns = []
    for col in old_columns:
        new_name = col

        if strip_underscores:
            new_name = new_name.strip('_')

        if remove_special:
            # Replace special chars with underscore first, then remove consecutive underscores
            new_name = re.sub(r'[^a-zA-Z0-9]', '_', new_name)
            # Remove consecutive underscores
            new_name = re.sub(r'_+', '_', new_name)
            # Strip leading/trailing underscores
            new_name = new_name.strip('_')

        # Handle snakecase - only for camelCase, not all-caps
        if snakecase:
            # Add underscores before uppercase letters (camelCase detection)
            # Only if there are lowercase letters (to avoid splitting all-caps like "CITY")
            if any(c.islower() for c in new_name):
                new_name = re.sub(r'(?<!^)(?=[A-Z])', '_', new_name)
            new_name = new_name.replace(' ', '_').replace('-', '_')

        if remove_special:
            # Replace special chars with underscore first
            new_name = re.sub(r'[^a-zA-Z0-9]', '_', new_name)
            # Remove consecutive underscores
            new_name = re.sub(r'_+', '_', new_name)

        if case_type == 'lower':
            new_name = new_name.lower()
        elif case_type == 'upper':
            new_name = new_name.upper()

        # Strip leading/trailing underscores
        new_name = new_name.strip('_')

        # Handle duplicate names
        if new_name in [c[1] for c in new_columns]:
            new_name = f"{new_name}_dup"

        new_columns.append((col, new_name))

    # Build SELECT statement with renamed columns
    select_parts = [f'{_quote_id(old)} AS {_quote_id(new)}' for old, new in new_columns]
    table_name = _register_relation(conn, relation)
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def remove_columns(relation: duckdb.DuckDBPyRelation,
                   columns: Union[str, List[str]],
                   conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Remove specified columns.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str
        Column name(s) to remove.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with columns removed.
    """
    # conn = conn or duckdb.connect()  # Removed: relation is bound to its connection
    
    if isinstance(columns, str):
        columns = [columns]
    
    old_columns = relation.columns
    keep_columns = [col for col in old_columns if col not in columns]
    
    if not keep_columns:
        raise ValueError("Cannot remove all columns")
    
    select_parts = [_quote_id(col) for col in keep_columns]
    table_name = _register_relation(conn, relation)
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    
    return conn.query(query)


def add_column(relation: duckdb.DuckDBPyRelation, column_name: str,
               values: Union[Any, List[Any], str],
               fill_value: Optional[Any] = None,
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Add a new column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column_name : str
        Name of the new column.
    values : scalar, list, or SQL expression
        Values for the column.
    fill_value : scalar, optional
        Fill value if values is shorter.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with new column added.
    """
    
    if isinstance(values, str):
        # SQL expression
        table_name = _register_relation(conn, relation)
        query = f"SELECT *, ({values}) AS {_quote_id(column_name)} FROM {table_name}"
    elif isinstance(values, list):
        # Create from list - need to use a different approach
        # For simplicity, convert to pandas, add column, convert back
        df = relation.df()
        if len(values) < len(df):
            if fill_value is not None:
                values = values + [fill_value] * (len(df) - len(values))
            else:
                values = values + [None] * (len(df) - len(values))
        df[column_name] = values[:len(df)]
        return conn.from_df(df)
    else:
        # Scalar value
        if isinstance(values, str):
            values = _sql_literal(values)
        table_name = _register_relation(conn, relation)
        query = f"SELECT *, {values} AS {_quote_id(column_name)} FROM {table_name}"
        return conn.query(query)

    return conn.query(query)


def rename_column(relation: duckdb.DuckDBPyRelation, old_name: str,
                  new_name: str,
                    conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Rename a column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    old_name : str
        Current column name.
    new_name : str
        New column name.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with column renamed.
    """
    # conn = conn or duckdb.connect()  # Removed: relation is bound to its connection
    old_columns = relation.columns
    
    if old_name not in old_columns:
        raise ValueError(f"Column '{old_name}' not found")
    
    select_parts = []
    for col in old_columns:
        if col == old_name:
            select_parts.append(f'{_quote_id(col)} AS {_quote_id(new_name)}')
        else:
            select_parts.append(_quote_id(col))
    
    table_name = _register_relation(conn, relation)
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    
    return conn.query(query)


def dropna(relation: duckdb.DuckDBPyRelation, subset: Optional[Union[str, List[str]]] = None,
           how: str = 'any',
                    conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Remove rows with missing values.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    subset : str or list of str, optional
        Column(s) to check.
    how : str
        'any' or 'all'.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with missing values removed.
    """
    table_name = _register_relation(conn, relation)
    
    if subset is None:
        subset = relation.columns
    elif isinstance(subset, str):
        subset = [subset]
    
    if how == 'any':
        # Keep rows where ALL checked columns are non-null.
        conditions = [f'{_quote_id(col)} IS NOT NULL' for col in subset]
        where_clause = ' AND '.join(conditions)
    elif how == 'all':
        # Keep rows where NOT ALL checked columns are null.
        null_conditions = [f'{_quote_id(col)} IS NULL' for col in subset]
        where_clause = 'NOT (' + ' AND '.join(null_conditions) + ')'
    else:
        raise ValueError("how must be 'any' or 'all'")
    
    query = f"SELECT * FROM {table_name} WHERE {where_clause}"
    
    return conn.query(query)


def remove_empty(relation: duckdb.DuckDBPyRelation,
                 conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Remove empty rows and columns.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with empty rows/columns removed.
    """
    table_name = _register_relation(conn, relation)
    old_columns = relation.columns
    non_empty_cols = []

    for col in old_columns:
        # A column is non-empty if it has at least one non-null, non-empty value.
        query = (
            f"SELECT COUNT(*) FROM {table_name} "
            f"WHERE {_quote_id(col)} IS NOT NULL "
            f"AND CAST({_quote_id(col)} AS VARCHAR) != ''"
        )
        result = conn.execute(query).fetchone()[0]
        if result > 0:
            non_empty_cols.append(col)

    if not non_empty_cols:
        raise ValueError("All columns are empty")

    # Remove rows where every non-empty column is null or empty.
    select_parts = [_quote_id(col) for col in non_empty_cols]
    row_conditions = [
        f"({_quote_id(col)} IS NOT NULL AND CAST({_quote_id(col)} AS VARCHAR) != '')"
        for col in non_empty_cols
    ]
    where_clause = ' OR '.join(row_conditions)

    query = f"SELECT {', '.join(select_parts)} FROM {table_name} WHERE {where_clause}"

    return conn.query(query)


def filter_column(relation: duckdb.DuckDBPyRelation, column: str,
                  criteria: Union[Callable, str],
                  conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Filter rows based on column values.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to filter on.
    criteria : callable or str
        Filter criteria.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Filtered relation.
    """
    if isinstance(criteria, str):
        # SQL WHERE clause
        table_name = _register_relation(conn, relation)
        query = f"SELECT * FROM {table_name} WHERE {criteria}"
    elif callable(criteria):
        # Convert callable to SQL - this is a simplification
        # For complex callables, need to evaluate in Python
        if conn is None:
            raise ValueError(
                "A DuckDB connection is required when using a callable criteria."
            )
        df = relation.df()
        mask = criteria(df[column])
        df = df[mask]
        return conn.from_df(df)
    else:
        raise ValueError("criteria must be a callable or SQL string")
    
    return conn.query(query)


def coalesce(relation: duckdb.DuckDBPyRelation, columns: List[str],
             target_column: str,
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Coalesce multiple columns into a single column.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : list of str
        Columns to coalesce.
    target_column : str
        Name of resulting column.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with coalesced column.
    """
    table_name = _register_relation(conn, relation)

    coalesce_expr = f"COALESCE({', '.join(_quote_id(col) for col in columns)}) AS {_quote_id(target_column)}"

    # Get all columns except the ones being coalesced
    old_columns = relation.columns
    other_cols = [col for col in old_columns if col not in columns]

    select_parts = [_quote_id(col) for col in other_cols] + [coalesce_expr]
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def encode_categorical(relation: duckdb.DuckDBPyRelation, column: str,
                       col_name: Optional[str] = None,
                    conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Encode a column as categorical (factorize).
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to encode.
    col_name : str, optional
        New column name.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with encoded column.
    """
    table_name = _register_relation(conn, relation)
    
    if col_name is None:
        col_name = f"{column}_cat"
    
    # Use DENSE_RANK to create categorical codes
    encode_expr = f"DENSE_RANK() OVER (ORDER BY {_quote_id(column)}) - 1 AS {_quote_id(col_name)}"
    
    old_columns = relation.columns
    other_cols = [col for col in old_columns if col != column]
    
    select_parts = [_quote_id(col) for col in other_cols] + [encode_expr]
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    
    return conn.query(query)


def get_dummies(relation: duckdb.DuckDBPyRelation, columns: Union[str, List[str]],
                prefix: Optional[str] = None,
                    conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    One-hot encode categorical columns.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str
        Column(s) to encode.
    prefix : str, optional
        Prefix for dummy columns.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with dummy columns.
    """
    table_name = _register_relation(conn, relation)
    
    if isinstance(columns, str):
        columns = [columns]
    
    # Get unique values for each column
    all_dummies = []
    old_columns = relation.columns
    
    for col in columns:
        query = f"SELECT DISTINCT {_quote_id(col)} FROM {table_name} WHERE {_quote_id(col)} IS NOT NULL"
        unique_vals = [row[0] for row in conn.execute(query).fetchall()]
        
        for val in unique_vals:
            if prefix:
                dummy_name = f"{prefix}_{col}_{val}"
            else:
                dummy_name = f"{col}_{val}"
            
            # Clean the name (remove spaces, special chars)
            dummy_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(dummy_name))
            
            # Create CASE expression for dummy variable
            case_expr = (
                f"CASE WHEN {_quote_id(col)} = {_sql_literal(val)} THEN 1 ELSE 0 END "
                f"AS {_quote_id(dummy_name)}"
            )
            
            all_dummies.append(case_expr)
    
    # Keep non-encoded columns
    other_cols = [col for col in old_columns if col not in columns]
    select_parts = [_quote_id(col) for col in other_cols] + all_dummies
    
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def select_columns(relation: duckdb.DuckDBPyRelation, columns: Union[str, List[str]],
                   conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Select specific columns from the relation.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str
        Column(s) to select.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with selected columns.
    """
    table_name = _register_relation(conn, relation)
    
    if isinstance(columns, str):
        columns = [columns]
    
    select_parts = [_quote_id(col) for col in columns]
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    
    return conn.query(query)


def select_rows(relation: duckdb.DuckDBPyRelation, indices: Optional[Union[List[int], str]] = None,
                criteria: Optional[str] = None,
                conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Select specific rows by index or condition.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    indices : list of int or str, optional
        Row indices to select, or a SQL criteria string.
    criteria : str, optional
        SQL WHERE clause condition for filtering rows.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with selected rows.
    """
    table_name = _register_relation(conn, relation)
    
    if indices is not None and criteria is None:
        # Select by row indices
        if isinstance(indices, str):
            # Treat as a criteria string
            criteria = indices
        elif isinstance(indices, list):
            # Use ROW_NUMBER() to filter by 1-based positions.
            positions = ', '.join(str(i + 1) for i in indices)
            query = f"""
                SELECT * EXCLUDE (row_num) FROM (
                    SELECT *, ROW_NUMBER() OVER () AS row_num FROM {table_name}
                ) WHERE row_num IN ({positions})
            """
            return conn.query(query)
    
    if criteria is not None:
        # Filter by SQL criteria
        query = f"SELECT * FROM {table_name} WHERE {criteria}"
        return conn.query(query)
    
    # If no criteria, return all rows
    return conn.query(f"SELECT * FROM {table_name}")


def transform_column(relation: duckdb.DuckDBPyRelation, column: str,
                     func: Union[str, Callable],
                     target_column: Optional[str] = None,
                     conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Transform a column using a function or SQL expression.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to transform.
    func : str or callable
        Transformation function or SQL expression.
    target_column : str, optional
        Name of the new column. If None, replaces the original.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with transformed column.
    """
    old_columns = relation.columns
    
    if isinstance(func, str):
        # SQL expression
        transform_expr = f"({func})"
    else:
        # For callable, we need to use a different approach
        # Convert to pandas, apply function, convert back
        table_name = _register_relation(conn, relation)
        query = f"SELECT * FROM {table_name}"
        df = conn.query(query).fetchdf()
        
        if target_column is None:
            target_column = column
        
        df[target_column] = df[column].apply(func)
        
        return conn.from_df(df)
    
    table_name = _register_relation(conn, relation)
    
    if target_column is None or target_column == column:
        # Replace the column
        select_parts = [_quote_id(c) for c in old_columns if c != column]
        select_parts.append(f"{transform_expr} AS {_quote_id(column)}")
    else:
        # Add a new column
        select_parts = [_quote_id(c) for c in old_columns]
        select_parts.append(f"{transform_expr} AS {_quote_id(target_column)}")
    
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    
    return conn.query(query)


def transform_columns(relation: duckdb.DuckDBPyRelation, columns: Union[str, List[str]],
                      func: Union[str, Callable],
                      target_columns: Optional[Union[str, List[str]]] = None,
                      conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Transform multiple columns using a function or SQL expression.
    """
    if isinstance(columns, str):
        columns = [columns]
    if target_columns is None:
        target_columns = columns
    elif isinstance(target_columns, str):
        target_columns = [target_columns]
    
    # For now, use transform_column for each column
    result = relation
    for i, col in enumerate(columns):
        target = target_columns[i] if i < len(target_columns) else col
        result = transform_column(result, col, func, target, conn)
    
    return result


def filter_on(relation: duckdb.DuckDBPyRelation, criteria: str,
              complement: bool = False,
              conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Filter rows based on a SQL-like criteria string.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    criteria : str
        SQL WHERE clause condition (e.g., "age > 25", "name LIKE '%John%'").
        Note: Use single quotes for string literals in DuckDB (e.g., 'NYC').
    complement : bool
        If True, return rows that DON'T match the criteria.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Filtered relation.
    """
    table_name = _register_relation(conn, relation)
    
    if complement:
        query = f"SELECT * FROM {table_name} WHERE NOT ({criteria})"
    else:
        query = f"SELECT * FROM {table_name} WHERE {criteria}"
    
    return conn.query(query)


def filter_string(relation: duckdb.DuckDBPyRelation, column: str, search_string: str,
                  complement: bool = False, case: bool = True,
                  regex: bool = True,
                  conn: Optional[duckdb.DuckDBPyConnection] = None) -> duckdb.DuckDBPyRelation:
    """
    Filter rows based on whether a string column contains a substring.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to search in.
    search_string : str
        String to search for.
    complement : bool
        If True, return rows that DON'T contain the search string.
    case : bool
        If True, case-sensitive matching.
    regex : bool
        If True, treat search_string as regex pattern.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use.
        
    Returns
    -------
    duckdb.DuckDBPyRelation
        Filtered relation.
    """
    table_name = _register_relation(conn, relation)
    col = _quote_id(column)
    pat = _sql_literal(search_string)
    
    if regex:
        op = 'NOT ' if complement else ''
        query = f"SELECT * FROM {table_name} WHERE {op}regexp_matches({col}, {pat})"
    else:
        if case:
            like_expr = f"{col} LIKE '%' || {pat} || '%'"
        else:
            like_expr = f"LOWER({col}) LIKE '%' || LOWER({pat}) || '%'"
        if complement:
            query = f"SELECT * FROM {table_name} WHERE NOT ({like_expr})"
        else:
            query = f"SELECT * FROM {table_name} WHERE {like_expr}"
    
    return conn.query(query)
