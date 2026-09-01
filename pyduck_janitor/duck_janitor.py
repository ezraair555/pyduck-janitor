"""
DuckJanitor - Main class for DuckDB-backed data cleaning.

This module provides the DuckJanitor class that wraps DuckDB relations
and provides a method-chaining API for data cleaning operations.
"""

import pandas as pd
import duckdb
from typing import Optional, Union, List, Dict, Any, Callable
from pathlib import Path
import re


class DuckJanitor:
    """
    DuckDB-backed DataFrame wrapper for high-performance data cleaning.
    
    DuckJanitor provides a method-chaining API similar to pyjanitor,
    but leverages DuckDB for faster execution and out-of-core processing.
    
    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The DuckDB relation to wrap.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from pyduck_janitor import DuckJanitor
    >>> 
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6]})
    >>> dj = DuckJanitor.from_pandas(df)
    >>> cleaned = dj.dropna().collect()
    """
    _shared_conn: Optional[duckdb.DuckDBPyConnection] = None

    @classmethod
    def get_shared_connection(cls) -> duckdb.DuckDBPyConnection:
        """Get or create the global shared DuckDB connection."""
        if cls._shared_conn is None:
            cls._shared_conn = duckdb.connect()
        return cls._shared_conn
    
    def __init__(self, relation: duckdb.DuckDBPyRelation, connection: Optional[duckdb.DuckDBPyConnection] = None) -> None:
        """
        Initialize DuckJanitor with a DuckDB relation.
        
        Parameters
        ----------
        relation : duckdb.DuckDBPyRelation
            The DuckDB relation.
        connection : duckdb.DuckDBPyConnection, optional
            The DuckDB connection that owns the relation. If omitted, the global
            shared connection is used.
        """
        if connection is None:
            connection = DuckJanitor.get_shared_connection()
            try:
                temp_name = f"_validate_{id(relation)}"
                connection.register(temp_name, relation)
                connection.execute(f"SELECT 1 FROM {temp_name} LIMIT 0")
            except Exception as exc:
                raise ValueError(
                    "Could not derive a matching DuckDB connection from the relation. "
                    "Pass the connection used to create the relation explicitly."
                ) from exc

        # Validate that the relation belongs to the provided connection.
        try:
            temp_name = f"_validate_{id(relation)}"
            connection.register(temp_name, relation)
            connection.execute(f"SELECT 1 FROM {temp_name} LIMIT 0")
        except Exception as exc:
            raise ValueError(
                "The relation does not belong to the provided DuckDB connection."
            ) from exc

        self._relation = relation
        self._connection = connection
    
    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> 'DuckJanitor':
        """
        Create a DuckJanitor from a pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame.
            
        Returns
        -------
        DuckJanitor
            A DuckJanitor instance.
        """
        conn = cls.get_shared_connection()
        relation = conn.from_df(df)
        return cls(relation, connection=conn)
    
    @classmethod
    def from_parquet(cls, path: Union[str, Path, List[str]]) -> 'DuckJanitor':
        """
        Create a DuckJanitor from Parquet file(s).
        
        Parameters
        ----------
        path : str, Path, or list of str/Path
            Path(s) to Parquet file(s). Can be local or remote (s3://, http://).
            
        Returns
        -------
        DuckJanitor
            A DuckJanitor instance.
            
        Examples
        --------
        >>> dj = DuckJanitor.from_parquet('data.parquet')
        >>> dj = DuckJanitor.from_parquet(['part1.parquet', 'part2.parquet'])
        >>> dj = DuckJanitor.from_parquet('s3://bucket/data.parquet')
        """
        conn = cls.get_shared_connection()

        if isinstance(path, list):
            path_list = [str(p) for p in path]
            relation = conn.query(
                f"SELECT * FROM read_parquet([{', '.join(repr(p) for p in path_list)}])"
            )
        else:
            relation = conn.query(f"SELECT * FROM read_parquet({repr(str(path))})")

        return cls(relation, connection=conn)
    
    @classmethod
    def from_csv(cls, path: Union[str, Path], **kwargs: Any) -> 'DuckJanitor':
        """
        Create a DuckJanitor from a CSV file.
        
        Parameters
        ----------
        path : str or Path
            Path to the CSV file.
        **kwargs
            Additional arguments passed to DuckDB's read_csv.
            
        Returns
        -------
        DuckJanitor
            A DuckJanitor instance.
        """
        conn = cls.get_shared_connection()
        relation = conn.read_csv(str(path), **kwargs)
        return cls(relation, connection=conn)

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path, List[Union[str, Path]]],
        format: str = 'auto',
        **kwargs: Any,
    ) -> 'DuckJanitor':
        """
        Create a DuckJanitor from JSON / NDJSON file(s).

        Leverages DuckDB's native ``read_json_auto`` — no pandas bridge,
        so schema inference, compression detection, and glob patterns
        all work out of the box.

        Parameters
        ----------
        path : str, Path, or list of str/Path
            Path(s) to JSON file(s). Supports:
            - Single file (``data.json``)
            - NDJSON / JSONL (``logs.jsonl``)
            - Glob patterns (``data/*.json``)
            - List of files (``['a.json', 'b.json']``)
            - Remote URLs (``s3://``, ``https://``) with httpfs loaded
            - Gzip-compressed (``.json.gz``) — auto-detected
        format : str, optional
            ``'auto'`` (default), ``'array'``, or ``'newline_delimited'``.
            ``'auto'`` lets DuckDB infer from file content/extension.
        **kwargs
            Extra arguments forwarded to DuckDB's ``read_json_auto``
            (e.g. ``columns``, ``records``, ``maximum_object_size``).

        Returns
        -------
        DuckJanitor
            A DuckJanitor instance.

        Examples
        --------
        >>> dj = DuckJanitor.from_json('data.json')
        >>> dj = DuckJanitor.from_json('logs.jsonl')  # NDJSON
        >>> dj = DuckJanitor.from_json('s3://bucket/data.json')
        >>> dj = DuckJanitor.from_json('data/*.json')  # glob
        >>> dj = DuckJanitor.from_json(['part1.json', 'part2.json'])
        >>> dj = DuckJanitor.from_json('data.json', format='array')
        """
        conn = cls.get_shared_connection()

        if isinstance(path, list):
            path_repr = '[' + ', '.join(repr(str(p)) for p in path) + ']'
            query = f"SELECT * FROM read_json_auto({path_repr}, format='{format}'"
        else:
            query = f"SELECT * FROM read_json_auto({repr(str(path))}, format='{format}'"

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, str):
                    query += f", {k}='{v}'"
                elif isinstance(v, bool):
                    query += f", {k}={str(v).lower()}"
                elif v is None:
                    query += f", {k}=NULL"
                else:
                    query += f", {k}={v}"
        query += ')'

        relation = conn.query(query)
        return cls(relation, connection=conn)

    @classmethod
    def from_excel(
        cls,
        path: Union[str, Path],
        sheet_name: Union[str, int, None] = 0,
        header: int = 0,
        usecols: Optional[Union[str, List[str], List[int]]] = None,
        skiprows: Optional[Union[int, List[int]]] = None,
        nrows: Optional[int] = None,
        na_values: Optional[Any] = None,
        keep_default_na: bool = True,
        dtype: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
        **kwargs: Any,
    ) -> 'DuckJanitor':
        """
        Create a DuckJanitor from an Excel file (.xlsx, .xls).

        Uses pandas.read_excel as a bridge into DuckDB. This keeps the
        API consistent with from_csv / from_parquet while relying on
        the mature openpyxl/xlrd ecosystem for parsing.

        Parameters
        ----------
        path : str or Path
            Path to the Excel file.
        sheet_name : str, int, or None, optional
            Sheet to read. 0-based index or sheet name. ``None`` reads
            all sheets (returns a dict in pandas; only the first is
            used here — pass a specific name/index to control).
        header : int, optional
            Row (0-indexed) to use as column names. Default 0.
        usecols : str, list, or None, optional
            Column selector accepted by pandas.read_excel
            (e.g. ``"A:C"``, ``["Name", "Age"]``, ``[0, 2]``).
        skiprows : int or list, optional
            Rows to skip at the start of the sheet.
        nrows : int, optional
            Number of rows to read.
        na_values : scalar, str, list, or dict, optional
            Additional strings to recognise as NaN.
        keep_default_na : bool, optional
            Whether to keep pandas' default NaN recognisers.
        dtype : dict, optional
            Column → dtype overrides.
        engine : str, optional
            'openpyxl' (xlsx) or 'xlrd' (xls). Auto-detected if omitted.
        **kwargs
            Extra keyword args forwarded to ``pandas.read_excel``.

        Returns
        -------
        DuckJanitor
            A DuckJanitor instance.

        Examples
        --------
        >>> dj = DuckJanitor.from_excel('data.xlsx')
        >>> dj = DuckJanitor.from_excel('report.xls', sheet_name='Summary')
        >>> dj = DuckJanitor.from_excel('data.xlsx', usecols='A:D', skiprows=2)
        """
        df = pd.read_excel(
            str(path),
            sheet_name=sheet_name,
            header=header,
            usecols=usecols,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            dtype=dtype,
            engine=engine,
            **kwargs,
        )
        # If sheet_name=None, pandas returns a dict of DataFrames.
        if isinstance(df, dict):
            if not df:
                raise ValueError(
                    f"Excel file '{path}' contains no sheets."
                )
            # Take the first sheet deterministically.
            first_key = next(iter(df))
            df = df[first_key]
        return cls.from_pandas(df)

    @classmethod
    def from_sql(cls, query: str, connection: Optional[duckdb.DuckDBPyConnection] = None) -> 'DuckJanitor':
        """
        Create a DuckJanitor from a SQL query.
        
        Parameters
        ----------
        query : str
            SQL query to execute.
        connection : duckdb.DuckDBPyConnection, optional
            DuckDB connection to use.
            
        Returns
        -------
        DuckJanitor
            A DuckJanitor instance.
        """
        conn = connection or cls.get_shared_connection()
        relation = conn.query(query)
        return cls(relation, connection=conn)
    
    def collect(self) -> pd.DataFrame:
        """
        Execute the pipeline and return results as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            The cleaned data.
            
        Examples
        --------
        >>> result = dj.clean_names().remove_empty().collect()
        """
        return self._relation.df()
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Return the first n rows.
        
        Parameters
        ----------
        n : int, optional
            Number of rows. Default is 5.
            
        Returns
        -------
        pd.DataFrame
            The first n rows.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        return self._relation.limit(n).df()
    
    def clean_names(self, strip_underscores: bool = True, case_type: str = 'lower', 
                    remove_special: bool = True, snakecase: bool = True) -> 'DuckJanitor':
        """
        Clean column names.
        
        Parameters
        ----------
        strip_underscores : bool, optional
            Remove leading/trailing underscores.
        case_type : str, optional
            Case conversion ('lower', 'upper', 'original').
        remove_special : bool, optional
            Remove special characters.
        snakecase : bool, optional
            Convert to snake_case.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import clean_names as _clean_names
        new_relation = _clean_names(self._relation, strip_underscores, case_type, remove_special, snakecase, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def remove_columns(self, columns: Union[str, List[str]]) -> 'DuckJanitor':
        """
        Remove specified columns.
        
        Parameters
        ----------
        columns : str or list of str
            Column name(s) to remove.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import remove_columns as _remove_columns
        new_relation = _remove_columns(self._relation, columns, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def add_column(self, column_name: str, values: Union[Any, List[Any], str], 
                   fill_value: Optional[Any] = None) -> 'DuckJanitor':
        """
        Add a new column.
        
        Parameters
        ----------
        column_name : str
            Name of the new column.
        values : scalar, list, or SQL expression
            Values for the column. Can be a scalar, list, or SQL expression string.
        fill_value : scalar, optional
            Fill value if values is shorter than DataFrame.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import add_column as _add_column
        new_relation = _add_column(self._relation, column_name, values, fill_value, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def rename_column(self, old_name: str, new_name: str) -> 'DuckJanitor':
        """
        Rename a column.
        
        Parameters
        ----------
        old_name : str
            Current column name.
        new_name : str
            New column name.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import rename_column as _rename_column
        new_relation = _rename_column(self._relation, old_name, new_name, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def dropna(self, subset: Optional[Union[str, List[str]]] = None, 
               how: str = 'any') -> 'DuckJanitor':
        """
        Remove rows with missing values.
        
        Parameters
        ----------
        subset : str or list of str, optional
            Column(s) to check for missing values.
        how : str, optional
            'any' or 'all' - whether to drop rows with any or all missing values.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import dropna as _dropna
        new_relation = _dropna(self._relation, subset, how, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def remove_empty(self) -> 'DuckJanitor':
        """
        Remove empty rows and columns.
        
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import remove_empty as _remove_empty
        new_relation = _remove_empty(self._relation, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def filter_column(self, column: str, criteria: Union[Callable, str]) -> 'DuckJanitor':
        """
        Filter rows based on column values.
        
        Parameters
        ----------
        column : str
            Column to filter on.
        criteria : callable or str
            Filter criteria. Can be a callable (lambda) or SQL WHERE clause string.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
            
        Examples
        --------
        >>> dj.filter_column('age', lambda x: x > 18)
        >>> dj.filter_column('sales', 'sales > 1000')
        """
        from .cleaning_ops import filter_column as _filter_column
        new_relation = _filter_column(self._relation, column, criteria, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def coalesce(self, columns: List[str], target_column: str) -> 'DuckJanitor':
        """
        Coalesce multiple columns into a single column.
        
        Parameters
        ----------
        columns : list of str
            Columns to coalesce.
        target_column : str
            Name of the resulting column.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import coalesce as _coalesce
        new_relation = _coalesce(self._relation, columns, target_column, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def encode_categorical(self, column: str, col_name: Optional[str] = None) -> 'DuckJanitor':
        """
        Encode a column as categorical.
        
        Parameters
        ----------
        column : str
            Column to encode.
        col_name : str, optional
            New column name. Defaults to original name.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import encode_categorical as _encode_categorical
        new_relation = _encode_categorical(self._relation, column, col_name, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def get_dummies(self, columns: Union[str, List[str]], prefix: Optional[str] = None) -> 'DuckJanitor':
        """
        One-hot encode categorical columns.
        
        Parameters
        ----------
        columns : str or list of str
            Column(s) to encode.
        prefix : str, optional
            Prefix for dummy column names.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops import get_dummies as _get_dummies
        new_relation = _get_dummies(self._relation, columns, prefix, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def filter_on(self, criteria: str, complement: bool = False) -> 'DuckJanitor':
        """Filter rows based on a SQL-like criteria string."""
        from .cleaning_ops import filter_on as _filter_on
        new_relation = _filter_on(self._relation, criteria, complement, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def filter_string(self, column: str, search_string: str, complement: bool = False,
                      case: bool = True, regex: bool = True) -> 'DuckJanitor':
        """Filter rows based on whether a string column contains a substring."""
        from .cleaning_ops import filter_string as _filter_string
        new_relation = _filter_string(self._relation, column, search_string, complement, case, regex, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def select_columns(self, columns: Union[str, List[str]]) -> 'DuckJanitor':
        """Select specific columns."""
        from .cleaning_ops import select_columns as _select_columns
        new_relation = _select_columns(self._relation, columns, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def select_rows(self, indices: Optional[Union[List[int], str]] = None,
                    criteria: Optional[str] = None) -> 'DuckJanitor':
        """Select specific rows by index or condition."""
        from .cleaning_ops import select_rows as _select_rows
        new_relation = _select_rows(self._relation, indices, criteria, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def transform_column(self, column: str, func: Union[str, Callable],
                         target_column: Optional[str] = None) -> 'DuckJanitor':
        """Transform a column using a function or SQL expression."""
        from .cleaning_ops import transform_column as _transform_column
        new_relation = _transform_column(self._relation, column, func, target_column, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def transform_columns(self, columns: Union[str, List[str]],
                         func: Union[str, Callable],
                         target_columns: Optional[Union[str, List[str]]] = None) -> 'DuckJanitor':
        """Transform multiple columns using a function or SQL expression."""
        from .cleaning_ops import transform_columns as _transform_columns
        new_relation = _transform_columns(self._relation, columns, func, target_columns, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def sql(self, query: str) -> 'DuckJanitor':
        """
        Execute a custom SQL query on the current relation.
        
        Parameters
        ----------
        query : str
            SQL query. Use 'self' to refer to the current relation.
            
        Returns
        -------
        DuckJanitor
            Self for method chaining.
            
        Examples
        --------
        >>> dj.sql("SELECT * FROM self WHERE age > 18")
        """
        # Register the current relation as a temporary table
        temp_name = f"_temp_{id(self._relation)}"
        self._connection.register(temp_name, self._relation)
        query = re.sub(r"\bself\b", temp_name, query)
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)
    
    def bin_numeric(self, column: str, target_column: str, bins: Union[int, List[float]] = 5,
                    strategy: str = 'quantile') -> 'DuckJanitor':
        """Bin a numeric column into discrete intervals."""
        from .cleaning_ops_extended import bin_numeric as _bin_numeric
        new_relation = _bin_numeric(self._relation, column, target_column, bins, strategy, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def change_type(self, column: str, dtype: str) -> 'DuckJanitor':
        """Change the data type of a column."""
        from .cleaning_ops_extended import change_type as _change_type
        new_relation = _change_type(self._relation, column, dtype, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def concatenate_columns(self, columns: List[str], sep: str = '_',
                            target_column: str = 'concatenated') -> 'DuckJanitor':
        """Concatenate multiple columns into a single column."""
        from .cleaning_ops_extended import concatenate_columns as _concatenate_columns
        new_relation = _concatenate_columns(self._relation, columns, sep, target_column, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def deconcatenate_column(self, column: str, sep: str,
                             target_columns: List[str]) -> 'DuckJanitor':
        """Split a column into multiple columns based on a delimiter."""
        from .cleaning_ops_extended import deconcatenate_column as _deconcatenate_column
        new_relation = _deconcatenate_column(self._relation, column, sep, target_columns, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def drop_constant_columns(self) -> 'DuckJanitor':
        """Remove columns that have only one unique value."""
        from .cleaning_ops_extended import drop_constant_columns as _drop_constant_columns
        new_relation = _drop_constant_columns(self._relation, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def fill(self, column: str, value: Optional[Any] = None,
             direction: str = 'forward',
             group_by: Optional[Union[str, List[str]]] = None) -> 'DuckJanitor':
        """Fill missing values in a column."""
        from .cleaning_ops_extended import fill as _fill
        new_relation = _fill(self._relation, column, value, direction, group_by, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def fill_empty(self, column: str, value: str = '') -> 'DuckJanitor':
        """Fill empty strings in a column with a specified value."""
        from .cleaning_ops_extended import fill_empty as _fill_empty
        new_relation = _fill_empty(self._relation, column, value, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def flag_nulls(self, columns: Optional[Union[str, List[str]]] = None,
                   prefix: str = 'is_null_', present_value: Any = 1,
                   absent_value: Any = 0) -> 'DuckJanitor':
        """Flag null values in specified columns with binary indicators."""
        from .cleaning_ops_extended import flag_nulls as _flag_nulls
        new_relation = _flag_nulls(self._relation, columns, prefix, present_value, absent_value, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def limit_column_characters(self, column: str, max_chars: int,
                                suffix: str = '...') -> 'DuckJanitor':
        """Limit the number of characters in a string column."""
        from .cleaning_ops_extended import limit_column_characters as _limit_column_characters
        new_relation = _limit_column_characters(self._relation, column, max_chars, suffix, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def min_max_scale(self, column: str, target_column: str,
                      min_val: float = 0, max_val: float = 1) -> 'DuckJanitor':
        """Apply Min-Max scaling to a numeric column."""
        from .cleaning_ops_extended import min_max_scale as _min_max_scale
        new_relation = _min_max_scale(self._relation, column, target_column, min_val, max_val, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def groupby_agg(self, by: Union[str, List[str]],
                    aggregations: Dict[str, Union[str, Dict]]) -> 'DuckJanitor':
        """Perform groupby aggregation."""
        from .cleaning_ops_extended import groupby_agg as _groupby_agg
        new_relation = _groupby_agg(self._relation, by, aggregations, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def groupby_topk(self, by: Union[str, List[str]], column: str,
                     k: int, ascending: bool = False) -> 'DuckJanitor':
        """Get top k rows within each group based on a column."""
        from .cleaning_ops_extended import groupby_topk as _groupby_topk
        new_relation = _groupby_topk(self._relation, by, column, k, ascending, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def case_when(self, conditions: List[tuple], target_column: str,
                  default: Optional[Any] = None) -> 'DuckJanitor':
        """Create a column based on multiple conditions (SQL CASE WHEN)."""
        from .cleaning_ops_extended import case_when as _case_when
        new_relation = _case_when(self._relation, conditions, target_column, default, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def currency_column_to_numeric(self, column: str,
                                   target_column: Optional[str] = None) -> 'DuckJanitor':
        """Convert a currency column to numeric."""
        from .cleaning_ops_extended import currency_column_to_numeric as _currency_column_to_numeric
        new_relation = _currency_column_to_numeric(self._relation, column, target_column, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def convert_date(self, column: str, target_column: Optional[str] = None,
                     date_format: Optional[str] = None) -> 'DuckJanitor':
        """Convert a column to date type."""
        from .cleaning_ops_extended import convert_date as _convert_date
        new_relation = _convert_date(self._relation, column, target_column, date_format, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def truncate_datetime(self, column: str, unit: str = 'day',
                         target_column: Optional[str] = None) -> 'DuckJanitor':
        """Truncate a datetime column to a specified unit."""
        from .cleaning_ops_extended import truncate_datetime as _truncate_datetime
        new_relation = _truncate_datetime(self._relation, column, unit, target_column, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def conditional_join(self, other: 'DuckJanitor', on: List[tuple],
                         how: str = 'inner') -> 'DuckJanitor':
        """Perform conditional (non-equi) joins."""
        from .cleaning_ops_final import conditional_join as _conditional_join
        new_relation = _conditional_join(self._relation, other._relation, on, how, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def get_dupes(self, columns: Optional[Union[str, List[str]]] = None) -> 'DuckJanitor':
        """Return duplicate rows."""
        from .cleaning_ops_final import get_dupes as _get_dupes
        new_relation = _get_dupes(self._relation, columns, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def dropnotnull(self, subset: Optional[Union[str, List[str]]] = None,
                    how: str = 'any') -> 'DuckJanitor':
        """Remove rows where values are NOT null (keep nulls)."""
        from .cleaning_ops_final import dropnotnull as _dropnotnull
        new_relation = _dropnotnull(self._relation, subset, how, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def expand_column(self, column: str, sep: str = '|',
                      prefix: Optional[str] = None) -> 'DuckJanitor':
        """Expand a delimited column into dummy variables."""
        from .cleaning_ops_final import expand_column as _expand_column
        new_relation = _expand_column(self._relation, column, sep, prefix, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def impute(self, column: str, value: Optional[Any] = None,
               statistic: str = 'mean',
               group_by: Optional[Union[str, List[str]]] = None) -> 'DuckJanitor':
        """Impute missing values."""
        from .cleaning_ops_final import impute as _impute
        new_relation = _impute(self._relation, column, value, statistic, group_by, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def jitter(self, column: str, target_column: str,
               scale: float = 0.01, seed: Optional[int] = None) -> 'DuckJanitor':
        """Add random noise (jitter) to a numeric column."""
        from .cleaning_ops_final import jitter as _jitter
        new_relation = _jitter(self._relation, column, target_column, scale, seed, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def label_encode(self, columns: Union[str, List[str]],
                     suffix: str = '_encoded') -> 'DuckJanitor':
        """Encode categorical columns with numerical labels."""
        from .cleaning_ops_final import label_encode as _label_encode
        new_relation = _label_encode(self._relation, columns, suffix, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def find_replace(self, column: str, value_pairs: Dict[str, str],
                     target_column: Optional[str] = None) -> 'DuckJanitor':
        """Find and replace values in a column."""
        from .cleaning_ops_final import find_replace as _find_replace
        new_relation = _find_replace(self._relation, column, value_pairs, target_column, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def count_cumulative_unique(self, column: str,
                                 dest_column: str = 'cumulative_unique') -> 'DuckJanitor':
        """Return a column with cumulative count of unique values."""
        from .cleaning_ops_final import count_cumulative_unique as _count_cumulative_unique
        new_relation = _count_cumulative_unique(self._relation, column, dest_column, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def complete(self, columns: Union[str, List[str]],
                 fill_value: Any = None) -> 'DuckJanitor':
        """Expand relation to include all possible combinations of specified columns."""
        from .cleaning_ops_final import complete as _complete
        new_relation = _complete(self._relation, columns, fill_value, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def also(self, func: Callable) -> 'DuckJanitor':
        """Apply a Python function with side effects (materializes data)."""
        from .cleaning_ops_final import also as _also
        result = _also(self, func)
        return result
    
    def alias(self, alias: Union[str, Callable]) -> 'DuckJanitor':
        """Rename all columns using a string or callable."""
        from .cleaning_ops_final import alias as _alias
        new_relation = _alias(self._relation, alias, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def drop_duplicate_columns(self) -> 'DuckJanitor':
        """Remove columns that are exact duplicates of other columns."""
        from .cleaning_ops_final import drop_duplicate_columns as _drop_duplicate_columns
        new_relation = _drop_duplicate_columns(self._relation, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def compare_df_cols(self, other: 'DuckJanitor') -> pd.DataFrame:
        """Compare columns between two DuckJanitor instances."""
        from .cleaning_ops_final import compare_df_cols as _compare_df_cols
        return _compare_df_cols(self, other, self._connection)

    def join_apply(self, other: 'DuckJanitor', on: Union[str, List[str]],
                   func: Callable, new_column_name: str) -> 'DuckJanitor':
        """Perform join then apply Python function to each row."""
        from .cleaning_ops_final import join_apply as _join_apply
        return _join_apply(self, other, on, func, new_column_name, self._connection)

    def process_text(self, column: str, func: Union[Callable, str],
                     new_column_name: str) -> 'DuckJanitor':
        """Apply text processing function to a column."""
        from .cleaning_ops_final import process_text as _process_text
        return _process_text(self, column, func, new_column_name, self._connection)
    
    def mutate(self, **kwargs: Any) -> 'DuckJanitor':
        """Create or modify columns using a dictionary (convenience wrapper)."""
        from .cleaning_ops_final import mutate as _mutate
        result = _mutate(self, **kwargs)
        return result
    
    def pivot_wider(self, id_cols: Union[str, List[str]],
                    name_col: str, value_col: str) -> 'DuckJanitor':
        """Pivot data from long to wide format."""
        from .cleaning_ops_extended import pivot_wider as _pivot_wider
        new_relation = _pivot_wider(self._relation, id_cols, name_col, value_col, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def pivot_longer(self, cols: Union[str, List[str]],
                     names_to: str = 'variable', values_to: str = 'value') -> 'DuckJanitor':
        """Pivot data from wide to long format."""
        from .cleaning_ops_extended import pivot_longer as _pivot_longer
        new_relation = _pivot_longer(self._relation, cols, names_to, values_to, self._connection)
        return DuckJanitor(new_relation, self._connection)
    
    def explain(self) -> str:
        """
        Show the query plan for the current pipeline.
        
        Returns
        -------
        str
            The query plan.
        """
        temp_name = f'_temp_explain_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = f"EXPLAIN SELECT * FROM {temp_name}"
        return str(self._connection.execute(query).fetchall())

    # ---------------------------------------------------------------
    # pyjanitor parity aliases (R/pyjanitor function name alignment).
    # These thin wrappers align the function name with pyjanitor so
    # users migrating from pyjanitor do not have to learn new names.
    # ---------------------------------------------------------------

    def rename_columns(self, old_name: str, new_name: str) -> 'DuckJanitor':
        """Alias of :meth:`rename_column` for the pyjanitor plural name."""
        return self.rename_column(old_name, new_name)

    def truncate_datetime_dataframe(
        self,
        column: str,
        unit: str = 'day',
        target_column: Optional[str] = None,
    ) -> 'DuckJanitor':
        """Alias of :meth:`truncate_datetime` matching pyjanitor's name."""
        return self.truncate_datetime(column=column, unit=unit,
                                          target_column=target_column)

    def convert_to_date(self, column: str, date_format: Optional[str] = None,
                         target_column: Optional[str] = None,
                         **kwargs) -> 'DuckJanitor':
        """Alias of :meth:`convert_date` matching pyjanitor's name."""
        return self.convert_date(column=column, target_column=target_column,
                                    date_format=date_format)

    def convert_to_datetime(self, column: str, date_format: Optional[str] = None,
                              target_column: Optional[str] = None,
                              **kwargs) -> 'DuckJanitor':
        """Alias of :meth:`convert_date` matching pyjanitor's name."""
        return self.convert_date(column=column, target_column=target_column,
                                    date_format=date_format)

    def convert_unix_date(self, column: str,
                            unit: str = 'seconds',
                            target_column: Optional[str] = None) -> 'DuckJanitor':
        """Coerce a UNIX/epoch numeric column into a DuckDB TIMESTAMP.

        Parameters
        ----------
        column : str
            Name of the column holding the Unix epoch value.
        unit : str, default 'seconds'
            One of {'seconds', 'milliseconds', 'microseconds'}.
        target_column : str, optional
            Name of the output column. Defaults to ``column + '_datetime'``.
        """
        out = target_column or f'{column}_datetime'
        multipliers = {'seconds': 1, 'milliseconds': 1000, 'microseconds': 1000000}
        if unit not in multipliers:
            raise ValueError(
                f'convert_unix_date(): unit must be one of '
                f'{sorted(multipliers)}; got {unit!r}.'
            )
        multiplier = multipliers[unit]
        temp_name = f'_unix_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # Use TO_TIMESTAMP which accepts DOUBLE seconds directly.
        query = (
            f"SELECT *, TO_TIMESTAMP(CAST({column} AS DOUBLE) / {multiplier}) AS {out} "
            f"FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def convert_excel_date(self, column: str,
                              target_column: Optional[str] = None) -> 'DuckJanitor':
        """Coerce an Excel serial date number into a DuckDB TIMESTAMP.

        Excel's serial date origin is 1899-12-30 (with the 1900 leap-year
        bug adjustment). 1 = 1900-01-01.
        """
        out = target_column or f'{column}_datetime'
        temp_name = f'_excel_date_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # Use TO_TIMESTAMP which accepts seconds-from-epoch DOUBLE.
        query = (
            f"SELECT *, TO_TIMESTAMP(CAST({column} AS DOUBLE) * 86400.0 - 25569 * 86400.0) AS "
            f"{out} FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def convert_matlab_date(self, column: str,
                              target_column: Optional[str] = None) -> 'DuckJanitor':
        """Coerce a MATLAB serial date number into a DuckDB TIMESTAMP.

        MATLAB datenum origin is 0000-01-01; 1 = 0000-01-01. The offset to
        the DuckDB TIMESTAMP epoch (1970-01-01) is 719529 days.
        """
        out = target_column or f'{column}_datetime'
        temp_name = f'_matlab_date_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = (
            f"SELECT *, TO_TIMESTAMP(CAST({column} AS DOUBLE) * 86400.0 - 719529 * 86400.0) AS "
            f"{out} FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def fill_direction(self, column: str, direction: str = 'forward',
                        value=None, **kwargs) -> 'DuckJanitor':
        """Alias of :meth:`fill` matching pyjanitor's name."""
        return self.fill(column=column, value=value, direction=direction,
                          **kwargs)

    def filter_column_isin(self, column: str, values,
                            complement: bool = False) -> 'DuckJanitor':
        """Filter rows where ``column`` IS IN ``values``.

        ``values`` may be any iterable of scalars. Implemented as a
        direct DuckDB SQL filter because :meth:`filter_column` requires
        a callable/SQL-fragment criteria.
        """
        if column not in self._relation.columns:
            raise ValueError(
                f'filter_column_isin(): column {column!r} not in '
                f'{list(self._relation.columns)}'
            )
        try:
            values = list(values)
        except TypeError:
            raise TypeError(
                'filter_column_isin(): values must be an iterable of scalars.'
            )
        if not values:
            # Empty list returns empty relation.
            return self.filter_on('1 = 0', complement=False)

        def _format(v):
            if v is None:
                return 'NULL'
            if isinstance(v, bool):
                return 'TRUE' if v else 'FALSE'
            if isinstance(v, (int, float)):
                return repr(v)
            escaped = str(v).replace("'", "''")
            return f"'{escaped}'"

        in_list = ', '.join(_format(v) for v in values)
        op = 'NOT IN' if complement else 'IN'
        where_clause = f'"{column}" {op} ({in_list})'
        return self.filter_on(where_clause, complement=False)

    def add_columns(self, column_values: dict) -> 'DuckJanitor':
        """Alias of :meth:`add_column` accepting a dict of column→values
        so multiple columns can be added in a single chained call.
        """
        out = self
        for col, vals in column_values.items():
            out = out.add_column(col, vals)
        return out

    def assign(self, **kwargs) -> 'DuckJanitor':
        """Alias of :meth:`mutate` matching pyjanitor's name.

        ``assign`` accepts keyword arguments of ``column=value`` or
        ``column=callable``, exactly like :meth:`mutate`.
        """
        return self.mutate(**kwargs)

    def ungroup(self, *groups, **kwargs) -> 'DuckJanitor':
        """Identity helper that matches pyjanitor's ``ungroup`` verb.

        DuckDB relations are inherently ungrouped; this is a no-op that
        simply returns the current DuckJanitor, kept as a chainable verb.
        """
        return self

    def get_columns(self, *names) -> 'DuckJanitor':
        """Select columns by name (alias of :meth:`select_columns`)."""
        return self.select_columns(list(names))

    def move(self, source: str, target: str, position: str = 'before',
              **kwargs) -> 'DuckJanitor':
        """Move ``source`` column relative to ``target`` column.

        position: ``'before'`` (default) or ``'after'``.
        """
        cur_cols = list(self._relation.columns)
        if source not in cur_cols or target not in cur_cols:
            raise ValueError(
                f'move(): source={source!r} and target={target!r} must both '
                f'exist in the current columns: {cur_cols}'
            )
        new_order = [c for c in cur_cols if c != source]
        insert_at = new_order.index(target)
        if position == 'after':
            insert_at += 1
        new_order.insert(insert_at, source)
        cols = ', '.join(self._quote(c) for c in new_order)
        temp_name = f'_move_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = f'SELECT {cols} FROM {temp_name}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def reorder_columns(self, new_order) -> 'DuckJanitor':
        """Reorder the relation's columns to match ``new_order``.

        Columns not listed are dropped (matching pyjanitor's behaviour,
        where columns must be enumerated fully).
        """
        cur = list(self._relation.columns)
        missing = [c for c in new_order if c not in cur]
        if missing:
            raise ValueError(f'reorder_columns(): unknown columns {missing}')
        cols = ', '.join(self._quote(c) for c in new_order)
        temp_name = f'_reorder_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = f'SELECT {cols} FROM {temp_name}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def get_index_labels(self) -> List[str]:
        """Return the current column names as a list (label-only)."""
        return list(self._relation.columns)

    @staticmethod
    def _quote(col: str) -> str:
        return f'"{col.replace(chr(34), chr(34) * 2)}"'

    @staticmethod
    def _sql_value(v) -> str:
        if v is None:
            return 'NULL'
        if isinstance(v, bool):
            return 'TRUE' if v else 'FALSE'
        if isinstance(v, (int, float)):
            return repr(v)
        escaped = str(v).replace("'", "''")
        return f"'{escaped}'"

    # =============================================================
    # pyjanitor parity batch: small DuckDB-trivial helpers
    # =============================================================

    def shuffle(self, seed: Optional[int] = None) -> 'DuckJanitor':
        """Return a relation with all rows in random order (R: ``shuffle``).

        Parameters
        ----------
        seed : int, optional
            If provided, drives DuckDB's ``random()`` PRNG for reproducibility.
            The seed is normalized into ``[-1.0, 1.0]`` because DuckDB's
            ``setseed`` only accepts that range.
        """
        if seed is not None:
            normalized = (abs(int(seed)) % (2 ** 31)) / (2 ** 31)
            seed_clause = f'setseed({normalized})'
        else:
            seed_clause = None
        temp_name = f'_shuffle_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        if seed_clause is not None:
            self._connection.execute(f'SELECT {seed_clause}')
        query = f'SELECT * FROM {temp_name} ORDER BY random()'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def toset(self, column: str) -> list:
        """Return the unique sorted values of ``column`` as a Python list (R: ``toset``)."""
        temp_name = f'_toset_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        rows = self._connection.execute(
            f'SELECT DISTINCT "{column}" FROM {temp_name} ORDER BY "{column}"'
        ).fetchall()
        self._connection.unregister(temp_name)
        return [r[0] for r in rows]

    def take_first(self, n: int = 1) -> 'DuckJanitor':
        """Return a relation containing only the first ``n`` rows (R: ``take_first``)."""
        if n < 0:
            raise ValueError(f'take_first(): n must be >= 0; got {n}')
        temp_name = f'_take_first_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # LIMIT on an inner subquery to avoid WHERE-on-window-function binder error.
        query = f'SELECT * FROM (SELECT * FROM {temp_name}) LIMIT {n}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def excel_time_to_numeric(self, column: str,
                                target_column: Optional[str] = None) -> 'DuckJanitor':
        """Convert an Excel time-fraction column (``0.0``–``1.0``) to seconds.

        Excel stores time-of-day as the fractional part of a day; multiplying
        by ``86400`` yields seconds.
        """
        out = target_column or f'{column}_seconds'
        temp_name = f'_excel_time_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = (
            f'SELECT *, CAST({column} * 86400.0 AS DOUBLE) AS {out} '
            f'FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def sas_numeric_to_date(self, column: str,
                              target_column: Optional[str] = None) -> 'DuckJanitor':
        """Convert a SAS numeric date column (days since 1960-01-01) to TIMESTAMP."""
        out = target_column or f'{column}_datetime'
        temp_name = f'_sasdate_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = (
            f'SELECT *, TO_TIMESTAMP(CAST({column} AS DOUBLE) * 86400.0 - 315619200.0) AS {out} '
            f'FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def round_to_fraction(self, column: str, denominator: Union[int, float]) -> 'DuckJanitor':
        """Round ``column`` to the nearest fraction ``1/denominator`` (R: ``round_to_fraction``)."""
        if denominator == 0:
            raise ValueError('round_to_fraction(): denominator must be non-zero')
        temp_name = f'_rfrac_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = (
            f'SELECT *, ROUND(CAST({column} AS DOUBLE) * {denominator}) / '
            f'{denominator} AS "{column}_rounded" FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def scale_mad(self, column: str, by: str = 'all') -> 'DuckJanitor':
        """Median-abs-deviation standardisation (R: ``scale_mad``).

        ``by`` is one of ``'all'`` (default) or ``'column'``.
        """
        if by not in {'all', 'column'}:
            raise ValueError(f"scale_mad(): 'by' must be 'all' or 'column'; got {by!r}")
        temp_name = f'_mad_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = (
            f'SELECT *, ({column} - (SELECT MEDIAN({column}) FROM {temp_name})) / '
            f'(1.4826 * (SELECT MEDIAN(ABS({column} - (SELECT MEDIAN({column}) FROM {temp_name}))) '
            f'FROM {temp_name})) AS "{column}_scaled" '
            f'FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def cartesian_product(self, other: 'DuckJanitor') -> 'DuckJanitor':
        """Return the cartesian (cross) product of ``self`` and ``other``."""
        if not isinstance(other, DuckJanitor):
            raise TypeError(
                f'cartesian_product(): other must be a DuckJanitor; got {type(other).__name__}'
            )
        left_name = f'_cp_left_{id(self._relation)}'
        right_name = f'_cp_right_{id(other._relation)}'
        self._connection.register(left_name, self._relation)
        other._connection.register(right_name, other._relation)
        query = f'SELECT * FROM {left_name} CROSS JOIN {right_name}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def then(self, *funcs) -> 'DuckJanitor':
        """Compose further verbs in sequence (R: ``then`` / ``DF_to_pandas``).

        Each callable is invoked with the current DuckJanitor and must return
        a DuckJanitor. Useful for ``pipe``-style chaining across modules.
        """
        out: DuckJanitor = self
        for func in funcs:
            res = func(out)
            if not isinstance(res, DuckJanitor):
                raise TypeError(
                    f'then(): func {getattr(func, "__name__", repr(func))} '
                    f'must return DuckJanitor; got {type(res).__name__}'
                )
            out = res
        return out

    def compare_df_cols_same(self, *others: 'DuckJanitor') -> bool:
        """Compare the current relation's columns to other relations (R: ``compare_df_cols_same``)."""
        cur_cols = list(self._relation.columns)
        return all(cur_cols == list(o._relation.columns) for o in others)

    def __repr__(self) -> str:
        return f"DuckJanitor(relation={self._relation.columns}, lazy=True)"
