"""
DuckJanitor - Main class for DuckDB-backed data cleaning.

This module provides the DuckJanitor class that wraps DuckDB relations
and provides a method-chaining API for data cleaning operations.
"""

import re as _re_mod
from dataclasses import dataclass

import pandas as pd
import duckdb
from typing import Optional, Union, List, Dict, Any, Callable
from pathlib import Path
import re


class patterns(str):
    """pyjanitor-parity regex helper.

    In pyjanitor, ``patterns(regex_pattern)`` converts a string into a
    compiled regular expression for use in selection DSLs.  We subclass
    ``str`` so that the compiled pattern can also flow through the
    ``select_columns`` DSL unchanged, while still being usable directly
    via ``re.search(patterns('foo'), text)``.
    """

    @property
    def compiled(self):
        return _re_mod.compile(str(self))

    def search(self, text: str):
        return self.compiled.search(text)

    def match(self, text: str):
        return self.compiled.match(text)

    def findall(self, text: str):
        return self.compiled.findall(text)


@dataclass
class DropLabel:
    """pyjanitor-parity label-drop sentinel for the select DSL.

    Wraps a column label so that ``select_columns`` excludes it from the
    output, matching pyjanitor's ``DropLabel`` dataclass semantics::

        dj.select_columns([DropLabel('unwanted'), 'keep_me'])
    """

    label: str


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
        """Select specific columns.

        Accepts a list, a single column name, a comma-separated string,
        a shell-glob pattern (e.g. ``"v*"``), or a regex prefix (``"re:^v_"``).
        See :func:`pyduck_janitor.cleaning_ops.select_columns` for details.
        """
        from .cleaning_ops import select_columns as _select_columns
        new_relation = _select_columns(self._relation, columns, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def select(self, *args, **kwargs) -> 'DuckJanitor':
        """Convenience alias for the pyjanitor ``select`` DSL.

        The pyjanitor API page has ``select`` at the top of the
        ``select`` family, but their own docs state::

            This function has been deprecated.  Kindly use
            ``jn.select_columns`` or ``jn.select_rows``.

        We expose it as a chainable convenience whose only supported
        form is the ``columns=`` keyword. Positional ``args`` and other
        kwargs (``index=``, ``axis=``, ``invert=``) are accepted but
        not implemented; they will raise ``NotImplementedError``.
        """
        if args or any(k in kwargs for k in ('index', 'axis', 'invert', 'rows')):
            raise NotImplementedError(
                'select(): only the columns= shorthand is supported; '
                'use .select_columns(...).select_rows(...) for the '
                'full pyjanitor DSL.'
            )
        return self.select_columns(kwargs.get('columns', []))
    
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

    def describe_class(self, strict_description: bool = True) -> pd.DataFrame:
        """Describe the column types of this relation (pyjanitor ``describe_class`` parity).

        Parameters
        ----------
        strict_description : bool, default True
            When True (default), raise a descriptive ValueError if the
            relation has no columns.  When False, return an empty frame
            instead of raising.

        Returns
        -------
        pd.DataFrame
            One row per column with ``column_name`` and ``column_type``.
        """
        cur_cols = list(self._relation.columns)
        if not cur_cols:
            if strict_description:
                raise ValueError(
                    'describe_class(): relation has no columns to describe'
                )
            return pd.DataFrame(columns=['column_name', 'column_type'])
        temp_name = f'_describe_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        rows = self._connection.execute(
            f'DESCRIBE SELECT * FROM {temp_name}'
        ).fetchall()
        self._connection.unregister(temp_name)
        return pd.DataFrame(
            [{'column_name': r[0], 'column_type': r[1]} for r in rows],
            columns=['column_name', 'column_type'],
        )

    # =============================================================
    # pyjanitor parity batch 2 — medium helpers
    # =============================================================

    def row_to_names(self, row_number: int = 0,
                      remove_row: bool = True,
                      reset_index: bool = False) -> 'DuckJanitor':
        """Promote one row to the column headers (R: ``row_to_names``).

        Parameters
        ----------
        row_number : int, default 0
            Which (0-indexed) row to lift.
        remove_row : bool, default True
            Drop the promoted row from the resulting body.
        reset_index : bool, default False
            Kept for signature parity; DuckDB relations have no integer
            index to reset.
        """
        if row_number < 0:
            raise ValueError(f'row_to_names(): row_number must be >= 0; got {row_number}')
        temp_name = f'_row2n_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # Pull the row values to use as column names.
        n_cols = len(self._relation.columns)
        placeholders = ','.join(['?'] * n_cols)
        row_sql = (
            f'SELECT * FROM {temp_name} '
            f'LIMIT 1 OFFSET {row_number}'
        )
        row_values = self._connection.execute(row_sql).fetchone()
        if row_values is None:
            raise ValueError(f'row_to_names(): row {row_number} does not exist')
        # Build the new column projection with the promoted row as aliases.
        col_selects = ', '.join(
            f'CAST("{self._relation.columns[i]}" AS VARCHAR) AS "{row_values[i]}"'
            for i in range(n_cols)
        )
        if remove_row:
            # Strip out the lifted row by row_number().
            query = (
                f'SELECT {col_selects} FROM ('
                f'SELECT *, row_number() OVER () AS _rn FROM {temp_name}'
                f') WHERE _rn <> {row_number + 1}'
            )
        else:
            query = f'SELECT {col_selects} FROM {temp_name}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def rle_id(self) -> 'DuckJanitor':
        """Run-length id (R: ``rle_id``) — assign an integer id per change.

        Implementation uses a CTE chain that hashes all columns on each row
        and uses ``CONDITIONAL_TRUE_EVENT`` (via a stale flag) to break the
        run-length counter when the hash changes.
        """
        temp_name = f'_rle_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        cols = ', '.join(f'CAST("{c}" AS VARCHAR)' for c in self._relation.columns)
        query = (
            f'SELECT *, (SUM(CASE WHEN col_hash = prev_hash THEN 0 ELSE 1 END) '
            f'OVER (ORDER BY ord)) AS _rle_id FROM ('
            f'SELECT *, hash({cols}) AS col_hash, '
            f'LAG(hash({cols})) OVER (ORDER BY ord) AS prev_hash '
            f'FROM (SELECT *, row_number() OVER () AS ord FROM {temp_name})'
            f')'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def factorize_columns(self, columns=None,
                            append: bool = False) -> 'DuckJanitor':
        """Integer-encode each category in ``columns`` (R: ``factorize_columns``).

        ``columns`` defaults to all string-typed columns in the relation.
        """
        cur_cols = list(self._relation.columns)
        if columns is None:
            # Inspect the DuckDB types and use VARCHAR columns as candidates.
            desc_table = f'__dj_factorize_t_{id(self._relation)}'
            self._connection.register(desc_table, self._relation)
            try:
                rows = self._connection.execute(
                    f"SELECT column_name FROM (DESCRIBE SELECT * FROM {desc_table}) "
                    f"WHERE column_type LIKE 'VARCHAR%'"
                ).fetchall()
            finally:
                self._connection.unregister(desc_table)
            columns = [r[0] for r in rows] if rows else cur_cols
        else:
            missing = [c for c in columns if c not in cur_cols]
            if missing:
                raise ValueError(f'factorize_columns(): unknown columns {missing}')

        temp_name = f'_factorize_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        new_selects = [f'"{c}" AS "{c}"' for c in cur_cols]
        for col in columns:
            new_name = f'{col}_factor'
            new_selects.append(
                f'DENSE_RANK() OVER (ORDER BY "{col}") AS "{new_name}"'
            )
        cols_csv = ', '.join(new_selects)
        query = f'SELECT {cols_csv} FROM {temp_name}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def sort_naturally(self, column: str) -> 'DuckJanitor':
        """Natural-sort order for a column (R: ``sort_naturally``)."""
        import re as _re
        temp_name = f'_natsort_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = f'SELECT * FROM {temp_name}'
        material = self._connection.query(query).df()

        def _key(v):
            return [
                (int(chunk) if chunk.isdigit() else chunk)
                for chunk in _re.split(r'(\d+)', str(v))
            ]

        # Defensive cast to string for the sort key.
        material_sorted = material.assign(
            **{'_natsort_key': material[column].astype(str)}
        ).sort_values(by='_natsort_key', key=lambda s: s.map(_key))
        material_sorted = material_sorted.drop(columns=['_natsort_key']).reset_index(drop=True)
        return DuckJanitor.from_pandas(material_sorted)

    def sort_column_value_order(self, column: str,
                                  order: List[str]) -> 'DuckJanitor':
        """Sort rows by an explicit string ordering of ``column`` (R: ``sort_column_value_order``)."""
        if column not in self._relation.columns:
            raise ValueError(f'sort_column_value_order(): column {column!r} not in {list(self._relation.columns)}')
        temp_name = f'_sortorder_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # Validate that every value in `order` exists in the column.
        cur_values = {
            r[0] for r in self._connection.execute(
                f'SELECT DISTINCT "{column}" FROM {temp_name}'
            ).fetchall()
        }
        missing = [v for v in order if v not in cur_values]
        if missing:
            raise ValueError(
                f'sort_column_value_order(): values not in column {column!r}: {missing}'
            )
        # Use LIST_VALUE() to build the desired ordering; list_position
        # returns a numeric index per row, so missing keys sort at the end.
        order_list = '[' + ', '.join(repr(v) for v in order) + ']'
        query = (
            f'SELECT * FROM {temp_name} ORDER BY list_position({order_list}, "{column}")'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def filter_date(self, column: str, start_date=None, end_date=None) -> 'DuckJanitor':
        """Range-filter rows by a datetime ``column`` (R: ``filter_date``)."""
        parts = []
        if start_date is not None:
            parts.append(f'"{column}" >= {self._sql_value(start_date)}')
        if end_date is not None:
            parts.append(f'"{column}" <= {self._sql_value(end_date)}')
        if not parts:
            return self
        where = ' AND '.join(parts)
        return self.filter_on(where, complement=False)

    def update_where(self, columns: dict, conditions: str) -> 'DuckJanitor':
        """Update ``columns`` where a SQL ``conditions`` clause holds (R: ``update_where``).

        ``columns`` maps column-name to a SQL expression string.
        """
        if not isinstance(columns, dict) or not columns:
            raise ValueError('update_where(): columns must be a non-empty dict')
        cur_cols = list(self._relation.columns)
        missing = [c for c in columns if c not in cur_cols]
        if missing:
            raise ValueError(f'update_where(): unknown columns {missing}')
        temp_name = f'_updwhere_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # Build CASE WHEN update expressions per column.
        case_clauses = []
        for col, expr in columns.items():
            case_clauses.append(
                f'CASE WHEN {conditions} THEN ({expr}) ELSE "{col}" END AS "{col}"'
            )
        select_list = ', '.join(case_clauses)
        query = f'SELECT {select_list} FROM {temp_name}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def unionize_dataframe_categories(self, *others: 'DuckJanitor',
                                        column_names=None) -> 'DuckJanitor':
        """Cast all factor-like columns across many DuckJanitors to consistent string types.

        Useful as a preprocessing step before concat().
        """
        if not others:
            return self
        # Determine the superset of column names.
        all_cols = list(self._relation.columns)
        for o in others:
            for c in o._relation.columns:
                if c not in all_cols:
                    all_cols.append(c)
        targets = column_names if column_names is not None else all_cols
        # Cast each target column to VARCHAR in the relation.
        self._connection.register('_union_self', self._relation)
        select_list = []
        for c in self._relation.columns:
            if c in targets:
                select_list.append(f'CAST("{c}" AS VARCHAR) AS "{c}"')
            else:
                select_list.append(f'"{c}"')
        query = 'SELECT ' + ', '.join(select_list) + ' FROM _union_self'
        new_relation = self._connection.query(query)
        # Note: full per-column coercion across *all* relations is left to user-level
        # concat orchestration; this method only coerces the calling relation.
        return DuckJanitor(new_relation, self._connection)

    # =============================================================
    # pyjanitor parity batch 3 — heavyweight helpers
    # =============================================================

    def expand(self, columns: List[str], on=None) -> 'DuckJanitor':
        """Cartesian-expand across the unique values of ``columns`` (R: ``expand``).

        ``on`` is unused for now; in R it controls the iteration order.
        """
        if not columns:
            raise ValueError('expand(): columns must be a non-empty list')
        for c in columns:
            if c not in self._relation.columns:
                raise ValueError(f'expand(): column {c!r} not in {list(self._relation.columns)}')
        temp_name = f'_expand_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # Use DuckDB GROUP BY ALL to preserve distinct combinations.
        query = (
            f'SELECT DISTINCT {", ".join(self._quote(c) for c in columns)} '
            f'FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def expand_grid(self, *tables) -> 'DuckJanitor':
        """Cross-join an arbitrary number of relations or value lists (R: ``expand_grid``).

        Each argument may be either a DuckJanitor or an iterable of value
        combinations to be materialised.
        """
        if not tables:
            raise ValueError('expand_grid(): need at least one input')
        # Materialise each input into a relation, then cross-join them all.
        temp_self = f'_expand_grid_self_{id(self._relation)}'
        self._connection.register(temp_self, self._relation)
        result_name = temp_self
        result_cols = [f'{result_name}."{c}"' for c in self._relation.columns]
        for i, t in enumerate(tables):
            try:
                # Validate DuckJanitor.
                rel = t._relation
                tname = f'_expand_grid_x_{i}_{id(rel)}'
                self._connection.register(tname, t._relation)
                result_name = f'_expand_grid_res_{i}_{id(rel)}'
                self._connection.execute(
                    f'CREATE OR REPLACE TEMPORARY TABLE {result_name} AS '
                    f'SELECT {", ".join(result_cols + [f"{tname}.*"])} '
                    f'FROM (SELECT * FROM {tname}) CROSS JOIN (SELECT * FROM {temp_self})'
                )
                # After the CREATE, the result_name relation is the table itself.
                # Re-point for the next iteration.
                new_relation = self._connection.table(result_name)
                # Update temp_self to result_name for next round.
                temp_self = result_name
                result_cols = [f'{result_name}."{c}"' for c in new_relation.columns]
            except AttributeError:
                # Otherwise treat as an iterable of single-column rows.
                rows = list(t)
                if not rows:
                    raise ValueError(f'expand_grid(): arg {i} is empty')
                # Build a single-column relation.
                col_name = f'value_{i}'
                aux_name = f'_expand_grid_aux_{i}_{id(self._relation)}'
                self._connection.execute(
                    f"CREATE OR REPLACE TEMPORARY TABLE {aux_name}({col_name}) AS "
                    f"SELECT * FROM (VALUES " +
                    ', '.join(['(\'' + str(v).replace("'", "''") + '\')' for v in rows]) +
                    f")"
                )
                result_name = f'_expand_grid_res_{i}_{id(self._relation)}'
                self._connection.execute(
                    f'CREATE OR REPLACE TEMPORARY TABLE {result_name} AS '
                    f'SELECT {", ".join(result_cols + [f"{aux_name}.*"])} '
                    f'CROSS JOIN (SELECT * FROM {aux_name})'
                )
                new_relation = self._connection.table(result_name)
                temp_self = result_name
                result_cols = [f'{result_name}."{c}"' for c in new_relation.columns]
        final_name = f'_expand_grid_final_{id(self._relation)}'
        self._connection.execute(f'CREATE OR REPLACE TEMPORARY VIEW {final_name} AS SELECT * FROM {result_name}')
        new_relation = self._connection.view(final_name)
        # Best-effort cleanup of aux temp tables.
        for i in range(len(tables)):
            try:
                self._connection.execute(f'DROP TABLE IF EXISTS _expand_grid_aux_{i}_{id(self._relation)}')
            except Exception:
                pass
        return DuckJanitor(new_relation, self._connection)

    def change_index_dtype(self, dtype: str,
                             target_name: Optional[str] = None) -> 'DuckJanitor':
        """Create a cast version of the FIRST column with the desired ``dtype`` (R: ``change_index_dtype``).

        DuckDB relations have no intrinsic integer index; we mimic by
        projecting a typed copy of the first column.
        """
        cur_cols = list(self._relation.columns)
        if not cur_cols:
            raise ValueError('change_index_dtype(): relation has no columns')
        src = cur_cols[0]
        out = target_name or f'{src}_idx_typed'
        temp_name = f'_idx_dtype_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        query = (
            f'SELECT *, CAST("{src}" AS {dtype}) AS "{out}" '
            f'FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def collapse_levels(self, sep: str = '_',
                           column: Optional[str] = None) -> 'DuckJanitor':
        """Collapse a tuple-named index column by joining on ``sep`` (R: ``collapse_levels``).

        For simplicity, when ``column`` is None this concatenates all
        columns together. When ``column`` is given, only that column is
        collapsed (no-op since DuckDB columns are flat).
        """
        cur_cols = list(self._relation.columns)
        if column and column not in cur_cols:
            raise ValueError(f'collapse_levels(): unknown column {column!r}')
        temp_name = f'_collapse_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        if column:
            # No-op: a single column is already flat; return as-is.
            new_relation = self._connection.query(f'SELECT * FROM {temp_name}')
        else:
            select_parts = [
                f'CAST("{c}" AS VARCHAR) AS "{c}"' for c in cur_cols
            ]
            query = (
                f'SELECT *, '
                f'CONCAT(' +
                ', '.join(f'CAST("{c}" AS VARCHAR)' for c in cur_cols) +
                f') AS "{sep.join(cur_cols)}" '
                f'FROM {temp_name}'
            )
            new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def explode_index(self, column: str,
                        names: Optional[List[str]] = None,
                        separator: str = '_') -> 'DuckJanitor':
        """Split ``column`` into multiple sub-fields (R: ``explode_index``).

        ``names`` lists the new column names. By default a single new
        column called ``<column>_parsed`` is created.
        """
        if column not in self._relation.columns:
            raise ValueError(f'explode_index(): unknown column {column!r}')
        temp_name = f'_explode_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        cur_cols = list(self._relation.columns)
        out_cols = names or [f'{column}_parsed']
        selects = [self._quote(c) for c in cur_cols]
        # Toy implementation: extract numeric sequences as a single column.
        # DuckDB's regex support allows richer extraction; this is a stub.
        query = (
            f'SELECT {", ".join(selects)}, '
            f'regexp_extract(CAST("{column}" AS VARCHAR), \'(\\d+)\', 1) AS "{out_cols[0]}" '
            f'FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def summarise(self, group_by: Optional[List[str]] = None,
                    agg_spec: Optional[dict] = None) -> 'DuckJanitor':
        """Group-by summarisation helper (R: ``summarise`` / ``summarize``).

        Parameters
        ----------
        group_by : list of str
            Columns to group by; ``None`` means no grouping.
        agg_spec : dict
            Mapping of ``new_column_name -> (source_column, agg_function_string)``.
            Examples::

                {'avg_age': ('age', 'AVG'), 'n': ('*', 'COUNT')}
        """
        agg_spec = agg_spec or {}
        cur_cols = list(self._relation.columns)
        if group_by:
            missing = [c for c in group_by if c not in cur_cols]
            if missing:
                raise ValueError(f'summarise(): unknown group_by columns {missing}')
        temp_name = f'_summarise_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        select_parts = []
        if group_by:
            select_parts.extend(self._quote(c) for c in group_by)
        for new_col, (src, agg) in agg_spec.items():
            agg = agg.upper()
            if src == '*':
                expr = f'COUNT(*) AS {new_col}'
            else:
                if src not in cur_cols:
                    raise ValueError(f'summarise(): unknown source {src!r}')
                expr = f'{agg}("{src}") AS {new_col}'
            select_parts.append(expr)
        if not select_parts:
            raise ValueError('summarise(): no group_by or aggregation specified')
        group_by_clause = (
            f'GROUP BY {", ".join(self._quote(c) for c in group_by)}'
            if group_by else ''
        )
        query = f'SELECT {", ".join(select_parts)} FROM {temp_name} {group_by_clause}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def pivot_longer_spec(self, id_cols: List[str],
                            value_cols: List[str],
                            names_to: str = 'name',
                            values_to: str = 'value',
                            names_sep: Optional[str] = None) -> 'DuckJanitor':
        """Long-form pivot driven by a column-name spec (R: ``pivot_longer_spec``)."""
        if not value_cols:
            raise ValueError('pivot_longer_spec(): value_cols must be non-empty')
        missing_vals = [c for c in value_cols if c not in self._relation.columns]
        if missing_vals:
            raise ValueError(f'pivot_longer_spec(): unknown value_cols {missing_vals}')
        cur_cols = list(self._relation.columns)
        select_parts = [self._quote(c) for c in cur_cols if c not in value_cols]
        if names_sep:
            # Apply split on each value column name via DuckDB transform.
            for vc in value_cols:
                split_parts = vc.split(names_sep)
                for k, part in enumerate(split_parts):
                    alias = f'{names_to}_{k}' if k < len(split_parts) - 1 else values_to
                    if k < len(split_parts) - 1:
                        select_parts.append(f"'{part}' AS {alias}")
                # Last part becomes the values column.
                select_parts.append(f'CAST("{vc}" AS DOUBLE) AS "{values_to}_{vc}"')
        else:
            select_parts.append(f'UNNEST({", ".join(self._quote(v) for v in value_cols)}) AS "{values_to}"')
        temp_name = f'_plspec_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # Without UNNEST it's awkward to pivot properly here; emulate with cross-join.
        # Practical implementation: emit one row per (id × values) using unnest of an array.
        value_arr = '[' + ', '.join(self._quote(v) for v in value_cols) + ']'
        final_parts = []
        for c in cur_cols:
            if c in value_cols:
                continue
            final_parts.append(self._quote(c))
        # Build a structured unpivot using DuckDB's UNPIVOT extension.
        unpivot_query = f"SELECT * FROM {temp_name} UNPIVOT ({values_to} FOR {names_to} IN ({', '.join(self._quote(v) for v in value_cols)}))"
        new_relation = self._connection.query(unpivot_query)
        return DuckJanitor(new_relation, self._connection)

    def pivot_wider_spec(self, id_cols: List[str],
                            names_from: str,
                            values_from: str,
                            names_glue: str = '_') -> 'DuckJanitor':
        """Wide pivot driven by a column-name spec (R: ``pivot_wider_spec``)."""
        if names_from not in self._relation.columns:
            raise ValueError(f'pivot_wider_spec(): unknown names_from {names_from!r}')
        if values_from not in self._relation.columns:
            raise ValueError(f'pivot_wider_spec(): unknown values_from {values_from!r}')
        temp_name = f'_pwspec_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        # DuckDB's PIVOT requires a literal list of values. Collect them first.
        distinct_values = [
            r[0] for r in self._connection.execute(
                f'SELECT DISTINCT "{names_from}" FROM {temp_name} ORDER BY "{names_from}"'
            ).fetchall()
        ]
        if not distinct_values:
            raise ValueError('pivot_wider_spec(): no values in names_from')
        values_csv = ', '.join(self._sql_value(v) for v in distinct_values)
        id_csv = ', '.join(self._quote(c) for c in id_cols)
        query = (
            f'SELECT {id_csv} FROM {temp_name} '
            f'PIVOT (FIRST({values_from}) FOR {names_from} IN ({values_csv}))'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def join_agg(self, other: 'DuckJanitor', on: tuple,
                  aggs: dict) -> 'DuckJanitor':
        """Aggregate join (R: ``join_agg``) — left-join with arbitrary aggregations.

        ``aggs`` is a dict mapping ``new_col -> ("COL", AGG)``.
        """
        cur_cols = list(self._relation.columns)
        other_cols = list(other._relation.columns)
        if len(on) != 3:
            raise ValueError('join_agg(): on must be (left, right, op)')
        left_on, right_on, op = on
        if left_on not in cur_cols:
            raise ValueError(f'join_agg(): left_on {left_on!r} not in current columns')
        if right_on not in other_cols:
            raise ValueError(f'join_agg(): right_on {right_on!r} not in other columns')
        if op == '==':
            raise ValueError(
                "join_agg(): equality joins are not supported; use a non-equality op "
                "or use conditional_join() for equality."
            )
        left_name = f'_jagg_left_{id(self._relation)}'
        right_name = f'_jagg_right_{id(other._relation)}'
        self._connection.register(left_name, self._relation)
        other._connection.register(right_name, other._relation)
        agg_expressions = ', '.join(
            f'{agg.upper()}("{col}") AS {new_col}'
            for new_col, (col, agg) in aggs.items()
        )
        # Build a fully-qualified projection list so the join key columns
        # can be quoted explicitly to avoid "ambiguous column" binder errors.
        full_cur = ', '.join(f'"{left_name}"."{c}" AS "{c}"' for c in cur_cols)
        agg_columns = ', '.join(f'g."{new_col}" AS "{new_col}"' for new_col in aggs.keys())
        query = (
            f'SELECT {full_cur}, {agg_columns} '
            f'FROM {left_name} LEFT JOIN ('
            f'SELECT "{right_on}", {agg_expressions} '
            f'FROM {right_name} GROUP BY "{right_on}"'
            f') g ON "{left_name}"."{left_on}" {op} g."{right_on}"'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def get_join_indices(self, other: 'DuckJanitor', conditions) -> dict:
        """Compute join key indices without materialising the join (R: ``get_join_indices``)."""
        if not isinstance(other, DuckJanitor):
            raise TypeError('get_join_indices(): other must be a DuckJanitor')
        if isinstance(conditions, tuple) and len(conditions) == 3:
            conditions = [conditions]
        else:
            try:
                conditions = list(conditions)
            except TypeError:
                raise TypeError(
                    'get_join_indices(): conditions must be a (left, right, op) tuple or list'
                )
        left_name = f'_gjidx_l_{id(self._relation)}'
        right_name = f'_gjidx_r_{id(other._relation)}'
        self._connection.register(left_name, self._relation)
        other._connection.register(right_name, other._relation)
        result = {}
        for left_on, right_on, op in conditions:
            if left_on not in self._relation.columns:
                raise ValueError(f'get_join_indices(): left_on {left_on!r} missing')
            if right_on not in other._relation.columns:
                raise ValueError(f'get_join_indices(): right_on {right_on!r} missing')
            # Pull all values to compute indices in Python.
            left_vals = [
                r[0] for r in self._connection.execute(
                    f'SELECT "{left_on}" FROM {left_name}'
                ).fetchall()
            ]
            right_vals = [
                r[0] for r in other._connection.execute(
                    f'SELECT "{right_on}" FROM {right_name}'
                ).fetchall()
            ]
            # Generic dispatcher using Python operator.
            import operator as _op
            opmap = {'<': _op.lt, '<=': _op.le, '>': _op.gt, '>=': _op.ge,
                       '==': _op.eq, '!=': _op.ne}
            py_op = opmap.get(op)
            if py_op is None:
                raise ValueError(f'get_join_indices(): unsupported op {op!r}')
            pairs = [(i, j) for i, l in enumerate(left_vals)
                              for j, r in enumerate(right_vals)
                              if py_op(l, r)]
            result[(left_on, right_on, op)] = pairs
        return result

    def to_datetime(self, column: str, format: Optional[str] = None,
                      target_column: Optional[str] = None) -> 'DuckJanitor':
        """Cast ``column`` to a TIMESTAMP using DuckDB ``strptime`` (R: ``to_datetime``)."""
        out = target_column or f'{column}_ts'
        if column not in self._relation.columns:
            raise ValueError(f'to_datetime(): unknown column {column!r}')
        temp_name = f'_todt_{id(self._relation)}'
        self._connection.register(temp_name, self._relation)
        format_arg = f', {self._sql_value(format)}' if format is not None else ''
        query = (
            f'SELECT *, TRY_CAST(strptime(CAST("{column}" AS VARCHAR)'
            f'{format_arg}) AS TIMESTAMP) AS "{out}" '
            f'FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def __repr__(self) -> str:
        return f"DuckJanitor(relation={self._relation.columns}, lazy=True)"
