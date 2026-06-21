"""
DuckJanitor - Main class for DuckDB-backed data cleaning.

This module provides the DuckJanitor class that wraps DuckDB relations
and provides a method-chaining API for data cleaning operations.
"""

import pandas as pd
import duckdb
from typing import Optional, Union, List, Dict, Any, Callable
from pathlib import Path


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
    
    def __init__(self, relation: duckdb.DuckDBPyRelation, connection: Optional[duckdb.DuckDBPyConnection] = None):
        """
        Initialize DuckJanitor with a DuckDB relation.
        
        Parameters
        ----------
        relation : duckdb.DuckDBPyRelation
            The DuckDB relation.
        connection : duckdb.DuckDBPyConnection, optional
            The DuckDB connection that owns the relation. If omitted, an attempt
            is made to use a fresh connection; this only succeeds for relations
            that can be registered there (e.g., created by the same process).
        """
        if connection is None:
            connection = duckdb.connect()
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
        conn = duckdb.connect()
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
        conn = duckdb.connect()

        if isinstance(path, list):
            path_list = [str(p) for p in path]
            relation = conn.query(
                f"SELECT * FROM read_parquet([{', '.join(repr(p) for p in path_list)}])"
            )
        else:
            relation = conn.query(f"SELECT * FROM read_parquet({repr(str(path))})")

        return cls(relation, connection=conn)
    
    @classmethod
    def from_csv(cls, path: Union[str, Path], **kwargs) -> 'DuckJanitor':
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
        conn = duckdb.connect()
        relation = conn.read_csv(str(path), **kwargs)
        return cls(relation, connection=conn)
    
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
        conn = connection or duckdb.connect()
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
        query = query.replace('self', temp_name)
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
        """Expand DataFrame to include all possible combinations."""
        from .cleaning_ops_final import complete as _complete
        result = _complete(self._relation, columns, fill_value, self._connection)
        return result
    
    def also(self, func: Callable) -> 'DuckJanitor':
        """Apply a Python function with side effects (materializes data)."""
        from .cleaning_ops_final import also as _also
        result = _also(self, func)
        return result
    
    def alias(self, alias: Union[str, Callable]) -> 'DuckJanitor':
        """Rename all columns using a string or callable (materializes data)."""
        from .cleaning_ops_final import alias as _alias
        result = _alias(self, alias)
        return result
    
    def mutate(self, **kwargs) -> 'DuckJanitor':
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
    
    def __repr__(self) -> str:
        return f"DuckJanitor(relation={self._relation.columns}, lazy=True)"
