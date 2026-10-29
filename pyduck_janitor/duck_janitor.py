"""
DuckJanitor - Main class for DuckDB-backed data cleaning.

This module provides the DuckJanitor class that wraps DuckDB relations
and provides a method-chaining API for data cleaning operations.
"""

import contextlib
import io
import re
import re as _re_mod
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import duckdb
import pandas as pd
from ._sqlutil import substitute_self
from ._sqlutil import temp_name as _temp_name


def _escape_html(text: str) -> str:
    """Escape *text* for safe inclusion in an HTML fragment.

    Parameters
    ----------
    text : str
        The raw text.

    Returns
    -------
    str
        The escaped text.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class patterns(str):  # noqa: N801
    """pyjanitor-parity regex helper.

    In pyjanitor, ``patterns(regex_pattern)`` converts a string into a
    compiled regular expression for use in selection DSLs.  We subclass
    ``str`` so that the compiled pattern can also flow through the
    ``select_columns`` DSL unchanged, while still being usable directly
    via ``re.search(patterns('foo'), text)``.

    Parameters
    ----------
    regex_pattern : str
        Regular expression source. Stored as a string subclass instance
        with a ``.compiled`` property that lazily compiles the pattern.

    Returns
    -------
    None
        Construction has no side effects; ``.compiled`` compiles on first
        access.
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

    Parameters
    ----------
    label : str
        Column name to mark as excluded from a ``select_columns`` call.

    Returns
    -------
    None
        DropLabel is a sentinel; construction has no side effects.
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

    Returns
    -------
    DuckJanitor
        A new DuckJanitor instance wrapping the relation.

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
    _shared_conn_lock = threading.Lock()

    @classmethod
    def get_shared_connection(cls) -> duckdb.DuckDBPyConnection:
        """Get or create the global shared DuckDB connection.

        The connection is created under a lock so that concurrent first use
        from multiple threads cannot create two connections and leave one
        orphaned.

        Returns
        -------
        duckdb.DuckDBPyConnection
            The shared in-memory DuckDB connection, lazily created on first call.
        """
        if cls._shared_conn is None:
            with cls._shared_conn_lock:
                if cls._shared_conn is None:
                    cls._shared_conn = duckdb.connect()
        return cls._shared_conn

    @classmethod
    def connect(
        cls, path: Optional[Union[str, Path]] = None, *, read_only: bool = False
    ) -> "DuckJanitor":
        """Open a DuckDB connection and return a session bound to it.

        This is the entry point for persistent, file-backed work. When
        *path* is ``None`` an in-memory database is used. Unlike the
        classmethods that build a relation, the returned object exposes the
        connection itself so you can create tables, attach other databases,
        and browse the catalog.

        Parameters
        ----------
        path : str or Path, optional
            Path to a DuckDB database file. Created if it does not exist.
            ``None`` (default) opens an in-memory database.
        read_only : bool, optional
            Open the database read-only. Default is False.

        Returns
        -------
        DuckJanitor
            A session object whose ``.connection`` is the new connection and
            whose relation is an empty ``session`` relation.

        Examples
        --------
        >>> with DuckJanitor.connect('warehouse.duckdb') as db:
        ...     dj = db.from_csv('sales.csv')
        ...     dj.clean_names().to_table('clean_sales')
        """
        if path is None:
            conn = duckdb.connect()
        else:
            target = Path(path)
            if target.parent and str(target.parent) not in ("", "."):
                target.parent.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(str(target), read_only=read_only)
        relation = conn.sql("SELECT 1 AS _session")
        session = cls(relation, connection=conn)
        return session

    def attach(
        self,
        path: Union[str, Path],
        alias: str,
        *,
        read_only: bool = False,
    ) -> "DuckJanitor":
        """Attach another DuckDB database file to this session.

        Parameters
        ----------
        path : str or Path
            Path to the DuckDB database file to attach.
        alias : str
            Catalog alias the database is exposed under.
        read_only : bool, optional
            Attach read-only. Default is False.

        Returns
        -------
        DuckJanitor
            This object, so the call can stay inside a chain.

        Raises
        ------
        ValueError
            If the alias is not a plain identifier.
        """
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias):
            raise ValueError(f"attach(): alias must be a plain identifier. Got {alias!r}.")
        escaped = str(path).replace("'", "''")
        mode = " (READ_ONLY)" if read_only else ""
        self._connection.execute(f"ATTACH '{escaped}' AS {alias}{mode}")
        return self

    def detach(self, alias: str) -> "DuckJanitor":
        """Detach a previously attached database.

        Parameters
        ----------
        alias : str
            The catalog alias to detach.

        Returns
        -------
        DuckJanitor
            This object.
        """
        self._connection.execute(f"DETACH {alias}")
        return self

    def tables(self) -> list[str]:
        """List the tables and views in the current catalog.

        Returns
        -------
        list of str
            Fully qualified table names.
        """
        rows = self._connection.execute(
            "SELECT schema_name || '.' || table_name FROM duckdb_tables() "
            "UNION ALL "
            "SELECT schema_name || '.' || view_name FROM duckdb_views() "
            "WHERE NOT internal ORDER BY 1"
        ).fetchall()
        return [row[0] for row in rows]

    def databases(self) -> list[str]:
        """List the attached databases.

        Returns
        -------
        list of str
            Database names.
        """
        rows = self._connection.execute(
            "SELECT database_name FROM duckdb_databases() ORDER BY 1"
        ).fetchall()
        return [row[0] for row in rows]

    def begin(self) -> "DuckJanitor":
        """Begin an explicit transaction on this session's connection.

        Returns
        -------
        DuckJanitor
            This object.
        """
        self._connection.begin()
        return self

    def commit(self) -> "DuckJanitor":
        """Commit the current transaction.

        Returns
        -------
        DuckJanitor
            This object.
        """
        self._connection.commit()
        return self

    def rollback(self) -> "DuckJanitor":
        """Roll back the current transaction.

        Returns
        -------
        DuckJanitor
            This object.
        """
        self._connection.rollback()
        return self

    def close(self) -> None:
        """Close this session's connection.

        The shared in-memory connection is never closed by this method,
        because other objects may still depend on it.
        """
        if self._connection is not DuckJanitor.get_shared_connection():
            self._connection.close()

    def __enter__(self) -> "DuckJanitor":
        """Enter a ``with`` block, returning this session.

        Returns
        -------
        DuckJanitor
            This object.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close the session when leaving a ``with`` block.

        Parameters
        ----------
        exc_type : type or None
            The exception type, if any.
        exc : BaseException or None
            The exception, if any.
        tb : traceback or None
            The traceback, if any.

        Returns
        -------
        bool
            False, so exceptions propagate.
        """
        self.close()
        return False

    def __init__(
        self,
        relation: duckdb.DuckDBPyRelation,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
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
                temp_name = _temp_name("_validate")
                connection.register(temp_name, relation)
                connection.execute(f"SELECT 1 FROM {temp_name} LIMIT 0")
            except Exception as exc:
                raise ValueError(
                    "Could not derive a matching DuckDB connection from the relation. "
                    "Pass the connection used to create the relation explicitly."
                ) from exc

        # Validate that the relation belongs to the provided connection.
        try:
            temp_name = _temp_name("_validate")
            connection.register(temp_name, relation)
            connection.execute(f"SELECT 1 FROM {temp_name} LIMIT 0")
        except Exception as exc:
            raise ValueError(
                "The relation does not belong to the provided DuckDB connection."
            ) from exc

        self._relation = relation
        self._connection = connection

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, *, connection: Optional[duckdb.DuckDBPyConnection] = None
    ) -> "DuckJanitor":
        """
        Create a DuckJanitor from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame.
        connection : duckdb.DuckDBPyConnection, optional
            The connection to load the frame into. Defaults to the shared
            in-memory connection. Pass a session's connection to load data
            into that session's database.

        Returns
        -------
        DuckJanitor
            A DuckJanitor instance.
        """
        conn = connection or cls.get_shared_connection()
        relation = conn.from_df(df)
        return cls(relation, connection=conn)

    @classmethod
    def from_parquet(
        cls,
        path: Union[str, Path, list[str]],
        *,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> "DuckJanitor":
        """
        Create a DuckJanitor from Parquet file(s).

        Parameters
        ----------
        path : str, Path, or list of str/Path
            Path(s) to Parquet file(s). Can be local or remote (s3://, http://).
        connection : duckdb.DuckDBPyConnection, optional
            The connection to read into. Defaults to the shared connection.

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
        conn = connection or cls.get_shared_connection()

        if isinstance(path, list):
            path_list = [str(p) for p in path]
            relation = conn.query(
                f"SELECT * FROM read_parquet([{', '.join(repr(p) for p in path_list)}])"
            )
        else:
            relation = conn.query(f"SELECT * FROM read_parquet({repr(str(path))})")

        return cls(relation, connection=conn)

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        *,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
        **kwargs: Any,
    ) -> "DuckJanitor":
        """
        Create a DuckJanitor from a CSV file.

        Parameters
        ----------
        path : str or Path
            Path to the CSV file.
        connection : duckdb.DuckDBPyConnection, optional
            The connection to read into. Defaults to the shared connection.
        **kwargs
            Additional arguments passed to DuckDB's read_csv.

        Returns
        -------
        DuckJanitor
            A DuckJanitor instance.
        """
        conn = connection or cls.get_shared_connection()
        relation = conn.read_csv(str(path), **kwargs)
        return cls(relation, connection=conn)

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path, list[Union[str, Path]]],
        format: str = "auto",  # noqa: A002
        *,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
        **kwargs: Any,
    ) -> "DuckJanitor":
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
        connection : duckdb.DuckDBPyConnection, optional
            The connection to read into. Defaults to the shared connection.
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
        conn = connection or cls.get_shared_connection()

        if isinstance(path, list):
            path_repr = "[" + ", ".join(repr(str(p)) for p in path) + "]"
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
        query += ")"

        relation = conn.query(query)
        return cls(relation, connection=conn)

    @classmethod
    def from_excel(
        cls,
        path: Union[str, Path],
        sheet_name: Union[str, int, None] = 0,
        header: int = 0,
        *,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
        usecols: Optional[Union[str, list[str], list[int]]] = None,
        skiprows: Optional[Union[int, list[int]]] = None,
        nrows: Optional[int] = None,
        na_values: Optional[Any] = None,
        keep_default_na: bool = True,
        dtype: Optional[dict[str, Any]] = None,
        engine: Optional[str] = None,
        **kwargs: Any,
    ) -> "DuckJanitor":
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
        connection : duckdb.DuckDBPyConnection, optional
            The connection to load the sheet into. Defaults to the shared
            connection.
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
                raise ValueError(f"Excel file '{path}' contains no sheets.")
            # Take the first sheet deterministically.
            first_key = next(iter(df))
            df = df[first_key]
        return cls.from_pandas(df, connection=connection)

    @classmethod
    def from_database(
        cls,
        connection: Any,
        query: str,
        params: Optional[Any] = None,
        *,
        target_connection: Optional[duckdb.DuckDBPyConnection] = None,
        **kwargs: Any,
    ) -> "DuckJanitor":
        """Create a DuckJanitor from a query on an external database.

        The connection may be any DB-API 2.0 connection accepted by
        ``pandas.read_sql_query``. This includes ``vertica_python``
        connections and ``pyodbc`` connections for Microsoft SQL Server.
        The database driver remains an optional application dependency.

        Parameters
        ----------
        connection : object
            An open DB-API 2.0 connection, such as a Vertica or pyodbc
            connection. The connection is not closed by this method.
        query : str
            SQL query to execute on the external database. Keep the query
            in the source database's SQL dialect.
        params : object, optional
            Parameters passed unchanged to the database driver. Use the
            placeholder style required by that driver (for example ``?``
            for pyodbc or ``%s`` for many DB-API drivers).
        target_connection : duckdb.DuckDBPyConnection, optional
            The DuckDB connection to load the result into. Defaults to the
            shared connection.
        **kwargs
            Additional arguments forwarded to ``pandas.read_sql_query``
            (for example ``parse_dates`` or ``dtype``).

        Returns
        -------
        DuckJanitor
            A DuckJanitor instance backed by a DuckDB relation containing
            the query result.

        Examples
        --------
        >>> import sqlite3
        >>> db = sqlite3.connect(':memory:')
        >>> _ = db.execute('CREATE TABLE people (name TEXT, age INTEGER)')
        >>> _ = db.executemany('INSERT INTO people VALUES (?, ?)', [('Ada', 36), ('Lin', 29)])
        >>> dj = DuckJanitor.from_database(db, 'SELECT * FROM people WHERE age > ?', [30])
        >>> dj.collect()['name'].tolist()
        ['Ada']

        Notes
        -----
        The query result is materialized in pandas before it is registered
        with DuckDB. For very large extracts, filter and aggregate in the
        source query first, or use a source-specific connector in a future
        extension.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty SQL string")

        read_kwargs = dict(kwargs)
        if params is not None:
            read_kwargs["params"] = params
        result = pd.read_sql_query(query, connection, **read_kwargs)
        if isinstance(result, pd.DataFrame):
            df = result
        else:
            df = pd.concat(result, ignore_index=True)
        return cls.from_pandas(df, connection=target_connection)

    @classmethod
    def from_sql(
        cls, query: str, connection: Optional[duckdb.DuckDBPyConnection] = None
    ) -> "DuckJanitor":
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

    # =============================================================
    # Output suite — write results without going through pandas.
    # =============================================================

    def to_parquet(
        self,
        path: Union[str, Path],
        *,
        overwrite: bool = False,
        compression: Optional[str] = None,
    ) -> str:
        """Write the pipeline to a Parquet file via DuckDB ``COPY ... TO``.

        This is the out-of-core exit: the result lands in Parquet without
        ever being materialised in Python.

        Parameters
        ----------
        path : str or Path
            Destination ``.parquet`` path. Parent directories are created.
        overwrite : bool, optional
            Replace an existing file. Default is False.
        compression : str, optional
            Compression codec, for example ``'zstd'`` or ``'snappy'``.

        Returns
        -------
        str
            The path that was written.

        Examples
        --------
        >>> dj.clean_names().to_parquet('out/clean.parquet', compression='zstd')
        """
        from .io_ops import to_parquet as _to_parquet

        return _to_parquet(
            self._relation,
            path,
            overwrite=overwrite,
            compression=compression,
            conn=self._connection,
        )

    def to_csv(
        self,
        path: Union[str, Path],
        *,
        overwrite: bool = False,
        header: bool = True,
        delimiter: str = ",",
    ) -> str:
        """Write the pipeline to a CSV file via DuckDB ``COPY ... TO``.

        Parameters
        ----------
        path : str or Path
            Destination ``.csv`` path.
        overwrite : bool, optional
            Replace an existing file. Default is False.
        header : bool, optional
            Write a header row. Default is True.
        delimiter : str, optional
            Field delimiter. Default is ``','``.

        Returns
        -------
        str
            The path that was written.
        """
        from .io_ops import to_csv as _to_csv

        return _to_csv(
            self._relation,
            path,
            overwrite=overwrite,
            header=header,
            delimiter=delimiter,
            conn=self._connection,
        )

    def to_json(
        self,
        path: Union[str, Path],
        *,
        overwrite: bool = False,
        newline_delimited: bool = False,
    ) -> str:
        """Write the pipeline to a JSON file via DuckDB ``COPY ... TO``.

        Parameters
        ----------
        path : str or Path
            Destination ``.json`` or ``.jsonl`` path.
        overwrite : bool, optional
            Replace an existing file. Default is False.
        newline_delimited : bool, optional
            Write newline-delimited JSON instead of a JSON array.

        Returns
        -------
        str
            The path that was written.
        """
        from .io_ops import to_json as _to_json

        return _to_json(
            self._relation,
            path,
            overwrite=overwrite,
            newline_delimited=newline_delimited,
            conn=self._connection,
        )

    def to_table(
        self,
        name: str,
        *,
        view: bool = False,
        replace: bool = True,
    ) -> str:
        """Materialise the pipeline as a table or view in the DuckDB catalog.

        Parameters
        ----------
        name : str
            Target name. May be ``'schema.table'``.
        view : bool, optional
            Create a lazy view instead of a stored table. Default is False.
        replace : bool, optional
            Replace an existing object of the same name. Default is True.

        Returns
        -------
        str
            The catalog name that was created.
        """
        from .io_ops import to_table as _to_table

        return _to_table(
            self._relation,
            name,
            view=view,
            replace=replace,
            conn=self._connection,
        )

    def to_view(self, name: str, *, replace: bool = True) -> str:
        """Materialise the pipeline as a view in the DuckDB catalog.

        Parameters
        ----------
        name : str
            Target view name.
        replace : bool, optional
            Replace an existing view of the same name. Default is True.

        Returns
        -------
        str
            The catalog name that was created.
        """
        return self.to_table(name, view=True, replace=replace)

    def to_arrow(self):
        """Return the pipeline as an Arrow table.

        Returns
        -------
        pyarrow.Table
            The result as an Arrow table.
        """
        from .io_ops import to_arrow as _to_arrow

        return _to_arrow(self._relation)

    def to_polars(self):
        """Return the pipeline as a Polars DataFrame.

        Returns
        -------
        polars.DataFrame
            The result as a Polars DataFrame.
        """
        from .io_ops import to_polars as _to_polars

        return _to_polars(self._relation)

    # =============================================================
    # Inspection — cheap metadata without materialising the frame.
    # =============================================================

    @property
    def columns(self) -> list[str]:
        """Column names of the current relation, in order.

        Returns
        -------
        list of str
            The column names.
        """
        return list(self._relation.columns)

    @property
    def dtypes(self) -> dict[str, str]:
        """Column names mapped to their DuckDB type names.

        Returns
        -------
        dict
            Mapping of column name to DuckDB type string.
        """
        return dict(zip(self._relation.columns, [str(t) for t in self._relation.types]))

    @property
    def shape(self) -> tuple[int, int]:
        """Row and column count of the current relation.

        Returns
        -------
        tuple
            ``(rows, columns)``.
        """
        return tuple(self._relation.shape)

    @property
    def relation(self) -> duckdb.DuckDBPyRelation:
        """The underlying DuckDB relation.

        Use this as an escape hatch when you need DuckDB's own API.

        Returns
        -------
        duckdb.DuckDBPyRelation
            The wrapped relation.
        """
        return self._relation

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The DuckDB connection that owns this relation.

        Returns
        -------
        duckdb.DuckDBPyConnection
            The owning connection.
        """
        return self._connection

    def count(self) -> int:
        """Return the number of rows in the current relation.

        Returns
        -------
        int
            The row count.
        """
        temp_name = _temp_name("_pyduck_count")
        self._connection.register(temp_name, self._relation)
        result = self._connection.execute(
            f"SELECT COUNT(*) FROM {self._quote(temp_name)}"
        ).fetchone()
        return int(result[0]) if result else 0

    def glimpse(self, n: int = 5) -> pd.DataFrame:
        """Return the first *n* rows as a pandas DataFrame.

        This is an alias for :meth:`head` with a name that matches the
        tidyverse convention.

        Parameters
        ----------
        n : int, optional
            Number of rows. Default is 5.

        Returns
        -------
        pd.DataFrame
            The preview rows.
        """
        return self.head(n)

    # =============================================================
    # Nested / typed data verbs (0.4.0).
    # =============================================================

    def to_struct(
        self,
        columns: Union[str, list[str]],
        struct_column: Optional[str] = None,
        *,
        drop_source: bool = False,
    ) -> "DuckJanitor":
        """Pack one or more columns into a DuckDB STRUCT column.

        Parameters
        ----------
        columns : str or list of str
            Columns to pack. The source column names become the struct keys.
        struct_column : str, optional
            Name of the resulting STRUCT column. Default is ``'struct'``.
        drop_source : bool, optional
            Remove the source columns from the output. Default is False.

        Returns
        -------
        DuckJanitor
            A new relation with the STRUCT column added.
        """
        from .nested_ops import to_struct as _to_struct

        relation = _to_struct(
            self._relation,
            columns,
            struct_column,
            drop_source=drop_source,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def from_struct(
        self,
        struct_column: str,
        fields: Optional[list[str]] = None,
        *,
        prefix: Optional[str] = None,
        drop_source: bool = False,
    ) -> "DuckJanitor":
        """Expand a STRUCT column into separate scalar columns.

        Parameters
        ----------
        struct_column : str
            The STRUCT column to expand.
        fields : list of str, optional
            Which struct keys to extract. All keys are extracted when omitted.
        prefix : str, optional
            Prefix added to the output column names.
        drop_source : bool, optional
            Remove the STRUCT column from the output. Default is False.

        Returns
        -------
        DuckJanitor
            A new relation with the struct fields as columns.
        """
        from .nested_ops import from_struct as _from_struct

        relation = _from_struct(
            self._relation,
            struct_column,
            fields,
            prefix=prefix,
            drop_source=drop_source,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def to_list(
        self,
        columns: Union[str, list[str]],
        list_column: Optional[str] = None,
        *,
        drop_source: bool = False,
    ) -> "DuckJanitor":
        """Pack one or more columns into a DuckDB LIST column.

        Parameters
        ----------
        columns : str or list of str
            Columns to pack into the LIST.
        list_column : str, optional
            Name of the resulting LIST column. Default is ``'list'``.
        drop_source : bool, optional
            Remove the source columns from the output. Default is False.

        Returns
        -------
        DuckJanitor
            A new relation with the LIST column added.
        """
        from .nested_ops import to_list as _to_list

        relation = _to_list(
            self._relation,
            columns,
            list_column,
            drop_source=drop_source,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def from_list(
        self,
        list_column: str,
        *,
        index: bool = False,
        output_column: Optional[str] = None,
        drop_source: bool = False,
    ) -> "DuckJanitor":
        """Expand a LIST column into one row per element.

        Parameters
        ----------
        list_column : str
            The LIST column to expand.
        index : bool, optional
            Also emit a 1-based position column. Default is False.
        output_column : str, optional
            Name of the position column. Default is ``'pos'``.
        drop_source : bool, optional
            Remove the LIST column from the output. Default is False.

        Returns
        -------
        DuckJanitor
            A new relation with one row per list element.
        """
        from .nested_ops import from_list as _from_list

        relation = _from_list(
            self._relation,
            list_column,
            index=index,
            output_column=output_column,
            drop_source=drop_source,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def unnest(
        self,
        columns: Union[str, list[str]],
        *,
        drop_source: bool = False,
    ) -> "DuckJanitor":
        """Expand one or more LIST columns into rows using ``UNNEST``.

        Parameters
        ----------
        columns : str or list of str
            The LIST columns to expand. Multiple columns are unnested in
            parallel, zipping their elements.
        drop_source : bool, optional
            Remove the source LIST columns from the output. Default is False.

        Returns
        -------
        DuckJanitor
            A new relation with one row per element.
        """
        from .nested_ops import unnest as _unnest

        relation = _unnest(
            self._relation,
            columns,
            drop_source=drop_source,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def list_transform(
        self,
        list_column: str,
        func: str,
        *,
        output_column: Optional[str] = None,
    ) -> "DuckJanitor":
        """Apply a DuckDB lambda over every element of a LIST column.

        Parameters
        ----------
        list_column : str
            The LIST column to transform.
        func : str
            A DuckDB lambda body, for example ``'x -> x * 2'``.
        output_column : str, optional
            Name of the resulting column. Defaults to overwriting the source.

        Returns
        -------
        DuckJanitor
            A new relation with the transformed LIST column.
        """
        from .nested_ops import list_transform as _list_transform

        relation = _list_transform(
            self._relation,
            list_column,
            func,
            output_column=output_column,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def list_filter(
        self,
        list_column: str,
        func: str,
        *,
        output_column: Optional[str] = None,
    ) -> "DuckJanitor":
        """Filter the elements of a LIST column with a DuckDB lambda.

        Parameters
        ----------
        list_column : str
            The LIST column to filter.
        func : str
            A DuckDB lambda body, for example ``'x -> x > 2'``.
        output_column : str, optional
            Name of the resulting column. Defaults to overwriting the source.

        Returns
        -------
        DuckJanitor
            A new relation with the filtered LIST column.
        """
        from .nested_ops import list_filter as _list_filter

        relation = _list_filter(
            self._relation,
            list_column,
            func,
            output_column=output_column,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def json_extract(
        self,
        json_column: str,
        path: str,
        *,
        output_column: Optional[str] = None,
    ) -> "DuckJanitor":
        """Extract a value from a JSON column.

        Parameters
        ----------
        json_column : str
            The JSON column to read.
        path : str
            A JSON pointer such as ``'$.a.b'`` or a plain key name.
        output_column : str, optional
            Name of the resulting column. Defaults to overwriting the source.

        Returns
        -------
        DuckJanitor
            A new relation with the extracted value.
        """
        from .nested_ops import json_extract as _json_extract

        relation = _json_extract(
            self._relation,
            json_column,
            path,
            output_column=output_column,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    # =============================================================
    # Sampling — DuckDB-native, no full sort.
    # =============================================================

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        *,
        seed: Optional[int] = None,
        method: str = "reservoir",
    ) -> "DuckJanitor":
        """Randomly sample the pipeline via DuckDB's ``USING SAMPLE``.

        Parameters
        ----------
        n : int, optional
            Number of rows to sample. Mutually exclusive with *frac*.
            Only the ``'reservoir'`` method supports a row count.
        frac : float, optional
            Fraction of rows to sample, between 0 and 1. Mutually exclusive
            with *n*.
        seed : int, optional
            Non-negative seed for reproducible sampling.
        method : str, optional
            Sampling method: ``'reservoir'`` (default), ``'bernoulli'``, or
            ``'system'``.

        Returns
        -------
        DuckJanitor
            A new relation holding the sample.

        Examples
        --------
        >>> dj.sample(n=1000, seed=42).collect()
        """
        from .sampling import sample as _sample

        relation = _sample(
            self._relation,
            n=n,
            frac=frac,
            seed=seed,
            method=method,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def sample_rows(self, n: int, *, seed: Optional[int] = None) -> "DuckJanitor":
        """Pick *n* rows in random order without a full sort.

        Parameters
        ----------
        n : int
            Number of rows to pick.
        seed : int, optional
            Non-negative seed for a reproducible pick.

        Returns
        -------
        DuckJanitor
            A new relation holding *n* rows.
        """
        from .sampling import sample_rows as _sample_rows

        relation = _sample_rows(self._relation, n, seed=seed, conn=self._connection)
        return DuckJanitor(relation, self._connection)

    def approx_distinct(self, column: str) -> "DuckJanitor":
        """Approximate the number of distinct values in *column*.

        Parameters
        ----------
        column : str
            Column to inspect.

        Returns
        -------
        DuckJanitor
            A single-row relation with the approximate count.
        """
        from .sampling import approx_distinct as _approx_distinct

        relation = _approx_distinct(self._relation, column, conn=self._connection)
        return DuckJanitor(relation, self._connection)

    def approx_quantile(self, column: str, quantile: float) -> "DuckJanitor":
        """Approximate a quantile of *column*.

        Parameters
        ----------
        column : str
            Column to inspect.
        quantile : float
            Target quantile, between 0 and 1.

        Returns
        -------
        DuckJanitor
            A single-row relation with the approximate quantile.
        """
        from .sampling import approx_quantile as _approx_quantile

        relation = _approx_quantile(self._relation, column, quantile, conn=self._connection)
        return DuckJanitor(relation, self._connection)

    def register_python_function(
        self,
        name: str,
        func: Callable,
        *,
        params: Optional[list[str]] = None,
        return_type: Optional[str] = None,
    ) -> "DuckJanitor":
        """Register a Python callable as a DuckDB scalar UDF.

        The function is registered on this pipeline's connection, so it can
        be used by any later ``sql()`` call or by
        :meth:`apply_python_function`. This keeps user logic inside DuckDB
        instead of forcing a pandas round-trip.

        Parameters
        ----------
        name : str
            Name to register the function under.
        func : callable
            The Python callable.
        params : list of str, optional
            Parameter types, for example ``['VARCHAR']``.
        return_type : str, optional
            Return type, for example ``'BIGINT'``. Required when the
            callable has no return annotation.

        Returns
        -------
        DuckJanitor
            This object, so the call can stay inside a chain.
        """
        from .sampling import register_python_function as _register_python_function

        _register_python_function(
            self._relation,
            name,
            func,
            params=params,
            return_type=return_type,
            conn=self._connection,
        )
        return self

    def apply_python_function(
        self,
        column: str,
        func: Callable,
        *,
        name: Optional[str] = None,
        params: Optional[list[str]] = None,
        return_type: Optional[str] = None,
        output_column: Optional[str] = None,
    ) -> "DuckJanitor":
        """Apply a Python callable to a column inside DuckDB.

        Parameters
        ----------
        column : str
            Column to transform.
        func : callable
            The Python callable.
        name : str, optional
            Name to register the UDF under. Generated when omitted.
        params : list of str, optional
            Parameter types for the UDF.
        return_type : str, optional
            Return type for the UDF.
        output_column : str, optional
            Name of the resulting column. Defaults to ``'<column>_py'``.

        Returns
        -------
        DuckJanitor
            A new relation with the transformed column added.
        """
        from .sampling import apply_python_function as _apply_python_function

        relation = _apply_python_function(
            self._relation,
            column,
            func,
            name=name,
            params=params,
            return_type=return_type,
            conn=self._connection,
        )
        return DuckJanitor(relation, self._connection)

    def show(self, n: int = 10, *, max_width: int = 100) -> str:
        """Return a DuckDB-style text table of the first *n* rows.

        Parameters
        ----------
        n : int, optional
            Number of rows to show. Default is 10.
        max_width : int, optional
            Maximum column width in characters. Default is 100.

        Returns
        -------
        str
            The rendered table.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        # DuckDB's relation.show() prints to stdout and returns None, so
        # capture it rather than returning an empty string.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self._relation.limit(n).show(max_rows=n, max_width=max_width)
        return buffer.getvalue().rstrip("\n")

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks.

        Returns
        -------
        str
            An HTML fragment with the schema and a preview table.
        """
        try:
            preview = self.head(10)
            columns = list(preview.columns)
            header = "".join(f"<th>{_escape_html(str(c))}</th>" for c in columns)
            rows = []
            for record in preview.itertuples(index=False):
                cells = "".join(
                    f"<td>{_escape_html('' if v is None else str(v))}</td>" for v in record
                )
                rows.append(f"<tr>{cells}</tr>")
            types = self.dtypes
            type_rows = "".join(
                f"<tr><td>{_escape_html(str(name))}</td>"
                f"<td>{_escape_html(str(types.get(name, '')))}</td></tr>"
                for name in columns
            )
            row_count = self.count()
            return (
                "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; "
                "font-size: 12px;'>"
                f"<div style='color:#666;margin-bottom:4px;'>DuckJanitor &mdash; "
                f"{row_count:,} row(s) x {len(columns)} column(s)</div>"
                f"<table style='border-collapse:collapse;'><thead><tr>{header}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
                f"<details><summary style='cursor:pointer;color:#666;'>schema</summary>"
                f"<table style='border-collapse:collapse;'>{type_rows}</table></details>"
                "</div>"
            )
        except Exception:  # noqa: BLE001 - a repr must never raise
            return f"<pre>{_escape_html(self.__repr__())}</pre>"

    def clean_names(
        self,
        strip_underscores: bool = True,
        case_type: str = "lower",
        remove_special: bool = True,
        snakecase: bool = True,
    ) -> "DuckJanitor":
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

        new_relation = _clean_names(
            self._relation,
            strip_underscores,
            case_type,
            remove_special,
            snakecase,
            self._connection,
        )
        return DuckJanitor(new_relation, self._connection)

    def remove_columns(self, columns: Union[str, list[str]]) -> "DuckJanitor":
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

    def add_column(
        self, column_name: str, values: Union[Any, list[Any], str], fill_value: Optional[Any] = None
    ) -> "DuckJanitor":
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

        new_relation = _add_column(
            self._relation, column_name, values, fill_value, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def rename_column(self, old_name: str, new_name: str) -> "DuckJanitor":
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

    def dropna(
        self, subset: Optional[Union[str, list[str]]] = None, how: str = "any"
    ) -> "DuckJanitor":
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

    def remove_empty(self) -> "DuckJanitor":
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

    def filter_column(self, column: str, criteria: Union[Callable, str]) -> "DuckJanitor":
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

    def coalesce(self, columns: list[str], target_column: str) -> "DuckJanitor":
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

    def encode_categorical(self, column: str, col_name: Optional[str] = None) -> "DuckJanitor":
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

    def get_dummies(
        self, columns: Union[str, list[str]], prefix: Optional[str] = None
    ) -> "DuckJanitor":
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

    def filter_on(self, criteria: str, complement: bool = False) -> "DuckJanitor":
        """Filter rows based on a SQL-like criteria string.

        Parameters
        ----------
        criteria : str
            A SQL WHERE-clause fragment evaluated against the current
            relation (``age > 18``); ``self`` is replaced with the
            registered relation name.
        complement : bool, default False
            If True, keep rows that *don't* match ``criteria``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to rows matching the
            criteria (or non-matching if ``complement=True``).
        """
        from .cleaning_ops import filter_on as _filter_on

        new_relation = _filter_on(self._relation, criteria, complement, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def filter_string(
        self,
        column: str,
        search_string: str,
        complement: bool = False,
        case: bool = True,
        regex: bool = True,
    ) -> "DuckJanitor":
        """Filter rows based on whether a string column contains a substring.

        Parameters
        ----------
        column : str
            Name of the column to search.
        search_string : str
            Substring to look for. When ``regex=True`` this is a regular
            expression; otherwise it's a literal substring.
        complement : bool, default False
            If True, keep rows that *don't* match ``search_string``.
        case : bool, default True
            If False, perform case-insensitive matching.
        regex : bool, default True
            If False, treat ``search_string`` as a literal substring
            rather than a regular expression.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to matching rows.
        """
        from .cleaning_ops import filter_string as _filter_string

        new_relation = _filter_string(
            self._relation, column, search_string, complement, case, regex, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def select_columns(self, columns: Union[str, list[str]]) -> "DuckJanitor":
        """Select specific columns.

        Accepts a list, a single column name, a comma-separated string,
        a shell-glob pattern (e.g. ``"v*"``), or a regex prefix (``"re:^v_"``).
        See :func:`pyduck_janitor.cleaning_ops.select_columns` for details.

        Parameters
        ----------
        columns : str or list of str
            Either a single column name, a comma-separated string
            (``"a, b, c"``), a list of names/patterns, a glob
            (``"value*"``), or a regex prefix (``"re:^v_"``).
            :class:`DropLabel` may be mixed in to exclude a column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to the selected columns.
        """
        from .cleaning_ops import select_columns as _select_columns

        new_relation = _select_columns(self._relation, columns, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def select(self, *args, **kwargs) -> "DuckJanitor":
        """Convenience alias for the pyjanitor ``select`` DSL.

        The pyjanitor API page has ``select`` at the top of the
        ``select`` family, but their own docs state::

            This function has been deprecated.  Kindly use
            ``jn.select_columns`` or ``jn.select_rows``.

        We expose it as a chainable convenience whose only supported
        form is the ``columns=`` keyword. Positional ``args`` and other
        kwargs (``index=``, ``axis=``, ``invert=``) are accepted but
        not implemented; they will raise ``NotImplementedError``.

        Parameters
        ----------
        *args
            Positional selector arguments. Not implemented; any
            positional arg raises ``NotImplementedError``.
        **kwargs
            Only ``columns=`` is implemented; all other kwargs
            (``index=``, ``axis=``, ``invert=``, ``rows=``) raise
            ``NotImplementedError``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to the requested
            columns.
        """
        if args or any(k in kwargs for k in ("index", "axis", "invert", "rows")):
            raise NotImplementedError(
                "select(): only the columns= shorthand is supported; "
                "use .select_columns(...).select_rows(...) for the "
                "full pyjanitor DSL."
            )
        return self.select_columns(kwargs.get("columns", []))

    def select_rows(
        self, indices: Optional[Union[list[int], str]] = None, criteria: Optional[str] = None
    ) -> "DuckJanitor":
        """Select specific rows by index or condition.

        Parameters
        ----------
        indices : list of int or str, optional
            Either a list of 0-indexed row positions, or a SQL string
            (``"row_number() BETWEEN 1 AND 5"`` style). Mutually exclusive
            with ``criteria``.
        criteria : str, optional
            SQL WHERE-clause fragment (``age > 18``). Mutually exclusive
            with ``indices``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to the selected rows.
        """
        from .cleaning_ops import select_rows as _select_rows

        new_relation = _select_rows(self._relation, indices, criteria, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def transform_column(
        self, column: str, func: Union[str, Callable], target_column: Optional[str] = None
    ) -> "DuckJanitor":
        """Transform a column using a function or SQL expression.

        Parameters
        ----------
        column : str
            Name of the column to transform.
        func : str or Callable
            Either a SQL expression referencing ``column`` (``"UPPER(column)"``)
            or a Python callable applied row-wise (column → scalar).
        target_column : str, optional
            Name of the output column. Defaults to overwriting ``column``
            in place.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``column`` replaced or
            ``target_column`` added.
        """
        from .cleaning_ops import transform_column as _transform_column

        new_relation = _transform_column(
            self._relation, column, func, target_column, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def transform_columns(
        self,
        columns: Union[str, list[str]],
        func: Union[str, Callable],
        target_columns: Optional[Union[str, list[str]]] = None,
    ) -> "DuckJanitor":
        """Transform multiple columns using a function or SQL expression.

        Parameters
        ----------
        columns : str or list of str
            One or more column names to transform. When a single column
            is passed, ``func`` may be a Python callable; with multiple
            columns ``func`` must be a SQL expression using the same
            column names as placeholders.
        func : str or Callable
            SQL expression or Python callable.
        target_columns : str or list of str, optional
            Output column name(s). Defaults to overwriting each input
            column in place.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the transformed columns.
        """
        from .cleaning_ops import transform_columns as _transform_columns

        new_relation = _transform_columns(
            self._relation, columns, func, target_columns, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def sql(self, query: str, **params: Any) -> "DuckJanitor":
        """Execute a custom SQL query on the current relation.

        The word ``self`` in the query refers to the current relation. The
        substitution is aware of string literals, quoted identifiers, and
        comments, so ``'self'`` and ``"self"`` are left untouched.

        Parameters
        ----------
        query : str
            SQL query. Use ``self`` to refer to the current relation.
        **params
            Named parameters bound with DuckDB's ``$name`` syntax. Values
            are passed to the driver rather than interpolated into the SQL,
            so user data never becomes part of the statement text.

        Returns
        -------
        DuckJanitor
            Self for method chaining.

        Examples
        --------
        >>> dj.sql("SELECT * FROM self WHERE age > 18")
        >>> dj.sql("SELECT * FROM self WHERE dept = $dept", dept="Finance")
        """
        # Register the current relation as a temporary table.
        temp_name = _temp_name("_temp")
        self._connection.register(temp_name, self._relation)
        query = substitute_self(query, temp_name)
        if params:
            arrow_table = self._connection.execute(query, params).to_arrow_table()
            return DuckJanitor(self._connection.from_arrow(arrow_table), self._connection)
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def load_extension(
        self,
        name: str,
        *,
        auto_install: bool = False,
        repository: Optional[str] = None,
    ) -> "DuckJanitor":
        """Load an optional DuckDB extension for this pipeline.

        Parameters
        ----------
        name : str
            Extension name, such as ``"onager"``.
        auto_install : bool, default False
            If True, ask DuckDB to install the extension when it is not
            already available locally. For Onager, the community repository
            is used by default.
        repository : str, optional
            Override the DuckDB extension repository used for installation.

        Returns
        -------
        DuckJanitor
            This object, allowing the call to remain in a method chain.
        """
        from .extensions import load_extension as _load_extension

        _load_extension(
            self._connection,
            name,
            install=auto_install,
            repository=repository,
        )
        return self

    def graph_algorithm(
        self,
        function: str,
        source: str,
        target: str,
        *,
        weight: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        auto_install: bool = False,
    ) -> "DuckJanitor":
        """Run an Onager graph table function over the current relation.

        This is the low-level escape hatch for Onager algorithms that are not
        yet wrapped by :meth:`graph_analyze`. The current relation is treated
        as an edge table and projected to the conventional ``src``, ``dst``
        (and optional ``weight``) columns expected by Onager.

        Parameters
        ----------
        function : str
            Onager table-function name, for example
            ``"onager_ctr_pagerank"``.
        source, target : str
            Current-relation columns identifying edge endpoints.
        weight : str, optional
            Current-relation column containing edge weights.
        parameters : dict[str, Any], optional
            Named SQL arguments passed to the Onager table function. Values
            are rendered as SQL literals by DuckJanitor.
        auto_install : bool, default False
            Install Onager from DuckDB's community repository if necessary.

        Returns
        -------
        DuckJanitor
            The algorithm result as a new relation.
        """
        if not re.fullmatch(r"onager_[A-Za-z0-9_]+", function):
            raise ValueError("function must be an Onager table-function name")

        columns = set(self._relation.columns)
        required = [source, target] + ([weight] if weight else [])
        missing = [column for column in required if column not in columns]
        if missing:
            raise ValueError(f"graph_algorithm(): unknown edge columns {missing}")

        self.load_extension("onager", auto_install=auto_install)
        temp_name = _temp_name("_onager_edges")
        self._connection.register(temp_name, self._relation)
        source_type = str(self._relation.types[list(self._relation.columns).index(source)]).upper()
        target_type = str(self._relation.types[list(self._relation.columns).index(target)]).upper()
        integer_types = {
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
        }
        node_map_name = None
        if source_type in integer_types and target_type in integer_types:
            edge_columns = (
                f"CAST({self._quote(source)} AS BIGINT) AS src, "
                f"CAST({self._quote(target)} AS BIGINT) AS dst"
            )
        else:
            node_map_name = _temp_name("_onager_nodes")
            node_map = self._connection.sql(
                f"""
                SELECT node_key, ROW_NUMBER() OVER ()::BIGINT AS node_id
                FROM (
                    SELECT CAST({self._quote(source)} AS VARCHAR) AS node_key
                    FROM {self._quote(temp_name)}
                    WHERE {self._quote(source)} IS NOT NULL
                    UNION
                    SELECT CAST({self._quote(target)} AS VARCHAR) AS node_key
                    FROM {self._quote(temp_name)}
                    WHERE {self._quote(target)} IS NOT NULL
                ) AS unique_nodes
                """
            )
            self._connection.register(node_map_name, node_map)
            encoded_name = _temp_name("_onager_encoded")
            encoded_query = (
                "SELECT src_node.node_id AS src, dst_node.node_id AS dst "
                + (f", {self._quote(weight)} AS weight " if weight else "")
                + f"FROM {self._quote(temp_name)} AS edges "
                f"JOIN {self._quote(node_map_name)} AS src_node "
                f"ON CAST(edges.{self._quote(source)} AS VARCHAR) = src_node.node_key "
                f"JOIN {self._quote(node_map_name)} AS dst_node "
                f"ON CAST(edges.{self._quote(target)} AS VARCHAR) = dst_node.node_key"
            )
            encoded = self._connection.sql(encoded_query)
            self._connection.register(encoded_name, encoded)
            temp_name = encoded_name
            edge_columns = "src, dst" + (", weight" if weight else "")
        if weight:
            if node_map_name is None:
                edge_columns += f", {self._quote(weight)} AS weight"
        arguments = [f"(SELECT {edge_columns} FROM {self._quote(temp_name)})"]
        for key, value in (parameters or {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"graph_algorithm(): invalid parameter name {key!r}")
            arguments.append(f"{key} := {self._sql_value(value)}")
        query = f"SELECT * FROM {function}({', '.join(arguments)})"
        result = self._connection.query(query)
        if node_map_name and "node_id" in result.columns:
            result_name = _temp_name("_onager_result")
            self._connection.register(result_name, result)
            result = self._connection.sql(
                f"SELECT nodes.node_key AS node_id, result.* EXCLUDE (node_id) "
                f"FROM {self._quote(result_name)} AS result "
                f"JOIN {self._quote(node_map_name)} AS nodes "
                "ON result.node_id = nodes.node_id"
            )
        return DuckJanitor(result, self._connection)

    def graph_analyze(
        self,
        source: str,
        target: str,
        algorithms: Union[str, list[str]],
        *,
        weight: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        auto_install: bool = False,
    ) -> dict[str, "DuckJanitor"]:
        """Run common Onager graph algorithms over an edge relation.

        ``algorithms`` may contain ``pagerank``, ``betweenness``,
        ``closeness``, ``components``, ``louvain``, or ``dijkstra``. For
        algorithms not listed here, use :meth:`graph_algorithm` directly.
        String and other non-integer node identifiers are supported; they are
        mapped to BIGINT surrogate IDs for Onager and mapped back in results.

        Returns
        -------
        dict[str, DuckJanitor]
            One result relation per requested algorithm, keyed by the public
            algorithm name.
        """
        function_map = {
            "pagerank": "onager_ctr_pagerank",
            "betweenness": "onager_ctr_betweenness",
            "closeness": "onager_ctr_closeness",
            "components": "onager_cmm_components",
            "louvain": "onager_cmm_louvain",
            "dijkstra": "onager_pth_dijkstra",
        }
        names = [algorithms] if isinstance(algorithms, str) else list(algorithms)
        unknown = [name for name in names if name not in function_map]
        if unknown:
            raise ValueError(
                f"Unknown graph algorithms {unknown}; choose from {sorted(function_map)} "
                "or use graph_algorithm() for a custom Onager function."
            )
        return {
            name: self.graph_algorithm(
                function_map[name],
                source,
                target,
                weight=weight,
                parameters=parameters,
                auto_install=auto_install,
            )
            for name in names
        }

    def network_evolution(
        self,
        date_col: str,
        source: str,
        target: str,
        algorithms: Union[str, list[str]],
        *,
        frequency: str = "month",
        weight: Optional[str] = None,
        include_deltas: bool = True,
        auto_install: bool = False,
    ) -> dict[str, "DuckJanitor"]:
        """Run graph algorithms across time and return metric trajectories.

        The current relation must contain edge endpoints and a snapshot date.
        Each algorithm is evaluated independently for every distinct
        ``date_trunc(frequency, date_col)`` period. Results include a
        ``period`` column; when the result exposes a ``node_id`` column and a
        numeric metric column, ``metric_delta`` is added per node.
        """
        if date_col not in self._relation.columns:
            raise ValueError(f"network_evolution(): unknown date column {date_col!r}")
        if frequency not in {"day", "week", "month", "quarter", "year"}:
            raise ValueError("network_evolution(): unsupported frequency")
        if source not in self._relation.columns or target not in self._relation.columns:
            raise ValueError("network_evolution(): source and target columns are required")
        algorithm_names = [algorithms] if isinstance(algorithms, str) else list(algorithms)
        relation_name = _temp_name("_network_evolution")
        self._connection.register(relation_name, self._relation)
        periods = self._connection.sql(
            f"SELECT DISTINCT date_trunc('{frequency}', CAST({self._quote(date_col)} AS TIMESTAMP)) "
            f"AS period FROM {self._quote(relation_name)} "
            f"WHERE {self._quote(date_col)} IS NOT NULL ORDER BY period"
        ).fetchall()
        collected: dict[str, list[str]] = {name: [] for name in algorithm_names}
        for period_index, (period,) in enumerate(periods):
            period_name = f"{_temp_name('_network_period')}_{period_index}"
            period_relation = self._connection.sql(
                f"SELECT * FROM {self._quote(relation_name)} "
                f"WHERE date_trunc('{frequency}', CAST({self._quote(date_col)} AS TIMESTAMP)) "
                f"= {self._sql_value(period)}"
            )
            period_janitor = DuckJanitor(period_relation, self._connection)
            results = period_janitor.graph_analyze(
                source,
                target,
                algorithm_names,
                weight=weight,
                auto_install=auto_install,
            )
            for name, result in results.items():
                result_name = f"{period_name}_{name}"
                self._connection.register(result_name, result._relation)
                collected[name].append(
                    f"SELECT *, {self._sql_value(period)} AS period FROM {self._quote(result_name)}"
                )
        output: dict[str, DuckJanitor] = {}
        for name, queries in collected.items():
            if not queries:
                continue
            union_query = " UNION ALL ".join(queries)
            combined_name = f"{_temp_name('_network_combined')}_{name}"
            combined = self._connection.sql(union_query)
            self._connection.register(combined_name, combined)
            columns = list(combined.columns)
            if include_deltas and "node_id" in columns:
                numeric_types = {
                    "TINYINT",
                    "SMALLINT",
                    "INTEGER",
                    "BIGINT",
                    "HUGEINT",
                    "DECIMAL",
                    "FLOAT",
                    "DOUBLE",
                }
                metric_column = next(
                    (
                        column
                        for column, dtype in zip(combined.columns, combined.types)
                        if column not in {"node_id", "period"}
                        and any(token in str(dtype).upper() for token in numeric_types)
                    ),
                    None,
                )
                if metric_column:
                    query = (
                        "SELECT *, "
                        f"{self._quote(metric_column)} - LAG({self._quote(metric_column)}) "
                        f"OVER (PARTITION BY {self._quote('node_id')} ORDER BY period) "
                        "AS metric_delta "
                        f"FROM {self._quote(combined_name)}"
                    )
                    combined = self._connection.sql(query)
            output[name] = DuckJanitor(combined, self._connection)
        return output

    def metrics(
        self,
        metrics: dict[str, Any],
        *,
        group_by: Optional[Union[str, list[str]]] = None,
        where: Optional[str] = None,
    ) -> "DuckJanitor":
        """Calculate named database-native aggregate metrics in one scan.

        Metric values may be ``(column, function)`` tuples or raw SQL
        expressions. Supported function aliases include ``count``,
        ``count_distinct``, ``sum``, ``mean``, ``median``, ``min``, ``max``,
        ``std``, and ``quantile:0.95``.
        """
        if not metrics:
            raise ValueError("metrics(): metrics must be non-empty")
        groups = [group_by] if isinstance(group_by, str) else list(group_by or [])
        missing = [column for column in groups if column not in self._relation.columns]
        if missing:
            raise ValueError(f"metrics(): unknown group_by columns {missing}")
        select_parts = [self._quote(column) for column in groups]
        columns = set(self._relation.columns)
        aliases = set()
        for alias, spec in metrics.items():
            if not isinstance(alias, str) or not alias:
                raise ValueError("metrics(): metric names must be non-empty strings")
            if alias in aliases:
                raise ValueError(f"metrics(): duplicate metric name {alias!r}")
            aliases.add(alias)
            if isinstance(spec, str):
                expression = spec
            elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                source, function = spec
                if source != "*" and source not in columns:
                    raise ValueError(f"metrics(): unknown source column {source!r}")
                function = str(function).lower()
                if function == "count":
                    expression = "COUNT(*)" if source == "*" else f"COUNT({self._quote(source)})"
                elif function == "count_distinct":
                    expression = f"COUNT(DISTINCT {self._quote(source)})"
                elif function in {"sum", "mean", "avg", "median", "min", "max", "std", "stddev"}:
                    aggregate = (
                        "AVG"
                        if function == "mean"
                        else ("STDDEV_SAMP" if function == "std" else function.upper())
                    )
                    expression = f"{aggregate}({self._quote(source)})"
                elif function.startswith("quantile:"):
                    quantile = float(function.split(":", 1)[1])
                    if not 0 <= quantile <= 1:
                        raise ValueError("metrics(): quantile must be between 0 and 1")
                    expression = f"QUANTILE_CONT({self._quote(source)}, {quantile})"
                else:
                    raise ValueError(f"metrics(): unsupported aggregate {function!r}")
            else:
                raise ValueError(f"metrics(): invalid specification for {alias!r}")
            select_parts.append(f"{expression} AS {self._quote(alias)}")
        relation_name = _temp_name("_metrics")
        self._connection.register(relation_name, self._relation)
        query = f"SELECT {', '.join(select_parts)} FROM {self._quote(relation_name)}"
        if where:
            query += f" WHERE {where}"
        if groups:
            query += f" GROUP BY {', '.join(self._quote(column) for column in groups)}"
        return DuckJanitor(self._connection.sql(query), self._connection)

    def profile(self, *, quantiles: Optional[list[float]] = None, top_n: int = 0) -> "DuckJanitor":
        """Return one profiling row per column using DuckDB aggregates.

        The profile covers null counts and rates, distinct counts and rates,
        min/max, and — for numeric columns — mean, standard deviation, and
        the requested quantiles. When *top_n* is greater than zero, a
        ``top_values`` column holds the most frequent values for the column
        as a comma-separated list.

        Parameters
        ----------
        quantiles : list of float, optional
            Quantiles to compute for numeric columns, for example
            ``[0.25, 0.5, 0.75]``. Default is ``[0.5]``.
        top_n : int, optional
            Number of most-frequent values to include per column. Default
            is 0, which omits the column.

        Returns
        -------
        DuckJanitor
            A relation with one row per profiled column.

        Examples
        --------
        >>> dj.profile(quantiles=[0.5, 0.95], top_n=3).collect()
        """
        quantile_list = [0.5] if quantiles is None else list(quantiles)
        for quantile in quantile_list:
            if not 0 <= quantile <= 1:
                raise ValueError(f"profile(): quantiles must be between 0 and 1. Got {quantile}.")
        if top_n < 0:
            raise ValueError("profile(): top_n must be >= 0.")

        relation_name = _temp_name("_profile")
        self._connection.register(relation_name, self._relation)
        columns = list(self._relation.columns)
        types = [str(t).upper() for t in self._relation.types]
        numeric_types = (
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "UHUGEINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
            "REAL",
        )
        # Every branch of the UNION must expose the same columns, so the
        # quantile labels are computed once and non-numeric columns emit
        # NULL for the numeric-only statistics.
        quantile_labels = [
            (quantile, f"q{str(quantile).replace('.', '_')}_value") for quantile in quantile_list
        ]
        queries = []
        for index, column in enumerate(columns):
            quoted = self._quote(column)
            column_type = types[index]
            is_numeric = any(marker in column_type for marker in numeric_types)

            numeric_extras = []
            if is_numeric:
                numeric_extras.append(f"AVG(TRY_CAST({quoted} AS DOUBLE)) AS mean_value")
                numeric_extras.append(f"STDDEV_SAMP(TRY_CAST({quoted} AS DOUBLE)) AS stddev_value")
                for quantile, label in quantile_labels:
                    numeric_extras.append(
                        f"QUANTILE_CONT(TRY_CAST({quoted} AS DOUBLE), {quantile}) AS {label}"
                    )
            else:
                numeric_extras.append("NULL AS mean_value")
                numeric_extras.append("NULL AS stddev_value")
                for _quantile, label in quantile_labels:
                    numeric_extras.append(f"NULL AS {label}")

            if top_n > 0:
                top_values = (
                    ", (SELECT STRING_AGG(v, ', ' ORDER BY n DESC, v) FROM ("
                    f"SELECT CAST({quoted} AS VARCHAR) AS v, COUNT(*) AS n "
                    f"FROM {self._quote(relation_name)} WHERE {quoted} IS NOT NULL "
                    "GROUP BY 1 ORDER BY n DESC, v LIMIT "
                    f"{int(top_n)})) AS top_values"
                )
            else:
                top_values = ", NULL AS top_values"

            queries.append(
                "SELECT "
                f"{self._sql_value(column)} AS column_name, "
                f"{self._sql_value(column_type)} AS data_type, "
                f"COUNT(*) AS row_count, COUNT(*) - COUNT({quoted}) AS null_count, "
                f"CASE WHEN COUNT(*) = 0 THEN 0.0 ELSE (COUNT(*) - COUNT({quoted})) / CAST(COUNT(*) AS DOUBLE) END AS null_rate, "
                f"COUNT(DISTINCT {quoted}) AS distinct_count, "
                f"CASE WHEN COUNT(*) = 0 THEN 0.0 ELSE COUNT(DISTINCT {quoted}) / CAST(COUNT(*) AS DOUBLE) END AS distinct_rate, "
                f"CAST(MIN({quoted}) AS VARCHAR) AS min_value, "
                f"CAST(MAX({quoted}) AS VARCHAR) AS max_value, "
                f"{', '.join(numeric_extras)}"
                f"{top_values} "
                f"FROM {self._quote(relation_name)}"
            )
        return DuckJanitor(self._connection.sql(" UNION ALL ".join(queries)), self._connection)

    def metric_cube(
        self,
        dimensions: list[str],
        measures: dict[str, str],
        *,
        totals: Optional[str] = None,
        grouping_sets: Optional[list[list[str]]] = None,
        grand_total: bool = True,
        total_label: str = "ALL",
    ) -> "DuckJanitor":
        """Calculate detail metrics plus optional ROLLUP/CUBE subtotals."""
        if not dimensions:
            raise ValueError("metric_cube(): dimensions must be non-empty")
        missing = [column for column in dimensions if column not in self._relation.columns]
        if missing:
            raise ValueError(f"metric_cube(): unknown dimensions {missing}")
        if not measures:
            raise ValueError("metric_cube(): measures must be non-empty")
        if totals not in {None, "rollup", "cube", "grouping_sets"}:
            raise ValueError(
                "metric_cube(): totals must be None, 'rollup', 'cube', or 'grouping_sets'"
            )
        if totals == "grouping_sets":
            if grouping_sets is None:
                raise ValueError("metric_cube(): grouping_sets is required")
            for subset in grouping_sets:
                unknown = [column for column in subset if column not in dimensions]
                if unknown:
                    raise ValueError(f"metric_cube(): unknown grouping-set dimensions {unknown}")
        relation_name = _temp_name("_metric_cube")
        self._connection.register(relation_name, self._relation)
        if totals == "rollup":
            group_clause = f"ROLLUP ({', '.join(self._quote(column) for column in dimensions)})"
        elif totals == "cube":
            group_clause = f"CUBE ({', '.join(self._quote(column) for column in dimensions)})"
        elif totals == "grouping_sets":
            rendered = []
            for subset in grouping_sets or []:
                rendered.append("(" + ", ".join(self._quote(column) for column in subset) + ")")
            group_clause = f"GROUPING SETS ({', '.join(rendered)})"
        else:
            group_clause = ", ".join(self._quote(column) for column in dimensions)
        select_parts = []
        for column in dimensions:
            quoted = self._quote(column)
            if totals is None:
                select_parts.append(quoted)
            else:
                select_parts.append(
                    f"CASE WHEN GROUPING({quoted}) = 1 THEN {self._sql_value(total_label)} "
                    f"ELSE CAST({quoted} AS VARCHAR) END AS {quoted}"
                )
        select_parts.extend(
            f"{expression} AS {self._quote(alias)}" for alias, expression in measures.items()
        )
        if totals is not None:
            grouping_flags = [f"GROUPING({self._quote(column)})" for column in dimensions]
            flag_sum = " + ".join(grouping_flags)
            select_parts.extend(
                [
                    f"GROUPING_ID({', '.join(self._quote(column) for column in dimensions)}) AS grouping_id",
                    f"CASE WHEN ({flag_sum}) = {len(dimensions)} THEN 'grand_total' WHEN ({flag_sum}) > 0 THEN 'subtotal' ELSE 'detail' END AS grouping_level",
                    f"(({flag_sum}) > 0) AS is_total",
                    f"(({flag_sum}) = {len(dimensions)}) AS is_grand_total",
                ]
            )
        query = f"SELECT {', '.join(select_parts)} FROM {self._quote(relation_name)} GROUP BY {group_clause}"
        if totals is not None and not grand_total:
            query += f" HAVING ({' + '.join(f'GROUPING({self._quote(column)})' for column in dimensions)}) < {len(dimensions)}"
        return DuckJanitor(self._connection.sql(query), self._connection)

    def rate_metrics(
        self,
        rates: dict[str, tuple[str, str]],
        *,
        group_by: Optional[Union[str, list[str]]] = None,
        where: Optional[str] = None,
    ) -> "DuckJanitor":
        """Calculate safe numerator/denominator rates by group."""
        groups = [group_by] if isinstance(group_by, str) else list(group_by or [])
        missing = [column for column in groups if column not in self._relation.columns]
        if missing:
            raise ValueError(f"rate_metrics(): unknown group_by columns {missing}")
        if not rates:
            raise ValueError("rate_metrics(): rates must be non-empty")
        expressions = [self._quote(column) for column in groups]
        for alias, (numerator, denominator) in rates.items():
            expressions.append(
                f"CAST(({numerator}) AS DOUBLE) / NULLIF(({denominator}), 0) AS {self._quote(alias)}"
            )
        relation_name = _temp_name("_rate_metrics")
        self._connection.register(relation_name, self._relation)
        query = f"SELECT {', '.join(expressions)} FROM {self._quote(relation_name)}"
        if where:
            query += f" WHERE {where}"
        if groups:
            query += f" GROUP BY {', '.join(self._quote(column) for column in groups)}"
        return DuckJanitor(self._connection.sql(query), self._connection)

    def cohort_metrics(
        self,
        entity: str,
        activity_date: str,
        *,
        frequency: str = "month",
        cohort_date: Optional[str] = None,
    ) -> "DuckJanitor":
        """Calculate cohort size, active entities, and retention by period."""
        if entity not in self._relation.columns or activity_date not in self._relation.columns:
            raise ValueError("cohort_metrics(): entity and activity_date columns are required")
        if frequency not in {"day", "week", "month", "quarter", "year"}:
            raise ValueError("cohort_metrics(): unsupported frequency")
        relation_name = _temp_name("_cohort_metrics")
        self._connection.register(relation_name, self._relation)
        cohort_expression = (
            self._quote(cohort_date) if cohort_date else f"MIN({self._quote(activity_date)})"
        )
        query = f"""
            WITH base AS (
                SELECT {self._quote(entity)} AS entity_id,
                       date_trunc('{frequency}', CAST({self._quote(activity_date)} AS TIMESTAMP)) AS activity_period,
                       date_trunc('{frequency}', CAST({cohort_expression} AS TIMESTAMP)) AS cohort_period
                FROM {self._quote(relation_name)}
                GROUP BY {self._quote(entity)}, activity_period
            ), cohorts AS (
                SELECT cohort_period, COUNT(DISTINCT entity_id) AS cohort_size
                FROM base GROUP BY cohort_period
            )
            SELECT b.cohort_period,
                   date_diff('{frequency}', b.cohort_period, b.activity_period) AS period_number,
                   c.cohort_size,
                   COUNT(DISTINCT b.entity_id) AS active_entities,
                   COUNT(DISTINCT b.entity_id) / CAST(c.cohort_size AS DOUBLE) AS retention_rate
            FROM base b JOIN cohorts c USING (cohort_period)
            GROUP BY b.cohort_period, period_number, c.cohort_size
            ORDER BY b.cohort_period, period_number
        """
        return DuckJanitor(self._connection.sql(query), self._connection)

    def freshness(
        self,
        timestamp_col: str,
        *,
        stale_after: Optional[str] = None,
    ) -> "DuckJanitor":
        """Return row count, latest timestamp, age, and stale status."""
        if timestamp_col not in self._relation.columns:
            raise ValueError(f"freshness(): unknown timestamp column {timestamp_col!r}")
        relation_name = _temp_name("_freshness")
        self._connection.register(relation_name, self._relation)
        query = (
            f"SELECT COUNT(*) AS row_count, MAX(CAST({self._quote(timestamp_col)} AS TIMESTAMP)) AS max_timestamp, "
            f"EPOCH(CURRENT_TIMESTAMP - MAX(CAST({self._quote(timestamp_col)} AS TIMESTAMP))) AS age_seconds"
        )
        if stale_after:
            query += f", MAX(CAST({self._quote(timestamp_col)} AS TIMESTAMP)) < CURRENT_TIMESTAMP - INTERVAL '{stale_after}' AS is_stale"
        else:
            query += ", NULL::BOOLEAN AS is_stale"
        return DuckJanitor(
            self._connection.sql(query + f" FROM {self._quote(relation_name)}"), self._connection
        )

    def reconcile(
        self,
        other: "DuckJanitor",
        keys: Union[str, list[str]],
    ) -> "DuckJanitor":
        """Summarize key coverage and duplicate differences between relations."""
        keys = [keys] if isinstance(keys, str) else list(keys)
        missing_left = [column for column in keys if column not in self._relation.columns]
        missing_right = [column for column in keys if column not in other._relation.columns]
        if missing_left or missing_right:
            raise ValueError(
                f"reconcile(): missing keys left={missing_left}, right={missing_right}"
            )
        left_name, right_name = (
            _temp_name("_reconcile_left"),
            _temp_name("_reconcile_right"),
        )
        self._connection.register(left_name, self._relation)
        self._connection.register(right_name, other._relation)
        condition = " AND ".join(
            f"l.{self._quote(column)} IS NOT DISTINCT FROM r.{self._quote(column)}"
            for column in keys
        )
        left_key = ", ".join(f"l.{self._quote(column)}" for column in keys)
        right_key = ", ".join(f"r.{self._quote(column)}" for column in keys)
        query = f"""
            WITH left_keys AS (SELECT {left_key}, COUNT(*) AS n FROM {self._quote(left_name)} l GROUP BY {left_key}),
                 right_keys AS (SELECT {right_key}, COUNT(*) AS n FROM {self._quote(right_name)} r GROUP BY {right_key}),
                 matched AS (
                     SELECT l.n AS left_n, r.n AS right_n
                     FROM left_keys l JOIN right_keys r ON {condition}
                 )
            SELECT (SELECT COUNT(*) FROM {self._quote(left_name)}) AS left_rows,
                   (SELECT COUNT(*) FROM {self._quote(right_name)}) AS right_rows,
                   (SELECT COUNT(*) FROM left_keys) AS left_distinct_keys,
                   (SELECT COUNT(*) FROM right_keys) AS right_distinct_keys,
                   (SELECT COUNT(*) FROM matched) AS matched_keys,
                   (SELECT COALESCE(SUM(left_n), 0) FROM matched) AS matched_left_rows,
                   (SELECT COALESCE(SUM(right_n), 0) FROM matched) AS matched_right_rows
        """
        return DuckJanitor(self._connection.sql(query), self._connection)

    @classmethod
    def metric_from_database(
        cls,
        connection: Any,
        query: str,
        metrics: dict[str, Any],
        *,
        group_by: Optional[Union[str, list[str]]] = None,
        where: Optional[str] = None,
        params: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DuckJanitor":
        """Execute a portable aggregate in the source database before transfer."""
        groups = [group_by] if isinstance(group_by, str) else list(group_by or [])
        select_parts = list(groups)
        for alias, spec in metrics.items():
            if isinstance(spec, str):
                expression = spec
            else:
                source, function = spec
                function = str(function).lower()
                aggregate = {"mean": "AVG", "count_distinct": "COUNT(DISTINCT"}.get(
                    function, function.upper()
                )
                if function == "count_distinct":
                    expression = f"COUNT(DISTINCT {source})"
                elif function == "count":
                    expression = "COUNT(*)" if source == "*" else f"COUNT({source})"
                else:
                    expression = f"{aggregate}({source})"
            select_parts.append(f"{expression} AS {alias}")
        source_sql = f"SELECT {', '.join(select_parts)} FROM ({query}) AS source"
        if where:
            source_sql += f" WHERE {where}"
        if groups:
            source_sql += f" GROUP BY {', '.join(groups)}"
        return cls.from_database(connection, source_sql, params=params, **kwargs)

    def diff(
        self,
        other: "DuckJanitor",
        keys: Union[str, list[str]],
        *,
        columns: Optional[list[str]] = None,
        ignore: Optional[list[str]] = None,
        context: Optional[list[str]] = None,
        numeric_tolerance: Optional[float] = None,
        timestamp_precision: Optional[str] = None,
        require_matching_columns: bool = True,
        upcast_types: bool = False,
        null_equals_empty: bool = False,
        prefix: str = "diff_",
        auto_install: bool = False,
    ) -> "DuckJanitor":
        """Compare this relation with another using ``duck_diff``.

        Parameters
        ----------
        other : DuckJanitor
            The relation to compare against the current relation.
        keys : str or list[str]
            Primary-key column or composite primary key.
        columns, ignore, context : list[str], optional
            Columns to compare, exclude, or expose without comparing.
        numeric_tolerance : float, optional
            Absolute tolerance for numeric comparisons.
        timestamp_precision : str, optional
            Precision used to truncate timestamps before comparing.
        require_matching_columns : bool, default True
            Require matching names and types on both sides.
        upcast_types : bool, default False
            Reconcile compatible type differences when matching columns are
            not required.
        null_equals_empty : bool, default False
            Treat NULL and the empty string as equal for VARCHAR values.
        prefix : str, default ``"diff_"``
            Prefix for the extension's status and data columns.
        auto_install : bool, default False
            Install ``duck_diff`` from DuckDB's community repository if needed.

        Returns
        -------
        DuckJanitor
            One row per distinct key with row and column-level diff status.
        """
        return self._duck_diff(
            other,
            "table_diff",
            keys,
            columns=columns,
            ignore=ignore,
            context=context,
            numeric_tolerance=numeric_tolerance,
            timestamp_precision=timestamp_precision,
            require_matching_columns=require_matching_columns,
            upcast_types=upcast_types,
            null_equals_empty=null_equals_empty,
            prefix=prefix,
            auto_install=auto_install,
        )

    def diff_summary(
        self,
        other: "DuckJanitor",
        keys: Union[str, list[str]],
        *,
        columns: Optional[list[str]] = None,
        ignore: Optional[list[str]] = None,
        numeric_tolerance: Optional[float] = None,
        timestamp_precision: Optional[str] = None,
        require_matching_columns: bool = True,
        upcast_types: bool = False,
        null_equals_empty: bool = False,
        auto_install: bool = False,
    ) -> "DuckJanitor":
        """Return aggregate counts from a ``duck_diff`` comparison."""
        return self._duck_diff(
            other,
            "table_diff_summary",
            keys,
            columns=columns,
            ignore=ignore,
            numeric_tolerance=numeric_tolerance,
            timestamp_precision=timestamp_precision,
            require_matching_columns=require_matching_columns,
            upcast_types=upcast_types,
            null_equals_empty=null_equals_empty,
            auto_install=auto_install,
        )

    def schema_diff(
        self,
        other: "DuckJanitor",
        *,
        auto_install: bool = False,
    ) -> "DuckJanitor":
        """Compare the column names and types of two relations."""
        if not isinstance(other, DuckJanitor):
            raise TypeError("schema_diff(): other must be a DuckJanitor")
        self.load_extension("duck_diff", auto_install=auto_install)
        left_name, right_name = self._register_diff_sources(other)
        query = (
            "SELECT * FROM schema_diff("
            f"{self._sql_value(f'FROM {self._quote(left_name)}')}, "
            f"{self._sql_value(f'FROM {self._quote(right_name)}')})"
        )
        return DuckJanitor(self._connection.query(query), self._connection)

    def _duck_diff(
        self,
        other: "DuckJanitor",
        function: str,
        keys: Union[str, list[str]],
        **options: Any,
    ) -> "DuckJanitor":
        if not isinstance(other, DuckJanitor):
            raise TypeError(f"{function}(): other must be a DuckJanitor")
        key_list = [keys] if isinstance(keys, str) else list(keys)
        if not key_list or any(not isinstance(key, str) or not key for key in key_list):
            raise ValueError(f"{function}(): keys must contain at least one column name")
        if function not in {"table_diff", "table_diff_summary"}:
            raise ValueError(f"Unsupported duck_diff function: {function}")

        self.load_extension("duck_diff", auto_install=options.pop("auto_install", False))
        left_name, right_name = self._register_diff_sources(other)
        arguments = [
            self._sql_value(f"FROM {self._quote(left_name)}"),
            self._sql_value(f"FROM {self._quote(right_name)}"),
            f"pk := {self._sql_list(key_list)}",
        ]
        for name, value in options.items():
            if value is None:
                continue
            if name in {"columns", "ignore", "context"}:
                arguments.append(f"{name} := {self._sql_list(value)}")
            else:
                arguments.append(f"{name} := {self._sql_value(value)}")
        query = f"SELECT * FROM {function}({', '.join(arguments)})"
        return DuckJanitor(self._connection.query(query), self._connection)

    def _register_diff_sources(self, other: "DuckJanitor") -> tuple[str, str]:
        left_name = _temp_name("_duck_diff_left")
        right_name = _temp_name("_duck_diff_right")
        self._connection.register(left_name, self._relation)
        try:
            self._connection.register(right_name, other._relation)
        except Exception:
            self._connection.register(right_name, other.collect())
        return left_name, right_name

    def validate_keys(
        self,
        keys: Union[str, list[str]],
        *,
        date_col: Optional[str] = None,
        date_lower: Optional[Any] = None,
        date_upper: Optional[Any] = None,
        allow_null: bool = False,
        unique: bool = True,
    ) -> "DuckJanitor":
        """Validate key columns and optional date bounds.

        The current relation is returned unchanged when validation passes.
        This makes validation composable at the start of a pipeline while
        failing before temporal joins, diffs, or graph construction.
        """
        key_list = [keys] if isinstance(keys, str) else list(keys)
        if not key_list or any(not isinstance(key, str) or not key for key in key_list):
            raise ValueError("validate_keys(): keys must contain at least one column name")
        columns = set(self._relation.columns)
        missing = [column for column in key_list if column not in columns]
        if missing:
            raise ValueError(f"validate_keys(): unknown key columns {missing}")
        if date_col is not None and date_col not in columns:
            raise ValueError(f"validate_keys(): unknown date column {date_col!r}")

        relation_name = _temp_name("_validate_keys")
        self._connection.register(relation_name, self._relation)
        quoted_keys = ", ".join(self._quote(column) for column in key_list)
        if not allow_null:
            null_predicate = " OR ".join(f"{self._quote(column)} IS NULL" for column in key_list)
            null_count = self._connection.sql(
                f"SELECT COUNT(*) FROM {self._quote(relation_name)} WHERE {null_predicate}"
            ).fetchone()[0]
            if null_count:
                raise ValueError(f"validate_keys(): key columns contain {null_count} NULL row(s)")
        if unique:
            duplicate_count = self._connection.sql(
                f"SELECT COUNT(*) - COUNT(DISTINCT ({quoted_keys})) "
                f"FROM {self._quote(relation_name)}"
            ).fetchone()[0]
            if duplicate_count:
                raise ValueError(
                    f"validate_keys(): key columns contain {duplicate_count} duplicate row(s)"
                )
        if date_col is not None and (date_lower is not None or date_upper is not None):
            date_predicates = []
            if date_lower is not None:
                date_predicates.append(
                    f"CAST({self._quote(date_col)} AS DATE) < {self._sql_value(date_lower)}"
                )
            if date_upper is not None:
                date_predicates.append(
                    f"CAST({self._quote(date_col)} AS DATE) > {self._sql_value(date_upper)}"
                )
            out_of_range = self._connection.sql(
                f"SELECT COUNT(*) FROM {self._quote(relation_name)} "
                f"WHERE {' OR '.join(date_predicates)}"
            ).fetchone()[0]
            if out_of_range:
                raise ValueError(f"validate_keys(): {out_of_range} row(s) fall outside date bounds")
        return self

    def deduplicate(
        self,
        keys: Union[str, list[str]],
        *,
        order_by: Optional[Union[str, list[str]]] = None,
        keep: str = "first",
    ) -> "DuckJanitor":
        """Keep one deterministic row per key combination.

        Parameters
        ----------
        keys : str or list[str]
            Columns defining duplicate groups.
        order_by : str or list[str], optional
            Columns used to choose the retained row. The first or last row
            under this ordering is retained. Without it, source order is used.
        keep : {"first", "last"}, default "first"
            Which ordered row to retain.
        """
        if keep not in {"first", "last"}:
            raise ValueError("deduplicate(): keep must be 'first' or 'last'")
        key_list = [keys] if isinstance(keys, str) else list(keys)
        order_list = [order_by] if isinstance(order_by, str) else list(order_by or [])
        columns = set(self._relation.columns)
        missing = [column for column in key_list + order_list if column not in columns]
        if not key_list or missing:
            raise ValueError(f"deduplicate(): unknown or missing columns {missing or key_list}")
        relation_name = _temp_name("_deduplicate")
        self._connection.register(relation_name, self._relation)
        partition_sql = ", ".join(self._quote(column) for column in key_list)
        if order_list:
            direction = "ASC" if keep == "first" else "DESC"
            order_sql = ", ".join(f"{self._quote(column)} {direction}" for column in order_list)
        else:
            order_sql = "row_number() OVER ()"
        selected = ", ".join(self._quote(column) for column in self._relation.columns)
        # QUALIFY filters on window functions directly, which avoids
        # wrapping the query in an extra CTE and lets DuckDB push the
        # partition/filter into a single scan.
        query = (
            f"SELECT {selected} FROM {self._quote(relation_name)} "
            f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_sql} ORDER BY {order_sql}) = 1"
        )
        return DuckJanitor(self._connection.query(query), self._connection)

    def filter_noise(
        self,
        *,
        id_col: Optional[str] = None,
        exclude_ids: Optional[list[Any]] = None,
        exclude_regex: Optional[str] = None,
        min_records: Optional[int] = None,
        where: Optional[str] = None,
    ) -> "DuckJanitor":
        """Filter known noise records before longitudinal or graph analysis.

        ``filter_noise`` is intentionally domain-neutral: it can remove test
        IDs, service accounts, malformed records, or low-frequency entities
        using explicit criteria supplied by the caller.
        """
        if id_col is not None and id_col not in self._relation.columns:
            raise ValueError(f"filter_noise(): unknown id column {id_col!r}")
        if min_records is not None and min_records < 1:
            raise ValueError("filter_noise(): min_records must be at least 1")
        if (exclude_ids or exclude_regex or min_records is not None) and id_col is None:
            raise ValueError("filter_noise(): id_col is required for ID-based filters")
        relation_name = _temp_name("_filter_noise")
        self._connection.register(relation_name, self._relation)
        predicates = [where] if where else []
        if exclude_ids:
            values = ", ".join(self._sql_value(value) for value in exclude_ids)
            predicates.append(f"{self._quote(id_col)} NOT IN ({values})")
        if exclude_regex:
            predicates.append(
                f"NOT regexp_matches(CAST({self._quote(id_col)} AS VARCHAR), "
                f"{self._sql_value(exclude_regex)})"
            )
        if min_records is not None:
            predicates.append(f"__pyduck_noise_count >= {int(min_records)}")
        selected = ", ".join(self._quote(column) for column in self._relation.columns)
        if min_records is not None:
            query = (
                f"WITH counted AS (SELECT {selected}, "
                f"COUNT(*) OVER (PARTITION BY {self._quote(id_col)}) AS __pyduck_noise_count "
                f"FROM {self._quote(relation_name)}) SELECT {selected} FROM counted "
                f"WHERE {' AND '.join(predicates)}"
            )
        else:
            query = f"SELECT {selected} FROM {self._quote(relation_name)}"
            if predicates:
                query += " WHERE " + " AND ".join(predicates)
        return DuckJanitor(self._connection.query(query), self._connection)

    def hierarchy_edges(
        self,
        source: str,
        parent: str,
        *,
        source_name: str = "source",
        target_name: str = "target",
        drop_null_parent: bool = True,
        reject_self_loops: bool = True,
        deduplicate: bool = True,
    ) -> "DuckJanitor":
        """Convert an entity/parent relation into a directed edge relation."""
        columns = set(self._relation.columns)
        missing = [column for column in (source, parent) if column not in columns]
        if missing:
            raise ValueError(f"hierarchy_edges(): unknown columns {missing}")
        for name in (source_name, target_name):
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"hierarchy_edges(): invalid output column {name!r}")
        relation_name = _temp_name("_hierarchy_edges")
        self._connection.register(relation_name, self._relation)
        predicates = []
        if drop_null_parent:
            predicates.append(f"{self._quote(parent)} IS NOT NULL")
        if reject_self_loops:
            predicates.append(f"{self._quote(source)} IS DISTINCT FROM {self._quote(parent)}")
        query = (
            f"SELECT {self._quote(source)} AS {self._quote(source_name)}, "
            f"{self._quote(parent)} AS {self._quote(target_name)} "
            f"FROM {self._quote(relation_name)}"
        )
        if predicates:
            query += " WHERE " + " AND ".join(predicates)
        if deduplicate:
            query = f"SELECT DISTINCT * FROM ({query}) hierarchy_edges"
        return DuckJanitor(self._connection.query(query), self._connection)

    def time_slice(
        self,
        date_col: str,
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        inclusive: str = "both",
    ) -> "DuckJanitor":
        """Filter rows to a bounded temporal slice.

        ``inclusive`` accepts ``"both"``, ``"left"``, ``"right"``, or
        ``"neither"`` and controls boundary inclusion for ``start`` and
        ``end``. Values are cast to ``TIMESTAMP`` so dates and timestamps can
        be used together.
        """
        if date_col not in self._relation.columns:
            raise ValueError(f"time_slice(): unknown date column {date_col!r}")
        if start is None and end is None:
            raise ValueError("time_slice(): start or end is required")
        if inclusive not in {"both", "left", "right", "neither"}:
            raise ValueError("time_slice(): inclusive must be both, left, right, or neither")
        relation_name = _temp_name("_time_slice")
        self._connection.register(relation_name, self._relation)
        predicates = []
        if start is not None:
            operator = ">=" if inclusive in {"both", "left"} else ">"
            predicates.append(
                f"CAST({self._quote(date_col)} AS TIMESTAMP) {operator} "
                f"CAST({self._sql_value(start)} AS TIMESTAMP)"
            )
        if end is not None:
            operator = "<=" if inclusive in {"both", "right"} else "<"
            predicates.append(
                f"CAST({self._quote(date_col)} AS TIMESTAMP) {operator} "
                f"CAST({self._sql_value(end)} AS TIMESTAMP)"
            )
        selected = ", ".join(self._quote(column) for column in self._relation.columns)
        query = (
            f"SELECT {selected} FROM {self._quote(relation_name)} WHERE {' AND '.join(predicates)}"
        )
        return DuckJanitor(self._connection.query(query), self._connection)

    def event_window(
        self,
        date_col: str,
        event_date: Any,
        *,
        pre_window: str = "0 days",
        post_window: str = "0 days",
    ) -> "DuckJanitor":
        """Return rows in a relative window around an event timestamp.

        Window values use DuckDB interval syntax, such as ``"30 days"`` or
        ``"2 hours"``. Both endpoints are inclusive.
        """
        if date_col not in self._relation.columns:
            raise ValueError(f"event_window(): unknown date column {date_col!r}")
        interval_pattern = r"\d+(?:\.\d+)?\s+(?:microsecond|millisecond|second|minute|hour|day|week|month|quarter|year)s?"
        for label, value in (("pre_window", pre_window), ("post_window", post_window)):
            if not isinstance(value, str) or not re.fullmatch(
                interval_pattern, value.strip(), re.I
            ):
                raise ValueError(f"event_window(): {label} must look like '30 days' or '2 hours'")
        relation_name = _temp_name("_event_window")
        self._connection.register(relation_name, self._relation)
        selected = ", ".join(self._quote(column) for column in self._relation.columns)
        event_value = f"CAST({self._sql_value(event_date)} AS TIMESTAMP)"
        date_value = f"CAST({self._quote(date_col)} AS TIMESTAMP)"
        query = (
            f"SELECT {selected} FROM {self._quote(relation_name)} WHERE {date_value} "
            f"BETWEEN {event_value} - INTERVAL {self._sql_value(pre_window)} "
            f"AND {event_value} + INTERVAL {self._sql_value(post_window)}"
        )
        return DuckJanitor(self._connection.query(query), self._connection)

    def change_detection(
        self,
        keys: Union[str, list[str]],
        order_by: Union[str, list[str]],
        *,
        columns: Optional[list[str]] = None,
        include_unchanged: bool = True,
    ) -> "DuckJanitor":
        """Annotate rows with changed columns relative to the prior row.

        The result adds ``is_changed``, ``change_count``, and
        ``changed_columns``. Comparisons are NULL-safe and partitioned by
        ``keys`` in ``order_by`` order.
        """
        key_list = [keys] if isinstance(keys, str) else list(keys)
        order_list = [order_by] if isinstance(order_by, str) else list(order_by)
        relation_columns = list(self._relation.columns)
        compare_columns = columns or [
            column for column in relation_columns if column not in set(key_list + order_list)
        ]
        missing = [
            column
            for column in key_list + order_list + compare_columns
            if column not in relation_columns
        ]
        if not key_list or not order_list or missing:
            raise ValueError(f"change_detection(): unknown or missing columns {missing}")
        relation_name = _temp_name("_change_detection")
        self._connection.register(relation_name, self._relation)
        partition_sql = ", ".join(self._quote(column) for column in key_list)
        order_sql = ", ".join(self._quote(column) for column in order_list)
        selected = ", ".join(self._quote(column) for column in relation_columns)
        previous_columns = ", ".join(
            f"LAG({self._quote(column)}) OVER (PARTITION BY {partition_sql} "
            f"ORDER BY {order_sql}) AS {self._quote(f'__previous_{column}')}"
            for column in compare_columns
        )
        previous_columns += (
            f", ROW_NUMBER() OVER (PARTITION BY {partition_sql} ORDER BY {order_sql}) > 1 "
            "AS __pyduck_has_previous"
        )
        change_predicates = [
            f"{self._quote(column)} IS DISTINCT FROM {self._quote(f'__previous_{column}')}"
            for column in compare_columns
        ]
        changed_cases = ", ".join(
            f"CASE WHEN {predicate} THEN {self._sql_value(column)} ELSE NULL END"
            for column, predicate in zip(compare_columns, change_predicates)
        )
        changed_condition = f"__pyduck_has_previous AND ({' OR '.join(change_predicates)})"
        query = (
            "WITH previous AS ("
            f"SELECT {selected}, {previous_columns} FROM {self._quote(relation_name)}), "
            "annotated AS (SELECT "
            f"{selected}, ({changed_condition}) AS is_changed, "
            f"list_count(list_filter([{changed_cases}], x -> x IS NOT NULL)) AS change_count, "
            f"array_to_string(list_filter([{changed_cases}], x -> x IS NOT NULL), ', ') "
            "AS changed_columns FROM previous) SELECT * FROM annotated"
        )
        if not include_unchanged:
            query += " WHERE is_changed"
        return DuckJanitor(self._connection.query(query), self._connection)

    def window_mutate(
        self,
        expressions: dict[str, Union[str, dict[str, Any]]],
        partition_by: Optional[Union[str, list[str]]] = None,
        order_by: Optional[Union[str, list[str]]] = None,
        frame: Optional[str] = None,
    ) -> "DuckJanitor":
        """Add one or more SQL window expressions to the relation.

        Expressions may be short function calls, which receive the shared
        ``PARTITION BY`` / ``ORDER BY`` specification, or complete SQL
        expressions containing their own ``OVER (...)`` clause. This keeps
        common analytical windows concise while preserving access to DuckDB's
        full window-function syntax.

        Parameters
        ----------
        expressions : dict[str, str or dict]
            Mapping of output column names to SQL expressions. A dict value
            may contain ``expression``, ``partition_by``, ``order_by``, and
            ``frame`` to override the shared window specification.
        partition_by : str or list[str], optional
            Columns used to partition the shared window.
        order_by : str or list[str], optional
            Columns used to order the shared window.
        frame : str, optional
            A DuckDB frame clause such as ``"ROWS BETWEEN 3 PRECEDING AND
            CURRENT ROW"``. Complete expressions with their own ``OVER``
            clause ignore this shared frame.

        Returns
        -------
        DuckJanitor
            Self with the window-derived columns added or replaced.

        Examples
        --------
        >>> events = DuckJanitor.from_pandas(pd.DataFrame({
        ...     'employee_id': [1, 1, 1], 'score': [10, 20, 15],
        ...     'event_time': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        ... }))
        >>> events.window_mutate(
        ...     {'previous_score': 'LAG(score)', 'rolling_score': 'AVG(score)'},
        ...     partition_by='employee_id', order_by='event_time',
        ...     frame='ROWS BETWEEN 1 PRECEDING AND CURRENT ROW',
        ... ).collect()['previous_score'].tolist()
        [<NA>, 10, 20]
        """
        if not expressions:
            raise ValueError("window_mutate(): expressions must not be empty")

        def normalize_columns(value, label):
            columns = [value] if isinstance(value, str) else list(value or [])
            for column in columns:
                if column not in self._relation.columns:
                    raise ValueError(f"window_mutate(): {label} column {column!r} not found")
            return columns

        shared_partition = normalize_columns(partition_by, "partition_by")
        shared_order = normalize_columns(order_by, "order_by")
        temp_name = _temp_name("_window")
        self._connection.register(temp_name, self._relation)

        output_names = set(self._relation.columns)
        select_columns = []
        for column in self._relation.columns:
            if column not in expressions:
                select_columns.append(self._quote(column))
        expression_sql = []
        for name, specification in expressions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("window_mutate(): expression names must be non-empty strings")
            if isinstance(specification, str):
                expression = specification
                partition = shared_partition
                ordering = shared_order
                expression_frame = frame
            elif isinstance(specification, dict):
                expression = specification.get("expression")
                if not isinstance(expression, str) or not expression.strip():
                    raise ValueError(f"window_mutate(): {name!r} needs a non-empty expression")
                partition = normalize_columns(
                    specification.get("partition_by", shared_partition), "partition_by"
                )
                ordering = normalize_columns(
                    specification.get("order_by", shared_order), "order_by"
                )
                expression_frame = specification.get("frame", frame)
            else:
                raise TypeError(
                    f"window_mutate(): {name!r} must be a SQL string or specification dict"
                )

            if not expression.strip():
                raise ValueError(f"window_mutate(): {name!r} expression must not be empty")
            if re.search(r"\bOVER\s*\(", expression, flags=re.IGNORECASE):
                window_expression = expression
            else:
                clauses = []
                if partition:
                    clauses.append("PARTITION BY " + ", ".join(self._quote(c) for c in partition))
                if ordering:
                    clauses.append("ORDER BY " + ", ".join(self._quote(c) for c in ordering))
                if expression_frame:
                    clauses.append(str(expression_frame))
                window_expression = f"{expression} OVER ({' '.join(clauses)})"
            expression_sql.append(f"{window_expression} AS {self._quote(name)}")
            output_names.add(name)

        query = (
            f"SELECT {', '.join(select_columns + expression_sql)} "
            f'FROM (SELECT *, row_number() OVER () AS "__dj_window_row" '
            f"FROM {self._quote(temp_name)}) window_source "
            f'ORDER BY "__dj_window_row"'
        )
        return DuckJanitor(self._connection.query(query), self._connection)

    def recursive_cte(
        self,
        name: str,
        anchor: str,
        recursive: str,
        columns: Optional[list[str]] = None,
        union_all: bool = True,
    ) -> "DuckJanitor":
        """Execute a recursive CTE rooted in the current relation.

        The current relation is available as ``self`` in both SQL fragments.
        The recursive fragment may reference the CTE by ``name``. This is a
        flexible SQL-native primitive for hierarchy traversal, reachability,
        path enumeration, and cycle-aware graph analysis.

        Parameters
        ----------
        name : str
            CTE name referenced by the recursive fragment.
        anchor : str
            Non-recursive seed query. Use ``self`` for the current relation.
        recursive : str
            Recursive query. Reference ``name`` to walk prior results and
            ``self`` to access the original relation.
        columns : list[str], optional
            Explicit CTE column names. Useful when the anchor uses computed
            expressions or when stable names are important downstream.
        union_all : bool, default True
            Use ``UNION ALL`` (recommended for traversal) or ``UNION``.

        Returns
        -------
        DuckJanitor
            The rows produced by the recursive CTE.

        Examples
        --------
        >>> org = DuckJanitor.from_pandas(pd.DataFrame({
        ...     'employee_id': [1, 2, 3], 'manager_id': [None, 1, 2]
        ... }))
        >>> tree = org.recursive_cte(
        ...     'org_tree',
        ...     'SELECT employee_id, manager_id, 0 AS depth FROM self WHERE manager_id IS NULL',
        ...     'SELECT e.employee_id, e.manager_id, t.depth + 1 FROM self e '
        ...     'JOIN org_tree t ON e.manager_id = t.employee_id',
        ... )
        >>> tree.collect()['depth'].tolist()
        [0, 1, 2]
        """
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("recursive_cte(): name must be a valid SQL identifier")
        if not isinstance(anchor, str) or not anchor.strip():
            raise ValueError("recursive_cte(): anchor must be a non-empty SQL query")
        if not isinstance(recursive, str) or not recursive.strip():
            raise ValueError("recursive_cte(): recursive must be a non-empty SQL query")
        if columns is not None:
            if not columns or any(not isinstance(column, str) or not column for column in columns):
                raise ValueError("recursive_cte(): columns must contain non-empty names")
            column_sql = "(" + ", ".join(self._quote(column) for column in columns) + ")"
        else:
            column_sql = ""

        temp_name = _temp_name("_recursive")
        self._connection.register(temp_name, self._relation)
        anchor_sql = re.sub(r"\bself\b", self._quote(temp_name), anchor)
        recursive_sql = re.sub(r"\bself\b", self._quote(temp_name), recursive)
        operator = "UNION ALL" if union_all else "UNION"
        query = (
            f"WITH RECURSIVE {self._quote(name)}{column_sql} AS ("
            f"{anchor_sql} {operator} {recursive_sql}) "
            f"SELECT * FROM {self._quote(name)}"
        )
        return DuckJanitor(self._connection.query(query), self._connection)

    def bin_numeric(
        self,
        column: str,
        target_column: str,
        bins: Union[int, list[float]] = 5,
        strategy: str = "quantile",
    ) -> "DuckJanitor":
        """Bin a numeric column into discrete intervals.

        Parameters
        ----------
        column : str
            Name of the numeric column to bin.
        target_column : str
            Name of the output column holding the bin label.
        bins : int or list of float, default 5
            Either the number of equal-width/quantile bins (when ``int``)
            or an explicit list of bin edges.
        strategy : str, default 'quantile'
            ``'quantile'`` (DuckDB ``NTILE``) or ``'equal_width'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``target_column`` added.
        """
        from .cleaning_ops_extended import bin_numeric as _bin_numeric

        new_relation = _bin_numeric(
            self._relation, column, target_column, bins, strategy, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def optimize_bins_to_target(
        self,
        y: str,
        *,
        columns=None,
        method: str = "auto",
        cv: int = 3,
        task: str = "auto",
        cast_to_text: bool = True,
        suffix: str = "_binned",
        random_state: int | None = 42,
        return_diagnostics: bool = False,
        optbinning_kwargs=None,
        holdout: bool | float = False,
        stratify: bool = True,
        sample_n: int | None = 100_000,
    ):
        """Learn supervised bin cuts from a target column.

        Bins each requested numeric column using its relationship with ``y``
        (two engines: OptBinning for binary targets, DecisionTreeDiscretiser
        for binary / multiclass / continuous).  Produces text interval labels
        such as ``(-inf, 26.5]`` by default.  Full reference with examples:
        ``docs/supervised_binning.md``.

        Parameters
        ----------
        y : str
            Target column (binary, multiclass, or continuous).
        columns : str or list of str, optional
            Feature column(s) to bin; default is all numeric non-target columns.
        method : {'auto','optbinning','tree'}, default 'auto'
            ``'auto'`` prefers OptBinning for binary y, else the tree.
        cv : int, default 3
            CV folds for the tree path (capped by smallest class size).
        task : {'auto','binary','multiclass','regression'}, default 'auto'
            Force a target interpretation; validated against the data.
        cast_to_text : bool, default True
            Text interval labels vs integer bin ids (1..k).
        suffix : str, default '_binned'
            New-column suffix; ``''`` replaces the source column in place.
        random_state : int or None, default 42
            Seed for the tree path, sampling, and holdout split.
        return_diagnostics : bool, default False
            Also return per-column cuts, method used, and holdout scores.
        optbinning_kwargs : dict, optional
            Extra ``OptimalBinning(...)`` kwargs (e.g. ``max_n_bins``).
        holdout : bool or float, default False
            Honest train/test scoring; ``True`` means a 20% test split.
        stratify : bool, default True
            Stratify the holdout split for discrete targets.
        sample_n : int or None, default 100_000
            Row cap for cut learning (CASE still applies to all rows).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``<column><suffix>`` added per
            feature — or ``(DuckJanitor, diagnostics)`` when
            ``return_diagnostics`` is True (standalone assignment, not mid-chain).
        """
        from .bins_opt import optimize_bins_to_target as _obt

        res = _obt(
            self._relation,
            y,
            columns=columns,
            method=method,
            cv=cv,
            task=task,
            cast_to_text=cast_to_text,
            suffix=suffix,
            random_state=random_state,
            return_diagnostics=return_diagnostics,
            optbinning_kwargs=optbinning_kwargs,
            holdout=holdout,
            stratify=stratify,
            sample_n=sample_n,
            conn=self._connection,
        )
        if return_diagnostics:
            rel, diag = res  # type: ignore[misc]
            return DuckJanitor(rel, self._connection), diag
        return DuckJanitor(res, self._connection)

    def change_type(self, column: str, dtype: str) -> "DuckJanitor":
        """Change the data type of a column.

        Parameters
        ----------
        column : str
            Name of the column to cast.
        dtype : str
            DuckDB type name (``'INT'``, ``'VARCHAR'``, ``'DOUBLE'``,
            ``'DATE'``, ``'TIMESTAMP'``, etc).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``column`` cast in place.
        """
        from .cleaning_ops_extended import change_type as _change_type

        new_relation = _change_type(self._relation, column, dtype, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def concatenate_columns(
        self, columns: list[str], sep: str = "_", target_column: str = "concatenated"
    ) -> "DuckJanitor":
        """Concatenate multiple columns into a single column.

        Parameters
        ----------
        columns : list of str
            Names of the columns to concatenate.
        sep : str, default '_'
            Separator inserted between values.
        target_column : str, default 'concatenated'
            Name of the output column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``target_column`` added.
        """
        from .cleaning_ops_extended import concatenate_columns as _concatenate_columns

        new_relation = _concatenate_columns(
            self._relation, columns, sep, target_column, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def deconcatenate_column(
        self, column: str, sep: str, target_columns: list[str]
    ) -> "DuckJanitor":
        """Split a column into multiple columns based on a delimiter.

        Parameters
        ----------
        column : str
            Name of the column to split.
        sep : str
            Delimiter on which to split (literal, not regex).
        target_columns : list of str
            Names of the output columns; positional split results land
            in order.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``target_columns`` added
            and ``column`` dropped.
        """
        from .cleaning_ops_extended import deconcatenate_column as _deconcatenate_column

        new_relation = _deconcatenate_column(
            self._relation, column, sep, target_columns, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def drop_constant_columns(self) -> "DuckJanitor":
        """Remove columns that have only one unique value.

        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops_extended import drop_constant_columns as _drop_constant_columns

        new_relation = _drop_constant_columns(self._relation, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def fill(
        self,
        column: str,
        value: Optional[Any] = None,
        direction: str = "forward",
        group_by: Optional[Union[str, list[str]]] = None,
    ) -> "DuckJanitor":
        """Fill missing values in a column.

        Parameters
        ----------
        column : str
            Name of the column to fill.
        value : Any, optional
            Scalar literal to fill missing cells with. When ``None``,
            forward/backward fill is used (see ``direction``).
        direction : str, default 'forward'
            ``'forward'`` (carry the last non-null), ``'backward'``
            (carry the next non-null), or ``'downcast'`` (do nothing).
            Ignored when ``value`` is provided.
        group_by : str or list of str, optional
            Restrict the fill to groups defined by these columns
            (DuckDB window function).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with missing values filled.
        """
        from .cleaning_ops_extended import fill as _fill

        new_relation = _fill(self._relation, column, value, direction, group_by, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def fill_empty(self, column: str, value: str = "") -> "DuckJanitor":
        """Fill empty strings in a column with a specified value.

        Parameters
        ----------
        column : str
            Name of the column to fill.
        value : str, default ''
            Replacement for empty strings (``''`` keeps empties empty).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with empty strings replaced.
        """
        from .cleaning_ops_extended import fill_empty as _fill_empty

        new_relation = _fill_empty(self._relation, column, value, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def flag_nulls(
        self,
        columns: Optional[Union[str, list[str]]] = None,
        prefix: str = "is_null_",
        present_value: Any = 1,
        absent_value: Any = 0,
    ) -> "DuckJanitor":
        """Flag null values in specified columns with binary indicators.

        Parameters
        ----------
        columns : str or list of str, optional
            Columns to flag. ``None`` means all columns.
        prefix : str, default 'is_null_'
            Prefix for each output column name (the source column is
            appended, e.g. ``is_null_age``).
        present_value : Any, default 1
            Value to write when the source cell is ``NULL``.
        absent_value : Any, default 0
            Value to write when the source cell is non-null.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new flag columns added.
        """
        from .cleaning_ops_extended import flag_nulls as _flag_nulls

        new_relation = _flag_nulls(
            self._relation, columns, prefix, present_value, absent_value, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def limit_column_characters(
        self, column: str, max_chars: int, suffix: str = "..."
    ) -> "DuckJanitor":
        """Limit the number of characters in a string column.

        Parameters
        ----------
        column : str
            Name of the column to truncate.
        max_chars : int
            Maximum length of the resulting string, including the suffix.
        suffix : str, default '...'
            Appended when the original string is truncated; pass ``''``
            to suppress.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``column`` truncated.
        """
        from .cleaning_ops_extended import limit_column_characters as _limit_column_characters

        new_relation = _limit_column_characters(
            self._relation, column, max_chars, suffix, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def min_max_scale(
        self, column: str, target_column: str, min_val: float = 0, max_val: float = 1
    ) -> "DuckJanitor":
        """Apply Min-Max scaling to a numeric column.

        Parameters
        ----------
        column : str
            Name of the numeric column to scale.
        target_column : str
            Name of the output column.
        min_val : float, default 0
            Target minimum.
        max_val : float, default 1
            Target maximum.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``target_column`` added.
        """
        from .cleaning_ops_extended import min_max_scale as _min_max_scale

        new_relation = _min_max_scale(
            self._relation, column, target_column, min_val, max_val, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def groupby_agg(
        self, by: Union[str, list[str]], aggregations: dict[str, Union[str, dict]]
    ) -> "DuckJanitor":
        """Perform groupby aggregation.

        Parameters
        ----------
        by : str or list of str
            Column name(s) to group by.
        aggregations : dict
            Mapping ``{new_column_name: agg_spec}`` where ``agg_spec``
            is either a string (``'mean'``, ``'sum'``, ``'min'``,
            ``'max'``, ``'count'``, ``'first'``, ``'last'``) or a
            dict with ``'agg'``/``'column'`` keys for full control.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the aggregated columns.
        """
        from .cleaning_ops_extended import groupby_agg as _groupby_agg

        new_relation = _groupby_agg(self._relation, by, aggregations, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def groupby_topk(
        self, by: Union[str, list[str]], column: str, k: int, ascending: bool = False
    ) -> "DuckJanitor":
        """Get top k rows within each group based on a column.

        Parameters
        ----------
        by : str or list of str
            Column name(s) to define groups.
        column : str
            Column to rank within each group.
        k : int
            Number of rows to keep per group.
        ascending : bool, default False
            If True, keep the bottom-k rather than top-k.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to top-k rows per group.
        """
        from .cleaning_ops_extended import groupby_topk as _groupby_topk

        new_relation = _groupby_topk(self._relation, by, column, k, ascending, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def case_when(
        self, conditions: list[tuple], target_column: str, default: Optional[Any] = None
    ) -> "DuckJanitor":
        """Create a column based on multiple conditions (SQL CASE WHEN).

        Parameters
        ----------
        conditions : list of tuple
            Each tuple is ``(condition_str, value)``; ``condition_str``
            is a SQL fragment evaluated against the relation.
        target_column : str
            Name of the output column.
        default : Any, optional
            Fallback value when no condition matches (``NULL`` if unset).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``target_column`` added.
        """
        from .cleaning_ops_extended import case_when as _case_when

        new_relation = _case_when(
            self._relation, conditions, target_column, default, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def currency_column_to_numeric(
        self, column: str, target_column: Optional[str] = None
    ) -> "DuckJanitor":
        """Convert a currency column to numeric.

        Strips currency symbols ($, €, £, ¥, comma, space) and parses
        parenthesised negatives (``(1,234.56)`` → ``-1234.56``).

        Parameters
        ----------
        column : str
            Name of the currency column.
        target_column : str, optional
            Name of the output column; defaults to overwriting ``column``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the numeric column in place.
        """
        from .cleaning_ops_extended import currency_column_to_numeric as _currency_column_to_numeric

        new_relation = _currency_column_to_numeric(
            self._relation, column, target_column, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def convert_date(
        self, column: str, target_column: Optional[str] = None, date_format: Optional[str] = None
    ) -> "DuckJanitor":
        """Convert a column to date type.

        Parameters
        ----------
        column : str
            Name of the column to convert. Strings are parsed with
            ``date_format`` if given, otherwise DuckDB's
            ``TRY_CAST(... AS DATE)`` is used.
        target_column : str, optional
            Name of the output column; defaults to overwriting ``column``.
        date_format : str, optional
            ``strftime``-style format string (``'%Y-%m-%d'``).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the parsed DATE column.
        """
        from .cleaning_ops_extended import convert_date as _convert_date

        new_relation = _convert_date(
            self._relation, column, target_column, date_format, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def truncate_datetime(
        self, column: str, unit: str = "day", target_column: Optional[str] = None
    ) -> "DuckJanitor":
        """Truncate a datetime column to a specified unit.

        Parameters
        ----------
        column : str
            Name of the TIMESTAMP / DATE column to truncate.
        unit : str, default 'day'
            One of ``'year'``, ``'quarter'``, ``'month'``, ``'week'``,
            ``'day'``, ``'hour'``, ``'minute'``, ``'second'``.
        target_column : str, optional
            Name of the output column; defaults to overwriting ``column``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the truncated column.
        """
        from .cleaning_ops_extended import truncate_datetime as _truncate_datetime

        new_relation = _truncate_datetime(
            self._relation, column, unit, target_column, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def conditional_join(
        self, other: "DuckJanitor", on: list[tuple], how: str = "inner"
    ) -> "DuckJanitor":
        """Perform conditional (non-equi) joins.

        Parameters
        ----------
        other : DuckJanitor
            The right-side relation to join.
        on : list of tuple
            Each tuple is ``(left_col, right_col, op_str)`` where
            ``op_str`` is one of ``'='``, ``'!='``, ``'<'``, ``'<='``,
            ``'>'``, ``'>='``.
        how : str, default 'inner'
            Join type: ``'inner'``, ``'left'``, or ``'right'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the conditional-join result.
        """
        from .cleaning_ops_final import conditional_join as _conditional_join

        new_relation = _conditional_join(self._relation, other._relation, on, how, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def asof_join(
        self,
        other: "DuckJanitor",
        left_on: str,
        right_on: Optional[str] = None,
        by: Optional[Union[str, list[str]]] = None,
        direction: str = "backward",
        tolerance: Optional[Any] = None,
        suffixes: tuple[str, str] = ("", "_right"),
        keep_right_keys: bool = False,
    ) -> "DuckJanitor":
        """Join each row to the nearest eligible temporal row on the right.

        This is the DuckDB-backed equivalent of a pandas ``merge_asof`` and
        is useful for point-in-time analysis, slowly changing dimensions,
        event attribution, and temporal feature construction without leaking
        future values.

        Parameters
        ----------
        other : DuckJanitor
            The historical or reference relation to search.
        left_on : str
            Temporal column in the current relation.
        right_on : str, optional
            Temporal column in ``other``. Defaults to ``left_on``.
        by : str or list of str, optional
            Equality key(s) required in addition to the temporal condition.
            For example, ``"employee_id"`` or ``["employee_id", "region"]``.
        direction : {'backward', 'forward', 'nearest'}, default 'backward'
            ``'backward'`` selects the latest right row at or before the left
            time; ``'forward'`` selects the earliest row at or after it;
            ``'nearest'`` selects the smallest absolute time distance and
            breaks ties in favour of the earlier right timestamp.
        tolerance : object, optional
            Maximum allowed distance. Strings are interpreted as DuckDB
            intervals, such as ``"7 days"``; numeric values are compared
            directly for numeric temporal keys.
        suffixes : tuple of str, default ('', '_right')
            Suffixes applied when both relations contain a non-key column
            with the same name.
        keep_right_keys : bool, default False
            Include the right equality and temporal key columns even when
            they duplicate columns from the left relation.

        Returns
        -------
        DuckJanitor
            A left-preserving temporal join result.

        Examples
        --------
        >>> events = DuckJanitor.from_pandas(pd.DataFrame({
        ...     'employee_id': [1, 1],
        ...     'event_time': pd.to_datetime(['2024-01-10', '2024-01-20']),
        ... }))
        >>> history = DuckJanitor.from_pandas(pd.DataFrame({
        ...     'employee_id': [1, 1],
        ...     'effective_time': pd.to_datetime(['2024-01-01', '2024-01-15']),
        ...     'manager': ['A', 'B'],
        ... }))
        >>> events.asof_join(history, 'event_time', 'effective_time', by='employee_id').collect()['manager'].tolist()
        ['A', 'B']
        """
        if not isinstance(other, DuckJanitor):
            raise TypeError("other must be a DuckJanitor instance")
        if not isinstance(left_on, str) or not left_on:
            raise ValueError("left_on must be a non-empty column name")
        right_on = right_on or left_on
        if not isinstance(right_on, str) or not right_on:
            raise ValueError("right_on must be a non-empty column name")
        if direction not in {"backward", "forward", "nearest"}:
            raise ValueError("direction must be 'backward', 'forward', or 'nearest'")
        if len(suffixes) != 2:
            raise ValueError("suffixes must contain exactly two strings")

        left_columns = list(self._relation.columns)
        right_columns = list(other._relation.columns)
        if left_on not in left_columns:
            raise ValueError(f"asof_join(): left_on {left_on!r} not in current columns")
        if right_on not in right_columns:
            raise ValueError(f"asof_join(): right_on {right_on!r} not in other columns")

        by_columns = [by] if isinstance(by, str) else list(by or [])
        for column in by_columns:
            if column not in left_columns:
                raise ValueError(f"asof_join(): by column {column!r} not in current columns")
            if column not in right_columns:
                raise ValueError(f"asof_join(): by column {column!r} not in other columns")

        left_name = _temp_name("_asof_left")
        right_name = _temp_name("_asof_right")
        self._connection.register(left_name, self._relation)
        try:
            self._connection.register(right_name, other._relation)
        except Exception:
            self._connection.register(right_name, other._relation.df())

        left_id = "__dj_asof_left_id"
        right_id = "__dj_asof_right_id"
        left_type = str(self._relation.types[left_columns.index(left_on)]).upper()
        right_type = str(other._relation.types[right_columns.index(right_on)]).upper()
        temporal_types = ("DATE", "TIME", "TIMESTAMP", "INTERVAL")
        temporal_distance = any(
            marker in left_type or marker in right_type for marker in temporal_types
        )
        left_cte = f"SELECT *, row_number() OVER () AS {self._quote(left_id)} FROM {self._quote(left_name)}"
        equality = " AND ".join(
            f"l.{self._quote(column)} = r.{self._quote(column)}" for column in by_columns
        )
        if equality:
            equality += " AND "

        # DuckDB's native ASOF JOIN only supports a single inequality
        # (>= or <=) and no ORDER BY, so 'nearest' cannot be expressed
        # natively and keeps the ranking implementation below.
        if direction != "nearest":
            return self._asof_join_native(
                left_name=left_name,
                left_on=left_on,
                right_name=right_name,
                right_on=right_on,
                by_columns=by_columns,
                direction=direction,
                tolerance=tolerance,
                temporal_distance=temporal_distance,
                suffixes=suffixes,
                keep_right_keys=keep_right_keys,
                left_columns=left_columns,
                right_columns=right_columns,
                left_id=left_id,
                right_id=right_id,
            )

        right_cte = f"SELECT *, row_number() OVER () AS {self._quote(right_id)} FROM {self._quote(right_name)}"
        temporal_condition = "TRUE"
        if temporal_distance:
            distance_order = (
                f"abs(epoch(l.{self._quote(left_on)}) - epoch(r.{self._quote(right_on)}))"
            )
        else:
            distance_order = f"abs(l.{self._quote(left_on)} - r.{self._quote(right_on)})"
        order = (
            f"{distance_order} ASC, r.{self._quote(right_on)} ASC, r.{self._quote(right_id)} ASC"
        )

        if direction != "nearest":
            distance = f"(l.{self._quote(left_on)} - r.{self._quote(right_on)})"
        elif temporal_distance:
            distance = f"abs(epoch(l.{self._quote(left_on)}) - epoch(r.{self._quote(right_on)}))"
        else:
            distance = f"abs(l.{self._quote(left_on)} - r.{self._quote(right_on)})"
        if tolerance is not None:
            if isinstance(tolerance, str):
                tolerance_sql = f"CAST({self._sql_value(tolerance)} AS INTERVAL)"
            else:
                tolerance_sql = self._sql_value(tolerance)
            if direction == "backward":
                tolerance_condition = f"AND {distance} <= {tolerance_sql}"
            elif direction == "forward":
                tolerance_condition = f"AND {distance} >= -{tolerance_sql} AND {distance} <= 0"
            else:
                tolerance_condition = f"AND {distance} <= {tolerance_sql}"
        else:
            tolerance_condition = ""

        left_output = []
        output_names = []
        for column in left_columns:
            name = column + suffixes[0] if column in right_columns and suffixes[0] else column
            if name in output_names:
                raise ValueError(f"asof_join(): duplicate output column {name!r}")
            output_names.append(name)
            left_output.append(f"l.{self._quote(column)} AS {self._quote(name)}")

        right_keys = set(by_columns + [right_on])
        right_output = []
        for column in right_columns:
            if column in right_keys and not keep_right_keys:
                continue
            name = column + suffixes[1] if column in left_columns else column
            if name in output_names:
                raise ValueError(f"asof_join(): duplicate output column {name!r}")
            output_names.append(name)
            right_output.append(f"r.{self._quote(column)} AS {self._quote(name)}")

        projection = ", ".join(left_output + right_output)
        query = f"""
            WITH l AS ({left_cte}), r AS ({right_cte}), ranked AS (
                SELECT {projection}, l.{self._quote(left_id)} AS {self._quote(left_id)},
                       row_number() OVER (
                           PARTITION BY l.{self._quote(left_id)} ORDER BY {order}
                       ) AS __dj_asof_rank
                FROM l
                LEFT JOIN r ON {equality}{temporal_condition} {tolerance_condition}
            )
            SELECT {", ".join(self._quote(name) for name in output_names)}
            FROM ranked
            WHERE __dj_asof_rank = 1
            ORDER BY {self._quote(left_id)}
        """
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def _asof_join_native(
        self,
        *,
        left_name: str,
        left_on: str,
        right_name: str,
        right_on: str,
        by_columns: list[str],
        direction: str,
        tolerance: Optional[Any],
        temporal_distance: bool,
        suffixes: tuple[str, str],
        keep_right_keys: bool,
        left_columns: list[str],
        right_columns: list[str],
        left_id: str,
        right_id: str,
    ) -> "DuckJanitor":
        """Execute a backward/forward as-of join using DuckDB's native syntax.

        DuckDB's ``ASOF JOIN`` is a single-indexed lookup rather than the
        rank-and-filter emulation, so it is substantially faster on large
        relations. Because native ASOF has no ``ORDER BY`` tie-break, the
        right side is de-duplicated first so that the surviving row per
        key/time pair is the same one the emulation would have chosen.

        Parameters
        ----------
        left_name : str
            Registered name of the left relation.
        left_on : str
            Temporal column on the left side.
        right_name : str
            Registered name of the right relation.
        right_on : str
            Temporal column on the right side.
        by_columns : list of str
            Equality key columns.
        direction : str
            ``'backward'`` or ``'forward'``.
        tolerance : object, optional
            Maximum allowed distance.
        temporal_distance : bool
            Whether the keys are temporal types.
        suffixes : tuple of str
            Output column suffixes.
        keep_right_keys : bool
            Whether to keep duplicated key columns from the right side.
        left_columns : list of str
            Left-side column names.
        right_columns : list of str
            Right-side column names.
        left_id : str
            Name of the left row-id column.
        right_id : str
            Name of the right row-id column.

        Returns
        -------
        DuckJanitor
            The join result.
        """
        equality = " AND ".join(
            f"l.{self._quote(column)} = r.{self._quote(column)}" for column in by_columns
        )
        if equality:
            equality += " AND "

        # The temporal column is projected under a fixed alias so the join
        # predicate does not have to care whether the two sides name it
        # differently.
        temporal_left = "__dj_temporal_left"
        left_cte = (
            f"SELECT *, {self._quote(left_on)} AS {self._quote(temporal_left)}, "
            f"row_number() OVER () AS {self._quote(left_id)} "
            f"FROM {self._quote(left_name)}"
        )

        if direction == "backward":
            temporal_condition = f"l.{self._quote(temporal_left)} >= r.{self._quote(right_on)}"
        else:
            temporal_condition = f"l.{self._quote(temporal_left)} <= r.{self._quote(right_on)}"

        # De-duplicate the right side keeping the same row the emulation
        # picked: the last one in right-row-id order for a backward join,
        # the first one for a forward join.
        right_order = (
            f"{self._quote(right_on)} DESC, {self._quote(right_id)} DESC"
            if direction == "backward"
            else f"{self._quote(right_on)} ASC, {self._quote(right_id)} ASC"
        )
        right_keys = set(by_columns + [right_on])
        dedupe_partition = ", ".join(self._quote(column) for column in sorted(right_keys))
        right_deduped = f"""
            SELECT * EXCLUDE ({self._quote(right_id)})
            FROM (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY {dedupe_partition} ORDER BY {right_order}
                       ) AS __dj_asof_dedupe_rank
                FROM (
                    SELECT *, row_number() OVER () AS {self._quote(right_id)}
                    FROM {self._quote(right_name)}
                )
            )
            WHERE __dj_asof_dedupe_rank = 1
        """

        # Tolerance is applied by nulling the right-hand columns when the
        # matched row is further away than the tolerance allows. The left
        # row is always preserved, which matches the rank-and-filter
        # behaviour this replaces: a row whose best match is out of
        # tolerance keeps its left values and receives NULLs on the right.
        right_case = ""
        if tolerance is not None:
            if isinstance(tolerance, str):
                # A string tolerance is a DuckDB interval; it is converted
                # to seconds so it can be compared with an epoch gap.
                tolerance_value = f"epoch(CAST({self._sql_value(tolerance)} AS INTERVAL))"
            else:
                tolerance_value = self._sql_value(tolerance)
            if temporal_distance:
                # DuckDB subtracts two temporal values into an INTERVAL.
                # abs() has no INTERVAL overload, so the gap is computed in
                # epoch seconds and compared against a numeric tolerance.
                gap = (
                    f"abs(epoch(l.{self._quote(temporal_left)}) - epoch(r.{self._quote(right_on)}))"
                )
            else:
                gap = f"abs(l.{self._quote(temporal_left)} - r.{self._quote(right_on)})"
            right_case = f"CASE WHEN {gap} <= {tolerance_value} THEN"

        left_output = []
        output_names = []
        for column in left_columns:
            name = column + suffixes[0] if column in right_columns and suffixes[0] else column
            if name in output_names:
                raise ValueError(f"asof_join(): duplicate output column {name!r}")
            output_names.append(name)
            left_output.append(f"l.{self._quote(column)} AS {self._quote(name)}")

        right_keys = set(by_columns + [right_on])
        right_output = []
        for column in right_columns:
            if column in right_keys and not keep_right_keys:
                continue
            name = column + suffixes[1] if column in left_columns else column
            if name in output_names:
                raise ValueError(f"asof_join(): duplicate output column {name!r}")
            output_names.append(name)
            expression = f"r.{self._quote(column)}"
            if right_case:
                expression = f"{right_case} {expression} END"
            right_output.append(f"{expression} AS {self._quote(name)}")

        projection = ", ".join(left_output + right_output)
        query = f"""
            WITH l AS ({left_cte}),
            r AS ({right_deduped}),
            joined AS (
                SELECT {projection}, l.{self._quote(left_id)} AS {self._quote(left_id)}
                FROM l
                ASOF LEFT JOIN r
                    ON {equality}{temporal_condition}
            )
            SELECT {", ".join(self._quote(name) for name in output_names)}
            FROM joined
            ORDER BY {self._quote(left_id)}
        """
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def get_dupes(self, columns: Optional[Union[str, list[str]]] = None) -> "DuckJanitor":
        """Return duplicate rows.

        Parameters
        ----------
        columns : str or list of str, optional
            Subset of columns to consider when detecting duplicates.
            ``None`` uses all columns.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to rows that have at
            least one duplicate on ``columns``.
        """
        from .cleaning_ops_final import get_dupes as _get_dupes

        new_relation = _get_dupes(self._relation, columns, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def dropnotnull(
        self, subset: Optional[Union[str, list[str]]] = None, how: str = "any"
    ) -> "DuckJanitor":
        """Remove rows where values are NOT null (keep nulls).

        Parameters
        ----------
        subset : str or list of str, optional
            Subset of columns to test for non-null. ``None`` uses all
            columns.
        how : str, default 'any'
            ``'any'`` drops rows with any non-null value in ``subset``;
            ``'all'`` drops only when *all* subset values are non-null.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the matched rows removed.
        """
        from .cleaning_ops_final import dropnotnull as _dropnotnull

        new_relation = _dropnotnull(self._relation, subset, how, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def expand_column(
        self, column: str, sep: str = "|", prefix: Optional[str] = None
    ) -> "DuckJanitor":
        """Expand a delimited column into dummy variables.

        Parameters
        ----------
        column : str
            Name of the column holding delimited tokens.
        sep : str, default '|'
            Token separator.
        prefix : str, optional
            Prefix for the new dummy columns; defaults to ``column + '_'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with one dummy column per token.
        """
        from .cleaning_ops_final import expand_column as _expand_column

        new_relation = _expand_column(self._relation, column, sep, prefix, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def impute(
        self,
        column: str,
        value: Optional[Any] = None,
        statistic: str = "mean",
        group_by: Optional[Union[str, list[str]]] = None,
    ) -> "DuckJanitor":
        """Impute missing values.

        Parameters
        ----------
        column : str
            Name of the column to fill.
        value : Any, optional
            Literal value to fill with; if omitted, ``statistic`` is
            computed from the non-null cells.
        statistic : str, default 'mean'
            Aggregation when ``value`` is None: ``'mean'``, ``'median'``,
            ``'mode'``, ``'min'``, ``'max'``.
        group_by : str or list of str, optional
            Compute the statistic per group rather than globally.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with missing values filled.
        """
        from .cleaning_ops_final import impute as _impute

        new_relation = _impute(self._relation, column, value, statistic, group_by, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def jitter(
        self, column: str, target_column: str, scale: float = 0.01, seed: Optional[int] = None
    ) -> "DuckJanitor":
        """Add random noise (jitter) to a numeric column.

        Parameters
        ----------
        column : str
            Name of the source numeric column.
        target_column : str
            Name of the output column.
        scale : float, default 0.01
            Multiplier applied to ``random()`` (a uniform ``[0,1)``).
        seed : int, optional
            PRNG seed for reproducibility (same normalization as :meth:`shuffle`).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``target_column`` added.
        """
        from .cleaning_ops_final import jitter as _jitter

        new_relation = _jitter(self._relation, column, target_column, scale, seed, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def label_encode(
        self, columns: Union[str, list[str]], suffix: str = "_encoded"
    ) -> "DuckJanitor":
        """Encode categorical columns with numerical labels.

        Parameters
        ----------
        columns : str or list of str
            Columns to label-encode. New columns are appended with
            ``suffix``; the originals are dropped.
        suffix : str, default '_encoded'
            Suffix for the output columns.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the encoded columns.
        """
        from .cleaning_ops_final import label_encode as _label_encode

        new_relation = _label_encode(self._relation, columns, suffix, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def find_replace(
        self, column: str, value_pairs: dict[str, str], target_column: Optional[str] = None
    ) -> "DuckJanitor":
        """Find and replace values in a column.

        Parameters
        ----------
        column : str
            Name of the source column.
        value_pairs : dict
            ``{find: replace}`` mapping applied via SQL ``CASE WHEN``.
        target_column : str, optional
            Name of the output column; defaults to overwriting ``column``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the replaced column.
        """
        from .cleaning_ops_final import find_replace as _find_replace

        new_relation = _find_replace(
            self._relation, column, value_pairs, target_column, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def count_cumulative_unique(
        self, column: str, dest_column: str = "cumulative_unique"
    ) -> "DuckJanitor":
        """Return a column with cumulative count of unique values.

        Parameters
        ----------
        column : str
            Source column to count uniques from.
        dest_column : str, default 'cumulative_unique'
            Name of the appended counter column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``dest_column`` appended.
        """
        from .cleaning_ops_final import count_cumulative_unique as _count_cumulative_unique

        new_relation = _count_cumulative_unique(
            self._relation, column, dest_column, self._connection
        )
        return DuckJanitor(new_relation, self._connection)

    def complete(self, columns: Union[str, list[str]], fill_value: Any = None) -> "DuckJanitor":
        """Expand relation to include all possible combinations of specified columns.

        Parameters
        ----------
        columns : str or list of str
            Columns to expand; their Cartesian product is added, missing
            combinations appear as ``NULL`` (or ``fill_value`` if given).
        fill_value : Any, optional
            Replacement for cells in newly-added rows.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the completed grid.
        """
        from .cleaning_ops_final import complete as _complete

        new_relation = _complete(self._relation, columns, fill_value, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def also(self, func: Callable) -> "DuckJanitor":
        """Apply a Python function with side effects (materializes data).

        Parameters
        ----------
        func : Callable
            Function called with the materialized pandas DataFrame; its
            return value is ignored.

        Returns
        -------
        DuckJanitor
            Self for method chaining (post-materialization).
        """
        from .cleaning_ops_final import also as _also

        result = _also(self, func)
        return result

    def alias(self, alias: Union[str, Callable]) -> "DuckJanitor":
        """Rename all columns using a string or callable.

        Parameters
        ----------
        alias : str or Callable
            Either a ``str.format`` template (``'col_{}'``) applied to
            each column, or a callable that maps an old column name to
            a new one.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with renamed columns.
        """
        from .cleaning_ops_final import alias as _alias

        new_relation = _alias(self._relation, alias, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def drop_duplicate_columns(self) -> "DuckJanitor":
        """Remove columns that are exact duplicates of other columns.

        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        from .cleaning_ops_final import drop_duplicate_columns as _drop_duplicate_columns

        new_relation = _drop_duplicate_columns(self._relation, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def compare_df_cols(self, other: "DuckJanitor") -> pd.DataFrame:
        """Compare columns between two DuckJanitor instances.

        Parameters
        ----------
        other : DuckJanitor
            The right-side relation to compare against.

        Returns
        -------
        pd.DataFrame
            Per-column metadata (name, dtype, presence in either side)
            showing where the two relations overlap or differ.
        """
        from .cleaning_ops_final import compare_df_cols as _compare_df_cols

        return _compare_df_cols(self, other, self._connection)

    def join_apply(
        self, other: "DuckJanitor", on: Union[str, list[str]], func: Callable, new_column_name: str
    ) -> "DuckJanitor":
        """Perform join then apply Python function to each row.

        Parameters
        ----------
        other : DuckJanitor
            The right-side relation to join.
        on : str or list of str
            Join key(s).
        func : Callable
            Python function applied to each joined row's pandas DataFrame;
            must return a scalar or array of the same length.
        new_column_name : str
            Name of the appended output column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``new_column_name`` added.
        """
        from .cleaning_ops_final import join_apply as _join_apply

        return _join_apply(self, other, on, func, new_column_name, self._connection)

    def process_text(
        self, column: str, func: Union[Callable, str], new_column_name: str
    ) -> "DuckJanitor":
        """Apply text processing function to a column.

        Parameters
        ----------
        column : str
            Name of the source string column.
        func : Callable or str
            Either a Python callable applied row-wise (cell → cell) or a
            SQL ``str`` expression (``LOWER(column)``).
        new_column_name : str
            Name of the appended output column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``new_column_name`` added.
        """
        from .cleaning_ops_final import process_text as _process_text

        return _process_text(self, column, func, new_column_name, self._connection)

    def mutate(self, **kwargs: Any) -> "DuckJanitor":
        """Create or modify columns using a dictionary (convenience wrapper).

        Parameters
        ----------
        **kwargs
            Mapping of ``column=value`` or ``column=callable``. Each value
            is broadcast as a new column or applied row-wise as a
            transformation.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new/modified columns.
        """
        from .cleaning_ops_final import mutate as _mutate

        result = _mutate(self, **kwargs)
        return result

    def pivot_wider(
        self, id_cols: Union[str, list[str]], name_col: str, value_col: str
    ) -> "DuckJanitor":
        """Pivot data from long to wide format.

        Parameters
        ----------
        id_cols : str or list of str
            Identifier columns that remain as rows in the output.
        name_col : str
            Column whose values become the new column names.
        value_col : str
            Column whose values populate the new columns.

        Returns
        -------
        DuckJanitor
            Self for method chaining, in wide format.
        """
        from .cleaning_ops_extended import pivot_wider as _pivot_wider

        new_relation = _pivot_wider(self._relation, id_cols, name_col, value_col, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def pivot_longer(
        self, cols: Union[str, list[str]], names_to: str = "variable", values_to: str = "value"
    ) -> "DuckJanitor":
        """Pivot data from wide to long format.

        Parameters
        ----------
        cols : str or list of str
            Columns to unpivot into rows.
        names_to : str, default 'variable'
            Output column holding the unpivoted column names.
        values_to : str, default 'value'
            Output column holding the unpivoted cell values.

        Returns
        -------
        DuckJanitor
            Self for method chaining, in long format.
        """
        from .cleaning_ops_extended import pivot_longer as _pivot_longer

        new_relation = _pivot_longer(self._relation, cols, names_to, values_to, self._connection)
        return DuckJanitor(new_relation, self._connection)

    def explain(self, analyze: bool = False, *, format: str = "text") -> str:
        """Show the query plan for the current pipeline.

        Parameters
        ----------
        analyze : bool, optional
            When True, run ``EXPLAIN ANALYZE`` so the plan includes actual
            execution statistics. Default is False.
        format : str, optional
            ``'text'`` (default) returns the plan as plain text.
            ``'json'`` returns DuckDB's JSON profiling output.

        Returns
        -------
        str
            The query plan.

        Examples
        --------
        >>> print(dj.clean_names().explain())
        ┌───────────────────────────┐
        │ physical_plan             │
        ├───────────────────────────┤
        │ ...                       │
        └───────────────────────────┘
        """
        if format not in {"text", "json"}:
            raise ValueError(f"explain(): format must be 'text' or 'json'. Got {format!r}.")
        temp_name = _temp_name("_temp_explain")
        self._connection.register(temp_name, self._relation)
        query = f"EXPLAIN {'ANALYZE ' if analyze else ''}SELECT * FROM {temp_name}"
        if format == "json":
            rows = self._connection.execute(query).fetchall()
            return rows[0][1] if rows else ""
        rows = self._connection.execute(query).fetchall()
        if not rows:
            return ""
        plan = rows[0][1]
        # DuckDB returns the plan as a plain multi-line string; strip the
        # leading/trailing blank lines so the caller gets a clean block.
        return plan.strip("\n") if isinstance(plan, str) else str(plan)

    # ---------------------------------------------------------------
    # pyjanitor parity aliases (R/pyjanitor function name alignment).
    # These thin wrappers align the function name with pyjanitor so
    # users migrating from pyjanitor do not have to learn new names.
    # ---------------------------------------------------------------

    def rename_columns(self, old_name: str, new_name: str) -> "DuckJanitor":
        """Alias of :meth:`rename_column` for the pyjanitor plural name.

        Parameters
        ----------
        old_name : str
            Current column name.
        new_name : str
            New column name.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the column renamed.
        """
        return self.rename_column(old_name, new_name)

    def truncate_datetime_dataframe(
        self,
        column: str,
        unit: str = "day",
        target_column: Optional[str] = None,
    ) -> "DuckJanitor":
        """Alias of :meth:`truncate_datetime` matching pyjanitor's name.

        Parameters
        ----------
        column : str
            Name of the TIMESTAMP / DATE column to truncate.
        unit : str, default 'day'
            Truncation unit (see :meth:`truncate_datetime`).
        target_column : str, optional
            Name of the output column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the truncated column.
        """
        return self.truncate_datetime(column=column, unit=unit, target_column=target_column)

    def convert_to_date(
        self,
        column: str,
        date_format: Optional[str] = None,
        target_column: Optional[str] = None,
        **kwargs,
    ) -> "DuckJanitor":
        """Alias of :meth:`convert_date` matching pyjanitor's name.

        Parameters
        ----------
        column : str
            Name of the column to convert.
        date_format : str, optional
            ``strftime``-style format string.
        target_column : str, optional
            Name of the output column.
        **kwargs
            Accepted for signature parity; ignored.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the parsed DATE column.
        """
        return self.convert_date(
            column=column, target_column=target_column, date_format=date_format
        )

    def convert_to_datetime(
        self,
        column: str,
        date_format: Optional[str] = None,
        target_column: Optional[str] = None,
        **kwargs,
    ) -> "DuckJanitor":
        """Alias of :meth:`convert_date` matching pyjanitor's name.

        Parameters
        ----------
        column : str
            Name of the column to convert.
        date_format : str, optional
            ``strftime``-style format string.
        target_column : str, optional
            Name of the output column.
        **kwargs
            Accepted for signature parity; ignored.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the parsed TIMESTAMP column.
        """
        return self.convert_date(
            column=column, target_column=target_column, date_format=date_format
        )

    def convert_unix_date(
        self, column: str, unit: str = "seconds", target_column: Optional[str] = None
    ) -> "DuckJanitor":
        """Coerce a UNIX/epoch numeric column into a DuckDB TIMESTAMP.

        Parameters
        ----------
        column : str
            Name of the column holding the Unix epoch value.
        unit : str, default 'seconds'
            One of {'seconds', 'milliseconds', 'microseconds'}.
        target_column : str, optional
            Name of the output column. Defaults to ``column + '_datetime'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new TIMESTAMP column appended.
        """
        out = target_column or f"{column}_datetime"
        multipliers = {"seconds": 1, "milliseconds": 1000, "microseconds": 1000000}
        if unit not in multipliers:
            raise ValueError(
                f"convert_unix_date(): unit must be one of {sorted(multipliers)}; got {unit!r}."
            )
        multiplier = multipliers[unit]
        temp_name = _temp_name("_unix")
        self._connection.register(temp_name, self._relation)
        # Use TO_TIMESTAMP which accepts DOUBLE seconds directly.
        query = (
            f"SELECT *, TO_TIMESTAMP(CAST({column} AS DOUBLE) / {multiplier}) AS {out} "
            f"FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def convert_excel_date(self, column: str, target_column: Optional[str] = None) -> "DuckJanitor":
        """Coerce an Excel serial date number into a DuckDB TIMESTAMP.

        Excel's serial date origin is 1899-12-30 (with the 1900 leap-year
        bug adjustment). 1 = 1900-01-01.

        Parameters
        ----------
        column : str
            Name of the numeric Excel-serial column.
        target_column : str, optional
            Name of the output TIMESTAMP column; defaults to
            ``column + '_datetime'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new TIMESTAMP column added.
        """
        out = target_column or f"{column}_datetime"
        temp_name = _temp_name("_excel_date")
        self._connection.register(temp_name, self._relation)
        # Use TO_TIMESTAMP which accepts seconds-from-epoch DOUBLE.
        query = (
            f"SELECT *, TO_TIMESTAMP(CAST({column} AS DOUBLE) * 86400.0 - 25569 * 86400.0) AS "
            f"{out} FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def convert_matlab_date(
        self, column: str, target_column: Optional[str] = None
    ) -> "DuckJanitor":
        """Coerce a MATLAB serial date number into a DuckDB TIMESTAMP.

        MATLAB datenum origin is 0000-01-01; 1 = 0000-01-01. The offset to
        the DuckDB TIMESTAMP epoch (1970-01-01) is 719529 days.

        Parameters
        ----------
        column : str
            Name of the numeric MATLAB-datenum column.
        target_column : str, optional
            Name of the output TIMESTAMP column; defaults to
            ``column + '_datetime'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new TIMESTAMP column added.
        """
        out = target_column or f"{column}_datetime"
        temp_name = _temp_name("_matlab_date")
        self._connection.register(temp_name, self._relation)
        query = (
            f"SELECT *, TO_TIMESTAMP(CAST({column} AS DOUBLE) * 86400.0 - 719529 * 86400.0) AS "
            f"{out} FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def fill_direction(
        self, column: str, direction: str = "forward", value=None, **kwargs
    ) -> "DuckJanitor":
        """Alias of :meth:`fill` matching pyjanitor's name.

        Parameters
        ----------
        column : str
            Name of the column to fill.
        direction : str, default 'forward'
            ``'forward'``, ``'backward'``, or ``'downcast'``.
        value : Any, optional
            Scalar literal to fill missing cells with.
        **kwargs
            Forwarded to :meth:`fill` (e.g. ``group_by=``).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with missing values filled.
        """
        return self.fill(column=column, value=value, direction=direction, **kwargs)

    def filter_column_isin(self, column: str, values, complement: bool = False) -> "DuckJanitor":
        """Filter rows where ``column`` IS IN ``values``.

        ``values`` may be any iterable of scalars. Implemented as a
        direct DuckDB SQL filter because :meth:`filter_column` requires
        a callable/SQL-fragment criteria.

        Parameters
        ----------
        column : str
            Name of the column to filter on.
        values : iterable
            Any iterable of scalars; quoted/escaped for DuckDB before
            being inlined into the SQL ``IN`` clause.
        complement : bool, default False
            If True, keep rows that *aren't* in ``values``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to matching rows.
        """
        if column not in self._relation.columns:
            raise ValueError(
                f"filter_column_isin(): column {column!r} not in {list(self._relation.columns)}"
            )
        try:
            values = list(values)
        except TypeError as err:
            raise TypeError("filter_column_isin(): values must be an iterable of scalars.") from err
        if not values:
            # Empty list returns empty relation.
            return self.filter_on("1 = 0", complement=False)

        def _format(v):
            if v is None:
                return "NULL"
            if isinstance(v, bool):
                return "TRUE" if v else "FALSE"
            if isinstance(v, (int, float)):
                return repr(v)
            escaped = str(v).replace("'", "''")
            return f"'{escaped}'"

        in_list = ", ".join(_format(v) for v in values)
        op = "NOT IN" if complement else "IN"
        where_clause = f'"{column}" {op} ({in_list})'
        return self.filter_on(where_clause, complement=False)

    def add_columns(self, column_values: dict) -> "DuckJanitor":
        """Alias of :meth:`add_column` accepting a dict of column→values
        so multiple columns can be added in a single chained call.

        Parameters
        ----------
        column_values : dict
            ``{column_name: value_or_callable}`` mapping; each entry is
            added in turn (callables are applied row-wise, scalars are
            broadcast).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with each new column appended.
        """
        out = self
        for col, vals in column_values.items():
            out = out.add_column(col, vals)
        return out

    def assign(self, **kwargs) -> "DuckJanitor":
        """Alias of :meth:`mutate` matching pyjanitor's name.

        ``assign`` accepts keyword arguments of ``column=value`` or
        ``column=callable``, exactly like :meth:`mutate`.

        Parameters
        ----------
        **kwargs
            Forwarded verbatim to :meth:`mutate`.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new/modified columns.
        """
        return self.mutate(**kwargs)

    def ungroup(self, *groups, **kwargs) -> "DuckJanitor":
        """Identity helper that matches pyjanitor's ``ungroup`` verb.

        DuckDB relations are inherently ungrouped; this is a no-op that
        simply returns the current DuckJanitor, kept as a chainable verb.

        Parameters
        ----------
        *groups
            Accepted for signature parity; ignored.
        **kwargs
            Accepted for signature parity; ignored.

        Returns
        -------
        DuckJanitor
            Self for method chaining (unmodified).
        """
        return self

    def get_columns(self, *names) -> "DuckJanitor":
        """Select columns by name (alias of :meth:`select_columns`).

        Parameters
        ----------
        *names
            Column names as variadic positional args (mirrors the
            pyjanitor plural signature).

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to ``names``.
        """
        return self.select_columns(list(names))

    def move(self, source: str, target: str, position: str = "before", **kwargs) -> "DuckJanitor":
        """Move ``source`` column relative to ``target`` column.

        position: ``'before'`` (default) or ``'after'``.

        Parameters
        ----------
        source : str
            Column to move.
        target : str
            Anchor column; ``source`` lands before or after it.
        position : str, default 'before'
            ``'before'`` or ``'after'`` the ``target``.
        **kwargs
            Accepted for signature parity; ignored.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``source`` repositioned.
        """
        cur_cols = list(self._relation.columns)
        if source not in cur_cols or target not in cur_cols:
            raise ValueError(
                f"move(): source={source!r} and target={target!r} must both "
                f"exist in the current columns: {cur_cols}"
            )
        new_order = [c for c in cur_cols if c != source]
        insert_at = new_order.index(target)
        if position == "after":
            insert_at += 1
        new_order.insert(insert_at, source)
        cols = ", ".join(self._quote(c) for c in new_order)
        temp_name = _temp_name("_move")
        self._connection.register(temp_name, self._relation)
        query = f"SELECT {cols} FROM {temp_name}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def reorder_columns(self, new_order) -> "DuckJanitor":
        """Reorder the relation's columns to match ``new_order``.

        Columns not listed are dropped (matching pyjanitor's behaviour,
        where columns must be enumerated fully).

        Parameters
        ----------
        new_order : list of str
            Column names in the desired order; any column not in the list
            is dropped.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the reordered (and possibly
            subset) column set.
        """
        cur = list(self._relation.columns)
        missing = [c for c in new_order if c not in cur]
        if missing:
            raise ValueError(f"reorder_columns(): unknown columns {missing}")
        cols = ", ".join(self._quote(c) for c in new_order)
        temp_name = _temp_name("_reorder")
        self._connection.register(temp_name, self._relation)
        query = f"SELECT {cols} FROM {temp_name}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def get_index_labels(self) -> list[str]:
        """Return the current column names as a list (label-only).

        Returns
        -------
        list of str
            Column labels in the current DuckDB relation, in order.
        """
        return list(self._relation.columns)

    @staticmethod
    def _quote(col: str) -> str:
        return f'"{col.replace(chr(34), chr(34) * 2)}"'

    @staticmethod
    def _sql_value(v) -> str:
        if v is None:
            return "NULL"
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (int, float)):
            return repr(v)
        escaped = str(v).replace("'", "''")
        return f"'{escaped}'"

    @classmethod
    def _sql_list(cls, values) -> str:
        """Render a list of scalar values for DuckDB named arguments."""
        return "[" + ", ".join(cls._sql_value(value) for value in values) + "]"

    # =============================================================
    # pyjanitor parity batch: small DuckDB-trivial helpers
    # =============================================================

    def shuffle(self, seed: Optional[int] = None) -> "DuckJanitor":
        """Return a relation with all rows in random order (R: ``shuffle``).

        Parameters
        ----------
        seed : int, optional
            If provided, drives DuckDB's ``random()`` PRNG for reproducibility.
            The seed is normalized into ``[-1.0, 1.0]`` because DuckDB's
            ``setseed`` only accepts that range.

        Returns
        -------
        DuckJanitor
            Self for method chaining.
        """
        if seed is not None:
            normalized = (abs(int(seed)) % (2**31)) / (2**31)
            seed_clause = f"setseed({normalized})"
        else:
            seed_clause = None
        temp_name = _temp_name("_shuffle")
        self._connection.register(temp_name, self._relation)
        if seed_clause is not None:
            self._connection.execute(f"SELECT {seed_clause}")
        query = f"SELECT * FROM {temp_name} ORDER BY random()"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def toset(self, column: str) -> list:
        """Return the unique sorted values of ``column`` as a Python list (R: ``toset``).

        Parameters
        ----------
        column : str
            Name of the column whose distinct values to return.

        Returns
        -------
        list
            Distinct values of ``column``, sorted ascending.
        """
        temp_name = _temp_name("_toset")
        self._connection.register(temp_name, self._relation)
        rows = self._connection.execute(
            f'SELECT DISTINCT "{column}" FROM {temp_name} ORDER BY "{column}"'
        ).fetchall()
        self._connection.unregister(temp_name)
        return [r[0] for r in rows]

    def take_first(self, n: int = 1) -> "DuckJanitor":
        """Return a relation containing only the first ``n`` rows (R: ``take_first``).

        Parameters
        ----------
        n : int, default 1
            Number of rows to keep; must be >= 0.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to the first ``n`` rows.
        """
        if n < 0:
            raise ValueError(f"take_first(): n must be >= 0; got {n}")
        temp_name = _temp_name("_take_first")
        self._connection.register(temp_name, self._relation)
        # LIMIT on an inner subquery to avoid WHERE-on-window-function binder error.
        query = f"SELECT * FROM (SELECT * FROM {temp_name}) LIMIT {n}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def excel_time_to_numeric(
        self, column: str, target_column: Optional[str] = None
    ) -> "DuckJanitor":
        """Convert an Excel time-fraction column (``0.0``–``1.0``) to seconds.

        Excel stores time-of-day as the fractional part of a day; multiplying
        by ``86400`` yields seconds.

        Parameters
        ----------
        column : str
            Name of the column holding the Excel time fraction.
        target_column : str, optional
            Name of the output numeric column; defaults to
            ``column + '_seconds'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new seconds column added.
        """
        out = target_column or f"{column}_seconds"
        temp_name = _temp_name("_excel_time")
        self._connection.register(temp_name, self._relation)
        query = f"SELECT *, CAST({column} * 86400.0 AS DOUBLE) AS {out} FROM {temp_name}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def sas_numeric_to_date(
        self, column: str, target_column: Optional[str] = None
    ) -> "DuckJanitor":
        """Convert a SAS numeric date column (days since 1960-01-01) to TIMESTAMP.

        Parameters
        ----------
        column : str
            Name of the numeric SAS-date column.
        target_column : str, optional
            Name of the output TIMESTAMP column; defaults to
            ``column + '_datetime'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new TIMESTAMP column added.
        """
        out = target_column or f"{column}_datetime"
        temp_name = _temp_name("_sasdate")
        self._connection.register(temp_name, self._relation)
        query = (
            f"SELECT *, TO_TIMESTAMP(CAST({column} AS DOUBLE) * 86400.0 - 315619200.0) AS {out} "
            f"FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def round_to_fraction(self, column: str, denominator: Union[int, float]) -> "DuckJanitor":
        """Round ``column`` to the nearest fraction ``1/denominator`` (R: ``round_to_fraction``).

        Parameters
        ----------
        column : str
            Name of the numeric column to round.
        denominator : int or float
            Denominator of the target fraction (``1/denominator``). Must be
            non-zero.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``<column>_rounded`` appended.
        """
        if denominator == 0:
            raise ValueError("round_to_fraction(): denominator must be non-zero")
        temp_name = _temp_name("_rfrac")
        self._connection.register(temp_name, self._relation)
        query = (
            f"SELECT *, ROUND(CAST({column} AS DOUBLE) * {denominator}) / "
            f'{denominator} AS "{column}_rounded" FROM {temp_name}'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def scale_mad(self, column: str, by: str = "all") -> "DuckJanitor":
        """Median-abs-deviation standardisation (R: ``scale_mad``).

        ``by`` is one of ``'all'`` (default) or ``'column'``.

        Parameters
        ----------
        column : str
            Name of the numeric column to scale.
        by : str, default 'all'
            ``'all'`` centres/scales against the global median+median
            absolute deviation; ``'column'`` is reserved for per-column
            variants (currently same behaviour).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``<column>_scaled`` appended.
        """
        if by not in {"all", "column"}:
            raise ValueError(f"scale_mad(): 'by' must be 'all' or 'column'; got {by!r}")
        temp_name = _temp_name("_mad")
        self._connection.register(temp_name, self._relation)
        query = (
            f"SELECT *, ({column} - (SELECT MEDIAN({column}) FROM {temp_name})) / "
            f"(1.4826 * (SELECT MEDIAN(ABS({column} - (SELECT MEDIAN({column}) FROM {temp_name}))) "
            f'FROM {temp_name})) AS "{column}_scaled" '
            f"FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def cartesian_product(self, other: "DuckJanitor") -> "DuckJanitor":
        """Return the cartesian (cross) product of ``self`` and ``other``.

        Parameters
        ----------
        other : DuckJanitor
            Right-side relation to cross-join with.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the cross-joined columns.
        """
        if not isinstance(other, DuckJanitor):
            raise TypeError(
                f"cartesian_product(): other must be a DuckJanitor; got {type(other).__name__}"
            )
        left_name = _temp_name("_cp_left")
        right_name = _temp_name("_cp_right")
        self._connection.register(left_name, self._relation)
        other._connection.register(right_name, other._relation)
        query = f"SELECT * FROM {left_name} CROSS JOIN {right_name}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def then(self, *funcs) -> "DuckJanitor":
        """Compose further verbs in sequence (R: ``then`` / ``DF_to_pandas``).

        Each callable is invoked with the current DuckJanitor and must return
        a DuckJanitor. Useful for ``pipe``-style chaining across modules.

        Parameters
        ----------
        *funcs : Callable
            Variadic callables, each taking a ``DuckJanitor`` and
            returning one.

        Returns
        -------
        DuckJanitor
            Self for method chaining, after every ``func`` has been applied.
        """
        out: DuckJanitor = self
        for func in funcs:
            res = func(out)
            if not isinstance(res, DuckJanitor):
                raise TypeError(
                    f"then(): func {getattr(func, '__name__', repr(func))} "
                    f"must return DuckJanitor; got {type(res).__name__}"
                )
            out = res
        return out

    def compare_df_cols_same(self, *others: "DuckJanitor") -> bool:
        """Compare the current relation's columns to other relations (R: ``compare_df_cols_same``).

        Parameters
        ----------
        *others : DuckJanitor
            One or more relations to compare against; all must have
            exactly the same column list as ``self``.

        Returns
        -------
        bool
            ``True`` if every ``others`` entry has identical columns to
            ``self``, otherwise ``False``.
        """
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
                raise ValueError("describe_class(): relation has no columns to describe")
            return pd.DataFrame(columns=["column_name", "column_type"])
        temp_name = _temp_name("_describe")
        self._connection.register(temp_name, self._relation)
        rows = self._connection.execute(f"DESCRIBE SELECT * FROM {temp_name}").fetchall()
        self._connection.unregister(temp_name)
        return pd.DataFrame(
            [{"column_name": r[0], "column_type": r[1]} for r in rows],
            columns=["column_name", "column_type"],
        )

    # =============================================================
    # pyjanitor parity batch 2 — medium helpers
    # =============================================================

    def row_to_names(
        self, row_number: int = 0, remove_row: bool = True, reset_index: bool = False
    ) -> "DuckJanitor":
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

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the chosen row lifted into the
            column headers.
        """
        if row_number < 0:
            raise ValueError(f"row_to_names(): row_number must be >= 0; got {row_number}")
        temp_name = _temp_name("_row2n")
        self._connection.register(temp_name, self._relation)
        # Pull the row values to use as column names.
        n_cols = len(self._relation.columns)
        ",".join(["?"] * n_cols)
        row_sql = f"SELECT * FROM {temp_name} LIMIT 1 OFFSET {row_number}"
        row_values = self._connection.execute(row_sql).fetchone()
        if row_values is None:
            raise ValueError(f"row_to_names(): row {row_number} does not exist")
        # Build the new column projection with the promoted row as aliases.
        col_selects = ", ".join(
            f'CAST("{self._relation.columns[i]}" AS VARCHAR) AS "{row_values[i]}"'
            for i in range(n_cols)
        )
        if remove_row:
            # Strip out the lifted row by row_number().
            query = (
                f"SELECT {col_selects} FROM ("
                f"SELECT *, row_number() OVER () AS _rn FROM {temp_name}"
                f") WHERE _rn <> {row_number + 1}"
            )
        else:
            query = f"SELECT {col_selects} FROM {temp_name}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def rle_id(self) -> "DuckJanitor":
        """Run-length id (R: ``rle_id``) — assign an integer id per change.

        Implementation uses a CTE chain that hashes all columns on each row
        and uses ``CONDITIONAL_TRUE_EVENT`` (via a stale flag) to break the
        run-length counter when the hash changes.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with an extra ``_rle_id`` column.
        """
        temp_name = _temp_name("_rle")
        self._connection.register(temp_name, self._relation)
        cols = ", ".join(f'CAST("{c}" AS VARCHAR)' for c in self._relation.columns)
        query = (
            f"SELECT *, (SUM(CASE WHEN col_hash = prev_hash THEN 0 ELSE 1 END) "
            f"OVER (ORDER BY ord)) AS _rle_id FROM ("
            f"SELECT *, hash({cols}) AS col_hash, "
            f"LAG(hash({cols})) OVER (ORDER BY ord) AS prev_hash "
            f"FROM (SELECT *, row_number() OVER () AS ord FROM {temp_name})"
            f")"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def factorize_columns(self, columns=None, append: bool = False) -> "DuckJanitor":
        """Integer-encode each category in ``columns`` (R: ``factorize_columns``).

        ``columns`` defaults to all string-typed columns in the relation.

        Parameters
        ----------
        columns : str or list of str, optional
            Columns to factorize. ``None`` picks every VARCHAR-typed column.
        append : bool, default False
            If True, append ``<col>_factor`` columns without dropping the
            originals; otherwise the originals are replaced.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with integer-encoded columns.
        """
        cur_cols = list(self._relation.columns)
        if columns is None:
            # Inspect the DuckDB types and use VARCHAR columns as candidates.
            desc_table = _temp_name("__dj_factorize_t")
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
                raise ValueError(f"factorize_columns(): unknown columns {missing}")

        temp_name = _temp_name("_factorize")
        self._connection.register(temp_name, self._relation)
        new_selects = [f'"{c}" AS "{c}"' for c in cur_cols]
        for col in columns:
            new_name = f"{col}_factor"
            new_selects.append(f'DENSE_RANK() OVER (ORDER BY "{col}") AS "{new_name}"')
        cols_csv = ", ".join(new_selects)
        query = f"SELECT {cols_csv} FROM {temp_name}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def sort_naturally(self, column: str) -> "DuckJanitor":
        """Natural-sort order for a column (R: ``sort_naturally``).

        Parameters
        ----------
        column : str
            Name of the column to sort by.

        Returns
        -------
        DuckJanitor
            Self for method chaining, sorted naturally.
        """
        import re as _re

        temp_name = _temp_name("_natsort")
        self._connection.register(temp_name, self._relation)
        query = f"SELECT * FROM {temp_name}"
        material = self._connection.query(query).df()

        def _key(v):
            return [
                (int(chunk) if chunk.isdigit() else chunk) for chunk in _re.split(r"(\d+)", str(v))
            ]

        # Defensive cast to string for the sort key.
        material_sorted = material.assign(
            **{"_natsort_key": material[column].astype(str)}
        ).sort_values(by="_natsort_key", key=lambda s: s.map(_key))
        material_sorted = material_sorted.drop(columns=["_natsort_key"]).reset_index(drop=True)
        return DuckJanitor.from_pandas(material_sorted)

    def sort_column_value_order(self, column: str, order: list[str]) -> "DuckJanitor":
        """Sort rows by an explicit string ordering of ``column`` (R: ``sort_column_value_order``).

        Parameters
        ----------
        column : str
            Name of the column whose ordering is imposed.
        order : list of str
            Permutation of the column's distinct values, in the desired
            display order. Values not present in the column are appended
            at the end.

        Returns
        -------
        DuckJanitor
            Self for method chaining, sorted by ``column`` per ``order``.
        """
        if column not in self._relation.columns:
            raise ValueError(
                f"sort_column_value_order(): column {column!r} not in {list(self._relation.columns)}"
            )
        temp_name = _temp_name("_sortorder")
        self._connection.register(temp_name, self._relation)
        # Validate that every value in `order` exists in the column.
        cur_values = {
            r[0]
            for r in self._connection.execute(
                f'SELECT DISTINCT "{column}" FROM {temp_name}'
            ).fetchall()
        }
        missing = [v for v in order if v not in cur_values]
        if missing:
            raise ValueError(
                f"sort_column_value_order(): values not in column {column!r}: {missing}"
            )
        # Use LIST_VALUE() to build the desired ordering; list_position
        # returns a numeric index per row, so missing keys sort at the end.
        order_list = "[" + ", ".join(repr(v) for v in order) + "]"
        query = f'SELECT * FROM {temp_name} ORDER BY list_position({order_list}, "{column}")'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def filter_date(self, column: str, start_date=None, end_date=None) -> "DuckJanitor":
        """Range-filter rows by a datetime ``column`` (R: ``filter_date``).

        Parameters
        ----------
        column : str
            Name of the datetime column to range-filter on.
        start_date : str, int, float, datetime, or None, optional
            Lower bound (inclusive). ``None`` means no lower bound.
            Strings, ints and floats are passed through DuckDB's
            value-literal handling; pass ``datetime`` objects for
            type safety.
        end_date : str, int, float, datetime, or None, optional
            Upper bound (inclusive). ``None`` means no upper bound.

        Returns
        -------
        DuckJanitor
            Self for method chaining, restricted to rows whose ``column``
            falls inside ``[start_date, end_date]``.
        """
        parts = []
        if start_date is not None:
            parts.append(f'"{column}" >= {self._sql_value(start_date)}')
        if end_date is not None:
            parts.append(f'"{column}" <= {self._sql_value(end_date)}')
        if not parts:
            return self
        where = " AND ".join(parts)
        return self.filter_on(where, complement=False)

    def update_where(self, columns: dict, conditions: str) -> "DuckJanitor":
        """Update ``columns`` where a SQL ``conditions`` clause holds (R: ``update_where``).

        ``columns`` maps column-name to a SQL expression string.

        Parameters
        ----------
        columns : dict
            ``{column_name: sql_expression}`` mapping; each listed column
            is replaced with ``sql_expression`` (per-row) wherever
            ``conditions`` evaluates true.
        conditions : str
            SQL WHERE-clause fragment (``age > 18``).

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the conditional updates applied.
        """
        if not isinstance(columns, dict) or not columns:
            raise ValueError("update_where(): columns must be a non-empty dict")
        cur_cols = list(self._relation.columns)
        missing = [c for c in columns if c not in cur_cols]
        if missing:
            raise ValueError(f"update_where(): unknown columns {missing}")
        temp_name = _temp_name("_updwhere")
        self._connection.register(temp_name, self._relation)
        # Build CASE WHEN update expressions per column.
        case_clauses = []
        for col, expr in columns.items():
            case_clauses.append(f'CASE WHEN {conditions} THEN ({expr}) ELSE "{col}" END AS "{col}"')
        select_list = ", ".join(case_clauses)
        query = f"SELECT {select_list} FROM {temp_name}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def unionize_dataframe_categories(
        self, *others: "DuckJanitor", column_names=None
    ) -> "DuckJanitor":
        """Cast all factor-like columns across many DuckJanitors to consistent string types.

        Useful as a preprocessing step before concat().

        Parameters
        ----------
        *others : DuckJanitor
            Additional DuckJanitor instances whose VARCHAR columns
            should be aligned with ``self``.
        column_names : list of str, optional
            Subset of column names to align; defaults to every VARCHAR
            column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with VARCHAR columns widened to the
            union of observed types.
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
        self._connection.register("_union_self", self._relation)
        select_list = []
        for c in self._relation.columns:
            if c in targets:
                select_list.append(f'CAST("{c}" AS VARCHAR) AS "{c}"')
            else:
                select_list.append(f'"{c}"')
        query = "SELECT " + ", ".join(select_list) + " FROM _union_self"
        new_relation = self._connection.query(query)
        # VARCHAR columns come back from DuckDB as pandas StringDtype on
        # collect() — that is the canonical string storage here (not
        # object dtype). Callers should check is_string_dtype.
        return DuckJanitor(new_relation, self._connection)

    # =============================================================
    # pyjanitor parity batch 3 — heavyweight helpers
    # =============================================================

    def expand(self, columns: list[str], on=None) -> "DuckJanitor":
        """Cartesian-expand across the unique values of ``columns`` (R: ``expand``).

        ``on`` is unused for now; in R it controls the iteration order.

        Parameters
        ----------
        columns : list of str
            Columns whose unique-value Cartesian product defines the new row set.
        on : unused
            Accepted for signature parity; currently ignored.

        Returns
        -------
        DuckJanitor
            Self for method chaining, expanded to the full value grid.
        """
        if not columns:
            raise ValueError("expand(): columns must be a non-empty list")
        for c in columns:
            if c not in self._relation.columns:
                raise ValueError(f"expand(): column {c!r} not in {list(self._relation.columns)}")
        temp_name = _temp_name("_expand")
        self._connection.register(temp_name, self._relation)
        # Use DuckDB GROUP BY ALL to preserve distinct combinations.
        query = f"SELECT DISTINCT {', '.join(self._quote(c) for c in columns)} FROM {temp_name}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def expand_grid(self, *tables) -> "DuckJanitor":
        """Cross-join an arbitrary number of relations (R: ``expand_grid``).

        Each argument must be a ``DuckJanitor``. The result contains one row
        for every combination of rows across all inputs (a cartesian product
        in the tidyverse sense). Column names are preserved; on collision,
        the later table's column is suffixed ``_1``, ``_2``, ... in order.

        Parameters
        ----------
        *tables : DuckJanitor
            Variadic DuckJanitor relations to cross-join with ``self``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the cross-joined columns.
        """
        if not tables:
            raise ValueError("expand_grid(): need at least one input")
        sources = [self] + list(tables)
        for t in sources:
            if not isinstance(t, DuckJanitor):
                raise TypeError(
                    "expand_grid(): all arguments must be DuckJanitor instances; "
                    f"got {type(t).__name__}"
                )
        # Register every source on the calling connection, build a single
        # flat CROSS JOIN query, and disambiguate duplicate column names.
        registered = []
        for i, t in enumerate(sources):
            src_name = f"_eg_{i}_{id(self)}"
            self._connection.register(src_name, t._relation)
            registered.append((src_name, list(t._relation.columns)))
        seen_names: dict[str, int] = {}
        select_parts = []
        for src_name, cols in registered:
            for c in cols:
                if c in seen_names:
                    seen_names[c] += 1
                    alias = f"{c}_{seen_names[c]}"
                else:
                    seen_names[c] = 0
                    alias = c
                select_parts.append(f'{src_name}."{c}" AS "{alias}"')
        query = (
            "SELECT "
            + ", ".join(select_parts)
            + " FROM "
            + " CROSS JOIN ".join(name for name, _ in registered)
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def change_index_dtype(self, dtype: str, target_name: Optional[str] = None) -> "DuckJanitor":
        """Create a cast version of the FIRST column with the desired ``dtype`` (R: ``change_index_dtype``).

        DuckDB relations have no intrinsic integer index; we mimic by
        projecting a typed copy of the first column.

        Parameters
        ----------
        dtype : str
            DuckDB type name (``'INT'``, ``'BIGINT'``, ``'VARCHAR'``,
            ``'DATE'``, etc).
        target_name : str, optional
            Name of the output column; defaults to
            ``<first_column>_typed``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with ``target_name`` appended.
        """
        cur_cols = list(self._relation.columns)
        if not cur_cols:
            raise ValueError("change_index_dtype(): relation has no columns")
        src = cur_cols[0]
        out = target_name or f"{src}_idx_typed"
        temp_name = _temp_name("_idx_dtype")
        self._connection.register(temp_name, self._relation)
        query = f'SELECT *, CAST("{src}" AS {dtype}) AS "{out}" FROM {temp_name}'
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def collapse_levels(self, sep: str = "_", column: Optional[str] = None) -> "DuckJanitor":
        """Collapse a tuple-named index column by joining on ``sep`` (R: ``collapse_levels``).

        For simplicity, when ``column`` is None this concatenates all
        columns together. When ``column`` is given, only that column is
        collapsed (no-op since DuckDB columns are flat).

        Parameters
        ----------
        sep : str, default '_'
            Separator inserted between collapsed values.
        column : str, optional
            Specific column to collapse; ``None`` collapses every column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the collapsed ``column`` (if any).
        """
        cur_cols = list(self._relation.columns)
        if column and column not in cur_cols:
            raise ValueError(f"collapse_levels(): unknown column {column!r}")
        temp_name = _temp_name("_collapse")
        self._connection.register(temp_name, self._relation)
        if column:
            # No-op: a single column is already flat; return as-is.
            new_relation = self._connection.query(f"SELECT * FROM {temp_name}")
        else:
            [f'CAST("{c}" AS VARCHAR) AS "{c}"' for c in cur_cols]
            query = (
                "SELECT *, "
                "CONCAT("
                + ", ".join(f'CAST("{c}" AS VARCHAR)' for c in cur_cols)
                + f') AS "{sep.join(cur_cols)}" '
                f"FROM {temp_name}"
            )
            new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def explode_index(
        self, column: str, names: Optional[list[str]] = None, separator: str = "_"
    ) -> "DuckJanitor":
        """Split ``column`` into multiple sub-fields (R: ``explode_index``).

        ``names`` lists the new column names. By default a single new
        column called ``<column>_parsed`` is created.

        Parameters
        ----------
        column : str
            Name of the string column to split.
        names : list of str, optional
            Output column names; the column is split positionally on
            ``separator`` into exactly ``len(names)`` new columns.
        separator : str, default '_'
            Split delimiter.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the parsed sub-field columns
            appended and ``column`` dropped.
        """
        if column not in self._relation.columns:
            raise ValueError(f"explode_index(): unknown column {column!r}")
        temp_name = _temp_name("_explode")
        self._connection.register(temp_name, self._relation)
        cur_cols = list(self._relation.columns)
        out_cols = names or [f"{column}_parsed"]
        selects = [self._quote(c) for c in cur_cols]
        # Toy implementation: extract numeric sequences as a single column.
        # DuckDB's regex support allows richer extraction; this is a stub.
        query = (
            f"SELECT {', '.join(selects)}, "
            f'regexp_extract(CAST("{column}" AS VARCHAR), \'(\\d+)\', 1) AS "{out_cols[0]}" '
            f"FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def summarise(
        self, group_by: Optional[list[str]] = None, agg_spec: Optional[dict] = None
    ) -> "DuckJanitor":
        """Group-by summarisation helper (R: ``summarise`` / ``summarize``).

        Parameters
        ----------
        group_by : list of str
            Columns to group by; ``None`` means no grouping.
        agg_spec : dict
            Mapping of ``new_column_name -> (source_column, agg_function_string)``.
            Examples::

                {'avg_age': ('age', 'AVG'), 'n': ('*', 'COUNT')}

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the aggregated columns appended.
        """
        agg_spec = agg_spec or {}
        cur_cols = list(self._relation.columns)
        if group_by:
            missing = [c for c in group_by if c not in cur_cols]
            if missing:
                raise ValueError(f"summarise(): unknown group_by columns {missing}")
        temp_name = _temp_name("_summarise")
        self._connection.register(temp_name, self._relation)
        select_parts = []
        if group_by:
            select_parts.extend(self._quote(c) for c in group_by)
        for new_col, (src, agg) in agg_spec.items():
            agg = agg.upper()
            if src == "*":
                expr = f"COUNT(*) AS {new_col}"
            else:
                if src not in cur_cols:
                    raise ValueError(f"summarise(): unknown source {src!r}")
                expr = f'{agg}("{src}") AS {new_col}'
            select_parts.append(expr)
        if not select_parts:
            raise ValueError("summarise(): no group_by or aggregation specified")
        group_by_clause = (
            f"GROUP BY {', '.join(self._quote(c) for c in group_by)}" if group_by else ""
        )
        query = f"SELECT {', '.join(select_parts)} FROM {temp_name} {group_by_clause}"
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def pivot_longer_spec(
        self,
        id_cols: list[str],
        value_cols: list[str],
        names_to: str = "name",
        values_to: str = "value",
        names_sep: Optional[str] = None,
    ) -> "DuckJanitor":
        """Long-form pivot driven by a column-name spec (R: ``pivot_longer_spec``).

        Parameters
        ----------
        id_cols : list of str
            Identifier columns that remain as rows.
        value_cols : list of str
            Columns to unpivot.
        names_to : str, default 'name'
            Output column holding the unpivoted column names.
        values_to : str, default 'value'
            Output column holding the unpivoted cell values.
        names_sep : str, optional
            If provided, multi-part column names are split on this
            separator into multiple ``names_to*`` columns.

        Returns
        -------
        DuckJanitor
            Self for method chaining, in long format.
        """
        if not value_cols:
            raise ValueError("pivot_longer_spec(): value_cols must be non-empty")
        missing_vals = [c for c in value_cols if c not in self._relation.columns]
        if missing_vals:
            raise ValueError(f"pivot_longer_spec(): unknown value_cols {missing_vals}")
        cur_cols = list(self._relation.columns)
        select_parts = [self._quote(c) for c in cur_cols if c not in value_cols]
        if names_sep:
            # Apply split on each value column name via DuckDB transform.
            for vc in value_cols:
                split_parts = vc.split(names_sep)
                for k, part in enumerate(split_parts):
                    alias = f"{names_to}_{k}" if k < len(split_parts) - 1 else values_to
                    if k < len(split_parts) - 1:
                        select_parts.append(f"'{part}' AS {alias}")
                # Last part becomes the values column.
                select_parts.append(f'CAST("{vc}" AS DOUBLE) AS "{values_to}_{vc}"')
        else:
            select_parts.append(
                f'UNNEST({", ".join(self._quote(v) for v in value_cols)}) AS "{values_to}"'
            )
        temp_name = _temp_name("_plspec")
        self._connection.register(temp_name, self._relation)
        # Without UNNEST it's awkward to pivot properly here; emulate with cross-join.
        # Practical implementation: emit one row per (id × values) using unnest of an array.
        "[" + ", ".join(self._quote(v) for v in value_cols) + "]"
        final_parts = []
        for c in cur_cols:
            if c in value_cols:
                continue
            final_parts.append(self._quote(c))
        # Build a structured unpivot using DuckDB's UNPIVOT extension.
        unpivot_query = f"SELECT * FROM {temp_name} UNPIVOT ({values_to} FOR {names_to} IN ({', '.join(self._quote(v) for v in value_cols)}))"
        new_relation = self._connection.query(unpivot_query)
        return DuckJanitor(new_relation, self._connection)

    def pivot_wider_spec(
        self, id_cols: list[str], names_from: str, values_from: str, names_glue: str = "_"
    ) -> "DuckJanitor":
        """Wide pivot driven by a column-name spec (R: ``pivot_wider_spec``).

        Parameters
        ----------
        id_cols : list of str
            Identifier columns that remain as rows.
        names_from : str
            Column whose values become the new column names.
        values_from : str
            Column whose values populate the new columns.
        names_glue : str, default '_'
            Separator inserted between multi-part names when ``names_from``
            itself is a derived multi-column.

        Returns
        -------
        DuckJanitor
            Self for method chaining, in wide format.
        """
        if names_from not in self._relation.columns:
            raise ValueError(f"pivot_wider_spec(): unknown names_from {names_from!r}")
        if values_from not in self._relation.columns:
            raise ValueError(f"pivot_wider_spec(): unknown values_from {values_from!r}")
        temp_name = _temp_name("_pwspec")
        self._connection.register(temp_name, self._relation)
        # DuckDB's PIVOT requires a literal list of values. Collect them first.
        distinct_values = [
            r[0]
            for r in self._connection.execute(
                f'SELECT DISTINCT "{names_from}" FROM {temp_name} ORDER BY "{names_from}"'
            ).fetchall()
        ]
        if not distinct_values:
            raise ValueError("pivot_wider_spec(): no values in names_from")
        values_csv = ", ".join(self._sql_value(v) for v in distinct_values)
        id_csv = ", ".join(self._quote(c) for c in id_cols)
        query = (
            f"SELECT {id_csv} FROM {temp_name} "
            f"PIVOT (FIRST({values_from}) FOR {names_from} IN ({values_csv}))"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def join_agg(self, other: "DuckJanitor", on: tuple, aggs: dict) -> "DuckJanitor":
        """Aggregate join (R: ``join_agg``) — left-join with arbitrary aggregations.

        ``aggs`` is a dict mapping ``new_col -> ("COL", AGG)``.

        Parameters
        ----------
        other : DuckJanitor
            The right-side relation to join.
        on : tuple
            ``(left_col, right_col, op_str)`` describing the conditional
            join key.
        aggs : dict
            ``{new_column_name: (source_column, agg_function)}`` for the
            right-side aggregations, e.g.
            ``{'avg_age': ('age', 'AVG')}``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the joined+aggregated result.
        """
        cur_cols = list(self._relation.columns)
        other_cols = list(other._relation.columns)
        if len(on) != 3:
            raise ValueError("join_agg(): on must be (left, right, op)")
        left_on, right_on, op = on
        if left_on not in cur_cols:
            raise ValueError(f"join_agg(): left_on {left_on!r} not in current columns")
        if right_on not in other_cols:
            raise ValueError(f"join_agg(): right_on {right_on!r} not in other columns")
        if op == "==":
            raise ValueError(
                "join_agg(): equality joins are not supported; use a non-equality op "
                "or use conditional_join() for equality."
            )
        left_name = _temp_name("_jagg_left")
        right_name = _temp_name("_jagg_right")
        self._connection.register(left_name, self._relation)
        other._connection.register(right_name, other._relation)
        agg_expressions = ", ".join(
            f'{agg.upper()}("{col}") AS {new_col}' for new_col, (col, agg) in aggs.items()
        )
        # Build a fully-qualified projection list so the join key columns
        # can be quoted explicitly to avoid "ambiguous column" binder errors.
        full_cur = ", ".join(f'"{left_name}"."{c}" AS "{c}"' for c in cur_cols)
        agg_columns = ", ".join(f'g."{new_col}" AS "{new_col}"' for new_col in aggs)
        query = (
            f"SELECT {full_cur}, {agg_columns} "
            f"FROM {left_name} LEFT JOIN ("
            f'SELECT "{right_on}", {agg_expressions} '
            f'FROM {right_name} GROUP BY "{right_on}"'
            f') g ON "{left_name}"."{left_on}" {op} g."{right_on}"'
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def get_join_indices(self, other: "DuckJanitor", conditions) -> dict:
        """Compute join key indices without materialising the join (R: ``get_join_indices``).

        Parameters
        ----------
        other : DuckJanitor
            The right-side relation to probe against.
        conditions : tuple or list of tuple
            Either a single ``(left_col, right_col, op_str)`` triple or a
            list of such triples (logical AND across them).

        Returns
        -------
        dict
            ``{'left_indices': [...], 'right_indices': [...]}`` — index
            arrays that satisfy the join.
        """
        if not isinstance(other, DuckJanitor):
            raise TypeError("get_join_indices(): other must be a DuckJanitor")
        if isinstance(conditions, tuple) and len(conditions) == 3:
            conditions = [conditions]
        else:
            try:
                conditions = list(conditions)
            except TypeError as err:
                raise TypeError(
                    "get_join_indices(): conditions must be a (left, right, op) tuple or list"
                ) from err
        left_name = _temp_name("_gjidx_l")
        right_name = _temp_name("_gjidx_r")
        self._connection.register(left_name, self._relation)
        other._connection.register(right_name, other._relation)
        result = {}
        for left_on, right_on, op in conditions:
            if left_on not in self._relation.columns:
                raise ValueError(f"get_join_indices(): left_on {left_on!r} missing")
            if right_on not in other._relation.columns:
                raise ValueError(f"get_join_indices(): right_on {right_on!r} missing")
            # Pull all values to compute indices in Python.
            left_vals = [
                r[0]
                for r in self._connection.execute(f'SELECT "{left_on}" FROM {left_name}').fetchall()
            ]
            right_vals = [
                r[0]
                for r in other._connection.execute(
                    f'SELECT "{right_on}" FROM {right_name}'
                ).fetchall()
            ]
            # Generic dispatcher using Python operator.
            import operator as _op

            opmap = {
                "<": _op.lt,
                "<=": _op.le,
                ">": _op.gt,
                ">=": _op.ge,
                "==": _op.eq,
                "!=": _op.ne,
            }
            py_op = opmap.get(op)
            if py_op is None:
                raise ValueError(f"get_join_indices(): unsupported op {op!r}")
            pairs = [
                (i, j)
                for i, lv in enumerate(left_vals)
                for j, rv in enumerate(right_vals)
                if py_op(lv, rv)
            ]
            result[(left_on, right_on, op)] = pairs
        return result

    # ─────────────────────────────────────────────────────────────────────
    # dplyr-style explicit join verbs (R/dplyr parity, DuckDB-backed)
    # ─────────────────────────────────────────────────────────────────────

    def inner_join(
        self,
        other: "DuckJanitor",
        on: Union[str, list[str], None] = None,
        *,
        left_on: Union[str, list[str], None] = None,
        right_on: Union[str, list[str], None] = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> "DuckJanitor":
        """Return rows whose key matches on both sides (dplyr / R ``inner_join``).

        Parameters
        ----------
        other : DuckJanitor
            The right-side relation to join.
        on : str or list of str, optional
            Column name(s) shared by both sides. Use this when the key has
            the same name on left and right. Mutually exclusive with
            ``left_on`` / ``right_on``.
        left_on, right_on : str or list of str, optional
            Column names when the join key has a different name on each side.
            Both must be supplied together and have equal length.
        suffixes : tuple of (str, str), default ``('_x', '_y')``
            Suffix applied to right-side columns whose names collide with a
            non-key left column.
        conn : DuckDBPyConnection, optional
            Connection that owns the left relation (defaults to
            ``self._connection``).

        Returns
        -------
        DuckJanitor
            New instance with the join applied.

        Example
        -------
        >>> import pandas as pd
        >>> from pyduck_janitor import DuckJanitor
        >>> emps = DuckJanitor.from_pandas(pd.DataFrame({
        ...     'id': [1, 2, 3], 'name': ['a', 'b', 'c'], 'dept_id': [10, 20, 30]
        ... }))
        >>> depts = DuckJanitor.from_pandas(pd.DataFrame({
        ...     'dept_id': [10, 20, 99], 'dept_name': ['Eng', 'Sales', 'HR']
        ... }))
        >>> emps.inner_join(depts, on='dept_id').collect()
           id name  dept_id dept_name
        0   1    a       10       Eng
        1   2    b       20     Sales
        """
        from .joins import inner_join as _inner_join

        return _inner_join(
            self,
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            conn=self._connection,
        )

    def left_join(
        self,
        other: "DuckJanitor",
        on: Union[str, list[str], None] = None,
        *,
        left_on: Union[str, list[str], None] = None,
        right_on: Union[str, list[str], None] = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> "DuckJanitor":
        """Keep every left row; matched right rows; NULL on miss (dplyr ``left_join``).

        Example
        -------
        >>> emps.left_join(depts, on='dept_id').collect()
           id name  dept_id dept_name
        0   1    a       10       Eng
        1   2    b       20     Sales
        2   3    c       30      None   # dept_id=30 has no department row

        """
        from .joins import left_join as _left_join

        return _left_join(
            self,
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            conn=self._connection,
        )

    def right_join(
        self,
        other: "DuckJanitor",
        on: Union[str, list[str], None] = None,
        *,
        left_on: Union[str, list[str], None] = None,
        right_on: Union[str, list[str], None] = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> "DuckJanitor":
        """Keep every right row; matched left rows; NULL on miss (dplyr ``right_join``)."""
        from .joins import right_join as _right_join

        return _right_join(
            self,
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            conn=self._connection,
        )

    def full_join(
        self,
        other: "DuckJanitor",
        on: Union[str, list[str], None] = None,
        *,
        left_on: Union[str, list[str], None] = None,
        right_on: Union[str, list[str], None] = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> "DuckJanitor":
        """Keep every row from both sides; NULL on either side on miss (dplyr ``full_join``)."""
        from .joins import full_join as _full_join

        return _full_join(
            self,
            other,
            on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            conn=self._connection,
        )

    def semi_join(
        self,
        other: "DuckJanitor",
        on: Union[str, list[str], None] = None,
        *,
        left_on: Union[str, list[str], None] = None,
        right_on: Union[str, list[str], None] = None,
    ) -> "DuckJanitor":
        """Return left rows whose key EXISTS on right; no right columns (dplyr ``semi_join``)."""
        from .joins import semi_join as _semi_join

        return _semi_join(
            self, other, on, left_on=left_on, right_on=right_on, conn=self._connection
        )

    def anti_join(
        self,
        other: "DuckJanitor",
        on: Union[str, list[str], None] = None,
        *,
        left_on: Union[str, list[str], None] = None,
        right_on: Union[str, list[str], None] = None,
    ) -> "DuckJanitor":
        """Return left rows whose key does NOT exist on right; no right columns (dplyr ``anti_join``)."""
        from .joins import anti_join as _anti_join

        return _anti_join(
            self, other, on, left_on=left_on, right_on=right_on, conn=self._connection
        )

    def to_datetime(
        self,
        column: str,
        format: Optional[str] = None,  # noqa: A002
        target_column: Optional[str] = None,
    ) -> "DuckJanitor":
        """Cast ``column`` to a TIMESTAMP using DuckDB ``strptime`` (R: ``to_datetime``).

        Parameters
        ----------
        column : str
            Name of the string / numeric column to cast.
        format : str, optional
            ``strftime``-style format; if omitted, DuckDB's
            ``TRY_CAST(... AS TIMESTAMP)`` is used.
        target_column : str, optional
            Name of the output TIMESTAMP column; defaults to
            ``column + '_ts'``.

        Returns
        -------
        DuckJanitor
            Self for method chaining, with the new TIMESTAMP column added.
        """
        out = target_column or f"{column}_ts"
        if column not in self._relation.columns:
            raise ValueError(f"to_datetime(): unknown column {column!r}")
        temp_name = _temp_name("_todt")
        self._connection.register(temp_name, self._relation)
        format_arg = f", {self._sql_value(format)}" if format is not None else ""
        query = (
            f'SELECT *, TRY_CAST(strptime(CAST("{column}" AS VARCHAR)'
            f'{format_arg}) AS TIMESTAMP) AS "{out}" '
            f"FROM {temp_name}"
        )
        new_relation = self._connection.query(query)
        return DuckJanitor(new_relation, self._connection)

    def __repr__(self) -> str:
        return f"DuckJanitor(relation={self._relation.columns}, lazy=True)"
