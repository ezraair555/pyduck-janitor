"""Output and write-back verbs for pyduck-janitor.

Before this module, a :class:`~pyduck_janitor.DuckJanitor` pipeline had
exactly one exit: ``collect()``, which called ``relation.df()`` and pulled
the whole result into pandas. That defeated the out-of-core promise of the
package.

Every function here writes through DuckDB itself — ``COPY ... TO`` for
files, ``CREATE TABLE/VIEW AS`` for the catalog — so a pipeline can land
directly in Parquet, CSV, JSON, or a persistent DuckDB database without
passing through Python objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import duckdb

from ._sqlutil import quote_identifier, temp_name as _temp_name

__all__ = [
    "to_parquet",
    "to_csv",
    "to_json",
    "to_table",
    "to_view",
    "to_arrow",
    "to_polars",
    "to_numpy",
    "write_relation",
]

# DuckDB's COPY ... TO formats. JSONL is DuckDB's name for
# newline-delimited JSON.
_FORMATS = {"parquet", "csv", "json", "jsonl"}


def _register(conn: duckdb.DuckDBPyConnection, relation: duckdb.DuckDBPyRelation) -> str:
    """Register *relation* on *conn* and return its temporary SQL name."""
    if conn is None:
        raise ValueError("A DuckDB connection is required to write a relation.")
    name = _temp_name("_pyduck_write")
    conn.register(name, relation)
    return name


def write_relation(
    relation: duckdb.DuckDBPyRelation,
    path: Union[str, Path],
    *,
    fmt: str = "parquet",
    overwrite: bool = False,
    options: Optional[dict[str, Any]] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> str:
    """Write a relation to a file using DuckDB's native ``COPY ... TO``.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to write.
    path : str or Path
        Destination path. Parent directories are created automatically.
    fmt : str, optional
        Output format: ``'parquet'``, ``'csv'``, ``'json'``, or
        ``'newline_delimited'``. Default is ``'parquet'``.
    overwrite : bool, optional
        When True, an existing destination is replaced. Default is False,
        which raises if the destination already exists.
    options : dict, optional
        Extra ``COPY`` options, for example
        ``{"compression": "zstd"}`` or ``{"delimiter": ";"}``. Values are
        rendered as SQL literals; boolean and numeric values are rendered
        unquoted.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relation. Required.

    Returns
    -------
    str
        The path that was written.

    Raises
    ------
    ValueError
        If ``fmt`` is unsupported, ``conn`` is None, or ``options`` contains
        a key that is not a valid COPY option.
    """
    if conn is None:
        raise ValueError("A DuckDB connection is required to write a relation.")
    fmt = str(fmt).lower()
    if fmt not in _FORMATS:
        raise ValueError(f"write_relation(): fmt must be one of {sorted(_FORMATS)}. Got {fmt!r}.")

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"write_relation(): {destination} already exists. Pass overwrite=True to replace it."
        )
    if destination.parent and str(destination.parent) not in ("", "."):
        destination.parent.mkdir(parents=True, exist_ok=True)

    table_name = _register(conn, relation)
    target = str(destination).replace("'", "''")

    clauses: list[str] = []
    for key, value in (options or {}).items():
        if not str(key).replace("_", "").isalnum():
            raise ValueError(f"write_relation(): invalid COPY option name {key!r}.")
        option_name = str(key).upper()
        if isinstance(value, bool):
            # DuckDB COPY options are flags, not key=value pairs.
            if value:
                clauses.append(option_name)
        elif isinstance(value, (int, float)):
            clauses.append(f"{option_name} {value}")
        else:
            escaped = str(value).replace("'", "''")
            clauses.append(f"{option_name} '{escaped}'")

    option_sql = (", " + ", ".join(clauses)) if clauses else ""
    query = f"COPY {quote_identifier(table_name)} TO '{target}' (FORMAT {fmt}{option_sql})"
    conn.execute(query)
    return str(destination)


def to_parquet(
    relation: duckdb.DuckDBPyRelation,
    path: Union[str, Path],
    *,
    overwrite: bool = False,
    compression: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> str:
    """Write a relation to a Parquet file.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to write.
    path : str or Path
        Destination ``.parquet`` path.
    overwrite : bool, optional
        Replace an existing file. Default is False.
    compression : str, optional
        Parquet compression codec, for example ``'zstd'``, ``'snappy'``, or
        ``'gzip'``.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relation. Required.

    Returns
    -------
    str
        The path that was written.
    """
    options = {"compression": compression} if compression else None
    return write_relation(
        relation,
        path,
        fmt="parquet",
        overwrite=overwrite,
        options=options,
        conn=conn,
    )


def to_csv(
    relation: duckdb.DuckDBPyRelation,
    path: Union[str, Path],
    *,
    overwrite: bool = False,
    header: bool = True,
    delimiter: str = ",",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> str:
    """Write a relation to a CSV file.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to write.
    path : str or Path
        Destination ``.csv`` path.
    overwrite : bool, optional
        Replace an existing file. Default is False.
    header : bool, optional
        Write a header row. Default is True.
    delimiter : str, optional
        Field delimiter. Default is ``','``.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relation. Required.

    Returns
    -------
    str
        The path that was written.
    """
    return write_relation(
        relation,
        path,
        fmt="csv",
        overwrite=overwrite,
        options={"header": header, "delimiter": delimiter},
        conn=conn,
    )


def to_json(
    relation: duckdb.DuckDBPyRelation,
    path: Union[str, Path],
    *,
    overwrite: bool = False,
    newline_delimited: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> str:
    """Write a relation to a JSON file.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to write.
    path : str or Path
        Destination ``.json`` or ``.jsonl`` path.
    overwrite : bool, optional
        Replace an existing file. Default is False.
    newline_delimited : bool, optional
        Write newline-delimited JSON (one object per line) instead of a
        JSON array. Default is False.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relation. Required.

    Returns
    -------
    str
        The path that was written.
    """
    fmt = "jsonl" if newline_delimited else "json"
    if fmt == "json":
        # DuckDB's JSON copy format emits newline-delimited objects by
        # default. ARRAY true produces a single JSON array instead, which
        # is what a caller writing a '.json' file normally expects.
        return write_relation(
            relation,
            path,
            fmt=fmt,
            overwrite=overwrite,
            options={"array": True},
            conn=conn,
        )
    return write_relation(relation, path, fmt=fmt, overwrite=overwrite, conn=conn)


def to_table(
    relation: duckdb.DuckDBPyRelation,
    name: str,
    *,
    view: bool = False,
    replace: bool = True,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> str:
    """Materialise a relation as a table or view in the catalog.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to materialise.
    name : str
        Target table or view name. May be ``'schema.table'``.
    view : bool, optional
        Create a view instead of a table. Views are lazy and read-through;
        tables store the data. Default is False.
    replace : bool, optional
        Replace an existing object of the same name. Default is True.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relation. Required.

    Returns
    -------
    str
        The catalog name that was created.
    """
    if conn is None:
        raise ValueError("A DuckDB connection is required to create a table or view.")
    table_name = _register(conn, relation)
    kind = "VIEW" if view else "TABLE"
    action = "OR REPLACE" if replace else ""
    conn.execute(f"CREATE {action} {kind} {name} AS SELECT * FROM {quote_identifier(table_name)}")
    return name


def to_view(
    relation: duckdb.DuckDBPyRelation,
    name: str,
    *,
    replace: bool = True,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> str:
    """Materialise a relation as a view in the catalog.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to materialise.
    name : str
        Target view name. May be ``'schema.view'``.
    replace : bool, optional
        Replace an existing view of the same name. Default is True.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relation. Required.

    Returns
    -------
    str
        The catalog name that was created.
    """
    return to_table(relation, name, view=True, replace=replace, conn=conn)


def to_arrow(relation: duckdb.DuckDBPyRelation):
    """Return the relation as an Arrow table.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to convert.

    Returns
    -------
    pyarrow.Table
        The result as an Arrow table.
    """
    return relation.to_arrow_table()


def to_polars(relation: duckdb.DuckDBPyRelation):
    """Return the relation as a Polars DataFrame.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to convert.

    Returns
    -------
    polars.DataFrame
        The result as a Polars DataFrame.

    Raises
    ------
    ImportError
        If Polars is not installed.
    """
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "to_polars() requires Polars. Install it with `pip install polars`."
        ) from exc
    return pl.from_arrow(relation.to_arrow_table())


def to_numpy(relation: duckdb.DuckDBPyRelation):
    """Return the relation as a NumPy array.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The relation to convert.

    Returns
    -------
    numpy.ndarray
        The result as a NumPy array.
    """
    return relation.fetchnumpy()
