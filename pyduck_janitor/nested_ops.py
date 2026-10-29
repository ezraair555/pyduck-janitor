"""
Nested / typed-data operations for DuckJanitor.

This module provides DuckDB-native verbs for working with nested data:
``STRUCT`` columns, ``LIST`` columns and ``JSON`` payloads.  Every function
takes a :class:`duckdb.DuckDBPyRelation` and returns a *new* relation; the
input relation is never mutated.

All SQL is generated with the helpers in :mod:`pyduck_janitor.cleaning_ops`
(``_quote_id``, ``_sql_literal``, ``_register_relation``,
``_ensure_columns_exist`` and ``_validate_sql_fragment``) so identifiers and
literals are always quoted safely and user supplied SQL fragments are
validated before they reach the database.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Union

import duckdb

from .cleaning_ops import (
    _require_conn,
    _ensure_columns_exist,
    _quote_id,
    _register_relation,
    _sql_literal,
    _validate_sql_fragment,
)

__all__ = [
    "to_struct",
    "from_struct",
    "to_list",
    "from_list",
    "list_transform",
    "list_filter",
    "list_transforms",
    "json_extract",
    "unnest",
]

# Prefix used for internal aliases of the columns that are carried through
# an inner ``SELECT``.  It is unlikely to collide with a user column name and
# keeps the generated SQL readable.
_TMP = "__dj_nested_"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_list(columns: Union[str, list[str], tuple[str, ...]]) -> list[str]:
    """Normalise a single column name / collection of names into a list."""
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    if not columns:
        raise ValueError("columns must contain at least one column name")
    return columns


def _column_types(relation: duckdb.DuckDBPyRelation) -> dict[str, str]:
    """Return a mapping of column name -> DuckDB type string."""
    types = list(relation.types)
    columns = list(relation.columns)
    return {col: str(typ) for col, typ in zip(columns, types)}


def _is_struct(type_string: str) -> bool:
    """Return True when the DuckDB type string denotes a STRUCT."""
    return type_string.upper().startswith("STRUCT")


def _is_list(type_string: str) -> bool:
    """Return True when the DuckDB type string denotes a LIST."""
    return type_string.upper().endswith("[]")


def _require_struct(relation: duckdb.DuckDBPyRelation, column: str) -> None:
    """Raise ValueError when *column* is not a STRUCT column."""
    type_string = _column_types(relation).get(column, "")
    if not _is_struct(type_string):
        raise ValueError(
            f"Column '{column}' is not a STRUCT column (type: {type_string or 'unknown'})."
        )


def _require_list(relation: duckdb.DuckDBPyRelation, column: str) -> None:
    """Raise ValueError when *column* is not a LIST column."""
    type_string = _column_types(relation).get(column, "")
    if not _is_list(type_string):
        raise ValueError(
            f"Column '{column}' is not a LIST column (type: {type_string or 'unknown'})."
        )


def _struct_keys(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    column: str,
) -> list[str]:
    """Return the key names of the STRUCT stored in *column*.

    The keys are read from the result description of a zero-row
    ``UNNEST`` probe.  This works even when every value in the column is
    ``NULL`` and returns the exact key names (including keys that contain
    spaces or punctuation).
    """
    probe = f"SELECT * FROM (SELECT UNNEST({_quote_id(column)}) FROM {table_name} LIMIT 0)"
    try:
        description = conn.execute(probe).description
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not inspect the STRUCT type of column '{column}'.") from exc
    return [entry[0] for entry in description]


def _carry(
    kept_columns: list[str],
) -> tuple[str, list[tuple[str, str]]]:
    """Build the inner alias expressions and the outer restore mapping."""
    inner_parts = []
    restore: list[tuple[str, str]] = []
    for index, column in enumerate(kept_columns):
        alias = f"{_TMP}{index}"
        inner_parts.append(f"{_quote_id(column)} AS {_quote_id(alias)}")
        restore.append((alias, column))
    return ", ".join(inner_parts), restore


def _outer_select(restore: list[tuple[str, str]], extra: list[str]) -> str:
    """Build the outer ``SELECT`` list restoring original column names."""
    parts = [f"{_quote_id(alias)} AS {_quote_id(name)}" for alias, name in restore]
    parts.extend(extra)
    return ", ".join(parts)


def _default_json_output(json_column: str, path: str) -> str:
    """Derive the default output column name for :func:`json_extract`.

    The last segment of the pointer (``"$.a.b"`` -> ``"b"``) is used; when the
    path carries no usable segment the column name is suffixed with
    ``"_value"``.
    """
    pointer = path.strip()
    if pointer.startswith("$"):
        pointer = pointer.lstrip("$")
    pointer = pointer.lstrip(".")
    segments = [segment for segment in pointer.split(".") if segment]
    candidate = segments[-1] if segments else f"{json_column}_value"
    if not candidate or candidate == json_column:
        candidate = f"{json_column}_value"
    return candidate


@lru_cache(maxsize=1)
def _supports_list_transform_returning_list() -> bool:
    """Probe whether the installed DuckDB can transform a list into a list.

    ``list_transform(l, x -> [x, x * 2])`` is only available in newer DuckDB
    releases.  The probe runs a tiny query against a throw-away in-memory
    connection so the answer reflects the *installed* DuckDB version.
    """
    try:
        probe_conn = duckdb.connect()
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        probe_conn.execute("SELECT list_transform([1, 2], x -> [x, x * 2])")
        return True
    except Exception:
        return False
    finally:
        try:
            probe_conn.close()
        except Exception:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Public verbs
# ---------------------------------------------------------------------------


def to_struct(
    relation: duckdb.DuckDBPyRelation,
    columns: Union[str, list[str]],
    struct_column: Optional[str] = None,
    *,
    drop_source: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Build a STRUCT column from one or more columns.

    The generated STRUCT uses the source column names as struct keys.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str
        Column(s) to pack into the STRUCT.
    struct_column : str, optional
        Name of the resulting STRUCT column.  Defaults to ``"struct"``.
    drop_source : bool
        When True the source columns are removed from the output.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation containing the new STRUCT column.

    Raises
    ------
    ValueError
        If ``columns`` is empty, references an unknown column, or
        ``struct_column`` is not a non-empty string.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import to_struct
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    >>> to_struct(rel, ["a", "b"], "addr", conn=conn).columns
    ['a', 'b', 'addr']
    """
    columns = _as_list(columns)
    if struct_column is None:
        struct_column = "struct"
    if not isinstance(struct_column, str) or not struct_column.strip():
        raise ValueError("struct_column must be a non-empty string")

    _ensure_columns_exist(relation.columns, columns)

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    old_columns = list(relation.columns)

    packed = ", ".join(f"{_quote_id(col)} := {_quote_id(col)}" for col in columns)
    struct_expr = f"struct_pack({packed}) AS {_quote_id(struct_column)}"

    source = set(columns)
    kept = [
        col
        for col in old_columns
        if (not drop_source or col not in source) and col != struct_column
    ]
    select_parts = [_quote_id(col) for col in kept] + [struct_expr]
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def from_struct(
    relation: duckdb.DuckDBPyRelation,
    struct_column: str,
    fields: Optional[list[str]] = None,
    *,
    prefix: Optional[str] = None,
    drop_source: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Explode a STRUCT column into separate scalar columns.

    When ``fields`` is given the requested keys are extracted with
    :func:`struct_extract` (safe for keys containing spaces or punctuation).
    When ``fields`` is ``None`` the whole STRUCT is expanded with ``UNNEST``
    so every key becomes its own column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    struct_column : str
        The STRUCT column to explode.
    fields : list of str, optional
        Struct keys to extract.  ``None`` expands every key.
    prefix : str, optional
        Prefix prepended to the output column names (e.g. ``"addr_"``).
    drop_source : bool
        When True the source STRUCT column is removed from the output.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with the exploded columns.

    Raises
    ------
    ValueError
        If the column is unknown, is not a STRUCT column, or a requested
        field is not a key of the STRUCT.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import from_struct
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"s": [{"x": 1, "y": 2}]}))
    >>> from_struct(rel, "s", ["x"], prefix="p_", conn=conn).columns
    ['s', 'p_x']
    """
    _ensure_columns_exist(relation.columns, [struct_column])
    _require_struct(relation, struct_column)

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    all_keys = _struct_keys(conn, table_name, struct_column)

    if fields is None:
        selected = list(all_keys)
    else:
        selected = _as_list(fields)
        unknown = [field for field in selected if field not in all_keys]
        if unknown:
            raise ValueError(f"Unknown struct field(s): {unknown}. Available fields: {all_keys}")

    old_columns = list(relation.columns)
    kept = [col for col in old_columns if not (drop_source and col == struct_column)]

    def _output_name(field: str) -> str:
        return f"{prefix}{field}" if prefix else field

    if fields is None:
        # Whole-struct expansion via UNNEST: the struct keys become columns.
        inner, restore = _carry(kept)
        inner_parts = [inner] if inner else []
        inner_parts.append(f"UNNEST({_quote_id(struct_column)})")
        extra = [f"{_quote_id(field)} AS {_quote_id(_output_name(field))}" for field in selected]
        query = (
            f"SELECT {_outer_select(restore, extra)} "
            f"FROM (SELECT {', '.join(inner_parts)} FROM {table_name})"
        )
        return conn.query(query)

    # Selective extraction via struct_extract.
    select_parts = [_quote_id(col) for col in kept]
    for field in selected:
        expr = (
            f"struct_extract({_quote_id(struct_column)}, {_sql_literal(field)}) "
            f"AS {_quote_id(_output_name(field))}"
        )
        select_parts.append(expr)
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    return conn.query(query)


def to_list(
    relation: duckdb.DuckDBPyRelation,
    columns: Union[str, list[str]],
    list_column: Optional[str] = None,
    *,
    drop_source: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Build a LIST column from one or more columns.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str
        Column(s) to pack into the LIST.
    list_column : str, optional
        Name of the resulting LIST column.  Defaults to ``"list"``.
    drop_source : bool
        When True the source columns are removed from the output.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation containing the new LIST column.

    Raises
    ------
    ValueError
        If ``columns`` is empty, references an unknown column, or
        ``list_column`` is not a non-empty string.

    Notes
    -----
    DuckDB's ``list_value`` requires all elements to share one type.  When
    the source columns have different DuckDB types they are cast to
    ``VARCHAR`` so the LIST can still be built.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import to_list
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    >>> to_list(rel, ["a", "b"], "vals", conn=conn).columns
    ['a', 'b', 'vals']
    """
    columns = _as_list(columns)
    if list_column is None:
        list_column = "list"
    if not isinstance(list_column, str) or not list_column.strip():
        raise ValueError("list_column must be a non-empty string")

    _ensure_columns_exist(relation.columns, columns)

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    old_columns = list(relation.columns)

    types = _column_types(relation)
    source_types = {types.get(col, "") for col in columns}
    needs_cast = len(source_types) > 1

    parts = []
    for col in columns:
        if needs_cast:
            parts.append(f"CAST({_quote_id(col)} AS VARCHAR)")
        else:
            parts.append(_quote_id(col))
    list_expr = f"list_value({', '.join(parts)}) AS {_quote_id(list_column)}"

    source = set(columns)
    kept = [
        col for col in old_columns if (not drop_source or col not in source) and col != list_column
    ]
    select_parts = [_quote_id(col) for col in kept] + [list_expr]
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def from_list(
    relation: duckdb.DuckDBPyRelation,
    list_column: str,
    *,
    index: bool = False,
    output_column: Optional[str] = None,
    drop_source: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Expand a LIST column into rows using ``UNNEST``.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    list_column : str
        The LIST column to expand.
    index : bool
        When True a 1-based position column is emitted next to every
        exploded value.
    output_column : str, optional
        Name of the position column.  Defaults to ``"pos"``.  Only used
        when ``index`` is True.
    drop_source : bool
        When True the source LIST column is removed from the output.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with one row per LIST element.

    Raises
    ------
    ValueError
        If the column is unknown or is not a LIST column.

    Notes
    -----
    The exploded values are emitted in a column named
    ``f"{list_column}_value"`` so the original LIST column can be kept
    side-by-side.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import from_list
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"vals": [[1, 2]]}))
    >>> from_list(rel, "vals", index=True, conn=conn).columns
    ['vals', 'vals_value', 'pos']
    """
    _ensure_columns_exist(relation.columns, [list_column])
    _require_list(relation, list_column)

    if output_column is None:
        output_column = "pos"
    if not isinstance(output_column, str) or not output_column.strip():
        raise ValueError("output_column must be a non-empty string")

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    old_columns = list(relation.columns)
    kept = [col for col in old_columns if not (drop_source and col == list_column)]

    inner, restore = _carry(kept)
    inner_parts = [inner] if inner else []
    inner_parts.append(f"UNNEST({_quote_id(list_column)}) AS {_quote_id(_TMP + 'v')}")
    if index:
        inner_parts.append(
            f"generate_subscripts({_quote_id(list_column)}, 1) AS {_quote_id(_TMP + 'i')}"
        )

    extra = [f"{_quote_id(_TMP + 'v')} AS {_quote_id(f'{list_column}_value')}"]
    if index:
        extra.append(f"{_quote_id(_TMP + 'i')} AS {_quote_id(output_column)}")

    query = (
        f"SELECT {_outer_select(restore, extra)} "
        f"FROM (SELECT {', '.join(inner_parts)} FROM {table_name})"
    )
    return conn.query(query)


def list_transform(
    relation: duckdb.DuckDBPyRelation,
    list_column: str,
    func: str,
    *,
    output_column: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Apply a DuckDB lambda over a LIST column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    list_column : str
        The LIST column to transform.
    func : str
        A SQL lambda body, e.g. ``"x -> x * 2"``.
    output_column : str, optional
        Name of the resulting column.  When ``None`` the source column is
        replaced in place.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with the transformed column.

    Raises
    ------
    ValueError
        If the column is unknown, is not a LIST column, or ``func`` is not
        a valid single SQL fragment.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import list_transform
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"vals": [[1, 2]]}))
    >>> list_transform(rel, "vals", "x -> x * 2", conn=conn).df()["vals"].tolist()
    [[2, 4]]
    """
    _ensure_columns_exist(relation.columns, [list_column])
    _require_list(relation, list_column)
    if not isinstance(func, str):
        raise ValueError("func must be a SQL lambda body string")
    _validate_sql_fragment(func, context="List transform lambda")

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    old_columns = list(relation.columns)

    expr = f"list_transform({_quote_id(list_column)}, {func})"
    if output_column is None or output_column == list_column:
        select_parts = [_quote_id(col) for col in old_columns if col != list_column]
        select_parts.append(f"{expr} AS {_quote_id(list_column)}")
    else:
        select_parts = [_quote_id(col) for col in old_columns]
        select_parts.append(f"{expr} AS {_quote_id(output_column)}")

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    return conn.query(query)


def list_filter(
    relation: duckdb.DuckDBPyRelation,
    list_column: str,
    func: str,
    *,
    output_column: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Apply a DuckDB predicate lambda over a LIST column.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    list_column : str
        The LIST column to filter.
    func : str
        A SQL lambda body, e.g. ``"x -> x > 2"``.
    output_column : str, optional
        Name of the resulting column.  When ``None`` the source column is
        replaced in place.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with the filtered column.

    Raises
    ------
    ValueError
        If the column is unknown, is not a LIST column, or ``func`` is not
        a valid single SQL fragment.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import list_filter
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"vals": [[1, 2, 3]]}))
    >>> list_filter(rel, "vals", "x -> x > 1", conn=conn).df()["vals"].tolist()
    [[2, 3]]
    """
    _ensure_columns_exist(relation.columns, [list_column])
    _require_list(relation, list_column)
    if not isinstance(func, str):
        raise ValueError("func must be a SQL lambda body string")
    _validate_sql_fragment(func, context="List filter lambda")

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    old_columns = list(relation.columns)

    expr = f"list_filter({_quote_id(list_column)}, {func})"
    if output_column is None or output_column == list_column:
        select_parts = [_quote_id(col) for col in old_columns if col != list_column]
        select_parts.append(f"{expr} AS {_quote_id(list_column)}")
    else:
        select_parts = [_quote_id(col) for col in old_columns]
        select_parts.append(f"{expr} AS {_quote_id(output_column)}")

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    return conn.query(query)


def list_transforms(
    relation: duckdb.DuckDBPyRelation,
    list_column: str,
    func: str,
    *,
    output_column: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Apply a DuckDB lambda over a LIST column where the lambda returns a LIST.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    list_column : str
        The LIST column to transform.
    func : str
        A SQL lambda body returning a list, e.g. ``"x -> [x, x * 2]"``.
    output_column : str, optional
        Name of the resulting column.  When ``None`` the source column is
        replaced in place.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with the transformed column.

    Raises
    ------
    NotImplementedError
        If the installed DuckDB cannot transform a LIST into a LIST.
    ValueError
        If the column is unknown, is not a LIST column, or ``func`` is not
        a valid single SQL fragment.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import list_transforms
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"vals": [[1, 2]]}))
    >>> list_transforms(rel, "vals", "x -> [x, x * 2]", conn=conn).df()["vals"].tolist()
    [[1, 2], [2, 4]]
    """
    if not _supports_list_transform_returning_list():
        raise NotImplementedError(
            "The installed DuckDB version does not support transforming a LIST into a "
            "LIST (list_transform with a lambda returning a list). "
            "Use list_transform() with a scalar lambda instead."
        )

    _ensure_columns_exist(relation.columns, [list_column])
    _require_list(relation, list_column)
    if not isinstance(func, str):
        raise ValueError("func must be a SQL lambda body string")
    _validate_sql_fragment(func, context="List transform lambda")

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    old_columns = list(relation.columns)

    expr = f"list_transform({_quote_id(list_column)}, {func})"
    if output_column is None or output_column == list_column:
        select_parts = [_quote_id(col) for col in old_columns if col != list_column]
        select_parts.append(f"{expr} AS {_quote_id(list_column)}")
    else:
        select_parts = [_quote_id(col) for col in old_columns]
        select_parts.append(f"{expr} AS {_quote_id(output_column)}")

    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"
    return conn.query(query)


def json_extract(
    relation: duckdb.DuckDBPyRelation,
    json_column: str,
    path: str,
    *,
    output_column: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Extract a value from a JSON column using DuckDB's ``json_extract``.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    json_column : str
        Column holding the JSON payload (a ``VARCHAR`` or ``JSON`` column).
    path : str
        A JSON pointer such as ``"$.a.b"`` or a plain key name such as
        ``"a"``.
    output_column : str, optional
        Name of the extracted column.  When ``None`` the name is derived
        from the last segment of the path.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with the extracted value added.

    Raises
    ------
    ValueError
        If the column is unknown or ``path`` is not a non-empty string.

    Notes
    -----
    The extracted value keeps DuckDB's ``JSON`` type; cast it in SQL (for
    example ``::VARCHAR``) when a plain string is required.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import json_extract
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"j": ['{"a": {"b": 5}}']}))
    >>> json_extract(rel, "j", "$.a.b", conn=conn).columns
    ['j', 'b']
    """
    _ensure_columns_exist(relation.columns, [json_column])
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    if output_column is None:
        output_column = _default_json_output(json_column, path)
    if not isinstance(output_column, str) or not output_column.strip():
        raise ValueError("output_column must be a non-empty string")

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    old_columns = list(relation.columns)

    expr = (
        f"json_extract({_quote_id(json_column)}, {_sql_literal(path)}) "
        f"AS {_quote_id(output_column)}"
    )
    select_parts = [_quote_id(col) for col in old_columns]
    select_parts.append(expr)
    query = f"SELECT {', '.join(select_parts)} FROM {table_name}"

    return conn.query(query)


def unnest(
    relation: duckdb.DuckDBPyRelation,
    columns: Union[str, list[str]],
    *,
    drop_source: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Expand one or more LIST columns into rows using ``UNNEST``.

    When several LIST columns are given DuckDB unnests them together in a
    parallel (zip) fashion: the shortest list decides the row count and the
    longer lists are padded with ``NULL``.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str
        LIST column(s) to expand.
    drop_source : bool
        When True the source LIST columns are removed from the output.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation with one row per LIST element.

    Raises
    ------
    ValueError
        If ``columns`` is empty, references an unknown column, or a column
        is not a LIST column.

    Notes
    -----
    The exploded values of column ``C`` are emitted in a column named
    ``f"{C}_value"``.

    Examples
    --------
    >>> import duckdb, pandas as pd
    >>> from pyduck_janitor.nested_ops import unnest
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"vals": [[1, 2]]}))
    >>> unnest(rel, ["vals"], conn=conn).columns
    ['vals', 'vals_value']
    """
    columns = _as_list(columns)
    _ensure_columns_exist(relation.columns, columns)
    for column in columns:
        _require_list(relation, column)

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    old_columns = list(relation.columns)

    source = set(columns)
    kept = [col for col in old_columns if not (drop_source and col in source)]

    inner, restore = _carry(kept)
    inner_parts = [inner] if inner else []
    extra: list[str] = []
    for index, column in enumerate(columns):
        alias = f"{_TMP}v{index}"
        inner_parts.append(f"UNNEST({_quote_id(column)}) AS {_quote_id(alias)}")
        extra.append(f"{_quote_id(alias)} AS {_quote_id(f'{column}_value')}")

    query = (
        f"SELECT {_outer_select(restore, extra)} "
        f"FROM (SELECT {', '.join(inner_parts)} FROM {table_name})"
    )
    return conn.query(query)
