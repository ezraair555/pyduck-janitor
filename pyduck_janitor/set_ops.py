"""
DuckDB-native set operations for DuckJanitor.

This module provides the SQL set operations (and their multi-way
concatenation) as *module-level functions* that take a
:class:`duckdb.DuckDBPyRelation` as their first argument and return a **new**
relation.  The input relation is never mutated.

Verbs
-----
``union``
    ``UNION`` / ``UNION ALL`` between two relations, optionally matched by
    column *name* instead of position.
``intersect``
    ``INTERSECT`` -- the distinct rows present in both relations.
``except_``
    ``EXCEPT`` -- the distinct rows of the left relation that are absent from
    the right one.  Also exported as ``except_rows``.
``concat_rows``
    Stack a list of relations into one relation (the "many files" verb).
``select_except``
    ``COLUMNS(* EXCLUDE (...))`` / ``COLUMNS(* REPLACE (...))`` star
    expressions.

Every generated statement is built with the shared helpers in
:mod:`pyduck_janitor.cleaning_ops` (``_quote_id``, ``_register_relation``,
``_ensure_columns_exist``, ``_validate_sql_fragment``) so identifiers and user
supplied SQL fragments are always quoted / validated safely.

DuckDB 1.5.2 behaviour that this module relies on
-------------------------------------------------
* ``SELECT * FROM a UNION [ALL] BY NAME SELECT * FROM b`` **is** supported,
  including the ``ALL`` variant.  This is the main win over a positional
  ``UNION``: two frames with differently ordered (or differently named)
  columns still combine correctly.
* ``UNION BY NAME`` fills columns that are missing on one side with ``NULL``.
* ``INTERSECT`` and ``EXCEPT`` have **no** ``BY NAME`` variant in DuckDB
  1.5.2 (the parser rejects ``INTERSECT BY NAME``).  Their ``by_name=True``
  mode is therefore emulated by padding both sides to the union of the two
  column sets with ``NULL`` and then running the positional statement -- so
  the "unmatched columns become NULL" rule matches ``UNION BY NAME``.
* ``SELECT COLUMNS(* EXCLUDE (...))`` and ``SELECT COLUMNS(* REPLACE (...))``
  are supported and may be combined in a single star expression.  A column
  may not appear in both lists, and the star expression may not resolve to an
  empty select list.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import duckdb

from .cleaning_ops import (
    _require_conn,
    _ensure_columns_exist,
    _quote_id,
    _register_relation,
    _validate_sql_fragment,
)

__all__ = [
    "union",
    "intersect",
    "except_",
    "except_rows",
    "concat_rows",
    "select_except",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_relation(obj: Any, label: str = "relation") -> duckdb.DuckDBPyRelation:
    """Return the underlying :class:`duckdb.DuckDBPyRelation` of *obj*.

    Accepts a raw DuckDB relation or any DuckJanitor-like wrapper that
    exposes a ``relation`` property.

    Parameters
    ----------
    obj : Any
        The object to unwrap.
    label : str, default ``'relation'``
        Label used in error messages.

    Returns
    -------
    duckdb.DuckDBPyRelation
        The unwrapped relation.

    Raises
    ------
    ValueError
        If *obj* is neither a DuckDB relation nor a DuckJanitor-like object.
    """
    if isinstance(obj, duckdb.DuckDBPyRelation):
        return obj
    inner = getattr(obj, "relation", None)
    if isinstance(inner, duckdb.DuckDBPyRelation):
        return inner
    raise ValueError(
        f"{label} must be a duckdb.DuckDBPyRelation or a DuckJanitor-like object "
        f"exposing a 'relation' property. Got {type(obj).__name__}."
    )


def _owning_connection(obj: Any) -> Optional[duckdb.DuckDBPyConnection]:
    """Return the connection that owns *obj*, when it can be determined.

    DuckJanitor-like objects expose a ``connection`` property.  Raw
    :class:`duckdb.DuckDBPyRelation` objects do not, in which case ``None`` is
    returned and the caller must supply the connection explicitly.
    """
    conn = getattr(obj, "connection", None)
    if isinstance(conn, duckdb.DuckDBPyConnection):
        return conn
    return None


def _resolve_connection(
    conn: Optional[duckdb.DuckDBPyConnection], *objects: Any
) -> duckdb.DuckDBPyConnection:
    """Return *conn*, or infer it from *objects* when it was not supplied.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection, optional
        The caller supplied connection.
    objects : Any
        Candidate objects whose owning connection may be used.

    Returns
    -------
    duckdb.DuckDBPyConnection
        The connection to use.

    Raises
    ------
    ValueError
        If no connection could be determined.
    """
    if conn is not None:
        return conn
    for obj in objects:
        found = _owning_connection(obj)
        if found is not None:
            return found
    raise ValueError(
        "A DuckDB connection is required. Pass the connection that owns the "
        "relation(s) via the 'conn' keyword argument."
    )


def _register(
    conn: duckdb.DuckDBPyConnection,
    relation: duckdb.DuckDBPyRelation,
    label: str,
) -> str:
    """Register *relation* on *conn* and return its temporary SQL name.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        The connection that owns the relation.
    relation : duckdb.DuckDBPyRelation
        The relation to register.
    label : str
        Label used in error messages.

    Returns
    -------
    str
        The temporary SQL name.

    Raises
    ------
    ValueError
        If the relation belongs to a different connection than *conn*.
    """
    try:
        conn = _require_conn(conn)
        return _register_relation(conn, relation)
    except duckdb.InvalidInputException as exc:
        raise ValueError(
            f"{label} was created on a different DuckDB connection than the one in use. "
            "All relations combined by a set operation must live on the same connection."
        ) from exc


def _column_types(relation: duckdb.DuckDBPyRelation) -> dict[str, str]:
    """Return a mapping of column name to DuckDB type string."""
    return {name: str(t) for name, t in zip(relation.columns, relation.types)}


def _check_column_counts(
    left: duckdb.DuckDBPyRelation, right: duckdb.DuckDBPyRelation, verb: str
) -> None:
    """Raise ``ValueError`` when the two relations have different column counts.

    DuckDB's positional ``UNION`` / ``INTERSECT`` / ``EXCEPT`` require the
    same number of result columns; a mismatch surfaces as a binder error.
    This helper converts it into a friendly, actionable ``ValueError``.
    """
    left_count = len(left.columns)
    right_count = len(right.columns)
    if left_count != right_count:
        raise ValueError(
            f"{verb} requires both relations to have the same number of columns. "
            f"The left relation has {left_count} column(s) {list(left.columns)} and the "
            f"right relation has {right_count} column(s) {list(right.columns)}. "
            "Use by_name=True to combine relations with different columns."
        )


def _padded_projection(
    available_columns: list[str],
    order: list[str],
    types: dict[str, str],
) -> str:
    """Build a ``SELECT`` list padding a frame out to *order*.

    Columns that are missing from *order* are dropped; columns of *order*
    that are missing from the frame become ``CAST(NULL AS <type>) AS "name"``
    so the projection keeps the exact position and type of the padded frame.

    Parameters
    ----------
    available_columns : list of str
        The columns actually present in the frame.
    order : list of str
        The desired column order.
    types : dict of {column_name: duckdb_type}
        Type to use when padding a missing column with ``NULL``.

    Returns
    -------
    str
        The comma separated select list.
    """
    available = set(available_columns)
    parts = []
    for column in order:
        if column in available:
            parts.append(_quote_id(column))
        else:
            parts.append(f"CAST(NULL AS {types[column]}) AS {_quote_id(column)}")
    return ", ".join(parts)


def _padded_frame(
    table_name: str,
    order: list[str],
    types: dict[str, str],
    available_columns: list[str],
) -> str:
    """Build a padded sub-query over *table_name*.

    Parameters
    ----------
    table_name : str
        The temporary SQL name of the registered relation.
    order : list of str
        The desired column order.
    types : dict of {column_name: duckdb_type}
        Types used when padding a missing column with ``NULL``.
    available_columns : list of str
        The columns actually present in the relation.

    Returns
    -------
    str
        A ``SELECT ... FROM <table_name>`` fragment.
    """
    projection = _padded_projection(available_columns, order, types)
    return f"SELECT {projection} FROM {table_name}"


def _union_column_order(left: duckdb.DuckDBPyRelation, right: duckdb.DuckDBPyRelation) -> list[str]:
    """Return the union of both column sets, left-side order first."""
    order = list(left.columns)
    seen = set(order)
    for column in right.columns:
        if column not in seen:
            order.append(column)
            seen.add(column)
    return order


def _shared_types(left: duckdb.DuckDBPyRelation, right: duckdb.DuckDBPyRelation) -> dict[str, str]:
    """Return the type used for each column in a padded by-name frame.

    When a column exists on both sides the left type wins (DuckDB unifies
    the two branches anyway).  When it exists on only one side that side's
    type is used so the padded ``NULL`` matches the real column exactly.
    """
    left_types = _column_types(left)
    right_types = _column_types(right)
    types: dict[str, str] = {}
    for column in _union_column_order(left, right):
        types[column] = left_types.get(column) or right_types[column]
    return types


def _star_set_sql(
    verb: str,
    left_name: str,
    right_name: str,
    by_name: bool,
    all_rows: bool = False,
) -> str:
    """Build the SQL text for a two-relation set operation.

    Parameters
    ----------
    verb : str
        ``'UNION'``, ``'INTERSECT'`` or ``'EXCEPT'``.
    left_name, right_name : str
        Temporary SQL names of the registered relations.
    by_name : bool
        Whether to match columns by name (``UNION BY NAME``).
    all_rows : bool
        Whether to keep duplicate rows (``UNION ALL``).

    Returns
    -------
    str
        The SQL statement.
    """
    keyword = f"{verb} ALL" if all_rows else verb
    if by_name:
        keyword = f"{keyword} BY NAME"
    return f"SELECT * FROM {left_name} {keyword} SELECT * FROM {right_name}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def union(
    relation: Any,
    other: duckdb.DuckDBPyRelation,
    *,
    by_name: bool = False,
    all: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Combine two relations with ``UNION`` / ``UNION ALL``.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The left relation.
    other : duckdb.DuckDBPyRelation
        The right relation.
    by_name : bool, default ``False``
        When ``True``, DuckDB's ``UNION BY NAME`` is used and the two
        relations are matched on column *name* instead of position.  Columns
        that only exist on one side are added to the result and filled with
        ``NULL`` for the rows coming from the other side.
    all : bool, default ``False``
        When ``True``, duplicate rows are kept (``UNION ALL``).  When
        ``False``, the result contains distinct rows only.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relations.  Required unless a relation
        is a DuckJanitor-like object that exposes its connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A new relation holding the combined rows.  The input relations are
        not modified.

    Raises
    ------
    ValueError
        If no connection is available, if the relations live on different
        connections, or -- when ``by_name`` is ``False`` -- if the two
        relations have a different number of columns.

    Notes
    -----
    ``UNION ALL BY NAME`` **is** supported by DuckDB 1.5.2, so ``all=True``
    may be combined with ``by_name=True``.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> conn = duckdb.connect()
    >>> left = conn.from_df(pd.DataFrame({"a": [1, 2]}))
    >>> right = conn.from_df(pd.DataFrame({"a": [2, 3]}))
    >>> union(left, right, all=True, conn=conn).df()["a"].tolist()
    [1, 2, 2, 3]
    >>> sorted(union(left, right, conn=conn).df()["a"].tolist())
    [1, 2, 3]
    """
    left = _as_relation(relation, "relation")
    right = _as_relation(other, "other")
    conn = _resolve_connection(conn, relation, other)

    if not by_name:
        _check_column_counts(left, right, "union()")

    left_name = _register(conn, left, "relation")
    right_name = _register(conn, right, "other")

    sql = _star_set_sql("UNION", left_name, right_name, by_name, all_rows=all)
    return conn.query(sql)


def intersect(
    relation: Any,
    other: duckdb.DuckDBPyRelation,
    *,
    by_name: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Return the distinct rows present in both relations.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The left relation.
    other : duckdb.DuckDBPyRelation
        The right relation.
    by_name : bool, default ``False``
        When ``True``, rows are matched on column *name* instead of on column
        position.  DuckDB 1.5.2 has no ``INTERSECT BY NAME`` statement, so
        the operation is emulated: both sides are padded to the union of the
        two column sets with ``NULL`` and the positional ``INTERSECT`` is
        then applied.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relations.  Required unless a relation
        is a DuckJanitor-like object that exposes its connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A new relation holding the rows of the left relation that also occur
        in the right relation, without duplicates.

    Raises
    ------
    ValueError
        If no connection is available, if the relations live on different
        connections, or -- when ``by_name`` is ``False`` -- if the column
        counts differ.

    Notes
    -----
    With ``by_name=True`` the result keeps the *left* relation's column
    order, padded with the right relation's extra columns (filled with
    ``NULL`` for the rows that came from the left).  ``NULL`` values compare
    equal inside DuckDB set operations, so ``NULL`` matches ``NULL``.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> conn = duckdb.connect()
    >>> left = conn.from_df(pd.DataFrame({"a": [1, 2, 2]}))
    >>> right = conn.from_df(pd.DataFrame({"a": [2, 3]}))
    >>> intersect(left, right, conn=conn).df()["a"].tolist()
    [2]
    """
    left = _as_relation(relation, "relation")
    right = _as_relation(other, "other")
    conn = _resolve_connection(conn, relation, other)

    left_name = _register(conn, left, "relation")
    right_name = _register(conn, right, "other")

    if not by_name:
        _check_column_counts(left, right, "intersect()")
        return conn.query(_star_set_sql("INTERSECT", left_name, right_name, False))

    order = _union_column_order(left, right)
    types = _shared_types(left, right)
    left_frame = _padded_frame(left_name, order, types, list(left.columns))
    right_frame = _padded_frame(right_name, order, types, list(right.columns))
    sql = f"{left_frame} INTERSECT {right_frame}"
    return conn.query(sql)


def except_(
    relation: Any,
    other: duckdb.DuckDBPyRelation,
    *,
    by_name: bool = False,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Return the distinct rows of the left relation that are absent from the right.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The left relation.
    other : duckdb.DuckDBPyRelation
        The right relation.
    by_name : bool, default ``False``
        When ``True``, rows are matched on column *name* instead of on column
        position.  DuckDB 1.5.2 has no ``EXCEPT BY NAME`` statement, so the
        operation is emulated: both sides are padded to the union of the two
        column sets with ``NULL`` and the positional ``EXCEPT`` is then
        applied.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relations.  Required unless a relation
        is a DuckJanitor-like object that exposes its connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A new relation holding the rows of the left relation that do not occur
        in the right relation, without duplicates.

    Raises
    ------
    ValueError
        If no connection is available, if the relations live on different
        connections, or -- when ``by_name`` is ``False`` -- if the column
        counts differ.

    Notes
    -----
    The function is named ``except_`` because ``except`` is a Python keyword.
    ``except_rows`` is exported as an alias for discoverability.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> conn = duckdb.connect()
    >>> left = conn.from_df(pd.DataFrame({"a": [1, 2]}))
    >>> right = conn.from_df(pd.DataFrame({"a": [2]}))
    >>> except_(left, right, conn=conn).df()["a"].tolist()
    [1]
    """
    left = _as_relation(relation, "relation")
    right = _as_relation(other, "other")
    conn = _resolve_connection(conn, relation, other)

    left_name = _register(conn, left, "relation")
    right_name = _register(conn, right, "other")

    if not by_name:
        _check_column_counts(left, right, "except_()")
        return conn.query(_star_set_sql("EXCEPT", left_name, right_name, False))

    order = _union_column_order(left, right)
    types = _shared_types(left, right)
    left_frame = _padded_frame(left_name, order, types, list(left.columns))
    right_frame = _padded_frame(right_name, order, types, list(right.columns))
    sql = f"{left_frame} EXCEPT {right_frame}"
    return conn.query(sql)


#: Alias for :func:`except_` -- ``except`` is a Python keyword.
except_rows = except_


def concat_rows(
    relation: Any,
    others: list[duckdb.DuckDBPyRelation],
    *,
    by_name: bool = True,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Stack several relations into a single relation.

    This is the "load many files" verb: it appends the rows of every
    relation in the list below the rows of *relation*.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The first relation.
    others : list of duckdb.DuckDBPyRelation
        The remaining relations to stack.  Must be a non-empty list; every
        entry must be a DuckDB relation or a DuckJanitor-like object.
    by_name : bool, default ``True``
        When ``True`` the relations are combined with ``UNION ALL BY NAME``,
        so they may have different column names and orders.  Columns that are
        missing from one relation are filled with ``NULL`` for its rows.
        When ``False`` the relations are combined positionally and must have
        the same number of columns.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relations.  Required unless a relation
        is a DuckJanitor-like object that exposes its connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A new relation holding the stacked rows.

    Raises
    ------
    ValueError
        If ``others`` is not a list, is empty, holds an object that is not a
        relation, if no connection is available, if the relations live on
        different connections, or -- when ``by_name`` is ``False`` -- if the
        relations have different numbers of columns.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> conn = duckdb.connect()
    >>> first = conn.from_df(pd.DataFrame({"a": [1]}))
    >>> second = conn.from_df(pd.DataFrame({"a": [2]}))
    >>> third = conn.from_df(pd.DataFrame({"a": [3]}))
    >>> concat_rows(first, [second, third], conn=conn).df()["a"].tolist()
    [1, 2, 3]
    """
    if not isinstance(others, (list, tuple)):
        raise ValueError(f"others must be a list of relations, got {type(others).__name__}.")
    if len(others) == 0:
        raise ValueError("others must contain at least one relation to concatenate.")

    items = [relation, *others]
    relations = [_as_relation(relation, "relation")]
    relations.extend(_as_relation(item, f"others[{index}]") for index, item in enumerate(others))

    conn = _resolve_connection(conn, *items)

    if not by_name:
        counts = {len(item.columns) for item in relations}
        if len(counts) > 1:
            details = ", ".join(
                f"{list(item.columns)} ({len(item.columns)} column(s))" for item in relations
            )
            raise ValueError(
                "concat_rows(by_name=False) requires every relation to have the same "
                f"number of columns. Got: {details}. Use by_name=True to stack "
                "relations with different columns."
            )

    names = [_register(conn, item, f"relation[{index}]") for index, item in enumerate(relations)]

    parts = [f"SELECT * FROM {name}" for name in names]
    separator = " UNION ALL BY NAME " if by_name else " UNION ALL "
    return conn.query(separator.join(parts))


def select_except(
    relation: Any,
    columns: Union[str, list[str]],
    *,
    replace: Optional[dict[str, str]] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Exclude and/or replace columns with a DuckDB star expression.

    Builds ``SELECT COLUMNS(* EXCLUDE (...) REPLACE (...))`` so that columns
    can be dropped and rewritten in one pass, without having to spell out the
    full column list.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    columns : str or list of str
        Name(s) of the columns to exclude.  An empty list excludes nothing.
    replace : dict of {column_name: sql_expression}, optional
        Columns to replace in place.  Each key must name an existing column
        and each value is a SQL expression evaluated against the relation.
    conn : duckdb.DuckDBPyConnection, optional
        The connection that owns the relation.  Required unless the relation
        is a DuckJanitor-like object that exposes its connection.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A new relation with the requested columns excluded / replaced.

    Raises
    ------
    ValueError
        If no connection is available, if a named column does not exist, if a
        column appears both in ``columns`` and in ``replace``, if a
        replacement expression is not a valid SQL fragment, or if the
        exclusion list would remove every column.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> conn = duckdb.connect()
    >>> rel = conn.from_df(pd.DataFrame({"a": [1], "b": [2]}))
    >>> select_except(rel, "b", conn=conn).df().columns.tolist()
    ['a']
    >>> select_except(rel, [], replace={"a": "a * 10"}, conn=conn).df()["a"].tolist()
    [10]
    """
    left = _as_relation(relation, "relation")
    conn = _resolve_connection(conn, relation)

    if isinstance(columns, str):
        exclude = [columns]
    elif isinstance(columns, (list, tuple)):
        exclude = list(columns)
    else:
        raise ValueError(
            f"columns must be a column name or a list of column names, "
            f"got {type(columns).__name__}."
        )
    if not all(isinstance(column, str) for column in exclude):
        raise ValueError("columns must contain only column names (strings).")

    available = list(left.columns)
    _ensure_columns_exist(available, exclude)

    replace_map: dict[str, str] = {}
    if replace is not None:
        if not isinstance(replace, dict):
            raise ValueError(
                f"replace must be a dict of {{column_name: sql_expression}}, "
                f"got {type(replace).__name__}."
            )
        for key, expression in replace.items():
            if not isinstance(key, str):
                raise ValueError("replace keys must be column names (strings).")
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError(f"replace[{key!r}] must be a non-empty SQL expression string.")
            _validate_sql_fragment(expression, context=f"replace[{key!r}] expression")
        _ensure_columns_exist(available, list(replace.keys()))
        overlap = sorted(set(replace) & set(exclude))
        if overlap:
            raise ValueError(
                f"Column(s) {overlap} appear in both the exclude list and the replace "
                "mapping. A column may not be excluded and replaced at the same time."
            )
        replace_map = dict(replace)

    remaining = [column for column in available if column not in exclude]
    if not remaining:
        raise ValueError(
            f"Cannot exclude every column of the relation ({available}). "
            "At least one column must remain."
        )

    clauses = []
    if exclude:
        excluded_list = ", ".join(_quote_id(column) for column in exclude)
        clauses.append(f"EXCLUDE ({excluded_list})")
    if replace_map:
        replacements = ", ".join(
            f"({expression}) AS {_quote_id(column)}" for column, expression in replace_map.items()
        )
        clauses.append(f"REPLACE ({replacements})")

    star = "COLUMNS(* " + " ".join(clauses) + ")" if clauses else "COLUMNS(*)"

    table_name = _register(conn, left, "relation")
    return conn.query(f"SELECT {star} FROM {table_name}")
