"""
Sampling and Python-UDF verbs for pyduck-janitor.

This module provides DuckDB-native sampling primitives plus registration of
Python callables as DuckDB scalar UDFs. Every verb operates on a
``duckdb.DuckDBPyRelation`` and returns a new relation, so the calls can be
chained with the rest of the cleaning API.

Verbs
-----
``sample``
    Random sample via DuckDB's ``USING SAMPLE`` clause.
``sample_rows``
    Deterministic N-row pick without a full sort.
``approx_distinct``
    Approximate distinct count (``approx_count_distinct``).
``approx_quantile``
    Approximate quantile (``approx_quantile``).
``register_python_function``
    Register a Python callable as a DuckDB scalar UDF on a connection.
``apply_python_function``
    Register a Python callable and apply it to a column.

DuckDB 1.5.2 sampling methods
-----------------------------
``USING SAMPLE`` accepts exactly three methods: ``reservoir``, ``bernoulli``
and ``system``. Only ``reservoir`` supports a discrete row count
(``n ROWS``); ``bernoulli`` and ``system`` are percentage-only and DuckDB
raises a parser error when they are combined with ``ROWS``.
"""

from __future__ import annotations

import inspect
import re
import uuid
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import duckdb

from .cleaning_ops import (
    _ensure_columns_exist,
    _quote_id,
    _register_relation,
    _require_conn,
)

__all__ = [
    "sample",
    "sample_rows",
    "approx_distinct",
    "approx_quantile",
    "register_python_function",
    "apply_python_function",
]


#: Sampling methods supported by DuckDB 1.5.2's ``USING SAMPLE`` clause.
SAMPLING_METHODS = ("reservoir", "bernoulli", "system")

#: Methods that accept a discrete ``n ROWS`` sample size.
ROW_COUNT_METHODS = ("reservoir",)

#: Python annotation -> DuckDB type string.
_PYTHON_TO_DUCKDB_TYPE = {
    bool: "BOOLEAN",
    int: "BIGINT",
    float: "DOUBLE",
    str: "VARCHAR",
    bytes: "BLOB",
}


def _format_percentage(frac: float) -> str:
    """Render *frac* (0..1) as a DuckDB percentage literal such as ``12.5%``."""
    return f"{frac * 100:.10g}%"


def _validate_seed(seed: Optional[int]) -> Optional[int]:
    """Validate the ``seed`` argument and return it unchanged."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validate_count(n: Any) -> int:
    """Validate a row-count argument and return it as an ``int``."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("n must be a non-negative integer")
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    return n


def _validate_fraction(frac: Any) -> float:
    """Validate a fraction argument and return it as a ``float``."""
    if isinstance(frac, bool) or not isinstance(frac, (int, float)):
        raise ValueError("frac must be a number between 0 and 1")
    frac = float(frac)
    if frac < 0 or frac > 1:
        raise ValueError("frac must be a number between 0 and 1")
    return frac


def _type_from_annotation(annotation: Any) -> Optional[str]:
    """Map a Python type annotation to a DuckDB type string, or ``None``."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return None
    try:
        if annotation in _PYTHON_TO_DUCKDB_TYPE:
            return _PYTHON_TO_DUCKDB_TYPE[annotation]
    except TypeError:  # pragma: no cover - unhashable annotation
        return None
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        if origin is Union:  # noqa: UP007 - runtime check of typing.Union
            for arg in args:
                if arg is type(None):
                    continue
                resolved = _type_from_annotation(arg)
                if resolved:
                    return resolved
            return None
        inner = args[0] if args else None
        resolved = _type_from_annotation(inner)
        if resolved:
            return resolved
    return None


def _positional_annotations(func: Callable[..., Any]) -> list[Any]:
    """Return the positional parameter annotations of *func*."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    annotations = []
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            annotations.append(parameter.annotation)
    return annotations


def _return_annotation(func: Callable[..., Any]) -> Any:
    """Return the return annotation of *func* (or ``inspect.Parameter.empty``)."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return inspect.Parameter.empty
    return signature.return_annotation


def _infer_param_types(func: Callable[..., Any]) -> Optional[list[str]]:
    """Infer DuckDB parameter types from *func*'s annotations, if possible."""
    annotations = _positional_annotations(func)
    if not annotations:
        return None
    resolved = []
    for annotation in annotations:
        duckdb_type = _type_from_annotation(annotation)
        if duckdb_type is None:
            return None
        resolved.append(duckdb_type)
    return resolved


def _infer_return_type(func: Callable[..., Any]) -> Optional[str]:
    """Infer the DuckDB return type of *func* from its annotation, if possible."""
    return _type_from_annotation(_return_annotation(func))


def _create_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    func: Callable[..., Any],
    param_types: Optional[list[str]],
    return_type: Optional[str],
) -> None:
    """Create a DuckDB scalar UDF, translating DuckDB errors into ValueError."""
    for attempt in range(2):
        try:
            conn.create_function(name, func, parameters=param_types, return_type=return_type)
            return
        except duckdb.NotImplementedException as exc:
            message = str(exc)
            if "already" in message.lower() and attempt == 0:
                # DuckDB 1.5.2 refuses to redefine an existing UDF; replace it.
                conn.remove_function(name)
                continue
            raise ValueError(f"Could not register Python UDF {name!r}: {message}") from exc
        except duckdb.InvalidInputException as exc:
            message = str(exc)
            if "Could not infer the return type" in message:
                raise ValueError(
                    f"Could not infer the DuckDB return type of the Python function "
                    f"{name!r}. Pass return_type explicitly, for example "
                    f"return_type='VARCHAR' (or 'BIGINT', 'DOUBLE', 'BOOLEAN')."
                ) from exc
            raise ValueError(f"Could not register Python UDF {name!r}: {message}") from exc
        except duckdb.Error as exc:
            raise ValueError(f"Could not register Python UDF {name!r}: {exc}") from exc


def sample(
    relation: duckdb.DuckDBPyRelation,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    *,
    seed: Optional[int] = None,
    method: str = "reservoir",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Draw a random sample of rows using DuckDB's ``USING SAMPLE`` clause.

    Exactly one of ``n`` (a row count) or ``frac`` (a fraction of the
    relation) must be supplied.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    n : int, optional
        Number of rows to draw. Mutually exclusive with ``frac``.
    frac : float, optional
        Fraction of rows to draw, between 0 and 1 inclusive.
        Mutually exclusive with ``n``.
    seed : int, optional
        Non-negative seed. When given, DuckDB's ``REPEATABLE`` clause is
        used so the same seed always yields the same sample.
    method : str, default 'reservoir'
        Sampling method. One of ``'reservoir'``, ``'bernoulli'`` or
        ``'system'``. Only ``'reservoir'`` supports a discrete ``n ROWS``
        sample size; ``'bernoulli'`` and ``'system'`` are percentage-only
        in DuckDB 1.5.2.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A new relation holding the sampled rows.

    Raises
    ------
    ValueError
        If both or neither of ``n``/``frac`` are given, if ``method`` is not
        supported, if ``seed`` is not a non-negative integer, or if a
        percentage-only method is combined with ``n``.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> from pyduck_janitor.sampling import sample
    >>>
    >>> conn = duckdb.connect()
    >>> df = pd.DataFrame({'id': range(100)})
    >>> relation = conn.from_df(df)
    >>> sampled = sample(relation, n=10, seed=42, conn=conn)
    >>> len(sampled.df())
    10
    """
    if (n is None) == (frac is None):
        raise ValueError("Exactly one of 'n' or 'frac' must be provided")

    if method not in SAMPLING_METHODS:
        raise ValueError(
            f"Unsupported sampling method {method!r}. "
            f"Supported methods are: {', '.join(SAMPLING_METHODS)}"
        )

    seed = _validate_seed(seed)

    if n is not None:
        row_count = _validate_count(n)
        if method not in ROW_COUNT_METHODS:
            raise ValueError(
                f"Sampling method {method!r} does not support a discrete row count in "
                f"DuckDB 1.5.2; only {', '.join(ROW_COUNT_METHODS)} support 'n ROWS'. "
                "Use 'frac' with this method instead."
            )
        size_clause = f"{row_count} ROWS"
    else:
        fraction = _validate_fraction(frac)
        size_clause = _format_percentage(fraction)

    repeatable = f" REPEATABLE ({seed})" if seed is not None else ""

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    query = f"SELECT * FROM {table_name} USING SAMPLE {method} ({size_clause}){repeatable}"

    return conn.query(query)


def sample_rows(
    relation: duckdb.DuckDBPyRelation,
    n: int,
    *,
    seed: Optional[int] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Deterministically pick ``n`` rows in random order without a full sort.

    The DuckDB-native path (``USING SAMPLE reservoir (n ROWS)``) is used
    whenever it is available, because reservoir sampling avoids sorting the
    input. When a ``seed`` is supplied the ``REPEATABLE`` clause makes the
    result reproducible across calls.

    If the native path is unavailable for the running DuckDB build, the
    function falls back to a hash-based deterministic pick:
    ``ORDER BY hash(<all columns>) LIMIT n``. The fallback sorts hash values
    (not the payload) and is therefore more expensive, but it returns the
    first ``n`` rows in a deterministic random order.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    n : int
        Number of rows to pick. Must be a non-negative integer.
    seed : int, optional
        Non-negative seed used for reproducible sampling.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A new relation holding ``n`` rows (fewer only if the input is
        smaller than ``n``).

    Raises
    ------
    ValueError
        If ``n`` is not a non-negative integer, if ``seed`` is not a
        non-negative integer, or if the relation has no columns and the
        hash-based fallback is required.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> from pyduck_janitor.sampling import sample_rows
    >>>
    >>> conn = duckdb.connect()
    >>> df = pd.DataFrame({'id': range(100)})
    >>> relation = conn.from_df(df)
    >>> picked = sample_rows(relation, 5, seed=7, conn=conn)
    >>> len(picked.df())
    5
    """
    row_count = _validate_count(n)
    seed = _validate_seed(seed)

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    repeatable = f" REPEATABLE ({seed})" if seed is not None else ""
    native_query = (
        f"SELECT * FROM {table_name} USING SAMPLE reservoir ({row_count} ROWS){repeatable}"
    )

    try:
        return conn.query(native_query)
    except duckdb.Error:
        # Fallback: deterministic pick based on hashed row values.
        columns = relation.columns
        if not columns:
            raise ValueError(
                "Cannot sample rows from a relation without columns using the hash-based fallback."
            )
        hash_expr = "hash(" + ", ".join(_quote_id(column) for column in columns) + ")"
        fallback_query = f"SELECT * FROM {table_name} ORDER BY {hash_expr} LIMIT {row_count}"
        return conn.query(fallback_query)


def approx_distinct(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    *,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Approximate distinct count of a column (``approx_count_distinct``).

    The result is a single-row relation with one column named
    ``approx_distinct``; read the scalar with ``result.fetchone()[0]``.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to count distinct values for.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Single-row relation with the approximate distinct count.

    Raises
    ------
    ValueError
        If ``column`` is not present in the relation.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> from pyduck_janitor.sampling import approx_distinct
    >>>
    >>> conn = duckdb.connect()
    >>> df = pd.DataFrame({'id': [1, 1, 2, 3]})
    >>> relation = conn.from_df(df)
    >>> approx_distinct(relation, 'id', conn=conn).fetchone()[0]
    3
    """
    _ensure_columns_exist(relation.columns, [column])
    table_name = _register_relation(conn, relation)
    conn = _require_conn(conn)
    table_name = _register_relation(conn, relation)
    query = (
        f"SELECT approx_count_distinct({_quote_id(column)}) AS "
        f"{_quote_id('approx_distinct')} FROM {table_name}"
    )
    return conn.query(query)


def approx_quantile(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    quantile: float,
    *,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Approximate quantile of a column (``approx_quantile``).

    The result is a single-row relation with one column named
    ``approx_quantile``; read the scalar with ``result.fetchone()[0]``.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column to compute the quantile for.
    quantile : float
        Quantile to compute, between 0 and 1 inclusive.
    conn : duckdb.DuckDBPyConnection, optional
        Connection that owns the relation.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Single-row relation with the approximate quantile.

    Raises
    ------
    ValueError
        If ``column`` is not present in the relation, if ``quantile`` is
        outside ``[0, 1]``, or if DuckDB cannot compute the quantile for
        the column type.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> from pyduck_janitor.sampling import approx_quantile
    >>>
    >>> conn = duckdb.connect()
    >>> df = pd.DataFrame({'v': [1, 2, 3, 4, 5]})
    >>> relation = conn.from_df(df)
    >>> approx_quantile(relation, 'v', 0.5, conn=conn).fetchone()[0]
    3
    """
    _ensure_columns_exist(relation.columns, [column])
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise ValueError("quantile must be a number between 0 and 1")
    quantile = float(quantile)
    if quantile < 0 or quantile > 1:
        raise ValueError("quantile must be a number between 0 and 1")

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    query = (
        f"SELECT approx_quantile({_quote_id(column)}, CAST({quantile!r} AS FLOAT)) AS "
        f"{_quote_id('approx_quantile')} FROM {table_name}"
    )
    try:
        return conn.query(query)
    except duckdb.Error as exc:
        raise ValueError(f"Could not compute approx_quantile for column {column!r}: {exc}") from exc


def register_python_function(
    relation: duckdb.DuckDBPyRelation,
    name: str,
    func: Callable[..., Any],
    *,
    params: Optional[Sequence[str]] = None,
    return_type: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Register a Python callable as a DuckDB scalar UDF on a connection.

    The UDF is registered on the *connection*, not on the data, so the
    relation is returned **unchanged**. That makes the call safe to use
    mid-chain::

        relation = register_python_function(relation, 'my_upper', my_upper, conn=conn)
        relation = apply_python_function(relation, 'name', my_upper, conn=conn)

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation. It is returned unchanged.
    name : str
        SQL name of the UDF.
    func : callable
        Python callable invoked by DuckDB. It receives one argument per
        declared parameter type.
    params : sequence of str, optional
        DuckDB type strings for the parameters, e.g. ``['VARCHAR']``. When
        omitted, the types are inferred from the function's annotations and
        fall back to ``['VARCHAR']`` when no annotation is available.
    return_type : str, optional
        DuckDB type string of the return value, e.g. ``'VARCHAR'``,
        ``'BIGINT'``, ``'DOUBLE'``. When omitted it is inferred from the
        function's return annotation; if it cannot be inferred a
        ``ValueError`` explains how to supply it explicitly.
    conn : duckdb.DuckDBPyConnection, optional
        Connection on which the UDF is registered.

    Returns
    -------
    duckdb.DuckDBPyRelation
        The *same* relation that was passed in, unchanged.

    Raises
    ------
    ValueError
        If no connection is supplied, if ``name`` is empty, if ``func`` is
        not callable, if the parameter/return types cannot be determined,
        or if DuckDB rejects the registration.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> from pyduck_janitor.sampling import register_python_function
    >>>
    >>> conn = duckdb.connect()
    >>> df = pd.DataFrame({'name': ['alice']})
    >>> relation = conn.from_df(df)
    >>> _ = register_python_function(
    ...     relation, 'my_upper', lambda s: s.upper(),
    ...     params=['VARCHAR'], return_type='VARCHAR', conn=conn,
    ... )
    >>> conn.execute("SELECT my_upper('bob')").fetchone()[0]
    'BOB'
    """
    if conn is None:
        raise ValueError(
            "A DuckDB connection is required. Pass the connection that owns the relation."
        )
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if not callable(func):
        raise TypeError("func must be callable")

    if params is None:
        param_types = _infer_param_types(func) or ["VARCHAR"]
    else:
        param_types = [str(param) for param in params]
        if not param_types:
            raise ValueError("params must contain at least one DuckDB type string")

    resolved_return_type = return_type if return_type is not None else _infer_return_type(func)

    _create_udf(conn, name, func, param_types, resolved_return_type)

    return relation


def apply_python_function(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    func: Callable[..., Any],
    *,
    name: Optional[str] = None,
    params: Optional[Sequence[str]] = None,
    return_type: Optional[str] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> duckdb.DuckDBPyRelation:
    """
    Register a Python callable and apply it to a column.

    The callable is registered on the connection under a generated name (or
    the caller-supplied ``name``) and evaluated for every row of ``column``.
    The result is appended as a new column named ``<column>_py``.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The input relation.
    column : str
        Column whose values are passed to ``func``.
    func : callable
        Python callable invoked by DuckDB.
    name : str, optional
        SQL name for the UDF. A unique name is generated when omitted.
    params : sequence of str, optional
        DuckDB type strings for the parameters. When omitted the types are
        inferred from the function's annotations, then from the type of
        ``column``, and finally default to ``['VARCHAR']``.
    return_type : str, optional
        DuckDB type string of the return value. When omitted it is inferred
        from the function's return annotation.
    conn : duckdb.DuckDBPyConnection, optional
        Connection on which the UDF is registered. Required.

    Returns
    -------
    duckdb.DuckDBPyRelation
        A new relation with the additional ``<column>_py`` column.

    Raises
    ------
    ValueError
        If no connection is supplied, if ``column`` is unknown, or if the
        UDF cannot be registered.

    Examples
    --------
    >>> import duckdb
    >>> import pandas as pd
    >>> from pyduck_janitor.sampling import apply_python_function
    >>>
    >>> conn = duckdb.connect()
    >>> df = pd.DataFrame({'name': ['alice', 'bob']})
    >>> relation = conn.from_df(df)
    >>> result = apply_python_function(
    ...     relation, 'name', lambda s: s.upper(),
    ...     params=['VARCHAR'], return_type='VARCHAR', conn=conn,
    ... )
    >>> result.df()['name_py'].tolist()
    ['ALICE', 'BOB']
    """
    if conn is None:
        raise ValueError("A DuckDB connection is required to register a Python function.")
    _ensure_columns_exist(relation.columns, [column])
    if not callable(func):
        raise TypeError("func must be callable")

    if name is None:
        safe_column = re.sub(r"[^A-Za-z0-9_]", "_", str(column))
        name = f"_pyfn_{safe_column}_{uuid.uuid4().hex[:12]}"
    elif not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    if params is None:
        param_types = _infer_param_types(func)
        if param_types is None:
            types = list(relation.types)
            column_index = list(relation.columns).index(column)
            param_types = [str(types[column_index])]
    else:
        param_types = [str(param) for param in params]
        if not param_types:
            raise ValueError("params must contain at least one DuckDB type string")

    register_python_function(
        relation,
        name,
        func,
        params=param_types,
        return_type=return_type,
        conn=conn,
    )

    table_name = _register_relation(conn, relation)

    conn = _require_conn(conn)

    table_name = _register_relation(conn, relation)
    target_column = f"{column}_py"
    query = (
        f"SELECT *, {_quote_id(name)}({_quote_id(column)}) AS {_quote_id(target_column)} "
        f"FROM {table_name}"
    )
    return conn.query(query)
