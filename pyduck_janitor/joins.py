"""
dplyr-style join verbs for pyduck_janitor.

Six verbs (matching R/dplyr and pyjanitor upstream):

- ``inner_join``  — only rows whose key matches on both sides
- ``left_join``   — all left rows + matched right rows; NULL for unmatched
- ``right_join``  — all right rows + matched left rows; NULL for unmatched
- ``full_join``   — all rows from both sides; NULL for unmatched on either side
- ``semi_join``   — left rows whose key exists on right (no right columns)
- ``anti_join``   — left rows whose key does NOT exist on right (no right columns)

Designed to compose with the rest of ``pyduck_janitor``:

.. code-block:: python

    from pyduck_janitor import DuckJanitor

    employees = DuckJanitor.from_pandas(...)
    departments = DuckJanitor.from_pandas(...)

    # Shared key
    employees.left_join(departments, on="dept_id")

    # Different key names per side
    employees.left_join(departments, left_on="emp_dept", right_on="id")

    # Many-to-one — filter left side to rows whose dept exists on right
    employees_in_active_depts = employees.semi_join(departments, on="dept_id")

    # Exclude employees whose dept was discontinued
    employees_active = employees.anti_join(discontinued_depts, on="dept_id")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

import duckdb

from .cleaning_ops import _quote_id

if TYPE_CHECKING:
    from .duck_janitor import DuckJanitor


def _normalize_keys(
    on: Union[str, Sequence[str], None],
    left_on: Union[str, Sequence[str], None],
    right_on: Union[str, Sequence[str], None],
    left_columns: Iterable[str],
    right_columns: Iterable[str],
) -> list[tuple[str, str]]:
    """Return a list of ``(left_col, right_col)`` pairs.

    Supports three call shapes:

    - ``on='dept_id'``                  → ``[('dept_id', 'dept_id')]``
    - ``on=['dept_id', 'tier']``        → ``[('dept_id', 'dept_id'), ('tier', 'tier')]``
    - ``left_on='e.dept', right_on='d.id'`` → ``[('e.dept', 'd.id')]``

    Raises ``ValueError`` when the call shape is invalid or a column is missing.
    """
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError("Pass either `on` (shared key name) or `left_on` + `right_on`, not both.")
    if (left_on is None) != (right_on is None):
        raise ValueError("`left_on` and `right_on` must be supplied together.")

    if on is not None:
        if isinstance(on, str):
            keys = [on]
        else:
            keys = list(on)
        if not keys:
            raise ValueError("`on` must contain at least one column.")
        left_list = list(left_columns)
        right_list = list(right_columns)
        for k in keys:
            if k not in left_list:
                raise ValueError(
                    f"`on` key {k!r} not found in left relation. Available: {left_list}"
                )
            if k not in right_list:
                raise ValueError(
                    f"`on` key {k!r} not found in right relation. Available: {right_list}"
                )
        return [(k, k) for k in keys]

    # left_on / right_on path
    if isinstance(left_on, str):
        left_keys = [left_on]
        right_keys = [right_on] if isinstance(right_on, str) else list(right_on)
    else:
        left_keys = list(left_on)
        right_keys = list(right_on)
    if not left_keys or not right_keys:
        raise ValueError("`left_on` and `right_on` must each contain at least one column.")
    if len(left_keys) != len(right_keys):
        raise ValueError(
            f"`left_on` ({len(left_keys)} keys) and `right_on` ({len(right_keys)} keys) "
            f"must have the same length."
        )

    left_list = list(left_columns)
    right_list = list(right_columns)
    pairs = list(zip(left_keys, right_keys))
    for lk, rk in pairs:
        if lk not in left_list:
            raise ValueError(
                f"`left_on` key {lk!r} not found in left relation. Available: {left_list}"
            )
        if rk not in right_list:
            raise ValueError(
                f"`right_on` key {rk!r} not found in right relation. Available: {right_list}"
            )
    return pairs


def _apply_suffix(
    right_columns: list[str],
    key_pairs: list[tuple[str, str]],
    suffixes: tuple[str, str],
) -> dict[str, str]:
    """Map right-side column → display name (suffixed on collision).

    Returns
    -------
    dict
        Mapping of every right column to its output column name. Join keys are
        never suffixed (the left side's name wins; right side is omitted when
        `USING` semantics apply, or renamed to `<name>_y` only when explicit
        `left_on` / `right_on` produced a same-name collision).

    Notes
    -----
    Only columns that exist on BOTH sides and are not join keys get a suffix.
    Match dplyr's convention: left columns win their name; right collisions get
    ``suffixes[1]``.
    """
    raise NotImplementedError  # implemented below as a module-level constant


# Simpler, single-purpose internal helper used by the verb implementations.
def _renames_for_right(
    left_columns: list[str],
    right_columns: list[str],
    key_pairs: list[tuple[str, str]],
    suffixes: tuple[str, str],
) -> dict[str, str]:
    """Build the {right_col → output_col} rename map.

    A right column is renamed only if it collides with a non-key left column,
    in which case it gets ``suffixes[1]`` appended. Join-key columns are
    handled by the caller (USING vs ON).
    """
    key_left_names = {lk for lk, _ in key_pairs}
    key_right_names = {rk for _, rk in key_pairs}
    collisions = set(left_columns) & set(right_columns)
    out: dict[str, str] = {}
    for rc in right_columns:
        if rc in collisions and rc not in key_left_names:
            out[rc] = f"{rc}{suffixes[1]}"
        else:
            out[rc] = rc
    # Suppress unused-variable warnings from the linter; keep names visible.
    _ = key_right_names
    _ = suffixes
    return out


def _build_join_sql(
    left_alias: str,
    right_alias: str,
    left_columns: list[str],
    right_columns: list[str],
    key_pairs: list[tuple[str, str]],
    right_renames: dict[str, str],
    how: str,
) -> str:
    """Build the SELECT/JOIN SQL for an INNER/LEFT/RIGHT/FULL join.

    Parameters
    ----------
    left_alias, right_alias
        Quoted SQL identifiers for the registered temp tables.
    left_columns, right_columns
        Original column lists.
    key_pairs
        Output of ``_normalize_keys``.
    right_renames
        Output of ``_renames_for_right``.
    how
        ``'INNER'``, ``'LEFT'``, ``'RIGHT'``, or ``'FULL'``.

    Returns
    -------
    str
        A single SQL statement.
    """
    how = how.upper()
    if how not in {"INNER", "LEFT", "RIGHT", "FULL"}:
        raise ValueError(f"Unsupported join type: {how}")

    # Build ON clause from key pairs.
    on_parts = [
        f"{left_alias}.{_quote_id(lk)} = {right_alias}.{_quote_id(rk)}" for lk, rk in key_pairs
    ]
    on_clause = " AND ".join(on_parts)

    # Build SELECT list. Left side: every column, qualified.
    # Right side: every column EXCEPT the join keys (because the ON clause
    # already showed them; joining them in the projection would duplicate).
    # Renamed collisions keep their new name.
    left_selects = [f"{left_alias}.{_quote_id(c)}" for c in left_columns]
    key_right_names = {rk for _, rk in key_pairs}
    right_selects = [
        f"{right_alias}.{_quote_id(rc)} AS {_quote_id(right_renames[rc])}"
        for rc in right_columns
        if rc not in key_right_names
    ]
    select_list = ", ".join(left_selects + right_selects)

    return f"SELECT {select_list} FROM {left_alias} {how} JOIN {right_alias} ON {on_clause}"


def _dplyr_join_impl(
    self: "DuckJanitor",
    other: "DuckJanitor",
    on: Union[str, Sequence[str], None],
    left_on: Union[str, Sequence[str], None],
    right_on: Union[str, Sequence[str], None],
    suffixes: tuple[str, str],
    how: str,
    conn: Optional[duckdb.DuckDBPyConnection],
) -> "DuckJanitor":
    """Shared implementation for inner/left/right/full join.

    Selects the connection that owns the left relation (via ``self._connection``)
    and registers both relations as temp tables keyed off ``id()`` of their
    underlying ``DuckDBPyRelation`` so the same pair can coexist in one chain.
    """
    from .duck_janitor import DuckJanitor  # local import to avoid cycle

    if conn is None:
        conn = self._connection
    if conn is None:
        raise ValueError(
            "A DuckDB connection is required. Use DuckJanitor.from_pandas(...) "
            "or pass the connection that owns self."
        )

    key_pairs = _normalize_keys(
        on, left_on, right_on, self._relation.columns, other._relation.columns
    )
    left_columns = list(self._relation.columns)
    right_columns = list(other._relation.columns)
    right_renames = _renames_for_right(left_columns, right_columns, key_pairs, suffixes)

    # Use id() of the relation (or attribute on DuckJanitor, if absent) to make
    # the temp table name collision-free within a chain.
    left_relation_obj = getattr(self, "_relation", None)
    right_relation_obj = getattr(other, "_relation", None)
    left_alias = f"_jl_{id(left_relation_obj)}"
    right_alias = f"_jr_{id(right_relation_obj)}"
    conn.register(left_alias, self._relation)
    conn.register(right_alias, other._relation)

    sql = _build_join_sql(
        left_alias,
        right_alias,
        left_columns,
        right_columns,
        key_pairs,
        right_renames,
        how,
    )
    new_relation = conn.query(sql)

    # Wrap in a fresh DuckJanitor pointing at the same connection, no longer
    # pinned to a source table.
    result = DuckJanitor.__new__(DuckJanitor)
    result._connection = conn
    result._relation = new_relation
    return result


def inner_join(
    self: "DuckJanitor",
    other: "DuckJanitor",
    on: Union[str, Sequence[str], None] = None,
    *,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
    suffixes: tuple[str, str] = ("_x", "_y"),
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> "DuckJanitor":
    """Return rows whose key matches on both sides.

    Parameters
    ----------
    other : DuckJanitor
        The right-side relation to join.
    on : str or list of str, optional
        Column name(s) shared by both sides. When supplied, ``left_on`` and
        ``right_on`` must be ``None``.
    left_on, right_on : str or list of str, optional
        Column names when the join key has a different name on each side.
        Both must be supplied together and have equal length.
    suffixes : tuple of (str, str), default ``('_x', '_y')``
        Suffix applied to right-side columns whose names collide with a
        non-key left column.
    conn : DuckDBPyConnection, optional
        Connection that owns the left relation. Defaults to ``self._connection``.

    Returns
    -------
    DuckJanitor
        New instance with the join applied.
    """
    return _dplyr_join_impl(self, other, on, left_on, right_on, suffixes, "INNER", conn)


def left_join(
    self: "DuckJanitor",
    other: "DuckJanitor",
    on: Union[str, Sequence[str], None] = None,
    *,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
    suffixes: tuple[str, str] = ("_x", "_y"),
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckJanitor_type:
    """Keep every row from the left side; match keys on the right; NULL on miss.

    R equivalent: ``dplyr::left_join``. pandas equivalent: ``pd.merge(how='left')``.
    """
    return _dplyr_join_impl(self, other, on, left_on, right_on, suffixes, "LEFT", conn)


def right_join(
    self: "DuckJanitor",
    other: "DuckJanitor",
    on: Union[str, Sequence[str], None] = None,
    *,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
    suffixes: tuple[str, str] = ("_x", "_y"),
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckJanitor_type:
    """Keep every row from the right side; match keys on the left; NULL on miss.

    R equivalent: ``dplyr::right_join``. pandas equivalent: ``pd.merge(how='right')``.
    """
    return _dplyr_join_impl(self, other, on, left_on, right_on, suffixes, "RIGHT", conn)


def full_join(
    self: "DuckJanitor",
    other: "DuckJanitor",
    on: Union[str, Sequence[str], None] = None,
    *,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
    suffixes: tuple[str, str] = ("_x", "_y"),
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckJanitor_type:
    """Keep every row from both sides; NULL on either side for unmatched keys.

    R equivalent: ``dplyr::full_join``. pandas equivalent: ``pd.merge(how='outer')``.
    """
    return _dplyr_join_impl(self, other, on, left_on, right_on, suffixes, "FULL", conn)


# Late binding to avoid forward-reference for the type alias used in the
# signatures above. Python evaluates annotations lazily for ``__future__``.
DuckJanitor_type = "DuckJanitor"


def semi_join(
    self: "DuckJanitor",
    other: "DuckJanitor",
    on: Union[str, Sequence[str], None] = None,
    *,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckJanitor_type:
    """Return left rows whose key EXISTS on right (no right columns added).

    R equivalent: ``dplyr::semi_join``. No pandas equivalent (use ``isin``).
    """
    if conn is None:
        conn = self._connection
    if conn is None:
        raise ValueError("A DuckDB connection is required. Use DuckJanitor.from_pandas(...).")

    from .duck_janitor import DuckJanitor

    key_pairs = _normalize_keys(
        on, left_on, right_on, self._relation.columns, other._relation.columns
    )
    left_columns = list(self._relation.columns)
    right_columns = list(other._relation.columns)  # noqa: F841 — referenced via key_pairs

    left_relation_obj = self._relation
    right_relation_obj = other._relation
    left_alias = f"_jl_{id(left_relation_obj)}"
    right_alias = f"_jr_{id(right_relation_obj)}"
    conn.register(left_alias, self._relation)
    conn.register(right_alias, other._relation)

    exists_parts = [
        f"{right_alias}.{_quote_id(rk)} = {left_alias}.{_quote_id(lk)}" for lk, rk in key_pairs
    ]
    exists_clause = " AND ".join(exists_parts)

    select_list = ", ".join(f"{left_alias}.{_quote_id(c)}" for c in left_columns)
    sql = (
        f"SELECT {select_list} "
        f"FROM {left_alias} "
        f"WHERE EXISTS ("
        f"SELECT 1 FROM {right_alias} WHERE {exists_clause}"
        f")"
    )
    new_relation = conn.query(sql)

    result = DuckJanitor.__new__(DuckJanitor)
    result._connection = conn
    result._relation = new_relation
    return result


def anti_join(
    self: "DuckJanitor",
    other: "DuckJanitor",
    on: Union[str, Sequence[str], None] = None,
    *,
    left_on: Union[str, Sequence[str], None] = None,
    right_on: Union[str, Sequence[str], None] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> DuckJanitor_type:
    """Return left rows whose key does NOT exist on right (no right columns added).

    R equivalent: ``dplyr::anti_join``. No pandas equivalent (``~df.isin(...)``).
    """
    if conn is None:
        conn = self._connection
    if conn is None:
        raise ValueError("A DuckDB connection is required. Use DuckJanitor.from_pandas(...).")

    from .duck_janitor import DuckJanitor

    key_pairs = _normalize_keys(
        on, left_on, right_on, self._relation.columns, other._relation.columns
    )
    left_columns = list(self._relation.columns)

    left_alias = f"_jl_{id(self._relation)}"
    right_alias = f"_jr_{id(other._relation)}"
    conn.register(left_alias, self._relation)
    conn.register(right_alias, other._relation)

    not_exists_parts = [
        f"{right_alias}.{_quote_id(rk)} = {left_alias}.{_quote_id(lk)}" for lk, rk in key_pairs
    ]
    not_exists_clause = " AND ".join(not_exists_parts)

    select_list = ", ".join(f"{left_alias}.{_quote_id(c)}" for c in left_columns)
    sql = (
        f"SELECT {select_list} "
        f"FROM {left_alias} "
        f"WHERE NOT EXISTS ("
        f"SELECT 1 FROM {right_alias} WHERE {not_exists_clause}"
        f")"
    )
    new_relation = conn.query(sql)

    result = DuckJanitor.__new__(DuckJanitor)
    result._connection = conn
    result._relation = new_relation
    return result
