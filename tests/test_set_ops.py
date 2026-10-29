"""
Tests for the DuckDB-native set operations in :mod:`pyduck_janitor.set_ops`.

Covers ``union`` (positional and ``BY NAME``), ``intersect``, ``except_`` /
``except_rows``, ``concat_rows`` and ``select_except``.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pyduck_janitor.set_ops import (
    concat_rows,
    except_,
    except_rows,
    intersect,
    select_except,
    union,
)


@pytest.fixture
def conn():
    """Return a fresh in-memory DuckDB connection."""
    return duckdb.connect()


def _rel(conn, data):
    """Build a relation from a dict of columns."""
    return conn.from_df(pd.DataFrame(data))


def _sorted_rows(relation):
    """Return the relation rows as a sorted list of tuples (NULLs sort last)."""

    def key(row):
        return tuple((1, "") if value is None else (0, str(value)) for value in row)

    return sorted((tuple(row) for row in relation.fetchall()), key=key)


# ---------------------------------------------------------------------------
# union
# ---------------------------------------------------------------------------


class TestUnion:
    """Tests for :func:`union`."""

    def test_union_all_is_the_default(self, conn):
        """``all=True`` keeps duplicate rows."""
        left = _rel(conn, {"a": [1, 2]})
        right = _rel(conn, {"a": [2, 3]})
        result = union(left, right, all=True, conn=conn)
        assert _sorted_rows(result) == [(1,), (2,), (2,), (3,)]

    def test_union_distinct(self, conn):
        """``all=False`` (the default) removes duplicate rows."""
        left = _rel(conn, {"a": [1, 2]})
        right = _rel(conn, {"a": [2, 3]})
        result = union(left, right, conn=conn)
        assert _sorted_rows(result) == [(1,), (2,), (3,)]

    def test_union_is_positional_by_default(self, conn):
        """Positional union matches column by position, not by name."""
        left = _rel(conn, {"x": [1], "y": ["a"]})
        right = _rel(conn, {"x": [2], "z": ["b"]})
        result = union(left, right, conn=conn)
        assert result.columns == ["x", "y"]
        assert _sorted_rows(result) == [(1, "a"), (2, "b")]

    def test_union_by_name_with_different_column_order(self, conn):
        """``by_name=True`` matches columns by name regardless of order."""
        left = _rel(conn, {"a": [1, 2], "b": ["x", "y"]})
        right = _rel(conn, {"b": ["z"], "a": [3]})
        result = union(left, right, by_name=True, conn=conn)
        assert result.columns == ["a", "b"]
        assert _sorted_rows(result) == [(1, "x"), (2, "y"), (3, "z")]

    def test_union_by_name_with_extra_column_becomes_null(self, conn):
        """A column present on only one side is filled with NULL."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [2], "extra": ["only-right"]})
        result = union(left, right, by_name=True, conn=conn)
        assert result.columns == ["a", "extra"]
        rows = {row[0]: row[1] for row in result.fetchall()}
        assert rows[1] is None
        assert rows[2] == "only-right"

    def test_union_by_name_all_is_supported(self, conn):
        """DuckDB 1.5.2 supports UNION ALL BY NAME -- duplicates survive."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [1], "b": [2]})
        result = union(left, right, by_name=True, all=True, conn=conn)
        assert result.columns == ["a", "b"]
        assert _sorted_rows(result) == [(1, 2), (1, None)]

    def test_union_by_name_all_false_removes_duplicates(self, conn):
        """``by_name=True, all=False`` yields distinct rows."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [1]})
        result = union(left, right, by_name=True, conn=conn)
        assert _sorted_rows(result) == [(1,)]

    def test_union_mismatched_column_counts_raises_value_error(self, conn):
        """A column-count mismatch is a ValueError, not a raw duckdb error."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [2], "b": [3]})
        with pytest.raises(ValueError, match="same number of columns"):
            union(left, right, conn=conn)

    def test_union_by_name_allows_different_column_counts(self, conn):
        """``by_name=True`` sidesteps the column-count restriction."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [2], "b": [3]})
        result = union(left, right, by_name=True, conn=conn)
        assert result.columns == ["a", "b"]

    def test_union_requires_a_connection(self, conn):
        """A connection must be supplied (or inferable) for raw relations."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [2]})
        with pytest.raises(ValueError, match="connection is required"):
            union(left, right)

    def test_union_rejects_relations_from_different_connections(self):
        """Relations on different connections raise a clear ValueError."""
        left_conn = duckdb.connect()
        right_conn = duckdb.connect()
        left = left_conn.from_df(pd.DataFrame({"a": [1]}))
        right = right_conn.from_df(pd.DataFrame({"a": [2]}))
        with pytest.raises(ValueError, match="different DuckDB connection"):
            union(left, right, conn=left_conn)

    def test_union_does_not_mutate_inputs(self, conn):
        """The input relations are untouched."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [2]})
        union(left, right, conn=conn)
        assert left.columns == ["a"]
        assert left.fetchall() == [(1,)]
        assert right.fetchall() == [(2,)]


# ---------------------------------------------------------------------------
# intersect
# ---------------------------------------------------------------------------


class TestIntersect:
    """Tests for :func:`intersect`."""

    def test_intersect_returns_shared_rows(self, conn):
        """Only rows present on both sides survive, without duplicates."""
        left = _rel(conn, {"a": [1, 2, 2]})
        right = _rel(conn, {"a": [2, 3]})
        result = intersect(left, right, conn=conn)
        assert _sorted_rows(result) == [(2,)]

    def test_intersect_multiple_columns(self, conn):
        """Intersection compares whole rows."""
        left = _rel(conn, {"a": [1, 2], "b": ["x", "y"]})
        right = _rel(conn, {"a": [2], "b": ["y"]})
        result = intersect(left, right, conn=conn)
        assert _sorted_rows(result) == [(2, "y")]

    def test_intersect_by_name_matches_by_column_name(self, conn):
        """``by_name=True`` matches columns by name, not position."""
        left = _rel(conn, {"a": [1], "b": ["x"]})
        right = _rel(conn, {"b": ["x"], "a": [1]})
        result = intersect(left, right, by_name=True, conn=conn)
        assert result.columns == ["a", "b"]
        assert _sorted_rows(result) == [(1, "x")]

    def test_intersect_by_name_with_different_columns(self, conn):
        """Padded by-name intersection compares the full padded rows.

        The left row is ``(1, 'x', NULL)`` and the right row is
        ``(1, NULL, 'extra')``; they differ, so nothing is shared.
        """
        left = _rel(conn, {"a": [1], "b": ["x"]})
        right = _rel(conn, {"a": [1], "c": ["extra"]})
        result = intersect(left, right, by_name=True, conn=conn)
        assert result.columns == ["a", "b", "c"]
        assert _sorted_rows(result) == []

    def test_intersect_by_name_matches_on_padded_shared_rows(self, conn):
        """Rows that are identical after padding do intersect."""
        left = _rel(conn, {"a": [1], "b": ["x"]})
        right = _rel(conn, {"a": [1], "b": ["x"], "c": [None]})
        result = intersect(left, right, by_name=True, conn=conn)
        assert result.columns == ["a", "b", "c"]
        assert _sorted_rows(result) == [(1, "x", None)]

    def test_intersect_mismatched_column_counts_raises_value_error(self, conn):
        """A column-count mismatch is a ValueError."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="same number of columns"):
            intersect(left, right, conn=conn)

    def test_intersect_empty_result(self, conn):
        """Disjoint relations produce an empty result."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [2]})
        assert intersect(left, right, conn=conn).fetchall() == []


# ---------------------------------------------------------------------------
# except_
# ---------------------------------------------------------------------------


class TestExcept:
    """Tests for :func:`except_` and its ``except_rows`` alias."""

    def test_except_returns_left_only_rows(self, conn):
        """Rows of the left relation that are absent from the right survive."""
        left = _rel(conn, {"a": [1, 2, 2]})
        right = _rel(conn, {"a": [2]})
        result = except_(left, right, conn=conn)
        assert _sorted_rows(result) == [(1,)]

    def test_except_rows_alias_is_the_same_function(self):
        """``except_rows`` is an alias of ``except_``."""
        assert except_rows is except_

    def test_except_by_name_matches_by_column_name(self, conn):
        """``by_name=True`` matches columns by name, not position."""
        left = _rel(conn, {"a": [1], "b": ["x"]})
        right = _rel(conn, {"b": ["x"], "a": [1]})
        result = except_(left, right, by_name=True, conn=conn)
        assert result.fetchall() == []

    def test_except_by_name_with_different_columns(self, conn):
        """Padded by-name except keeps the left column order."""
        left = _rel(conn, {"a": [1], "b": ["x"]})
        right = _rel(conn, {"a": [1], "c": ["extra"]})
        result = except_(left, right, by_name=True, conn=conn)
        assert result.columns == ["a", "b", "c"]
        assert _sorted_rows(result) == [(1, "x", None)]

    def test_except_mismatched_column_counts_raises_value_error(self, conn):
        """A column-count mismatch is a ValueError."""
        left = _rel(conn, {"a": [1]})
        right = _rel(conn, {"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="same number of columns"):
            except_(left, right, conn=conn)

    def test_except_with_identical_relations_is_empty(self, conn):
        """Everything in the right relation removes everything on the left."""
        left = _rel(conn, {"a": [1, 2]})
        right = _rel(conn, {"a": [1, 2]})
        assert except_(left, right, conn=conn).fetchall() == []


# ---------------------------------------------------------------------------
# concat_rows
# ---------------------------------------------------------------------------


class TestConcatRows:
    """Tests for :func:`concat_rows`."""

    def test_concat_rows_with_three_relations(self, conn):
        """Three relations are stacked into one."""
        first = _rel(conn, {"a": [1]})
        second = _rel(conn, {"a": [2]})
        third = _rel(conn, {"a": [3]})
        result = concat_rows(first, [second, third], conn=conn)
        assert result.columns == ["a"]
        assert _sorted_rows(result) == [(1,), (2,), (3,)]

    def test_concat_rows_keeps_duplicates(self, conn):
        """``concat_rows`` is a UNION ALL -- duplicates are preserved."""
        first = _rel(conn, {"a": [1]})
        second = _rel(conn, {"a": [1]})
        result = concat_rows(first, [second], conn=conn)
        assert _sorted_rows(result) == [(1,), (1,)]

    def test_concat_rows_by_name_with_different_columns(self, conn):
        """By default the relations may have different column sets."""
        first = _rel(conn, {"a": [1]})
        second = _rel(conn, {"b": [2]})
        result = concat_rows(first, [second], conn=conn)
        assert result.columns == ["a", "b"]
        assert _sorted_rows(result) == [(1, None), (None, 2)]

    def test_concat_rows_positional_requires_matching_counts(self, conn):
        """``by_name=False`` requires identical column counts."""
        first = _rel(conn, {"a": [1]})
        second = _rel(conn, {"a": [2], "b": [3]})
        with pytest.raises(ValueError, match="same number of columns"):
            concat_rows(first, [second], by_name=False, conn=conn)

    def test_concat_rows_rejects_empty_others(self, conn):
        """An empty list is rejected with a clear message."""
        first = _rel(conn, {"a": [1]})
        with pytest.raises(ValueError, match="at least one relation"):
            concat_rows(first, [], conn=conn)

    def test_concat_rows_rejects_non_list_others(self, conn):
        """``others`` must be a list."""
        first = _rel(conn, {"a": [1]})
        with pytest.raises(ValueError, match="must be a list"):
            concat_rows(first, first, conn=conn)

    def test_concat_rows_rejects_non_relation_entries(self, conn):
        """Entries that are not relations raise a clear ValueError."""
        first = _rel(conn, {"a": [1]})
        with pytest.raises(ValueError, match="must be a duckdb.DuckDBPyRelation"):
            concat_rows(first, ["not-a-relation"], conn=conn)

    def test_concat_rows_rejects_different_connections(self):
        """All relations must live on the same connection."""
        left_conn = duckdb.connect()
        right_conn = duckdb.connect()
        first = left_conn.from_df(pd.DataFrame({"a": [1]}))
        second = right_conn.from_df(pd.DataFrame({"a": [2]}))
        with pytest.raises(ValueError, match="different DuckDB connection"):
            concat_rows(first, [second], conn=left_conn)


# ---------------------------------------------------------------------------
# select_except
# ---------------------------------------------------------------------------


class TestSelectExcept:
    """Tests for :func:`select_except`."""

    def test_select_except_single_column(self, conn):
        """A single column name may be passed as a string."""
        rel = _rel(conn, {"a": [1], "b": [2], "c": [3]})
        result = select_except(rel, "b", conn=conn)
        assert result.columns == ["a", "c"]
        assert _sorted_rows(result) == [(1, 3)]

    def test_select_except_multiple_columns(self, conn):
        """Several columns may be excluded at once."""
        rel = _rel(conn, {"a": [1], "b": [2], "c": [3]})
        result = select_except(rel, ["b", "c"], conn=conn)
        assert result.columns == ["a"]
        assert _sorted_rows(result) == [(1,)]

    def test_select_except_with_replace(self, conn):
        """``replace`` rewrites a column in place."""
        rel = _rel(conn, {"a": [1, 2], "b": [10, 20]})
        result = select_except(rel, [], replace={"a": "a * 10"}, conn=conn)
        assert result.columns == ["a", "b"]
        assert _sorted_rows(result) == [(10, 10), (20, 20)]

    def test_select_except_with_replace_and_exclude(self, conn):
        """Exclude and replace combine in one star expression."""
        rel = _rel(conn, {"a": [1], "b": [2], "c": [3]})
        result = select_except(rel, "c", replace={"a": "a + 100"}, conn=conn)
        assert result.columns == ["a", "b"]
        assert _sorted_rows(result) == [(101, 2)]

    def test_select_except_unknown_column_raises_value_error(self, conn):
        """Excluding a missing column raises a ValueError."""
        rel = _rel(conn, {"a": [1]})
        with pytest.raises(ValueError, match="Unknown column"):
            select_except(rel, "nope", conn=conn)

    def test_select_except_replace_unknown_column_raises_value_error(self, conn):
        """Replacing a missing column raises a ValueError."""
        rel = _rel(conn, {"a": [1]})
        with pytest.raises(ValueError, match="Unknown column"):
            select_except(rel, [], replace={"nope": "1"}, conn=conn)

    def test_select_except_cannot_exclude_and_replace_same_column(self, conn):
        """A column may not be in both lists."""
        rel = _rel(conn, {"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="both the exclude list and the replace"):
            select_except(rel, "a", replace={"a": "a + 1"}, conn=conn)

    def test_select_except_cannot_exclude_every_column(self, conn):
        """At least one column must remain."""
        rel = _rel(conn, {"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Cannot exclude every column"):
            select_except(rel, ["a", "b"], conn=conn)

    def test_select_except_rejects_non_string_columns(self, conn):
        """Column names must be strings."""
        rel = _rel(conn, {"a": [1]})
        with pytest.raises(ValueError, match="only column names"):
            select_except(rel, [1], conn=conn)

    def test_select_except_rejects_destructive_replace_expression(self, conn):
        """A destructive SQL fragment in ``replace`` is rejected."""
        rel = _rel(conn, {"a": [1]})
        with pytest.raises(ValueError, match="disallowed SQL statement"):
            select_except(rel, [], replace={"a": "DROP TABLE foo"}, conn=conn)

    def test_select_except_preserves_column_order(self, conn):
        """The remaining columns keep their original order."""
        rel = _rel(conn, {"z": [1], "a": [2], "m": [3]})
        result = select_except(rel, "a", conn=conn)
        assert result.columns == ["z", "m"]


# ---------------------------------------------------------------------------
# DuckJanitor-like inputs and NULL semantics
# ---------------------------------------------------------------------------


class TestDuckJanitorLikeInputs:
    """The verbs also accept DuckJanitor-like wrappers."""

    def test_union_accepts_duckjanitor_objects(self):
        """A DuckJanitor exposes ``relation`` and ``connection``."""
        from pyduck_janitor import DuckJanitor

        left = DuckJanitor.from_pandas(pd.DataFrame({"a": [1]}))
        right = DuckJanitor.from_pandas(pd.DataFrame({"a": [2]}))
        result = union(left, right)
        assert _sorted_rows(result) == [(1,), (2,)]

    def test_concat_rows_accepts_duckjanitor_objects(self):
        """DuckJanitor objects may be stacked."""
        from pyduck_janitor import DuckJanitor

        first = DuckJanitor.from_pandas(pd.DataFrame({"a": [1]}))
        second = DuckJanitor.from_pandas(pd.DataFrame({"a": [2]}))
        result = concat_rows(first, [second])
        assert _sorted_rows(result) == [(1,), (2,)]

    def test_select_except_accepts_duckjanitor_object(self):
        """The connection is inferred from the DuckJanitor wrapper."""
        from pyduck_janitor import DuckJanitor

        dj = DuckJanitor.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
        result = select_except(dj, "b")
        assert result.columns == ["a"]


class TestNullSemantics:
    """DuckDB set operations treat NULL as comparable."""

    def test_intersect_matches_null_with_null(self, conn):
        """``NULL`` matches ``NULL`` inside INTERSECT."""
        left = _rel(conn, {"a": [None, 1]})
        right = _rel(conn, {"a": [None]})
        result = intersect(left, right, conn=conn)
        assert result.fetchall() == [(None,)]

    def test_except_drops_null_when_present_on_the_right(self, conn):
        """A NULL row on the right removes the NULL row on the left."""
        left = _rel(conn, {"a": [None, 1]})
        right = _rel(conn, {"a": [None]})
        result = except_(left, right, conn=conn)
        assert result.fetchall() == [(1,)]

    def test_union_by_name_fills_missing_column_with_null(self, conn):
        """The unmatched column is NULL for the rows of the other side."""
        left = _rel(conn, {"a": [1], "b": ["keep"]})
        right = _rel(conn, {"a": [2]})
        result = union(left, right, by_name=True, conn=conn)
        rows = {row[0]: row[1] for row in result.fetchall()}
        assert rows[1] == "keep"
        assert rows[2] is None
