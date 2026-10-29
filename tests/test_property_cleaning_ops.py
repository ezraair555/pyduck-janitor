"""
Property-based tests for pyduck-janitor cleaning_ops.

Uses Hypothesis to verify:
1. Idempotency — applying an operation twice yields the same result as once.
2. No-op on already-clean data — clean input is unchanged.

This file was previously an orphaned, gitignored ``tests/property_cleaning_ops.py``
that pytest never collected (the name does not match the ``test_*.py`` pattern).
It is revived here under a collectable name.

Hypothesis is an optional dependency: if it is not installed the whole module
skips cleanly so CI stays green.

Run:  pytest tests/test_property_cleaning_ops.py -v
"""

from __future__ import annotations

import string

import duckdb
import pandas as pd
import pytest

# Optional dependency — skip the whole module when hypothesis is absent.
try:  # pragma: no cover - exercised only when the optional dep is absent
    import hypothesis  # noqa: F401

    pytest.importorskip("hypothesis")
except ImportError as _exc:  # pragma: no cover - optional dependency missing
    pytest.skip(
        f"hypothesis is not installed ({_exc}) — skipping property-based tests",
        allow_module_level=True,
    )

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from pyduck_janitor.cleaning_ops import (  # noqa: E402
    clean_names,
    dropna,
    remove_empty,
    rename_column,
    select_columns,
)

# ---------------------------------------------------------------------------
# Helpers & strategies
# ---------------------------------------------------------------------------


def _make_relation(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    return conn.from_df(df)


# Clean column names: lowercase, alphanumeric + underscore, no leading/trailing _
clean_col_names = st.text(
    alphabet=string.ascii_lowercase + string.digits + "_",
    min_size=1,
    max_size=20,
).filter(lambda s: not s.startswith("_") and not s.endswith("_") and s != "_" and "__" not in s)


# Dirty column names: can contain spaces, special chars, mixed case, leading/trailing _
dirty_col_names = st.text(
    alphabet=string.ascii_letters + string.digits + " _-!@#$%",
    min_size=1,
    max_size=30,
).filter(lambda s: any(c.isalpha() for c in s))


# Small dataframes for testing
@st.composite
def small_dataframe(draw, max_cols=5, max_rows=10):
    n_cols = draw(st.integers(min_value=2, max_value=max_cols))
    n_rows = draw(st.integers(min_value=1, max_value=max_rows))

    cols = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        values = draw(
            st.lists(
                st.one_of(
                    st.integers(min_value=-1000, max_value=1000),
                    st.text(alphabet=string.ascii_letters, max_size=10),
                    st.none(),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        cols[col_name] = values
    return pd.DataFrame(cols)


@st.composite
def dirty_named_dataframe(draw, max_cols=5, max_rows=10):
    """DataFrames with deliberately dirty column names."""
    n_cols = draw(st.integers(min_value=2, max_value=max_cols))
    n_rows = draw(st.integers(min_value=1, max_value=max_rows))

    # Generate dirty column names
    dirty_names = draw(
        st.lists(
            dirty_col_names,
            min_size=n_cols,
            max_size=n_cols,
            unique=True,
        )
    )

    cols = {}
    for name in dirty_names:
        values = draw(
            st.lists(
                st.one_of(
                    st.integers(min_value=-1000, max_value=1000),
                    st.text(alphabet=string.ascii_letters, max_size=10),
                    st.none(),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        cols[name] = values
    return pd.DataFrame(cols)


@st.composite
def clean_named_dataframe(draw, max_cols=5, max_rows=10):
    """DataFrames with already-clean column names (lowercase, snake_case)."""
    n_cols = draw(st.integers(min_value=2, max_value=max_cols))
    n_rows = draw(st.integers(min_value=1, max_value=max_rows))

    clean_names_list = draw(
        st.lists(
            clean_col_names,
            min_size=n_cols,
            max_size=n_cols,
            unique=True,
        )
    )

    cols = {}
    for name in clean_names_list:
        values = draw(
            st.lists(
                st.one_of(
                    st.integers(min_value=-1000, max_value=1000),
                    st.text(alphabet=string.ascii_letters, max_size=10),
                    st.none(),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        cols[name] = values
    return pd.DataFrame(cols)


@st.composite
def dataframe_with_nulls(draw, max_cols=5, max_rows=10):
    """DataFrames guaranteed to have some null values."""
    n_cols = draw(st.integers(min_value=2, max_value=max_cols))
    n_rows = draw(st.integers(min_value=2, max_value=max_rows))

    cols = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        # Mix of values and None
        values = draw(
            st.lists(
                st.one_of(
                    st.integers(min_value=-100, max_value=100),
                    st.none(),
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        cols[col_name] = values
    return pd.DataFrame(cols)


# Strategy: DataFrames with all-integer, no-null, non-empty values
@st.composite
def clean_no_null_dataframe(draw, max_cols=5, max_rows=10):
    """DataFrames with clean column names, no nulls, and non-empty string values."""
    n_cols = draw(st.integers(min_value=2, max_value=max_cols))
    n_rows = draw(st.integers(min_value=1, max_value=max_rows))

    clean_names_list = draw(
        st.lists(
            clean_col_names,
            min_size=n_cols,
            max_size=n_cols,
            unique=True,
        )
    )

    cols = {}
    for name in clean_names_list:
        # Use only positive integers to avoid any "empty" ambiguity
        values = draw(
            st.lists(
                st.integers(min_value=1, max_value=999),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        cols[name] = values
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    return duckdb.connect()


# ---------------------------------------------------------------------------
# Property: clean_names is idempotent
# ---------------------------------------------------------------------------


class TestCleanNamesIdempotent:
    """clean_names applied twice == clean_names applied once."""

    @given(df=dirty_named_dataframe())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_clean_names_idempotent(self, conn, df):
        rel = _make_relation(conn, df)
        once = clean_names(rel, conn=conn).df()
        twice = clean_names(clean_names(rel, conn=conn), conn=conn).df()

        assert list(once.columns) == list(twice.columns)
        pd.testing.assert_frame_equal(once, twice)

    @given(df=clean_named_dataframe())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_clean_names_noop_on_clean(self, conn, df):
        """clean_names on already-clean names is a no-op (columns unchanged)."""
        original_cols = list(df.columns)
        rel = _make_relation(conn, df)
        result = clean_names(rel, conn=conn).df()

        assert list(result.columns) == original_cols


# ---------------------------------------------------------------------------
# Property: dropna is idempotent
# ---------------------------------------------------------------------------


class TestDropnaIdempotent:
    """dropna applied twice == dropna applied once (no rows removed second time)."""

    @given(df=dataframe_with_nulls())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_dropna_idempotent(self, conn, df):
        rel = _make_relation(conn, df)
        once = dropna(rel, conn=conn)
        once_df = once.df()
        twice = dropna(once, conn=conn)
        twice_df = twice.df()

        # Second dropna should remove zero rows
        assert len(once_df) == len(twice_df)
        pd.testing.assert_frame_equal(once_df, twice_df)

    @given(df=clean_no_null_dataframe())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_dropna_noop_on_clean(self, conn, df):
        """dropna on data with no nulls is a no-op."""
        original_len = len(df)
        rel = _make_relation(conn, df)
        result = dropna(rel, conn=conn).df()

        assert len(result) == original_len


# ---------------------------------------------------------------------------
# Property: remove_empty is idempotent
# ---------------------------------------------------------------------------


class TestRemoveEmptyIdempotent:
    """remove_empty applied twice == remove_empty applied once."""

    @given(df=dataframe_with_nulls())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_remove_empty_idempotent(self, conn, df):
        rel = _make_relation(conn, df)
        try:
            once = remove_empty(rel, conn=conn)
            once_df = once.df()
        except ValueError:
            # All columns empty — expected for some generated cases
            return

        twice = remove_empty(once, conn=conn)
        twice_df = twice.df()

        assert list(once_df.columns) == list(twice_df.columns)
        assert len(once_df) == len(twice_df)
        pd.testing.assert_frame_equal(once_df, twice_df)

    @given(df=clean_no_null_dataframe())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_remove_empty_noop_on_clean(self, conn, df):
        """remove_empty on data with no empty rows/cols is a no-op."""
        # Data is all positive integers — no nulls, no empty strings
        original_cols = list(df.columns)
        original_len = len(df)
        rel = _make_relation(conn, df)
        result = remove_empty(rel, conn=conn).df()

        assert list(result.columns) == original_cols
        assert len(result) == original_len


# ---------------------------------------------------------------------------
# Property: select_columns is idempotent
# ---------------------------------------------------------------------------


class TestSelectColumnsIdempotent:
    """select_columns applied to its own output is a no-op."""

    @given(df=clean_named_dataframe())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_select_columns_idempotent(self, conn, df):
        cols = list(df.columns)
        rel = _make_relation(conn, df)
        once = select_columns(rel, cols, conn=conn)
        once_df = once.df()
        twice = select_columns(once, cols, conn=conn)
        twice_df = twice.df()

        pd.testing.assert_frame_equal(once_df, twice_df)


# ---------------------------------------------------------------------------
# Property: rename_column is idempotent when renaming to the same name
# ---------------------------------------------------------------------------


class TestRenameColumnIdempotent:
    """rename_column(col, col) preserves column names and row count."""

    @given(df=clean_no_null_dataframe())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_rename_to_self_is_noop(self, conn, df):
        col = df.columns[0]
        rel = _make_relation(conn, df)
        result = rename_column(rel, col, col, conn=conn).df()

        # Column names preserved
        assert list(result.columns) == list(df.columns)
        # Same number of rows
        assert len(result) == len(df)
        # Compare values (DuckDB round-trip may change dtypes, so compare by value)
        for c in df.columns:
            assert list(result[c]) == list(df[c]), f"Column '{c}' values differ"


# ---------------------------------------------------------------------------
# Property: clean_names output is always "clean"
# ---------------------------------------------------------------------------


class TestCleanNamesProducesCleanOutput:
    """After clean_names, column names match the clean pattern: [a-z0-9_]+."""

    @given(df=dirty_named_dataframe())
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    )
    def test_output_names_are_clean(self, conn, df):
        import re

        rel = _make_relation(conn, df)
        result = clean_names(rel, conn=conn).df()

        for col in result.columns:
            assert re.match(r"^[a-z0-9_]+$", col), f"Column '{col}' is not clean"
            assert not col.startswith("_"), f"Column '{col}' has leading underscore"
            assert not col.endswith("_"), f"Column '{col}' has trailing underscore"
