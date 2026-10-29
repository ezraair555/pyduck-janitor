"""Tests for pyduck_janitor.nested_ops (DuckDB-native nested/typed-data verbs)."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pyduck_janitor.nested_ops import (
    from_list,
    from_struct,
    json_extract,
    list_filter,
    list_transform,
    list_transforms,
    to_list,
    to_struct,
    unnest,
)


def _lists(series):
    """Return a LIST column as a list of plain python lists."""
    return [list(item) if item is not None else None for item in series]


@pytest.fixture
def conn():
    """Fresh in-memory DuckDB connection."""
    return duckdb.connect()


@pytest.fixture
def struct_rel(conn):
    """Relation with a struct column."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "addr": [{"city": "NYC", "zip": 10001}, {"city": "LA", "zip": 90001}],
        }
    )
    return conn.from_df(df)


@pytest.fixture
def list_rel(conn):
    """Relation with a list column."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "vals": [[1, 2], [3], [4, 5, 6]],
            "words": [["x", "y"], ["z"], ["p", "q"]],
        }
    )
    return conn.from_df(df)


# ---------------------------------------------------------------------------
# to_struct
# ---------------------------------------------------------------------------


def test_to_struct_happy_path(conn):
    """to_struct packs columns into a STRUCT column."""
    rel = conn.from_df(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    out = to_struct(rel, ["a", "b"], "packed", conn=conn)
    assert out.columns == ["a", "b", "packed"]
    row = out.df().iloc[0]
    assert row["packed"]["a"] == 1
    assert row["packed"]["b"] == "x"


def test_to_struct_default_name(conn):
    """to_struct defaults the output column name to 'struct'."""
    rel = conn.from_df(pd.DataFrame({"a": [1], "b": [2]}))
    out = to_struct(rel, ["a", "b"], conn=conn)
    assert "struct" in out.columns


def test_to_struct_single_column(conn):
    """to_struct accepts a bare column name string."""
    rel = conn.from_df(pd.DataFrame({"a": [1, 2]}))
    out = to_struct(rel, "a", "one", conn=conn)
    assert out.columns == ["a", "one"]


def test_to_struct_drop_source(conn):
    """to_struct(drop_source=True) removes the source columns."""
    rel = conn.from_df(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    out = to_struct(rel, ["a", "b"], "packed", drop_source=True, conn=conn)
    assert out.columns == ["packed"]


def test_to_struct_unknown_column_raises(conn):
    """to_struct raises ValueError for unknown columns."""
    rel = conn.from_df(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="Unknown column"):
        to_struct(rel, ["a", "nope"], conn=conn)


# ---------------------------------------------------------------------------
# from_struct
# ---------------------------------------------------------------------------


def test_from_struct_whole_struct(struct_rel, conn):
    """from_struct expands every key of the struct."""
    out = from_struct(struct_rel, "addr", conn=conn)
    assert out.columns == ["id", "addr", "city", "zip"]
    df = out.df()
    assert df["city"].tolist() == ["NYC", "LA"]
    assert df["zip"].tolist() == [10001, 90001]


def test_from_struct_selected_fields(struct_rel, conn):
    """from_struct can extract a subset of keys."""
    out = from_struct(struct_rel, "addr", ["city"], conn=conn)
    assert out.columns == ["id", "addr", "city"]


def test_from_struct_prefix(struct_rel, conn):
    """from_struct applies a prefix to the output names."""
    out = from_struct(struct_rel, "addr", ["city", "zip"], prefix="addr_", conn=conn)
    assert out.columns == ["id", "addr", "addr_city", "addr_zip"]


def test_from_struct_drop_source(struct_rel, conn):
    """from_struct(drop_source=True) removes the struct column."""
    out = from_struct(struct_rel, "addr", ["city"], drop_source=True, conn=conn)
    assert out.columns == ["id", "city"]


def test_from_struct_unknown_field_raises(struct_rel, conn):
    """from_struct raises ValueError for a key that does not exist."""
    with pytest.raises(ValueError, match="Unknown struct field"):
        from_struct(struct_rel, "addr", ["nope"], conn=conn)


def test_from_struct_unknown_column_raises(conn):
    """from_struct raises ValueError for an unknown column."""
    rel = conn.from_df(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="Unknown column"):
        from_struct(rel, "nope", conn=conn)


def test_from_struct_non_struct_raises(conn):
    """from_struct raises ValueError when the column is not a struct."""
    rel = conn.from_df(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="not a STRUCT column"):
        from_struct(rel, "a", conn=conn)


# ---------------------------------------------------------------------------
# to_list
# ---------------------------------------------------------------------------


def test_to_list_happy_path(conn):
    """to_list packs columns into a LIST column."""
    rel = conn.from_df(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    out = to_list(rel, ["a", "b"], "vals", conn=conn)
    assert out.columns == ["a", "b", "vals"]
    assert _lists(out.df()["vals"]) == [[1, 3], [2, 4]]


def test_to_list_default_name(conn):
    """to_list defaults the output column name to 'list'."""
    rel = conn.from_df(pd.DataFrame({"a": [1]}))
    out = to_list(rel, ["a"], conn=conn)
    assert "list" in out.columns


def test_to_list_drop_source(conn):
    """to_list(drop_source=True) removes the source columns."""
    rel = conn.from_df(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    out = to_list(rel, ["a", "b"], "vals", drop_source=True, conn=conn)
    assert out.columns == ["vals"]


def test_to_list_mixed_types_cast(conn):
    """to_list casts mixed source types to VARCHAR so the list can be built."""
    rel = conn.from_df(pd.DataFrame({"a": [1], "b": ["x"]}))
    out = to_list(rel, ["a", "b"], "vals", conn=conn)
    assert _lists(out.df()["vals"]) == [["1", "x"]]


def test_to_list_unknown_column_raises(conn):
    """to_list raises ValueError for unknown columns."""
    rel = conn.from_df(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="Unknown column"):
        to_list(rel, ["nope"], conn=conn)


# ---------------------------------------------------------------------------
# from_list
# ---------------------------------------------------------------------------


def test_from_list_happy_path(list_rel, conn):
    """from_list expands a list into rows."""
    out = from_list(list_rel, "vals", conn=conn)
    assert out.columns == ["id", "vals", "words", "vals_value"]
    df = out.df()
    assert len(df) == 6
    assert df["vals_value"].tolist() == [1, 2, 3, 4, 5, 6]


def test_from_list_with_index(list_rel, conn):
    """from_list(index=True) emits a 1-based position column."""
    out = from_list(list_rel, "vals", index=True, conn=conn)
    assert "pos" in out.columns
    df = out.df()
    assert df["pos"].tolist() == [1, 2, 1, 1, 2, 3]


def test_from_list_custom_index_name(list_rel, conn):
    """from_list honours a custom position column name."""
    out = from_list(list_rel, "vals", index=True, output_column="idx", conn=conn)
    assert "idx" in out.columns
    assert "pos" not in out.columns


def test_from_list_drop_source(list_rel, conn):
    """from_list(drop_source=True) removes the source list column."""
    out = from_list(list_rel, "vals", drop_source=True, conn=conn)
    assert "vals" not in out.columns
    assert "vals_value" in out.columns


def test_from_list_unknown_column_raises(conn):
    """from_list raises ValueError for unknown columns."""
    rel = conn.from_df(pd.DataFrame({"vals": [[1]]}))
    with pytest.raises(ValueError, match="Unknown column"):
        from_list(rel, "nope", conn=conn)


def test_from_list_non_list_raises(conn):
    """from_list raises ValueError when the column is not a list."""
    rel = conn.from_df(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="not a LIST column"):
        from_list(rel, "a", conn=conn)


# ---------------------------------------------------------------------------
# list_transform / list_filter / list_transforms
# ---------------------------------------------------------------------------


def test_list_transform_happy_path(list_rel, conn):
    """list_transform applies a lambda to a list column."""
    out = list_transform(list_rel, "vals", "x -> x * 2", conn=conn)
    df = out.df()
    assert _lists(df["vals"]) == [[2, 4], [6], [8, 10, 12]]


def test_list_transform_output_column(list_rel, conn):
    """list_transform can write to a new column."""
    out = list_transform(list_rel, "vals", "x -> x + 1", output_column="vals2", conn=conn)
    assert out.columns == ["id", "vals", "words", "vals2"]
    assert _lists(out.df()["vals2"]) == [[2, 3], [4], [5, 6, 7]]


def test_list_transform_rejects_multistatement(list_rel, conn):
    """list_transform rejects multi-statement SQL fragments."""
    with pytest.raises(ValueError):
        list_transform(list_rel, "vals", "x -> x; DROP TABLE foo", conn=conn)


def test_list_filter_happy_path(list_rel, conn):
    """list_filter applies a predicate lambda to a list column."""
    out = list_filter(list_rel, "vals", "x -> x > 3", conn=conn)
    assert _lists(out.df()["vals"]) == [[], [], [4, 5, 6]]


def test_list_filter_output_column(list_rel, conn):
    """list_filter can write to a new column."""
    out = list_filter(list_rel, "vals", "x -> x > 3", output_column="big", conn=conn)
    assert out.columns == ["id", "vals", "words", "big"]


def test_list_transforms_happy_path(list_rel, conn):
    """list_transforms applies a lambda that returns a list."""
    try:
        out = list_transforms(list_rel, "vals", "x -> [x, x * 2]", conn=conn)
    except NotImplementedError:
        pytest.skip("Installed DuckDB cannot transform a LIST into a LIST")
    df = out.df()
    transformed = _lists(df["vals"])
    assert len(transformed) == 3
    assert [list(pair) for pair in transformed[0]] == [[1, 2], [2, 4]]
    assert [list(pair) for pair in transformed[1]] == [[3, 6]]
    assert [list(pair) for pair in transformed[2]] == [[4, 8], [5, 10], [6, 12]]


def test_list_transforms_output_column(list_rel, conn):
    """list_transforms can write to a new column."""
    try:
        out = list_transforms(list_rel, "vals", "x -> [x, x * 2]", output_column="pairs", conn=conn)
    except NotImplementedError:
        pytest.skip("Installed DuckDB cannot transform a LIST into a LIST")
    assert "pairs" in out.columns


def test_list_transforms_unknown_column_raises(list_rel, conn):
    """list_transforms raises ValueError for unknown columns."""
    with pytest.raises(ValueError, match="Unknown column"):
        list_transforms(list_rel, "nope", "x -> [x]", conn=conn)


# ---------------------------------------------------------------------------
# json_extract
# ---------------------------------------------------------------------------


def test_json_extract_pointer(conn):
    """json_extract supports a JSON pointer path."""
    rel = conn.from_df(pd.DataFrame({"j": ['{"a": {"b": 5}}', '{"a": {"b": 6}}']}))
    out = json_extract(rel, "j", "$.a.b", conn=conn)
    assert out.columns == ["j", "b"]
    assert out.df()["b"].tolist() == ["5", "6"]


def test_json_extract_key_name(conn):
    """json_extract supports a plain key name."""
    rel = conn.from_df(pd.DataFrame({"j": ['{"a": {"b": 5}}']}))
    out = json_extract(rel, "j", "a", conn=conn)
    assert out.columns == ["j", "a"]
    assert out.df()["a"].tolist() == ['{"b":5}']


def test_json_extract_output_column(conn):
    """json_extract honours an explicit output column name."""
    rel = conn.from_df(pd.DataFrame({"j": ['{"a": 1}']}))
    out = json_extract(rel, "j", "$.a", output_column="value", conn=conn)
    assert out.columns == ["j", "value"]
    assert out.df()["value"].tolist() == ["1"]


def test_json_extract_missing_key_is_null(conn):
    """json_extract returns NULL for a missing key."""
    rel = conn.from_df(pd.DataFrame({"j": ['{"a": 1}']}))
    out = json_extract(rel, "j", "$.zzz", output_column="v", conn=conn)
    assert out.df()["v"].isna().all()


def test_json_extract_unknown_column_raises(conn):
    """json_extract raises ValueError for unknown columns."""
    rel = conn.from_df(pd.DataFrame({"j": ["{}"]}))
    with pytest.raises(ValueError, match="Unknown column"):
        json_extract(rel, "nope", "$.a", conn=conn)


# ---------------------------------------------------------------------------
# unnest
# ---------------------------------------------------------------------------


def test_unnest_one_column(list_rel, conn):
    """unnest expands a single list column."""
    out = unnest(list_rel, ["vals"], conn=conn)
    assert out.columns == ["id", "vals", "words", "vals_value"]
    df = out.df()
    assert len(df) == 6
    assert df["vals_value"].tolist() == [1, 2, 3, 4, 5, 6]


def test_unnest_two_columns(list_rel, conn):
    """unnest expands two list columns in parallel (zip) fashion."""
    out = unnest(list_rel, ["vals", "words"], conn=conn)
    df = out.df()
    assert "vals_value" in df.columns
    assert "words_value" in df.columns
    # zip semantics: rows are limited by the shortest list per row
    assert len(df) == 6
    assert df["vals_value"].tolist() == [1, 2, 3, 4, 5, 6]
    assert df["words_value"].tolist()[:5] == ["x", "y", "z", "p", "q"]
    assert pd.isna(df["words_value"].tolist()[5])


def test_unnest_drop_source(list_rel, conn):
    """unnest(drop_source=True) removes the source list columns."""
    out = unnest(list_rel, ["vals"], drop_source=True, conn=conn)
    assert "vals" not in out.columns
    assert "vals_value" in out.columns


def test_unnest_unknown_column_raises(conn):
    """unnest raises ValueError for unknown columns."""
    rel = conn.from_df(pd.DataFrame({"vals": [[1]]}))
    with pytest.raises(ValueError, match="Unknown column"):
        unnest(rel, ["nope"], conn=conn)


def test_unnest_non_list_raises(conn):
    """unnest raises ValueError when the column is not a list."""
    rel = conn.from_df(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="not a LIST column"):
        unnest(rel, ["a"], conn=conn)
