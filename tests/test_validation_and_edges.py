"""Validation and edge-case tests for pyduck-janitor."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pyduck_janitor import DuckJanitor
from pyduck_janitor.cleaning_ops import filter_column, select_rows


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small fixture DataFrame."""
    return pd.DataFrame(
        {
            "name": ["alice", "bob", "charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "NYC"],
            "value": [1.0, None, 3.0],
        }
    )


@pytest.mark.parametrize(
    "criteria",
    [
        "age > 20; DROP TABLE users",
        "age > 20 -- inline comment",
        "age > 20 /* block comment */",
        "DELETE FROM t",
        "UPDATE t SET x = 1",
        "INSERT INTO t VALUES (1)",
        "DROP TABLE t",
        "ALTER TABLE t ADD COLUMN x INT",
    ],
)
def test_filter_on_rejects_destructive_sql(criteria: str, sample_df: pd.DataFrame) -> None:
    """filter_on should reject multi-statement and destructive SQL fragments."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="disallowed|cannot contain ';'|cannot contain SQL comments"):
        dj.filter_on(criteria).collect()


@pytest.mark.parametrize(
    "criteria",
    [
        "age > 30; select 1",
        "age > 30 -- x",
        "age > 30 /* x */",
        "create table x as select 1",
        "truncate table t",
        "call now()",
    ],
)
def test_filter_column_rejects_destructive_sql(criteria: str, sample_df: pd.DataFrame) -> None:
    """filter_column should reject destructive SQL in criteria strings."""
    conn = duckdb.connect()
    rel = conn.from_df(sample_df)
    with pytest.raises(ValueError):
        filter_column(rel, "age", criteria, conn=conn).df()


@pytest.mark.parametrize("indices", [[-1], [0, -2], [0, "1"], ["1"]])
def test_select_rows_rejects_invalid_indices(indices: list[object], sample_df: pd.DataFrame) -> None:
    """select_rows should reject invalid row index lists."""
    conn = duckdb.connect()
    rel = conn.from_df(sample_df)
    with pytest.raises(ValueError):
        select_rows(rel, indices=indices, conn=conn).df()


@pytest.mark.parametrize(
    "op_name, kwargs",
    [
        ("remove_columns", {"columns": ["missing"]}),
        ("dropna", {"subset": ["missing"]}),
        ("coalesce", {"columns": ["age", "missing"], "target_column": "x"}),
        ("encode_categorical", {"column": "missing"}),
        ("get_dummies", {"columns": ["missing"]}),
        ("select_columns", {"columns": ["missing"]}),
        ("transform_column", {"column": "missing", "func": "age + 1"}),
        ("bin_numeric", {"column": "missing", "target_column": "b"}),
        ("change_type", {"column": "missing", "dtype": "VARCHAR"}),
        ("fill", {"column": "missing"}),
        ("expand_column", {"column": "missing"}),
        ("impute", {"column": "missing"}),
        ("jitter", {"column": "missing", "target_column": "j"}),
        ("count_cumulative_unique", {"column": "missing"}),
    ],
)
def test_methods_raise_for_missing_columns(
    op_name: str, kwargs: dict[str, object], sample_df: pd.DataFrame
) -> None:
    """Column-referencing methods should raise actionable missing-column errors."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="Unknown column|not found"):
        getattr(dj, op_name)(**kwargs).collect()


@pytest.mark.parametrize("bad_n", [-1, -5])
def test_head_rejects_negative_n(bad_n: int, sample_df: pd.DataFrame) -> None:
    """head should reject negative row counts."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="n must be >= 0"):
        dj.head(bad_n)


@pytest.mark.parametrize(
    "dtype",
    [
        "VARCHAR; DROP TABLE t",
        "INT -- comment",
        "DOUBLE /* comment */",
    ],
)
def test_change_type_rejects_unsafe_dtype(dtype: str, sample_df: pd.DataFrame) -> None:
    """change_type should reject unsafe dtype fragments."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError):
        dj.change_type("age", dtype).collect()


@pytest.mark.parametrize("bins", [0, -1])
def test_bin_numeric_rejects_non_positive_int_bins(bins: int, sample_df: pd.DataFrame) -> None:
    """bin_numeric should reject non-positive integer bins."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="bins must be >= 1"):
        dj.bin_numeric("age", "age_bin", bins=bins).collect()


@pytest.mark.parametrize("bins", [[1.0], []])
def test_bin_numeric_rejects_short_edges(bins: list[float], sample_df: pd.DataFrame) -> None:
    """bin_numeric should require at least two edge values."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="at least two edge values"):
        dj.bin_numeric("age", "age_bin", bins=bins).collect()


@pytest.mark.parametrize("k", [0, -1, -10])
def test_groupby_topk_rejects_bad_k(k: int, sample_df: pd.DataFrame) -> None:
    """groupby_topk should require k >= 1."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="k must be >= 1"):
        dj.groupby_topk("city", "age", k=k).collect()


@pytest.mark.parametrize("scale", [-0.1, -1.0])
def test_jitter_rejects_negative_scale(scale: float, sample_df: pd.DataFrame) -> None:
    """jitter should reject negative jitter scale."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="scale must be >= 0"):
        dj.jitter("age", "age_j", scale=scale).collect()


@pytest.mark.parametrize("value_pairs", [{}, dict()])
def test_find_replace_rejects_empty_mapping(
    value_pairs: dict[object, object], sample_df: pd.DataFrame
) -> None:
    """find_replace should reject empty replacement maps."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="cannot be empty"):
        dj.find_replace("city", value_pairs).collect()


def test_empty_dataframe_roundtrip() -> None:
    """Basic no-op operations should work on an empty-row DataFrame."""
    df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
    result = DuckJanitor.from_pandas(df).clean_names().select_columns(["a", "b"]).collect()
    assert list(result.columns) == ["a", "b"]
    assert result.empty


def test_single_row_dataframe_mutate() -> None:
    """mutate should work on single-row data."""
    df = pd.DataFrame({"x": [5]})
    result = DuckJanitor.from_pandas(df).mutate(y="x * 10").collect()
    assert result.shape == (1, 2)
    assert result["y"].iloc[0] == 50


def test_all_null_column_impute_mean() -> None:
    """impute on all-null column should keep nulls when no statistic can be computed."""
    df = pd.DataFrame({"x": [None, None, None]})
    result = DuckJanitor.from_pandas(df).impute("x", statistic="mean").collect()
    assert result["x"].isna().all()


def test_duplicate_column_names_clean_names() -> None:
    """clean_names should produce unique columns when duplicates are present."""
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "A", "A!"])
    result = DuckJanitor.from_pandas(df).clean_names().collect()
    cols = list(result.columns)
    assert cols[0] == "a"
    assert len(cols) == len(set(cols))
    assert all(col.startswith("a") for col in cols)


def test_dropnotnull_all_keeps_only_all_null_rows() -> None:
    """dropnotnull(how='all') should keep rows where all subset cols are null."""
    df = pd.DataFrame({"a": [None, 1, None], "b": [None, None, 2]})
    result = DuckJanitor.from_pandas(df).dropnotnull(subset=["a", "b"], how="all").collect()
    assert len(result) == 1
    assert result["a"].isna().iloc[0] and result["b"].isna().iloc[0]


def test_join_apply_cross_connection_materialization() -> None:
    """join_apply should work when both sides come from different DuckDB connections."""
    conn_left = duckdb.connect()
    conn_right = duckdb.connect()
    left = DuckJanitor(conn_left.from_df(pd.DataFrame({"id": [1, 2], "a": [10, 20]})), conn_left)
    right = DuckJanitor(conn_right.from_df(pd.DataFrame({"id": [1, 2], "b": [1, 2]})), conn_right)
    result = left.join_apply(right, on="id", func=lambda row: row["a"] + row["b"], new_column_name="sum").collect()
    assert list(result["sum"]) == [11, 22]


def test_process_text_validation_errors(sample_df: pd.DataFrame) -> None:
    """process_text should validate column and new column names."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError):
        dj.process_text("missing", "upper(name)", "x")
    with pytest.raises(ValueError):
        dj.process_text("name", "upper(name)", "")


def test_mutate_multiple_columns(sample_df: pd.DataFrame) -> None:
    """mutate should support multiple SQL expressions in one call."""
    result = (
        DuckJanitor.from_pandas(sample_df)
        .mutate(double_age="age * 2", city_age="city || '_' || CAST(age AS VARCHAR)")
        .collect()
    )
    assert "double_age" in result.columns
    assert "city_age" in result.columns
    assert list(result["double_age"]) == [50, 60, 70]


@pytest.mark.parametrize("alias_value", ["", "   "])
def test_alias_rejects_empty_string(alias_value: str, sample_df: pd.DataFrame) -> None:
    """alias should reject empty string aliases."""
    with pytest.raises(ValueError):
        DuckJanitor.from_pandas(sample_df).alias(alias_value).collect()
