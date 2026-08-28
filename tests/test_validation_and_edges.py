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


# ---- H1/H2: SQL fragment validator should not block legitimate values ----

def test_filter_on_allows_string_literal_with_dashes(sample_df: pd.DataFrame) -> None:
    """filter_on should allow string literals containing -- (H1 fix)."""
    df = pd.DataFrame({"name": ["a--b", "c", "d"], "age": [1, 2, 3]})
    dj = DuckJanitor.from_pandas(df)
    result = dj.filter_on("name = 'a--b'").collect()
    assert len(result) == 1
    assert result["name"].iloc[0] == "a--b"


def test_filter_on_allows_string_literal_with_comment_markers() -> None:
    """filter_on should allow string literals containing /* */ (H1 fix)."""
    df = pd.DataFrame({"comment": ["see /* note */", "x", "y"]})
    dj = DuckJanitor.from_pandas(df)
    result = dj.filter_on("comment = 'see /* note */'").collect()
    assert len(result) == 1


def test_filter_on_allows_column_named_drop() -> None:
    """filter_on should allow column references whose names match reserved words (H2 fix)."""
    df = pd.DataFrame({"drop": [10, 20, 30]})
    dj = DuckJanitor.from_pandas(df)
    result = dj.filter_on("drop > 15").collect()
    assert len(result) == 2


def test_add_column_allows_column_named_update() -> None:
    """add_column should allow SQL expressions referencing columns named like keywords (H2 fix)."""
    df = pd.DataFrame({"update": [1, 2, 3]})
    dj = DuckJanitor.from_pandas(df)
    result = dj.add_column("y", "update + 1").collect()
    assert list(result["y"]) == [2, 3, 4]


def test_filter_on_still_blocks_select_from(sample_df: pd.DataFrame) -> None:
    """filter_on should still block SELECT ... FROM statements."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="disallowed SQL statement"):
        dj.filter_on("SELECT * FROM users").collect()


def test_filter_on_still_blocks_delete_from(sample_df: pd.DataFrame) -> None:
    """filter_on should still block DELETE FROM statements."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="disallowed SQL statement"):
        dj.filter_on("DELETE FROM users WHERE 1=1").collect()


# ---- H3: filter_string regex error handling ----

def test_filter_string_invalid_regex_raises_value_error(sample_df: pd.DataFrame) -> None:
    """filter_string should raise ValueError (not re.error) for invalid regex (H3 fix)."""
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        dj.filter_string("name", "[invalid", regex=True).collect()


def test_filter_string_valid_regex_works(sample_df: pd.DataFrame) -> None:
    """filter_string should work with a valid regex pattern."""
    dj = DuckJanitor.from_pandas(sample_df)
    result = dj.filter_string("name", "^a", regex=True).collect()
    assert len(result) == 1
    assert result["name"].iloc[0] == "alice"


# ---- L8: DuckJanitor.sql() word-boundary replacement ----

def test_sql_word_boundary_replacement() -> None:
    """sql() should only replace the standalone word 'self', not substrings."""
    df = pd.DataFrame({"selfish": [1, 2], "val": [10, 20]})
    dj = DuckJanitor.from_pandas(df)
    # 'self' should be replaced with the temp table name; 'selfish' should remain
    result = dj.sql("SELECT selfish, val FROM self").collect()
    assert "selfish" in result.columns
    assert list(result["selfish"]) == [1, 2]
    assert list(result["val"]) == [10, 20]


# ---- M4: add_column with literal string containing semicolon ----

def test_add_column_literal_string_with_semicolon() -> None:
    """add_column should fall back to literal for strings that fail SQL validation (M4 fix)."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    dj = DuckJanitor.from_pandas(df)
    result = dj.add_column("note", "abc; def").collect()
    assert list(result["note"]) == ["abc; def", "abc; def", "abc; def"]


# ---- M5: validation branch coverage ----

def test_clean_names_rejects_invalid_case_type(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="case_type must be one of"):
        dj.clean_names(case_type="invalid").collect()


def test_remove_columns_rejects_empty_list(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="at least one"):
        dj.remove_columns([]).collect()


def test_add_column_rejects_empty_name(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="column_name"):
        dj.add_column("", 42).collect()


def test_rename_column_rejects_empty_new_name(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="new_name"):
        dj.rename_column("age", "").collect()


def test_transform_columns_rejects_length_mismatch(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="same length"):
        dj.transform_columns(["age", "value"], "col + 1", target_columns=["x"]).collect()


def test_conditional_join_rejects_invalid_how(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    other = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="how must be one of"):
        dj.conditional_join(other, [("age", "age", ">")], how="outer").collect()


def test_flag_nulls_missing_column(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="Unknown column"):
        dj.flag_nulls(columns="missing").collect()


def test_limit_column_characters_rejects_negative(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="max_chars must be >= 0"):
        dj.limit_column_characters("name", -1).collect()


def test_get_dupes_rejects_empty_columns(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="at least one"):
        dj.get_dupes(columns=[]).collect()


def test_dropnotnull_rejects_empty_subset(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="at least one"):
        dj.dropnotnull(subset=[]).collect()


def test_label_encode_rejects_empty_columns(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="at least one"):
        dj.label_encode(columns=[]).collect()


def test_complete_rejects_empty_columns(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="at least one"):
        dj.complete(columns=[]).collect()


def test_case_when_rejects_empty_conditions(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="conditions"):
        dj.case_when([], "result").collect()


def test_alias_callable_returning_empty(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="non-empty string"):
        dj.alias(lambda col: "").collect()


def test_min_max_scale_rejects_empty_target(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="target_column"):
        dj.min_max_scale("age", "").collect()


def test_pivot_wider_rejects_empty_name_col(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="name_col"):
        dj.pivot_wider("name", "", "age").collect()


def test_pivot_longer_rejects_empty_names_to(sample_df: pd.DataFrame) -> None:
    dj = DuckJanitor.from_pandas(sample_df)
    with pytest.raises(ValueError, match="names_to"):
        dj.pivot_longer("age", names_to="", values_to="val").collect()


# ---- M2: find_replace with numeric keys/values ----

def test_find_replace_numeric_keys_values() -> None:
    """find_replace should accept numeric keys and values (M2 regression test)."""
    df = pd.DataFrame({"code": [1, 2, 3]})
    dj = DuckJanitor.from_pandas(df)
    result = dj.find_replace("code", {1: 100, 2: 200}, target_column="code_new").collect()
    assert result["code_new"].iloc[0] == 100
    assert result["code_new"].iloc[1] == 200
    assert result["code_new"].isna().iloc[2]  # unmatched row becomes NULL


# ---- M6: positive control for filter_column ----

def test_filter_column_positive_control(sample_df: pd.DataFrame) -> None:
    """filter_column should pass through legitimate SQL predicates (M6 fix)."""
    dj = DuckJanitor.from_pandas(sample_df)
    result = dj.filter_column("age", "age > 25").collect()
    assert len(result) == 2
    assert "charlie" in result["name"].tolist()


# ---- L6: head(n=0) edge case ----

def test_head_zero_returns_empty(sample_df: pd.DataFrame) -> None:
    """head(n=0) should return an empty DataFrame (L6)."""
    dj = DuckJanitor.from_pandas(sample_df)
    result = dj.head(0)
    assert result.empty
