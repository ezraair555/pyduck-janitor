"""
Tests for sampling and Python-UDF verbs.
"""

import duckdb
import pandas as pd
import pytest
from pyduck_janitor.sampling import (
    approx_distinct,
    approx_quantile,
    apply_python_function,
    register_python_function,
    sample,
    sample_rows,
)


@pytest.fixture
def conn():
    """Create a DuckDB connection."""
    return duckdb.connect()


@pytest.fixture
def relation(conn):
    """Create a 1000-row relation with an integer and a string column."""
    df = pd.DataFrame(
        {
            "id": list(range(1000)),
            "label": [f"row_{i}" for i in range(1000)],
            "value": [float(i) / 4 for i in range(1000)],
        }
    )
    return conn.from_df(df)


class TestSample:
    """Tests for `sample`."""

    def test_sample_with_n(self, conn, relation):
        """Sample an exact number of rows."""
        result = sample(relation, n=25, seed=42, conn=conn)
        assert len(result.df()) == 25
        assert result.columns == relation.columns

    def test_sample_with_frac(self, conn, relation):
        """Sample a fraction of the rows."""
        result = sample(relation, frac=0.1, seed=42, conn=conn)
        assert 70 <= len(result.df()) <= 130

    def test_sample_fraction_one(self, conn, relation):
        """A fraction of 1.0 keeps every row."""
        result = sample(relation, frac=1.0, conn=conn)
        assert len(result.df()) == 1000

    def test_sample_is_reproducible_with_seed(self, conn, relation):
        """The same seed yields the same rows."""
        first = sample(relation, n=20, seed=7, conn=conn).df()
        second = sample(relation, n=20, seed=7, conn=conn).df()
        assert first["id"].tolist() == second["id"].tolist()

    def test_sample_requires_n_or_frac(self, conn, relation):
        """Passing neither n nor frac raises."""
        with pytest.raises(ValueError, match="Exactly one"):
            sample(relation, conn=conn)

    def test_sample_rejects_both_n_and_frac(self, conn, relation):
        """Passing both n and frac raises."""
        with pytest.raises(ValueError, match="Exactly one"):
            sample(relation, n=10, frac=0.5, conn=conn)

    def test_sample_rejects_unknown_method(self, conn, relation):
        """An unsupported method raises and lists the supported ones."""
        with pytest.raises(ValueError) as excinfo:
            sample(relation, n=10, method="simple", conn=conn)
        message = str(excinfo.value)
        assert "reservoir" in message
        assert "bernoulli" in message
        assert "system" in message

    def test_sample_bernoulli_requires_frac(self, conn, relation):
        """bernoulli/system do not support a discrete row count."""
        with pytest.raises(ValueError, match="does not support a discrete row count"):
            sample(relation, n=10, method="bernoulli", conn=conn)

    def test_sample_bernoulli_with_frac(self, conn, relation):
        """bernoulli works with a fraction."""
        result = sample(relation, frac=0.1, method="bernoulli", seed=3, conn=conn)
        assert 70 <= len(result.df()) <= 130

    def test_sample_system_with_frac(self, conn, relation):
        """system works with a fraction."""
        result = sample(relation, frac=0.1, method="system", seed=3, conn=conn)
        assert len(result.df()) <= 1000

    def test_sample_rejects_negative_seed(self, conn, relation):
        """A negative seed raises."""
        with pytest.raises(ValueError, match="seed"):
            sample(relation, n=5, seed=-1, conn=conn)

    def test_sample_rejects_non_integer_seed(self, conn, relation):
        """A non-integer seed raises."""
        with pytest.raises(ValueError, match="seed"):
            sample(relation, n=5, seed="abc", conn=conn)

    def test_sample_rejects_negative_n(self, conn, relation):
        """A negative n raises."""
        with pytest.raises(ValueError, match="n must be"):
            sample(relation, n=-5, conn=conn)

    def test_sample_works_on_temp_view(self, conn):
        """The relation may be registered as a temporary view."""
        df = pd.DataFrame({"id": list(range(100))})
        conn.register("df_src", df)
        conn.execute("CREATE TEMP VIEW v AS SELECT * FROM df_src")
        view_relation = conn.query("SELECT * FROM v")
        result = sample(view_relation, n=10, seed=1, conn=conn)
        assert len(result.df()) == 10


class TestSampleRows:
    """Tests for `sample_rows`."""

    def test_sample_rows_returns_n_rows(self, conn, relation):
        """Exactly n rows are returned."""
        result = sample_rows(relation, 15, seed=11, conn=conn)
        assert len(result.df()) == 15

    def test_sample_rows_is_deterministic(self, conn, relation):
        """The same seed yields the same rows in the same order."""
        first = sample_rows(relation, 10, seed=5, conn=conn).df()
        second = sample_rows(relation, 10, seed=5, conn=conn).df()
        assert first["id"].tolist() == second["id"].tolist()

    def test_sample_rows_larger_than_input(self, conn, relation):
        """Requesting more rows than available returns everything."""
        result = sample_rows(relation, 5000, conn=conn)
        assert len(result.df()) == 1000

    def test_sample_rows_rejects_negative_n(self, conn, relation):
        """A negative n raises."""
        with pytest.raises(ValueError, match="n must be"):
            sample_rows(relation, -1, conn=conn)

    def test_sample_rows_rejects_bad_seed(self, conn, relation):
        """A bad seed raises."""
        with pytest.raises(ValueError, match="seed"):
            sample_rows(relation, 5, seed=-3, conn=conn)


class TestApprox:
    """Tests for `approx_distinct` and `approx_quantile`."""

    def test_approx_distinct(self, conn, relation):
        """Approximate distinct count of a column (HyperLogLog, so approximate)."""
        result = approx_distinct(relation, "id", conn=conn)
        count = result.fetchone()[0]
        assert 900 <= count <= 1300

    def test_approx_distinct_small_cardinality(self, conn):
        """A low-cardinality column is estimated closely."""
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 100})
        rel = conn.from_df(df)
        result = approx_distinct(rel, "cat", conn=conn)
        assert result.fetchone()[0] == pytest.approx(3, abs=1)

    def test_approx_distinct_unknown_column(self, conn, relation):
        """An unknown column raises."""
        with pytest.raises(ValueError, match="Unknown column"):
            approx_distinct(relation, "nope", conn=conn)

    def test_approx_quantile(self, conn, relation):
        """Approximate median of a numeric column."""
        result = approx_quantile(relation, "value", 0.5, conn=conn)
        assert result.fetchone()[0] == pytest.approx(124.875, abs=1.0)

    def test_approx_quantile_invalid_quantile(self, conn, relation):
        """A quantile outside [0, 1] raises."""
        with pytest.raises(ValueError, match="quantile"):
            approx_quantile(relation, "value", 1.5, conn=conn)
        with pytest.raises(ValueError, match="quantile"):
            approx_quantile(relation, "value", -0.1, conn=conn)

    def test_approx_quantile_unknown_column(self, conn, relation):
        """An unknown column raises."""
        with pytest.raises(ValueError, match="Unknown column"):
            approx_quantile(relation, "nope", 0.5, conn=conn)

    def test_approx_quantile_bad_type(self, conn, relation):
        """A non-numeric column raises a ValueError."""
        with pytest.raises(ValueError, match="approx_quantile"):
            approx_quantile(relation, "label", 0.5, conn=conn)


class TestPythonFunctions:
    """Tests for `register_python_function` and `apply_python_function`."""

    def test_register_returns_same_relation(self, conn, relation):
        """Registration returns the same relation unchanged."""
        result = register_python_function(
            relation,
            "my_upper",
            lambda s: s.upper(),
            params=["VARCHAR"],
            return_type="VARCHAR",
            conn=conn,
        )
        assert result is relation
        assert conn.execute("SELECT my_upper('bob')").fetchone()[0] == "BOB"

    def test_register_requires_connection(self, relation):
        """Registering without a connection raises."""
        with pytest.raises(ValueError, match="connection"):
            register_python_function(relation, "f", lambda s: s)

    def test_register_rejects_empty_name(self, conn, relation):
        """An empty name raises."""
        with pytest.raises(ValueError, match="name"):
            register_python_function(relation, "  ", lambda s: s, conn=conn)

    def test_register_infers_types_from_annotations(self, conn, relation):
        """Types are inferred from the callable's annotations."""

        def double(value: int) -> int:
            return value * 2

        register_python_function(relation, "double_it", double, conn=conn)
        assert conn.execute("SELECT double_it(21)").fetchone()[0] == 42

    def test_register_requires_return_type_when_uninferable(self, conn, relation):
        """A missing return type raises an explanatory ValueError."""

        def no_annotation(value):
            return value

        with pytest.raises(ValueError, match="return_type"):
            register_python_function(
                relation,
                "no_ret",
                no_annotation,
                params=["VARCHAR"],
                conn=conn,
            )

    def test_apply_python_function_adds_column(self, conn, relation):
        """The UDF result lands in a `<column>_py` column."""
        result = apply_python_function(
            relation,
            "label",
            lambda s: s.upper(),
            params=["VARCHAR"],
            return_type="VARCHAR",
            conn=conn,
        )
        df = result.df()
        assert "label_py" in df.columns
        assert df["label_py"].iloc[0] == "ROW_0"
        assert len(df) == 1000

    def test_apply_python_function_after_register(self, conn, relation):
        """A registered UDF can be reused in a following call."""
        register_python_function(
            relation,
            "shout",
            lambda s: s + "!",
            params=["VARCHAR"],
            return_type="VARCHAR",
            conn=conn,
        )
        result = apply_python_function(
            relation,
            "label",
            lambda s: s + "?",
            name="ask",
            params=["VARCHAR"],
            return_type="VARCHAR",
            conn=conn,
        )
        df = result.df()
        assert df["label_py"].iloc[1] == "row_1?"
        assert conn.execute("SELECT shout('hi')").fetchone()[0] == "hi!"

    def test_apply_python_function_unknown_column(self, conn, relation):
        """An unknown column raises."""
        with pytest.raises(ValueError, match="Unknown column"):
            apply_python_function(relation, "nope", lambda s: s, conn=conn)

    def test_apply_python_function_requires_connection(self, relation):
        """Applying without a connection raises."""
        with pytest.raises(ValueError, match="connection"):
            apply_python_function(relation, "label", lambda s: s)

    def test_apply_python_function_infers_column_type(self, conn, relation):
        """Without explicit params the column's own type is used."""

        def to_int(value):
            return value

        result = apply_python_function(
            relation,
            "id",
            to_int,
            return_type="BIGINT",
            conn=conn,
        )
        assert result.df()["id_py"].iloc[3] == 3

    def test_apply_python_function_numeric(self, conn, relation):
        """A numeric UDF works end to end."""

        def scaled(value: float) -> float:
            return value * 2

        result = apply_python_function(
            relation,
            "value",
            scaled,
            params=["DOUBLE"],
            return_type="DOUBLE",
            conn=conn,
        )
        assert result.df()["value_py"].iloc[2] == pytest.approx(1.0)
