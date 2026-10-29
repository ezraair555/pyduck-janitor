"""Tests for the 0.4.0 DuckDB-native additions.

Covers the output suite (write-back without pandas), the inspection
properties, the connection/persistence model, the literal-aware ``self``
substitution in :meth:`DuckJanitor.sql`, parameter binding, the enriched
profile, and the sampling / Python-UDF verbs.
"""

from __future__ import annotations

import json
import os

import duckdb
import pandas as pd
import pytest

from pyduck_janitor import DuckJanitor
from pyduck_janitor.io_ops import write_relation


@pytest.fixture()
def frame() -> pd.DataFrame:
    """Small deterministic frame used across the module."""
    return pd.DataFrame(
        {
            "Sales Month": ["Jan", "Feb", "Mar"],
            "Company 1": [150.0, 200.0, 300.0],
            "Company 2": [180.0, 250.0, None],
        }
    )


@pytest.fixture()
def dj(frame: pd.DataFrame) -> DuckJanitor:
    """A DuckJanitor over the sample frame."""
    return DuckJanitor.from_pandas(frame)


# ----------------------------------------------------------------------
# Output suite — write-back without pandas
# ----------------------------------------------------------------------


class TestOutputSuite:
    """Writing results through DuckDB rather than pandas."""

    def test_to_parquet_writes_a_readable_file(self, dj, tmp_path):
        target = tmp_path / "out.parquet"
        written = dj.to_parquet(target)
        assert os.path.exists(written)
        back = DuckJanitor.from_parquet(written)
        assert back.count() == 3
        assert set(back.columns) == set(dj.columns)

    def test_to_parquet_creates_parent_directories(self, dj, tmp_path):
        target = tmp_path / "nested" / "deeper" / "out.parquet"
        dj.to_parquet(target)
        assert target.exists()

    def test_to_parquet_refuses_to_overwrite_by_default(self, dj, tmp_path):
        target = tmp_path / "out.parquet"
        dj.to_parquet(target)
        with pytest.raises(FileExistsError):
            dj.to_parquet(target)

    def test_to_parquet_overwrite_replaces_the_file(self, dj, tmp_path):
        target = tmp_path / "out.parquet"
        dj.to_parquet(target)
        dj.to_parquet(target, overwrite=True)
        assert target.exists()

    def test_to_csv_writes_header_and_rows(self, dj, tmp_path):
        target = tmp_path / "out.csv"
        dj.to_csv(target)
        text = target.read_text()
        assert "Sales Month" in text
        assert len(text.strip().splitlines()) == 4

    def test_to_json_writes_records(self, dj, tmp_path):
        target = tmp_path / "out.json"
        dj.to_json(target)
        payload = json.loads(target.read_text())
        assert len(payload) == 3

    def test_to_json_newline_delimited_writes_one_object_per_line(self, dj, tmp_path):
        target = tmp_path / "out.jsonl"
        dj.to_json(target, newline_delimited=True)
        lines = [line for line in target.read_text().splitlines() if line.strip()]
        assert len(lines) == 3
        assert json.loads(lines[0])["Company 1"] == 150.0

    def test_to_table_creates_a_catalog_table(self, dj):
        dj.to_table("written_table")
        conn = dj.connection
        rows = conn.execute("SELECT COUNT(*) FROM written_table").fetchone()
        assert rows[0] == 3

    def test_to_view_creates_a_catalog_view(self, dj):
        dj.to_view("written_view")
        rows = dj.connection.execute("SELECT COUNT(*) FROM written_view").fetchone()
        assert rows[0] == 3

    def test_to_arrow_returns_an_arrow_table(self, dj):
        table = dj.to_arrow()
        assert table.num_rows == 3
        assert "Sales Month" in table.column_names

    def test_write_relation_rejects_unknown_format(self, dj, tmp_path):
        with pytest.raises(ValueError, match="fmt must be one of"):
            write_relation(dj.relation, tmp_path / "out.bin", fmt="binary", conn=dj.connection)

    def test_write_relation_requires_a_connection(self, dj, tmp_path):
        with pytest.raises(ValueError, match="connection is required"):
            write_relation(dj.relation, tmp_path / "out.parquet")


# ----------------------------------------------------------------------
# Inspection
# ----------------------------------------------------------------------


class TestInspection:
    """Cheap metadata without materialising the frame."""

    def test_columns_property(self, dj):
        assert dj.columns == ["Sales Month", "Company 1", "Company 2"]

    def test_dtypes_property(self, dj):
        types = dj.dtypes
        assert types["Sales Month"] == "VARCHAR"
        assert "DOUBLE" in types["Company 1"]

    def test_shape_property(self, dj):
        assert dj.shape == (3, 3)

    def test_count_method(self, dj):
        assert dj.count() == 3

    def test_glimpse_is_an_alias_for_head(self, dj):
        assert len(dj.glimpse(2)) == 2

    def test_show_returns_a_box_table(self, dj):
        rendered = dj.show(2)
        assert "Sales Month" in rendered
        assert "│" in rendered

    def test_relation_and_connection_properties_are_public(self, dj):
        assert isinstance(dj.relation, duckdb.DuckDBPyRelation)
        assert isinstance(dj.connection, duckdb.DuckDBPyConnection)

    def test_repr_html_renders_schema_and_preview(self, dj):
        html = dj._repr_html_()
        assert "<table" in html
        assert "Sales Month" in html
        assert "schema" in html


# ----------------------------------------------------------------------
# Connection / persistence model
# ----------------------------------------------------------------------


class TestConnectionModel:
    """File-backed sessions, catalog browsing, and transactions."""

    def test_connect_creates_a_persistent_database(self, tmp_path):
        target = tmp_path / "warehouse.duckdb"
        with DuckJanitor.connect(target) as session:
            DuckJanitor.from_pandas(
                pd.DataFrame({"a": [1, 2]}), connection=session.connection
            ).to_table("persisted")
        assert target.exists()
        with DuckJanitor.connect(target) as session:
            rows = session.connection.execute("SELECT COUNT(*) FROM persisted").fetchone()
            assert rows[0] == 2

    def test_connect_without_a_path_is_in_memory(self):
        with DuckJanitor.connect() as session:
            assert session.count() == 1

    def test_tables_lists_catalog_objects(self, dj):
        dj.to_table("catalog_table")
        assert any(name.endswith("catalog_table") for name in dj.tables())

    def test_databases_lists_attached_databases(self, dj):
        assert "memory" in dj.databases()

    def test_attach_and_detach(self, dj, tmp_path):
        other = tmp_path / "other.duckdb"
        conn = duckdb.connect(str(other))
        conn.execute("CREATE TABLE t AS SELECT 1 AS a")
        conn.close()
        dj.attach(other, "otherdb")
        assert "otherdb" in dj.databases()
        dj.detach("otherdb")
        assert "otherdb" not in dj.databases()

    def test_attach_rejects_an_unsafe_alias(self, dj):
        with pytest.raises(ValueError, match="plain identifier"):
            dj.attach("/tmp/nope.duckdb", "bad alias; DROP TABLE x")

    def test_transaction_helpers_run(self, dj):
        dj.begin()
        dj.to_table("tx_table")
        dj.commit()
        rows = dj.connection.execute("SELECT COUNT(*) FROM tx_table").fetchone()
        assert rows[0] == 3


# ----------------------------------------------------------------------
# sql() — literal-aware self substitution and parameter binding
# ----------------------------------------------------------------------


class TestSqlEscapeHatch:
    """The raw SQL escape hatch is safe to teach."""

    def test_self_is_replaced(self, dj):
        result = dj.sql('SELECT * FROM self WHERE "Company 1" > 200').collect()
        assert len(result) == 1

    def test_self_inside_a_string_literal_is_untouched(self, dj):
        result = dj.sql("SELECT 'self' AS marker FROM self LIMIT 1").collect()
        assert result["marker"].tolist() == ["self"]

    def test_self_inside_a_quoted_identifier_is_untouched(self, dj):
        # A column genuinely named "self" must survive the substitution.
        # If the quoted identifier were rewritten to the temp table name,
        # the query would fail to bind.
        framed = DuckJanitor.from_pandas(pd.DataFrame({"self": [1, 2]}))
        result = framed.sql('SELECT "self" FROM self').collect()
        assert result["self"].tolist() == [1, 2]

    def test_self_inside_a_comment_is_untouched(self, dj):
        result = dj.sql("SELECT 1 AS n -- self\nFROM self LIMIT 1").collect()
        assert result["n"].tolist() == [1]

    def test_named_parameters_are_bound(self, dj):
        result = dj.sql(
            'SELECT * FROM self WHERE "Company 1" > $threshold', threshold=200.0
        ).collect()
        assert len(result) == 1

    def test_parameters_are_not_interpolated_into_the_text(self, dj):
        # A value that would be dangerous if interpolated must round-trip
        # as a bound parameter instead.
        result = dj.sql("SELECT $value AS v", value="'; DROP TABLE x; --").collect()
        assert result["v"].tolist() == ["'; DROP TABLE x; --"]


# ----------------------------------------------------------------------
# profile()
# ----------------------------------------------------------------------


class TestProfile:
    """The enriched column profile."""

    def test_profile_reports_null_and_distinct_counts(self, dj):
        profile = dj.profile().collect()
        row = profile[profile["column_name"] == "Company 2"].iloc[0]
        assert row["null_count"] == 1
        assert row["distinct_count"] == 2

    def test_profile_adds_numeric_statistics(self, dj):
        profile = dj.profile(quantiles=[0.5]).collect()
        row = profile[profile["column_name"] == "Company 1"].iloc[0]
        assert row["mean_value"] == pytest.approx(216.6666666, rel=1e-6)
        assert row["q0_5_value"] == pytest.approx(200.0)

    def test_profile_reports_top_values(self, dj):
        profile = dj.profile(top_n=2).collect()
        row = profile[profile["column_name"] == "Sales Month"].iloc[0]
        assert "Jan" in row["top_values"]

    def test_profile_rejects_an_out_of_range_quantile(self, dj):
        with pytest.raises(ValueError, match="between 0 and 1"):
            dj.profile(quantiles=[1.5])

    def test_profile_rejects_a_negative_top_n(self, dj):
        with pytest.raises(ValueError, match="top_n must be"):
            dj.profile(top_n=-1)


# ----------------------------------------------------------------------
# Sampling and Python UDFs
# ----------------------------------------------------------------------


class TestSampling:
    """DuckDB-native sampling."""

    def test_sample_by_row_count(self, dj):
        assert dj.sample(n=2, seed=1).count() == 2

    def test_sample_by_fraction(self, dj):
        assert dj.sample(frac=0.5, seed=1).count() >= 1

    def test_sample_rejects_both_n_and_frac(self, dj):
        with pytest.raises(ValueError):
            dj.sample(n=1, frac=0.5)

    def test_sample_rows(self, dj):
        assert dj.sample_rows(2, seed=3).count() == 2

    def test_approx_distinct(self, dj):
        value = dj.approx_distinct("Sales Month").collect().iloc[0, 0]
        assert value == 3

    def test_approx_quantile(self, dj):
        value = dj.approx_quantile("Company 1", 0.5).collect().iloc[0, 0]
        assert value == pytest.approx(200.0)


class TestPythonUdf:
    """Python callables registered inside DuckDB."""

    def test_register_and_call_a_udf(self, dj):
        session = dj.register_python_function(
            "py_len", lambda s: len(s) if s else 0, params=["VARCHAR"], return_type="BIGINT"
        )
        result = session.sql("SELECT py_len('abcd') AS n").collect()
        assert result["n"].tolist() == [4]

    def test_apply_python_function_adds_a_column(self, dj):
        result = dj.apply_python_function(
            "Sales Month",
            lambda s: s.upper(),
            params=["VARCHAR"],
            return_type="VARCHAR",
        ).collect()
        assert result["Sales Month_py"].tolist() == ["JAN", "FEB", "MAR"]


# ----------------------------------------------------------------------
# asof_join now uses DuckDB's native ASOF JOIN
# ----------------------------------------------------------------------


class TestNativeAsofJoin:
    """The native ASOF path preserves the previous semantics."""

    def test_backward_join_picks_the_latest_prior_row(self):
        events = DuckJanitor.from_pandas(
            pd.DataFrame(
                {
                    "employee_id": [1, 1],
                    "event_time": pd.to_datetime(["2024-01-10", "2024-01-20"]),
                }
            )
        )
        history = DuckJanitor.from_pandas(
            pd.DataFrame(
                {
                    "employee_id": [1, 1],
                    "effective_time": pd.to_datetime(["2024-01-01", "2024-01-15"]),
                    "manager": ["A", "B"],
                }
            )
        )
        result = events.asof_join(
            history, "event_time", "effective_time", by="employee_id"
        ).collect()
        assert result["manager"].tolist() == ["A", "B"]

    def test_duplicate_right_timestamps_keep_the_last_row(self):
        events = DuckJanitor.from_pandas(
            pd.DataFrame({"id": [1], "ts": pd.to_datetime(["2024-01-20"])})
        )
        history = DuckJanitor.from_pandas(
            pd.DataFrame(
                {
                    "id": [1, 1],
                    "ts": pd.to_datetime(["2024-01-15", "2024-01-15"]),
                    "mgr": ["FIRST", "SECOND"],
                }
            )
        )
        result = events.asof_join(history, "ts", "ts", by="id").collect()
        # The pre-0.4.0 emulation kept the last registered right row.
        assert result["mgr"].tolist() == ["SECOND"]

    def test_tolerance_nulls_the_right_columns_but_keeps_the_left_row(self):
        left = DuckJanitor.from_pandas(
            pd.DataFrame(
                {
                    "employee_id": [1],
                    "event_time": pd.to_datetime(["2024-01-20"]),
                    "value": [100],
                }
            )
        )
        right = DuckJanitor.from_pandas(
            pd.DataFrame(
                {
                    "employee_id": [1],
                    "effective_time": pd.to_datetime(["2024-01-01"]),
                    "value": [10],
                }
            )
        )
        result = left.asof_join(
            right,
            "event_time",
            "effective_time",
            by="employee_id",
            tolerance="7 days",
            suffixes=("_event", "_history"),
        ).collect()
        assert result["value_event"].tolist() == [100]
        assert pd.isna(result["value_history"].iloc[0])


# ----------------------------------------------------------------------
# explain()
# ----------------------------------------------------------------------


class TestExplain:
    """The query-plan diagnostics."""

    def test_explain_returns_plan_text(self, dj):
        plan = dj.explain()
        assert isinstance(plan, str)
        # The old implementation returned the tuple repr of fetchall(),
        # which looked like "[('physical_plan', '...')]".
        assert not plan.startswith("[")
        assert "SCAN" in plan.upper()

    def test_explain_analyze_runs_the_query(self, dj):
        plan = dj.explain(analyze=True)
        assert isinstance(plan, str)
        assert plan.strip() != ""

    def test_explain_rejects_an_unknown_format(self, dj):
        with pytest.raises(ValueError, match="format must be"):
            dj.explain(format="xml")
