"""
Tests for the dplyr-style explicit join verbs:
    inner_join, left_join, right_join, full_join, semi_join, anti_join.

Mirrors the upstream ``pyjanitor`` verb tests where the semantics match
(``inner_join`` etc.) and adds coverage for ``anti_join`` /
``semi_join`` which have no pandas equivalent.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pyduck_janitor import (
    DuckJanitor,
    anti_join,
    left_join,
    semi_join,
)


@pytest.fixture
def employees() -> DuckJanitor:
    """Five employees across five departments; dept 50 has no matching dept row."""
    return DuckJanitor.from_pandas(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Carol", "Diana", "Eve"],
                "dept_id": [10, 20, 30, 40, 50],
            }
        )
    )


@pytest.fixture
def departments() -> DuckJanitor:
    """Four departments; dept 50 absent, dept 99 is unmatched."""
    return DuckJanitor.from_pandas(
        pd.DataFrame(
            {
                "dept_id": [10, 20, 30, 40, 99],
                "dept_name": ["Eng", "Sales", "Ops", "Mktg", "Other"],
            }
        )
    )


@pytest.fixture
def discontinued() -> DuckJanitor:
    """Two departments that are being shut down."""
    return DuckJanitor.from_pandas(pd.DataFrame({"dept_id": [30, 50]}))


class TestInnerJoin:
    def test_returns_only_matching_rows(self, employees, departments):
        """Only employees whose dept_id exists in the departments table."""
        result = employees.inner_join(departments, on="dept_id").collect()
        assert sorted(result["id"].tolist()) == [1, 2, 3, 4]
        assert result["dept_name"].isna().sum() == 0

    def test_left_on_and_right_on(self):
        """When keys have different names, both must be supplied."""
        emps = DuckJanitor.from_pandas(pd.DataFrame({"id": [1, 2], "team_code": ["E1", "S2"]}))
        teams = DuckJanitor.from_pandas(
            pd.DataFrame({"code": ["E1", "S2", "X9"], "team_name": ["Eng1", "Sales2", "Other"]})
        )
        result = emps.inner_join(teams, left_on="team_code", right_on="code").collect()
        assert sorted(result["id"].tolist()) == [1, 2]
        assert result["team_name"].tolist() == ["Eng1", "Sales2"]

    def test_rejects_on_and_left_on_together(self, employees, departments):
        """``on`` and ``left_on``/``right_on`` are mutually exclusive."""
        with pytest.raises(ValueError, match="not both"):
            employees.inner_join(departments, on="dept_id", left_on="id", right_on="dept_id")

    def test_raises_on_missing_key(self, employees):
        """An unknown join key produces a clear error."""
        empty = DuckJanitor.from_pandas(pd.DataFrame({"foo": [1]}))
        with pytest.raises(ValueError, match="`on` key 'bar' not found"):
            employees.inner_join(empty, on="bar")


class TestLeftJoin:
    def test_keeps_all_left_rows(self, employees, departments):
        """Eve (dept 50) appears with NULL dept_name."""
        result = employees.left_join(departments, on="dept_id").collect()
        assert sorted(result["id"].tolist()) == [1, 2, 3, 4, 5]
        eve = result[result["id"] == 5].iloc[0]
        assert pd.isna(eve["dept_name"])

    def test_right_only_rows_do_not_appear(self, employees, departments):
        """Unmatched right rows (dept 99) are absent from a left join."""
        result = employees.left_join(departments, on="dept_id").collect()
        assert "Other" not in result["dept_name"].dropna().tolist()


class TestRightJoin:
    def test_keeps_all_right_rows(self, employees, departments):
        """Dept 99 (Other) appears even though no employee has it."""
        result = employees.right_join(departments, on="dept_id").collect()
        assert "Other" in result["dept_name"].dropna().tolist()

    def test_left_only_rows_have_nulls(self, employees, departments):
        """Eve (dept 50) is absent from a right join."""
        result = employees.right_join(departments, on="dept_id").collect()
        assert 5 not in result["id"].dropna().tolist()


class TestFullJoin:
    def test_keeps_all_rows_from_both_sides(self, employees, departments):
        """Eve's row AND dept 99's row both appear, with NULLs on the opposite side."""
        result = employees.full_join(departments, on="dept_id").collect()
        # 5 employees + 1 unmatched department = at least 6 rows
        assert len(result) >= 6
        assert "Other" in result["dept_name"].dropna().tolist()
        eve_rows = result[result["id"] == 5]
        assert not eve_rows.empty
        assert eve_rows["dept_name"].isna().all()


class TestSemiJoin:
    def test_returns_left_rows_with_matching_key(self, employees, departments):
        """Employees whose dept exists in the departments table."""
        result = employees.semi_join(departments, on="dept_id").collect()
        assert sorted(result["id"].tolist()) == [1, 2, 3, 4]

    def test_does_not_add_right_columns(self, employees, departments):
        """semi_join must not pull dept_name through."""
        result = employees.semi_join(departments, on="dept_id").collect()
        assert "dept_name" not in result.columns
        assert set(result.columns) == {"id", "name", "dept_id"}

    def test_no_match_returns_empty(self):
        """When keys don't intersect, the result is empty."""
        emps = DuckJanitor.from_pandas(pd.DataFrame({"id": [1], "dept_id": [999]}))
        depts = DuckJanitor.from_pandas(pd.DataFrame({"dept_id": [1], "dept_name": ["X"]}))
        result = emps.semi_join(depts, on="dept_id").collect()
        assert len(result) == 0


class TestAntiJoin:
    def test_returns_left_rows_with_no_matching_key(self, employees, departments):
        """Eve's dept (50) is not in departments."""
        result = employees.anti_join(departments, on="dept_id").collect()
        assert result["id"].tolist() == [5]

    def test_does_not_add_right_columns(self, employees, departments):
        """anti_join must not pull dept_name through."""
        result = employees.anti_join(departments, on="dept_id").collect()
        assert "dept_name" not in result.columns

    def test_no_left_only_returns_all(self):
        """When every left row has a match, anti_join returns empty."""
        emps = DuckJanitor.from_pandas(pd.DataFrame({"id": [1, 2], "k": [1, 2]}))
        depts = DuckJanitor.from_pandas(pd.DataFrame({"k": [1, 2]}))
        result = emps.anti_join(depts, on="k").collect()
        assert len(result) == 0


class TestSuffixes:
    def test_default_suffixes_applied_on_collision(self):
        """Columns that collide get the right-side suffix."""
        left = DuckJanitor.from_pandas(
            pd.DataFrame({"id": [1, 2], "tag": ["x", "y"], "val": [10, 20]})
        )
        right = DuckJanitor.from_pandas(
            pd.DataFrame({"id": [1, 2], "tag": ["p", "q"], "label": ["apple", "banana"]})
        )
        result = left.inner_join(right, on="id").collect()
        # ``tag`` collides → left wins `tag`, right gets `tag_y`.
        assert "tag" in result.columns
        assert "tag_y" in result.columns
        assert result.loc[result["id"] == 1, "tag"].iloc[0] == "x"
        assert result.loc[result["id"] == 1, "tag_y"].iloc[0] == "p"

    def test_custom_suffixes(self):
        """Custom suffixes are honored."""
        left = DuckJanitor.from_pandas(pd.DataFrame({"id": [1], "tag": ["x"]}))
        right = DuckJanitor.from_pandas(pd.DataFrame({"id": [1], "tag": ["p"]}))
        result = left.inner_join(
            right, on="id", suffixes=("_left_only_placeholder", "_r")
        ).collect()
        assert "tag_r" in result.columns


class TestMethodAndFunctionParity:
    """The module-level functions are equivalent to the methods on DuckJanitor."""

    def test_module_function_matches_method(self, employees, departments):
        left_method = employees.left_join(departments, on="dept_id").collect()
        left_function = left_join(employees, departments, on="dept_id").collect()
        # Both should agree on id order independent of any DuckDB-returned order.
        assert sorted(left_method["id"].tolist()) == sorted(left_function["id"].tolist())

    def test_anti_join_function_parity(self, employees, departments):
        method_result = employees.anti_join(departments, on="dept_id").collect()
        function_result = anti_join(employees, departments, on="dept_id").collect()
        assert method_result["id"].tolist() == function_result["id"].tolist()

    def test_semi_join_function_parity(self, employees, departments):
        method_result = employees.semi_join(departments, on="dept_id").collect()
        function_result = semi_join(employees, departments, on="dept_id").collect()
        assert sorted(method_result["id"].tolist()) == sorted(function_result["id"].tolist())
