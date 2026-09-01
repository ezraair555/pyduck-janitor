"""
Tests for DuckJanitor class.
"""

import pandas as pd
import pytest
from pyduck_janitor import DuckJanitor


class TestDuckJanitor:
    """Tests for the DuckJanitor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "Name": ["Alice", "Bob", "Charlie", "Diana"],
                "Age": [25, 30, 35, 40],
                "City": ["NYC", "LA", "Chicago", "Houston"],
                "Salary": [50000, 60000, 70000, 80000],
            }
        )

    @pytest.fixture
    def data_with_nulls(self):
        """Create sample DataFrame with missing values."""
        return pd.DataFrame(
            {
                "A": [1, 2, None, 4, 5],
                "B": [None, 2, 3, 4, 5],
                "C": [1, 2, 3, None, 5],
            }
        )

    def test_from_pandas(self, sample_data):
        """Test creating DuckJanitor from pandas DataFrame."""
        dj = DuckJanitor.from_pandas(sample_data)

        assert isinstance(dj, DuckJanitor)
        assert len(dj.head()) == len(sample_data)

    def test_from_excel(self, tmp_path):
        """Test creating DuckJanitor from an Excel file."""
        df = pd.DataFrame(
            {
                "Name": ["Alice", "Bob", "Charlie"],
                "Age": [25, 30, 35],
                "Score": [88.5, 92.0, 76.3],
            }
        )
        xlsx_path = tmp_path / "test_data.xlsx"
        df.to_excel(xlsx_path, index=False, engine="openpyxl")

        dj = DuckJanitor.from_excel(xlsx_path)
        assert isinstance(dj, DuckJanitor)
        result = dj.collect()
        assert result.shape == df.shape
        assert list(result.columns) == ["Name", "Age", "Score"]
        assert result["Age"].tolist() == [25, 30, 35]

    def test_from_excel_sheet_name(self, tmp_path):
        """Test from_excel with a specific sheet name."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = pd.DataFrame({"X": ["a", "b"], "Y": ["c", "d"]})
        xlsx_path = tmp_path / "multi_sheet.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df1.to_excel(writer, sheet_name="first", index=False)
            df2.to_excel(writer, sheet_name="second", index=False)

        dj = DuckJanitor.from_excel(xlsx_path, sheet_name="second")
        result = dj.collect()
        assert list(result.columns) == ["X", "Y"]
        assert len(result) == 2

    def test_from_excel_usecols(self, tmp_path):
        """Test from_excel with usecols to select specific columns."""
        df = pd.DataFrame(
            {
                "Name": ["Alice", "Bob", "Charlie", "Diana"],
                "Age": [25, 30, 35, 40],
                "City": ["NYC", "LA", "Chicago", "Houston"],
                "Salary": [50000, 60000, 70000, 80000],
            }
        )
        xlsx_path = tmp_path / "cols_test.xlsx"
        df.to_excel(xlsx_path, index=False, engine="openpyxl")

        dj = DuckJanitor.from_excel(xlsx_path, usecols=["Name", "Salary"])
        result = dj.collect()
        assert list(result.columns) == ["Name", "Salary"]
        assert len(result) == 4
        assert result["Name"].tolist() == ["Alice", "Bob", "Charlie", "Diana"]

    def test_from_excel_skiprows(self, tmp_path):
        """Test from_excel with header offset to skip rows."""
        df = pd.DataFrame(
            {
                "Name": ["Alice", "Bob", "Charlie", "Diana"],
                "Age": [25, 30, 35, 40],
            }
        )
        xlsx_path = tmp_path / "skip_test.xlsx"
        df.to_excel(xlsx_path, index=False, engine="openpyxl")

        # header=1 means: use the second row as column names, skip row 0
        dj = DuckJanitor.from_excel(xlsx_path, header=1)
        result = dj.collect()
        assert len(result) == 3  # 4 rows - 1 used as header
        assert list(result.columns) == ["Alice", "25"]
        assert result["Alice"].tolist() == ["Bob", "Charlie", "Diana"]

    def test_from_excel_all_sheets(self, tmp_path):
        """Test from_excel with sheet_name=None picks first sheet."""
        df1 = pd.DataFrame({"A": [1, 2]})
        df2 = pd.DataFrame({"Z": [9, 8]})
        xlsx_path = tmp_path / "all_sheets.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df1.to_excel(writer, sheet_name="alpha", index=False)
            df2.to_excel(writer, sheet_name="beta", index=False)

        dj = DuckJanitor.from_excel(xlsx_path, sheet_name=None)
        result = dj.collect()
        # Should pick the first sheet deterministically
        assert "A" in result.columns
        assert len(result) == 2

    def test_from_json(self, tmp_path):
        """Test creating DuckJanitor from a JSON file."""
        import json as _json

        data = [
            {"Name": "Alice", "Age": 25, "Score": 88.5},
            {"Name": "Bob", "Age": 30, "Score": 92.0},
            {"Name": "Charlie", "Age": 35, "Score": 76.3},
        ]
        json_path = tmp_path / "test_data.json"
        with open(json_path, "w") as f:
            _json.dump(data, f)

        dj = DuckJanitor.from_json(json_path)
        assert isinstance(dj, DuckJanitor)
        result = dj.collect()
        assert result.shape == (3, 3)
        assert list(result.columns) == ["Name", "Age", "Score"]
        assert result["Age"].tolist() == [25, 30, 35]

    def test_from_json_ndjson(self, tmp_path):
        """Test creating DuckJanitor from NDJSON (JSONL) file."""
        import json as _json

        records = [
            {"event": "login", "user": "alice", "ts": "2026-01-01"},
            {"event": "logout", "user": "alice", "ts": "2026-01-02"},
            {"event": "login", "user": "bob", "ts": "2026-01-03"},
        ]
        jsonl_path = tmp_path / "events.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(_json.dumps(rec) + "\n")

        dj = DuckJanitor.from_json(jsonl_path)
        result = dj.collect()
        assert len(result) == 3
        assert list(result.columns) == ["event", "user", "ts"]
        assert result["event"].tolist() == ["login", "logout", "login"]

    def test_from_json_glob(self, tmp_path):
        """Test from_json with glob pattern to read multiple files."""
        import json as _json

        with open(tmp_path / "part1.json", "w") as f:
            _json.dump([{"a": 1, "b": "x"}], f)
        with open(tmp_path / "part2.json", "w") as f:
            _json.dump([{"a": 2, "b": "y"}], f)

        dj = DuckJanitor.from_json(str(tmp_path / "*.json"))
        result = dj.collect()
        assert len(result) == 2
        assert set(result.columns) == {"a", "b"}

    def test_from_json_file_list(self, tmp_path):
        """Test from_json with explicit list of file paths."""
        import json as _json

        f1 = tmp_path / "f1.json"
        f2 = tmp_path / "f2.json"
        with open(f1, "w") as f:
            _json.dump([{"a": 1}], f)
        with open(f2, "w") as f:
            _json.dump([{"a": 2}], f)

        dj = DuckJanitor.from_json([str(f1), str(f2)])
        result = dj.collect()
        assert len(result) == 2
        assert result["a"].tolist() == [1, 2]

    def test_from_json_gzipped(self, tmp_path):
        """Test from_json with gzip-compressed JSON."""
        import gzip
        import json as _json

        data = [{"x": 10, "y": "a"}, {"x": 20, "y": "b"}]
        gz_path = tmp_path / "data.json.gz"
        with gzip.open(gz_path, "wt") as f:
            _json.dump(data, f)

        dj = DuckJanitor.from_json(gz_path)
        result = dj.collect()
        assert len(result) == 2
        assert result["x"].tolist() == [10, 20]

    def test_from_json_format_explicit(self, tmp_path):
        """Test from_json with explicit format='array'."""
        import json as _json

        data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        json_path = tmp_path / "explicit.json"
        with open(json_path, "w") as f:
            _json.dump(data, f)

        dj = DuckJanitor.from_json(json_path, format="array")
        result = dj.collect()
        assert len(result) == 2
        assert result["a"].tolist() == [1, 2]

    def test_collect(self, sample_data):
        """Test collecting results back to pandas."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.collect()

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data.shape

    def test_head(self, sample_data):
        """Test head method."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.head(3)

        assert len(result) == 3

    def test_clean_names(self, sample_data):
        """Test cleaning column names."""
        # Create data with messy column names
        messy = sample_data.copy()
        messy.columns = ["  First Name  ", "Age-Years", "CITY!", "Annual Salary"]

        dj = DuckJanitor.from_pandas(messy)
        cleaned = dj.clean_names()
        result = cleaned.collect()

        expected_cols = ["first_name", "age_years", "city", "annual_salary"]
        assert list(result.columns) == expected_cols

    def test_remove_columns(self, sample_data):
        """Test removing columns."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.remove_columns(["City", "Salary"]).collect()

        assert "City" not in result.columns
        assert "Salary" not in result.columns
        assert "Name" in result.columns
        assert "Age" in result.columns

    def test_rename_column(self, sample_data):
        """Test renaming a column."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.rename_column("Name", "Full_Name").collect()

        assert "Full_Name" in result.columns
        assert "Name" not in result.columns

    def test_add_column_scalar(self, sample_data):
        """Test adding a column with a scalar value."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.add_column("Constant", 42).collect()

        assert "Constant" in result.columns
        assert (result["Constant"] == 42).all()

    def test_add_column_expression(self, sample_data):
        """Test adding a column with a SQL expression."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.add_column("Double_Age", "Age * 2").collect()

        assert "Double_Age" in result.columns
        assert (result["Double_Age"] == sample_data["Age"] * 2).all()

    def test_dropna(self, data_with_nulls):
        """Test dropping rows with missing values."""
        dj = DuckJanitor.from_pandas(data_with_nulls)
        result = dj.dropna().collect()

        assert result.isnull().sum().sum() == 0

    def test_dropna_subset(self, data_with_nulls):
        """Test dropping rows with missing values in subset."""
        dj = DuckJanitor.from_pandas(data_with_nulls)
        result = dj.dropna(subset=["A"]).collect()

        assert result["A"].isnull().sum() == 0

    def test_remove_empty(self, data_with_nulls):
        """Test removing empty rows and columns."""
        dj = DuckJanitor.from_pandas(data_with_nulls)
        result = dj.remove_empty().collect()

        assert len(result) > 0

    def test_filter_column_callable(self, sample_data):
        """Test filtering with a callable."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.filter_column("Age", lambda x: x > 30).collect()

        assert (result["Age"] > 30).all()

    def test_filter_column_sql(self, sample_data):
        """Test filtering with SQL string."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.filter_column("Age", "Age > 30").collect()

        assert (result["Age"] > 30).all()

    def test_coalesce(self, data_with_nulls):
        """Test coalescing columns."""
        dj = DuckJanitor.from_pandas(data_with_nulls)
        result = dj.coalesce(["A", "B", "C"], "coalesced").collect()

        assert "coalesced" in result.columns
        assert result["coalesced"].isnull().sum() == 0

    def test_encode_categorical(self, sample_data):
        """Test encoding categorical column."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.encode_categorical("City").collect()

        assert "City_cat" in result.columns

    def test_get_dummies(self, sample_data):
        """Test one-hot encoding."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.get_dummies("City", prefix="city").collect()

        # Should have dummy columns for each city
        assert any(col.startswith("city_") for col in result.columns)

    def test_sql(self, sample_data):
        """Test custom SQL query."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = dj.sql("SELECT * FROM self WHERE Age > 30").collect()

        assert (result["Age"] > 30).all()

    def test_method_chaining(self, sample_data):
        """Test method chaining."""
        dj = DuckJanitor.from_pandas(sample_data)
        result = (
            dj.clean_names()
            .remove_columns(["city"])
            .rename_column("name", "full_name")
            .add_column("double_salary", "salary * 2")
            .filter_column("age", "age > 25")
            .collect()
        )

        assert "full_name" in result.columns
        assert "city" not in result.columns
        assert "double_salary" in result.columns
        assert (result["age"] > 25).all()

    def test_explain(self, sample_data):
        """Test query plan explanation."""
        dj = DuckJanitor.from_pandas(sample_data)
        explanation = dj.explain()

        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_repr(self, sample_data):
        """Test string representation."""
        dj = DuckJanitor.from_pandas(sample_data)
        repr_str = repr(dj)

        assert "DuckJanitor" in repr_str
        assert "lazy" in repr_str.lower()

    # Regression tests for P0 fixes

    def test_remove_empty_rows_and_columns(self):
        """remove_empty should drop all-empty columns and all-empty rows."""
        df = pd.DataFrame(
            {
                "A": [1, None, None],
                "B": [None, None, None],
                "C": ["", "x", ""],
            }
        )
        result = DuckJanitor.from_pandas(df).remove_empty().collect()

        assert "B" not in result.columns
        assert len(result) == 2

    def test_dropna_how_all(self):
        """dropna(how='all') should keep rows where not all checked columns are null."""
        df = pd.DataFrame(
            {
                "A": [None, 1, None, 4],
                "B": [None, None, 2, 5],
            }
        )
        result = DuckJanitor.from_pandas(df).dropna(how="all").collect()

        assert len(result) == 3

    def test_select_rows_by_index(self):
        """select_rows should pick rows by 0-based index."""
        df = pd.DataFrame({"A": [10, 20, 30, 40]})
        result = DuckJanitor.from_pandas(df).select_rows([0, 2]).collect()

        assert list(result["A"]) == [10, 30]

    def test_case_when(self):
        """case_when should build a conditional column."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = (
            DuckJanitor.from_pandas(df)
            .case_when([("A > 1", "high")], "case_col", default="low")
            .collect()
        )

        assert list(result["case_col"]) == ["low", "high", "high"]

    def test_currency_column_to_numeric(self):
        """currency_column_to_numeric should strip currency symbols and commas."""
        df = pd.DataFrame({"price": ["$1,234.56", "$2,000.00"]})
        result = (
            DuckJanitor.from_pandas(df).currency_column_to_numeric("price", "price_num").collect()
        )

        assert list(result["price_num"]) == pytest.approx([1234.56, 2000.0])

    def test_convert_date(self):
        """convert_date should parse string dates."""
        df = pd.DataFrame({"dt": ["2023-01-15", "2023-02-20"]})
        result = DuckJanitor.from_pandas(df).convert_date("dt", "dt_parsed").collect()

        assert result["dt_parsed"].notna().all()

    def test_impute(self):
        """impute should fill nulls with the requested statistic."""
        df = pd.DataFrame({"A": [1, None, 3], "B": [10, 20, 30]})
        result = DuckJanitor.from_pandas(df).impute("A", statistic="mean").collect()

        assert list(result["A"]) == pytest.approx([1.0, 2.0, 3.0])

    def test_conditional_join(self):
        """conditional_join should use a single shared connection."""
        left = DuckJanitor.from_pandas(pd.DataFrame({"A": [1, 2, 3]}))
        right = DuckJanitor.from_pandas(pd.DataFrame({"D": [2, 3, 4]}))
        result = left.conditional_join(right, [("A", "D", ">")]).collect()

        assert len(result) == 1
        assert result["A"].iloc[0] == 3 and result["D"].iloc[0] == 2

    def test_min_max_scale_identical_values(self):
        """min_max_scale should not return all nulls when every value is identical."""
        df = pd.DataFrame({"A": [5, 5, 5]})
        result = DuckJanitor.from_pandas(df).min_max_scale("A", "scaled").collect()

        assert list(result["scaled"]) == pytest.approx([0.0, 0.0, 0.0])

    def test_init_rejects_mismatched_connection(self):
        """DuckJanitor should reject a relation from a different connection."""
        import duckdb

        rel = duckdb.connect().from_df(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError):
            DuckJanitor(rel, connection=duckdb.connect())

    def test_add_column_string_literal(self):
        """add_column should correctly treat a string scalar as a literal rather than a SQL expression."""
        df = pd.DataFrame({"A": [1, 2]})
        result = DuckJanitor.from_pandas(df).add_column("country", "USA").collect()
        assert list(result["country"]) == ["USA", "USA"]

    def test_filter_column_scalar_comparison(self):
        """filter_column should correctly fallback to equality comparison for string scalars."""
        df = pd.DataFrame({"city": ["NYC", "LA"]})
        result = DuckJanitor.from_pandas(df).filter_column("city", "NYC").collect()
        assert list(result["city"]) == ["NYC"]

    def test_impute_group_by(self):
        """impute should respect the group_by parameter."""
        df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [1.0, None, 10.0, None]})
        result = (
            DuckJanitor.from_pandas(df).impute("val", statistic="mean", group_by="cat").collect()
        )
        result = result.sort_values("cat").reset_index(drop=True)
        # Val for cat A should be imputed with 1.0, cat B with 10.0
        assert list(result["val"]) == pytest.approx([1.0, 1.0, 10.0, 10.0])

    def test_fill_empty_non_string(self):
        """fill_empty on a non-string column should return the relation as-is without crashing."""
        df = pd.DataFrame({"A": [1, 2]})
        result = DuckJanitor.from_pandas(df).fill_empty("A", "0").collect()
        assert list(result["A"]) == [1, 2]

    def test_currency_column_to_numeric_empty_string(self):
        """currency_column_to_numeric should return NULL for empty/invalid strings instead of crashing."""
        df = pd.DataFrame({"price": ["", "$", "$10.00"]})
        result = DuckJanitor.from_pandas(df).currency_column_to_numeric("price").collect()
        assert pd.isna(result["price"].iloc[0])
        assert pd.isna(result["price"].iloc[1])
        assert result["price"].iloc[2] == 10.0

    def test_clean_names_arbitrary_duplicates(self):
        """clean_names should correctly append unique suffixes when 3+ columns map to the same name."""
        df = pd.DataFrame([[1, 2, 3]], columns=["A!", "A@", "A#"])
        result = DuckJanitor.from_pandas(df).clean_names().collect()
        assert list(result.columns) == ["a", "a_dup", "a_dup_1"]

    def test_coalesce_collision(self):
        """coalesce should replace target column and not duplicate column name if target_column already exists."""
        df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "C"])
        result = DuckJanitor.from_pandas(df).coalesce(["A", "B"], "C").collect()
        assert list(result.columns) == ["C"]
        assert result["C"].iloc[0] == 1

    def test_complete_pure_sql(self):
        """complete should run in pure SQL and produce correct Cartesian combinations."""
        df = pd.DataFrame({"id": [1, 2], "cat": ["A", "B"], "val": [10, 20]})
        result = DuckJanitor.from_pandas(df).complete(["id", "cat"], fill_value=0).collect()
        assert len(result) == 4
        # Assert values are filled correctly
        assert result.loc[(result["id"] == 1) & (result["cat"] == "B"), "val"].iloc[0] == 0

    def test_alias_pure_sql(self):
        """alias should rename all columns using callable or string in pure SQL."""
        df = pd.DataFrame({"colA": [1], "colB": [2]})
        dj = DuckJanitor.from_pandas(df)
        result1 = dj.alias(lambda x: x.lower().replace("col", "x_")).collect()
        assert list(result1.columns) == ["x_a", "x_b"]
        result2 = dj.alias("new").collect()
        assert list(result2.columns) == ["new_0", "new_1"]

    def test_drop_duplicate_columns_pure_sql(self):
        """drop_duplicate_columns should run in pure SQL and drop exact duplicate columns."""
        df = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [3, 4]})
        result = DuckJanitor.from_pandas(df).drop_duplicate_columns().collect()
        assert list(result.columns) == ["A", "C"]

    def test_compare_df_cols(self):
        """compare_df_cols should compare column sets correctly."""
        dj1 = DuckJanitor.from_pandas(pd.DataFrame({"A": [1], "B": [2]}))
        dj2 = DuckJanitor.from_pandas(pd.DataFrame({"A": [1], "C": [3]}))
        result = dj1.compare_df_cols(dj2)
        assert len(result) == 1
        assert ("B", "BIGINT") in result["only_in_dj1"].iloc[0] or ("B", "int64") in result[
            "only_in_dj1"
        ].iloc[0]
        assert ("C", "BIGINT") in result["only_in_dj2"].iloc[0] or ("C", "int64") in result[
            "only_in_dj2"
        ].iloc[0]

    def test_join_apply(self):
        """join_apply should perform a join and apply a Python function row-wise."""
        dj1 = DuckJanitor.from_pandas(pd.DataFrame({"id": [1], "val1": [10]}))
        dj2 = DuckJanitor.from_pandas(pd.DataFrame({"id": [1], "val2": [20]}))
        result = dj1.join_apply(
            dj2, on="id", func=lambda row: row["val1"] + row["val2"], new_column_name="sum"
        ).collect()
        assert result["sum"].iloc[0] == 30

    def test_process_text(self):
        """process_text should apply string SQL expression or Python callable."""
        dj = DuckJanitor.from_pandas(pd.DataFrame({"name": ["alice", "bob"]}))
        result1 = dj.process_text("name", "upper(name)", "upper_name").collect()
        assert list(result1["upper_name"]) == ["ALICE", "BOB"]
        result2 = dj.process_text("name", lambda x: x[::-1], "rev_name").collect()
        assert list(result2["rev_name"]) == ["ecila", "bob"]

    def test_convert_date_invalid(self):
        """convert_date should return NULL for unparseable dates instead of crashing."""
        df = pd.DataFrame({"dt": ["invalid-date", "2023-01-15"]})
        result = DuckJanitor.from_pandas(df).convert_date("dt", "dt_parsed").collect()
        assert pd.isna(result["dt_parsed"].iloc[0])
        assert not pd.isna(result["dt_parsed"].iloc[1])

    def test_from_parquet_and_csv(self, tmp_path):
        """Test from_parquet and from_csv methods with shared connection."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        csv_file = tmp_path / "test.csv"
        df.to_csv(csv_file, index=False)

        parquet_file = tmp_path / "test.parquet"
        df.to_parquet(parquet_file, index=False)

        dj_csv = DuckJanitor.from_csv(csv_file)
        assert len(dj_csv.collect()) == 3

        dj_pq = DuckJanitor.from_parquet(parquet_file)
        assert len(dj_pq.collect()) == 3

    def test_bin_numeric(self):
        """bin_numeric should group numeric values into bins."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        # uniform strategy
        result = (
            DuckJanitor.from_pandas(df)
            .bin_numeric("A", "A_binned", bins=2, strategy="uniform")
            .collect()
        )
        assert "A_binned" in result.columns

        # quantile strategy
        result2 = (
            DuckJanitor.from_pandas(df)
            .bin_numeric("A", "A_binned", bins=2, strategy="quantile")
            .collect()
        )
        assert "A_binned" in result2.columns

        # custom edges
        result3 = DuckJanitor.from_pandas(df).bin_numeric("A", "A_binned", bins=[0, 3, 6]).collect()
        assert "A_binned" in result3.columns

    def test_change_type(self):
        """change_type should convert column types."""
        df = pd.DataFrame({"A": [1, 2]})
        result = DuckJanitor.from_pandas(df).change_type("A", "VARCHAR").collect()
        assert str(result["A"].dtype) in ("object", "string", "str")

    def test_concatenate_columns(self):
        """concatenate_columns should concatenate specified columns."""
        df = pd.DataFrame({"A": ["hello"], "B": ["world"]})
        result = DuckJanitor.from_pandas(df).concatenate_columns(["A", "B"], sep="-").collect()
        assert result["concatenated"].iloc[0] == "hello-world"

    def test_deconcatenate_column(self):
        """deconcatenate_column should split column values into multiple columns."""
        df = pd.DataFrame({"A": ["hello-world"]})
        result = DuckJanitor.from_pandas(df).deconcatenate_column("A", "-", ["B", "C"]).collect()
        assert result["B"].iloc[0] == "hello"
        assert result["C"].iloc[0] == "world"

    def test_drop_constant_columns(self):
        """drop_constant_columns should drop constant columns."""
        df = pd.DataFrame({"A": [1, 1, 1], "B": [1, 2, 3]})
        result = DuckJanitor.from_pandas(df).drop_constant_columns().collect()
        assert "A" not in result.columns
        assert "B" in result.columns

    def test_fill_directions(self):
        """fill should support forward and backward fill directions."""
        df = pd.DataFrame({"A": [1, None, 3]})
        # forward
        result_f = DuckJanitor.from_pandas(df).fill("A", direction="forward").collect()
        assert list(result_f["A"]) == [1.0, 1.0, 3.0]
        # backward
        result_b = DuckJanitor.from_pandas(df).fill("A", direction="backward").collect()
        assert list(result_b["A"]) == [1.0, 3.0, 3.0]
        # value
        result_v = DuckJanitor.from_pandas(df).fill("A", value=4, direction="value").collect()
        assert list(result_v["A"]) == [1.0, 4.0, 3.0]

    def test_flag_nulls(self):
        """flag_nulls should flag null columns with a binary indicator."""
        df = pd.DataFrame({"A": [1, None, 3]})
        result = DuckJanitor.from_pandas(df).flag_nulls("A").collect()
        assert "is_null_A" in result.columns
        assert list(result["is_null_A"]) == [0, 1, 0]

    def test_limit_column_characters(self):
        """limit_column_characters should limit column characters."""
        df = pd.DataFrame({"A": ["extremelylongstring"]})
        result = DuckJanitor.from_pandas(df).limit_column_characters("A", 8, suffix="...").collect()
        assert result["A"].iloc[0] == "extre..."

    def test_groupby_agg(self):
        """groupby_agg should support complex grouping and aggregates."""
        df = pd.DataFrame({"g": ["A", "A", "B"], "v": [10, 20, 30]})
        result = DuckJanitor.from_pandas(df).groupby_agg("g", {"v": "sum"}).collect()
        result = result.sort_values("g").reset_index(drop=True)
        assert list(result["v_sum"]) == [30, 30]

    def test_groupby_topk(self):
        """groupby_topk should return top k rows in each group."""
        df = pd.DataFrame({"g": ["A", "A", "A", "B", "B"], "v": [1, 2, 3, 10, 20]})
        result = DuckJanitor.from_pandas(df).groupby_topk("g", "v", k=2, ascending=False).collect()
        assert len(result) == 4

    def test_truncate_datetime(self):
        """truncate_datetime should truncate datetime values."""
        df = pd.DataFrame({"dt": ["2023-05-15 12:30:45"]})
        dj = DuckJanitor.from_pandas(df).change_type("dt", "TIMESTAMP")
        result = dj.truncate_datetime("dt", "day").collect()
        assert str(result["dt"].iloc[0]).split()[0] == "2023-05-15"

    def test_pivot_wider_and_longer(self):
        """pivot_wider and pivot_longer should widen and lengthen datasets."""
        df = pd.DataFrame(
            {"id": [1, 1, 2, 2], "var": ["A", "B", "A", "B"], "val": [10, 20, 30, 40]}
        )
        # wider
        wider = DuckJanitor.from_pandas(df).pivot_wider("id", "var", "val").collect()
        assert "A" in wider.columns
        assert "B" in wider.columns
        # longer
        longer = (
            DuckJanitor(DuckJanitor.get_shared_connection().from_df(wider))
            .pivot_longer(["A", "B"], names_to="var", values_to="val")
            .collect()
        )
        assert "var" in longer.columns
        assert "val" in longer.columns

    def test_flag_nulls_errors(self):
        """flag_nulls should raise ValueError for invalid inputs."""
        df = pd.DataFrame({"A": [1]})
        dj = DuckJanitor.from_pandas(df)
        with pytest.raises(ValueError):
            dj.change_type("B", "INT")

    def test_get_dupes(self):
        """get_dupes should find duplicate rows."""
        df = pd.DataFrame({"A": [1, 2, 1], "B": [2, 3, 2]})
        result = DuckJanitor.from_pandas(df).get_dupes().collect()
        assert len(result) == 2

    def test_dropnotnull(self):
        """dropnotnull should drop non-null rows."""
        df = pd.DataFrame({"A": [1, None, 3]})
        result = DuckJanitor.from_pandas(df).dropnotnull().collect()
        assert len(result) == 1
        assert pd.isna(result["A"].iloc[0])

    def test_expand_column(self):
        """expand_column should expand delimited columns to dummies."""
        df = pd.DataFrame({"A": ["x|y", "x", None]})
        result = DuckJanitor.from_pandas(df).expand_column("A", sep="|").collect()
        assert "A_x" in result.columns
        assert "A_y" in result.columns

    def test_jitter(self):
        """jitter should add random noise to numeric columns."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        result = DuckJanitor.from_pandas(df).jitter("A", "A_jit", scale=0.1, seed=42).collect()
        assert "A_jit" in result.columns
        assert not (result["A"] == result["A_jit"]).all()

    def test_label_encode(self):
        """label_encode should encode categorical columns with integers."""
        df = pd.DataFrame({"A": ["cat", "dog", "cat"]})
        result = DuckJanitor.from_pandas(df).label_encode("A").collect()
        assert "A_encoded" in result.columns

    def test_find_replace(self):
        """find_replace should replace values in columns."""
        df = pd.DataFrame({"A": ["cat", "dog"]})
        result = (
            DuckJanitor.from_pandas(df)
            .find_replace("A", {"cat": "feline", "dog": "canine"})
            .collect()
        )
        assert list(result["A"]) == ["feline", "canine"]

    def test_count_cumulative_unique(self):
        """count_cumulative_unique should return running count of unique values."""
        df = pd.DataFrame({"A": ["x", "y", "x"]})
        result = DuckJanitor.from_pandas(df).count_cumulative_unique("A").collect()
        assert "cumulative_unique" in result.columns

    def test_also_and_mutate(self):
        """also and mutate should work as convenience wrappers."""
        df = pd.DataFrame({"A": [1, 2]})
        dj = DuckJanitor.from_pandas(df)

        # also
        side_effect = []

        def my_side_effect(d):
            side_effect.append(len(d))

        dj.also(my_side_effect)
        assert side_effect == [2]

        # mutate
        result = dj.mutate(B="A * 10").collect()
        assert list(result["B"]) == [10, 20]

    def test_sql_literal_exceptions_and_edge_cases(self):
        """Test _sql_literal and _register_relation edge cases/exceptions."""
        import pytest
        from pyduck_janitor.cleaning_ops import _register_relation, _sql_literal

        assert _sql_literal(None) == "NULL"
        assert _sql_literal(True) == "TRUE"
        assert _sql_literal(False) == "FALSE"

        # Test ValueError in _register_relation when conn is None
        df = pd.DataFrame({"a": [1]})
        rel = DuckJanitor.get_shared_connection().from_df(df)
        with pytest.raises(ValueError):
            _register_relation(None, rel)

    def test_clean_names_upper(self):
        """clean_names with case_type='upper' should upper-case names."""
        df = pd.DataFrame({"a_col": [1]})
        result = DuckJanitor.from_pandas(df).clean_names(case_type="upper").collect()
        assert list(result.columns) == ["A_COL"]

    def test_remove_columns_edge_cases(self):
        """remove_columns single string and error when removing all."""
        df = pd.DataFrame({"A": [1], "B": [2]})
        # Single string column
        result = DuckJanitor.from_pandas(df).remove_columns("A").collect()
        assert list(result.columns) == ["B"]
        # Error when removing all
        with pytest.raises(ValueError):
            DuckJanitor.from_pandas(df).remove_columns(["A", "B"]).collect()

    def test_add_column_list_values(self):
        """add_column when passing list values."""
        df = pd.DataFrame({"A": [1, 2]})
        result1 = DuckJanitor.from_pandas(df).add_column("B", [10, 20]).collect()
        assert list(result1["B"]) == [10, 20]
        # values length shorter than df
        result2 = DuckJanitor.from_pandas(df).add_column("B", [10], fill_value=99).collect()
        assert list(result2["B"]) == [10, 99]

    def test_rename_column_not_found(self):
        """rename_column should raise ValueError if column not found."""
        df = pd.DataFrame({"A": [1]})
        with pytest.raises(ValueError):
            DuckJanitor.from_pandas(df).rename_column("B", "C").collect()

    def test_dropna_subset_single_string_and_invalid_how(self):
        """dropna single string subset and invalid how value."""
        df = pd.DataFrame({"A": [1, None], "B": [3, 4]})
        result = DuckJanitor.from_pandas(df).dropna(subset="A").collect()
        assert len(result) == 1
        with pytest.raises(ValueError):
            DuckJanitor.from_pandas(df).dropna(how="invalid").collect()

    def test_remove_empty_all_columns_empty(self):
        """remove_empty should raise ValueError if all columns are empty."""
        df = pd.DataFrame({"A": [None, None]})
        with pytest.raises(ValueError):
            DuckJanitor.from_pandas(df).remove_empty().collect()

    def test_filter_column_callable_no_conn(self):
        """filter_column with callable and no connection should raise ValueError."""
        from pyduck_janitor.cleaning_ops import filter_column as _filter_column

        df = pd.DataFrame({"A": [1]})
        rel = DuckJanitor.get_shared_connection().from_df(df)
        with pytest.raises(ValueError):
            _filter_column(rel, "A", lambda x: True, conn=None)

    def test_get_dummies_no_prefix(self):
        """get_dummies should work when prefix is None."""
        df = pd.DataFrame({"A": ["x", "y"]})
        result = DuckJanitor.from_pandas(df).get_dummies("A", prefix=None).collect()
        assert any("A_" in col for col in result.columns)

    def test_select_columns(self):
        """select_columns should select specific columns (supports string or list)."""
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        dj = DuckJanitor.from_pandas(df)
        assert list(dj.select_columns("A").collect().columns) == ["A"]
        assert list(dj.select_columns(["A", "C"]).collect().columns) == ["A", "C"]

    def test_select_rows_criteria_and_no_args(self):
        """select_rows with criteria or no arguments."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        dj = DuckJanitor.from_pandas(df)
        assert len(dj.select_rows(indices="A > 1").collect()) == 2
        assert len(dj.select_rows(criteria="A = 3").collect()) == 1
        assert len(dj.select_rows().collect()) == 3

    def test_transform_column(self):
        """transform_column SQL expression and callable."""
        df = pd.DataFrame({"A": [1, 2]})
        dj = DuckJanitor.from_pandas(df)
        assert list(dj.transform_column("A", "A + 10").collect()["A"]) == [11, 12]
        assert list(dj.transform_column("A", "A + 10", "B").collect()["B"]) == [11, 12]
        assert list(dj.transform_column("A", lambda x: x * 2).collect()["A"]) == [2, 4]

    def test_transform_columns(self):
        """transform_columns with single column string or target mapping."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        dj = DuckJanitor.from_pandas(df)
        result1 = dj.transform_columns("A", "A + 5").collect()
        assert list(result1["A"]) == [6, 7]
        result2 = dj.transform_columns(["A", "B"], "A + B", ["C", "D"]).collect()
        assert "C" in result2.columns

    def test_filter_on_complement(self):
        """filter_on should support complement parameter."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        dj = DuckJanitor.from_pandas(df)
        assert len(dj.filter_on("A > 1", complement=False).collect()) == 2
        assert len(dj.filter_on("A > 1", complement=True).collect()) == 1

    def test_filter_string_options(self):
        """filter_string should support regex, case sensitivity, and complement."""
        df = pd.DataFrame({"name": ["Alice", "bob", "Charlie"]})
        dj = DuckJanitor.from_pandas(df)
        assert len(dj.filter_string("name", "^A", complement=True, regex=True).collect()) == 2
        assert (
            len(dj.filter_string("name", "ice", complement=False, regex=False, case=True).collect())
            == 1
        )
        assert (
            len(
                dj.filter_string(
                    "name", "alice", complement=False, regex=False, case=False
                ).collect()
            )
            == 1
        )
        assert (
            len(
                dj.filter_string(
                    "name", "alice", complement=True, regex=False, case=False
                ).collect()
            )
            == 2
        )

    def test_constructor_invalid_relation_connection_fallback(self):
        """Constructor should raise ValueError when connection derivation fails."""
        import duckdb

        other_conn = duckdb.connect()
        rel = other_conn.from_df(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError):
            DuckJanitor(rel, connection=None)

    def test_from_parquet_list_of_paths(self, tmp_path):
        """from_parquet should accept a list of paths."""
        df1 = pd.DataFrame({"A": [1]})
        df2 = pd.DataFrame({"A": [2]})

        file1 = tmp_path / "1.parquet"
        file2 = tmp_path / "2.parquet"
        df1.to_parquet(file1, index=False)
        df2.to_parquet(file2, index=False)

        dj = DuckJanitor.from_parquet([file1, file2])
        assert len(dj.collect()) == 2

    def test_from_sql(self):
        """from_sql should initialize DuckJanitor directly from a SQL query."""
        dj = DuckJanitor.from_sql("SELECT 42 AS answer")
        assert dj.collect()["answer"].iloc[0] == 42
