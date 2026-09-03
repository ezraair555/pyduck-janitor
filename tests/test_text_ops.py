"""Tests for pyduck_janitor.text_ops (icu + fts extensions).

All tests use a fresh in-memory DuckDB connection so extension
loading state doesn't leak between tests. Extension availability
is probed up-front; tests skip when an extension can't be loaded
(typically on a development machine without internet, or a
platform without that extension built).
"""

from __future__ import annotations

import os
import sys

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyduck_janitor import DuckJanitor
from pyduck_janitor.extensions import (
    EXTENSIONS,
    ExtensionNotAvailable,
    extension_loaded,
    load_extension,
)
from pyduck_janitor.text_ops import (
    build_fts_index,
    drop_fts_index,
    keyword_filter,
    search_text,
    text_normalize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_conn() -> duckdb.DuckDBPyConnection:
    """A fresh in-memory DuckDB connection per test."""
    conn = duckdb.connect(":memory:")
    yield conn
    try:
        conn.close()
    except Exception:
        pass


@pytest.fixture
def small_corpus(fresh_conn: duckdb.DuckDBPyConnection) -> DuckJanitor:
    """A small text corpus for FTS tests."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "text": [
                "the quick brown fox jumps over the lazy dog",
                "a journey of a thousand miles begins with a single step",
                "to be or not to be that is the question",
                "all that glitters is not gold",
                "the fox is quick and brown",
                "lazy dogs are not quick",
                "brown foxes are quick and brown",
            ],
        }
    )
    return DuckJanitor.from_pandas(df)


def _extension_available(fresh_conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    try:
        load_extension(fresh_conn, name, install=False)
        return True
    except ExtensionNotAvailable:
        return False
    except duckdb.Error:
        return False


@pytest.fixture
def icu_available(fresh_conn: duckdb.DuckDBPyConnection) -> bool:
    return _extension_available(fresh_conn, "icu")


@pytest.fixture
def fts_available(fresh_conn: duckdb.DuckDBPyConnection) -> bool:
    return _extension_available(fresh_conn, "fts")


# ---------------------------------------------------------------------------
# icu: text_normalize
# ---------------------------------------------------------------------------


class TestTextNormalize:
    def test_lowercase(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["HELLO World", "Foo Bar"]})
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(dj, "name", strip_accents=False).collect()
        assert out["name"].tolist() == ["hello world", "foo bar"]

    def test_strip_whitespace(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["   hello   ", "\t\nfoo\n"]})
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(dj, "name").collect()
        assert out["name"].tolist() == ["hello", "foo"]

    def test_collapse_whitespace(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["hello   world", "a\tb"]})
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(dj, "name").collect()
        assert out["name"].tolist() == ["hello world", "a b"]

    def test_strip_accents(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["Café", "résumé", "NAÏVE"]})
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(dj, "name").collect()
        # Accents gone, lowercased, whitespace collapsed.
        assert out["name"].tolist() == ["cafe", "resume", "naive"]

    def test_preserves_nulls(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["A", None, pd.NA]})
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(dj, "name").collect()
        assert out.loc[0, "name"] == "a"
        assert pd.isna(out.loc[1, "name"])
        assert pd.isna(out.loc[2, "name"])

    def test_disable_lowercase(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["HELLO"]})
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(
            dj, "name", lower=False, strip_accents=False
        ).collect()
        assert out["name"].tolist() == ["HELLO"]

    def test_target_column(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["HELLO"]})
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(
            dj, "name", target_columns="name_clean", strip_accents=False
        ).collect()
        assert "name" in out.columns
        assert "name_clean" in out.columns
        assert out["name_clean"].tolist() == ["hello"]

    def test_target_column_with_accent_strip(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["Café"]})
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(dj, "name", target_columns="name_clean").collect()
        assert out["name"].tolist() == ["Café"]
        assert out["name_clean"].tolist() == ["cafe"]

    def test_multiple_columns(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame(
            {"a": ["HELLO"], "b": ["WORLD"]}
        )
        dj = DuckJanitor.from_pandas(df)
        out = text_normalize(
            dj, ["a", "b"], strip_accents=False
        ).collect()
        assert out["a"].tolist() == ["hello"]
        assert out["b"].tolist() == ["world"]

    def test_invalid_form_raises(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["x"]})
        dj = DuckJanitor.from_pandas(df)
        with pytest.raises(ValueError, match="form must be one of"):
            text_normalize(dj, "name", form="INVALID")

    def test_form_nfkc_changes_compatibility_chars(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"name": ["ℌ𝔢𝔩𝔩𝔬 ﬁ"]})
        dj = DuckJanitor.from_pandas(df)
        out_nfc = text_normalize(
            dj,
            "name",
            form="NFC",
            strip_accents=False,
            lower=False,
            collapse_whitespace=False,
            strip=False,
        ).collect()
        out_nfkc = text_normalize(
            dj,
            "name",
            form="NFKC",
            strip_accents=False,
            lower=False,
            collapse_whitespace=False,
            strip=False,
        ).collect()
        assert out_nfc.loc[0, "name"] != out_nfkc.loc[0, "name"]
        assert out_nfkc.loc[0, "name"] == "Hello fi"

    def test_mismatched_lengths_raise(self, fresh_conn, icu_available):
        if not icu_available:
            pytest.skip("icu extension not available")
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        dj = DuckJanitor.from_pandas(df)
        with pytest.raises(ValueError, match="same length"):
            text_normalize(
                dj, ["a", "b"], target_columns=["only_one"]
            )


# ---------------------------------------------------------------------------
# fts: build / search / drop
# ---------------------------------------------------------------------------


class TestFTSIndex:
    def test_build_and_search(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        dj = build_fts_index(small_corpus, "text")
        results = search_text(dj, "text", "fox quick", top_k=5)
        assert len(results) >= 1
        assert "score" in results.columns
        # Rows about foxes should rank highest.
        top_texts = results["text"].tolist()
        assert any("fox" in t for t in top_texts[:3])

    def test_search_top_k(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        dj = build_fts_index(small_corpus, "text")
        results = search_text(dj, "text", "fox", top_k=2)
        assert len(results) == 2

    def test_threshold(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        dj = build_fts_index(small_corpus, "text")
        # Threshold of 0 should keep rows with positive scores.
        results = search_text(
            dj, "text", "fox", threshold=0.0
        )
        # At least one match (we know 'fox' rows are in the corpus).
        assert len(results) >= 1
        # A very high threshold drops everything.
        results_high = search_text(
            dj, "text", "fox", threshold=1000.0
        )
        assert len(results_high) == 0

    def test_return_relation(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        dj = build_fts_index(small_corpus, "text")
        rel = search_text(
            dj, "text", "fox", return_relation=True
        )
        assert isinstance(rel, DuckJanitor)

    def test_drop(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        dj = build_fts_index(small_corpus, "text")
        assert hasattr(dj, "_fts_index")
        dj2 = drop_fts_index(dj)
        assert not hasattr(dj2, "_fts_index")
        # Searching again should fail (no index recorded).
        with pytest.raises(ValueError, match="No FTS index"):
            search_text(dj2, "text", "fox")

    def test_search_no_index_raises(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        with pytest.raises(ValueError, match="No FTS index"):
            search_text(small_corpus, "text", "fox")

    def test_query_with_apostrophe(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        dj = build_fts_index(small_corpus, "text")
        # Apostrophes in queries should not crash SQL.
        results = search_text(dj, "text", "fox's quick")
        # No assertion on row count; just no exception.
        assert isinstance(results, pd.DataFrame)


class TestKeywordFilter:
    def test_any(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        out = keyword_filter(small_corpus, "text", ["fox", "gold"])
        rows = out.collect()
        assert any("fox" in t for t in rows["text"])
        assert any("gold" in t for t in rows["text"])

    def test_all(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        out = keyword_filter(
            small_corpus, "text", ["quick", "fox"], mode="all"
        )
        rows = out.collect()
        # Every result must contain both quick and fox.
        for t in rows["text"]:
            assert "quick" in t.lower()
            assert "fox" in t.lower()

    def test_case_sensitive_default_off(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        out = keyword_filter(small_corpus, "text", ["FOX"])
        rows = out.collect()
        # 'FOX' should match 'fox' rows because we lowercase by default.
        assert any("fox" in t.lower() for t in rows["text"])

    def test_case_sensitive_on(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        out = keyword_filter(
            small_corpus, "text", ["FOX"], case_sensitive=True
        )
        rows = out.collect()
        # 'FOX' is uppercase, so no rows match (all our text is lowercase).
        assert len(rows) == 0

    def test_invalid_mode_raises(self, small_corpus, fts_available):
        if not fts_available:
            pytest.skip("fts extension not available")
        with pytest.raises(ValueError, match="mode must be"):
            keyword_filter(small_corpus, "text", ["fox"], mode="bogus")
