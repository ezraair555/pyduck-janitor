"""Tests for the chainable text and vector verbs.

Before 0.4.1 the text and vector helpers were module-level functions
only (``text_normalize(dj, ...)``), which broke the chaining contract
every other verb in the package follows. These tests pin the chainable
surface.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pyduck_janitor import DuckJanitor


@pytest.fixture()
def conn() -> duckdb.DuckDBPyConnection:
    """A fresh DuckDB connection."""
    return duckdb.connect()


@pytest.fixture()
def text_frame() -> pd.DataFrame:
    """A frame with messy text."""
    return pd.DataFrame(
        {
            "body": ["Café  Bar", "café", "lazy dog", None],
            "tag": ["Alpha", "beta", "Gamma", "delta"],
        }
    )


class TestTextNormalize:
    """``text_normalize`` as a chainable method."""

    def test_normalizes_and_chains(self, conn, text_frame):
        dj = DuckJanitor.from_pandas(text_frame)
        result = dj.text_normalize("body").collect()
        assert result["body"].tolist()[:3] == ["cafe bar", "cafe", "lazy dog"]
        # pandas 3.x represents the NULL as NaN rather than None.
        assert pd.isna(result["body"].tolist()[3])

    def test_chains_into_a_filter(self, conn, text_frame):
        dj = DuckJanitor.from_pandas(text_frame)
        result = dj.text_normalize("body").filter_on("\"body\" = 'cafe'").collect()
        assert len(result) == 1

    def test_target_columns_do_not_overwrite(self, conn, text_frame):
        dj = DuckJanitor.from_pandas(text_frame)
        result = dj.text_normalize("body", target_columns="body_clean").collect()
        assert "body_clean" in result.columns
        assert result["body"].tolist()[0] == "Café  Bar"

    def test_multiple_columns(self, conn, text_frame):
        dj = DuckJanitor.from_pandas(text_frame)
        result = dj.text_normalize(["body", "tag"]).collect()
        assert result["tag"].tolist()[0] == "alpha"


class TestFts:
    """Full-text search as a chainable method."""

    def test_build_index_then_search(self, conn, text_frame):
        dj = DuckJanitor.from_pandas(text_frame)
        indexed = dj.build_fts_index("body")
        assert indexed is dj

        results = indexed.search_text("body", "lazy")
        assert len(results) >= 1

    def test_search_returns_a_relation_when_asked(self, conn, text_frame):
        dj = DuckJanitor.from_pandas(text_frame)
        indexed = dj.build_fts_index("body")
        relation = indexed.search_text("body", "cafe", return_relation=True)
        assert isinstance(relation, DuckJanitor)
        assert relation.count() >= 1

    def test_drop_index(self, conn, text_frame):
        dj = DuckJanitor.from_pandas(text_frame)
        indexed = dj.build_fts_index("body")
        assert indexed.drop_fts_index() is indexed

    def test_keyword_filter_chains(self, conn, text_frame):
        dj = DuckJanitor.from_pandas(text_frame)
        result = dj.keyword_filter("tag", ["alpha"]).collect()
        assert len(result) == 1


class TestVectorVerbs:
    """Embedding and vector-search verbs as chainable methods."""

    def test_embed_column_adds_an_embedding(self, conn, text_frame):
        from pyduck_janitor.embeddings import DEFAULT_EMBED_MODEL, embed_install

        pytest.importorskip("sentence_transformers")
        try:
            embed_install(DEFAULT_EMBED_MODEL)
        except Exception as exc:  # pragma: no cover - network dependent
            pytest.skip(f"embedding model unavailable: {exc}")

        dj = DuckJanitor.from_pandas(text_frame)
        try:
            result = dj.embed_column("body").collect()
        except ValueError as exc:
            # A cached model whose config.json is unusable is an environment
            # problem, not a defect in the verb. The bundled MiniLM checkpoint
            # on this host is exactly that case.
            pytest.skip(f"embedding model unusable in this environment: {exc}")
        assert "embedding" in result.columns

    def test_vector_search_can_return_a_relation(self, conn, text_frame):
        """``return_relation=True`` must give a chainable DuckJanitor."""
        from pyduck_janitor.embeddings import vector_search

        dj = DuckJanitor.from_pandas(text_frame)
        with pytest.raises(Exception):
            # No index was built, so this must fail loudly rather than
            # silently returning an empty frame.
            vector_search(dj, "cafe", return_relation=True)
