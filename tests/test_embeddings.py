"""Tests for pyduck_janitor.embeddings.

These tests focus on the cache management surface (install / list /
remove) and the vss operators. The actual model download + embedding
calls are mocked, since the goal here is to verify the API contract
without a 90MB download on every CI run.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyduck_janitor import DuckJanitor
from pyduck_janitor.embeddings import (
    DEFAULT_EMBED_MODEL,
    EmbeddingsNotAvailable,
    _is_installed,
    _model_dir,
    cache_dir,
    embed_install,
    embed_list_installed,
    embed_remove,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path: Path) -> Path:
    """Redirect the embedding cache to a tmp dir per test."""
    cache_root = tmp_path / "embed_cache"
    monkeypatch.setenv("PYDUCK_EMBED_CACHE", str(cache_root))
    return cache_root


@pytest.fixture
def fake_model_dir(tmp_path: Path) -> Path:
    """A directory that looks like a valid sentence-transformers model."""
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "bert"}')
    (model_dir / "model.safetensors").write_bytes(b"fake weights")
    (model_dir / "tokenizer.json").write_text("{}")
    return model_dir


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


class TestCacheManagement:
    def test_cache_dir_creates(self, isolated_cache):
        path = cache_dir()
        assert path == isolated_cache
        assert path.exists()
        assert path.is_dir()

    def test_is_installed_false_for_missing(self, isolated_cache):
        assert not _is_installed(DEFAULT_EMBED_MODEL)

    def test_embed_list_installed_empty(self, isolated_cache):
        df = embed_list_installed()
        assert len(df) == 0

    def test_install_from_local(self, isolated_cache, fake_model_dir):
        target = embed_install(str(fake_model_dir), allow_hf_fallback=False)
        # Local-path install with same name should copy the model.
        assert target.exists()
        assert (target / "config.json").exists()
        assert (target / "model.safetensors").exists()
        assert _is_installed(fake_model_dir.name)

    def test_install_idempotent(self, isolated_cache, fake_model_dir):
        target1 = embed_install(str(fake_model_dir), allow_hf_fallback=False)
        target2 = embed_install(str(fake_model_dir), allow_hf_fallback=False)
        assert target1 == target2
        # Both runs hit the same cache entry.

    def test_remove(self, isolated_cache, fake_model_dir):
        embed_install(str(fake_model_dir), allow_hf_fallback=False)
        assert _is_installed(fake_model_dir.name)
        removed = embed_remove(fake_model_dir.name)
        assert removed is True
        assert not _is_installed(fake_model_dir.name)

    def test_remove_missing(self, isolated_cache):
        assert embed_remove("never/existed") is False

    def test_list_after_install(self, isolated_cache, fake_model_dir):
        embed_install(str(fake_model_dir), allow_hf_fallback=False)
        df = embed_list_installed()
        assert len(df) == 1
        assert "size_bytes" in df.columns
        assert df.iloc[0]["has_config"]
        assert df.iloc[0]["has_weights"]

    def test_missing_bundled_raises_when_hf_disabled(self, isolated_cache):
        # No bundled wheel + HF disabled → clear error.
        with pytest.raises(EmbeddingsNotAvailable) as exc_info:
            embed_install("nonexistent-model", allow_hf_fallback=False)
        # The error message should hint at the install command.
        assert "Hint" in str(exc_info.value)
        assert "embed_install" in str(exc_info.value)


# ---------------------------------------------------------------------------
# embed_column error paths
# ---------------------------------------------------------------------------


class TestEmbedColumnErrors:
    def test_raises_when_model_missing(self, isolated_cache, monkeypatch):
        monkeypatch.setattr(
            "pyduck_janitor.embeddings._is_installed",
            lambda model: False,
        )
        # The verb must surface a clear error, not a silent download.
        df = pd.DataFrame({"text": ["hello"]})
        dj = DuckJanitor.from_pandas(df)
        with pytest.raises(EmbeddingsNotAvailable) as exc_info:
            from pyduck_janitor.embeddings import embed_column

            embed_column(dj, "text")
        assert "sentence-transformers" in str(exc_info.value) or \
               "Model" in str(exc_info.value)
        assert exc_info.value.install_command is not None


# ---------------------------------------------------------------------------
# vss verbs — DuckDB-side only (no embedding model)
# ---------------------------------------------------------------------------


def _extension_available(name: str) -> bool:
    conn = duckdb.connect(":memory:")
    try:
        from pyduck_janitor.extensions import load_extension

        try:
            load_extension(conn, name, install=False)
            return True
        except Exception:
            return False
    finally:
        conn.close()


@pytest.fixture
def vss_available() -> bool:
    return _extension_available("vss")


class TestVSSErrors:
    def test_search_without_index_raises(self, isolated_cache, vss_available):
        if not vss_available:
            pytest.skip("vss extension not available")
        from pyduck_janitor.embeddings import vector_search

        df = pd.DataFrame({"text": ["hello"]})
        dj = DuckJanitor.from_pandas(df)
        # No HNSW index recorded.
        with pytest.raises(ValueError, match="No HNSW index"):
            vector_search(dj, query=[0.1] * 4)

    def test_build_index_with_synthetic_embeddings(
        self, isolated_cache, vss_available
    ):
        if not vss_available:
            pytest.skip("vss extension not available")
        from pyduck_janitor.embeddings import build_vector_index

        # Fake an embedding column with random vectors. We don't
        # need a real model for this test.
        import numpy as np

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "embedding": [
                    list(map(float, rng.normal(size=4))),
                    list(map(float, rng.normal(size=4))),
                    list(map(float, rng.normal(size=4))),
                ],
            }
        )
        dj = DuckJanitor.from_pandas(df)
        dj2 = build_vector_index(dj, metric="cosine")
        assert hasattr(dj2, "_vss_index")
        assert dj2._vss_metric == "cosine"
