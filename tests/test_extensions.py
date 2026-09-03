"""Regression tests for pyduck_janitor.extensions."""

from __future__ import annotations

import duckdb
import pytest
from pyduck_janitor.extensions import ExtensionNotAvailable, load_extension


def test_skip_extensions_env_disables_loading(monkeypatch):
    conn = duckdb.connect(":memory:")
    monkeypatch.setenv("PYDUCK_SKIP_EXTENSIONS", "1")
    try:
        with pytest.raises(ExtensionNotAvailable, match="disabled"):
            load_extension(conn, "json")
    finally:
        conn.close()
