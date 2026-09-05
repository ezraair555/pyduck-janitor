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


def test_onager_install_uses_community_repository(monkeypatch):
    statements = []

    class FakeConnection:
        def execute(self, statement):
            statements.append(statement)

    monkeypatch.delenv("PYDUCK_SKIP_EXTENSIONS", raising=False)
    load_extension(FakeConnection(), "onager")

    assert statements == ["INSTALL onager FROM community", "LOAD onager"]
