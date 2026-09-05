"""
DuckDB extension management for pyduck-janitor.

Loads and caches DuckDB extensions (``icu``, ``fts``, ``vss``, ``json``,
``onager``)
on a per-connection basis. Extensions are an opt-in capability: the
core pyduck-janitor verbs do not require them, and a missing extension
is surfaced as a clear ``ExtensionNotAvailable`` error pointing at the
exact pip extra or environment variable to fix it.

Extension loading is lazy and idempotent — calling ``load_extension``
twice is a no-op. Failed loads raise ``ExtensionNotAvailable`` with
the original DuckDB error attached for diagnostics.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from typing import Optional

import duckdb

__all__ = [
    "EXTENSIONS",
    "ExtensionNotAvailable",
    "load_extension",
    "load_extensions",
    "extension_loaded",
]


# Map from extension logical name -> (pip extra, DuckDB name).
# The pip extra is what the user installs to make the extension
# available; the DuckDB name is what we pass to LOAD.
EXTENSIONS: dict[str, tuple[str, str]] = {
    "icu": ("icu", "icu"),
    "fts": ("fts", "fts"),
    "vss": ("vss", "vss"),
    "json": ("json", "json"),
    "onager": ("graph", "onager"),
    "duck_diff": ("diff", "duck_diff"),
}

_COMMUNITY_EXTENSIONS = {"onager", "duck_diff"}


class ExtensionNotAvailable(RuntimeError):
    """Raised when a DuckDB extension cannot be loaded.

    Attributes
    ----------
    name : str
        Logical extension name (e.g. ``"icu"``).
    pip_extra : str or None
        The pip extra that would install the supporting binary,
        or ``None`` if no extra is known.
    cause : Exception or None
        The original DuckDB error, if any.
    """

    def __init__(
        self,
        name: str,
        message: str,
        pip_extra: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.name = name
        self.pip_extra = pip_extra
        self.cause = cause
        hint = ""
        if pip_extra:
            hint = (
                f"\n\nHint: install the supporting pip extra:\n"
                f"    pip install pyduck-janitor[{pip_extra}]\n"
                f"or, for all extensions:\n"
                f"    pip install pyduck-janitor[text]"
            )
        super().__init__(message + hint)


# Per-connection set of "successfully loaded" extensions. A second
# load of the same extension is a no-op.
_lock = threading.Lock()
_loaded: dict[int, set[str]] = {}


def _conn_id(conn: duckdb.DuckDBPyConnection) -> int:
    # DuckDB connections are Python objects; id() is a stable handle
    # for the lifetime of the process and is sufficient for caching.
    return id(conn)


def extension_loaded(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    """Return True if ``name`` has been successfully loaded on ``conn``."""
    with _lock:
        return name in _loaded.get(_conn_id(conn), set())


def load_extension(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    install: bool = True,
    repository: Optional[str] = None,
) -> None:
    """Load a DuckDB extension onto a connection.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        The connection to load the extension onto.
    name : str
        Logical extension name (one of ``icu``, ``fts``, ``vss``, ``json``,
        or ``onager``).
    install : bool, default True
        If True, run ``INSTALL <name>`` before ``LOAD <name>``. Set
        False when the extension has already been installed system-wide
        or you want to surface install errors separately.
    repository : str, optional
        DuckDB extension repository used by ``INSTALL``. Onager defaults to
        ``"community"``; other extensions use DuckDB's default repository.

    Raises
    ------
    ExtensionNotAvailable
        When the extension cannot be loaded. The original DuckDB
        error is attached as ``__cause__``.
    """
    if name not in EXTENSIONS:
        raise ValueError(f"Unknown extension '{name}'. Known: {sorted(EXTENSIONS)}")

    cid = _conn_id(conn)
    with _lock:
        if name in _loaded.get(cid, set()):
            return

    pip_extra, duckdb_name = EXTENSIONS[name]

    if _env_disabled():
        raise ExtensionNotAvailable(
            name,
            f"Extension loading disabled by PYDUCK_SKIP_EXTENSIONS for '{duckdb_name}'.",
            pip_extra=pip_extra,
        )

    # Skip install if the user opts out (e.g. they manage extensions
    # via a custom DuckDB configuration or a Docker image).
    if install:
        try:
            install_sql = f"INSTALL {duckdb_name}"
            repository = repository or ("community" if name in _COMMUNITY_EXTENSIONS else None)
            if repository:
                if not repository.replace("_", "").isalnum():
                    raise ValueError("repository must contain only letters, numbers, or _")
                install_sql += f" FROM {repository}"
            conn.execute(install_sql)
        except duckdb.Error as exc:
            raise ExtensionNotAvailable(
                name,
                f"Failed to install DuckDB extension '{duckdb_name}': {exc}",
                pip_extra=pip_extra,
                cause=exc,
            ) from exc

    try:
        conn.execute(f"LOAD {duckdb_name}")
    except duckdb.Error as exc:
        raise ExtensionNotAvailable(
            name,
            f"Failed to load DuckDB extension '{duckdb_name}': {exc}",
            pip_extra=pip_extra,
            cause=exc,
        ) from exc

    with _lock:
        _loaded.setdefault(cid, set()).add(name)


def load_extensions(
    conn: duckdb.DuckDBPyConnection,
    names: Iterable[str],
    *,
    install: bool = True,
) -> None:
    """Load multiple DuckDB extensions on a single connection.

    Failures on the first extension raise immediately; later
    extensions are not attempted.
    """
    for name in names:
        load_extension(conn, name, install=install)


# Allow opt-out via environment variable for environments where
# extension installation must be controlled by ops (e.g. air-gapped
# CI, locked-down containers). Set ``PYDUCK_SKIP_EXTENSIONS=1`` to
# make ``load_extension`` raise ``ExtensionNotAvailable`` instead of
# attempting to install.
def _env_disabled() -> bool:
    return os.environ.get("PYDUCK_SKIP_EXTENSIONS", "").lower() in (
        "1",
        "true",
        "yes",
    )
