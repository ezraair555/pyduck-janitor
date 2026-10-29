"""
Remote-file support for pyduck-janitor.

The README of this package advertises ``from_parquet('s3://bucket/data.parquet')``
and ``from_json('s3://...')``, but the package itself never loaded DuckDB's
``httpfs`` extension nor configured S3 credentials. This module makes that
capability real.

Three things live here:

* :func:`load_httpfs` — loads the ``httpfs`` extension on a connection.
  ``httpfs`` is a DuckDB *core* extension: it ships inside the standard
  ``duckdb`` wheel, so on a normal install ``LOAD httpfs`` works with no
  network access and no ``INSTALL`` step. When the extension really is
  missing, the helper raises
  :class:`~pyduck_janitor.extensions.ExtensionNotAvailable` with the
  environment fix spelled out.

* :func:`configure_s3` / :func:`create_s3_secret` — put S3 credentials on a
  connection. DuckDB 1.5.2 supports both the modern persistent/temporary
  *secret manager* (``CREATE SECRET``) and the legacy ``SET s3_*`` settings.
  ``configure_s3`` prefers the secret manager and falls back to ``SET``.

  **Secret handling rule:** credential values are never printed, never
  logged, and never embedded in an exception message. Values are passed to
  DuckDB as bind parameters (``?``) rather than interpolated into SQL text,
  so they cannot leak through query logs either. When a statement involving
  a credential fails, the raised error deliberately drops the original
  DuckDB message (DuckDB echoes offending values in some conversion
  errors) and says only that the value was redacted.

* :func:`read_remote` / :func:`list_remote` — read and enumerate remote
  objects (S3, HTTP(S), GCS) through DuckDB.

Environment variables
---------------------
When the corresponding argument is omitted, the credential helpers fall back
to these environment variables:

===============================  ====================================
Variable                         Meaning
===============================  ====================================
``PYDUCK_S3_ACCESS_KEY``         S3 access key id
``PYDUCK_S3_SECRET_KEY``         S3 secret access key
``PYDUCK_S3_SESSION_TOKEN``      Optional session token (STS)
``PYDUCK_S3_REGION``             S3 region, e.g. ``us-east-1``
``PYDUCK_S3_ENDPOINT``           Custom S3 endpoint (MinIO, Ceph, ...)
===============================  ====================================

Explicit arguments always win over the environment.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Union

import duckdb

from .extensions import ExtensionNotAvailable, load_extension

__all__ = [
    "REMOTE_PREFIXES",
    "configure_s3",
    "create_s3_secret",
    "is_remote_path",
    "list_remote",
    "load_httpfs",
    "read_remote",
]


#: URI prefixes treated as "remote" by :func:`is_remote_path`.
REMOTE_PREFIXES: tuple[str, ...] = ("s3://", "http://", "https://", "gs://")

# Environment variable names consulted when an argument is not supplied.
ENV_ACCESS_KEY = "PYDUCK_S3_ACCESS_KEY"
ENV_SECRET_KEY = "PYDUCK_S3_SECRET_KEY"
ENV_SESSION_TOKEN = "PYDUCK_S3_SESSION_TOKEN"
ENV_REGION = "PYDUCK_S3_REGION"
ENV_ENDPOINT = "PYDUCK_S3_ENDPOINT"

# ``httpfs`` ships inside the standard DuckDB wheel, so it has no pip
# extra. It is registered in ``extensions.EXTENSIONS`` so the existing
# ``extensions.load_extension`` helper can be reused: that gives us its
# per-connection idempotency cache and its ``ExtensionNotAvailable``
# error mapping for free.

# DuckDB secret names are identifiers; we interpolate the name into SQL
# (the credential *values* always travel as bind parameters), so it must
# be validated.
_SECRET_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Supported ``read_remote`` formats mapped to the DuckDB table function
# used to read them.
_FORMAT_READERS: dict[str, str] = {
    "parquet": "read_parquet",
    "csv": "read_csv",
    "csv_auto": "read_csv_auto",
    "json": "read_json_auto",
    "json_auto": "read_json_auto",
    "text": "read_text",
}

_HTTPFS_UNAVAILABLE = (
    "DuckDB's 'httpfs' extension is required for remote file access "
    "(s3://, gs://, http://, https://) but it could not be loaded.\n"
    "'httpfs' is a core extension and ships inside the standard DuckDB "
    "wheel, so the usual fix is to install or upgrade DuckDB:\n"
    "    pip install --upgrade 'duckdb>=1.5.2'\n"
    "Alternatively make the extension available on this host by running\n"
    "    INSTALL httpfs;\n"
    "once with network access, or point the DUCKDB_EXTENSION_DIRECTORY "
    "environment variable at a directory containing "
    "httpfs.duckdb_extension."
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _env(name: str) -> Optional[str]:
    """Return a non-empty environment value, or None."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _first_env(explicit: Optional[str], env_name: str) -> Optional[str]:
    """Explicit argument wins; otherwise consult the environment."""
    if explicit is not None:
        return explicit
    return _env(env_name)


def _set_setting(
    conn: duckdb.DuckDBPyConnection,
    setting: str,
    value: object,
    *,
    sensitive: bool = False,
) -> None:
    """Apply a single ``SET <setting> = ?`` statement.

    The value is bound as a parameter, so it never appears in the SQL text
    DuckDB might log. For credential-bearing settings the original DuckDB
    error is suppressed entirely (``from None``) because DuckDB echoes the
    offending value in some conversion errors.
    """
    try:
        conn.execute(f"SET {setting} = ?", [value])
    except duckdb.Error as exc:
        if sensitive:
            raise RuntimeError(
                f"Failed to set DuckDB setting '{setting}' "
                f"({type(exc).__name__}); the value was redacted and is not "
                f"included in this error."
            ) from None
        raise RuntimeError(f"Failed to set DuckDB setting '{setting}': {exc}") from exc


def _looks_like_httpfs_error(exc: BaseException) -> bool:
    """True when a DuckDB error is really 'httpfs is not available'."""
    message = str(exc).lower()
    return "httpfs" in message


def _glob_files(conn: duckdb.DuckDBPyConnection, pattern: str) -> list[str]:
    """Return the files matching a DuckDB ``glob()`` pattern."""
    rows = conn.execute("SELECT file FROM glob(?)", [pattern]).fetchall()
    return [row[0] for row in rows]


def _has_glob_magic(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


# --------------------------------------------------------------------------- #
# extension loading
# --------------------------------------------------------------------------- #
def load_httpfs(
    conn: duckdb.DuckDBPyConnection,
    *,
    install: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Load DuckDB's ``httpfs`` extension on ``conn``.

    ``httpfs`` is a DuckDB *core* extension: it is bundled in the standard
    ``duckdb`` wheel, so ``LOAD httpfs`` normally succeeds without any
    download. This helper is idempotent — calling it twice on the same
    connection is a no-op (the existing
    :func:`pyduck_janitor.extensions.load_extension` helper keeps a
    per-connection cache of what it already loaded).

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Connection to load the extension onto.
    install : bool, default False
        When True, the helper falls back to running ``INSTALL httpfs``
        (which needs network access) if the plain ``LOAD`` fails. It is
        never attempted up front, so the common path stays offline-safe.

    Returns
    -------
    duckdb.DuckDBPyConnection
        The same connection, for chaining.

    Raises
    ------
    ExtensionNotAvailable
        When the extension cannot be loaded. The message names the pip
        requirement (``duckdb>=1.5.2``) and the environment fix
        (``INSTALL httpfs`` / ``DUCKDB_EXTENSION_DIRECTORY``). The
        original DuckDB error is attached as ``__cause__``.
    """
    attempts = (False, True) if install else (False,)
    failure: Optional[ExtensionNotAvailable] = None
    for use_install in attempts:
        try:
            load_extension(conn, "httpfs", install=use_install)
            return conn
        except ExtensionNotAvailable as exc:
            failure = exc
    assert failure is not None  # attempts is never empty
    raise ExtensionNotAvailable(
        "httpfs",
        f"{_HTTPFS_UNAVAILABLE}\n\nDuckDB reported:\n{failure}",
        cause=failure,
    ) from failure


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
def create_s3_secret(
    conn: duckdb.DuckDBPyConnection,
    name: str = "pyduck_s3",
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: Optional[str] = None,
    session_token: Optional[str] = None,
    endpoint: Optional[str] = None,
    *,
    use_ssl: Optional[bool] = None,
    persistent: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Create an S3 secret in DuckDB's secret manager.

    DuckDB 1.5.2 supports ``CREATE SECRET (TYPE S3, ...)``; this helper
    uses the *temporary* (in-memory) form by default so credentials are
    never written to disk.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Connection that receives the secret.
    name : str, default ``"pyduck_s3"``
        Name of the secret. Must be a plain identifier; an existing
        secret with the same name is replaced.
    access_key, secret_key : str, optional
        Credentials. When omitted they fall back to ``PYDUCK_S3_ACCESS_KEY``
        and ``PYDUCK_S3_SECRET_KEY``. Both are required.
    region : str, optional
        S3 region; falls back to ``PYDUCK_S3_REGION``.
    session_token : str, optional
        Optional STS session token; falls back to
        ``PYDUCK_S3_SESSION_TOKEN``.
    endpoint : str, optional
        Custom endpoint (MinIO, Ceph, ...); falls back to
        ``PYDUCK_S3_ENDPOINT``.
    use_ssl : bool, optional
        Whether to use TLS for the endpoint. Left untouched when None.
    persistent : bool, default False
        When True the secret is written to DuckDB's on-disk secret store
        (``~/.duckdb/stored_secrets``) and survives restarts. Off by
        default so credentials stay in memory.

    Returns
    -------
    duckdb.DuckDBPyConnection
        The same connection, for chaining.

    Raises
    ------
    ValueError
        If the secret name is not a valid identifier, or the credentials
        are missing.
    RuntimeError
        If DuckDB rejects the statement. The message never contains the
        credential values.
    """
    if not isinstance(name, str) or not _SECRET_NAME_RE.match(name):
        raise ValueError(
            f"Invalid secret name {name!r}. A secret name must start with a "
            f"letter or underscore and contain only letters, digits and "
            f"underscores."
        )

    key_id = _first_env(access_key, ENV_ACCESS_KEY)
    secret = _first_env(secret_key, ENV_SECRET_KEY)
    reg = _first_env(region, ENV_REGION)
    token = _first_env(session_token, ENV_SESSION_TOKEN)
    ep = _first_env(endpoint, ENV_ENDPOINT)

    if not key_id or not secret:
        raise ValueError(
            "create_s3_secret requires both an access key and a secret key "
            "(pass access_key=/secret_key= or set "
            f"{ENV_ACCESS_KEY} and {ENV_SECRET_KEY})."
        )

    clauses = ["TYPE S3", "KEY_ID ?", "SECRET ?"]
    params: list[object] = [key_id, secret]
    if reg:
        clauses.append("REGION ?")
        params.append(reg)
    if token:
        clauses.append("SESSION_TOKEN ?")
        params.append(token)
    if ep:
        clauses.append("ENDPOINT ?")
        params.append(ep)
    if use_ssl is not None:
        clauses.append("USE_SSL ?")
        params.append(bool(use_ssl))

    kind = "PERSISTENT" if persistent else "TEMPORARY"
    sql = f"CREATE OR REPLACE {kind} SECRET {name} ({', '.join(clauses)})"
    try:
        conn.execute(sql, params)
    except duckdb.Error as exc:
        raise RuntimeError(
            f"Failed to create DuckDB S3 secret '{name}' "
            f"({type(exc).__name__}); credential values are redacted and are "
            f"not included in this error."
        ) from None
    return conn


def configure_s3(
    conn: duckdb.DuckDBPyConnection,
    *,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    session_token: Optional[str] = None,
    region: Optional[str] = None,
    endpoint: Optional[str] = None,
    use_ssl: bool = True,
    name: str = "pyduck_s3",
) -> duckdb.DuckDBPyConnection:
    """Configure S3 access on a connection.

    Two mechanisms are used, in this order:

    1. **Secret manager (preferred).** When both an access key and a secret
       key are available, a temporary S3 secret is created via
       :func:`create_s3_secret`. This is DuckDB's modern mechanism: the
       credentials live in the connection's secret manager, where DuckDB
       redacts them when listing secrets.
    2. **Legacy settings (fallback / mirror).** The equivalent
       ``SET s3_access_key_id``, ``SET s3_secret_access_key``,
       ``SET s3_session_token``, ``SET s3_region``, ``SET s3_endpoint`` and
       ``SET s3_use_ssl`` statements are applied as well, so the
       configuration is visible through ``current_setting('s3_region')`` and
       friends, and so the function still works on DuckDB builds where the
       secret manager rejects the statement.

    A setting whose value is ``None`` is left untouched.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Connection to configure.
    access_key, secret_key : str, optional
        Credentials; fall back to ``PYDUCK_S3_ACCESS_KEY`` and
        ``PYDUCK_S3_SECRET_KEY`` when omitted.
    session_token : str, optional
        Falls back to ``PYDUCK_S3_SESSION_TOKEN``.
    region : str, optional
        Falls back to ``PYDUCK_S3_REGION``.
    endpoint : str, optional
        Falls back to ``PYDUCK_S3_ENDPOINT``.
    use_ssl : bool, default True
        Value applied to ``s3_use_ssl``.
    name : str, default ``"pyduck_s3"``
        Name of the secret created in the secret manager.

    Returns
    -------
    duckdb.DuckDBPyConnection
        The same connection, for chaining.

    Raises
    ------
    RuntimeError
        If DuckDB rejects a credential-bearing statement. The message
        never contains the credential values.
    """
    key_id = _first_env(access_key, ENV_ACCESS_KEY)
    secret = _first_env(secret_key, ENV_SECRET_KEY)
    token = _first_env(session_token, ENV_SESSION_TOKEN)
    reg = _first_env(region, ENV_REGION)
    ep = _first_env(endpoint, ENV_ENDPOINT)

    if key_id and secret:
        try:
            create_s3_secret(
                conn,
                name=name,
                access_key=key_id,
                secret_key=secret,
                region=reg,
                session_token=token,
                endpoint=ep,
                use_ssl=use_ssl,
            )
        except (duckdb.Error, RuntimeError, ValueError):
            # Secret manager unavailable or unhappy: the legacy SET path
            # below is the documented fallback.
            pass

    # Credential-bearing settings are applied first so that a failure on
    # them is surfaced (and redacted) before any other setting is touched.
    for setting, value, sensitive in (
        ("s3_access_key_id", key_id, True),
        ("s3_secret_access_key", secret, True),
        ("s3_session_token", token, True),
        ("s3_region", reg, False),
        ("s3_endpoint", ep, False),
        ("s3_use_ssl", use_ssl, False),
    ):
        if value is None:
            continue
        _set_setting(conn, setting, value, sensitive=sensitive)

    return conn


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def is_remote_path(path: Union[str, os.PathLike[str]]) -> bool:
    """Return True when ``path`` looks like a remote object URI.

    Recognised prefixes: ``s3://``, ``http://``, ``https://`` and
    ``gs://``.
    """
    if path is None:
        return False
    text = str(path).strip().lower()
    return text.startswith(REMOTE_PREFIXES)


def read_remote(
    path: Union[str, os.PathLike[str]],
    *,
    format: str = "parquet",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    install_httpfs: bool = True,
) -> duckdb.DuckDBPyRelation:
    """Read a (possibly remote) file into a DuckDB relation.

    The path may be local or remote (``s3://``, ``http://``, ``https://``,
    ``gs://``). Remote access is provided by DuckDB's ``httpfs``
    extension, which is loaded first when ``install_httpfs`` is True.

    Parameters
    ----------
    path : str or os.PathLike
        File to read. Local paths work too — they simply exercise the same
        reader-selection code path without touching the network.
    format : str, default ``"parquet"``
        One of ``parquet``, ``csv``, ``csv_auto``, ``json``, ``json_auto``,
        ``text``. The format selects the DuckDB table function:
        ``read_parquet``, ``read_csv``, ``read_csv_auto``, ``read_json_auto``
        or ``read_text``.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to read on. A new in-memory connection is created when
        omitted; the returned relation keeps it alive.
    install_httpfs : bool, default True
        Load the ``httpfs`` extension before reading. Set to False when the
        extension is already loaded or when only local files are read.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Relation over the file's contents.

    Raises
    ------
    ValueError
        If ``format`` is not supported. The message lists the supported
        formats.
    ExtensionNotAvailable
        If the ``httpfs`` extension is required but unavailable.
    """
    fmt = str(format).strip().lower()
    reader = _FORMAT_READERS.get(fmt)
    if reader is None:
        raise ValueError(
            f"Unsupported format {format!r} for read_remote. "
            f"Supported formats: {', '.join(sorted(_FORMAT_READERS))}."
        )

    connection = duckdb.connect() if conn is None else conn
    if install_httpfs:
        load_httpfs(connection)

    try:
        relation = connection.sql(f"SELECT * FROM {reader}(?)", params=[str(path)])
    except duckdb.Error as exc:
        if _looks_like_httpfs_error(exc):
            raise ExtensionNotAvailable(
                "httpfs",
                f"{_HTTPFS_UNAVAILABLE}\n\nDuckDB reported:\n{exc}",
                cause=exc,
            ) from exc
        raise
    return relation


def list_remote(
    path: Union[str, os.PathLike[str]],
    *,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    install_httpfs: bool = True,
) -> list[str]:
    """List files at a remote location using DuckDB's ``glob()``.

    ``path`` is a glob pattern, e.g. ``s3://bucket/data/*.parquet``. When
    the pattern contains no glob metacharacter and matches nothing, it is
    retried as ``<path>/*`` so that a bare prefix or directory listing
    still works.

    Parameters
    ----------
    path : str or os.PathLike
        Glob pattern to expand.
    conn : duckdb.DuckDBPyConnection, optional
        Connection to use. A new in-memory connection is created when
        omitted.
    install_httpfs : bool, default True
        Load the ``httpfs`` extension before listing.

    Returns
    -------
    list[str]
        Matching file paths.

    Raises
    ------
    ExtensionNotAvailable
        If the ``httpfs`` extension is required but unavailable.
    """
    connection = duckdb.connect() if conn is None else conn
    if install_httpfs:
        load_httpfs(connection)

    pattern = str(path)
    try:
        files = _glob_files(connection, pattern)
        if not files and not _has_glob_magic(pattern):
            files = _glob_files(connection, pattern.rstrip("/") + "/*")
    except duckdb.Error as exc:
        if _looks_like_httpfs_error(exc):
            raise ExtensionNotAvailable(
                "httpfs",
                f"{_HTTPFS_UNAVAILABLE}\n\nDuckDB reported:\n{exc}",
                cause=exc,
            ) from exc
        raise
    return files
