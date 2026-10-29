"""Tests for pyduck_janitor.remote (remote-file support helpers).

The environment has no network access to S3, so these tests exercise the
plumbing only: URI detection, reader selection against local files,
credential plumbing through DuckDB's settings and secret manager, and the
graceful degradation path when the ``httpfs`` extension is unavailable.
"""

from __future__ import annotations

import duckdb
import pytest

from pyduck_janitor.extensions import ExtensionNotAvailable
from pyduck_janitor.remote import (
    configure_s3,
    create_s3_secret,
    is_remote_path,
    list_remote,
    load_httpfs,
    read_remote,
)

SECRET_VALUE = "sUpErSeCrEtVaLuE-0123456789"


@pytest.fixture()
def conn():
    connection = duckdb.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def csv_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,2\n3,4\n")
    return path


@pytest.fixture()
def json_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')
    return path


@pytest.fixture()
def parquet_file(tmp_path, conn):
    path = tmp_path / "data.parquet"
    conn.execute(f"COPY (SELECT 1 AS x, 'z' AS y) TO '{path}' (FORMAT PARQUET)")
    return path


# --------------------------------------------------------------------------- #
# is_remote_path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [
        "s3://bucket/data.parquet",
        "http://example.com/data.csv",
        "https://example.com/data.csv",
        "gs://bucket/data.parquet",
    ],
)
def test_is_remote_path_true_for_remote_prefixes(path):
    assert is_remote_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/data.parquet",
        "data.parquet",
        "./relative/data.csv",
        "file:///tmp/data.csv",
        "s3:/bucket/data.parquet",
    ],
)
def test_is_remote_path_false_for_local_paths(path):
    assert is_remote_path(path) is False


def test_is_remote_path_is_case_insensitive():
    assert is_remote_path("S3://bucket/key") is True
    assert is_remote_path("HTTPS://example.com/x") is True


def test_is_remote_path_accepts_pathlike(tmp_path):
    assert is_remote_path(tmp_path / "x.parquet") is False


# --------------------------------------------------------------------------- #
# load_httpfs
# --------------------------------------------------------------------------- #
def test_load_httpfs_returns_connection(conn):
    assert load_httpfs(conn) is conn


def test_load_httpfs_is_idempotent(conn):
    load_httpfs(conn)
    load_httpfs(conn)
    load_httpfs(conn)
    assert load_httpfs(conn) is conn


def test_load_httpfs_actually_loads_the_extension(conn):
    load_httpfs(conn)
    rows = conn.execute(
        "SELECT loaded FROM duckdb_extensions() WHERE extension_name = 'httpfs'"
    ).fetchall()
    assert rows == [(True,)]


def test_load_httpfs_raises_extension_not_available_when_disabled(conn, monkeypatch):
    monkeypatch.setenv("PYDUCK_SKIP_EXTENSIONS", "1")
    with pytest.raises(ExtensionNotAvailable) as excinfo:
        load_httpfs(conn)
    message = str(excinfo.value)
    assert "httpfs" in message
    assert "duckdb" in message.lower()


def test_load_httpfs_error_names_the_environment_fix(conn, monkeypatch):
    monkeypatch.setenv("PYDUCK_SKIP_EXTENSIONS", "1")
    with pytest.raises(ExtensionNotAvailable) as excinfo:
        load_httpfs(conn)
    message = str(excinfo.value)
    assert "INSTALL httpfs" in message or "DUCKDB_EXTENSION_DIRECTORY" in message


# --------------------------------------------------------------------------- #
# configure_s3
# --------------------------------------------------------------------------- #
def test_configure_s3_sets_duckdb_settings(conn):
    configure_s3(
        conn,
        access_key="AKIAEXAMPLE",
        secret_key=SECRET_VALUE,
        region="us-east-1",
        session_token="session-token-value",
        endpoint="https://s3.us-east-1.amazonaws.com",
        use_ssl=True,
    )
    assert conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == (
        "AKIAEXAMPLE"
    )
    assert conn.execute("SELECT current_setting('s3_region')").fetchone()[0] == "us-east-1"
    assert conn.execute("SELECT current_setting('s3_session_token')").fetchone()[0] == (
        "session-token-value"
    )
    assert conn.execute("SELECT current_setting('s3_endpoint')").fetchone()[0] == (
        "https://s3.us-east-1.amazonaws.com"
    )
    assert conn.execute("SELECT current_setting('s3_use_ssl')").fetchone()[0] is True


def test_configure_s3_returns_connection(conn):
    assert configure_s3(conn, access_key="AKIA", secret_key="SK") is conn


def test_configure_s3_leaves_none_settings_untouched(conn):
    conn.execute("SET s3_region = 'pre-existing'")
    configure_s3(conn, access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE)
    assert conn.execute("SELECT current_setting('s3_region')").fetchone()[0] == "pre-existing"


def test_configure_s3_reads_environment_variables(conn, monkeypatch):
    monkeypatch.setenv("PYDUCK_S3_ACCESS_KEY", "AKIAFROMENV")
    monkeypatch.setenv("PYDUCK_S3_SECRET_KEY", "SECRETFROMENV")
    monkeypatch.setenv("PYDUCK_S3_REGION", "eu-west-1")
    monkeypatch.setenv("PYDUCK_S3_ENDPOINT", "https://env.example.com")
    monkeypatch.setenv("PYDUCK_S3_SESSION_TOKEN", "env-token")
    configure_s3(conn)
    assert conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == (
        "AKIAFROMENV"
    )
    assert conn.execute("SELECT current_setting('s3_region')").fetchone()[0] == "eu-west-1"
    assert conn.execute("SELECT current_setting('s3_endpoint')").fetchone()[0] == (
        "https://env.example.com"
    )
    assert conn.execute("SELECT current_setting('s3_session_token')").fetchone()[0] == ("env-token")


def test_configure_s3_explicit_arguments_beat_environment(conn, monkeypatch):
    monkeypatch.setenv("PYDUCK_S3_ACCESS_KEY", "AKIAFROMENV")
    monkeypatch.setenv("PYDUCK_S3_SECRET_KEY", "SECRETFROMENV")
    monkeypatch.setenv("PYDUCK_S3_REGION", "eu-west-1")
    configure_s3(conn, access_key="AKIAEXPLICIT", secret_key="SECRETEXPLICIT", region="ap-south-1")
    assert conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == (
        "AKIAEXPLICIT"
    )
    assert conn.execute("SELECT current_setting('s3_region')").fetchone()[0] == "ap-south-1"


def test_configure_s3_creates_a_secret_in_the_secret_manager(conn):
    configure_s3(conn, access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE, region="us-east-1")
    names = [row[0] for row in conn.execute("SELECT name FROM duckdb_secrets()").fetchall()]
    assert "pyduck_s3" in names


def test_configure_s3_secret_manager_redacts_the_secret_value(conn):
    configure_s3(conn, access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE, region="us-east-1")
    secret_strings = [
        row[0] for row in conn.execute("SELECT secret_string FROM duckdb_secrets()").fetchall()
    ]
    joined = "\n".join(secret_strings)
    assert SECRET_VALUE not in joined
    assert "redacted" in joined


def test_configure_s3_does_not_leak_the_secret_in_an_exception(conn):
    class ExplodingConnection:
        def execute(self, statement, params=None):
            # DuckDB echoes offending values in some conversion errors;
            # the helper must strip that from the surfaced message.
            raise duckdb.Error(f"Invalid Input Error: bad value {params} for {statement}")

    with pytest.raises(RuntimeError) as excinfo:
        configure_s3(
            ExplodingConnection(),
            access_key="AKIAEXAMPLE",
            secret_key=SECRET_VALUE,
            region="us-east-1",
        )
    message = str(excinfo.value)
    assert SECRET_VALUE not in message
    assert "redacted" in message


def test_configure_s3_does_not_leak_the_secret_in_a_repr(conn):
    configure_s3(conn, access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE, region="us-east-1")
    assert SECRET_VALUE not in repr(conn)
    assert SECRET_VALUE not in repr(configure_s3(conn))


def test_configure_s3_falls_back_to_set_when_the_secret_manager_fails(conn):
    calls = []
    real_execute = conn.execute

    class SecretManagerBrokenConnection:
        """Delegates to the real connection but rejects secret DDL."""

        def execute(self, statement, *args, **kwargs):
            calls.append(statement)
            upper = statement.upper()
            if "CREATE" in upper and "SECRET" in upper:
                raise duckdb.Error("Binder Error: secret manager unavailable")
            return real_execute(statement, *args, **kwargs)

    configure_s3(
        SecretManagerBrokenConnection(),
        access_key="AKIAEXAMPLE",
        secret_key=SECRET_VALUE,
        region="us-east-1",
    )
    assert any("SET s3_access_key_id" in stmt for stmt in calls)
    assert conn.execute("SELECT current_setting('s3_region')").fetchone()[0] == "us-east-1"
    assert conn.execute("SELECT current_setting('s3_access_key_id')").fetchone()[0] == (
        "AKIAEXAMPLE"
    )


# --------------------------------------------------------------------------- #
# create_s3_secret
# --------------------------------------------------------------------------- #
def test_create_s3_secret_creates_a_temporary_secret(conn):
    create_s3_secret(conn, access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE, region="us-east-1")
    rows = conn.execute(
        "SELECT name, type, persistent FROM duckdb_secrets() WHERE name = 'pyduck_s3'"
    ).fetchall()
    assert rows == [("pyduck_s3", "s3", False)]


def test_create_s3_secret_is_not_persistent_by_default(conn):
    create_s3_secret(conn, access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE)
    rows = conn.execute(
        "SELECT persistent FROM duckdb_secrets() WHERE name = 'pyduck_s3'"
    ).fetchall()
    assert rows == [(False,)]


def test_create_s3_secret_accepts_a_custom_name(conn):
    create_s3_secret(conn, name="my_bucket", access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE)
    names = [row[0] for row in conn.execute("SELECT name FROM duckdb_secrets()").fetchall()]
    assert "my_bucket" in names


def test_create_s3_secret_rejects_invalid_names(conn):
    with pytest.raises(ValueError, match="Invalid secret name"):
        create_s3_secret(conn, name="bad name; DROP", access_key="K", secret_key="S")


def test_create_s3_secret_requires_credentials(conn):
    with pytest.raises(ValueError, match="access key"):
        create_s3_secret(conn)


def test_create_s3_secret_reads_the_environment(conn, monkeypatch):
    monkeypatch.setenv("PYDUCK_S3_ACCESS_KEY", "AKIAFROMENV")
    monkeypatch.setenv("PYDUCK_S3_SECRET_KEY", "SECRETFROMENV")
    create_s3_secret(conn)
    secret_strings = [
        row[0] for row in conn.execute("SELECT secret_string FROM duckdb_secrets()").fetchall()
    ]
    assert "key_id=AKIAFROMENV" in "\n".join(secret_strings)
    assert "SECRETFROMENV" not in "\n".join(secret_strings)


def test_create_s3_secret_error_does_not_leak_the_secret(conn):
    class ExplodingConnection:
        def execute(self, statement, params=None):
            raise duckdb.Error(f"Invalid Input Error: bad value {params}")

    with pytest.raises(RuntimeError) as excinfo:
        create_s3_secret(ExplodingConnection(), access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE)
    assert SECRET_VALUE not in str(excinfo.value)


def test_create_s3_secret_returns_the_connection(conn):
    assert create_s3_secret(conn, access_key="AKIAEXAMPLE", secret_key=SECRET_VALUE) is conn


# --------------------------------------------------------------------------- #
# read_remote
# --------------------------------------------------------------------------- #
def test_read_remote_reads_a_local_csv(csv_file):
    relation = read_remote(csv_file, format="csv")
    assert relation.fetchall() == [(1, 2), (3, 4)]


def test_read_remote_reads_a_local_csv_auto(csv_file):
    relation = read_remote(str(csv_file), format="csv_auto")
    assert relation.fetchall() == [(1, 2), (3, 4)]


def test_read_remote_reads_a_local_json(json_file):
    relation = read_remote(json_file, format="json")
    rows = relation.fetchall()
    assert len(rows) == 2
    assert rows[0][0] == 1


def test_read_remote_reads_a_local_parquet(parquet_file):
    relation = read_remote(parquet_file, format="parquet")
    assert relation.fetchall() == [(1, "z")]


def test_read_remote_accepts_a_connection(conn, csv_file):
    relation = read_remote(csv_file, format="csv", conn=conn)
    assert relation.fetchall() == [(1, 2), (3, 4)]


def test_read_remote_returns_a_duckdb_relation(csv_file):
    relation = read_remote(csv_file, format="csv")
    assert isinstance(relation, duckdb.DuckDBPyRelation)


def test_read_remote_rejects_unsupported_formats(csv_file):
    with pytest.raises(ValueError) as excinfo:
        read_remote(csv_file, format="xlsx")
    message = str(excinfo.value)
    assert "parquet" in message
    assert "csv" in message
    assert "json" in message
    assert "text" in message


def test_read_remote_format_is_case_insensitive(csv_file):
    assert read_remote(csv_file, format="CSV").fetchall() == [(1, 2), (3, 4)]


def test_read_remote_raises_extension_not_available_when_httpfs_disabled(csv_file, monkeypatch):
    monkeypatch.setenv("PYDUCK_SKIP_EXTENSIONS", "1")
    with pytest.raises(ExtensionNotAvailable) as excinfo:
        read_remote(csv_file, format="csv")
    assert "httpfs" in str(excinfo.value)


def test_read_remote_can_skip_httpfs_loading(csv_file, monkeypatch):
    monkeypatch.setenv("PYDUCK_SKIP_EXTENSIONS", "1")
    relation = read_remote(csv_file, format="csv", install_httpfs=False)
    assert relation.fetchall() == [(1, 2), (3, 4)]


# --------------------------------------------------------------------------- #
# list_remote
# --------------------------------------------------------------------------- #
def test_list_remote_lists_local_files(tmp_path):
    (tmp_path / "a.parquet").write_text("")
    (tmp_path / "b.parquet").write_text("")
    (tmp_path / "c.csv").write_text("")
    files = list_remote(str(tmp_path / "*.parquet"))
    assert sorted(files) == sorted([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")])


def test_list_remote_returns_an_empty_list_when_nothing_matches(tmp_path):
    assert list_remote(str(tmp_path / "*.parquet")) == []


def test_list_remote_accepts_a_connection(conn, tmp_path):
    (tmp_path / "a.parquet").write_text("")
    files = list_remote(str(tmp_path / "*.parquet"), conn=conn)
    assert files == [str(tmp_path / "a.parquet")]


def test_list_remote_expands_a_bare_directory(tmp_path):
    (tmp_path / "a.parquet").write_text("")
    (tmp_path / "b.parquet").write_text("")
    files = list_remote(str(tmp_path))
    assert sorted(files) == sorted([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")])


def test_list_remote_raises_extension_not_available_when_httpfs_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("PYDUCK_SKIP_EXTENSIONS", "1")
    with pytest.raises(ExtensionNotAvailable) as excinfo:
        list_remote(str(tmp_path / "*.parquet"))
    assert "httpfs" in str(excinfo.value)
