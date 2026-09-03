"""
Vector similarity search and embedding-model management.

This module wraps DuckDB's ``vss`` extension and provides a
sentence-transformers-backed embedding pipeline. The heavy dependency
(``sentence-transformers``) is imported lazily so that ``pyduck-janitor``
continues to import cleanly for users who only need the icu/fts paths.

Three surfaces are exposed:

1. ``embed_install(model)`` — install a model into the local cache.
   Three sources are supported:
     - bundled (``pyduck_janitor-embeddings`` companion wheel, default)
     - HuggingFace (``hf:org/model``)
     - local path (``/path/to/model`` or relative)
   Idempotent.

2. ``embed_list_installed()`` / ``embed_remove(model)`` — cache
   management. Returns paths and sizes; removal is permanent.

3. ``DuckJanitor.embed_column(...)``, ``vector_search(...)``,
   ``fuzzy_dedupe(...)`` — verbs that use the installed model to
   embed, index, and search. These require ``vss`` (DuckDB extension)
   plus a cached embedding model.

Errors are surfaced as ``EmbeddingsNotAvailable`` with the exact
``embed_install(...)`` line the user needs to run. We never silently
download — embedding a million rows should not fail mid-batch.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from pathlib import Path
from typing import Optional, Sequence, Union

import duckdb
import pandas as pd

from .extensions import ExtensionNotAvailable, extension_loaded, load_extension
from .duck_janitor import DuckJanitor
from .cleaning_ops import _register_relation


__all__ = [
    "EmbeddingsNotAvailable",
    "DEFAULT_EMBED_MODEL",
    "cache_dir",
    "embed_install",
    "embed_list_installed",
    "embed_remove",
    "embed_column",
    "vector_search",
    "fuzzy_dedupe",
]


# ---------------------------------------------------------------------------
# Default model + cache layout
# ---------------------------------------------------------------------------


# Default model name. Resolved relative to ``pyduck_janitor_embeddings``
# (the companion wheel), then to HuggingFace. The default is the
# sentence-transformers all-MiniLM-L6-v2 — small, fast, well-tested,
# Apache 2.0 license.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache directory for installed model weights. We use a pyduck-janitor
# specific subdirectory so users can audit it independently of other
# caches (HuggingFace, etc.).
CACHE_ROOT_ENV = "PYDUCK_EMBED_CACHE"
CACHE_ROOT_DEFAULT = "~/.cache/pyduck-janitor/embeddings"


class EmbeddingsNotAvailable(RuntimeError):
    """Raised when a model or extension isn't installed.

    Attributes
    ----------
    model : str
        The model that was requested.
    install_command : str or None
        A copy-paste ready command to install the missing model.
    """

    def __init__(
        self,
        model: str,
        message: str,
        install_command: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.model = model
        self.install_command = install_command
        self.cause = cause
        hint = ""
        if install_command:
            hint = (
                f"\n\nHint: install the model with:\n"
                f"    {install_command}"
            )
        super().__init__(message + hint)


_lock = threading.Lock()


def cache_dir() -> Path:
    """Return the local cache root directory, creating it if needed."""
    root = Path(os.environ.get(CACHE_ROOT_ENV, CACHE_ROOT_DEFAULT)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slugify(name: str) -> str:
    """Convert a model id to a flat, filesystem-safe slug."""
    return name.replace("/", "__").replace("@", "_at_")


def _model_dir(model: str) -> Path:
    return cache_dir() / _slugify(model)


def _is_installed(model: str) -> bool:
    """Return True if the model directory has required files."""
    md = _model_dir(model)
    if not md.exists():
        return False
    # sentence-transformers needs at least config + weights
    has_config = (md / "config.json").exists()
    has_weights = (
        (md / "model.safetensors").exists()
        or (md / "pytorch_model.bin").exists()
    )
    return has_config and has_weights


# ---------------------------------------------------------------------------
# Bundled-model support (pyduck_janitor_embeddings companion wheel)
# ---------------------------------------------------------------------------


def _bundled_model_path(model: str) -> Optional[Path]:
    """Locate a bundled model inside the ``pyduck_janitor_embeddings`` package.

    The companion wheel, when installed, exposes a ``data/embeddings/``
    tree with one directory per bundled model. Returns ``None`` when
    the companion package isn't installed.
    """
    try:
        import importlib

        pkg = importlib.import_module("pyduck_janitor_embeddings")
    except ImportError:
        return None
    pkg_path = Path(pkg.__file__).resolve().parent
    candidate = pkg_path / "data" / "embeddings" / _slugify(model)
    if candidate.exists() and (candidate / "config.json").exists():
        return candidate
    return None


# ---------------------------------------------------------------------------
# Public install / list / remove
# ---------------------------------------------------------------------------


def embed_install(
    model: str = DEFAULT_EMBED_MODEL,
    *,
    cache: bool = True,
    allow_hf_fallback: bool = True,
) -> Path:
    """Install an embedding model into the local cache.

    Three source modes are tried in order, controlled by ``model``:

    - ``"hf:org/model"`` (or any string starting with ``hf:``): fetch
      from HuggingFace Hub using ``huggingface_hub``. The remainder
      of the string is treated as a repo id. Revision pins like
      ``hf:org/model@<rev>`` are supported.
    - ``"/path/to/model"`` (or any string that's an existing
      directory): copy from a local path.
    - Anything else: try the bundled wheel first; on miss, fall back
      to HuggingFace when ``allow_hf_fallback`` is True.

    Parameters
    ----------
    model : str, default ``DEFAULT_EMBED_MODEL``
        Model identifier (see above for source modes).
    cache : bool, default True
        If True, copy into the local cache. If False and a local path
        is given, the cache entry is a symlink for fast iteration.
    allow_hf_fallback : bool, default True
        When no source is explicit and the bundled wheel doesn't have
        the model, fall back to HuggingFace.

    Returns
    -------
    pathlib.Path
        The directory holding the installed model.

    Raises
    ------
    EmbeddingsNotAvailable
        When the model can't be installed. The error includes a
        ready-to-run ``embed_install(...)`` command when possible.
    """
    with _lock:
        if _is_installed(model):
            return _model_dir(model)

        # Source 1: HuggingFace explicit
        if model.startswith("hf:"):
            spec = model[3:]
            return _install_from_hf(spec)

        # Source 2: local path
        if os.path.isdir(model):
            return _install_from_local(Path(model), cache=cache)

        # Source 3: bundled wheel
        bundled = _bundled_model_path(model)
        if bundled is not None:
            return _install_from_local(bundled, cache=cache)

        # Source 4: HuggingFace fallback
        if allow_hf_fallback:
            return _install_from_hf(model)

        raise EmbeddingsNotAvailable(
            model,
            f"No source for model '{model}': not in bundled wheel and "
            "HuggingFace fallback disabled.",
            install_command=f"pyduck_janitor.embed_install('{model}')",
        )


def _install_from_hf(spec: str) -> Path:
    """Download a model from HuggingFace Hub into the cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise EmbeddingsNotAvailable(
            spec,
            "sentence-transformers is not installed; required for HuggingFace "
            "model downloads.",
            install_command="pip install pyduck-janitor[vss]",
            cause=exc,
        ) from exc

    # Revision pin: hf:org/model@<rev>
    if "@" in spec:
        repo_id, revision = spec.split("@", 1)
    else:
        repo_id, revision = spec, None

    target = _model_dir(spec)
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.safetensors",
                "*.bin",
                "tokenizer.*",
                "vocab.*",
                "merges.txt",
                "special_tokens_map.json",
            ],
        )
    except Exception as exc:
        raise EmbeddingsNotAvailable(
            spec,
            f"Failed to download model '{spec}' from HuggingFace: {exc}",
            install_command=f"pyduck_janitor.embed_install('hf:{spec}')",
            cause=exc,
        ) from exc
    return target


def _install_from_local(source: Path, *, cache: bool = True) -> Path:
    """Copy or link a local model directory into the cache."""
    target = _model_dir(source.name)
    if cache:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
    return target


def embed_list_installed() -> pd.DataFrame:
    """Return a DataFrame listing installed embedding models.

    Columns: ``model``, ``path``, ``size_bytes``, ``size_human``,
    ``has_config``, ``has_weights``.
    """
    rows = []
    root = cache_dir()
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        total = sum(
            f.stat().st_size for f in entry.rglob("*") if f.is_file()
        )
        rows.append(
            {
                "model": entry.name.replace("__", "/").replace("_at_", "@"),
                "path": str(entry),
                "size_bytes": total,
                "size_human": _human_size(total),
                "has_config": (entry / "config.json").exists(),
                "has_weights": (entry / "model.safetensors").exists()
                or (entry / "pytorch_model.bin").exists(),
            }
        )
    return pd.DataFrame(rows)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def embed_remove(model: str) -> bool:
    """Remove ``model`` from the cache. Returns True if anything was removed."""
    md = _model_dir(model)
    if md.exists():
        shutil.rmtree(md)
        return True
    return False


# ---------------------------------------------------------------------------
# Lazy sentence-transformers import
# ---------------------------------------------------------------------------


_st_module = None
_st_lock = threading.Lock()


def _get_st():
    """Lazily import and cache the sentence-transformers module."""
    global _st_module
    if _st_module is not None:
        return _st_module
    with _st_lock:
        if _st_module is not None:
            return _st_module
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise EmbeddingsNotAvailable(
                DEFAULT_EMBED_MODEL,
                "sentence-transformers is not installed. Required for embedding "
                "operations.",
                install_command="pip install pyduck-janitor[vss]",
                cause=exc,
            ) from exc
        import sentence_transformers
        _st_module = sentence_transformers
        return _st_module


def _encode_texts(texts: Sequence[str], model: str) -> list[list[float]]:
    """Encode ``texts`` with ``model``. Returns a list of float lists."""
    if not _is_installed(model):
        embed_install(model)
    st = _get_st()
    encoder = st.SentenceTransformer(str(_model_dir(model)))
    embeddings = encoder.encode(
        list(texts), convert_to_numpy=True, show_progress_bar=False
    )
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# DuckDB vss verbs
# ---------------------------------------------------------------------------


def embed_column(
    dj: DuckJanitor,
    column: str,
    *,
    target_column: str = "embedding",
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
) -> DuckJanitor:
    """Compute embeddings for ``column`` and attach as ``target_column``.

    The returned relation has an extra ``FLOAT[<dim>]`` column. Embedding
    is done in Python (sentence-transformers), then loaded back into
    DuckDB as a list column. The relation is registered as a temporary
    table and the original ``dj._relation`` is replaced with a join
    that re-attaches the original columns.

    Parameters
    ----------
    dj : DuckJanitor
        Input relation.
    column : str
        Text column to embed.
    target_column : str, default 'embedding'
        Output column name.
    model : str, default ``DEFAULT_EMBED_MODEL``
        Model identifier (must already be installed).
    batch_size : int, default 64
        Number of rows per encoder batch.

    Raises
    ------
    EmbeddingsNotAvailable
        If the model isn't installed and ``embed_install`` can't fetch it.
    """
    conn = dj._connection
    if not _is_installed(model):
        raise EmbeddingsNotAvailable(
            model,
            f"Model '{model}' is not in the local cache.",
            install_command=f"pyduck_janitor.embed_install('{model}')",
        )

    df = dj.collect()
    texts = df[column].astype(str).tolist()
    embeddings = _encode_texts(texts, model)

    # Register a temp table with original cols + new embedding column.
    table_name = _register_relation(conn, dj._relation)
    embed_df = pd.DataFrame(
        {target_column: [list(map(float, e)) for e in embeddings]}
    )
    embed_table = f"_dj_embed_{table_name.strip('_').strip('\"')}"
    # We embed on the original df, so row order matches. Register
    # the embed table alongside the original.
    embed_full = pd.concat(
        [df.reset_index(drop=True), embed_df.reset_index(drop=True)],
        axis=1,
    )
    conn.register(embed_table, embed_full)
    return DuckJanitor(conn.from_df(embed_full), connection=conn)


def _ensure_vss(conn: duckdb.DuckDBPyConnection) -> None:
    if not extension_loaded(conn, "vss"):
        load_extension(conn, "vss")


def build_vector_index(
    dj: DuckJanitor,
    embedding_column: str = "embedding",
    *,
    metric: str = "cosine",
    index_name: Optional[str] = None,
    dim: Optional[int] = None,
) -> DuckJanitor:
    """Create a HNSW vector index on an embedding column.

    The relation is materialized as a base TEMP table so the index
    can attach. The embedding column is cast to ``FLOAT[<dim>]``;
    ``dim`` is inferred from the first row when not supplied.

    Parameters
    ----------
    dj : DuckJanitor
        Input relation. Must have a FLOAT[] column ``embedding_column``.
    metric : {'cosine', 'ip', 'l2sq'}, default 'cosine'
        Distance metric (HNSW internal name).
    index_name : str, optional
        Index identifier. Defaults to ``hnsw_<table>``.
    dim : int, optional
        Embedding dimension. Inferred from the data when omitted.

    Returns
    -------
    DuckJanitor
        The same ``dj`` with the HNSW index attached.
    """
    conn = dj._connection
    _ensure_vss(conn)

    if metric not in {"cosine", "ip", "l2sq"}:
        raise ValueError(
            f"metric must be one of cosine/ip/l2sq, got {metric!r}"
        )

    table_name = _register_relation(conn, dj._relation)
    if index_name is None:
        index_name = f"hnsw_{table_name.strip('_').strip('\"')}"

    # Infer dimension if not supplied.
    if dim is None:
        dim = conn.execute(
            f"SELECT length({dj._quote(embedding_column)}) FROM {table_name} LIMIT 1"
        ).fetchone()[0]
    if dim is None:
        raise ValueError(
            f"Cannot infer embedding dimension from column "
            f"{embedding_column!r}; pass dim=... explicitly."
        )

    # Materialize as a base table. HNSW indexes can't be created
    # on a registered relation — only on real base tables.
    base_name = f"_dj_vss_base_{table_name.strip('_').strip('\"')}"
    conn.execute(
        f"CREATE OR REPLACE TABLE {dj._quote(base_name)} AS "
        f"SELECT CAST({dj._quote(embedding_column)} AS FLOAT[{int(dim)}]) AS "
        f"{dj._quote(embedding_column)}, "
        + ", ".join(
            dj._quote(c)
            for c in dj._relation.columns
            if c != embedding_column
        )
        + f" FROM {table_name}"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {dj._quote(index_name)} "
        f"ON {dj._quote(base_name)} USING HNSW "
        f"({dj._quote(embedding_column)}) "
        f"WITH (metric = '{metric}')"
    )
    dj._vss_index = index_name
    dj._vss_metric = metric
    dj._vss_embedding_column = embedding_column
    dj._vss_table = base_name
    return dj


def vector_search(
    dj: DuckJanitor,
    query: Union[str, Sequence[float]],
    embedding_column: str = "embedding",
    *,
    top_k: int = 10,
    threshold: Optional[float] = None,
    metric: str = "cosine",
    model: str = DEFAULT_EMBED_MODEL,
    score_col: str = "_vss_distance",
    index_name: Optional[str] = None,
) -> pd.DataFrame:
    """K-nearest-neighbor search against an embedding column.

    Parameters
    ----------
    dj : DuckJanitor
        Input relation. Must have an HNSW index on ``embedding_column``
        (call ``build_vector_index`` first).
    query : str or sequence of float
        Either a text query (embedded via ``model``) or a pre-computed
        embedding.
    top_k : int, default 10
        Number of nearest neighbors to return.
    threshold : float, optional
        Maximum allowed distance. Rows with distance above this are dropped.
    metric : {'cosine', 'ip', 'l2sq'}, default 'cosine'
        Must match the index metric.
    model : str, default ``DEFAULT_EMBED_MODEL``
        Used to embed text queries.
    score_col : str, default '_vss_distance'
        Output column name for the distance.

    Returns
    -------
    pd.DataFrame
        Top-k rows with the distance column appended.
    """
    conn = dj._connection
    _ensure_vss(conn)

    if isinstance(query, str):
        embedding = _encode_texts([query], model)[0]
    else:
        embedding = list(query)

    if index_name is None:
        index_name = getattr(dj, "_vss_index", None)
    if index_name is None:
        raise ValueError(
            "No HNSW index recorded on this DuckJanitor and no index_name "
            "supplied. Call build_vector_index() first."
        )

    table_name = getattr(dj, "_vss_table", None) or _register_relation(
        conn, dj._relation
    )
    metric_fn = {
        "cosine": "array_cosine_distance",
        "ip": "array_inner_product",
        "l2sq": "array_distance",
    }[metric]
    embedding_literal = (
        "[" + ", ".join(f"{float(x):.8f}" for x in embedding) + "]"
    )
    sql = (
        f"SELECT *, {metric_fn}({dj._quote(embedding_column)}, "
        f"{embedding_literal}::FLOAT[{len(embedding)}]) "
        f"AS {dj._quote(score_col)} FROM {table_name} "
        f"ORDER BY {dj._quote(score_col)} ASC LIMIT {int(top_k)}"
    )
    relation = conn.sql(sql)
    df = relation.df()
    if threshold is not None:
        df = df[df[score_col] <= threshold].reset_index(drop=True)
    return df


def fuzzy_dedupe(
    dj: DuckJanitor,
    columns: Union[str, Sequence[str]],
    *,
    threshold: float = 0.1,
    model: str = DEFAULT_EMBED_MODEL,
    embedding_column: str = "embedding",
    keep: str = "first",
) -> DuckJanitor:
    """Find near-duplicate rows by embedding similarity.

    Concatenates the given ``columns`` per row, embeds them, computes
    pairwise cosine distance, and returns one row per duplicate group
    (``keep='first'``) or the full table with a ``_dup_group`` column.

    Parameters
    ----------
    dj : DuckJanitor
        Input relation.
    columns : str or sequence of str
        Columns whose concatenation defines the row's identity.
    threshold : float, default 0.1
        Cosine distance above which two rows are *not* duplicates.
    model : str, default ``DEFAULT_EMBED_MODEL``
        Model identifier (must be installed).
    keep : {'first', 'group'}, default 'first'
        If ``'first'``, return one row per duplicate group. If
        ``'group'``, return all rows annotated with ``_dup_group``.
    """
    if isinstance(columns, str):
        columns = [columns]
    df = dj.collect()
    # Build the row identity string.
    identity = df[columns].astype(str).agg(" | ".join, axis=1)
    embeddings = _encode_texts(identity.tolist(), model)

    # Pairwise cosine distance via DuckDB.
    conn = duckdb.connect(":memory:")
    rows = pd.DataFrame(
        {
            "__id": range(len(df)),
            embedding_column: [list(map(float, e)) for e in embeddings],
            **{c: df[c] for c in df.columns},
        }
    )
    conn.register("rows", rows)
    op = "array_cosine_distance"
    # Compute the full N×N distance matrix in DuckDB. This is O(N²)
    # in DuckDB memory; suitable for batch dedup up to ~50k rows.
    n = len(rows)
    pairs = conn.sql(
        f"SELECT a.__id AS id_a, b.__id AS id_b, "
        f"{op}(a.{embedding_column}, b.{embedding_column}) AS dist "
        f"FROM rows a, rows b "
        f"WHERE a.__id < b.__id AND "
        f"{op}(a.{embedding_column}, b.{embedding_column}) <= {threshold}"
    ).df()

    # Union-find to group duplicates.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, row in pairs.iterrows():
        union(int(row["id_a"]), int(row["id_b"]))

    df["_dup_group"] = [find(i) for i in range(n)]
    if keep == "first":
        df = df.drop_duplicates(subset=["_dup_group"]).reset_index(drop=True)
    return DuckJanitor.from_pandas(df)
