"""
Text and full-text-search verbs for pyduck-janitor.

This module wraps DuckDB's ``icu`` and ``fts`` extensions into
pyduck-janitor verbs. Each function operates on a ``DuckJanitor``
instance and returns a new ``DuckJanitor`` (or a DataFrame for the
search verbs).

Verbs
-----
``text_normalize`` (icu)
    Lowercase, strip accents, collapse whitespace, normalize Unicode.
``search_text`` (fts)
    BM25-ranked text search over a column with optional full-text index.
``keyword_filter`` (fts)
    Boolean filter: keep rows whose text contains any/all of the phrases.
``build_fts_index`` / ``drop_fts_index``
    Manage the underlying FTS inverted index.

The ``search_text`` and ``keyword_filter`` verbs both lazily install and
load the ``fts`` extension on the connection. ``text_normalize`` does
the same with ``icu``. Failure surfaces as ``ExtensionNotAvailable``
with the exact pip extra to install.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union
import unicodedata

import duckdb
import pandas as pd

from .extensions import ExtensionNotAvailable, extension_loaded, load_extension
from .duck_janitor import DuckJanitor
from .cleaning_ops import _register_relation


__all__ = [
    "text_normalize",
    "build_fts_index",
    "drop_fts_index",
    "search_text",
    "keyword_filter",
]


# Default stopword list when the user doesn't supply one. DuckDB's
# default FTS stopword set is "english"; we keep the same name.
DEFAULT_STOPWORDS = "english"
DEFAULT_STEMMER = "porter"


def _fts_index_name(table_name: str) -> str:
    """Compute the FTS index name for a table.

    DuckDB names the index ``fts_main_<table>`` literally, including
    any underscores in the table name. We only strip surrounding
    quotes here.
    """
    clean = table_name.strip('"')
    return f"fts_main_{clean}"


def text_normalize(
    dj: DuckJanitor,
    columns: Union[str, Sequence[str]],
    *,
    target_columns: Optional[Union[str, Sequence[str]]] = None,
    form: str = "NFKC",
    strip_accents: bool = True,
    lower: bool = True,
    collapse_whitespace: bool = True,
    strip: bool = True,
) -> DuckJanitor:
    """Normalize text columns: lowercase, accent strip, whitespace collapse.

    Wraps DuckDB's ``icu`` extension where available. Accent stripping
    is done in Python via ``unicodedata.normalize('NFD', ...)`` plus
    a combining-mark filter — DuckDB's RE2 regex engine is byte-based
    and can't reliably remove accents without mangling multi-byte
    characters. The other transforms push down into DuckDB.

    Parameters
    ----------
    dj : DuckJanitor
        Input relation.
    columns : str or sequence of str
        Source text columns.
    target_columns : str or sequence of str, optional
        Output column names. Defaults to overwriting the source columns.
    form : {'NFC', 'NFD', 'NFKC', 'NFKD'}, default 'NFKC'
        Unicode normalization form. ``NFKC`` is the safest default for
        general text cleaning.
    strip_accents : bool, default True
        NFD normalize then drop combining diacritical marks.
    lower : bool, default True
        Apply ``lower()`` for ASCII-aware case folding.
    collapse_whitespace : bool, default True
        Collapse runs of whitespace into a single space.
    strip : bool, default True
        Strip leading/trailing whitespace.

    Returns
    -------
    DuckJanitor
        New relation with normalized columns.

    Examples
    --------
    >>> dj = DuckJanitor.from_pandas(df)
    >>> dj = text_normalize(dj, "name", strip_accents=True, lower=True)
    """
    conn = dj._connection
    if not extension_loaded(conn, "icu"):
        load_extension(conn, "icu")

    table_name = _register_relation(conn, dj._relation)

    if isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)
    if target_columns is None:
        target_columns = list(columns)
    elif isinstance(target_columns, str):
        target_columns = [target_columns]
    else:
        target_columns = list(target_columns)

    if len(columns) != len(target_columns):
        raise ValueError(
            f"columns ({len(columns)}) and target_columns "
            f"({len(target_columns)}) must have the same length"
        )

    form = form.upper()
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"form must be one of NFC/NFD/NFKC/NFKD, got {form!r}")

    # Non-NFC form normalization and accent stripping are done in Python
    # to preserve Unicode semantics and NULL values.
    sql_sources = list(columns)
    if strip_accents or form != "NFC":
        def _normalize_scalar(v: object) -> object:
            if pd.isna(v):
                return v
            normalized = unicodedata.normalize(form, str(v))
            if strip_accents:
                normalized = "".join(
                    c
                    for c in unicodedata.normalize("NFD", normalized)
                    if not unicodedata.combining(c)
                )
            return normalized

        df = dj.collect()
        transformed = df.copy()
        for src, dst in zip(columns, target_columns):
            transformed[dst] = df[src].map(_normalize_scalar)

        temp_name = f"_dj_textnorm_{table_name.strip('_').strip('\"')}"
        conn.register(temp_name, transformed)
        table_name = temp_name
        sql_sources = list(target_columns)

    pieces: list[str] = []
    for src, dst in zip(sql_sources, target_columns):
        # Build the expression by composing the transforms. Each
        # transform takes the previous expression and wraps it.
        expr: str = dj._quote(src)

        if lower:
            expr = f"lower({expr})"

        if collapse_whitespace:
            expr = f"regexp_replace({expr}, '\\s+', ' ', 'g')"

        if strip:
            expr = f"trim({expr})"

        pieces.append(f"{expr} AS {dj._quote(dst)}")

    # When the target column matches a source column, drop the
    # source from the projection so we overwrite instead of adding
    # name_1 alongside name.
    base_columns = (
        list(dj._relation.columns)
        if not (strip_accents or form != "NFC")
        else list(transformed.columns)
    )
    replacing = set(target_columns)
    other_cols = [
        dj._quote(c)
        for c in base_columns
        if c not in replacing
    ]
    prefix = ", ".join(other_cols) + ", " if other_cols else ""
    select_sql = (
        f"SELECT {prefix}"
        + ", ".join(pieces)
        + f" FROM {table_name}"
    )
    new_relation = conn.sql(select_sql)
    return DuckJanitor(new_relation, connection=conn)


# ---------------------------------------------------------------------------
# Full-text search (fts extension)
# ---------------------------------------------------------------------------


def _fts_create_args(
    stopwords: str,
    stemmer: str,
    lower: bool,
    overwrite: bool,
) -> str:
    """Build the option string for CREATE FTS INDEX ... """
    parts: list[str] = []
    if stemmer != "porter":
        parts.append(f"stemmer = '{stemmer}'")
    if stopwords != DEFAULT_STOPWORDS:
        parts.append(f"stopwords = '{stopwords}'")
    if not lower:
        parts.append("lower = false")
    if overwrite:
        parts.append("overwrite = true")
    return ", ".join(parts)


def build_fts_index(
    dj: DuckJanitor,
    columns: Union[str, Sequence[str]],
    *,
    stopwords: str = DEFAULT_STOPWORDS,
    stemmer: str = DEFAULT_STEMMER,
    lower: bool = True,
    overwrite: bool = True,
    rowid_col: Optional[str] = None,
) -> DuckJanitor:
    """Create a DuckDB FTS index over one or more columns.

    Parameters
    ----------
    dj : DuckJanitor
        Input relation.
    columns : str or sequence of str
        Columns to include in the index.
    stopwords : str, default 'english'
        Stopword set. See DuckDB docs for available sets.
    stemmer : str, default 'porter'
        Stemmer algorithm.
    lower : bool, default True
        Apply ``lower()`` before indexing.
    overwrite : bool, default True
        Recreate the index if it already exists.
    rowid_col : str, optional
        Name of the unique row-id column. Defaults to ``__pyduck_rowid``,
        a synthetic column added by ``build_fts_index``.

    Returns
    -------
    DuckJanitor
        The same ``dj``, after the index has been registered. The
        index lives in DuckDB's catalog and is referenced by
        ``fts_main_<table>``.
    """
    conn = dj._connection
    if not extension_loaded(conn, "fts"):
        load_extension(conn, "fts")

    if isinstance(columns, str):
        columns = [columns]

    # Materialize the relation as a TEMP table so FTS has a stable
    # target (FTS refuses to index views / arbitrary relations).
    table_name = _register_relation(conn, dj._relation)
    material_name = f"_dj_fts_material_{table_name.strip('_').strip('\"')}"
    if rowid_col is None:
        rowid_col = "__pyduck_rowid"
    conn.execute(
        f"CREATE OR REPLACE TABLE {dj._quote(material_name)} AS "
        f"SELECT row_number() OVER () AS {dj._quote(rowid_col)}, "
        f"{', '.join(dj._quote(c) for c in dj._relation.columns)} "
        f"FROM {table_name}"
    )

    cols_sql = ", ".join(f"'{c}'" for c in columns)
    opts = _fts_create_args(stopwords, stemmer, lower, overwrite)
    create_sql = (
        f"PRAGMA create_fts_index("
        f"'{material_name}', '{rowid_col}', {cols_sql}"
        + (f", {opts}" if opts else "")
        + ")"
    )
    conn.execute(create_sql)

    # Stash the index name so search_text() can find it.
    dj._fts_index = _fts_index_name(material_name)
    dj._fts_table = material_name
    return dj


def drop_fts_index(
    dj: DuckJanitor,
    table_name: Optional[str] = None,
) -> DuckJanitor:
    """Drop an FTS index previously created by ``build_fts_index``.

    DuckDB's ``drop_fts_index`` operates on the underlying material
    table, not on the index name. If ``table_name`` is not supplied,
    uses the material table recorded on ``dj`` by ``build_fts_index``.

    Returns the same ``DuckJanitor`` for chaining.
    """
    conn = dj._connection
    if not extension_loaded(conn, "fts"):
        load_extension(conn, "fts")
    name = table_name or getattr(dj, "_fts_table", None)
    if name is None:
        raise ValueError(
            "No FTS material table recorded on this DuckJanitor and no "
            "table_name supplied. Call build_fts_index() first or pass "
            "table_name=... explicitly."
        )
    # Drop the material table that hosts the FTS index.
    conn.execute(f"DROP TABLE IF EXISTS {dj._quote(name)}")
    if hasattr(dj, "_fts_index"):
        del dj._fts_index
    if hasattr(dj, "_fts_table"):
        del dj._fts_table
    return dj


def search_text(
    dj: DuckJanitor,
    column: str,
    query: str,
    *,
    index_name: Optional[str] = None,
    top_k: Optional[int] = None,
    score_col: Optional[str] = "score",
    threshold: Optional[float] = None,
    return_relation: bool = False,
) -> Union[DuckJanitor, pd.DataFrame]:
    """BM25-ranked text search over a column.

    Parameters
    ----------
    dj : DuckJanitor
        Input relation. Must already have an FTS index covering
        ``column`` (call ``build_fts_index`` first).
    column : str
        Column to search.
    query : str
        Free-text query. Quoted with DuckDB FTS rules (phrase = quoted).
    index_name : str, optional
        Name of the FTS index to use. Defaults to the index recorded
        by ``build_fts_index`` on ``dj``.
    top_k : int, optional
        If set, return only the ``top_k`` highest-scoring rows.
    score_col : str or None, default 'score'
        Name of the column to attach the BM25 score to. ``None`` to omit.
    threshold : float, optional
        Minimum score to include. DuckDB FTS scores are non-negative,
        with higher = more relevant.
    return_relation : bool, default False
        If True, return a ``DuckJanitor``; otherwise return a DataFrame.

    Returns
    -------
    DuckJanitor or pd.DataFrame
        Search results, sorted by score descending.
    """
    conn = dj._connection
    if not extension_loaded(conn, "fts"):
        load_extension(conn, "fts")

    if index_name is None:
        index_name = getattr(dj, "_fts_index", None)
    if index_name is None:
        raise ValueError(
            "No FTS index recorded on this DuckJanitor and no index_name "
            "supplied. Call build_fts_index() first or pass index_name=... "
            "explicitly."
        )

    material_table = getattr(dj, "_fts_table", None)
    if material_table is None:
        raise ValueError(
            "No FTS material table recorded on this DuckJanitor. "
            "Call build_fts_index() first."
        )

    # Escape single quotes in the query (DuckDB string literal).
    safe_query = query.replace("'", "''")
    bm25 = f"{index_name}.match_bm25(__pyduck_rowid, '{safe_query}')"

    select_cols = "*"
    if score_col is not None:
        select_cols = f"*, {bm25} AS {dj._quote(score_col)}"
    sql = (
        f"SELECT {select_cols} FROM {material_table} "
        f"WHERE {bm25} IS NOT NULL"
    )
    if threshold is not None:
        sql += f" AND {bm25} >= {threshold}"
    sql += f" ORDER BY {bm25} DESC"
    if top_k is not None:
        sql += f" LIMIT {int(top_k)}"

    relation = conn.sql(sql)
    if return_relation:
        return DuckJanitor(relation, connection=conn)
    return relation.df()


def keyword_filter(
    dj: DuckJanitor,
    column: str,
    phrases: Union[str, Sequence[str]],
    *,
    mode: str = "any",
    case_sensitive: bool = False,
) -> DuckJanitor:
    """Boolean contains filter: keep rows whose text contains any/all phrases.

    Faster than ``filter_string`` for repeated queries because it pushes
    the matching into DuckDB's vectorized engine. For ranked results
    with stemming and stopwords, use ``search_text`` instead.

    Parameters
    ----------
    dj : DuckJanitor
        Input relation.
    column : str
        Column to filter on.
    phrases : str or sequence of str
        Phrases to look for.
    mode : {'any', 'all'}, default 'any'
        Whether to require any phrase (OR) or all phrases (AND).
    case_sensitive : bool, default False
        If False, both column and phrases are lower-cased before matching.

    Returns
    -------
    DuckJanitor
        Filtered relation.
    """
    conn = dj._connection
    if isinstance(phrases, str):
        phrases = [phrases]
    if mode not in {"any", "all"}:
        raise ValueError(f"mode must be 'any' or 'all', got {mode!r}")

    table_name = _register_relation(conn, dj._relation)
    col = dj._quote(column)
    parts: list[str] = []
    for p in phrases:
        esc = p.replace("'", "''")
        if case_sensitive:
            parts.append(f"{col} LIKE '%' || '{esc}' || '%'")
        else:
            parts.append(f"lower({col}) LIKE '%' || lower('{esc}') || '%'")
    joiner = " OR " if mode == "any" else " AND "
    where = joiner.join(parts)
    relation = conn.sql(
        f"SELECT * FROM {table_name} WHERE {where}"
    )
    return DuckJanitor(relation, connection=conn)
