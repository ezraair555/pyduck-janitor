# Text & Similarity Verbs

`pyduck-janitor` ships three optional DuckDB-extension-backed verb
families for working with messy text and embedding-based search.
Each extension is **lazy-loaded**: the core package works fine
without it, and a missing extension surfaces as a clear
`ExtensionNotAvailable` pointing at the exact pip extra to install.

## At a glance

| Verb | Backend | What it does |
|---|---|---|
| `text_normalize` | `icu` | Lowercase, accent strip, whitespace collapse, Unicode normalize |
| `search_text` | `fts` | BM25-ranked full-text search |
| `keyword_filter` | `fts` | Boolean contains filter (any/all) |
| `build_fts_index` / `drop_fts_index` | `fts` | Manage the FTS index lifecycle |
| `embed_column` | `vss` + sentence-transformers | Compute embeddings for a text column |
| `build_vector_index` | `vss` | Create a HNSW index over embeddings |
| `vector_search` | `vss` | K-nearest-neighbor search |
| `fuzzy_dedupe` | `vss` | Find near-duplicate rows by embedding similarity |

## Installing

The extensions are part of the core DuckDB Python wheel; you only need
pyduck-janitor itself. Heavy extras (sentence-transformers) come via:

```bash
# Slim install — text_ops work, no embeddings
pip install pyduck-janitor

# Full install — embeddings + HNSW vector search
pip install pyduck-janitor[vss]

# Optional: bundle a model with the package via the companion wheel
pip install pyduck-janitor[vss] pyduck-janitor-embeddings
```

The package never downloads models without an explicit
`embed_install(...)` call.

## `text_normalize` (icu extension)

```python
import pandas as pd
from pyduck_janitor import DuckJanitor, text_normalize

df = pd.DataFrame({"name": ["Café", "NAÏVE", "  Hello World  "]})
dj = DuckJanitor.from_pandas(df)
clean = text_normalize(dj, "name").collect()
#           name
# 0          cafe
# 1          naive
# 2  hello world
```

All options:

| Parameter | Default | Effect |
|---|---|---|
| `target_columns` | overwrites source | Where to put the normalized text |
| `form` | `"NFKC"` | Unicode normalization form (NFC/NFD/NFKC/NFKD) |
| `strip_accents` | `True` | NFD normalize then drop combining marks |
| `lower` | `True` | ASCII-aware case folding |
| `collapse_whitespace` | `True` | Runs of whitespace → single space |
| `strip` | `True` | Trim leading/trailing whitespace |

Accent stripping happens in Python via `unicodedata.normalize('NFD', ...)`
because DuckDB's RE2 regex engine is byte-based and would mangle
multi-byte UTF-8. The other transforms push down into DuckDB.

## `search_text` and `build_fts_index` (fts extension)

```python
from pyduck_janitor import (
    DuckJanitor, build_fts_index, search_text,
)

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "text": [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be that is the question",
        "All that glitters is not gold",
        "The fox is quick and brown",
    ],
})
dj = DuckJanitor.from_pandas(df)
dj = build_fts_index(dj, "text")        # BM25, porter stemmer, english stopwords

results = search_text(dj, "text", "fox quick", top_k=3)
print(results)
#    __pyduck_rowid  id                                         text     score
# 0               5   5                   The fox is quick and brown  0.816063
# 1               1   1  The quick brown fox jumps over the lazy dog  0.597475
```

Options on `build_fts_index`:

- `stopwords` — `"english"` (default), `"none"`, or a custom list
- `stemmer` — `"porter"` (default), any of the 30+ language stemmers DuckDB ships
- `lower` — `True` (default), apply `lower()` before indexing
- `overwrite` — `True` (default), recreate the index if it exists
- `rowid_col` — `None`, defaults to a synthetic `__pyduck_rowid`

`search_text` returns a DataFrame by default; pass
`return_relation=True` to get a `DuckJanitor` for chaining.

## `keyword_filter`

```python
from pyduck_janitor import keyword_filter

filtered = keyword_filter(dj, "text", ["fox", "gold"], mode="any")
filtered_all = keyword_filter(dj, "text", ["quick", "fox"], mode="all")
```

`case_sensitive=True` makes both the column and phrases compared
verbatim; default is to lowercase both sides.

## Embedding model management

Three install modes:

```python
import pyduck_janitor as pj

# 1. Default — bundled model from pyduck-janitor-embeddings wheel
pj.embed_install()

# 2. Upgrade — pull a HuggingFace model
pj.embed_install("hf:BAAI/bge-small-en-v1.5")
pj.embed_install("hf:sentence-transformers/all-mpnet-base-v2")

# 3. Local path you've prepared
pj.embed_install("/opt/models/my-finetuned-encoder")
```

`embed_install` is **idempotent**: re-running on an installed model
is a no-op. Models are cached under
`~/.cache/pyduck-janitor/embeddings/` by default (override with the
`PYDUCK_EMBED_CACHE` env var).

Inspect and remove models:

```python
pj.embed_list_installed()  # DataFrame with model, path, size_bytes, ...
pj.embed_remove("hf:BAAI/bge-small-en-v1.5")
```

If `HF_TOKEN` is set, gated HuggingFace models are accessible
without extra configuration.

## `embed_column`, `build_vector_index`, `vector_search`

```python
from pyduck_janitor import (
    DuckJanitor, embed_column, build_vector_index, vector_search,
)

dj = DuckJanitor.from_pandas(df)
dj = embed_column(dj, "text")             # adds FLOAT[384] 'embedding' column
dj = build_vector_index(dj, metric="cosine")

# text query → embed → kNN
hits = vector_search(dj, "lazy dog", top_k=5)
```

`embed_column` requires the model to already be installed; it raises
`EmbeddingsNotAvailable` (not a silent download) if missing.

`vector_search` accepts either a text string (which is embedded on
the fly) or a pre-computed embedding vector.

## `fuzzy_dedupe`

```python
from pyduck_janitor import fuzzy_dedupe

deduped = fuzzy_dedupe(
    dj, columns=["text"],
    threshold=0.1,            # cosine distance threshold
    keep="first",             # one row per duplicate group
)
```

`fuzzy_dedupe` is O(N²) in DuckDB memory and is intended for batches
up to ~50k rows. For larger inputs, chunk and union-find externally.

## Why three extensions?

| Need | Pick |
|---|---|
| Strip accents, fix casing | `icu` (`text_normalize`) |
| Substring / regex match | `keyword_filter` |
| Stemming, stopwords, BM25 ranking | `fts` (`search_text`) |
| Near-duplicate detection | `vss` (`fuzzy_dedupe`) |
| Semantic / embedding similarity | `vss` (`vector_search`) |
| Hybrid: keyword + semantic re-rank | Chain `search_text` then `vector_search` |

## Error handling

Every verb raises a typed exception when its extension isn't
available or a model isn't installed:

```python
try:
    dj = text_normalize(dj, "name")
except ExtensionNotAvailable as exc:
    # exc.pip_extra == "icu"; install with `pip install pyduck-janitor[icu]`
    ...

try:
    dj = embed_column(dj, "text")
except EmbeddingsNotAvailable as exc:
    # exc.install_command is a copy-paste ready line
    print(exc.install_command)  # pyduck_janitor.embed_install(...)
```

No silent network calls, no half-failed batch jobs.
