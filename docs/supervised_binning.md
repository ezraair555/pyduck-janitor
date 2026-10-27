# Supervised binning with `optimize_bins_to_target`

Learn optimal cut points for numeric columns **from their relationship with a
target column** — the supervised counterpart to the unsupervised
[`bin_numeric()`](api/functions.md#bin_numeric) (quantile / uniform splits).

```python
import pandas as pd
from pyduck_janitor import DuckJanitor

df = pd.DataFrame({
    "age": [22, 25, 31, 38, 44, 52, 61, 70] * 25,
    "bad_flag": [0, 0, 0, 0, 1, 1, 1, 1] * 25,
})
dj_binned, diag = (
    DuckJanitor.from_pandas(df)
    .optimize_bins_to_target("bad_flag", columns=["age"],
                             holdout=0.2, return_diagnostics=True)
)
print(diag["age"]["cuts"])   # learned split points, e.g. [41.0]
```

Each requested column gets a new `<column>_binned` text column with interval
labels such as `(-inf, 26.5]`, `(26.5, 47.5]`, …, plus `Missing` for NULLs.
The CASE expression is applied inside DuckDB, so the full relation is binned
even when cuts were learned on a sample.

## Installation

```bash
pip install "pyduck-janitor[binning]"   # optbinning + feature_engine + scikit-learn + scipy
```

Without the extra, `method="tree"` still works via a built-in sklearn
fallback, and `method="auto"` degrades gracefully (OptBinning skipped when
the package is missing).

## Quick start: binary target

```python
import duckdb
import numpy as np
import pandas as pd
from pyduck_janitor import DuckJanitor

rng = np.random.default_rng(42)
n = 6000
ages = rng.integers(18, 78, n)
inc = rng.lognormal(10.7, 0.35, n)
logit = -2.8 + 0.045 * (ages - 40) - 0.00003 * inc + (ages < 28) * 1.2
bad = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"age": ages, "income": inc, "bad": bad})

dj, diag = (
    DuckJanitor.from_pandas(df)
    .optimize_bins_to_target("bad", columns=["age", "income"],
                             holdout=0.2, return_diagnostics=True)
)
print(diag["age"]["cuts"])                    # e.g. [27.5, 32.5, 36.5, 47.5, 67.5]
print(diag["age"]["scores_holdout"]["auc"])   # honest test AUC, e.g. 0.63
print(dj.collect()[["age", "age_binned"]].head(3).to_string(index=False))
```

Expected shape of output:

```
 age   age_binned
  23  (-inf, 27.5]
  64  (47.5, 67.5]
  57  (47.5, 67.5]
```

## Which engine runs? (`method`)

| `method` | Engine | Binary | Multiclass | Continuous |
|---|---|---|---|---|
| `"auto"` (default) | OptBinning first, tree fallback | ✅ | tree | tree |
| `"optbinning"` | `optbinning.OptimalBinning` (IV/WoE-optimal) | ✅ only | falls back to tree + warning | falls back to tree + warning |
| `"tree"` | `feature_engine.DecisionTreeDiscretiser` (or sklearn fallback) | ✅ | ✅ | ✅ (regression tree) |

`method="auto"` prefers OptBinning for binary targets because it optimizes
Information Value under business constraints (`max_n_bins`, `min_bin_size`,
`monotonic_trend`) and its verdicts are stable and explainable. Two
auto-fallbacks keep it robust:

1. **OptBinning unavailable or errors** → tree, with a warning.
2. **OptBinning returns zero splits** (its conservative answer on perfect
   separation or very weak IV) → tree gets one chance; if the tree also finds
   nothing, the single bin stands and `method` stays `"optbinning"`.

Pass `method="optbinning"` explicitly when you want the IV-optimal answer or
nothing (a hard `RuntimeError` if OptBinning cannot run). Pass
`method="tree"` when you always want splits, or for non-binary targets.

`optbinning_kwargs` forwards constructor options, e.g.
`optbinning_kwargs={"max_n_bins": 4, "monotonic_trend": "ascending"}`.
Diagnostics record the effective `solver` (`cp` when `ortools` is installed,
else `mip`); a failing solver is retried once with the alternate backend.

## Target types (`task`)

Target kind is auto-detected (NaNs ignored):

- **binary** — exactly 2 distinct non-null values, any dtype (ints, strings,
  bools all work; string labels are factorized to 0/1 in sorted order).
- **multiclass** — 3+ string/bool/categorical values; ints up to 100 distinct;
  floats up to 50 distinct.
- **continuous** — floats with >50 distinct values, ints with >100 distinct.

Override with `task="binary" | "multiclass" | "regression"`. A mismatch
raises immediately (`task='binary'` on 3-class y → `ValueError`) instead of
failing deep inside cross-validation.

```python
# Multiclass: income tier as target
df["risk_grade"] = np.select([df["income"] < 30_000, df["income"] < 60_000], [0, 1], 2)
dj_m, d_m = (
    DuckJanitor.from_pandas(df)
    .optimize_bins_to_target("risk_grade", columns=["income"],
                             method="tree", holdout=True, return_diagnostics=True)
)
print(d_m["income"]["scores_holdout"])
# {'f1_macro': 1.0, 'balanced_accuracy': 1.0, 'n_bins': 3, 'n': 1200}

# Continuous: bin age against a price target
df["price"] = np.log(df["income"]) + rng.normal(0, 0.30, n)
dj_c, d_c = (
    DuckJanitor.from_pandas(df)
    .optimize_bins_to_target("price", columns=["age"],
                             method="tree", holdout=0.2, return_diagnostics=True)
)
print(d_c["age"]["scores_holdout"])
# {'r2': ..., 'rmse': ..., 'n_bins': 4, 'n': 1200}
```

## Columns, labels, and in-place mode

- `columns` accepts one name or a list; default is every numeric non-target
  column. Non-numeric columns are **skipped with a warning** (categoricals
  are already discrete) and recorded as `"method": "skipped"` in diagnostics.
- `cast_to_text=True` (default) emits `VARCHAR` interval labels. Use
  `cast_to_text=False` for integer bin ids `1..k` (handy before one-hot
  encoding or WoE mapping).
- `suffix="_binned"` names outputs `<col>_binned`; `suffix=""` replaces the
  source column in place (via `SELECT * EXCLUDE`). An existing target name
  raises `ValueError` rather than silently duplicating the column.
- NULL inputs always map to a `Missing` bin; a feature with no usable splits
  yields a single `Single` bin.

## Honest holdout scoring

`holdout=False` (default) learns cuts on all rows. Pass `holdout=True`
(20% test) or a fraction in (0, 0.5) for an honest evaluation:

1. stratified train/test split (for discrete y; plain shuffle for continuous),
2. cuts learned on **train only**,
3. per-bin statistics (event rate / majority class / bin mean) learned on
   **train only**,
4. scores computed on **test** by mapping test rows through train statistics
   (unseen bins fall back to the train global stat).

The returned relation carries the **train cuts applied to all rows**, and
diagnostics gain `scores_holdout`, `holdout_n`, `train_n`, plus `learned_on`
(`"train n=..."` vs `"full n=..."`).

Metrics per target kind (all from cheap per-bin statistics — no model fit):

| Target | Metrics |
|---|---|
| binary | `auc`, `f1` (0.5 threshold on per-bin event rate), `gini = 2·AUC−1`, `cramers_v` |
| multiclass | `f1_macro`, `balanced_accuracy` (per-bin majority vote) |
| continuous | `r2`, `rmse` (per-bin mean predictor) |

```python
dj_h, d_h = (
    DuckJanitor.from_pandas(df)
    .optimize_bins_to_target("bad", columns=["age"], holdout=0.2,
                             return_diagnostics=True)
)
print(d_h["age"]["learned_on"])       # 'train n=4800'
print(d_h["age"]["scores_holdout"])   # {'auc': 0.63, 'f1': ..., ...}
```

## Large tables (`sample_n`)

Cut learning materialises the requested columns into pandas. To bound memory,
`sample_n=100_000` (default) learns cuts on a stratified random sample past
that size; the resulting CASE is still applied to **every row** in DuckDB.
Diagnostics record `fit_rows`, `total_rows`, and `sampled` so the cap is
auditable. Set `sample_n=None` to force full-data learning (can OOM past
~1M rows depending on width).

`random_state` (default 42) seeds the tree search, the sampler, and the
holdout split together — same seed, same cuts.

## Diagnostics reference

With `return_diagnostics=True` the call returns `(DuckJanitor, diag)` where
`diag` maps each requested column to (illustrative values):

```text
{
    "age": {
        "cuts": [27.5, 32.5, 36.5, 47.5, 67.5],  # learned split points
        "method": "optbinning",                   # engine actually used
        "y_kind": "binary",
        "solver": "cp",                           # optbinning only
        "learned_on": "train n=4800",             # or "full n=..." / "sample n=..."
        "fit_rows": 6000, "sampled": False, "total_rows": 6000,
        "scores_holdout": {"auc": ..., ...},      # only when holdout is set
        "holdout_frac": 0.2, "holdout_n": 1200, "train_n": 4800,
    },
    "segment": {"cuts": None, "method": "skipped", ...},  # non-numeric input
}
```

## Best practices

1. **Start with `method="auto"` + `holdout=0.2` + `return_diagnostics=True`.**
   The diagnostics tell you which engine won, how many bins resulted, and
   whether the bins carry signal out of sample. Promote to
   `method="optbinning"` once binary targets are validated.
2. **Read the holdout scores, not the bin count.** More bins ≠ better bins.
   A 6-bin split with test AUC 0.52 is noise; a 3-bin split with AUC 0.68 is
   signal. Drop binned features whose holdout scores hover near chance
   (AUC≈0.5, R²≤0, F1-macro near the majority-class baseline).
3. **Constrain OptBinning for production.** Defaults (`max_n_bins=6`,
   `min_prebin_size=0.05`, `monotonic_trend="auto"`) suit exploration. For
   scorecards, tighten via `optbinning_kwargs`, e.g.
   `{"max_n_bins": 4, "min_bin_size": 0.05, "monotonic_trend": "ascending"}`
   for a monotone risk ordering.
4. **Keep `stratify=True`** (default) for imbalanced discrete targets so
   rare classes survive the holdout split.
5. **Mind the `task=` override.** Use it when labels are ambiguous (e.g.
   60 distinct integer codes that are classes, not measurements — auto
   already handles this up to 100, but explicit beats implicit).
6. **Watch `sampled` on big tables.** If diagnostics show sampling kicked in,
   re-run with a larger `sample_n` (or `None` if memory allows) to confirm
   cuts are stable before shipping them.
7. **Treat `Missing` as a real bin.** NULLs get their own bin rather than
   being dropped — check its event rate; missingness is often predictive.
8. **Pin `random_state`** for reproducible cuts across runs; vary it once to
   check cut stability (cuts that jump around under re-seed are fragile).
9. **One target at a time.** Cuts are target-specific — bins learned against
   `bad_flag` do not transfer to `risk_grade`. Re-run per target.
10. **Chain last, diagnose first.** `return_diagnostics=True` breaks chaining
    (returns a tuple). Explore with diagnostics, then drop the flag for the
    production chain.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cuts: []`, single `Single` bin | No supervised signal (or OB conservative) | Check holdout scores; try `method="tree"`; loosen `optbinning_kwargs` (`min_prebin_size`, `max_n_bins`) |
| `ValueError: task=... conflicts with y` | Forced task ≠ detected kind | Use `task="auto"` or the matching task |
| `ValueError: target column ... already exists` | Output name collision | Change `suffix` or drop the stale column |
| `cv reduced 3->2 ...` warning | Tiny minority class | Expected on small data; add rows or accept fewer folds |
| `OptBinning ... solver ... retrying` warning | Missing/failing solver backend | `pip install ortools`; diagnostics `solver` shows what ran |
| `MethodFallbackWarning: DTD failed ... sklearn tree` | `feature_engine` edge case (e.g. NaNs survived) | Usually harmless — sklearn fallback produces the cuts |
| `y has only 1 distinct value` | Constant target | Nothing to bin on; check the target column |
| Scores all `nan` | Too few test rows or single-class holdout | More data, smaller `holdout`, or `stratify=True` |

## See also

- API entry: [`optimize_bins_to_target`](api/functions.md#optimize_bins_to_target)
- Unsupervised alternative: [`bin_numeric`](api/functions.md#bin_numeric)
  (quantile / uniform / explicit edges, no target needed)
- Runnable demo: `examples/optimize_bins_example.py`
