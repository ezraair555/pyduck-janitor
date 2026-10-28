"""
Supervised binning — learn cut points from y.

Backends (two packages):
  • OptBinning (optbinning)  — binary y only; WoE/IV-optimal, monotonic when available
  • DecisionTreeDiscretiser (feature_engine) — binary + multiclass + continuous
       (regression=True path)

OptBinning is the default because it optimizes IV under business constraints
(min_bin_size, monotonic_trend) and ships WoE/event-rate diagnostics; the
tree path is the fallback when y is not binary or when iv is unavailable.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*HIGHS.*", category=UserWarning)

from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd

from .cleaning_ops import _ensure_columns_exist, _quote_id, _register_relation, _sql_literal


def _detect_y_kind(y: pd.Series, task: str = "auto", n_sample: int | None = None) -> str:
    """Return 'binary' | 'multiclass' | 'continuous'.

    NaNs are ignored for kind detection (a binary y with missing values
    is still binary).  When ``task`` is not "auto" it wins:
    "binary" | "multiclass" -> discrete, "regression" -> continuous.

    Auto rules (in order):
    * 2 distinct non-null values -> binary (any dtype).
    * <=1 distinct -> error (no signal to bin on).
    * object / string / bool / categorical dtype -> multiclass (never
      continuous — text labels are classes, not measurements).
    * float dtype with >50 distinct -> continuous.
    * int dtype with >100 distinct -> continuous (allows up to 100-class
      integer-labelled multiclass); int with <=100 distinct -> multiclass.
    * anything else discrete with >2 values -> multiclass.
    """
    if task != "auto":
        t = task.lower()
        if t in ("binary", "multiclass"):
            return t
        if t in ("regression", "continuous", "cont"):
            return "continuous"
        raise ValueError("task must be 'auto', 'binary', 'multiclass' or 'regression'")
    y_nn = y.dropna()
    nunique = y_nn.nunique()
    if nunique <= 1:
        raise ValueError(f"y has only {nunique} distinct non-null value(s); supervised binning needs at least 2")
    if nunique == 2:
        return "binary"
    # Non-numeric labels are classes, never measurements.
    if y.dtype == object or str(y.dtype) in ("string", "str") or str(y.dtype).startswith("categor") or str(y.dtype) == "bool":
        return "multiclass"
    if pd.api.types.is_float_dtype(y.dtype):
        if nunique > 50:
            return "continuous"
        return "multiclass"
    if pd.api.types.is_integer_dtype(y.dtype):
        if nunique > 100:
            return "continuous"
        return "multiclass"
    return "multiclass"


def _tree_splits(
    x: pd.Series,
    y: pd.Series,
    *,
    method: str,
    cv: int = 3,
    task: str = "auto",
    random_state: Optional[int] = 42,
    suppress_nan_warnings: bool = False,
) -> List[float]:
    """Learn splits with a 1-d decision tree.

    For binary/multiclass y we use feature_engine's DecisionTreeDiscretiser
    (GridSearchCV over max_depth/min_samples_leaf) — one package.  For
    continuous y the ``regression=True`` path optimizes MSE (same package,
    different mode).  If feature_engine is unavailable we fall back to a
    direct sklearn tree so the feature still works.
    """
    # Validate an explicit task against the data before any GridSearchCV
    # so a mismatch fails fast with one clear error instead of pages of
    # nan-score CV warnings.
    actual_kind = _detect_y_kind(y, "auto")
    requested = task.lower() if isinstance(task, str) else "auto"
    if requested != "auto":
        exp = {"binary": "binary", "multiclass": "multiclass",
               "regression": "continuous", "continuous": "continuous",
               "cont": "continuous"}.get(requested, requested)
        if exp != actual_kind:
            # binary is a special case of 2-class multiclass scoring, but
            # roc_auc on 3+ classes always fails — refuse rather than
            # drown the user in nan-score warnings.
            raise ValueError(
                f"task={task!r} conflicts with y: detected {actual_kind} "
                f"({y.dropna().nunique()} distinct non-null values). "
                f"Use task='auto' or the matching task."
            )
    # ---- try feature_engine path first ----
    try:
        from feature_engine.discretisation import DecisionTreeDiscretiser as DTD

        y_kind = _detect_y_kind(y, task)
        # DTD needs a DataFrame; it also wants a target Series
        regression = y_kind == "continuous"
        scoring = "neg_mean_squared_error" if regression else ("roc_auc_ovr" if y_kind == "multiclass" and len(np.unique(pd.Series(y).dropna())) > 2 else ("roc_auc" if y_kind == "binary" else "f1_macro"))
        # Cap CV folds by the smallest class count so GridSearchCV never
        # runs folds with a single class (which yields nan + warnings).
        _cv = cv
        if not regression:
            try:
                _counts = pd.Series(y).dropna().value_counts()
                if len(_counts) > 1:
                    _cv = max(2, min(cv, int(_counts.min())))
                    if _cv < cv:
                        warnings.warn(f"cv reduced {cv}->{_cv} for {x.name!r}: smallest class has {_counts.min()} rows", stacklevel=3)
            except Exception:
                pass
        # Small-n grids must allow small leaves or no split is possible.
        _leaf_grid = [2, 5, 10] if len(x) < 200 else [10, 20]
        with warnings.catch_warnings():
            # GridSearchCV intentionally emits nan-score warnings when a
            # fold lacks a class; we already capped cv above, so silence
            # the residue — a genuine failure still raises below.
            warnings.filterwarnings("ignore", message=".*score.*nan.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*least populated class.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*ROC AUC.*not defined.*", category=UserWarning)
            d = DTD(
                variables=[x.name],
                regression=regression,
                scoring=scoring,
                cv=_cv,
                bin_output="prediction",  # we only need binner_dict_; transform returns predictions but we extract thresholds
                param_grid={"max_depth": [2, 3, 4], "min_samples_leaf": _leaf_grid},
                random_state=random_state,
            )
            df = pd.DataFrame({x.name: x.astype(float)})
            d.fit(df, y)
        gs = d.binner_dict_.get(x.name)
        if gs is None:
            raise RuntimeError("DTD binner not found")
        best = gs.best_estimator_
        cuts = sorted(float(t) for t in best.tree_.threshold if t != -2)
        return cuts
    except ModuleNotFoundError:
        pass
    except Exception as e:
        warnings.warn(f"DTD failed for {x.name!r}: {e} — falling back to sklearn tree", stacklevel=3)

    # ---- sklearn fallback (same behaviour, no extra deps beyond sklearn) ----
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    # drop NA rows for fitting; splits should not include the NA sentinel
    mask = x.notna() & y.notna()
    xx = x[mask].astype(float).to_numpy()
    yy = y[mask].to_numpy()
    if len(xx) < 10:
        return []
    y_kind = _detect_y_kind(pd.Series(yy), task)
    if y_kind == "continuous":
        clf = DecisionTreeRegressor(max_depth=4, min_samples_leaf=max(2, len(xx)//20), random_state=random_state or 42)
    else:
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(2, len(xx)//20), random_state=random_state or 42)
    # Optionally suppress nan-score warnings from sklearn internals.
    # Scope the filter to this call: a bare warnings.filterwarnings() call
    # mutates the global filter list and leaks into the rest of the
    # process, silencing unrelated nan warnings program-wide.
    with warnings.catch_warnings():
        if suppress_nan_warnings:
            warnings.filterwarnings("ignore", message=".*nan.*", category=UserWarning)
        clf.fit(xx.reshape(-1, 1), yy)
    return sorted(float(t) for t in clf.tree_.threshold if t != -2)


def _optbinning_splits(x: pd.Series, y: pd.Series, **ob_kwargs) -> Optional[List[float]]:
    """Try OptBinning. Returns None if unavailable or unsuited (e.g. multiclass/continuous y)."""
    try:
        from optbinning import OptimalBinning
    except ModuleNotFoundError:
        return None
    y_kind = _detect_y_kind(y)
    if y_kind != "binary":
        return None  # OptBinning's IV/WoE formulation assumes 0/1 target
    # Sensible defaults; user can override via ob_kwargs.
    # solver='cp' needs ortools; if it is missing, prefer 'mip' up front
    # so the failure mode is one clear warning, not an obscure fit error.
    try:
        import ortools  # noqa: F401
        _default_solver = "cp"
    except ModuleNotFoundError:
        _default_solver = "mip"
        warnings.warn("ortools not installed — OptBinning solver defaulted to 'mip'; pip install ortools for 'cp'", stacklevel=3)
    kwargs = dict(
        dtype="numerical",
        max_n_prebins=20,
        min_prebin_size=0.05,
        min_n_bins=2,
        max_n_bins=6,
        monotonic_trend="auto",
        divergence="iv",
        solver=_default_solver,
        time_limit=30,
    )
    kwargs.update(ob_kwargs)
    try:
        ob = OptimalBinning(**kwargs)
    except Exception as e:
        # Bad solver name etc. — surface clearly, let auto fall back.
        warnings.warn(f"OptBinning init failed ({e}) — falling back", stacklevel=3)
        return None
    mask = x.notna() & y.notna()
    # OptBinning needs numeric 0/1; factorize string binary y (sorted classes -> 0/1)
    y_fit = y[mask]
    if not pd.api.types.is_numeric_dtype(y_fit.dtype):
        # map sorted unique -> 0..n-1; binary only here so 0/1
        mapping = {u: i for i, u in enumerate(sorted(y_fit.dropna().unique()))}
        y_fit = y_fit.map(mapping)
    try:
        ob.fit(x[mask].to_numpy(dtype=float), y_fit.to_numpy(dtype=float))
    except Exception as e:
        # Solver backend (cp/mip) is the usual culprit — retry once with
        # the other solver before giving up so one missing backend
        # doesn't force a silent tree fallback.
        _msg = str(e).lower()
        _solver_used = str(kwargs.get("solver", ""))
        if "solver" in _msg or "ortools" in _msg or "highs" in _msg or "mip" in _msg:
            _alt = "mip" if _solver_used == "cp" else "cp"
            try:
                warnings.warn(f"OptBinning solver {_solver_used!r} failed for {x.name!r} ({e}) — retrying solver={_alt!r}", stacklevel=3)
                ob_retry = OptimalBinning(**{**kwargs, "solver": _alt})
                ob_retry.fit(x[mask].to_numpy(dtype=float), y_fit.to_numpy(dtype=float))
                ob = ob_retry
            except Exception as e2:
                warnings.warn(f"OptBinning fit failed for {x.name!r}: {e2} — falling back", stacklevel=3)
                return None
        else:
            warnings.warn(f"OptBinning fit failed for {x.name!r}: {e} — falling back", stacklevel=3)
            return None
    status = getattr(ob, "status", "")
    if ob.splits is None or len(ob.splits) == 0:
        return None if status != "OPTIMAL" else []
    return [float(s) for s in ob.splits]


def _case_sql(col_id: str, cuts: List[float], numeric_labels: bool = False) -> str:
    """Build CASE WHEN ... END that maps bins to text labels (or 1..k integers if numeric_labels)."""
    # Labels: (-inf, c1] (c1, c2] ... (ck, inf) + None -> "Missing" / NULL
    # When numeric_labels=True we return small integers (DuckDB cast-friendly)
    parts = []
    for i, c in enumerate(cuts):
        if i == 0:
            # (-inf, c1]
            label = 1 if numeric_labels else f"(-inf, {c}]"
            parts.append(f"WHEN {col_id} <= {_sql_literal(c)} THEN {_sql_literal(label)}")
        else:
            label = i + 1 if numeric_labels else f"({cuts[i-1]}, {c}]"
            parts.append(f"WHEN {col_id} <= {_sql_literal(c)} THEN {_sql_literal(label)}")
    # tail
    if cuts:
        last = f"({cuts[-1]}, inf)"
        label = len(cuts) + 1 if numeric_labels else last
        parts.append(f"WHEN {col_id} > {_sql_literal(cuts[-1])} THEN {_sql_literal(label)}")
    # NULL -> Missing (preserves information)
    other_label = "Other" if cuts else "Single"
    return "CASE " + " ".join(parts) + f" WHEN {col_id} IS NULL THEN {_sql_literal('Missing')} ELSE {_sql_literal(other_label)} END"


def _choose_splits(
    x: pd.Series,
    y: pd.Series,
    *,
    method: str,
    cv: int,
    task: str,
    random_state: Optional[int],
    optbinning_kwargs: Optional[Dict[str, Any]],
) -> Tuple[List[float], str]:
    """Pick a method and return (cuts, used_method)."""
    y_kind = _detect_y_kind(y, task)
    if method not in ("optbinning", "tree", "auto"):
        raise ValueError("method must be 'optbinning', 'tree', or 'auto'")
    if method in ("optbinning", "auto"):
        ob_kwargs = dict(optbinning_kwargs or {})
        # continuous y never goes through OptBinning (tree regression instead)
        if y_kind != "continuous":
            cuts = _optbinning_splits(x, y, **(ob_kwargs or {}))
            if cuts is not None:
                if len(cuts) > 0 or method == "optbinning":
                    return cuts, "optbinning"
                # auto + OB optimal-but-empty: give the tree one chance
                # before settling on a single bin. OB is conservative on
                # perfect separation / weak-IV data where a tree still
                # finds usable signal.
                tree_cuts = _tree_splits(x, y, method=method, cv=cv, task=task, random_state=random_state)
                if tree_cuts:
                    warnings.warn(
                        f"OptBinning found no splits for {x.name!r}; auto fell back to tree ({len(tree_cuts)} cuts)",
                        stacklevel=3,
                    )
                    return tree_cuts, "tree"
                return cuts, "optbinning"  # both engines agree: single bin
        if method == "optbinning" and y_kind != "binary":
            # explicit optbinning on non-binary -> be loud
            warnings.warn(f"optbinning requested but y={y_kind!r} is not binary — falling back to tree", stacklevel=3)
            # fall through to tree path below
        elif method == "optbinning" and cuts is None:
            raise RuntimeError("OptBinning is unavailable or returned no splits; cannot satisfy method='optbinning'")
    # tree path (also the auto fallback for multiclass/continuous or when OB unavailable)
    cuts = _tree_splits(x, y, method=method, cv=cv, task=task, random_state=random_state)
    return cuts, "tree"



def _assign_bin(value: float | None, cuts: list[float]) -> str:
    """Map a scalar to its bin label consistent with _case_sql interval labels."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Missing"
    for i, c in enumerate(cuts):
        if value <= c:
            return f"(-inf, {c}]" if i == 0 else f"({cuts[i-1]}, {c}]"
    return f"({cuts[-1]}, inf)" if cuts else "Single"


def _score_binned_feature(
    x: pd.Series,
    y: pd.Series,
    cuts: list[float],
    y_kind: str,
    _train_map: dict | None = None,
) -> dict[str, float | int]:
    """Score how well the learned bins predict y (cheap — no model, just per-bin stats).

    When ``_train_map`` is given (bin -> stat learned on TRAIN), it is used
    to predict TEST rows (honest holdout).  Otherwise the mapping is learned
    on the passed ``x``/``y`` itself (in-sample / resubstitution score).
    """
    import pandas as _pd
    import numpy as _np
    bins = _pd.Series([_assign_bin(v, cuts) for v in x], index=x.index)
    mask = y.notna() & bins.notna()
    yv, bv = y[mask], bins[mask]
    if len(yv) < 10:
        return {"n": int(len(yv)), "note": "too few non-null rows for scoring"}
    if y_kind == "binary":
        # per-bin p(event) -> use as "proba" proxy to compute AUC/f1
        # Factorize non-numeric y to 0/1 (sorted classes) so mean/AUC work.
        if not _pd.api.types.is_numeric_dtype(yv):
            _classes = sorted(yv.dropna().unique())
            _ymap = {c: i for i, c in enumerate(_classes)}
            yv_num = yv.map(_ymap).astype(int)
        else:
            yv_num = yv.astype(int)
        if _train_map is not None:
            p_map = _train_map
        else:
            grp = _pd.DataFrame({"y": yv_num, "bin": bv}).groupby("bin", observed=True)["y"]
            p_map = grp.mean()  # p(event) per bin
        # map each row to its bin's p(event); that's the "prediction"
        # unseen bins (test-only) fall back to global mean
        _fallback = float(yv_num.mean())
        proba = bv.map(p_map).fillna(_fallback).astype(float)
        try:
            from sklearn.metrics import roc_auc_score, f1_score
            auc = float(roc_auc_score(yv_num, proba)) if yv_num.nunique() == 2 else float("nan")
        except Exception:
            auc = float("nan")
        try:
            pred = (proba >= 0.5).astype(int)
            f1 = float(f1_score(yv_num, pred, zero_division=0))
        except Exception:
            f1 = float("nan")
        # Cramer's V (chi2 on bins x y table)
        try:
            from scipy.stats import chi2_contingency
            tbl = _pd.crosstab(bv, yv)
            chi2 = chi2_contingency(tbl)[0]
            n = len(yv)
            r, k = tbl.shape
            cramer = float((chi2 / (n * min(r - 1, k - 1))) ** 0.5) if min(r - 1, k - 1) > 0 else float("nan")
        except Exception:
            cramer = float("nan")
        return {"auc": auc, "f1": f1, "gini": (2*auc-1) if _np.isfinite(auc) else float("nan"), "cramers_v": cramer, "n_bins": int(bv.nunique()), "n": int(len(yv))}
    if y_kind == "multiclass":
        try:
            from sklearn.metrics import f1_score as _f1m
            # per-bin majority class as prediction (train map when provided)
            if _train_map is not None:
                maj = _train_map
            else:
                maj = _pd.DataFrame({"y": yv, "bin": bv}).groupby("bin", observed=True)["y"].agg(lambda s: s.mode().iloc[0])
            _fb = _pd.Series(yv).mode().iloc[0]
            pred = bv.map(maj).fillna(_fb)
            f1m = float(_f1m(yv, pred, average="macro", zero_division=0))
        except Exception:
            f1m = float("nan")
        try:
            from sklearn.metrics import balanced_accuracy_score as _bacc
            _maj2 = _train_map if _train_map is not None else _pd.DataFrame({"y": yv, "bin": bv}).groupby("bin", observed=True)["y"].agg(lambda s: s.mode().iloc[0])
            _fb2 = _pd.Series(yv).mode().iloc[0]
            pred2 = bv.map(_maj2).fillna(_fb2)
            bacc = float(_bacc(yv, pred2))
        except Exception:
            bacc = float("nan")
        return {"f1_macro": f1m, "balanced_accuracy": bacc, "n_bins": int(bv.nunique()), "n": int(len(yv))}
    # continuous: per-bin mean as predictor -> r2/rmse (train map when provided)
    try:
        from sklearn.metrics import r2_score as _r2, mean_squared_error as _mse
        if _train_map is not None:
            bin_mean = _train_map
        else:
            bin_mean = _pd.DataFrame({"y": yv.astype(float), "bin": bv}).groupby("bin", observed=True)["y"].mean()
        _fbm = float(_pd.Series(yv.astype(float)).mean())
        pred = bv.map(bin_mean).fillna(_fbm).astype(float)
        r2 = float(_r2(yv.astype(float), pred))
        rmse = float(_mse(yv.astype(float), pred) ** 0.5)
    except Exception:
        r2, rmse = float("nan"), float("nan")
    return {"r2": r2, "rmse": rmse, "n_bins": int(bv.nunique()), "n": int(len(yv))}


# ── public API ─────────────────────────────────────────────────────────────
def optimize_bins_to_target(
    relation: duckdb.DuckDBPyRelation,
    y: str,
    *,
    columns: Optional[Union[str, List[str]]] = None,
    method: str = "auto",
    cv: int = 3,
    task: str = "auto",
    cast_to_text: bool = True,
    suffix: str = "_binned",
    random_state: Optional[int] = 42,
    return_diagnostics: bool = False,
    optbinning_kwargs: Optional[Dict[str, Any]] = None,
    holdout: Union[bool, float] = False,
    stratify: bool = True,
    sample_n: Optional[int] = 100_000,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> Union[duckdb.DuckDBPyRelation, Tuple[duckdb.DuckDBPyRelation, Dict[str, Any]]]:
    """Supervised binning: learn optimal cut points from y and apply them.

    Learns one set of cut points per ``columns`` entry using the
    relationship with ``y``.  Two engines from two packages:

    * **optbinning** (``method='optbinning'``) — IV/WoE optimal, with business
      constraints (``optbinning_kwargs``).  Only for binary y.
    * **tree / DTD** (``method='tree'``) — ``DecisionTreeDiscretiser``/
      sklearn tree; works for binary, multiclass, and continuous y.
    * ``method='auto'`` (default) prefers OptBinning when y is binary and
      the package is installed, otherwise the tree.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        Input relation.
    y : str
        Target column.  Binary / multiclass labels and continuous values
        are all accepted; ``method='auto'`` adapts automatically.
    columns : str or list of str, optional
        Feature column(s) to bin.  When omitted, all numeric columns in the
        relation except ``y`` are considered.
    method : {'auto','optbinning','tree'}, default 'auto'
        Which discretiser to try first.
    cv : int, default 3
        Cross-validation folds for the tree path.
    task : {'auto','binary','multiclass','regression'}, default 'auto'
        Passed through to the scoring selection.  ``'auto'`` is
        inferred from ``y``.
    cast_to_text : bool, default True
        When True and the source column is numeric (``INTEGER``/``DOUBLE``/
        ``DECIMAL`` etc.), the produced bin column is ``VARCHAR`` with
        interval labels ``(-inf, 10]``, ``(10, 25]`` …  When False the
        bin is left numeric (1, 2, …).
    suffix : str, default '_binned'
        Appended to the original column name for the new column.  Use
        ``''`` to replace in place (original column name reused).
    random_state : int or None
        Seeded into the tree path for reproducibility.
    return_diagnostics : bool
        Also return per-column diagnostics (cuts, used method, y-kind).
    optbinning_kwargs : dict, optional
        Extra ``OptimalBinning(...)`` kwargs forwarded to the optbinning
        constructor (e.g. ``{"monotonic_trend":"peak", "max_n_bins":8}``).
    holdout : bool or float, default False
        Honest train/test evaluation. ``False`` (default) learns cuts on
        the full data. ``True`` uses 0.2, a float in (0, 0.5) is the test
        fraction.  When set, cuts are learned on the TRAIN split only,
        bin statistics come from TRAIN, scores are computed on TEST, and
        the returned relation carries TRAIN cuts applied to all rows.
        Diagnostics gain ``scores_holdout`` / ``holdout_n`` / ``train_n``.
    stratify : bool, default True
        Stratify the hold-out split by y when y is discrete.
    sample_n : int or None, default 100_000
        Cap on rows materialised for cut learning.  When the input has
        more than ``sample_n`` rows, a stratified random sample of that
        size is used to learn cuts (and for the holdout split); the
        learned CASE is still applied to the full relation, so no rows
        are lost.  ``None`` disables sampling (full materialisation —
        can OOM past ~1M rows).  Diagnostics record ``fit_rows`` and
        ``sampled`` so the cap is auditable.
    conn : duckdb.DuckDBPyConnection, optional
        Explicit DuckDB connection.  When omitted it is derived from
        ``relation``.

    Returns
    -------
    DuckDBPyRelation
        Relation with one new ``<column><suffix>`` column per requested
        feature.  If ``return_diagnostics`` is True, the second value is
        ``{column: {"cuts": [...], "method": "...", "y_kind": "..."}}``.

    Examples
    --------
    >>> dj2 = optimize_bins_to_target(dj, y="bad_flag", columns=["age", "income"])
    >>> dj2 = optimize_bins_to_target(
    ...     dj, y="risk_grade", columns="income", method="tree", cv=5
    ... )
    >>> result, diag = optimize_bins_to_target(
    ...     dj, y="bad_flag", columns=["age"], return_diagnostics=True
    ... )
    """
    if conn is None:
        # Standalone functional call: copy into a fresh connection (eager but
        # robust — DuckDB relations are connection-bound and cannot be
        # re-registered across connections for replacement scans).
        conn = duckdb.connect()
        relation = conn.from_df(relation.df())
    cols_available = list(relation.columns)
    _ensure_columns_exist(cols_available, [y])

    if columns is None:
        # infer numeric non-target columns
        inferred: List[str] = []
        for col in cols_available:
            if col == y:
                continue
            try:
                # probe dtype
                from duckdb import SQLType  # noqa: F401
            except Exception:
                pass
            # heuristic: materialise 1 row; pandas dtypes -> numeric?
            inferred.append(col)
        columns = inferred
    elif isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

    if not columns:
        raise ValueError("columns produced no work: pass at least one column name, or a relation with non-target columns")
    _ensure_columns_exist(cols_available, columns)

    # Single materialisation for all columns.  Cap rows for learning so a
    # 10M-row Parquet doesn't OOM the fitter — the CASE is still applied
    # to the full relation afterwards.
    want = list(dict.fromkeys(columns + [y]))  # de-duplicate, keep order
    tbl_all = _register_relation(conn, relation)
    sel = ", ".join(_quote_id(c) for c in want)
    try:
        _total_n = conn.execute(f"SELECT COUNT(*) FROM {tbl_all}").fetchone()[0]
    except Exception:
        _total_n = None
    df_all = conn.execute(f"SELECT {sel} FROM {tbl_all}").df()
    _sampled = False
    if sample_n is not None and len(df_all) > sample_n:
        from sklearn.model_selection import train_test_split as _samp_split
        _strat = df_all[y] if _detect_y_kind(df_all[y], "auto") in ("binary", "multiclass") else None
        try:
            _idx, _ = _samp_split(range(len(df_all)), train_size=sample_n, random_state=random_state, stratify=_strat)
            df_all = df_all.iloc[sorted(_idx)].reset_index(drop=True)
            _sampled = True
            warnings.warn(f"optimize_bins_to_target: sampled {sample_n:,} of {_total_n:,} rows for cut learning; CASE applied to all rows", stacklevel=2)
        except Exception as e:
            warnings.warn(f"sampling failed ({e}) — learning on full {len(df_all):,} rows", stacklevel=2)
    y_series = df_all[y]
    y_kind = _detect_y_kind(y_series, task)

    # Learn cuts per feature
    diag: Dict[str, Dict[str, Any]] = {}
    cuts_per_col: Dict[str, List[float]] = {}
    for col in columns:
        x = df_all[col]
        # Skip non-numeric features: binning is defined for ordered values;
        # categorical features already are discrete.
        if not pd.api.types.is_numeric_dtype(x):
            warnings.warn(f"optimize_bins_to_target: {col!r} is non-numeric ({x.dtype}) — skipped", stacklevel=2)
            diag[col] = {"cuts": None, "method": "skipped", "y_kind": y_kind, "reason": f"non-numeric {x.dtype}"}
            continue
        cuts, used = _choose_splits(x, y_series, method=method, cv=cv, task=task, random_state=random_state, optbinning_kwargs=optbinning_kwargs)
        cuts_per_col[col] = cuts
        _entry: Dict[str, Any] = {"cuts": cuts, "method": used, "y_kind": y_kind, "learned_on": f"{'sample' if _sampled else 'full'} n={len(df_all)}", "fit_rows": len(df_all), "sampled": _sampled, "total_rows": _total_n}
        if used == "optbinning":
            if optbinning_kwargs and "solver" in optbinning_kwargs:
                _entry["solver"] = optbinning_kwargs["solver"]
            else:
                try:
                    import ortools  # noqa: F401
                    _entry["solver"] = "cp"
                except ModuleNotFoundError:
                    _entry["solver"] = "mip"
        diag[col] = _entry

    if not cuts_per_col:
        # nothing to do (e.g. all non-numeric) — return unchanged
        return (relation, diag) if return_diagnostics else relation

    # ── optional honest hold-out: learn cuts on TRAIN, score on TEST ──
    # When holdout is set, cuts are learned on the train split only and the
    # reported scores use train-derived bin statistics applied to test rows.
    # The returned relation carries the TRAIN cuts applied to all rows.
    if holdout is not False:
        from sklearn.model_selection import train_test_split as _tss
        frac = 0.2 if holdout is True else float(holdout)
        if not (0 < frac < 0.5):
            raise ValueError("holdout fraction must be in (0, 0.5)")
        strat = y_series if (stratify and y_kind in ("binary", "multiclass")) else None
        train_idx, test_idx = _tss(range(len(df_all)), test_size=frac, random_state=random_state, stratify=strat)
        df_train = df_all.iloc[train_idx]
        y_train = y_series.iloc[train_idx]
        # Re-learn cuts on train only (honest); overwrite train cuts
        for col in list(cuts_per_col.keys()):
            x_tr = df_train[col]
            try:
                cuts_tr, used_tr = _choose_splits(x_tr, y_train, method=method, cv=cv, task=task, random_state=random_state, optbinning_kwargs=optbinning_kwargs)
            except Exception as e:
                diag[col]["holdout_error"] = str(e)
                continue
            cuts_per_col[col] = cuts_tr
            diag[col]["cuts"] = cuts_tr
            diag[col]["method"] = used_tr
            diag[col]["learned_on"] = f"train n={len(df_train)}"
            # Build train mapping bin->stat, then score test with it
            import pandas as _pdh
            bins_tr = _pdh.Series([_assign_bin(v, cuts_tr) for v in x_tr], index=df_train.index)
            ytr = y_train
            if y_kind == "binary":
                if not _pdh.api.types.is_numeric_dtype(ytr):
                    _cl = sorted(ytr.dropna().unique())
                    _ym = {c: i for i, c in enumerate(_cl)}
                    ytr_num = ytr.map(_ym).astype(int)
                else:
                    ytr_num = ytr.astype(int)
                train_map = _pdh.DataFrame({"y": ytr_num, "bin": bins_tr}).groupby("bin", observed=True)["y"].mean().to_dict()
            elif y_kind == "multiclass":
                train_map = _pdh.DataFrame({"y": ytr, "bin": bins_tr}).groupby("bin", observed=True)["y"].agg(lambda ss: ss.mode().iloc[0]).to_dict()
            else:
                train_map = _pdh.DataFrame({"y": ytr.astype(float), "bin": bins_tr}).groupby("bin", observed=True)["y"].mean().to_dict()
            x_te = df_all.iloc[test_idx][col]
            y_te = y_series.iloc[test_idx]
            try:
                score = _score_binned_feature(x_te, y_te, cuts_tr, y_kind, _train_map=train_map)
            except Exception as e:
                score = {"error": str(e)}
            diag[col]["scores_holdout"] = score
            diag[col]["holdout_frac"] = frac
            diag[col]["holdout_n"] = int(len(test_idx))
            diag[col]["train_n"] = int(len(df_train))

    # Apply via pure DuckDB CASE — no second materialisation per column
    rel = relation
    tbl = _register_relation(conn, rel)
    for col, cuts in cuts_per_col.items():
        tgt = col if not suffix else f"{col}{suffix}"
        if tgt in list(rel.columns) and tgt != col:
            raise ValueError(f"target column {tgt!r} already exists; choose a different suffix")
        col_q = _quote_id(col)
        tgt_q = _quote_id(tgt)
        # cast_to_text only matters when source column is numeric (which it is, per above)
        case_expr = _case_sql(col_q, cuts, numeric_labels=not cast_to_text)
        if tgt == col:
            rel = conn.sql(f"SELECT * EXCLUDE ({col_q}), ({case_expr}) AS {tgt_q} FROM {tbl}")
        else:
            rel = conn.sql(f"SELECT *, ({case_expr}) AS {tgt_q} FROM {tbl}")
        # re-register for next iteration so bins accumulate
        tbl = _register_relation(conn, rel)

    if return_diagnostics:
        return rel, diag
    return rel
