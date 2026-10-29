"""Tests for :mod:`pyduck_janitor.bins_opt` — supervised binning.

Covers the public ``optimize_bins_to_target`` entry point: the happy path
(numeric feature + binary target), the categorical-feature skip path, the
diagnostics contract, the holdout scoring metrics, ``sample_n`` capping and
the error paths.

The heavy optional backends (optbinning / feature_engine / scikit-learn /
scipy) are imported lazily by the implementation, so this module skips
cleanly when they are absent — CI stays green without the ``binning`` extra.
"""

from __future__ import annotations

import math
import warnings

import duckdb
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn is required for supervised binning")

from pyduck_janitor.bins_opt import (  # noqa: E402
    _assign_bin,
    _case_sql,
    _detect_y_kind,
    _optbinning_splits,
    _score_binned_feature,
    _tree_splits,
    optimize_bins_to_target,
)

SEED = 42


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    connection = duckdb.connect()
    yield connection
    connection.close()


def _binary_frame(n: int = 2000, seed: int = SEED):
    """Feature ~ N(0,1); target = 1 when (feature + noise) > 0.

    The signal is real, so a good binning must show an increasing event
    rate from the lowest bin to the highest.
    """
    rs = np.random.RandomState(seed)
    x = rs.normal(size=n)
    y = (x + rs.normal(0.0, 0.4, size=n) > 0).astype(int)
    return pd.DataFrame({"feat": x, "y": y})


def _bin_event_rates(frame: pd.DataFrame, feature: str, cuts, target: str = "y"):
    """Assign rows to the learned bins and return the per-bin event rate."""
    edges = [-np.inf] + list(cuts) + [np.inf]
    binned = pd.cut(frame[feature], bins=edges, labels=False, include_lowest=True)
    rates = frame.assign(_bin=binned).groupby("_bin")[target].mean().sort_index()
    return rates


def _rel(conn, df):
    return conn.from_df(df)


# ---------------------------------------------------------------------------
# Happy path — numeric feature, binary target
# ---------------------------------------------------------------------------


class TestNumericBinaryHappyPath:
    def test_returns_new_binned_column(self, conn):
        df = _binary_frame()
        out = optimize_bins_to_target(_rel(conn, df), "y", columns=["feat"], conn=conn)
        cols = list(out.columns)
        assert "feat" in cols
        assert "feat_binned" in cols
        assert len(out.df()) == len(df)

    def test_cuts_are_sorted_and_unique(self, conn):
        df = _binary_frame()
        _out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], return_diagnostics=True, conn=conn
        )
        cuts = diag["feat"]["cuts"]
        assert isinstance(cuts, list)
        assert all(isinstance(c, float) for c in cuts)
        assert cuts == sorted(cuts), "cuts must be strictly ascending"
        assert len(set(cuts)) == len(cuts)

    def test_bin_count_matches_cuts(self, conn):
        df = _binary_frame()
        out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], return_diagnostics=True, conn=conn
        )
        cuts = diag["feat"]["cuts"]
        n_labels = out.df()["feat_binned"].nunique()
        # one label per cut plus the open-ended tail bin
        assert n_labels <= len(cuts) + 1

    def test_event_rate_is_monotonic_across_bins(self, conn):
        df = _binary_frame()
        _out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], return_diagnostics=True, conn=conn
        )
        cuts = diag["feat"]["cuts"]
        assert len(cuts) >= 1, "expected the learner to find at least one split"
        rates = _bin_event_rates(df, "feat", cuts)
        assert len(rates) >= 2
        values = rates.to_numpy()
        # Monotonic non-decreasing event rate: every bin is at least as
        # risky as the one below it (the signal is monotone by construction).
        assert np.all(np.diff(values) >= -1e-9), f"event rate not monotonic: {values}"

    def test_event_rate_spans_the_range(self, conn):
        df = _binary_frame()
        _out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], return_diagnostics=True, conn=conn
        )
        rates = _bin_event_rates(df, "feat", diag["feat"]["cuts"])
        assert rates.iloc[0] < 0.2
        assert rates.iloc[-1] > 0.8

    def test_binned_labels_are_interval_strings(self, conn):
        df = _binary_frame()
        out = optimize_bins_to_target(_rel(conn, df), "y", columns=["feat"], conn=conn)
        labels = set(out.df()["feat_binned"].dropna().unique())
        assert labels, "expected some bin labels"
        for label in labels:
            assert isinstance(label, str)
            assert label.startswith("(") or label in ("Single", "Missing", "Other")
            # interval labels are either "(a, b]" or the open-ended tail "(b, inf)"
            assert label.endswith("]") or label.endswith("inf)"), label

    def test_nulls_get_missing_bin(self, conn):
        df = _binary_frame()
        df.loc[df.index[:50], "feat"] = np.nan
        out = optimize_bins_to_target(_rel(conn, df), "y", columns=["feat"], conn=conn)
        labels = set(out.df()["feat_binned"].unique())
        assert "Missing" in labels

    def test_cast_to_text_false_gives_integer_bin_ids(self, conn):
        df = _binary_frame()
        out = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], cast_to_text=False, conn=conn
        )
        binned = out.df()["feat_binned"]
        assert pd.api.types.is_integer_dtype(binned)
        assert binned.min() >= 1

    def test_cast_to_text_false_with_nulls_is_a_known_bug(self, conn):
        """KNOWN BUG (not fixed here — reported to the maintainer).

        With ``cast_to_text=False`` the CASE still emits the string literal
        ``'Missing'`` for NULL rows, which DuckDB cannot cast to INT32:

            Conversion Error: Could not convert string 'Missing' to INT32

        The label is built by ``_case_sql(..., numeric_labels=True)`` which
        hardcodes the ``'Missing'`` branch.  The test pins the current
        (broken) behaviour; it will XPASS once the bug is fixed.
        """
        df = _binary_frame(n=400)
        df.loc[df.index[:20], "feat"] = np.nan
        with pytest.raises(Exception) as excinfo:
            out = optimize_bins_to_target(
                _rel(conn, df), "y", columns=["feat"], cast_to_text=False, conn=conn
            )
            out.df()  # the conversion error surfaces on materialisation
        assert "Missing" in str(excinfo.value)

    def test_suffix_empty_replaces_in_place(self, conn):
        df = _binary_frame()
        out = optimize_bins_to_target(_rel(conn, df), "y", columns=["feat"], suffix="", conn=conn)
        cols = list(out.columns)
        assert "feat" in cols
        assert "feat_binned" not in cols

    def test_multiple_columns(self, conn):
        rs = np.random.RandomState(SEED)
        n = 2000
        x = rs.normal(size=n)
        z = rs.normal(size=n)
        y = (x + rs.normal(0.0, 0.4, size=n) > 0).astype(int)
        df = pd.DataFrame({"a": x, "b": z, "y": y})
        out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["a", "b"], return_diagnostics=True, conn=conn
        )
        cols = list(out.columns)
        assert "a_binned" in cols and "b_binned" in cols
        assert set(diag) == {"a", "b"}

    def test_columns_accepts_a_bare_string(self, conn):
        df = _binary_frame(n=400)
        out = optimize_bins_to_target(_rel(conn, df), "y", columns="feat", conn=conn)
        assert "feat_binned" in list(out.columns)


# ---------------------------------------------------------------------------
# Categorical feature path
# ---------------------------------------------------------------------------


class TestCategoricalFeature:
    def test_categorical_column_is_skipped_with_warning(self, conn):
        rs = np.random.RandomState(SEED)
        n = 300
        df = pd.DataFrame(
            {
                "feat": rs.normal(size=n),
                "segment": [["red", "green", "blue"][i % 3] for i in range(n)],
                "y": rs.randint(0, 2, n),
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, diag = optimize_bins_to_target(
                _rel(conn, df), "y", columns=["segment"], return_diagnostics=True, conn=conn
            )
        assert diag["segment"]["method"] == "skipped"
        assert diag["segment"]["cuts"] is None
        assert "reason" in diag["segment"]
        assert any("non-numeric" in str(w.message) for w in caught)
        # Nothing to bin -> relation returned unchanged (no new column).
        assert "segment_binned" not in list(out.columns)

    def test_mixed_columns_bins_only_the_numeric_one(self, conn):
        rs = np.random.RandomState(SEED)
        n = 300
        df = pd.DataFrame(
            {
                "feat": rs.normal(size=n),
                "segment": [["red", "green", "blue"][i % 3] for i in range(n)],
                "y": rs.randint(0, 2, n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, diag = optimize_bins_to_target(
                _rel(conn, df), "y", columns=["feat", "segment"], return_diagnostics=True, conn=conn
            )
        assert "feat_binned" in list(out.columns)
        assert diag["segment"]["method"] == "skipped"
        assert diag["feat"]["cuts"] is not None


# ---------------------------------------------------------------------------
# Diagnostics contract
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_documented_keys_present(self, conn):
        df = _binary_frame()
        _out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], return_diagnostics=True, conn=conn
        )
        entry = diag["feat"]
        for key in ("cuts", "method", "y_kind", "learned_on", "fit_rows", "sampled", "total_rows"):
            assert key in entry, f"missing diagnostics key {key!r}"
        assert entry["method"] in ("optbinning", "tree")
        assert entry["y_kind"] == "binary"
        assert entry["fit_rows"] == len(df)
        assert entry["total_rows"] == len(df)
        assert entry["sampled"] is False
        assert entry["learned_on"] == f"full n={len(df)}"

    def test_no_diagnostics_returns_relation_only(self, conn):
        df = _binary_frame()
        out = optimize_bins_to_target(_rel(conn, df), "y", columns=["feat"], conn=conn)
        assert isinstance(out, duckdb.DuckDBPyRelation)

    def test_holdout_adds_scoring_keys(self, conn):
        df = _binary_frame()
        _out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], holdout=0.2, return_diagnostics=True, conn=conn
        )
        entry = diag["feat"]
        for key in ("scores_holdout", "holdout_frac", "holdout_n", "train_n"):
            assert key in entry, f"missing holdout diagnostics key {key!r}"
        assert entry["holdout_frac"] == 0.2
        assert entry["train_n"] + entry["holdout_n"] == len(df)
        assert entry["learned_on"].startswith("train n=")

    def test_binary_metric_values_are_float_or_nan(self, conn):
        df = _binary_frame()
        _out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], holdout=0.2, return_diagnostics=True, conn=conn
        )
        scores = diag["feat"]["scores_holdout"]
        for key in ("auc", "f1", "gini", "cramers_v"):
            assert key in scores, f"missing metric {key!r}"
            value = scores[key]
            assert isinstance(value, float)
            assert math.isnan(value) or math.isfinite(value)
        assert isinstance(scores["n_bins"], int)
        assert isinstance(scores["n"], int)

    def test_metric_values_are_finite_for_a_clean_signal(self, conn):
        df = _binary_frame()
        _out, diag = optimize_bins_to_target(
            _rel(conn, df), "y", columns=["feat"], holdout=0.2, return_diagnostics=True, conn=conn
        )
        scores = diag["feat"]["scores_holdout"]
        assert not math.isnan(scores["auc"])
        assert scores["auc"] > 0.7

    def test_silent_nan_is_the_current_contract(self, conn):
        """A single-class hold-out yields NaN metrics without raising.

        This pins the documented behaviour ("Scores all nan -> too few test
        rows or single-class holdout"): the implementation swallows the
        metric failure and reports NaN rather than raising.
        """
        rs = np.random.RandomState(SEED)
        n = 300
        df = pd.DataFrame(
            {
                "feat": rs.normal(size=n),
                "y": np.array([1] + [0] * (n - 1)),  # one positive only
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                holdout=0.1,
                stratify=False,
                return_diagnostics=True,
                conn=conn,
            )
        scores = diag["feat"]["scores_holdout"]
        assert "auc" in scores
        assert math.isnan(scores["auc"]), "expected silent NaN for a single-class holdout"
        assert math.isnan(scores["cramers_v"])

    def test_holdout_failure_is_recorded_as_holdout_error(self, conn):
        """When the train split collapses to a single class, the failure is
        recorded under ``holdout_error`` and *no* ``scores_holdout`` key is
        written at all (the key is simply absent, not None).

        The single positive lands in the test split for this seed/size, so
        the train split has a constant target and the re-learn fails.
        """
        rs = np.random.RandomState(SEED)
        n = 400
        df = pd.DataFrame(
            {
                "feat": rs.normal(size=n),
                "y": np.array([1] + [0] * (n - 1)),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                holdout=0.1,
                stratify=False,
                return_diagnostics=True,
                conn=conn,
            )
        entry = diag["feat"]
        assert "holdout_error" in entry
        assert "scores_holdout" not in entry

    def test_too_few_rows_for_scoring_reports_note(self, conn):
        rs = np.random.RandomState(SEED)
        df = pd.DataFrame({"feat": rs.normal(size=12), "y": rs.randint(0, 2, 12)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                holdout=0.4,
                return_diagnostics=True,
                conn=conn,
            )
        scores = diag["feat"]["scores_holdout"]
        assert "note" in scores
        assert scores["note"] == "too few non-null rows for scoring"

    def test_multiclass_and_continuous_target_kinds(self, conn):
        rs = np.random.RandomState(SEED)
        n = 400
        df = pd.DataFrame(
            {
                "feat": rs.normal(size=n),
                "grade": rs.randint(0, 3, n),
                "price": rs.normal(size=n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "grade",
                columns=["feat"],
                method="tree",
                holdout=0.2,
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["y_kind"] == "multiclass"
        mc = diag["feat"]["scores_holdout"]
        assert "f1_macro" in mc and "balanced_accuracy" in mc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "price",
                columns=["feat"],
                method="tree",
                holdout=0.2,
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["y_kind"] == "continuous"
        cont = diag["feat"]["scores_holdout"]
        assert "r2" in cont and "rmse" in cont
        assert isinstance(cont["r2"], float)


# ---------------------------------------------------------------------------
# sample_n behaviour
# ---------------------------------------------------------------------------


class TestSampleN:
    def test_sampling_caps_the_fit_rows_but_keeps_all_rows(self, conn):
        rs = np.random.RandomState(SEED)
        n = 3000
        x = rs.normal(size=n)
        y = (x + rs.normal(0.0, 0.4, size=n) > 0).astype(int)
        df = pd.DataFrame({"feat": x, "y": y})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                sample_n=200,
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["sampled"] is True
        assert diag["feat"]["fit_rows"] == 200
        assert diag["feat"]["total_rows"] == n
        assert diag["feat"]["learned_on"].startswith("sample n=")
        assert any("sampled" in str(w.message) for w in caught)
        # The CASE is still applied to every row.
        assert len(out.df()) == n
        assert "feat_binned" in out.columns

    def test_sampled_result_is_still_usable(self, conn):
        rs = np.random.RandomState(SEED)
        n = 3000
        x = rs.normal(size=n)
        y = (x + rs.normal(0.0, 0.4, size=n) > 0).astype(int)
        df = pd.DataFrame({"feat": x, "y": y})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                sample_n=200,
                return_diagnostics=True,
                conn=conn,
            )
        cuts = diag["feat"]["cuts"]
        assert isinstance(cuts, list)
        assert cuts == sorted(cuts)
        rates = _bin_event_rates(df, "feat", cuts)
        assert len(rates) >= 2
        # Do not assert exact values — only that the sampled fit is usable:
        # the highest bin is riskier than the lowest one.
        assert rates.iloc[-1] > rates.iloc[0]

    def test_sample_n_none_learns_on_everything(self, conn):
        df = _binary_frame(n=800)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                sample_n=None,
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["sampled"] is False
        assert diag["feat"]["fit_rows"] == 800

    def test_sample_n_larger_than_frame_does_not_sample(self, conn):
        df = _binary_frame(n=300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                sample_n=10_000,
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["sampled"] is False
        assert diag["feat"]["fit_rows"] == 300

    def test_sampling_and_holdout_combine(self, conn):
        rs = np.random.RandomState(SEED)
        n = 3000
        x = rs.normal(size=n)
        y = (x + rs.normal(0.0, 0.4, size=n) > 0).astype(int)
        df = pd.DataFrame({"feat": x, "y": y})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                sample_n=500,
                holdout=0.2,
                return_diagnostics=True,
                conn=conn,
            )
        entry = diag["feat"]
        assert entry["sampled"] is True
        assert entry["fit_rows"] == 500
        assert entry["learned_on"].startswith("train n=")
        assert entry["train_n"] + entry["holdout_n"] == 500


# ---------------------------------------------------------------------------
# Engine selection / optbinning options
# ---------------------------------------------------------------------------


class TestEngineSelection:
    def test_optbinning_kwargs_are_forwarded(self, conn):
        df = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                optbinning_kwargs={"max_n_bins": 4, "monotonic_trend": "ascending"},
                return_diagnostics=True,
                conn=conn,
            )
        cuts = diag["feat"]["cuts"]
        assert isinstance(cuts, list)
        assert len(cuts) <= 3, "max_n_bins=4 should cap the number of cuts"

    def test_solver_is_recorded_for_optbinning(self, conn):
        pytest.importorskip("optbinning")
        df = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                method="optbinning",
                return_diagnostics=True,
                conn=conn,
            )
        entry = diag["feat"]
        assert entry["method"] == "optbinning"
        assert entry["solver"] in ("cp", "mip")

    def test_explicit_solver_is_recorded(self, conn):
        pytest.importorskip("optbinning")
        df = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                method="optbinning",
                optbinning_kwargs={"solver": "mip"},
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["solver"] == "mip"

    def test_tree_method_is_used_and_recorded(self, conn):
        df = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                method="tree",
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["method"] == "tree"
        assert isinstance(diag["feat"]["cuts"], list)

    def test_explicit_optbinning_on_multiclass_falls_back_with_warning(self, conn):
        rs = np.random.RandomState(SEED)
        df = pd.DataFrame({"feat": rs.normal(size=300), "y": rs.randint(0, 3, 300)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                method="optbinning",
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["method"] == "tree"
        assert any("not binary" in str(w.message) for w in caught)

    def test_explicit_optbinning_on_continuous_target_falls_back(self, conn):
        rs = np.random.RandomState(SEED)
        df = pd.DataFrame({"feat": rs.normal(size=300), "y": rs.normal(size=300)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                method="optbinning",
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["method"] == "tree"
        assert diag["feat"]["y_kind"] == "continuous"

    def test_string_binary_target_with_holdout(self, conn):
        rs = np.random.RandomState(SEED)
        n = 800
        x = rs.normal(size=n)
        y = np.where(x + rs.normal(0.0, 0.4, size=n) > 0, "yes", "no")
        df = pd.DataFrame({"feat": x, "y": y})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                holdout=0.2,
                return_diagnostics=True,
                conn=conn,
            )
        entry = diag["feat"]
        assert entry["y_kind"] == "binary"
        scores = entry["scores_holdout"]
        assert "auc" in scores
        assert not math.isnan(scores["auc"])

    def test_default_columns_infers_all_non_target_columns(self, conn):
        rs = np.random.RandomState(SEED)
        n = 400
        df = pd.DataFrame(
            {"a": rs.normal(size=n), "b": rs.normal(size=n), "y": rs.randint(0, 2, n)}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, diag = optimize_bins_to_target(
                _rel(conn, df), "y", return_diagnostics=True, conn=conn
            )
        assert set(diag) == {"a", "b"}
        cols = list(out.columns)
        assert "a_binned" in cols and "b_binned" in cols
        assert "y_binned" not in cols


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_unknown_feature_column_raises(self, conn):
        df = _binary_frame(n=200)
        with pytest.raises(ValueError, match="Unknown column"):
            optimize_bins_to_target(_rel(conn, df), "y", columns=["nope"], conn=conn)

    def test_unknown_target_column_raises(self, conn):
        df = _binary_frame(n=200)
        with pytest.raises(ValueError, match="Unknown column"):
            optimize_bins_to_target(_rel(conn, df), "not_a_target", columns=["feat"], conn=conn)

    def test_constant_target_raises(self, conn):
        df = _binary_frame(n=200)
        df["y"] = 0
        with pytest.raises(ValueError, match="distinct non-null value"):
            optimize_bins_to_target(_rel(conn, df), "y", columns=["feat"], conn=conn)

    def test_non_numeric_target_is_accepted_as_binary_labels(self, conn):
        """String labels are factorised to 0/1 — not an error."""
        rs = np.random.RandomState(SEED)
        n = 400
        x = rs.normal(size=n)
        y = np.where(x + rs.normal(0.0, 0.4, size=n) > 0, "yes", "no")
        df = pd.DataFrame({"feat": x, "y": y})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df), "y", columns=["feat"], return_diagnostics=True, conn=conn
            )
        assert diag["feat"]["y_kind"] == "binary"
        assert len(diag["feat"]["cuts"]) >= 1

    def test_task_conflict_raises(self, conn):
        rs = np.random.RandomState(SEED)
        df = pd.DataFrame({"feat": rs.normal(size=300), "y": rs.randint(0, 3, 300)})
        with pytest.raises(ValueError, match="conflicts with y"):
            optimize_bins_to_target(
                _rel(conn, df), "y", columns=["feat"], task="binary", method="tree", conn=conn
            )

    def test_invalid_task_raises(self, conn):
        df = _binary_frame(n=200)
        with pytest.raises(ValueError, match="task must be"):
            optimize_bins_to_target(
                _rel(conn, df), "y", columns=["feat"], task="nonsense", conn=conn
            )

    def test_too_few_rows_for_requested_bins_yields_single_bin(self, conn):
        """With fewer rows than the learner needs, no splits are invented."""
        df = pd.DataFrame({"feat": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                method="tree",
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["cuts"] == []
        assert set(out.df()["feat_binned"].unique()) == {"Single"}

    def test_negative_sample_n_is_rejected_or_falls_back(self, conn):
        """A negative sample_n must never silently corrupt the fit.

        The current implementation warns and learns on the full frame
        (train_test_split refuses a negative train_size), which is safe.
        """
        df = _binary_frame(n=300)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _out, diag = optimize_bins_to_target(
                _rel(conn, df),
                "y",
                columns=["feat"],
                sample_n=-1,
                return_diagnostics=True,
                conn=conn,
            )
        assert diag["feat"]["sampled"] is False
        assert diag["feat"]["fit_rows"] == 300
        assert any("sampling failed" in str(w.message) for w in caught)

    def test_holdout_fraction_out_of_range_raises(self, conn):
        df = _binary_frame(n=300)
        for bad in (0.0, 0.6, -0.1, 0.5):
            with pytest.raises(ValueError, match="holdout fraction"):
                optimize_bins_to_target(
                    _rel(conn, df), "y", columns=["feat"], holdout=bad, conn=conn
                )

    def test_existing_output_column_raises(self, conn):
        df = _binary_frame(n=200)
        df["feat_binned"] = 1
        with pytest.raises(ValueError, match="already exists"):
            optimize_bins_to_target(_rel(conn, df), "y", columns=["feat"], conn=conn)

    def test_invalid_method_raises(self, conn):
        df = _binary_frame(n=200)
        with pytest.raises(ValueError, match="method must be"):
            optimize_bins_to_target(
                _rel(conn, df), "y", columns=["feat"], method="magic", conn=conn
            )


# ---------------------------------------------------------------------------
# Internal helpers (interval labelling)
# ---------------------------------------------------------------------------


class TestBinHelpers:
    def test_assign_bin_intervals(self):
        cuts = [-0.5, 0.5]
        assert _assign_bin(-1.0, cuts) == "(-inf, -0.5]"
        assert _assign_bin(0.0, cuts) == "(-0.5, 0.5]"
        assert _assign_bin(1.0, cuts) == "(0.5, inf)"
        assert _assign_bin(None, cuts) == "Missing"
        assert _assign_bin(float("nan"), cuts) == "Missing"
        assert _assign_bin(1.0, []) == "Single"

    def test_case_sql_mentions_every_cut(self):
        sql = _case_sql('"feat"', [-0.5, 0.5])
        assert "CASE" in sql and "END" in sql
        assert "-0.5" in sql and "0.5" in sql
        assert "IS NULL" in sql

    def test_detect_y_kind(self):
        assert _detect_y_kind(pd.Series([0, 1, 1, 0])) == "binary"
        assert _detect_y_kind(pd.Series(["a", "b", "c"])) == "multiclass"
        assert _detect_y_kind(pd.Series(np.linspace(0, 1, 200))) == "continuous"
        assert _detect_y_kind(pd.Series([0, 1, 0]), task="binary") == "binary"
        assert _detect_y_kind(pd.Series([0.0, 1.0, 2.0]), task="regression") == "continuous"
        assert _detect_y_kind(pd.Series([0, 1, 0]), task="multiclass") == "multiclass"
        assert _detect_y_kind(pd.Series(range(30))) == "multiclass"
        assert _detect_y_kind(pd.Series(range(200))) == "continuous"
        assert _detect_y_kind(pd.Series(["a", "b", "c"], dtype="category")) == "multiclass"
        assert _detect_y_kind(pd.Series([True, False, True])) == "binary"
        with pytest.raises(ValueError):
            _detect_y_kind(pd.Series([0, 1]), task="bogus")
        with pytest.raises(ValueError):
            _detect_y_kind(pd.Series([1, 1, 1]))


# ---------------------------------------------------------------------------
# Scoring helpers (per-bin metric computation)
# ---------------------------------------------------------------------------


class TestScoreBinnedFeature:
    def test_binary_scores(self):
        rs = np.random.RandomState(SEED)
        n = 300
        x = rs.normal(size=n)
        y = pd.Series((x > 0).astype(int))
        scores = _score_binned_feature(pd.Series(x), y, [-0.5, 0.5], "binary")
        assert set(scores) == {"auc", "f1", "gini", "cramers_v", "n_bins", "n"}
        assert 0.0 <= scores["auc"] <= 1.0
        assert scores["gini"] == pytest.approx(2 * scores["auc"] - 1)
        assert scores["n"] == n
        assert scores["n_bins"] == 3

    def test_string_binary_target_is_factorised(self):
        rs = np.random.RandomState(SEED)
        n = 300
        x = rs.normal(size=n)
        y = pd.Series(np.where(x > 0, "yes", "no"))
        scores = _score_binned_feature(pd.Series(x), y, [-0.5, 0.5], "binary")
        assert 0.0 <= scores["auc"] <= 1.0
        assert scores["n"] == n

    def test_multiclass_scores(self):
        rs = np.random.RandomState(SEED)
        n = 300
        x = rs.normal(size=n)
        y = pd.Series(rs.randint(0, 3, n))
        scores = _score_binned_feature(pd.Series(x), y, [-0.5, 0.5], "multiclass")
        assert set(scores) == {"f1_macro", "balanced_accuracy", "n_bins", "n"}
        assert 0.0 <= scores["f1_macro"] <= 1.0
        assert 0.0 <= scores["balanced_accuracy"] <= 1.0

    def test_continuous_scores(self):
        rs = np.random.RandomState(SEED)
        n = 300
        x = rs.normal(size=n)
        y = pd.Series(x * 2.0 + rs.normal(size=n))
        scores = _score_binned_feature(pd.Series(x), y, [-0.5, 0.5], "continuous")
        assert set(scores) == {"r2", "rmse", "n_bins", "n"}
        assert isinstance(scores["r2"], float)
        assert scores["rmse"] >= 0.0

    def test_too_few_rows_returns_note(self):
        x = pd.Series([1.0, 2.0, 3.0])
        y = pd.Series([0, 1, 0])
        scores = _score_binned_feature(x, y, [1.5], "binary")
        assert scores == {"n": 3, "note": "too few non-null rows for scoring"}

    def test_train_map_is_used_for_holdout_scoring(self):
        rs = np.random.RandomState(SEED)
        n = 300
        x = rs.normal(size=n)
        y = pd.Series((x > 0).astype(int))
        train_map = {"(-inf, -0.5]": 0.1, "(-0.5, 0.5]": 0.5, "(0.5, inf)": 0.9}
        scores = _score_binned_feature(pd.Series(x), y, [-0.5, 0.5], "binary", _train_map=train_map)
        assert scores["n"] == n
        assert scores["n_bins"] == 3


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


class TestBackendHelpers:
    def test_optbinning_splits_returns_none_for_non_binary_target(self):
        rs = np.random.RandomState(SEED)
        x = pd.Series(rs.normal(size=200))
        y = pd.Series(rs.randint(0, 3, 200))
        assert _optbinning_splits(x, y) is None

    def test_optbinning_splits_for_binary_target(self):
        pytest.importorskip("optbinning")
        rs = np.random.RandomState(SEED)
        n = 400
        x = pd.Series(rs.normal(size=n))
        y = pd.Series((x > 0).astype(int))
        cuts = _optbinning_splits(x, y)
        assert cuts is not None
        assert cuts == sorted(cuts)

    def test_tree_splits_returns_sorted_thresholds(self):
        rs = np.random.RandomState(SEED)
        n = 400
        x = pd.Series(rs.normal(size=n), name="feat")
        y = pd.Series((x > 0).astype(int), name="y")
        cuts = _tree_splits(x, y, method="tree", cv=3, task="auto", random_state=SEED)
        assert isinstance(cuts, list)
        assert cuts == sorted(cuts)

    def test_tree_splits_returns_empty_for_tiny_input(self):
        x = pd.Series([1.0, 2.0, 3.0], name="feat")
        y = pd.Series([0, 1, 0], name="y")
        assert _tree_splits(x, y, method="tree", cv=3, task="auto", random_state=SEED) == []

    def test_tree_splits_handles_an_unnamed_series(self):
        """An unnamed Series falls back to the sklearn tree (feature_engine
        needs a column name) but still returns usable cuts."""
        rs = np.random.RandomState(SEED)
        n = 400
        x = pd.Series(rs.normal(size=n))
        y = pd.Series((x > 0).astype(int))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cuts = _tree_splits(x, y, method="tree", cv=3, task="auto", random_state=SEED)
        assert isinstance(cuts, list)
        assert cuts == sorted(cuts)


# ---------------------------------------------------------------------------
# Standalone (no explicit connection) entry point
# ---------------------------------------------------------------------------


class TestStandaloneCall:
    def test_works_without_an_explicit_connection(self, conn):
        df = _binary_frame(n=400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = optimize_bins_to_target(_rel(conn, df), "y", columns=["feat"])
        assert "feat_binned" in list(out.columns)
        assert len(out.df()) == len(df)

    def test_empty_columns_raises(self, conn):
        df = pd.DataFrame({"y": [0, 1, 0, 1]})
        with pytest.raises(ValueError, match="no work"):
            optimize_bins_to_target(_rel(conn, df), "y", columns=[], conn=conn)
