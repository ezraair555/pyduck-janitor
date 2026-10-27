"""optimize_bins_to_target — supervised binning example.

Install the binning deps first:  pip install "pyduck-janitor[binning]"
"""
import duckdb
import numpy as np
import pandas as pd
from pyduck_janitor import DuckJanitor
from pyduck_janitor.bins_opt import optimize_bins_to_target

rng = np.random.default_rng(42)
n = 6000
ages = rng.integers(18, 78, n)
inc = rng.lognormal(10.7, 0.35, n)
logit = -2.8 + 0.045 * (ages - 40) - 0.00003 * inc + (ages < 28) * 1.2
bad = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"age": ages, "income": inc, "bad": bad})
df["risk_grade"] = np.select([inc < 30_000, inc < 60_000], [0, 1], 2)
df["price"] = np.log(inc) + rng.normal(0, 0.30, n)

# Binary y (default auto uses OptBinning when available)
con = duckdb.connect()
rel = con.from_df(df)
rel_bin, diag = optimize_bins_to_target(
    rel, y="bad", columns=["age", "income"],
    method="auto", holdout=0.2, return_diagnostics=True, conn=con,
)
print("Binary  ->", diag["age"]["cuts"], "AUC", round(diag["age"]["scores_holdout"]["auc"], 3))
print(rel_bin.df().head().to_string(index=False))

# Multiclass y (tree)
rel2, d2 = optimize_bins_to_target(rel, y="risk_grade", columns=["income"], method="tree", holdout=True, return_diagnostics=True, conn=con)
print("\nMulticlass ->", d2["income"])

# Continuous y (tree regressor)
rel3, d3 = optimize_bins_to_target(rel, y="price", columns=["age"], method="tree", holdout=0.2, return_diagnostics=True, conn=con)
print("\nContinuous ->", d3["age"])

# Chainable on DuckJanitor
from pyduck_janitor import DuckJanitor as DJ
res = DJ.from_pandas(df).optimize_bins_to_target("bad", columns=["age"], method="tree", holdout=0.2).collect()
print("\nChain head:\n", res[["age", "age_binned"]].head().to_string())
