"""Estimation helpers shared by the modelling scripts (OLS with entity-clustered SEs, design matrices)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

CONTROLS_PART = ["ln_base", "ln_plazo", "plazo_miss"]
FE_PART = ["year", "unspsc4", "orden"]


def part_mask(df: pd.DataFrame, ratio_lo: float = 0.30) -> pd.Series:
    """Participation-sample definition with a configurable lower trimming bound for award/budget."""
    s2 = df["plataforma"] == "SECOP II"
    return (s2 & df["tipo_contrato"].eq("OBRA") & df["regime"].isin(["minima", "menor", "licitacion"])
            & df["add_ratio_raw"].notna() & df["n_bidders"].notna() & df["precio_base"].gt(0)
            & df["award_ratio"].between(ratio_lo, 1.0001))


def prepare_part(df: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    """Participation sample with derived variables used in every model."""
    d = df.loc[(df["s_part"] == 1) if mask is None else mask].copy()
    d["ln_plazo_raw"] = np.log(d["plazo_proc_dias"].where(d["plazo_proc_dias"] > 0))
    d["plazo_miss"] = d["ln_plazo_raw"].isna().astype(float)
    d["ln_plazo"] = d["ln_plazo_raw"].fillna(d["ln_plazo_raw"].median())
    d["net_saving"] = 1 - d["final_ratio_w"]                       # = 1 - final price / reference budget
    d["rebate_raw"] = 1 - d["award_ratio"].clip(upper=1.0)
    d["add_base"] = d["rebate_raw"] - d["net_saving"]               # additions as share of reference budget (identity)
    d["rebate_pp"] = 100 * d["rebate_raw"]
    d["net_saving_pp"] = 100 * d["net_saving"]
    d["add_base_pp"] = 100 * d["add_base"]
    d["add_any_pp"] = 100 * d["add_any"]
    d["add_ratio_pp"] = 100 * d["add_ratio"]
    d["bid_s"] = (d["n_bidders"] - d["n_bidders"].mean()) / d["n_bidders"].std()
    d["ent_g"] = pd.factorize(d["ent_id"])[0]
    d["unspsc4"] = d["unspsc4"].astype(str)
    d["year"] = d["year"].astype(int).astype(str)
    return d


def design(d: pd.DataFrame, x: list[str], fe: list[str], extra_dummies: list[str] | None = None) -> pd.DataFrame:
    X = d[x].astype(float).copy()
    for f in fe + (extra_dummies or []):
        X = pd.concat([X, pd.get_dummies(d[f].astype(str), prefix=f, drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    # drop constant-within-sample columns (collinear dummies in small subsamples)
    keep = [c for c in X.columns if c == "const" or X[c].std() > 0]
    return X[keep]


def ols_cluster(d: pd.DataFrame, y: str, x: list[str], fe: list[str], cluster: str = "ent_g",
                extra_dummies: list[str] | None = None, weights: str | None = None):
    X = design(d, x, fe, extra_dummies)
    yv = d[y].astype(float)
    mask = yv.notna() & X.notna().all(axis=1)
    X, yv, g = X[mask], yv[mask], d.loc[mask, cluster]
    mod = sm.OLS(yv, X) if weights is None else sm.WLS(yv, X, weights=d.loc[mask, weights])
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0], "use_correction": True})
    res.n_clusters = int(g.nunique())
    res.X_, res.y_, res.g_ = X, yv, g
    return res


def pstar(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
