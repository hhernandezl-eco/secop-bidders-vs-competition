"""STEP 07 - Supply side: do observable bidder capabilities predict renegotiation?

Design choices:
  * outcome measured on the same footing in SECOP I and II (no SECOP II 'zeros');
  * experience measured strictly before signing; track record uses *completed* contracts only;
  * within-firm estimates exclude the lagged-outcome regressor (mechanical Nickell bias; shown separately);
  * regressors standardised so magnitudes are comparable (SD units).
Also reports out-of-sample predictive validity (AUC) with a time split, which is what a screening
criterion has to deliver.

Outputs: output/latex/tables/tab_firms.tex, tab_auc.tex, output/numbers.json, output/tables/firms_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ANALYSIS_PQ, N_BOOT, SEED, TABLES
from tex import coef_cell, fint, fnum, put_number, write_numbers_tex, write_table


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def prepare_firms(df: pd.DataFrame) -> pd.DataFrame:
    d = df.loc[df["s_firm"] == 1].copy()
    d["ln_plazo_raw"] = np.log(d["plazo_dias"].where(d["plazo_dias"] > 0))
    d["plazo_miss"] = d["ln_plazo_raw"].isna().astype(float)
    d["ln_plazo"] = d["ln_plazo_raw"].fillna(d["ln_plazo_raw"].median())
    d["z_spec"] = zscore(d["ln_spec_exp"])
    d["z_vol"] = zscore(d["ln_vol_exp"])
    d["hist_miss"] = d["hist_mod"].isna().astype(float)
    mu, sd = d["hist_mod"].mean(), d["hist_mod"].std()
    d["z_hist"] = ((d["hist_mod"] - mu) / sd).fillna(0.0)
    d["add_any_pp"] = 100 * d["add_any"]
    d["add_ratio_pp"] = 100 * d["add_ratio"]
    d["s2"] = (d["plataforma"] == "SECOP II").astype(float)
    d["ent_g"] = pd.factorize(d["ent_id"])[0]
    d["year"] = d["year"].astype(int).astype(str)
    d["unspsc4"] = d["unspsc4"].astype(str)
    d["reg"] = d["modalidad_marco"].fillna("NA")
    return d


def dummies(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.concat([pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float) for c in cols], axis=1)


def fit(d: pd.DataFrame, y: str, xs: list[str], fe_cols: list[str], absorb: str | None = None):
    X = pd.concat([d[xs].astype(float), dummies(d, fe_cols)], axis=1)
    yv = d[y].astype(float)
    if absorb is None:
        X = sm.add_constant(X)
    else:                                          # within transformation
        g = d[absorb]
        X = X - X.groupby(g).transform("mean")
        yv = yv - yv.groupby(g).transform("mean")
    X = X.loc[:, X.std() > 1e-12] if absorb else X.loc[:, (X.std() > 0) | (X.columns == "const")]
    res = sm.OLS(yv, X).fit(cov_type="cluster", cov_kwds={"groups": d["ent_g"].values, "use_correction": True})
    return res


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC."""
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    s = pd.Series(p)
    ranks = s.rank(method="average").values
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> None:
    df = pd.read_parquet(ANALYSIS_PQ)
    d = prepare_firms(df)
    put_number("n firm", len(d), ",d")
    put_number("nfirms firm", d["firm_id"].nunique(), ",d")
    put_number("n firm secopone", int((d.plataforma == "SECOP I").sum()), ",d")
    put_number("n firm secoptwo", int((d.plataforma == "SECOP II").sum()), ",d")
    base_fe = ["year", "departamento", "orden", "tipo_contrato", "reg", "unspsc4"]
    base_x = ["ln_valor", "ln_plazo", "plazo_miss", "s2"]
    rows = []
    out = {}
    for y in ["add_any_pp", "add_ratio_pp"]:
        specs = {
            "(1)": (base_x, base_fe, None),
            "(2)": (base_x + ["z_vol", "z_spec"], base_fe, None),
            "(3)": (base_x + ["z_vol", "z_spec", "z_hist", "hist_miss"], base_fe, None),
            "(4)": (base_x + ["z_vol", "z_spec"], ["year"], "firm_id"),
            "(5)": (base_x + ["z_vol", "z_spec", "z_hist", "hist_miss"], ["year", "tipo_contrato"], "ent_id"),
        }
        for k, (xs, fe, ab) in specs.items():
            sub = d
            if ab == "firm_id":                       # firms observed at least twice
                cnt = d.groupby("firm_id")["id_contrato"].transform("size")
                sub = d[cnt >= 2]
            res = fit(sub, y, xs, fe, ab)
            out[(y, k)] = (res, len(sub), sub["firm_id"].nunique() if ab == "firm_id" else None)
            for v in ["z_vol", "z_spec", "z_hist"]:
                if v in res.params.index:
                    rows.append(dict(y=y, spec=k, var=v, b=res.params[v], se=res.bse[v], p=res.pvalues[v], n=int(res.nobs)))
    pd.DataFrame(rows).to_csv(TABLES / "firms_reg.csv", index=False)

    def block(y: str, title: str) -> list[str]:
        lines = [f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for v, lab in [("z_vol", "Volume of experience (z)"), ("z_spec", "Specific experience, same UNSPSC (z)"),
                       ("z_hist", "Track record: prior addition rate (z)")]:
            l1, l2 = [lab], [""]
            for k in ["(2)", "(3)", "(4)", "(5)"]:                  # spec (1), controls only, is the AUC benchmark
                res = out[(y, k)][0]
                if v in res.params.index:
                    c1, c2 = coef_cell(res.params[v], res.bse[v], res.pvalues[v], 2)
                else:
                    c1, c2 = "", ""
                l1.append(c1)
                l2.append(c2)
            lines += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
        return lines

    body = block("add_any_pp", "Panel A. Outcome: any addition (0/100)") + ["\\addlinespace"] + \
        block("add_ratio_pp", "Panel B. Outcome: addition as \\% of initial value")
    nrow = " & ".join(["Observations"] + [fint(out[("add_any_pp", k)][1]) for k in ["(2)", "(3)", "(4)", "(5)"]]) + " \\\\"
    body += ["\\addlinespace", nrow,
             "Fixed effects & Year, dept., order, type, regime, UNSPSC4 & Same as (1) & Firm, year & Entity, year, type \\\\"]
    header = "& (1) Experience & (2) + Track record & (3) Firm FE & (4) Entity FE"
    notes = ("OLS with standard errors clustered by contracting entity. Regressors in standard-deviation units. Experience is counted strictly before the "
             "signing date; the track record is the share of the firm's earlier \\emph{completed} works contracts with an addition (set to zero with a "
             "missing-value indicator for firms without history). All columns control for $\\ln$ initial value, $\\ln$ planned duration and a SECOP~II indicator. "
             "Column (3) uses only within-firm variation (firms with at least two contracts) and omits the track-record regressor, which is mechanically "
             "negatively correlated with the within-firm deviation of the outcome (lagged-outcome/Nickell bias; see text). "
             "* $p<0.10$, ** $p<0.05$, *** $p<0.01$.")
    write_table("tab_firms", "\n".join(body), "", "tab:firms", notes, "lccccc", header)
    for v in ["z_vol", "z_spec", "z_hist"]:
        for k, nm in [("(3)", "pool"), ("(4)", "fe")]:
            if v in out[("add_any_pp", k)][0].params.index:
                r = out[("add_any_pp", k)][0]
                put_number(f"firm {nm} {v}", r.params[v], ".2")
                put_number(f"firm p {nm} {v}", r.pvalues[v], ".3")

    # --- Nickell illustration: within-firm coefficient on the lagged track record -------------
    cnt = d.groupby("firm_id")["id_contrato"].transform("size")
    sub = d[cnt >= 2]
    r_nick = fit(sub, "add_any_pp", base_x + ["z_vol", "z_spec", "z_hist", "hist_miss"], ["year"], "firm_id")
    put_number("nickell hist", r_nick.params["z_hist"], ".1")

    # --- predictive validity: time split ---------------------------------------------------
    d = d.dropna(subset=["add_any"]).copy()
    train = d[d["year"].astype(int) <= 2021]
    test = d[d["year"].astype(int) >= 2022]
    sets = {
        "Contract controls only": base_x,
        "+ volume of experience": base_x + ["z_vol"],
        "+ specific experience": base_x + ["z_spec"],
        "+ track record": base_x + ["z_hist", "hist_miss"],
        "All capabilities": base_x + ["z_vol", "z_spec", "z_hist", "hist_miss"],
    }
    fe_cols = ["orden", "tipo_contrato", "reg"]
    rng = np.random.default_rng(SEED)
    res_auc = []
    for nm, xs in sets.items():
        Xtr = sm.add_constant(pd.concat([train[xs].astype(float), dummies(train, fe_cols)], axis=1))
        Xte = sm.add_constant(pd.concat([test[xs].astype(float), dummies(test, fe_cols)], axis=1))
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
        keep = Xtr.columns[(Xtr.std() > 0) | (Xtr.columns == "const")]
        m = sm.Logit(train["add_any"].values, Xtr[keep]).fit(disp=0, maxiter=200, method="bfgs")
        p = m.predict(Xte[keep])
        y = test["add_any"].values.astype(int)
        a = auc(y, p.values)
        bs = []
        for _ in range(300):
            ix = rng.integers(0, len(y), len(y))
            bs.append(auc(y[ix], p.values[ix]))
        res_auc.append(dict(model=nm, auc=a, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n_train=len(train), n_test=len(test)))
    A = pd.DataFrame(res_auc)
    A.to_csv(TABLES / "firms_auc.csv", index=False)
    body = []
    for _, r in A.iterrows():
        body.append(f"{r['model']} & {fnum(r['auc'], 3)} & [{fnum(r['lo'], 3)}; {fnum(r['hi'], 3)}] \\\\")
    put_number("auc base", A.loc[0, "auc"], ".3")
    put_number("auc all", A.loc[4, "auc"], ".3")
    put_number("auc spec", A.loc[2, "auc"], ".3")
    put_number("auc hist", A.loc[3, "auc"], ".3")
    put_number("auc vol", A.loc[1, "auc"], ".3")
    put_number("auc ntrain", len(train), ",d")
    put_number("auc ntest", len(test), ",d")
    header = "Model & AUC (test sample) & 95\\% CI"
    notes = (f"Logit models estimated on contracts signed in 2018--2021 ($N$ = {len(train):,}) and evaluated on contracts signed in 2022--2024 ($N$ = {len(test):,}). "
             "AUC = probability that a randomly chosen contract with an addition receives a higher predicted risk than one without (0.5 = no discrimination). "
             "Bootstrap intervals (300 resamples of the test sample). Later contracts may still be in execution, which understates additions in the test sample.")
    write_table("tab_auc", "\n".join(body), "", "tab:auc", notes, "lcc", header)
    write_numbers_tex()
    print(A.round(4).to_string())
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
