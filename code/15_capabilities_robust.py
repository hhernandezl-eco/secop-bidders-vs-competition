"""STEP 15 - Capabilities: platform-specific estimates and a leave-one-entity-out track record.

(i)  Table of capabilities by platform (SECOP I only, SECOP II only) and test of platform interactions.
(ii) Leave-one-entity-out (LOEO) track record on the broad SECOP II works universe: the firm's addition rate in
     earlier *completed* contracts signed with entities other than the focal one. It is free of the mechanical
     (Nickell) bias of a lagged own-outcome regressor and of market segmentation, because the focal entity's own
     practice never enters the regressor. Entity fixed effects are then added.
(iii) Do winners' track records rise with participation (selection of renegotiation-prone entrants)?

Outputs: tab_cap_platform.tex, tab_loeo.tex, numbers
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ANALYSIS_PQ, INTERIM, TABLES
from models import CONTROLS_PART, design, ols_cluster
from tex import coef_cell, fint, fnum, put_number, write_numbers_tex, write_table

LN2 = np.log(2.0)


def loeo_rates(focal: pd.DataFrame, pan: pd.DataFrame) -> pd.DataFrame:
    """focal: nit, ent, t ; pan: nit, ent, end, y. Returns n and rate of earlier completed contracts with *other* entities,
    and the same quantities including the focal entity ('own')."""
    out = pd.DataFrame(index=focal.index, columns=["n_loeo", "rate_loeo", "n_all", "rate_all"], dtype=float)
    groups = {k: g for k, g in pan.groupby("nit")}
    for k, g in focal.groupby("nit"):
        p = groups.get(k)
        if p is None:
            out.loc[g.index, ["n_loeo", "n_all"]] = 0.0
            continue
        e_end = p["end"].values.astype("datetime64[ns]")
        e_ent = p["ent"].values
        y = p["y"].values
        for idx, ent, t in zip(g.index, g["ent"].values, g["t"].values.astype("datetime64[ns]")):
            done = e_end <= t
            m = done & (e_ent != ent)
            out.at[idx, "n_all"] = done.sum()
            out.at[idx, "n_loeo"] = m.sum()
            if done.any():
                out.at[idx, "rate_all"] = y[done].mean()
            if m.any():
                out.at[idx, "rate_loeo"] = y[m].mean()
    return out


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def main() -> None:
    f07 = importlib.import_module("07_firms")
    br = importlib.import_module("08_robustness")

    # ---------------- (i) capabilities by platform ----------------
    df = pd.read_parquet(ANALYSIS_PQ)
    d = f07.prepare_firms(df)
    base_x = ["ln_valor", "ln_plazo", "plazo_miss"]
    fe = ["year", "departamento", "orden", "tipo_contrato", "reg", "unspsc4"]
    xs = base_x + ["z_vol", "z_spec", "z_hist", "hist_miss"]
    cols = []
    for lab, sub in [("SECOP I only", d[d.plataforma == "SECOP I"]), ("SECOP II only", d[d.plataforma == "SECOP II"])]:
        res = f07.fit(sub, "add_any_pp", xs, fe)
        cols.append((lab, res, len(sub)))
    # platform interactions (pooled, platform-specific slopes tested jointly)
    dd = d.copy()
    for v in ["z_vol", "z_spec", "z_hist"]:
        dd[v + "_s2"] = dd[v] * dd["s2"]
    resi = f07.fit(dd, "add_any_pp", base_x + ["s2", "z_vol", "z_spec", "z_hist", "hist_miss", "z_vol_s2", "z_spec_s2", "z_hist_s2"], fe)
    wt = resi.wald_test("z_vol_s2 = 0, z_spec_s2 = 0, z_hist_s2 = 0", use_f=True, scalar=True)
    put_number("plat int p", float(wt.pvalue), ".3")
    body = []
    for v, lab in [("z_vol", "Volume of experience (z)"), ("z_spec", "Specific experience, same UNSPSC (z)"), ("z_hist", "Track record: prior addition rate (z)")]:
        l1, l2 = [lab], [""]
        for _, res, _n in cols:
            c1, c2 = coef_cell(res.params[v], res.bse[v], res.pvalues[v], 2)
            l1.append(c1)
            l2.append(c2)
            put_number(f"cap {_ + ' '.join(['x'])[0:0]}{'one' if _.startswith('SECOP I only') else 'two'} {v}", res.params[v], ".2")
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    body += ["\\addlinespace", "Observations & " + " & ".join(fint(n) for _, _, n in cols) + " \\\\",
             f"$p$-value, joint test of platform differences in slopes & \\multicolumn{{2}}{{c}}{{{float(wt.pvalue):.3f}}} \\\\"]
    write_table("tab_cap_platform", "\n".join(body), "", "tab:capplatform",
                "Capability regressions of \\cref{tab:firms}, column (2), estimated separately for each platform. The outcome is the probability of an addition (pp), observed in SECOP I and derived in SECOP II; regressors in standard-deviation units; "
                "same controls and fixed effects. The last row tests the joint equality of the three slopes across platforms in a pooled model with platform interactions. Standard errors clustered by entity.",
                "lcc", "& SECOP I only & SECOP II only")

    # ---------------- (ii) leave-one-entity-out track record, broad SECOP II universe ----------------
    works = pd.read_parquet(INTERIM / "secop2_works_universe.parquet")
    wk = works.loc[(works.outcome_ok == 1) & works.nit_proveedor.notna() & works.nit_entidad.notna()].copy()
    from utils import clean_nit
    wk["nit"] = wk["nit_proveedor"].map(clean_nit)
    pan = pd.DataFrame({"nit": wk["nit"], "ent": wk["nit_entidad"], "sign": pd.to_datetime(wk["fecha_firma"]),
                        "end": pd.to_datetime(wk["fecha_fin"].fillna(wk["fecha_firma"] + pd.Timedelta(days=365))),
                        "y": (wk["add_ratio_raw"] > 0.001).astype(float)}).dropna()
    pan["end"] = np.maximum(pan["end"].values.astype("datetime64[ns]"), (pan["sign"] + pd.Timedelta(days=1)).values.astype("datetime64[ns]"))
    w = br.broad_sample()
    w["nit"] = w["nit_proveedor"].map(clean_nit)
    w = w.dropna(subset=["nit"]).copy()
    foc = pd.DataFrame({"nit": w["nit"].values, "ent": w["nit_entidad"].values, "t": pd.to_datetime(w["fecha_firma"]).values}, index=w.index)
    r = loeo_rates(foc, pan)
    w = w.join(r)
    w["add_pp"] = w["add_any_pp"]
    for nm, col, ncol in [("own", "rate_all", "n_all"), ("loeo", "rate_loeo", "n_loeo")]:
        w[f"{nm}_miss"] = w[col].isna().astype(float)
        mu, sd = w[col].mean(), w[col].std()
        w[f"z_{nm}"] = ((w[col] - mu) / sd).fillna(0.0)
    put_number("loeo share", 100 * (1 - w["rate_loeo"].isna().mean()), ".0")
    put_number("loeo n", len(w), ",d")
    w["ent_str"] = w["ent_g"].astype(str)
    ctrl = ["ln_base", "ln_plazo", "plazo_miss"]
    fe_b = ["year", "unspsc4", "orden", "regime"]
    specs = [("(1) Own-history track record, no entity FE", ["z_own", "own_miss"], False),
             ("(2) Own-history track record, entity FE", ["z_own", "own_miss"], True),
             ("(3) LOEO track record, no entity FE", ["z_loeo", "loeo_miss"], False),
             ("(4) LOEO track record, entity FE", ["z_loeo", "loeo_miss"], True),
             ("(5) Both, entity FE", ["z_own", "own_miss", "z_loeo", "loeo_miss"], True)]
    body, rows = [], []
    header_cells = []
    for lab, xs_, use_fe in specs:
        res = ols_cluster(w, "add_pp", xs_ + ctrl, fe_b, extra_dummies=["ent_str"] if use_fe else None)
        rows.append((lab, res, use_fe))
    def cell(res, v):
        return coef_cell(res.params[v], res.bse[v], res.pvalues[v], 2) if v in res.params.index else ("", "")
    for v, lab in [("z_own", "Own-history track record (z)"), ("z_loeo", "Leave-one-entity-out track record (z)")]:
        l1, l2 = [lab], [""]
        for _, res, _f in rows:
            c1, c2 = cell(res, v)
            l1.append(c1)
            l2.append(c2)
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    body += ["\\addlinespace", "Observations & " + " & ".join(fint(res.nobs) for _, res, _ in rows) + " \\\\",
             "Entity fixed effects & " + " & ".join("Yes" if f else "No" for _, _, f in rows) + " \\\\"]
    for i, (lab, res, use_fe) in enumerate(rows, 1):
        for v in ["z_own", "z_loeo"]:
            if v in res.params.index:
                put_number(f"loeo spec{['one','two','three','four','five'][i-1]} {v}", res.params[v], ".2")
                put_number(f"loeop spec{['one','two','three','four','five'][i-1]} {v}", res.pvalues[v], ".3")
    write_table("tab_loeo", "\n".join(body), "", "tab:loeo",
                "Broad SECOP II works universe. Outcome: probability of an addition (pp). The own-history track record is the share of the firm's earlier completed works contracts with an addition; the leave-one-entity-out (LOEO) track record uses only earlier "
                "contracts signed with entities other than the focal one, so that the focal entity's practice never enters the regressor. Both standardised among firms with history (zero and a missing indicator otherwise). Controls: $\\ln$ budget, $\\ln$ duration; fixed effects for year, UNSPSC-4, entity order and regime. "
                "Standard errors clustered by entity.", "lccccc", "& (1) & (2) & (3) & (4) & (5)")

    # ---------------- (iii) do winners' track records rise with participation? ----------------
    for g, sub in [("scored", w[w.regime.isin(["menor", "licitacion"])]), ("minima", w[w.regime == "minima"])]:
        s = sub.dropna(subset=["rate_all"])
        fe_g = ["year", "unspsc4", "orden"] + (["regime"] if g == "scored" else [])
        s = s.assign(rate_pp=100 * s["rate_all"])
        r1 = ols_cluster(s, "rate_pp", ["ln_bidders"] + ctrl, fe_g)
        r2 = ols_cluster(s, "rate_pp", ["ln_bidders"] + ctrl, ["year", "unspsc4"] + (["regime"] if g == "scored" else []), extra_dummies=["ent_str"])
        put_number(f"winhist {g}", r1.params["ln_bidders"] * LN2, ".2")
        put_number(f"winhistp {g}", r1.pvalues["ln_bidders"], ".3")
        put_number(f"winhistfe {g}", r2.params["ln_bidders"] * LN2, ".2")
        put_number(f"winhistfep {g}", r2.pvalues["ln_bidders"], ".3")
        put_number(f"winhist n {g}", int(r1.nobs), ",d")
        # 95% intervals (pp per doubling of bidders): the test has limited power, so the intervals matter more than the p-values
        ci1, ci2 = r1.conf_int().loc["ln_bidders"] * LN2, r2.conf_int().loc["ln_bidders"] * LN2
        put_number(f"winhistlo {g}", ci1[0], ".1")
        put_number(f"winhisthi {g}", ci1[1], ".1")
        put_number(f"winhistfelo {g}", ci2[0], ".1")
        put_number(f"winhistfehi {g}", ci2[1], ".1")
        put_number(f"winhist mean {g}", s["rate_pp"].mean(), ".0")
        put_number(f"winhist sd {g}", s["rate_pp"].std(), ".0")
    write_numbers_tex()
    print("ok")


if __name__ == "__main__":
    main()
