"""STEP 14 - Right-censoring / selection into 'closed' contracts.

The analysis sample keeps contracts that are closed or terminated at the snapshot date. If contracts with difficult
renegotiations stay open longer, and if that depends on participation, the closed sample is selected.
We therefore (i) document the share of signed works contracts that are closed, by signing year and regime,
(ii) test whether being closed is related to ln(bidders) conditional on controls (cohorts 2018-2021, long since
signed), and (iii) re-estimate the main effects with inverse-probability weights for being closed.

Outputs: tab_closed.tex, tab_closed_test.tex, numbers
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import INTERIM, SECOP2_CONTRACTS_PQ, SECOP2_PROCESSES_PQ, TABLES
from models import CONTROLS_PART, design, ols_cluster
from tex import coef_cell, fint, fnum, put_number, write_numbers_tex, write_table

LN2 = np.log(2.0)


def q(p):
    return str(p).replace("\\", "/")


def main() -> None:
    con = duckdb.connect()
    tmp = INTERIM / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{q(tmp)}'")
    con.execute("SET memory_limit='2500MB'")
    con.execute("SET threads=2")
    con.execute(f"""
        CREATE TABLE c AS
        SELECT * EXCLUDE rn FROM (
          SELECT id_contrato, proceso_compra, estado_contrato, fecha_firma, modalidad, nit_entidad, orden_entidad,
                 substr(unspsc_raw, 4, 6) AS unspsc6, substr(unspsc_raw, 4, 4) AS unspsc4, valor_contrato,
                 row_number() OVER (PARTITION BY id_contrato ORDER BY fecha_firma NULLS LAST) AS rn
          FROM read_parquet('{q(SECOP2_CONTRACTS_PQ)}')
          WHERE (substr(unspsc_raw, 4, 2) IN ('72','95') OR substr(unspsc_raw, 4, 6) = '811015')
            AND fecha_firma BETWEEN DATE '2018-01-01' AND DATE '2024-12-31') WHERE rn = 1""")
    con.execute(f"""
        CREATE TABLE a AS
        SELECT id_portafolio, any_value(prov_unicos_resp) AS n_bidders, any_value(precio_base) AS precio_base,
               any_value(valor_adjudicado) AS valor_adjudicado, any_value(duracion) AS duracion, any_value(unidad_duracion) AS unidad,
               count(DISTINCT (id_proceso, nit_adjudicado, valor_adjudicado)) AS n_award
        FROM read_parquet('{q(SECOP2_PROCESSES_PQ)}')
        WHERE adjudicado = 'Si' AND id_portafolio IN (SELECT proceso_compra FROM c)
        GROUP BY 1""")
    d = con.execute("SELECT c.*, a.n_bidders, a.precio_base, a.valor_adjudicado, a.duracion, a.unidad FROM c JOIN a ON a.id_portafolio = c.proceso_compra WHERE a.n_award = 1").fetchdf()
    d["regime"] = d["modalidad"].map(lambda m: ("minima" if "nima cuant" in str(m).lower() else "menor" if "menor cuant" in str(m).lower()
                                                 else "licitacion" if "licitaci" in str(m).lower() else "otra"))
    d = d[d.regime.isin(["minima", "menor", "licitacion"]) & d.n_bidders.gt(0) & d.precio_base.gt(0) & d.valor_adjudicado.gt(0)].copy()
    d = d[(d.valor_adjudicado / d.precio_base).between(0.30, 1.0001)]
    d["year"] = pd.to_datetime(d["fecha_firma"]).dt.year
    d["closed"] = d["estado_contrato"].isin(["Cerrado", "terminado"]).astype(float)
    d["ln_bidders"] = np.log(d["n_bidders"].astype(float))
    d["ln_base"] = np.log(d["precio_base"].astype(float))
    dur = d["duracion"].astype(float) * np.select([d.unidad.fillna("").str.lower().str.startswith("m"), d.unidad.fillna("").str.lower().str.startswith("d"),
                                                    d.unidad.fillna("").str.lower().str.startswith("a")], [30.0, 1.0, 365.0], default=np.nan)
    d["ln_plazo_raw"] = np.log(dur.where(dur > 0))
    d["plazo_miss"] = d["ln_plazo_raw"].isna().astype(float)
    d["ln_plazo"] = d["ln_plazo_raw"].fillna(d["ln_plazo_raw"].median())
    d["orden"] = d["orden_entidad"].fillna("NA")
    d["year"] = d["year"].astype(str)
    d["unspsc4"] = d["unspsc4"].astype(str)
    d["ent_g"] = pd.factorize(d["nit_entidad"])[0]
    put_number("cens n", len(d), ",d")
    put_number("cens closed share", 100 * d.closed.mean(), ".0")

    # (i) closed share by year and regime
    tab = d.pivot_table(index="year", columns="regime", values="closed", aggfunc="mean") * 100
    cnt = d.groupby("year").size()
    body = []
    for yr, r in tab.iterrows():
        body.append(f"{yr} & {fint(cnt[yr])} & {r['minima']:.0f} & {r['menor']:.0f} & {r['licitacion']:.0f} \\\\")
    tab.to_csv(TABLES / "closed_share.csv")
    write_table("tab_closed", "\n".join(body), "", "tab:closed",
                "Share (\\%) of signed SECOP II works contracts in the participation-sample definition (single award, bidders observed, positive budget, award/budget ratio in [0.30, 1.00]) that are closed or terminated at the extraction date, by signing year and award regime. "
                "Open contracts are excluded from all analysis samples.", "lcccc", "Year signed & Contracts & M\\'inima (\\%) & Menor (\\%) & Licitaci\\'on (\\%)")
    for yr, nm in [("2018", "first"), ("2021", "mid"), ("2022", "later"), ("2023", "late"), ("2024", "last")]:
        put_number(f"cens share lic {nm}", tab.loc[yr, "licitacion"], ".0")
        put_number(f"cens share min {nm}", tab.loc[yr, "minima"], ".0")

    # (ii) is being closed related to participation? cohorts 2018-2021
    old = d[d.year.astype(int) <= 2021]
    body = []
    for g, lab in [("minima", "M\\'inima cuant\\'ia"), ("menor", "Menor cuant\\'ia"), ("licitacion", "Licitaci\\'on p\\'ublica"), ("scored", "Scored (menor + licitaci\\'on)")]:
        s = old[old.regime.isin(["menor", "licitacion"])] if g == "scored" else old[old.regime == g]
        r = ols_cluster(s.assign(closed_pp=100 * s["closed"]), "closed_pp", ["ln_bidders"] + CONTROLS_PART, ["year", "unspsc4", "orden"] + (["regime"] if g == "scored" else []))
        c1, c2 = coef_cell(r.params["ln_bidders"] * LN2, r.bse["ln_bidders"] * LN2, r.pvalues["ln_bidders"], 2)
        body += [f"{lab} & {c1} & {fint(r.nobs)} & {s['closed'].mean() * 100:.0f} \\\\", f" & {c2} & & \\\\"]
        put_number(f"cens test {g}", r.params["ln_bidders"] * LN2, ".2")
        put_number(f"censp test {g}", r.pvalues["ln_bidders"], ".3")
    write_table("tab_closed_test", "\n".join(body), "", "tab:closedtest",
                "Linear probability model for a contract being closed or terminated (pp), signing years 2018--2021: effect of doubling bidders, with baseline controls and fixed effects for year, UNSPSC-4 class and entity order. Standard errors clustered by entity.",
                "lccc", "& Effect on Pr(closed), pp & $N$ & Closed (\\%)")

    # (iii) inverse-probability weights for being closed (observables), main scored effects, outcomes from the closed sample
    br = __import__("importlib").import_module("08_robustness")
    w = br.broad_sample()
    w["year"] = w["year"].astype(str)
    key = w.merge(d[["id_contrato"]].assign(in_d=1), on="id_contrato", how="left")
    X = design(d, ["ln_bidders", "ln_base", "ln_plazo", "plazo_miss"], ["year", "unspsc4", "orden", "regime"])
    lg = sm.Logit(d["closed"].values, X).fit(disp=0, maxiter=200, method="bfgs")
    d["p_closed"] = lg.predict(X)
    wmap = (1 / d.set_index("id_contrato")["p_closed"].clip(lower=0.05))
    w["ipw"] = w["id_contrato"].map(wmap)
    out = []
    for g, lab in [("minima", "M\\'inima"), ("scored", "Scored")]:
        s = (w[w.regime.isin(["menor", "licitacion"])] if g == "scored" else w[w.regime == g]).dropna(subset=["ipw"])
        for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
            r = ols_cluster(s, y, ["ln_bidders"] + CONTROLS_PART, ["year", "unspsc4", "orden"] + (["regime"] if g == "scored" else []), weights="ipw")
            put_number(f"ipw {g} {y}", r.params["ln_bidders"] * LN2, ".2")
            put_number(f"ipwp {g} {y}", r.pvalues["ln_bidders"], ".3")
    # (iv) include contracts that are still open (cohorts 2018-2022), measuring additions accumulated to the snapshot,
    #      and Manski-type worst-case bounds for the contracts whose final outcome is not yet realised
    d["add_any_pp"] = 100 * ((d["valor_contrato"] / d["valor_adjudicado"] - 1) > 0.001).astype(float)
    d["rebate_pp"] = 100 * (1 - (d["valor_adjudicado"] / d["precio_base"]).clip(upper=1.0))
    d["net_saving_pp"] = 100 * (1 - (d["valor_contrato"] / d["precio_base"]).clip(upper=2.0))
    cohort = d[d.year.astype(int) <= 2022].copy()
    put_number("open cohort n", len(cohort), ",d")
    put_number("open cohort closed", 100 * cohort.closed.mean(), ".0")
    cohort["add_up"] = np.where(cohort.closed == 1, cohort["add_any_pp"], 100.0)      # open contracts may still receive an addition
    cohort["net_low"] = np.where(cohort.closed == 1, cohort["net_saving_pp"], -100.0)  # ... and the net saving cannot fall below -100%
    body = []
    for g, lab in [("minima", "M\\'inima cuant\\'ia"), ("scored", "Scored regimes")]:
        s0 = cohort[cohort.regime == "minima"] if g == "minima" else cohort[cohort.regime.isin(["menor", "licitacion"])]
        fe_g = ["year", "unspsc4", "orden"] + (["regime"] if g == "scored" else [])
        rows_ = []
        for nm, data, ya, yn in [("Closed contracts only", s0[s0.closed == 1], "add_any_pp", "net_saving_pp"),
                                 ("Closed and open, additions to the snapshot", s0, "add_any_pp", "net_saving_pp"),
                                 ("Worst-case bound (open contracts at the extreme)", s0, "add_up", "net_low")]:
            ra = ols_cluster(data, ya, ["ln_bidders"] + CONTROLS_PART, fe_g)
            rn = ols_cluster(data, yn, ["ln_bidders"] + CONTROLS_PART, fe_g)
            rows_.append((nm, ra, rn, len(data)))
        for nm, ra, rn, n in rows_:
            c1, c2 = coef_cell(ra.params["ln_bidders"] * LN2, ra.bse["ln_bidders"] * LN2, ra.pvalues["ln_bidders"], 2)
            c3, c4 = coef_cell(rn.params["ln_bidders"] * LN2, rn.bse["ln_bidders"] * LN2, rn.pvalues["ln_bidders"], 2)
            body += [f"{lab}: {nm} & {c1} & {c3} & {fint(n)} \\\\", f" & {c2} & {c4} & \\\\"]
        put_number(f"open {g} all add", rows_[1][1].params["ln_bidders"] * LN2, ".2")
        put_number(f"openp {g} all add", rows_[1][1].pvalues["ln_bidders"], ".3")
        put_number(f"open {g} all net", rows_[1][2].params["ln_bidders"] * LN2, ".2")
        put_number(f"openp {g} all net", rows_[1][2].pvalues["ln_bidders"], ".3")
        put_number(f"open {g} add up", rows_[2][1].params["ln_bidders"] * LN2, ".2")
        put_number(f"open {g} net low", rows_[2][2].params["ln_bidders"] * LN2, ".2")
    write_table("tab_open", "\n".join(body), "", "tab:open",
                "Contracts signed in 2018--2022, closed and still open at the extraction date. Open contracts carry the value reported to the snapshot, a lower bound of their final additions. The worst-case rows set the addition indicator of every open contract to one (upper bound for additions) and its net saving to $-100$ pp (lower bound for the net saving), the Manski-type extremes consistent with any future outcome. "
                "Effect (pp) of doubling bidders; baseline controls and fixed effects; standard errors clustered by entity.", "lccc",
                "& Any addition & Net saving & $N$")
    write_numbers_tex()
    print("ok", len(d))


if __name__ == "__main__":
    main()
