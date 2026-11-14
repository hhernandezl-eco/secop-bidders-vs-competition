"""STEP 17 - Award rule versus project scale: comparison of contracts of the same size in minimum wages.

The legal ceiling of m\'inima cuant\'ia is 10% of the entity's ceiling for menor cuant\'ia, and the latter depends on the entity's annual
budget in minimum monthly wages (SMMLV): 28, 45, 65, 85 or 100 SMMLV for m\'inima cuant\'ia (Ley 1150 of 2007, art. 2, as regulated).
Hence between 28 and 100 SMMLV the same contract (same size in SMMLV) is awarded by lowest price in an entity with a large budget
and by a scored rule in an entity with a small budget. We compare the participation slopes of the two groups on the window
(28, 100] SMMLV, where scale differs little, and control for the entity's size, imputed from its annual contracting in SECOP II,
and then for the entity itself (entities that use both rules in the window, because their budget band or the wage changes).

Outputs: tab_zone.tex, tables/zone.csv, numbers
"""
from __future__ import annotations

import importlib

import duckdb
import numpy as np
import pandas as pd

from config import INTERIM, SECOP2_CONTRACTS_PQ, TABLES
from models import CONTROLS_PART, ols_cluster
from tex import fint, fnum, put_number, star, write_numbers_tex, write_table

LN2 = np.log(2.0)
SMMLV = {2018: 781242, 2019: 828116, 2020: 877803, 2021: 908526, 2022: 1000000, 2023: 1160000, 2024: 1300000}
LO, HI = 28.0, 100.0


def entity_year_contracting() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET memory_limit='2500MB'")
    con.execute("SET threads=2")
    q = str(SECOP2_CONTRACTS_PQ).replace("\\", "/")
    ent = con.execute(f"""SELECT nit_entidad, year(fecha_firma) AS yr, sum(valor_contrato) AS tot
                          FROM (SELECT DISTINCT id_contrato, nit_entidad, fecha_firma, valor_contrato FROM read_parquet('{q}'))
                          WHERE fecha_firma >= DATE '2017-01-01' GROUP BY 1, 2""").fetchdf()
    ent["tot_smmlv"] = ent["tot"] / ent["yr"].map(SMMLV)
    return ent


def main() -> None:
    br = importlib.import_module("08_robustness")
    w = br.broad_sample()
    w["yr"] = w["year"].astype(int)
    w["b_smmlv"] = w["precio_base"] / w["yr"].map(SMMLV)
    w = w.merge(entity_year_contracting()[["nit_entidad", "yr", "tot_smmlv"]], on=["nit_entidad", "yr"], how="left")
    w["ln_ent"] = np.log(w["tot_smmlv"].clip(lower=1).fillna(1))
    w["ent_band"] = np.select([w.tot_smmlv >= 120000, w.tot_smmlv >= 80000, w.tot_smmlv >= 40000, w.tot_smmlv >= 15000], ["b5", "b4", "b3", "b2"], "b1")
    z = w[(w.b_smmlv > LO) & (w.b_smmlv <= HI) & w.regime.isin(["minima", "menor"])].copy()
    z["is_min"] = (z.regime == "minima").astype(float)
    put_number("zone n", len(z), ",d")
    put_number("zone n minima", int(z.is_min.sum()), ",d")
    put_number("zone n scored", int((1 - z.is_min).sum()), ",d")
    put_number("zone median minima", float(z.loc[z.is_min == 1, "b_smmlv"].median()), ".0f")
    put_number("zone median scored", float(z.loc[z.is_min == 0, "b_smmlv"].median()), ".0f")
    put_number("zone entities scored", int(z.loc[z.is_min == 0, "nit_entidad"].nunique()), ",d")
    c = np.log(z["b_smmlv"]) - np.log(z["b_smmlv"]).median()
    for k in (1, 2, 3):
        z[f"c{k}"] = c ** k
    z["lnb_min"] = z["ln_bidders"] * z["is_min"]
    z["lnb_sc"] = z["ln_bidders"] * (1 - z["is_min"])
    z["c1_Nm"] = z["c1"] * z["lnb_min"]
    z["c1_Ns"] = z["c1"] * z["lnb_sc"]
    z["ent_str"] = z["ent_g"].astype(str)
    both = z.groupby("ent_str")["is_min"].agg(["min", "max"])
    both = both.index[(both["min"] == 0) & (both["max"] == 1)]
    put_number("zone entities both", len(both), ",d")
    zb = z[z.ent_str.isin(both)].copy()
    put_number("zone n both", len(zb), ",d")

    # Discretionary choice of modality: the legal minima-cuantia ceiling implied by the entity's imputed size band.
    # A menor-cuantia contract at or below that ceiling (or a minima contract above it) departs from the imputed legal rule:
    # either the entity chose the scored process for a contract that legally fits minima cuantia (e.g. to ask for the RUP or to score
    # technical factors) or the imputed band is wrong. Check (4)-(5) drops those contracts.
    lim = z["ent_band"].map({"b1": 28.0, "b2": 45.0, "b3": 65.0, "b4": 85.0, "b5": 100.0})
    z["rule_match"] = ((z["b_smmlv"] <= lim) == (z["is_min"] == 1))
    sc_ = z[z.is_min == 0]
    put_number("zone share match", 100 * float(z.rule_match.mean()), ".0f")
    put_number("zone share scored below ceiling", 100 * float((sc_["b_smmlv"] <= lim[sc_.index]).mean()), ".0f")
    put_number("zone n match", int(z.rule_match.sum()), ",d")
    zm = z[z.rule_match].copy()
    bm = zm.groupby("ent_str")["is_min"].agg(["min", "max"])
    bm = bm.index[(bm["min"] == 0) & (bm["max"] == 1)]
    zbm = zm[zm.ent_str.isin(bm)].copy()
    put_number("zone entities both match", len(bm), ",d")
    put_number("zone n scored match", int((1 - zm.is_min).sum()), ",d")

    base = ["c1", "c2", "c3", "c1_Nm", "c1_Ns", "ln_ent"] + ["ln_plazo", "plazo_miss"]
    specs = [("(1) Same-size contracts, 28--100 SMMLV", z, base, ["year", "unspsc4", "orden"], None),
             ("(2) + entity-size band", z, base, ["year", "unspsc4", "orden", "ent_band"], None),
             ("(3) Entities using both rules (entity fixed effects)", zb, [b for b in base if b != "ln_ent"], ["year", "unspsc4"], "ent_str"),
             ("(4) As (2), contracts whose rule matches the imputed legal band", zm, base, ["year", "unspsc4", "orden", "ent_band"], None),
             ("(5) As (3), contracts whose rule matches the imputed legal band", zbm, [b for b in base if b != "ln_ent"], ["year", "unspsc4"], "ent_str")]
    body, rows = [], []
    for lab, dd, xs, fe, ent_fe in specs:
        l1, l2 = [lab], [""]
        for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
            res = ols_cluster(dd, y, ["lnb_min", "lnb_sc", "is_min"] + xs, fe, extra_dummies=[ent_fe] if ent_fe else None)
            b, se, p = res.params["lnb_min"] * LN2, res.bse["lnb_min"] * LN2, res.pvalues["lnb_min"]
            b2, se2, p2 = res.params["lnb_sc"] * LN2, res.bse["lnb_sc"] * LN2, res.pvalues["lnb_sc"]
            dif = res.t_test("lnb_min - lnb_sc = 0")
            d_, pd_ = (res.params["lnb_min"] - res.params["lnb_sc"]) * LN2, float(dif.pvalue)
            l1.append(f"{fnum(b, 2)}{star(p)} / {fnum(b2, 2)}{star(p2)}")
            l2.append(f"diff. {fnum(d_, 2)} ($p$={fnum(pd_, 3)})")
            rows.append(dict(spec=lab, y=y, minima=b, scored=b2, diff=d_, p_diff=pd_, n=int(res.nobs)))
            tag = "ABCDE"[int(lab[1]) - 1]
            put_number(f"zone{tag} {y} minima", b, ".2")
            put_number(f"zone{tag} {y} scored", b2, ".2")
            put_number(f"zone{tag} {y} diff", d_, ".2")
            put_number(f"zone{tag} {y} p", pd_, ".3")
        l1.append(fint(res.nobs))
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    pd.DataFrame(rows).to_csv(TABLES / "zone.csv", index=False)
    write_table("tab_zone", "\n".join(body), "", "tab:zone",
                "Effect (pp) of doubling bidders in m\\'inima cuant\\'ia / in menor cuant\\'ia, for contracts whose reference budget lies between 28 and 100 monthly minimum wages (SMMLV), "
                "where the same contract is awarded by lowest price in entities with a large annual budget and by a scored rule in entities with a small one. "
                "Regime-specific slopes, with a cubic polynomial in $\\ln$ budget (in SMMLV, centred at the median of the window), $\\ln B\\times\\ln N$ by regime, $\\ln$ duration, year and UNSPSC-4 fixed effects; "
                "(1) adds entity-order fixed effects and the log of the entity's annual contracting in SECOP II as a proxy of its size, (2) fixed effects for five size bands that mirror the legal bands, (3) contracting-entity fixed effects, "
                "using entities that award contracts in the window under both rules; (4) and (5) repeat (2) and (3) after dropping the contracts whose observed rule departs from the legal band imputed to the entity "
                "(a scored process for a contract that legally fits m\\'inima cuant\\'ia, or the reverse), which removes discretionary choice of modality and classification error. Standard errors clustered by entity. * $p<0.10$, ** $p<0.05$, *** $p<0.01$.",
                "lccc c".replace(" ", ""), "& Rebate & Any addition & Net saving & $N$")
    write_numbers_tex()
    print("zone done", len(z), len(zb))


if __name__ == "__main__":
    main()
