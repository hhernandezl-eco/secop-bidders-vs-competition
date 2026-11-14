"""STEP 18 - What explains the gap in the frequency of additions between SECOP I (reported) and SECOP II (derived)?

In the firm sample, 17% of SECOP I contracts and 40% of SECOP II contracts show an addition. Three explanations compete:
(i) under-reporting in SECOP I (additions not uploaded), (ii) the derived SECOP II measure also picks up larger quantities of work
and price adjustments that raise the contract value, mostly in small amounts, and (iii) composition (entities, years, sectors
and contract types that migrated first). We examine (a) the distribution of the size of the addition by platform,
(b) the platform gap after controlling for composition and for municipality fixed effects (municipalities with contracts on both platforms; entity identifiers differ across platforms),
and (c) the agreement between the derived measure and the number of added days, which SECOP II reports for a fraction of contracts.

Outputs: tab_addgap.tex, tables/addgap.csv, numbers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ANALYSIS_PQ, TABLES
from models import ols_cluster
from tex import coef_cell, fint, fnum, put_number, write_numbers_tex, write_table

TH = {0.001: "any", 0.10: "ten", 0.25: "twentyfive"}
BINS = [(0.001, 0.01, "0.1--1\\%"), (0.01, 0.05, "1--5\\%"), (0.05, 0.10, "5--10\\%"), (0.10, 0.25, "10--25\\%"), (0.25, 0.475, "25--47.5\\%"), (0.475, 1.0, "47.5--100\\%")]


def main() -> None:
    df = pd.read_parquet(ANALYSIS_PQ)
    f = df[df.s_firm == 1].copy()
    f["s2"] = (f.plataforma == "SECOP II").astype(float)
    r = f["add_ratio_raw"]
    # ---- (a) distribution of the size of additions, as a share of all contracts ----
    body = []
    for lo, hi, lab in BINS:
        c = []
        for plat in ["SECOP I", "SECOP II"]:
            m = f.plataforma == plat
            sh = 100 * ((r[m] > lo) & (r[m] <= hi)).mean()
            c.append(sh)
        body.append(f"Addition of {lab} of the initial value & {fnum(c[0], 1)} & {fnum(c[1], 1)} & {fnum(c[1] - c[0], 1)} \\\\")
    tot = [100 * (r[f.plataforma == p] > 0.001).mean() for p in ["SECOP I", "SECOP II"]]
    body.append("\\midrule")
    body.append(f"Any addition ($>0.1\\%$) & {fnum(tot[0], 1)} & {fnum(tot[1], 1)} & {fnum(tot[1] - tot[0], 1)} \\\\")
    put_number("addgap total secopone", tot[0], ".1")
    put_number("addgap total secoptwo", tot[1], ".1")
    # share of the gap in additions above 10%
    big = [100 * (r[f.plataforma == p] > 0.10).mean() for p in ["SECOP I", "SECOP II"]]
    put_number("addgap big secopone", big[0], ".1")
    put_number("addgap big secoptwo", big[1], ".1")
    put_number("addgap share of gap above ten", (big[1] - big[0]) / (tot[1] - tot[0]) * 100, ".0f")

    # ---- (b) platform gap with composition controls and entity fixed effects ----
    f["ln_valor"] = f["ln_valor"].astype(float)
    f["year_s"] = f["year"].astype(str)
    f["tipo"] = f["tipo_contrato"].fillna("NA").astype(str)
    f["ent_s"] = f["departamento"].astype(str).str.lower().str.strip() + "|" + f["municipio"].astype(str).str.lower().str.strip()
    plats = f.groupby("ent_s")["s2"].agg(["min", "max"])
    both = plats.index[(plats["min"] == 0) & (plats["max"] == 1)]
    put_number("addgap municipalities both", len(both), ",d")
    fb = f[f.ent_s.isin(both)]
    put_number("addgap n both", len(fb), ",d")
    body.append("\\midrule")
    body.append("\\multicolumn{4}{l}{\\emph{Platform gap (SECOP II minus SECOP I), pp, from linear probability models}} \\\\")
    body.append("& No controls & Controls & Municipality FE \\\\")
    for y_lab, thr in [("Any addition ($>0.1\\%$)", 0.001), ("Addition above 10\\%", 0.10), ("Addition above 25\\%", 0.25)]:
        f["y"] = 100 * (f["add_ratio_raw"] > thr).astype(float)
        fb = f[f.ent_s.isin(both)].copy()
        cells = []
        for nm, d, xs, fe, ex in [("raw", f, ["s2"], [], None),
                                  ("ctrl", f, ["s2", "ln_valor"], ["year_s", "unspsc4", "tipo"], None),
                                  ("fe", fb, ["s2", "ln_valor"], ["year_s", "tipo"], ["ent_s"])]:
            res = ols_cluster(d, "y", xs, fe, cluster="ent_id", extra_dummies=ex)
            c1, c2 = coef_cell(res.params["s2"], res.bse["s2"], res.pvalues["s2"], 1)
            cells.append(f"{c1} {c2}")
            put_number(f"addgap {nm} {TH[thr]}", res.params["s2"], ".1")
        body.append(f"{y_lab} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    write_table("tab_addgap", "\n".join(body), "", "tab:addgap",
                "Firm sample. Top: share of contracts (\\%) whose addition falls in each size band, by platform. Bottom: coefficient on a SECOP~II indicator, in percentage points, from linear probability models; the first column has no controls, the second adds $\\ln$ value, year, UNSPSC-4 and contract-type fixed effects, "
                "and the third adds municipality fixed effects and uses only municipalities with contracts on both platforms (entity identifiers differ across platforms, so entities cannot be matched). Standard errors clustered by entity in parentheses.",
                "lccc", "& SECOP I & SECOP II & Difference")
    write_numbers_tex()
    print("addgap done")


if __name__ == "__main__":
    main()
