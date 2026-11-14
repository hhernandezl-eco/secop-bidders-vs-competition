"""STEP 16 - Construction of the broad works universe and its split by Documentos Tipo scope.

(a) Sample-construction cascade of the broad SECOP II works universe, from the raw contract file to the estimation sample.
    It is written as Panel B of the sample-construction table (Panel A is the transport frame, from step 02/05).
(b) Split of the broad universe into contracts that belong to the transport-infrastructure frame, to which the
    Documentos Tipo apply directly, and all other works (buildings, other civil works, ...), whose tender documents
    are drawn up by each entity under Ley 80/Ley 1150. Participation effects are estimated separately, so that we can tell
    whether the blocked price margin in the scored regimes is specific to the TRM-based formula of the Documentos Tipo
    or is a general feature of scored selection in Colombia.

Outputs: tab_flow.tex (two panels), tab_dtsplit.tex, tables/broad_cascade.csv, tables/dtsplit.csv, numbers
"""
from __future__ import annotations

import importlib
import re

import numpy as np
import pandas as pd

from config import FRAME_PQ, INTERIM, LOGS, TABLES
from models import CONTROLS_PART, ols_cluster
from tex import coef_cell, fint, fnum, put_number, star, write_numbers_tex, write_table

LN2 = np.log(2.0)


def regime_of(m) -> str:
    m = str(m).lower()
    return "minima" if "nima cuant" in m else "menor" if "menor cuant" in m else "licitacion" if "licitaci" in m else "otra"


def cascade() -> list[tuple[str, int]]:
    log = (LOGS / "03_build_secop2_works.log").read_text(encoding="utf-8")
    raw, dist = [int(x.replace(",", "")) for x in re.search(r"raw rows ([\d,]+); distinct id_contrato ([\d,]+)", log).groups()]
    w = pd.read_parquet(INTERIM / "secop2_works_universe.parquet")
    steps = [("SECOP II contract records, all sectors (raw file)", raw),
             ("distinct contracts", dist),
             ("works-type (UNSPSC 72, 95, 811015), closed or terminated, signed 2015 onwards", len(w))]
    w = w[w.outcome_ok == 1]
    steps.append(("single award row and positive awarded value (renegotiation measurable)", len(w)))
    w = w[w.prov_unicos_resp.gt(0) & w.precio_base.gt(0)]
    steps.append(("bidders observed and reference budget $>$ 0", len(w)))
    w = w.assign(regime=w["modalidad"].map(regime_of))
    w = w[w.regime.isin(["minima", "menor", "licitacion"])]
    steps.append(("m\\'inima cuant\\'ia, menor cuant\\'ia or licitaci\\'on p\\'ublica", len(w)))
    w = w[(w["valor_adjudicado"] / w["precio_base"]).between(0.30, 1.0001)]
    steps.append(("award/reference-budget ratio in $[0.30,\\,1.00]$", len(w)))
    w = w[pd.to_datetime(w["fecha_firma"]).dt.year.between(2018, 2024)]
    steps.append(("signed 2018--2024 $\\rightarrow$ broad works universe", len(w)))
    return steps


def split_table(w: pd.DataFrame) -> None:
    f = pd.read_parquet(FRAME_PQ, columns=["id_contrato", "plataforma"])
    ids = set(f.loc[f.plataforma == "SECOP II", "id_contrato"])
    w = w.copy()
    w["in_frame"] = w["id_contrato"].isin(ids).astype(float)
    w["lnN_f"] = w["ln_bidders"] * w["in_frame"]
    w["lnN_nf"] = w["ln_bidders"] * (1 - w["in_frame"])
    put_number("dtsplit n frame", int(w.in_frame.sum()), ",d")
    put_number("dtsplit n other", int((1 - w.in_frame).sum()), ",d")
    top = (w[w.in_frame == 0].groupby("unspsc4").size().sort_values(ascending=False).head(6))
    top.to_csv(TABLES / "dtsplit_top_unspsc_other.csv")
    print(top)
    rows, body = [], []
    for g, lab in [("minima", "M\\'inima cuant\\'ia"), ("scored", "Scored (menor + licitaci\\'on)")]:
        s = w[w.regime == "minima"] if g == "minima" else w[w.regime.isin(["menor", "licitacion"])]
        for y_lab, y in [("Rebate", "rebate_pp"), ("Any addition", "add_any_pp"), ("Net saving", "net_saving_pp")]:
            fe = ["year", "unspsc4", "orden"] + (["regime"] if g == "scored" else [])
            res = ols_cluster(s, y, ["lnN_f", "lnN_nf", "in_frame"] + CONTROLS_PART, fe)
            a, sa, pa = res.params["lnN_f"] * LN2, res.bse["lnN_f"] * LN2, res.pvalues["lnN_f"]
            b, sb, pb = res.params["lnN_nf"] * LN2, res.bse["lnN_nf"] * LN2, res.pvalues["lnN_nf"]
            pd_ = float(res.t_test("lnN_f - lnN_nf = 0").pvalue)
            rows.append(dict(regime=g, y=y, frame=a, other=b, p_equal=pd_))
            put_number(f"dtsplit {g} {y} frame", a, ".2")
            put_number(f"dtsplit {g} {y} other", b, ".2")
            put_number(f"dtsplit {g} {y} p", pd_, ".3")
            body.append(f"{lab if y == 'rebate_pp' else ''} & {y_lab} & {fnum(a, 2)}{star(pa)} & {fnum(b, 2)}{star(pb)} & {fnum(pd_, 3)} \\\\")
            body.append(f" & & ({fnum(sa, 2)}) & ({fnum(sb, 2)}) & \\\\")
        body.append(f"\\addlinespace")
        body.append(f" & $N$ & {fint(int(((s.in_frame == 1)).sum()))} & {fint(int(((s.in_frame == 0)).sum()))} & \\\\")
        body.append("\\midrule")
    body = body[:-1]
    pd.DataFrame(rows).to_csv(TABLES / "dtsplit.csv", index=False)
    write_table("tab_dtsplit", "\n".join(body), "", "tab:dtsplit",
                "Effect (pp) of doubling bidders in the broad works universe, estimated separately for contracts in the transport-infrastructure frame, to which the Documentos Tipo apply directly, and for all other works, whose tender documents are drawn up by each entity. "
                "One pooled regression per row group with regime-specific slopes for the two groups; the last column is the $p$-value of equality of the two slopes. Year, UNSPSC-4, entity-order (and regime in the scored rows) fixed effects and baseline controls; standard errors clustered by entity. "
                "Frame membership is the classification of the sampling frame (UNSPSC code and keyword rules), used here as an approximation to the scope of the Documentos Tipo. * $p<0.10$, ** $p<0.05$, *** $p<0.01$.",
                "llccc", "Regime & Outcome & Transport frame & Other works & $p$ (equal)")


def main() -> None:
    br = importlib.import_module("08_robustness")
    steps = cascade()
    pd.DataFrame(steps, columns=["step", "n"]).to_csv(TABLES / "broad_cascade.csv", index=False)
    w = br.broad_sample()
    assert steps[-1][1] == len(w), (steps[-1][1], len(w))
    put_number("n broad cascade raw", steps[0][1], ",d")
    # ---- two-panel sample construction table ----
    F = pd.read_csv(TABLES / "sample_flow.csv")
    A = []
    for _, r in F.iterrows():
        s = r["step"].replace("->", "$\\rightarrow$").replace("%", "\\%").replace("[0.30, 1.00]", "$[0.30,\\,1.00]$").replace("_", "\\_")
        A.append(f"{s} & {fint(r['n'])} \\\\")
    Bp = [f"{s} & {fint(n)} \\\\" for s, n in steps]
    body = (["\\multicolumn{2}{l}{\\emph{Panel A. Broad SECOP II works universe}} \\\\"] + Bp +
            ["\\addlinespace", "\\multicolumn{2}{l}{\\emph{Panel B. Transport-infrastructure frame (sectoral case)}} \\\\"] + A)
    write_table("tab_flow", "\n".join(body), "", "tab:flow",
                "Sample construction. Panel A: all SECOP II contract records are filtered by UNSPSC class (72 building and construction services, 95 land and building structures, 811015 civil-engineering services); the universe includes the frame contracts that are in SECOP II and any other works not classified as transport infrastructure. Panel B: the frame is the stratified sampling frame of transport-infrastructure contracts compiled by Gandour Consultores (2025). "
                "SECOP~II contracts are linked to their selection process through the portfolio identifier; contracts whose process has several award rows (lots) are excluded because the lot-to-contract mapping is ambiguous.",
                "lr", "Step & Contracts", size="\\small")
    split_table(w)
    write_numbers_tex()
    print("cascade done")


if __name__ == "__main__":
    main()
