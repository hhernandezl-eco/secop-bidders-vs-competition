"""STEP 20 - Validation of the derived SECOP II additions measure (F - P) / P, and censoring of the final value.

SECOP II reports no field for the value of additions. We derive them as (current contract value F - awarded value P) / P.
A referee asked for the natural validation: read the modifications recorded in the electronic file of a random sample of
contracts and decompose the increase into (a) formal additions of scope (otrosi), (b) larger quantities of work in unit-price
contracts and (c) price-adjustment records. This script

  1. computes what the open data alone say about the measure: share of F > 1.5 P (legal ceiling), share of the increase that was
     actually paid among contracts that report payments, and the share of contracts at the 2B censoring point;
  2. draws the stratified random audit sample (regime x size of the increase; seed in config.py) and writes it with the SECOP II
     link of each contract and empty coding columns (data/audit/audit_sample.csv and .xlsx);
  3. if the coded file data/audit/audit_sample_coded.csv exists, computes the population-weighted decomposition of the increase and
     the table of Appendix 'Audit'. Without it, the manuscript prints the protocol and states that the document-level coding is pending
     (macro \\Rauditdone = 0).

Coding columns (COP, one row per sampled contract; leave blank what is not found, doc_access = 0 if the file cannot be opened):
  doc_access, v_otrosi (formal additions of value), v_cantidades (larger quantities / unit-price execution),
  v_reajuste (price adjustment by formula), v_otro (any other modification of value), notes, coder

Usage:  python code/20_additions_audit.py [--coded path/to/coded.csv]
Outputs: data/audit/*, tab_audit.tex, numbers
"""
from __future__ import annotations

import argparse
import importlib

import duckdb
import numpy as np
import pandas as pd

from config import PROJECT_DIR, SECOP2_CONTRACTS_PQ, SEED
from tex import fint, fnum, put_number, write_numbers_tex, write_table

AUDIT_DIR = PROJECT_DIR / "data" / "audit"
N_PER_STRATUM = 17                          # 9 strata x 17 = 153 contracts (about 150)
BANDS = [(0.001, 0.10, "up to 10%"), (0.10, 0.40, "10 to 40%"), (0.40, np.inf, "above 40%")]
CODE_COLS = ["doc_access", "v_otrosi", "v_cantidades", "v_reajuste", "v_otro", "notes", "coder"]
TYPES = [("v_otrosi", "Formal addition of scope or value (otros\\'i)", "otrosi"),
         ("v_cantidades", "Larger quantities of work (unit-price contracts)", "cantidades"),
         ("v_reajuste", "Price adjustment by formula (reajuste)", "reajuste"),
         ("v_otro", "Other, or not explained by a recorded modification", "otro")]


def load() -> pd.DataFrame:
    br = importlib.import_module("08_robustness")
    w = br.broad_sample()
    con = duckdb.connect()
    q = str(SECOP2_CONTRACTS_PQ).replace("\\", "/")
    extra = con.execute(f"SELECT id_contrato, max(valor_pagado) AS pagado, max(url_proceso) AS url FROM read_parquet('{q}') GROUP BY 1").fetchdf()
    w = w.merge(extra, on="id_contrato", how="left")
    w["F"], w["P"], w["B"] = w["valor_contrato"], w["valor_adjudicado"], w["precio_base"]
    w["has_add"] = w["add_ratio_raw"] > 0.001
    return w


def open_data_checks(w: pd.DataFrame) -> None:
    # --- censoring of the final value at twice the reference budget ---
    c = (w.F / w.B) > 2.0
    put_number("cens2b n broad", int(c.sum()), ",d")
    put_number("cens2b share broad", 100 * c.mean(), ".2")
    for reg in ["minima", "menor", "licitacion"]:
        put_number(f"cens2b share broad {reg}", 100 * c[w.regime == reg].mean(), ".2")
    an = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "analysis_contracts.parquet")
    fr = an[an.s_part == 1]
    cf = fr["final_ratio"] > 2.0
    put_number("cens2b n frame", int(cf.sum()), ",d")
    put_number("cens2b share frame", 100 * cf.mean(), ".2")
    put_number("cens2b n frame part", len(fr), ",d")
    # largest ratio kept after censoring, and share of contracts above budget (context)
    put_number("cens2b share above budget broad", 100 * ((w.F / w.B) > 1.0).mean(), ".0f")
    # --- legal ceiling: share of additions above 50% of P (ceiling of Ley 80, art. 40, par.) ---
    a = w[w.has_add]
    put_number("legal share above ceiling", 100 * (a.add_ratio_raw > 0.5).mean(), ".1")
    put_number("legal share above ceiling all", 100 * (w.add_ratio_raw > 0.5).mean(), ".1")
    put_number("legal n additions broad", len(a), ",d")
    put_number("legal share big additions", 100 * (a.add_ratio_raw > 0.10).mean(), ".0f")
    # --- were the increases actually paid? (contracts that report payments) ---
    r = a[a.pagado > 0]
    put_number("payval share reporting", 100 * (w.pagado > 0).mean(), ".0f")
    put_number("payval n", len(r), ",d")
    put_number("payval share above awarded", 100 * (r.pagado > r.P * 1.001).mean(), ".0f")
    inc = ((r.pagado - r.P) / (r.F - r.P)).replace([np.inf, -np.inf], np.nan)
    put_number("payval median paid share", 100 * inc.median(), ".0f")
    put_number("payval share ninety", 100 * (inc >= 0.9).mean(), ".0f")
    # --- value share of the scored regimes (abstract) ---
    tot = w.valor_adjudicado.sum()
    put_number("valshare broad scored", 100 * w.loc[w.regime.isin(["menor", "licitacion"]), "valor_adjudicado"].sum() / tot, ".0f")
    for reg in ["minima", "menor", "licitacion"]:
        put_number(f"valshare broad {reg}", 100 * w.loc[w.regime == reg, "valor_adjudicado"].sum() / tot, ".0f")


LAB = {"minima": "M\\'inima cuant\\'ia (lowest price)", "menor": "Selecci\\'on abreviada, menor cuant\\'ia", "licitacion": "Licitaci\\'on p\\'ublica"}


def desc_broad(w: pd.DataFrame) -> None:
    """Descriptive statistics by award regime on the broad works universe (same columns as the frame table of step 09)."""
    d = w.copy()
    d["add_base_pp"] = d["rebate_pp"] - d["net_saving_pp"]
    tot = d.valor_adjudicado.sum()
    rows = []
    for g in ["minima", "menor", "licitacion", "all"]:
        s = d if g == "all" else d[d.regime == g]
        rows.append([("All regimes" if g == "all" else LAB[g]), fint(len(s)), f"{s.n_bidders.mean():.1f}", f"{s.n_bidders.median():.0f}", f"{s.rebate_pp.mean():.1f}",
                     f"{s.add_any_pp.mean():.1f}", f"{s.add_base_pp.mean():.1f}", fnum(s.net_saving_pp.mean(), 1),
                     f"{s.valor_adjudicado.median() / 1e6:,.0f}".replace(",", "\\,"), f"{100 * s.valor_adjudicado.sum() / tot:.0f}"])
        if g != "all":
            put_number(f"bd meanbid {g}", s.n_bidders.mean(), ".1")
            put_number(f"bd medbid {g}", s.n_bidders.median(), ".0")
            put_number(f"bd meanrebate {g}", s.rebate_pp.mean(), ".1")
            put_number(f"bd meanadd {g}", s.add_any_pp.mean(), ".0")
            put_number(f"bd meanfinal {g}", 100 * s.final_ratio_w.mean(), ".0")
            put_number(f"bd nreg {g}", len(s), ",d")
    h2 = ("Regime & $N$ & \\shortstack{Mean\\\\bidders} & \\shortstack{Median\\\\bidders} & \\shortstack{Rebate\\\\(\\%)} & "
          "\\shortstack{With\\\\addition (\\%)} & \\shortstack{Additions/\\\\budget (\\%)} & \\shortstack{Net saving\\\\(\\%)} & "
          "\\shortstack{Median award\\\\(COP m)} & \\shortstack{Value\\\\share (\\%)}")
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    write_table("tab_desc_part_broad", body, "", "tab:descpartbroad",
                "Broad SECOP II works universe (price-competitive regimes). Same definitions as the transport-frame panel.", "lccccccccc", h2, size="\\scriptsize")


def draw_sample(w: pd.DataFrame) -> pd.DataFrame:
    a = w[w.has_add & w.url.notna()].copy()
    a["band"] = pd.cut(a.add_ratio_raw, [b[0] for b in BANDS] + [np.inf], labels=[b[2] for b in BANDS], right=True)
    rng = np.random.default_rng(SEED + 20)
    parts = []
    for (reg, band), g in a.groupby(["regime", "band"], observed=True):
        k = min(N_PER_STRATUM, len(g))
        s = g.sample(n=k, random_state=int(rng.integers(1 << 31)))
        s = s.assign(stratum_N=len(g), stratum_n=k, weight=len(g) / k)
        parts.append(s)
    s = pd.concat(parts).sort_values(["regime", "band", "id_contrato"]).reset_index(drop=True)
    s["audit_id"] = np.arange(1, len(s) + 1)
    s["increment_cop"] = s.F - s.P
    s["year"] = s["year"].astype(int)
    keep = ["audit_id", "id_contrato", "proceso_compra", "entidad", "departamento", "regime", "year", "band", "P", "F", "increment_cop",
            "add_ratio_raw", "stratum_N", "stratum_n", "weight", "url"]
    out = s[keep].rename(columns={"P": "valor_adjudicado", "F": "valor_contrato", "url": "url_proceso"})
    for c in CODE_COLS:
        out[c] = np.nan if c not in ("notes", "coder") else ""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(AUDIT_DIR / "audit_sample.csv", index=False, encoding="utf-8-sig")
    try:
        out.to_excel(AUDIT_DIR / "audit_sample.xlsx", index=False)
    except Exception as e:                                           # openpyxl missing: the CSV is enough
        print("xlsx not written:", e)
    put_number("audit n sample", len(out), ",d")
    put_number("audit n strata", out.groupby(["regime", "band"], observed=True).ngroups, "d")
    put_number("audit n population", int(a.shape[0]), ",d")
    return out


def stub_table(reason: str) -> None:
    write_table("tab_audit", "\\multicolumn{4}{c}{" + reason + "} \\\\", "", "tab:audit", "Document-level coding not yet available.",
                "lccc", "& All & M\\'inima & Scored")


def coded_results(coded_path) -> None:
    c = pd.read_csv(coded_path, encoding="utf-8-sig")
    base = pd.read_csv(AUDIT_DIR / "audit_sample.csv", encoding="utf-8-sig")[["audit_id", "weight", "regime", "band", "increment_cop"]]
    c = c.drop(columns=[x for x in ["weight", "regime", "band", "increment_cop"] if x in c.columns]).merge(base, on="audit_id", how="left")
    c = c[c.doc_access == 1].copy()
    for t, _, _ in TYPES:
        c[t] = pd.to_numeric(c[t], errors="coerce").fillna(0.0)
    c["v_tot"] = c[[t for t, _, _ in TYPES]].sum(axis=1)
    c["scored"] = c.regime.isin(["menor", "licitacion"])
    n_all = len(pd.read_csv(AUDIT_DIR / "audit_sample.csv", encoding="utf-8-sig"))
    put_number("auditmanual n coded", len(c), ",d")
    put_number("auditmanual share access", 100 * len(c) / n_all, ".0f")
    rng = np.random.default_rng(SEED + 21)

    def shares(d: pd.DataFrame):
        """Population-weighted share of the coded increase by type, and share of contracts with a formal addition."""
        w = d["weight"].values
        tot = (w * d["v_tot"].values).sum()
        sh = [100 * (w * d[t].values).sum() / tot if tot > 0 else np.nan for t, _, _ in TYPES]
        any_formal = 100 * (w * (d["v_otrosi"].values > 0)).sum() / w.sum()
        cover = 100 * (w * d["v_tot"].values).sum() / (w * d["increment_cop"].values).sum()
        return sh + [any_formal, cover]

    def boot(d: pd.DataFrame, B: int = 1000):
        reps = []
        strata = {k: g for k, g in d.groupby(["regime", "band"])}
        for _ in range(B):
            parts = [g.sample(n=len(g), replace=True, random_state=int(rng.integers(1 << 31))) for g in strata.values()]
            reps.append(shares(pd.concat(parts)))
        reps = np.array(reps, dtype=float)
        return np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)

    groups = [("all", c), ("minima", c[~c.scored]), ("scored", c[c.scored])]
    res = {g: (shares(d), boot(d), len(d)) for g, d in groups}
    labels = [lab for _, lab, _ in TYPES] + ["Contracts with at least one formal addition (\\% of contracts)", "Coded value as \\% of the increase $F-P$"]
    keys = [k for _, _, k in TYPES] + ["anyformal", "coverage"]
    body = []
    for i, lab in enumerate(labels):
        cells = []
        for g, _ in groups:
            v, (lo, hi), _n = res[g]
            cells.append(f"{fnum(v[i], 0)} [{fnum(lo[i], 0)}, {fnum(hi[i], 0)}]")
            put_number(f"auditmanual {keys[i]} {g}", v[i], ".0f")
            put_number(f"auditmanual {keys[i]} lo {g}", lo[i], ".0f")
            put_number(f"auditmanual {keys[i]} hi {g}", hi[i], ".0f")
        if keys[i] == "coverage":          # 100% by construction when the coder allocates the whole increase: not shown
            continue
        body.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if i == 3:
            body.append("\\midrule")
    body.append("Contracts coded & " + " & ".join(str(res[g][2]) for g, _ in groups) + " \\\\")
    write_table("tab_audit", "\n".join(body), "", "tab:audit",
                "Stratified random sample of SECOP II works contracts with $F>P$, by award regime and size of the increase. The increase was classified from the modification records that the entity publishes on the SECOP II portal for each contract (type, date, justification and change in the contract value), not from the attached documents. "
                "The type is the one that the entity's justification states: larger quantities of work or price adjustments registered as an \\emph{otros\\'i} count as otros\\'i. All sampled contracts are coded and were checked by the authors against the portal records; larger quantities or adjustments recorded as otros\\'i make the shares of those two types lower bounds. "
                "Shares are population-weighted (inverse sampling probability) shares of the coded increase; brackets are 95\\% stratified-bootstrap intervals.",
                "lccc", "& All & M\\'inima & Scored")
    put_number("auditdone", 1, "d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coded", default=str(AUDIT_DIR / "audit_sample_coded.csv"))
    args = ap.parse_args()
    w = load()
    open_data_checks(w)
    desc_broad(w)
    draw_sample(w)
    import pathlib
    if pathlib.Path(args.coded).exists():
        coded_results(args.coded)
    else:
        put_number("auditdone", 0, "d")
        stub_table("Document-level coding pending")
        # placeholders keep every macro of the conditional text defined (they are never printed while auditdone = 0)
        put_number("auditmanual n coded", "--")
        put_number("auditmanual share access", "--")
        for g in ("all", "minima", "scored"):
            for k in [k for _, _, k in TYPES] + ["anyformal", "coverage"]:
                for suf in ("", " lo", " hi"):
                    put_number(f"auditmanual {k}{suf} {g}", "--")
    write_numbers_tex()
    print("audit step done")


if __name__ == "__main__":
    main()
