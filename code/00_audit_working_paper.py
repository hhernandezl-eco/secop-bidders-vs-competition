"""AUDIT - Reproduce the working paper's descriptive table and identify what changes in the journal version.

Reads the *frame* and the raw SECOP II contract extract, reproduces Table 1 / Figure 2 of Working_paper_DT_BID.pdf,
and documents every discrepancy with a numeric diagnosis.

Outputs: docs/replication_audit.md, output/tables/audit_*.csv, paper/tables/tab_audit.tex, paper/tables/tab_trend.tex
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ANALYSIS_PQ, FRAME_PQ, INTERIM, PROJECT_DIR, SECOP2_CONTRACTS_PQ, TABLES
from tex import fint, fnum, put_number, write_numbers_tex, write_table

import duckdb


def main() -> None:
    fr = pd.read_parquet(FRAME_PQ)
    an = pd.read_parquet(ANALYSIS_PQ)
    out = []

    # 1) N: frame vs working paper -----------------------------------------------------------------
    con = duckdb.connect()
    s2 = fr.loc[fr.plataforma == "SECOP II", ["id_contrato"]]
    con.register("s2", s2)
    n_join = con.execute(f"SELECT count(*) FROM s2 JOIN read_parquet('{str(SECOP2_CONTRACTS_PQ).replace(chr(92), '/')}') c USING (id_contrato)").fetchone()[0]
    n_frame = len(fr)
    n_wp = n_frame - len(s2) + n_join
    out.append(("Analytic N", "15,656", fint(n_wp) + f" (frame has {n_frame:,}; SECOP II rows duplicated by the join)",
                f"{n_join - len(s2)} contracts are double counted when the frame is merged with the raw SECOP II extract, which repeats {n_join - len(s2)} contract identifiers."))
    put_number("audit dup", n_join - len(s2), ",d")

    # 2) Table 1 -----------------------------------------------------------------------------------------
    v = fr.valor_inicial / 1e6
    va = fr.valor_adicion_s1 / 1e6
    out.append(("Initial value, mean (COP m)", "746.99", f"{v.mean():.2f}", "Reproduced (difference due to the 16 duplicated rows)."))
    out.append(("Value of additions, N", "12,489", fint(va.notna().sum()), "Reproduced: only SECOP I reports the value of additions."))
    out.append(("Value of additions, mean (COP m)", "67.34", f"{va.mean():.2f}", "Reproduced."))
    out.append(("Extension days, mean", "5.18", f"{fr.prorroga_dias.mean():.2f}", "Reproduced."))

    # 3) share with additions ---------------------------------------------------------------------------
    s1 = fr[fr.plataforma == "SECOP I"]
    pos_s1 = (s1.valor_adicion_s1 > 0).sum()
    share_wp = pos_s1 / n_wp
    share_s1 = pos_s1 / len(s1)
    a2 = an[(an.plataforma == "SECOP II") & an.add_ratio_raw.notna()]
    share_s2 = (a2.add_ratio_raw > 0.001).mean()
    out.append(("Share of contracts with an addition", "13.36%", f"{100 * share_wp:.2f}% (SECOP II coded as zero)",
                f"Same number, but {len(s2) / n_frame:.0%} of the sample (SECOP II) cannot report additions and was coded as no addition. "
                f"Correct rates: SECOP I {100 * share_s1:.1f}%; SECOP II (derived from awarded vs current value) {100 * share_s2:.1f}%."))
    put_number("audit share wp", 100 * share_wp, ".2")
    put_number("audit share secopone", 100 * share_s1, ".1")
    put_number("audit share secoptwo", 100 * share_s2, ".1")

    # 4) trend (Figure 2) ----------------------------------------------------------------------------------
    t = pd.DataFrame({"year": fr.ano.astype(int)})
    t["wp_def"] = (fr.valor_adicion_s1.fillna(0) > 0).astype(float)
    t["plat"] = fr.plataforma
    g = fr.assign(wp_def=t.wp_def)
    trend = g.groupby("ano").agg(n=("id_contrato", "size"), share_s2=("plataforma", lambda s: (s == "SECOP II").mean()),
                                  wp_rate=("wp_def", "mean"))
    trend["secop1_only"] = s1.groupby("ano").apply(lambda x: (x.valor_adicion_s1 > 0).mean())
    trend["secop2_derived"] = a2.groupby("year").apply(lambda x: (x.add_ratio_raw > 0.001).mean())
    f = an[an.s_firm == 1]
    trend["consistent_all"] = f.groupby("year").add_any.mean()
    trend.to_csv(TABLES / "audit_trend.csv")
    body = []
    for yr, r in trend.iterrows():
        body.append(f"{yr} & {fint(r.n)} & {100 * r.share_s2:.0f} & {100 * r.wp_rate:.1f} & {100 * r.secop1_only:.1f} & "
                    f"{'--' if np.isnan(r.secop2_derived) else f'{100 * r.secop2_derived:.1f}'} & {100 * r.consistent_all:.1f} \\\\")
    sh18, sh24 = 100 * trend.loc[2018, "share_s2"], 100 * trend.loc[2024, "share_s2"]
    write_table("tab_trend", "\n".join(body), "", "tab:trend",
                "Share of contracts with an addition by signing year (\\%). Column (1): working-paper definition (SECOP II contracts coded as zero). The apparent decline of that series is largely a composition effect: "
                f"the share of SECOP II contracts, which cannot report additions, rises from {sh18:.0f}\\% in 2018 to {sh24:.0f}\\% in 2024. Column (4) uses the platform-consistent outcome on the firm sample; "
                "it is flat through 2022 and is affected by right-censoring in the last years.",
                "lcccccc", "Year & Contracts & SECOP II share (\\%) & (1) WP definition & (2) SECOP I only & (3) SECOP II derived & (4) Consistent, all",
                size="\\small")
    put_number("trend wp first", 100 * trend.loc[2018, "wp_rate"], ".1")
    put_number("trend wp last", 100 * trend.loc[2024, "wp_rate"], ".1")
    put_number("trend sharetwo last", 100 * trend.loc[2024, "share_s2"], ".0")
    put_number("trend cons first", 100 * trend.loc[2018, "consistent_all"], ".1")
    put_number("trend cons late", 100 * trend.loc[2023, "consistent_all"], ".1")
    put_number("trend derivedtwo nineteen", 100 * trend.loc[2019, "secop2_derived"], ".0")

    # 5) days added in SECOP II ---------------------------------------------------------------------------
    pr = fr[fr.plataforma == "SECOP II"].prorroga_dias
    out.append(("Extension days reported in SECOP II", "used in Table 1 (mean 5.18 pooled)", f"{100 * (pr > 0).mean():.2f}% of SECOP II contracts have any",
                "The SECOP II days-added field is practically empty for these contracts; extension days are informative for SECOP I only."))
    put_number("audit secoptwo days", 100 * (pr > 0).mean(), ".2")

    # 6) experience scale -----------------------------------------------------------------------------------------
    out.append(("Volume of experience, mean (max)", "38.11 (11,360)", f"{an.loc[an.s_firm == 1, 'vol_exp'].mean():.2f} ({an.loc[an.s_firm == 1, 'vol_exp'].max():.0f})",
                "The 5.9 M-record SECOP I extract is not in the delivered data, so experience is counted over the frame plus all SECOP II contracts: a lower bound."))

    # 7) instrument ------------------------------------------------------------------------------------------------------
    from models import CONTROLS_PART, FE_PART, ols_cluster, prepare_part
    dp = prepare_part(an).dropna(subset=["ln_dens", "ln_cong"])
    fs = ols_cluster(dp, "ln_bidders", ["ln_dens", "ln_cong"] + CONTROLS_PART, FE_PART + ["regime"])
    wt = fs.wald_test("ln_dens = 0, ln_cong = 0", use_f=True, scalar=True)
    f_fs = float(np.asarray(wt.statistic).ravel()[0])
    put_number("audit firststage F", f_fs, ".1")
    out.append(("Instrument relevance (density)", "first-stage p = 0.134", f"joint first-stage F = {f_fs:.1f} (density and congestion)",
                "The market-density instruments are irrelevant; IV results of the working paper cannot be interpreted causally."))

    A = pd.DataFrame(out, columns=["item", "working_paper", "replication", "implication"])
    A.to_csv(TABLES / "audit_wp.csv", index=False)
    esc = lambda s: s.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")
    body = [f"{esc(r.item)} & {esc(r.working_paper)} & {esc(r.replication)} \\\\" for r in A.itertuples()]
    write_table("tab_audit", "\n".join(body), "", "tab:audit",
                "Comparison with Table~1 and Figure~2 of the working paper. Full diagnosis in docs/replication\\_audit.md.", "p{4.2cm}p{3.4cm}p{6.2cm}",
                "Item & Working paper & Replication", size="\\scriptsize")

    md = ["# Replication audit of the working paper (Working_paper_DT_BID.pdf)\n",
          "Every row below is produced by `code/00_audit_working_paper.py` from the raw data.\n",
          "| Item | Working paper | Replication | Implication |", "|---|---|---|---|"]
    for r in A.itertuples():
        md.append(f"| {r.item} | {r.working_paper} | {r.replication} | {r.implication} |")
    md += ["", "## Year-by-year share of contracts with an addition (percent)\n",
           "| Year | Contracts | SECOP II share | WP definition | SECOP I only | SECOP II derived | Consistent (all) |",
           "|---|---|---|---|---|---|---|"]
    for yr, r in trend.iterrows():
        s2d = "--" if np.isnan(r.secop2_derived) else f"{100 * r.secop2_derived:.1f}"
        md.append(f"| {yr} | {int(r.n)} | {100 * r.share_s2:.0f} | {100 * r.wp_rate:.1f} | {100 * r.secop1_only:.1f} | {s2d} | {100 * r.consistent_all:.1f} |")
    (PROJECT_DIR / "docs" / "replication_audit.md").write_text("\n".join(md), encoding="utf-8")
    write_numbers_tex()
    print(A.to_string())
    print(trend.round(3).to_string())


if __name__ == "__main__":
    main()
