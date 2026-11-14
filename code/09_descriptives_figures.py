"""STEP 09 - Descriptive tables and figures.

Figures (PDF, vector) -> paper/figures ; tables -> paper/tables ; numbers -> paper/numbers.json
  fig_cap          additions bunch at the legal ceiling (+50 %)
  fig_bidders      distribution of the number of bidders by award regime
  fig_binned       outcomes against participation (binned means) by regime
  fig_decomp       what doubling participation buys: rebate, erosion by additions, net saving
  fig_auc          out-of-sample discrimination of bidder capabilities
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ANALYSIS_PQ, PAPER_FIGURES, TABLES
from models import prepare_part
from tex import fint, fnum, put_number, write_numbers_tex, write_table

COL = {"minima": "#0072B2", "menor": "#E69F00", "licitacion": "#009E73", "scored": "#CC79A7"}
LAB = {"minima": "M\u00ednima cuant\u00eda (lowest price)", "menor": "Selecci\u00f3n abreviada, menor cuant\u00eda",
       "licitacion": "Licitaci\u00f3n p\u00fablica"}
LAB_EN = {"minima": "Minima cuant\u00eda\n(lowest price)", "menor": "Menor cuant\u00eda\n(scored)", "licitacion": "Licitaci\u00f3n\np\u00fablica"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "axes.labelsize": 9, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans"})


def fig_cap(df: pd.DataFrame) -> None:
    f = df[df.s_firm == 1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, plat, col in zip(axes, ["SECOP I", "SECOP II"], ["#56B4E9", "#D55E00"]):
        r = f.loc[(f.plataforma == plat) & (f.add_ratio_raw > 0.001) & (f.add_ratio_raw <= 1.0), "add_ratio_raw"]
        bins = np.arange(0, 1.0001, 0.025)
        ax.hist(r, bins=bins, color=col, edgecolor="white", linewidth=0.4)
        ax.axvline(0.5, color="black", ls="--", lw=0.9)
        ax.set_title(f"{plat}  (N = {len(r):,} contracts with additions)")
        ax.set_xlabel("Addition / initial contract value")
        ax.set_xlim(0, 1)
        share = ((r > 0.475) & (r <= 0.5001)).mean()
        ax.text(0.52, ax.get_ylim()[1] * 0.85, f"{share:.0%} of additions\nsit at 47.5\u201350%", fontsize=8)
        put_number(f"cap share {plat.replace(' ', '')}", 100 * share, ".0f")
    axes[0].set_ylabel("Number of contracts")
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES / "fig_cap.pdf")
    plt.close(fig)


def fig_bidders(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    for g in ["minima", "menor", "licitacion"]:
        x = np.sort(d.loc[d.regime == g, "n_bidders"].values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COL[g], lw=1.8,
                label=f"{LAB[g]} (median {np.median(x):.0f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of unique bidders (log scale)")
    ax.set_ylabel("Cumulative share of contracts")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=1)
    fig.savefig(PAPER_FIGURES / "fig_bidders.pdf")
    plt.close(fig)


def binned(sub: pd.DataFrame, y: str, nb: int = 7):
    q = pd.qcut(sub["n_bidders"], nb, duplicates="drop")
    g = sub.groupby(q, observed=True)
    xm = g["n_bidders"].median()
    ym = g[y].mean()
    se = g[y].std() / np.sqrt(g[y].count())
    return xm.values, ym.values, 1.96 * se.values


def fig_binned(d: pd.DataFrame) -> None:
    outs = [("rebate_pp", "Price rebate (% of budget)"), ("add_any_pp", "Contracts with an addition (%)"),
            ("net_saving_pp", "Net saving (% of budget)")]
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.4), sharex="col")
    for j, g in enumerate(["minima", "menor", "licitacion"]):
        sub = d[d.regime == g]
        for i, (y, lab) in enumerate(outs):
            ax = axes[i, j]
            x, ym, e = binned(sub, y)
            ax.errorbar(x, ym, yerr=e, fmt="o-", color=COL[g], ms=3.5, lw=1.2, capsize=2)
            ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(LAB[g], fontsize=8)
            if i == 2:
                ax.set_xlabel("Bidders (log scale)")
            if y == "net_saving_pp":
                ax.axhline(0, color="grey", lw=0.6, ls=":")
        for i in range(3):
            ylim = [a.get_ylim() for a in axes[i]]
    for i in range(3):
        lo = min(a.get_ylim()[0] for a in axes[i])
        hi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES / "fig_binned.pdf")
    plt.close(fig)


def fig_decomp() -> None:
    D = pd.read_csv(TABLES / "decomposition.csv")
    order = ["minima", "menor", "licitacion", "scored"]
    D = D.set_index("regime").loc[[o for o in order if o in D.regime.values]]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    w = 0.26
    xs = np.arange(len(D))
    for k, (col, lab, c) in enumerate([("rebate", "Price rebate", "#0072B2"), ("add", "Erosion by additions", "#D55E00"),
                                         ("net", "Net saving", "#222222")]):
        y = D[col].values
        lo, hi = D[col + "_lo"].values, D[col + "_hi"].values
        ax.bar(xs + (k - 1) * w, y, w, color=c, label=lab, alpha=0.9)
        ax.errorbar(xs + (k - 1) * w, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt="none", ecolor="black", lw=0.8, capsize=2)
    ax.axhline(0, color="black", lw=0.6)
    names = {"minima": "M\u00ednima cuant\u00eda\n(lowest price)", "menor": "Menor cuant\u00eda", "licitacion": "Licitaci\u00f3n\np\u00fablica",
             "scored": "Scored regimes\n(menor + licitaci\u00f3n)"}
    ax.set_xticks(xs)
    ax.set_xticklabels([names[r] for r in D.index])
    ax.set_ylabel("Effect of doubling bidders\n(pp of reference budget)")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(PAPER_FIGURES / "fig_decomp.pdf")
    plt.close(fig)


def fig_decomp2() -> None:
    """Two panels: broad works universe (primary) and transport frame (focal case)."""
    panels = [("A. Broad works universe", "decomposition_broad.csv"), ("B. Transport frame", "decomposition.csv")]
    order = ["minima", "menor", "licitacion", "scored"]
    names = {"minima": "Mínima\ncuantía", "menor": "Menor\ncuantía", "licitacion": "Licitación\npública", "scored": "Scored\n(menor + lic.)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    w = 0.26
    for ax, (title, fname) in zip(axes, panels):
        D = pd.read_csv(TABLES / fname).set_index("regime").loc[order]
        xs = np.arange(len(D))
        for k, (col, lab, c) in enumerate([("rebate", "Price rebate", "#0072B2"), ("add", "Erosion by additions", "#D55E00"), ("net", "Net saving", "#222222")]):
            y = D[col].values
            lo, hi = D[col + "_lo"].values, D[col + "_hi"].values
            ax.bar(xs + (k - 1) * w, y, w, color=c, label=lab, alpha=0.9)
            ax.errorbar(xs + (k - 1) * w, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt="none", ecolor="black", lw=0.7, capsize=1.5)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([names[r] for r in D.index], fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_ylabel("Effect of doubling bidders\n(pp of reference budget)")
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES / "fig_decomp.pdf")
    plt.close(fig)


def fig_auc() -> None:
    A = pd.read_csv(TABLES / "firms_auc.csv")
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    y = np.arange(len(A))[::-1]
    ax.errorbar(A.auc, y, xerr=[A.auc - A.lo, A.hi - A.auc], fmt="o", color="#0072B2", capsize=3, ms=5)
    ax.axvline(A.auc.iloc[0], color="grey", ls=":", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(A.model)
    ax.set_xlabel("Out-of-sample AUC (2021\u20132022 contracts)")
    fig.savefig(PAPER_FIGURES / "fig_auc.pdf")
    plt.close(fig)


def table_desc(df: pd.DataFrame, d: pd.DataFrame) -> None:
    f = df[df.s_firm == 1]
    rows = []
    for plat in ["SECOP I", "SECOP II"]:
        s = f[f.plataforma == plat]
        pos = s[s.add_ratio_raw > 0.001]
        rows.append([plat, fint(len(s)), f"{100 * s.add_any.mean():.1f}", f"{100 * pos.add_ratio.mean():.1f}",
                     f"{100 * s.add_cap.mean():.1f}", f"{s.valor_ini.median() / 1e6:,.0f}".replace(",", "\\,"),
                     f"{(s.tipo_contrato == 'OBRA').mean() * 100:.0f}"])
        put_number(f"addshare {plat.replace(' ', '')}", 100 * s.add_any.mean(), ".1")
        put_number(f"addmean {plat.replace(' ', '')}", 100 * pos.add_ratio.mean(), ".0")
    body = [" & ".join(r) + " \\\\" for r in rows]
    h1 = ("Platform & Contracts & With addition (\\%) & Mean addition if any (\\% of value) & At the 50\\% ceiling (\\%) & "
          "Median initial value (COP m) & Works contracts (\\%)")
    write_table("tab_desc_firms", "\n".join(body), "", "tab:desc",
                "Firm sample (\\cref{tab:flow}). ``With addition'' = addition above 0.1\\% of the initial value. ``At the 50\\% ceiling'' = addition between 49\\% and 50\\% of the initial value. "
                "SECOP~I values come from the platform's reported total of additions; SECOP~II values are derived as the change between the awarded value and the current contract value.",
                "lcccccc", h1, size="\\scriptsize")

    rows = []
    tot_val = d.valor_adjudicado.sum()
    for g in ["minima", "menor", "licitacion"]:
        s = d[d.regime == g]
        rows.append([LAB[g].replace("\u00ed", "\\'i").replace("\u00f3", "\\'o").replace("\u00fa", "\\'u"), fint(len(s)), f"{s.n_bidders.mean():.1f}", f"{s.n_bidders.median():.0f}",
                     f"{s.rebate_pp.mean():.1f}", f"{s.add_any_pp.mean():.1f}", f"{s.add_base_pp.mean():.1f}",
                     f"{s.net_saving_pp.mean():.1f}", f"{s.valor_adjudicado.median() / 1e6:,.0f}".replace(",", "\\,"),
                     f"{100 * s.valor_adjudicado.sum() / tot_val:.0f}"])
        put_number(f"valshare {g}", 100 * s.valor_adjudicado.sum() / tot_val, ".0f")
        put_number(f"meanbid {g}", s.n_bidders.mean(), ".1")
        put_number(f"medbid {g}", s.n_bidders.median(), ".0")
        put_number(f"meanrebate {g}", s.rebate_pp.mean(), ".1")
        put_number(f"meanadd {g}", s.add_any_pp.mean(), ".0")
        put_number(f"meanfinal {g}", 100 * s.final_ratio_w.mean(), ".0")
        put_number(f"nreg {g}", len(s), ",d")
    s = d
    rows.append(["All regimes", fint(len(s)), f"{s.n_bidders.mean():.1f}", f"{s.n_bidders.median():.0f}", f"{s.rebate_pp.mean():.1f}",
                 f"{s.add_any_pp.mean():.1f}", f"{s.add_base_pp.mean():.1f}", f"{s.net_saving_pp.mean():.1f}",
                 f"{s.valor_adjudicado.median() / 1e6:,.0f}".replace(",", "\\,"), "100"])
    h2 = ("Regime & $N$ & \\shortstack{Mean\\\\bidders} & \\shortstack{Median\\\\bidders} & \\shortstack{Rebate\\\\(\\%)} & "
          "\\shortstack{With\\\\addition (\\%)} & \\shortstack{Additions/\\\\budget (\\%)} & \\shortstack{Net saving\\\\(\\%)} & "
          "\\shortstack{Median award\\\\(COP m)} & \\shortstack{Value\\\\share (\\%)}")
    body = [" & ".join(r) + " \\\\" for r in rows]
    write_table("tab_desc_part", "\n".join(body), "", "tab:descpart",
                "Participation sample (SECOP II works contracts, price-competitive regimes). Rebate $=$ 1 $-$ awarded value / reference budget; additions/budget $=$ (current contract value $-$ awarded value) / reference "
                "budget; net saving $=$ 1 $-$ current contract value / reference budget (capped at $-100\\%$).", "lccccccccc", h2, size="\\scriptsize")


def table_flow() -> None:
    F = pd.read_csv(TABLES / "sample_flow.csv")
    body = []
    for _, r in F.iterrows():
        s = r["step"].replace("->", "$\\rightarrow$").replace("%", "\\%").replace("[0.30, 1.00]", "$[0.30,\\,1.00]$").replace("_", "\\_")
        body.append(f"{s} & {fint(r['n'])} \\\\")
        if r["step"].strip().startswith("winner identified"):
            put_number("n firm sample", r["n"], ",d")
        if "participation sample" in r["step"]:
            put_number("n part sample", r["n"], ",d")
        if r["step"].startswith("Contracts in the sampling frame"):
            put_number("n frame", r["n"], ",d")
    write_table("tab_flow_frame", "\n".join(body), "", "tab:flowframe",
                "Sample construction. The frame is the stratified sampling frame of transport-infrastructure contracts compiled by Gandour Consultores (2025). "
                "SECOP~II contracts are linked to their selection process through the portfolio identifier; contracts whose process has several award rows (lots) are excluded because the lot-to-contract mapping is ambiguous.",
                "lr", "Step & Contracts", size="\\small")


def main() -> None:
    df = pd.read_parquet(ANALYSIS_PQ)
    d = prepare_part(df)
    put_number("n frame secopone", int((df.plataforma == "SECOP I").sum()), ",d")
    put_number("n frame secoptwo", int((df.plataforma == "SECOP II").sum()), ",d")
    fig_cap(df)
    fig_bidders(d)
    fig_binned(d)
    table_desc(df, d)
    table_flow()
    if (TABLES / "decomposition.csv").exists():
        fig_decomp2()
    if (TABLES / "firms_auc.csv").exists():
        fig_auc()
    write_numbers_tex()
    print("figures and descriptive tables written")


if __name__ == "__main__":
    main()
