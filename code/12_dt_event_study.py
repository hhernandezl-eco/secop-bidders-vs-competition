"""STEP 12 - Standard bidding documents (DT) for transport works: staggered adoption by award regime.

The DT for transport-infrastructure works entered into force at different dates for each regime
(data/reference/dt_adoption.csv, from the official summary table of Colombia Compra Eficiente):
licitacion 2019-04-01, menor cuantia 2020-02-17, minima cuantia 2020-06-10 (and a gap with no document
in force between 2022-03-29 and 2023-02-26). A process is treated when it is published while a DT version is in force.

Estimator: two-way fixed effects (regime and publication-month fixed effects, plus the baseline controls),
standard errors clustered by entity; event study in quarters around each regime's first adoption.
Samples: broad SECOP II works universe (main) and the frame sample.
Outputs: paper/tables/tab_dt.tex, output/tables/dt_did.csv, dt_event_study.csv, figure fig_dt_event.pdf
"""
from __future__ import annotations

import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ANALYSIS_PQ, PAPER_FIGURES, REFERENCE, TABLES
from models import CONTROLS_PART, ols_cluster, prepare_part
from tex import coef_cell, fint, fnum, put_number, write_numbers_tex, write_table

LN2 = np.log(2.0)
OUTS = [("ln_bidders", "ln(bidders)"), ("rebate_pp", "Rebate"), ("add_any_pp", "Any addition"), ("net_saving_pp", "Net saving")]


def in_force(reg: pd.Series, t: pd.Series, tab: pd.DataFrame) -> pd.Series:
    out = pd.Series(0.0, index=t.index)
    for _, r in tab.iterrows():
        end = r["end"] if pd.notna(r["end"]) else pd.Timestamp("2100-01-01")
        out[(reg == r["regime"]) & (t >= r["start"]) & (t <= end)] = 1.0
    return out


def prepare(df: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t"] = pd.to_datetime(df["fecha_publicacion"])
    df = df.dropna(subset=["t"])
    df["dt"] = in_force(df["regime"], df["t"], tab)
    first = tab.groupby("regime")["start"].min()
    df["first_dt"] = df["regime"].map(first)
    df["ym"] = df["t"].dt.to_period("M").astype(str)
    df["rel_q"] = np.floor((df["t"] - df["first_dt"]).dt.days / 91.3).clip(-8, 8)
    df["year"] = df["t"].dt.year.astype(int).astype(str)
    return df


def main() -> None:
    tab = pd.read_csv(REFERENCE / "dt_adoption.csv", parse_dates=["start", "end"])
    br = importlib.import_module("08_robustness")
    w = br.broad_sample()
    w["fecha_publicacion"] = w["fecha_publicacion"]
    w["orden_c"] = w["orden"]
    w = prepare(w, tab)
    d0 = prepare_part(pd.read_parquet(ANALYSIS_PQ))
    d0["orden_c"] = d0["orden"]
    d0 = prepare(d0, tab)
    rows, body = [], []
    fe = ["regime", "ym", "unspsc4", "orden_c"]
    for label, data in [("Broad works universe", w), ("Frame sample", d0)]:
        l1, l2 = [label], [""]
        for y, _ in OUTS:
            sub = data.dropna(subset=[y])
            res = ols_cluster(sub, y, ["dt"] + CONTROLS_PART, fe)
            b, se, p = res.params["dt"], res.bse["dt"], res.pvalues["dt"]
            c1, c2 = coef_cell(b, se, p, 2 if y != "ln_bidders" else 3)
            l1.append(c1)
            l2.append(c2)
            rows.append(dict(sample=label, y=y, b=b, se=se, p=p, n=int(res.nobs), clusters=res.n_clusters))
            key = "broad" if label.startswith("Broad") else "frame"
            put_number(f"dt {key} {y}", b, ".3" if y == "ln_bidders" else ".2")
            put_number(f"dtp {key} {y}", p, ".3")
        l1.append(fint(res.nobs))
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    pd.DataFrame(rows).to_csv(TABLES / "dt_did.csv", index=False)
    header = "& ln(bidders) & Rebate (pp) & Any addition (pp) & Net saving (pp) & $N$"
    notes = ("Two-way fixed effects: coefficient on an indicator that a DT version for the regime is in force on the process publication date. Fixed effects for award regime, publication month, UNSPSC-4 class and "
             "entity order; baseline controls. Adoption dates (licitaci\\'on 1 April 2019, menor cuant\\'ia 17 February 2020, m\\'inima cuant\\'ia 10 June 2020, with no m\\'inima document in force between 29 March 2022 and 26 February 2023) "
             "are those of the official table of Colombia Compra Eficiente. Standard errors clustered by entity.")
    write_table("tab_dt", "\n".join(body), "", "tab:dt", notes, "lccccc", header)

    # event study on the broad sample
    ev = []
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.6), sharex=True)
    for ax, (y, lab) in zip(axes, OUTS):
        sub = w.dropna(subset=[y]).copy()
        ks = [k for k in range(-8, 9) if k != -1]
        for k in ks:
            sub[f"e{k}"] = (sub["rel_q"] == k).astype(float)
        res = ols_cluster(sub, y, [f"e{k}" for k in ks] + CONTROLS_PART, fe)
        b = np.array([res.params.get(f"e{k}", np.nan) for k in ks])
        se = np.array([res.bse.get(f"e{k}", np.nan) for k in ks])
        for k, bb, ss in zip(ks, b, se):
            ev.append(dict(y=y, quarter=k, b=bb, se=ss))
        ax.errorbar(ks, b, yerr=1.96 * se, fmt="o", ms=3, lw=0.8, color="#0072B2", capsize=1.5)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(-0.5, color="black", ls="--", lw=0.7)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("Quarters from first DT")
        pre = [k for k in ks if k < 0 and f"e{k}" in res.params.index]
        wt = res.wald_test(", ".join(f"e{k} = 0" for k in pre), use_f=True, scalar=True)
        put_number(f"dt pretrend p {y}", float(wt.pvalue), ".3")
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES / "fig_dt_event.pdf")
    plt.close(fig)
    pd.DataFrame(ev).to_csv(TABLES / "dt_event_study.csv", index=False)
    put_number("dt n broad", int((w.dt == 1).sum()), ",d")
    write_numbers_tex()
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
