"""STEP 06 - Participation vs. competition: main results.

Question: when more firms bid, does the buyer get (i) a lower awarded price, (ii) fewer renegotiations,
(iii) a lower *final* cost?  Estimated separately for the three price-competitive award regimes
(minima cuantia = lowest price; seleccion abreviada de menor cuantia; licitacion publica) because the
mapping from bidders to price pressure depends on the award rule.

Outcomes (percentage points unless stated)
  rebate        1 - awarded value / reference budget
  add_any       contract received a value addition (> 0.1 %)  [0/100]
  add_base      (final contract value - awarded value) / reference budget
  net_saving    1 - final contract value / reference budget   = rebate - add_base   (identity)
Main regressor: ln(number of unique bidders). A coefficient b means: +b pp for a one-log-point increase;
we report effects per *doubling* of bidders (b x ln 2).

Outputs: paper/tables/tab_main.tex, tab_decomp.tex, tab_hurdle.tex, tab_enforce.tex, paper/numbers.json,
         output/tables/*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ANALYSIS_PQ, N_BOOT, SEED, TABLES
from models import CONTROLS_PART, FE_PART, ols_cluster, prepare_part
from tex import coef_cell, fint, fnum, put_number, write_numbers_tex, write_table

LN2 = np.log(2.0)
REGIMES = [("minima", "Minima cuant\\'ia (lowest price)"), ("menor", "Selecci\\'on abreviada, menor cuant\\'ia"),
           ("licitacion", "Licitaci\\'on p\\'ublica")]
SCORED = ("scored", "Scored regimes (menor cuant\\'ia + licitaci\\'on)")


def get_sub(d, rg):
    if rg == "pooled":
        return d
    if rg == "scored":
        return d[d.regime.isin(["menor", "licitacion"])]
    return d[d.regime == rg]


def fe_for(rg):
    return FE_PART + (["regime"] if rg in ("pooled", "scored") else [])


OUTCOMES = [("rebate_pp", "Rebate"), ("add_any_pp", "Any addition"), ("add_base_pp", "Additions / budget"),
            ("net_saving_pp", "Net saving")]


def main() -> None:
    df = pd.read_parquet(ANALYSIS_PQ)
    d = prepare_part(df)
    rows_csv = []

    # ---------------- Table: main pass-through by regime -------------------------------------
    cells = {}
    for rg, _ in REGIMES + [SCORED, ("pooled", "")]:
        sub = get_sub(d, rg)
        fe = fe_for(rg)
        for y, _lab in OUTCOMES:
            res = ols_cluster(sub, y, ["ln_bidders"] + CONTROLS_PART, fe)
            b, se, p = res.params["ln_bidders"], res.bse["ln_bidders"], res.pvalues["ln_bidders"]
            cells[(rg, y)] = (b, se, p, int(res.nobs), res.n_clusters, res.rsquared)
            rows_csv.append(dict(regime=rg, outcome=y, b=b, se=se, p=p, n=int(res.nobs), clusters=res.n_clusters, r2=res.rsquared))
            put_number(f"b {rg} {y}", b * LN2, ".2f")           # effect per doubling of bidders
            put_number(f"one {rg} {y}", b * LN2, ".1f")         # one-decimal version for abstract/intro
            put_number(f"se {rg} {y}", se * LN2, ".2f")
            put_number(f"p {rg} {y}", p, ".3f")
        put_number(f"n {rg}", cells[(rg, 'rebate_pp')][3], ",d")
        put_number(f"clusters {rg}", cells[(rg, 'rebate_pp')][4], ",d")

    body = []
    for rg, lab in REGIMES + [SCORED, ("pooled", "Pooled, all regimes (regime fixed effects)")]:
        sub = get_sub(d, rg)
        line_b, line_se = [f"{lab}"], [""]
        for y, _ in OUTCOMES:
            b, se, p, n, g, r2 = cells[(rg, y)]
            c1, c2 = coef_cell(b * LN2, se * LN2, p, 2)
            line_b.append(c1)
            line_se.append(c2)
        n = cells[(rg, 'rebate_pp')][3]
        body.append(" & ".join(line_b) + " \\\\")
        body.append(" & ".join(line_se) + " \\\\")
        body.append(f"\\quad \\emph{{N = {fint(n)}; clusters = {fint(cells[(rg, 'rebate_pp')][4])}; "
                    f"mean bidders = {sub['n_bidders'].mean():.1f}}} & & & & \\\\[3pt]")
    header = "& (1) Rebate & (2) Any addition & (3) Additions/budget & (4) Net saving \\\\"
    notes = ("OLS (linear probability model in column 2). Each cell is a separate regression of the outcome (in percentage points) on "
             "$\\ln(\\text{bidders})$; coefficients are scaled to the effect of \\emph{doubling} the number of bidders. "
             "All regressions include $\\ln$ reference budget, $\\ln$ planned duration (+ missing indicator) and fixed effects for signing year, "
             "UNSPSC 4-digit class and entity order (national / capital-city / other territorial); the pooled row adds regime fixed effects. "
             "Net saving $=$ rebate $-$ additions/budget by construction. Standard errors clustered by contracting entity in parentheses. "
             "* $p<0.10$, ** $p<0.05$, *** $p<0.01$.")
    write_table("tab_main", "\n".join(body), "", "tab:main", notes, "lcccc", header)

    # ---------------- Table: decomposition (rebate gained vs. eroded by additions) --------------
    rng = np.random.default_rng(SEED)
    dec = []
    for rg, lab in REGIMES + [SCORED]:
        sub = get_sub(d, rg).reset_index(drop=True)
        point = {y: cells[(rg, y)][0] * LN2 for y in ["rebate_pp", "add_base_pp", "net_saving_pp"]}
        # clustered bootstrap of the three coefficients
        groups = sub["ent_g"].unique()
        by = {g: sub.index[sub.ent_g == g].values for g in groups}
        draws = []
        for _ in range(N_BOOT):
            samp = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([by[g] for g in samp])
            bs = sub.loc[idx].reset_index(drop=True)
            bs["ent_g"] = np.repeat(np.arange(len(samp)), [len(by[g]) for g in samp])
            try:
                out = []
                for y in ["rebate_pp", "add_base_pp", "net_saving_pp"]:
                    r = ols_cluster(bs, y, ["ln_bidders"] + CONTROLS_PART, fe_for(rg))
                    out.append(r.params["ln_bidders"] * LN2)
                draws.append(out)
            except Exception:
                continue
        draws = np.array(draws)
        ci = np.percentile(draws, [2.5, 97.5], axis=0)
        diss = draws[:, 1] / draws[:, 0]
        dec.append(dict(regime=rg, label=lab, rebate=point["rebate_pp"], rebate_lo=ci[0, 0], rebate_hi=ci[1, 0],
                        add=point["add_base_pp"], add_lo=ci[0, 1], add_hi=ci[1, 1],
                        net=point["net_saving_pp"], net_lo=ci[0, 2], net_hi=ci[1, 2],
                        diss=point["add_base_pp"] / point["rebate_pp"] if point["rebate_pp"] != 0 else np.nan,
                        diss_med=np.nanmedian(diss), n=len(sub)))
        put_number(f"net lo {rg}", ci[0, 2], ".2f")
        put_number(f"net hi {rg}", ci[1, 2], ".2f")
        put_number(f"diss {rg}", 100 * dec[-1]["diss"], ".0f")
    dec = pd.DataFrame(dec)
    dec.to_csv(TABLES / "decomposition.csv", index=False)

    body = []
    for _, r in dec.iterrows():
        body.append(f"{r['label']} & {fnum(r.rebate, 2)} & [{fnum(r.rebate_lo, 2)}; {fnum(r.rebate_hi, 2)}] & "
                    f"{fnum(r['add'], 2)} & [{fnum(r['add_lo'], 2)}; {fnum(r['add_hi'], 2)}] & "
                    f"{fnum(r.net, 2)} & [{fnum(r.net_lo, 2)}; {fnum(r.net_hi, 2)}] \\\\")
    header = ("& \\multicolumn{2}{c}{Price rebate (pp)} & \\multicolumn{2}{c}{Erosion by additions (pp)} & "
              "\\multicolumn{2}{c}{Net saving (pp)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
              "& Effect & 95\\% CI & Effect & 95\\% CI & Effect & 95\\% CI")
    notes = ("Effects of doubling the number of bidders, from the specifications in \\cref{tab:main}, expressed in percentage points of the reference "
             f"budget. Net saving $=$ rebate $-$ erosion by additions. 95\\% intervals from a pairs-cluster bootstrap by contracting entity ({N_BOOT} replications).")
    write_table("tab_decomp", "\n".join(body), "", "tab:decomp", notes, "lcccccc", header)

    # ---------------- Table: hurdle model (two-part) for renegotiation -----------------------
    body, hur = [], []
    for rg, lab in REGIMES + [SCORED, ("pooled", "Pooled")]:
        sub = get_sub(d, rg)
        fe = fe_for(rg)
        from models import design
        X = design(sub, ["ln_bidders"] + CONTROLS_PART, fe)
        y1 = sub["add_any"].astype(float)
        ok = X.notna().all(axis=1) & y1.notna()
        try:
            pr = sm.Probit(y1[ok], X[ok]).fit(disp=0, cov_type="cluster",
                                              cov_kwds={"groups": pd.factorize(sub.loc[ok, "ent_g"])[0]}, maxiter=200)
            me = pr.get_margeff(at="overall")
            idx = list(X.columns).index("ln_bidders") - 1
            ame, ame_se, ame_p = me.margeff[idx] * LN2 * 100, me.margeff_se[idx] * LN2 * 100, me.pvalues[idx]
        except Exception:
            ame, ame_se, ame_p = np.nan, np.nan, np.nan
        pos = sub[(sub.add_any == 1) & (sub.add_ratio > 0)].copy()
        pos["ln_add_ratio"] = np.log(pos["add_ratio"])
        r2 = ols_cluster(pos, "ln_add_ratio", ["ln_bidders"] + CONTROLS_PART, fe)
        b2, se2, p2 = r2.params["ln_bidders"], r2.bse["ln_bidders"], r2.pvalues["ln_bidders"]
        hur.append(dict(regime=rg, ame_pp=ame, ame_se=ame_se, ame_p=ame_p, n1=int(ok.sum()), b2=b2 * LN2 * 100, se2=se2 * LN2 * 100,
                        p2=p2, n2=int(r2.nobs)))
        c1, c2 = coef_cell(ame, ame_se, ame_p, 2)
        c3, c4 = coef_cell(b2 * LN2 * 100, se2 * LN2 * 100, p2, 1)
        body.append(f"{lab} & {c1} & {c3} \\\\")
        body.append(f" & {c2} & {c4} \\\\")
        body.append(f"\\quad \\emph{{N = {fint(ok.sum())} / {fint(r2.nobs)}}} & & \\\\[3pt]")
        put_number(f"hurdle ame {rg}", ame, ".1f")
        put_number(f"hurdle p {rg}", ame_p, ".3f")
    pd.DataFrame(hur).to_csv(TABLES / "hurdle.csv", index=False)
    header = "& (1) Part 1: Pr(addition) & (2) Part 2: size of addition \\\\\n& average marginal effect, pp & \\% change, $\\ln$(addition/value)"
    notes = ("Two-part (hurdle) model, as in \\citet{cragg1971}. Part 1 is a probit for any addition; entries are average marginal effects (percentage points) of "
             "doubling the number of bidders. Part 2 is an OLS regression of $\\ln$(addition ratio) on the same covariates among contracts with an addition; entries are "
             "percent changes in the addition ratio from doubling bidders. Same controls as \\cref{tab:main}. Standard errors clustered by entity.")
    write_table("tab_hurdle", "\n".join(body), "", "tab:hurdle", notes, "lcc", header)

    # ---------------- Table: enforcement heterogeneity ---------------------------------------
    dd = d.dropna(subset=["ent_soft"]).copy()
    dd["soft_z"] = (dd["ent_soft"] - dd["ent_soft"].mean()) / dd["ent_soft"].std()
    dd["lnb_soft"] = dd["ln_bidders"] * dd["soft_z"]
    put_number("n enforce", len(dd), ",d")
    put_number("enf mean", dd["ent_soft"].mean() * 100, ".0f")
    put_number("enf sd", dd["ent_soft"].std() * 100, ".0f")
    body = []
    for rg, lab in REGIMES + [SCORED, ("pooled", "Pooled")]:
        sub = get_sub(dd, rg)
        fe = fe_for(rg)
        l1, l2, l3, l4 = [f"{lab}: $\\ln$(bidders)"], [""], [f"\\quad $\\times$ entity prior-addition rate (z)"], [""]
        for y, _ in [("rebate_pp", ""), ("add_any_pp", ""), ("net_saving_pp", "")]:
            res = ols_cluster(sub, y, ["ln_bidders", "soft_z", "lnb_soft"] + CONTROLS_PART, fe)
            for nm, la, lb in [("ln_bidders", l1, l2), ("lnb_soft", l3, l4)]:
                c1, c2 = coef_cell(res.params[nm] * LN2 if nm == "ln_bidders" else res.params[nm] * LN2,
                                   res.bse[nm] * LN2, res.pvalues[nm], 2)
                la.append(c1)
                lb.append(c2)
            put_number(f"enf int {rg} {y}", res.params["lnb_soft"] * LN2, ".2f")
            put_number(f"enf intp {rg} {y}", res.pvalues["lnb_soft"], ".3f")
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\", " & ".join(l3) + " \\\\", " & ".join(l4) + " \\\\"]
        body.append(f"\\quad \\emph{{N = {fint(len(sub))}}} & & & \\\\[3pt]")
    header = "& (1) Rebate & (2) Any addition & (3) Net saving"
    notes = ("The moderator is the share of the contracting entity's earlier \\emph{completed} works contracts that received an addition "
             "(requires at least five prior contracts), standardised to mean zero and unit variance; it uses only information available "
             "before the focal contract. Coefficients are scaled to the effect of doubling the number of bidders (pp). Same controls as "
             "\\cref{tab:main}. Standard errors clustered by entity. * $p<0.10$, ** $p<0.05$, *** $p<0.01$.")
    write_table("tab_enforce", "\n".join(body), "", "tab:enforce", notes, "lccc", header)

    # ---- non-linear check of the interaction (Ai & Norton): probit, average effect of doubling bidders at +-1 SD of the moderator ----
    from models import design
    for rg in ["scored", "pooled"]:
        sub = get_sub(dd, rg)
        X = design(sub, ["ln_bidders", "soft_z", "lnb_soft"] + CONTROLS_PART, fe_for(rg))
        y1 = sub["add_any"].astype(float)
        pr = sm.Probit(y1, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["ent_g"])[0]}, maxiter=200)
        put_number(f"probit int {rg}", pr.params["lnb_soft"], ".2")
        put_number(f"probit intp {rg}", pr.pvalues["lnb_soft"], ".3")
        for zlab, z in [("low", -1.0), ("high", 1.0)]:
            X0, X1 = X.copy(), X.copy()
            X0["soft_z"], X1["soft_z"] = z, z
            X0["lnb_soft"] = X0["ln_bidders"] * z
            X1["ln_bidders"] = X0["ln_bidders"] + LN2
            X1["lnb_soft"] = X1["ln_bidders"] * z
            put_number(f"probit ame {rg} {zlab}", 100 * float((pr.predict(X1) - pr.predict(X0)).mean()), ".1")
        put_number(f"base addrate {rg}", 100 * float(y1.mean()), ".0")
    hi_soft = dd.loc[dd["soft_z"] > 1, "add_any"].mean() * 100
    put_number("addrate high soft", hi_soft, ".0")
    put_number("addrate low soft", dd.loc[dd["soft_z"] < -1, "add_any"].mean() * 100, ".0")
    pd.DataFrame(rows_csv).to_csv(TABLES / "main_participation.csv", index=False)
    write_numbers_tex()
    print(pd.DataFrame(rows_csv).round(4).to_string())
    print(dec.round(3).to_string())


if __name__ == "__main__":
    main()
