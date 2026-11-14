"""STEP 13 - Broad SECOP II works universe as a co-primary sample: mechanisms.

Repeats, on all closed single-award SECOP II works contracts signed 2018-2024 (~14,000 contracts, ~1,000 entities),
(a) the pass-through by regime, (b) the two-part model of renegotiation and (c) the entity-commitment interaction,
the latter with the entity's *ex-ante* addition rate computed from the same universe (earlier completed works contracts).

Outputs: tab_main_broad.tex, tab_hurdle_broad.tex, tab_enforce_broad.tex, numbers
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import INTERIM, TABLES
from models import CONTROLS_PART, design, ols_cluster
from tex import coef_cell, fint, fnum, put_number, write_numbers_tex, write_table

LN2 = np.log(2.0)
FE = ["year", "unspsc4", "orden"]
REG = [("minima", "M\\'inima cuant\\'ia"), ("menor", "Menor cuant\\'ia"), ("licitacion", "Licitaci\\'on p\\'ublica"), ("scored", "Scored (menor + licitaci\\'on)")]


def sub(w, g):
    return w[w.regime.isin(["menor", "licitacion"])] if g == "scored" else w[w.regime == g]


def fe_of(g):
    return FE + (["regime"] if g == "scored" else [])


def main() -> None:
    br = importlib.import_module("08_robustness")
    f04 = importlib.import_module("04_build_features")
    w = br.broad_sample()
    w["add_base_pp"] = w["rebate_pp"] - w["net_saving_pp"]
    w["add_ratio_raw"] = w["add_ratio_raw"].astype(float)
    # ---- entity ex-ante addition rate from the works universe (completed contracts only) ----
    works = pd.read_parquet(INTERIM / "secop2_works_universe.parquet")
    we = works.loc[(works.outcome_ok == 1) & works.nit_entidad.notna()].copy()
    pan = pd.DataFrame({"nit_entidad": we["nit_entidad"], "sign": pd.to_datetime(we["fecha_firma"]),
                        "end": pd.to_datetime(we["fecha_fin"].fillna(we["fecha_firma"] + pd.Timedelta(days=365))),
                        "y": (we["add_ratio_raw"] > 0.001).astype(float)}).dropna()
    pan["end"] = np.maximum(pan["end"].values.astype("datetime64[ns]"), (pan["sign"] + pd.Timedelta(days=1)).values.astype("datetime64[ns]"))
    foc = pd.DataFrame({"nit_entidad": w["nit_entidad"].values, "t": pd.to_datetime(w["fecha_firma"]).values}, index=w.index).dropna()
    ent = f04.prior_rates(foc, pan, "nit_entidad", "ent")
    w["ent_n"] = ent["ent_n_completed"]
    w["ent_soft"] = ent["ent_rate_completed"].where(ent["ent_n_completed"] >= 5)

    # ---- (a) main pass-through ----
    body = []
    for g, lab in REG:
        s = sub(w, g)
        l1, l2 = [lab], [""]
        for y in ["rebate_pp", "add_any_pp", "add_base_pp", "net_saving_pp"]:
            r = ols_cluster(s, y, ["ln_bidders"] + CONTROLS_PART, fe_of(g))
            c1, c2 = coef_cell(r.params["ln_bidders"] * LN2, r.bse["ln_bidders"] * LN2, r.pvalues["ln_bidders"], 2)
            l1.append(c1)
            l2.append(c2)
            put_number(f"mb {g} {y}", r.params["ln_bidders"] * LN2, ".2")
        l1.append(fint(r.nobs))
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    # Panel B: entity fixed effects (within-entity variation) on the broad universe
    w["ent_str"] = w["ent_g"].astype(str)
    bodyB = []
    for g, lab in REG:
        s = sub(w, g)
        l1, l2 = [lab], [""]
        for y in ["rebate_pp", "add_any_pp", "add_base_pp", "net_saving_pp"]:
            r = ols_cluster(s, y, ["ln_bidders"] + CONTROLS_PART, ["year"] + (["regime"] if g == "scored" else []), extra_dummies=["ent_str"])
            c1, c2 = coef_cell(r.params["ln_bidders"] * LN2, r.bse["ln_bidders"] * LN2, r.pvalues["ln_bidders"], 2)
            l1.append(c1)
            l2.append(c2)
            put_number(f"mbfe {g} {y}", r.params["ln_bidders"] * LN2, ".2")
            put_number(f"mbfep {g} {y}", r.pvalues["ln_bidders"], ".3")
        l1.append(fint(r.nobs))
        bodyB += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    body = (["\\multicolumn{6}{l}{\\emph{Panel A. Year, UNSPSC-4 and entity-order fixed effects}} \\\\"] + body +
            ["\\addlinespace", "\\multicolumn{6}{l}{\\emph{Panel B. Entity fixed effects (within-entity variation)}} \\\\"] + bodyB)
    write_table("tab_main_broad", "\n".join(body), "", "tab:mainbroad",
                "Same specification as \\cref{tab:main}, on the broad works universe: effect (pp) of doubling bidders, baseline controls, fixed effects for year, UNSPSC-4 class and entity order (Panel A) or for year and contracting entity (Panel B); regime fixed effects in the scored row. "
                "Net saving $=$ rebate $-$ additions/budget by construction. Standard errors clustered by entity.", "lccccc",
                "& (1) Rebate & (2) Any addition & (3) Additions/budget & (4) Net saving & $N$")

    # ---- (b) hurdle ----
    body = []
    for g, lab in REG:
        s = sub(w, g)
        X = design(s, ["ln_bidders"] + CONTROLS_PART, fe_of(g))
        y1 = (s["add_ratio_raw"] > 0.001).astype(float)
        pr = sm.Probit(y1, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(s["ent_g"])[0]}, maxiter=200)
        me = pr.get_margeff(at="overall")
        i = list(X.columns).index("ln_bidders") - 1
        ame, se, p = me.margeff[i] * LN2 * 100, me.margeff_se[i] * LN2 * 100, me.pvalues[i]
        pos = s[(s["add_ratio_raw"] > 0.001)].copy()
        pos["ln_ar"] = np.log(pos["add_ratio_raw"].clip(upper=1.0))
        r2 = ols_cluster(pos, "ln_ar", ["ln_bidders"] + CONTROLS_PART, fe_of(g))
        c1, c2 = coef_cell(ame, se, p, 2)
        c3, c4 = coef_cell(r2.params["ln_bidders"] * LN2 * 100, r2.bse["ln_bidders"] * LN2 * 100, r2.pvalues["ln_bidders"], 1)
        body += [f"{lab} & {c1} & {c3} \\\\", f" & {c2} & {c4} \\\\", f"\\quad \\emph{{N = {fint(len(s))} / {fint(r2.nobs)}}} & & \\\\[3pt]"]
        put_number(f"hb {g} ame", ame, ".1")
        put_number(f"hb {g} size", r2.params["ln_bidders"] * LN2 * 100, ".0")
        put_number(f"hbp {g} size", r2.pvalues["ln_bidders"], ".3")
    write_table("tab_hurdle_broad", "\n".join(body), "", "tab:hurdlebroad",
                "Two-part model as in \\cref{tab:hurdle}, on the broad works universe: probit average marginal effects (pp) for any addition, and the percent change in the addition ratio among contracts with an addition. Standard errors clustered by entity.",
                "lcc", "& (1) Pr(addition), AME pp & (2) Size of addition, \\%")

    # ---- (c) commitment interaction ----
    dd = w.dropna(subset=["ent_soft"]).copy()
    dd["soft_z"] = (dd["ent_soft"] - dd["ent_soft"].mean()) / dd["ent_soft"].std()
    dd["lnb_soft"] = dd["ln_bidders"] * dd["soft_z"]
    put_number("eb n", len(dd), ",d")
    put_number("eb mean", 100 * dd["ent_soft"].mean(), ".0")
    body = []
    for g, lab in REG:
        s = sub(dd, g)
        l1, l2, l3, l4 = [f"{lab}: $\\ln$(bidders)"], [""], ["\\quad $\\times$ entity prior-addition rate (z)"], [""]
        for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
            r = ols_cluster(s, y, ["ln_bidders", "soft_z", "lnb_soft"] + CONTROLS_PART, fe_of(g))
            for nm, la, lb in [("ln_bidders", l1, l2), ("lnb_soft", l3, l4)]:
                c1, c2 = coef_cell(r.params[nm] * LN2, r.bse[nm] * LN2, r.pvalues[nm], 2)
                la.append(c1)
                lb.append(c2)
            put_number(f"eb {g} {y}", r.params["lnb_soft"] * LN2, ".2")
            put_number(f"ebp {g} {y}", r.pvalues["lnb_soft"], ".3")
        # probit check at +-1 SD
        X = design(s, ["ln_bidders", "soft_z", "lnb_soft"] + CONTROLS_PART, fe_of(g))
        pr = sm.Probit((s["add_ratio_raw"] > 0.001).astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(s["ent_g"])[0]}, maxiter=200)
        for zl, z in [("low", -1.0), ("high", 1.0)]:
            X0, X1 = X.copy(), X.copy()
            X0["soft_z"] = X1["soft_z"] = z
            X0["lnb_soft"] = X0["ln_bidders"] * z
            X1["ln_bidders"] = X0["ln_bidders"] + LN2
            X1["lnb_soft"] = X1["ln_bidders"] * z
            put_number(f"eb probit {g} {zl}", 100 * float((pr.predict(X1) - pr.predict(X0)).mean()), ".1")
        put_number(f"eb probit int {g}", pr.params["lnb_soft"], ".2")
        put_number(f"ebp probit int {g}", pr.pvalues["lnb_soft"], ".3")
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\", " & ".join(l3) + " \\\\", " & ".join(l4) + " \\\\", f"\\quad \\emph{{N = {fint(len(s))}}} & & & \\\\[3pt]"]
    write_table("tab_enforce_broad", "\n".join(body), "", "tab:enforcebroad",
                "Same specification as \\cref{tab:enforce}, on the broad works universe. The moderator is the share of the entity's earlier completed works contracts with an addition (at least five), standardised. Coefficients scaled to doubling bidders (pp). Standard errors clustered by entity.",
                "lccc", "& (1) Rebate & (2) Any addition & (3) Net saving")
    # ---- (d) exact decomposition on the broad universe (pairs-cluster bootstrap by entity) ----
    from config import SEED
    rng = np.random.default_rng(SEED)
    B_REP = 200
    dec = []
    for g, lab in REG:
        s_ = sub(w, g).reset_index(drop=True)
        point = {}
        for y in ["rebate_pp", "add_base_pp", "net_saving_pp"]:
            point[y] = ols_cluster(s_, y, ["ln_bidders"] + CONTROLS_PART, fe_of(g)).params["ln_bidders"] * LN2
        groups = s_["ent_g"].unique()
        by = {k: s_.index[s_.ent_g == k].values for k in groups}
        draws = []
        for _ in range(B_REP):
            samp = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([by[k] for k in samp])
            bs = s_.loc[idx].reset_index(drop=True)
            bs["ent_g"] = np.repeat(np.arange(len(samp)), [len(by[k]) for k in samp])
            try:
                draws.append([ols_cluster(bs, y, ["ln_bidders"] + CONTROLS_PART, fe_of(g)).params["ln_bidders"] * LN2
                              for y in ["rebate_pp", "add_base_pp", "net_saving_pp"]])
            except Exception:
                continue
        ci = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
        dec.append(dict(regime=g, label=lab, rebate=point["rebate_pp"], rebate_lo=ci[0, 0], rebate_hi=ci[1, 0],
                        add=point["add_base_pp"], add_lo=ci[0, 1], add_hi=ci[1, 1],
                        net=point["net_saving_pp"], net_lo=ci[0, 2], net_hi=ci[1, 2]))
        put_number(f"decb net lo {g}", ci[0, 2], ".2")
        put_number(f"decb net hi {g}", ci[1, 2], ".2")
    dec = pd.DataFrame(dec)
    dec.to_csv(TABLES / "decomposition_broad.csv", index=False)
    body = [f"{r['label']} & {fnum(r.rebate, 2)} & [{fnum(r.rebate_lo, 2)}; {fnum(r.rebate_hi, 2)}] & {fnum(r['add'], 2)} & [{fnum(r['add_lo'], 2)}; {fnum(r['add_hi'], 2)}] & "
            f"{fnum(r.net, 2)} & [{fnum(r.net_lo, 2)}; {fnum(r.net_hi, 2)}] \\\\" for _, r in dec.iterrows()]
    write_table("tab_decomp_broad", "\n".join(body), "", "tab:decompbroad",
                f"Decomposition of the effect of doubling participation on the broad works universe (pp of the reference budget); net saving $=$ rebate $-$ erosion by additions. 95\\% intervals from a pairs-cluster bootstrap by entity ({B_REP} replications).",
                "lcccccc", "& \\multicolumn{2}{c}{Price rebate} & \\multicolumn{2}{c}{Erosion by additions} & \\multicolumn{2}{c}{Net saving} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n& Effect & 95\\% CI & Effect & 95\\% CI & Effect & 95\\% CI")

    # ---- (e) robustness grid on the broad universe ----
    def spec_sets():
        yrs = w["year"].astype(int)
        return [("Baseline", w, {}),
                ("Add department fixed effects", w, dict(extra=["departamento"])),
                ("Entity fixed effects", w, dict(fe=["year"], extra=["ent_str"])),
                ("Drop top 5\% bidder counts (by regime)", w[w.n_bidders <= w.groupby("regime")["n_bidders"].transform(lambda x: x.quantile(.95))], {}),
                ("Signed 2018--2022 only", w[yrs <= 2022], {}),
                ("No cap on final value / budget", w.assign(net_saving_pp=100 * (1 - w["valor_contrato"] / w["precio_base"])).assign(add_base_pp=lambda x: x["rebate_pp"] - x["net_saving_pp"]), {}),
                ("Territorial entities only", w[w.territorial == 1], dict(fe=["year", "unspsc4"])),
                ("National-order entities only", w[w.territorial == 0], dict(fe=["year", "unspsc4"]))]
    body = []
    for nm, data, kw in spec_sets():
        l1, l2 = [nm], [""]
        for y, gs in [("rebate_pp", ["minima", "scored"]), ("add_any_pp", ["scored"]), ("net_saving_pp", ["minima", "scored"])]:
            for g in gs:
                s_ = sub(data, g)
                fe_ = list(kw.get("fe", FE)) + (["regime"] if g == "scored" else []) + list(kw.get("extra", []))
                fe_main = [f for f in fe_ if f != "ent_str"]
                r = ols_cluster(s_, y, ["ln_bidders"] + CONTROLS_PART, fe_main, extra_dummies=["ent_str"] if "ent_str" in kw.get("extra", []) else None)
                c1, c2 = coef_cell(r.params["ln_bidders"] * LN2, r.bse["ln_bidders"] * LN2, r.pvalues["ln_bidders"], 2)
                l1.append(c1)
                l2.append(c2)
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    write_table("tab_robust_broad", "\n".join(body), "", "tab:robustbroad",
                "Same layout as \\cref{tab:robust}, on the broad works universe: effect (pp) of doubling bidders under alternative specifications and samples; ``Scored'' pools menor cuant\\'ia and licitaci\\'on. Standard errors clustered by entity.",
                "lccccc", "& \\multicolumn{2}{c}{Rebate} & Any addition & \\multicolumn{2}{c}{Net saving} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}\n& M\\'inima & Scored & Scored & M\\'inima & Scored")

    write_numbers_tex()
    print("ok", len(w), len(dd))


if __name__ == "__main__":
    main()
