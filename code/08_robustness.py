"""STEP 08 - Robustness, inference and identification checks for the participation results.

R1  Specification grid (department FE, entity FE, trimming, bidder trimming, vintage, entity order)
R2  Wild-cluster bootstrap-t p-values (Webb weights) for the baseline coefficients
R3  Placebo: merit-based contests (concurso de meritos) where price is not competed
R4  Broad SECOP II works universe (all closed single-award works contracts, not only the frame)
R5  Coefficient stability (Oster 2019) for the scored-regime renegotiation effect
R6  Exploratory IV: calendar exposure of the bidding window; Anderson-Rubin confidence sets
R7  Mediation through price (product of coefficients, cluster bootstrap)

Outputs: output/latex/tables/tab_robust.tex, tab_wild.tex, tab_placebo.tex, tab_broad.tex, tab_iv.tex, tab_mediation.tex
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ANALYSIS_PQ, INTERIM, N_BOOT, SEED, TABLES
from models import CONTROLS_PART, FE_PART, design, ols_cluster, part_mask, prepare_part
from tex import coef_cell, fint, fnum, put_number, wild_cluster_p, write_numbers_tex, write_table

LN2 = np.log(2.0)
GROUPS = [("minima", "M\\'inima"), ("menor", "Menor"), ("licitacion", "Licit."), ("scored", "Scored")]


def sub_group(d: pd.DataFrame, g: str) -> pd.DataFrame:
    if g == "scored":
        return d[d.regime.isin(["menor", "licitacion"])]
    return d[d.regime == g]


def run(d: pd.DataFrame, y: str, g: str, extra_fe=None, fe=None, x=None):
    sub = sub_group(d, g)
    fe = list(fe if fe is not None else FE_PART) + (["regime"] if g == "scored" else []) + list(extra_fe or [])
    return ols_cluster(sub, y, (x or ["ln_bidders"]) + CONTROLS_PART, fe)


def per_doubling(res, name="ln_bidders"):
    return res.params[name] * LN2, res.bse[name] * LN2, res.pvalues[name]


# ---------------------------------------------------------------------------------------------
def r1_grid(df: pd.DataFrame) -> pd.DataFrame:
    d0 = prepare_part(df)
    specs = {
        "Baseline": lambda: (d0, {}),
        "Add department fixed effects": lambda: (d0, dict(extra_fe=["departamento"])),
        "Entity fixed effects": lambda: (d0, dict(fe=["year"], extra_fe=["ent_id"])),
        "No trimming (award/budget $\\geq$ 0.10)": lambda: (prepare_part(df, part_mask(df, 0.10)), {}),
        "Drop top 5\\% bidder counts (by regime)": lambda: (d0[d0.n_bidders <= d0.groupby("regime")["n_bidders"].transform(lambda s: s.quantile(.95))], {}),
        "Signed 2018--2022 only": lambda: (d0[d0.year.astype(int) <= 2022], {}),
        "Exclude national-order entities": lambda: (d0[d0.orden != "A"], dict(fe=["year", "unspsc4"])),
    }
    rows = []
    for nm, fn in specs.items():
        d, kw = fn()
        for g, _ in GROUPS:
            for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
                try:
                    res = run(d, y, g, **kw)
                    b, se, p = per_doubling(res)
                    rows.append(dict(spec=nm, group=g, y=y, b=b, se=se, p=p, n=int(res.nobs)))
                except Exception as e:                                   # tiny subsamples under entity FE
                    rows.append(dict(spec=nm, group=g, y=y, b=np.nan, se=np.nan, p=np.nan, n=0))
    R = pd.DataFrame(rows)
    R.to_csv(TABLES / "robust_grid.csv", index=False)
    body = []
    for nm in specs:
        l1, l2 = [nm], [""]
        for y, gs in [("rebate_pp", ["minima", "scored"]), ("add_any_pp", ["scored"]), ("net_saving_pp", ["minima", "scored"])]:
            for g in gs:
                r = R[(R.spec == nm) & (R.group == g) & (R.y == y)].iloc[0]
                c1, c2 = coef_cell(r.b, r.se, r.p, 2) if r.n > 0 else ("--", "")
                l1.append(c1)
                l2.append(c2)
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    header = ("& \\multicolumn{2}{c}{Rebate} & Any addition & \\multicolumn{2}{c}{Net saving} \\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-6}\n& M\\'inima & Scored & Scored & M\\'inima & Scored")
    notes = ("Each cell is the effect (pp) of doubling the number of bidders from a separate regression, with the baseline controls of \\cref{tab:main} "
             "unless the row says otherwise. ``Scored'' pools menor cuant\\'ia and licitaci\\'on (with regime fixed effects). The entity-fixed-effects row "
             "replaces the other fixed effects by entity dummies (identification from within-entity variation only). Standard errors clustered by entity.")
    write_table("tab_robust", "\n".join(body), "", "tab:robust", notes, "lccccc", header)
    b = R[(R.spec == "Entity fixed effects")]
    for g in ["minima", "scored"]:
        for y in ["rebate_pp", "net_saving_pp", "add_any_pp"]:
            r = b[(b.group == g) & (b.y == y)].iloc[0]
            put_number(f"efe {g} {y}", r.b, ".2")
            put_number(f"efep {g} {y}", r.p, ".3")
    return R


# ---------------------------------------------------------------------------------------------
def r2_wild(df: pd.DataFrame) -> None:
    d = prepare_part(df)
    rows = []
    for g, lab in GROUPS:
        for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
            res = run(d, y, g)
            X, yv, grp = res.X_.values, res.y_.values, pd.factorize(res.g_)[0]
            j = list(res.X_.columns).index("ln_bidders")
            pw = wild_cluster_p(yv, X, j, grp, B=999, seed=SEED)
            rows.append(dict(group=g, y=y, b=res.params["ln_bidders"] * LN2, p_cluster=res.pvalues["ln_bidders"],
                             p_wild=pw, clusters=res.n_clusters))
    W = pd.DataFrame(rows)
    W.to_csv(TABLES / "wild_bootstrap.csv", index=False)
    body = []
    for g, lab in GROUPS:
        l = [lab]
        for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
            r = W[(W.group == g) & (W.y == y)].iloc[0]
            l += [fnum(r.b, 2), fnum(r.p_cluster, 3), fnum(r.p_wild, 3)]
        l.append(fint(W[(W.group == g)].clusters.iloc[0]))
        body.append(" & ".join(l) + " \\\\")
    header = ("& \\multicolumn{3}{c}{Rebate} & \\multicolumn{3}{c}{Any addition} & \\multicolumn{3}{c}{Net saving} & \\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n& Effect & $p$ cl. & $p$ wild & Effect & $p$ cl. & $p$ wild & Effect & $p$ cl. & $p$ wild & Clusters")
    notes = ("Effect of doubling the number of bidders (pp) from the baseline specification. $p$ cl.: cluster-robust $t$ test; $p$ wild: wild-cluster "
             "bootstrap-t with Webb weights and 999 replications \\citep{cameron2008}, which is reliable with few clusters.")
    write_table("tab_wild", "\n".join(body), "", "tab:wild", notes, "lcccccccccc", header, size="\\scriptsize")
    for g in ["minima", "menor", "licitacion", "scored"]:
        for y in ["add_any_pp", "net_saving_pp"]:
            r = W[(W.group == g) & (W.y == y)].iloc[0]
            put_number(f"wild {g} {y}", r.p_wild, ".3")


# ---------------------------------------------------------------------------------------------
def r3_placebo(df: pd.DataFrame) -> None:
    c = df[df.s_concurso == 1].copy()
    c["ent_g"] = pd.factorize(c["ent_id"])[0]
    c["ln_plazo_raw"] = np.log(c["plazo_proc_dias"].where(c["plazo_proc_dias"] > 0))
    c["plazo_miss"] = c["ln_plazo_raw"].isna().astype(float)
    c["ln_plazo"] = c["ln_plazo_raw"].fillna(c["ln_plazo_raw"].median())
    c["year"] = c["year"].astype(int).astype(str)
    c["rebate_pp"] = 100 * (1 - c["award_ratio"].clip(upper=1.0))
    c["add_any_pp"] = 100 * c["add_any"]
    c["add_base_pp"] = 100 * ((c["valor_contrato_s2"] - c["valor_adjudicado"]) / c["precio_base"]).clip(-1, 1)
    c["final_pp"] = 100 * (c["final_ratio"].clip(upper=2.0))
    put_number("n concurso", len(c), ",d")
    put_number("concurso rebate mean", c["rebate_pp"].mean(), ".1")
    put_number("concurso bidders mean", c["n_bidders"].mean(), ".0")
    rows = []
    body = []
    l1, l2 = ["Placebo: merit-based contests"], [""]
    for y in ["rebate_pp", "add_any_pp", "add_base_pp"]:
        res = ols_cluster(c, y, ["ln_bidders", "ln_base", "ln_plazo", "plazo_miss"], ["year"])
        b, se, p = per_doubling(res)
        c1, c2 = coef_cell(b, se, p, 2)
        l1.append(c1)
        l2.append(c2)
        rows.append(dict(y=y, b=b, se=se, p=p, n=int(res.nobs)))
        put_number(f"placebo {y}", b, ".2")
        put_number(f"placebop {y}", p, ".3")
    body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\", f"\\quad \\emph{{N = {fint(len(c))}}} & & & \\\\"]
    pd.DataFrame(rows).to_csv(TABLES / "placebo_concurso.csv", index=False)
    header = "& (1) Rebate & (2) Any addition & (3) Additions/budget"
    notes = ("Concurso de m\\'eritos (consultancy and supervision contracts selected on quality, with the price fixed by the entity). Price does not enter the "
             "award, so the rebate should be close to zero by design. Effects (pp) of doubling bidders; controls: $\\ln$ budget, $\\ln$ duration, year effects.")
    write_table("tab_placebo", "\n".join(body), "", "tab:placebo", notes, "lccc", header)


# ---------------------------------------------------------------------------------------------
def r4_broad(df_unused) -> None:
    w = pd.read_parquet(INTERIM / "secop2_works_universe.parquet")
    w = w[(w.outcome_ok == 1) & w.prov_unicos_resp.gt(0) & w.precio_base.gt(0)].copy()
    w["regime"] = w["modalidad"].map(lambda m: ("minima" if "nima cuant" in str(m).lower() else "menor" if "menor cuant" in str(m).lower()
                                                 else "licitacion" if "licitaci" in str(m).lower() else "otra"))
    w = w[w.regime.isin(["minima", "menor", "licitacion"])]
    w["award_ratio"] = w["valor_adjudicado"] / w["precio_base"]
    w = w[w.award_ratio.between(0.30, 1.0001)]
    w["year"] = pd.to_datetime(w["fecha_firma"]).dt.year
    w = w[w.year.between(2018, 2024)]
    w["n_bidders"] = w["prov_unicos_resp"]
    w["ln_bidders"] = np.log(w["n_bidders"])
    w["add_ratio_raw"] = w["add_ratio_raw"]
    w["add_any_pp"] = 100 * (w["add_ratio_raw"] > 0.001).astype(float)
    w["rebate_pp"] = 100 * (1 - w["award_ratio"].clip(upper=1.0))
    w["final_ratio_w"] = (w["valor_contrato"] / w["precio_base"]).clip(upper=2.0)
    w["net_saving_pp"] = 100 * (1 - w["final_ratio_w"])
    w["ln_base"] = np.log(w["precio_base"])
    dur = w["duracion"].astype(float) * np.select([w.unidad_duracion.fillna("").str.lower().str.startswith("m"),
                                                    w.unidad_duracion.fillna("").str.lower().str.startswith("d"),
                                                    w.unidad_duracion.fillna("").str.lower().str.startswith("a")], [30.0, 1.0, 365.0], default=np.nan)
    w["ln_plazo_raw"] = np.log(dur.where(dur > 0))
    w["plazo_miss"] = w["ln_plazo_raw"].isna().astype(float)
    w["ln_plazo"] = w["ln_plazo_raw"].fillna(w["ln_plazo_raw"].median())
    w["unspsc4"] = w["unspsc4"].astype(str)
    w["orden"] = w["orden_entidad"].fillna("NA")
    w["year"] = w["year"].astype(str)
    w["ent_g"] = pd.factorize(w["nit_entidad"])[0]
    put_number("n broad", len(w), ",d")
    put_number("broad ratio", len(w) / int((df_unused["s_part"] == 1).sum()), ".0f")   # size relative to the participation sample
    put_number("clusters broad", w["ent_g"].nunique(), ",d")
    rows, body = [], []
    for g, lab in [("minima", "M\\'inima cuant\\'ia"), ("menor", "Menor cuant\\'ia"), ("licitacion", "Licitaci\\'on p\\'ublica"), ("scored", "Scored (menor + licit.)")]:
        sub = w[w.regime.isin(["menor", "licitacion"])] if g == "scored" else w[w.regime == g]
        fe = ["year", "unspsc4", "orden"] + (["regime"] if g == "scored" else [])
        l1, l2 = [lab], [""]
        for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
            res = ols_cluster(sub, y, ["ln_bidders", "ln_base", "ln_plazo", "plazo_miss"], fe)
            b, se, p = per_doubling(res)
            c1, c2 = coef_cell(b, se, p, 2)
            l1.append(c1)
            l2.append(c2)
            rows.append(dict(group=g, y=y, b=b, se=se, p=p, n=int(res.nobs), clusters=res.n_clusters))
            put_number(f"broad {g} {y}", b, ".2")
            put_number(f"broadp {g} {y}", p, ".3")
        l1.append(fint(len(sub)))
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    pd.DataFrame(rows).to_csv(TABLES / "broad_universe.csv", index=False)
    header = "& (1) Rebate & (2) Any addition & (3) Net saving & $N$"
    notes = ("All closed SECOP II contracts with UNSPSC codes 72*, 95* or 811015 (building, civil-works and related services) signed 2018--2024, with a single award "
             "row and the same trimming as the baseline sample. Effects (pp) of doubling the number of bidders; baseline controls; standard errors clustered by entity.")
    write_table("tab_broad", "\n".join(body), "", "tab:broad", notes, "lcccc", header)


# ---------------------------------------------------------------------------------------------
def oster(df: pd.DataFrame) -> None:
    d = prepare_part(df)
    sub = d[d.regime.isin(["menor", "licitacion"])]
    short = ols_cluster(sub, "add_any_pp", ["ln_bidders"], ["regime"])
    full = ols_cluster(sub, "add_any_pp", ["ln_bidders"] + CONTROLS_PART, FE_PART + ["regime"])
    bd, Rd = short.params["ln_bidders"], short.rsquared
    bt, Rt = full.params["ln_bidders"], full.rsquared
    Rmax = min(1.0, 1.3 * Rt)
    delta = bt * (Rt - Rd) / ((bd - bt) * (Rmax - Rt)) if (bd - bt) != 0 else np.inf
    put_number("oster delta", delta, ".1")
    put_number("oster short", bd * LN2, ".2")
    put_number("oster full", bt * LN2, ".2")
    pd.DataFrame([dict(beta_short=bd * LN2, r2_short=Rd, beta_full=bt * LN2, r2_full=Rt, r2max=Rmax, delta=delta)]).to_csv(
        TABLES / "oster.csv", index=False)


# ---------------------------------------------------------------------------------------------
def iv_ar(df: pd.DataFrame) -> None:
    d = prepare_part(df)
    d = d.dropna(subset=["win_holiday_days", "win_cal_days"]).copy()
    for c in ["win_holiday_days", "win_cal_days", "win_weekend_days", "ln_bidders"]:
        d[c] = d[c].astype(float)
    d["ln_cal"] = np.log(d["win_cal_days"].clip(lower=1))
    rows, body = [], []
    for g, lab in [("minima", "M\\'inima"), ("scored", "Scored"), ("pooled", "Pooled")]:
        sub = d if g == "pooled" else sub_group(d, g)
        fe = FE_PART + (["regime"] if g in ("scored", "pooled") else [])
        W = design(sub, CONTROLS_PART + ["ln_cal", "win_weekend_days"], fe).astype(float)
        grp = pd.factorize(sub["ent_g"])[0]
        out_cells = [lab]
        for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
            ok = sub[y].notna()
            Wm = W[ok].values
            x = sub.loc[ok, "ln_bidders"].to_numpy(dtype=float)
            yv = sub.loc[ok, y].to_numpy(dtype=float)
            z = sub.loc[ok, "win_holiday_days"].to_numpy(dtype=float)
            gg = grp[ok.values]
            P = Wm @ np.linalg.pinv(Wm.T @ Wm) @ Wm.T
            xr, yr, zr = x - P @ x, yv - P @ yv, z - P @ z
            G = len(np.unique(gg))

            def cl_t(dep, reg):
                b = (reg @ dep) / (reg @ reg)
                e = dep - b * reg
                s = np.zeros(G)
                np.add.at(s, pd.factorize(gg)[0], reg * e)
                v = (s ** 2).sum() / (reg @ reg) ** 2 * G / (G - 1)
                return b, np.sqrt(v)
            b1, se1 = cl_t(xr, zr)
            F = (b1 / se1) ** 2
            bt = (zr @ yr) / (zr @ xr)
            e = yr - bt * xr
            s = np.zeros(G)
            np.add.at(s, pd.factorize(gg)[0], zr * e)
            se_iv = np.sqrt((s ** 2).sum() / (zr @ xr) ** 2 * G / (G - 1))
            grid = np.linspace(-60, 60, 1201)
            keep = []
            for b0 in grid:
                bb, ss = cl_t(yr - b0 * xr, zr)
                keep.append(abs(bb / ss) < 1.96)
            keep = np.array(keep)
            if keep.all() or keep[0] or keep[-1]:
                ci = "unbounded"
            else:
                ci = f"[{grid[keep].min() * LN2:.1f}; {grid[keep].max() * LN2:.1f}]"
            rows.append(dict(group=g, y=y, first_stage_coef=b1, first_stage_F=F, iv_b=bt * LN2, iv_se=se_iv * LN2, ar_ci=ci, n=int(ok.sum())))
            out_cells += [f"{fnum(bt * LN2, 1)} ({fnum(se_iv * LN2, 1)})", ci]
        out_cells.append(fnum(rows[-1]["first_stage_F"], 1))
        body.append(" & ".join(out_cells) + " \\\\")
        put_number(f"ivF {g}", rows[-1]["first_stage_F"], ".1")
    pd.DataFrame(rows).to_csv(TABLES / "iv_ar.csv", index=False)
    header = ("& \\multicolumn{2}{c}{Rebate} & \\multicolumn{2}{c}{Any addition} & \\multicolumn{2}{c}{Net saving} & \\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n& 2SLS (s.e.) & AR 95\\% set & 2SLS (s.e.) & AR 95\\% set & 2SLS (s.e.) & AR 95\\% set & First-stage $F$")
    notes = ("2SLS effects (pp) of doubling bidders, instrumenting $\\ln$(bidders) with the number of public holidays falling on weekdays inside the bidding "
             "window (publication to offer deadline), controlling for the window's calendar length and weekend days. The first-stage $F$ is far below conventional "
             "thresholds, so we do \\emph{not} rely on 2SLS; Anderson--Rubin sets \\citep{anderson1949} are valid under weak instruments and are reported as the "
             "inference of record (``unbounded'' = uninformative).")
    write_table("tab_iv", "\n".join(body), "", "tab:iv", notes, "lcccccc c".replace(" ", ""), header, size="\\scriptsize")


# ---------------------------------------------------------------------------------------------
def mediation(df: pd.DataFrame) -> None:
    d = prepare_part(df)
    sub = d[d.regime.isin(["menor", "licitacion"])].reset_index(drop=True)
    sub["z_rebate"] = (sub["rebate_pp"] - sub["rebate_pp"].mean()) / sub["rebate_pp"].std()
    fe = FE_PART + ["regime"]

    def est(s):
        tot = ols_cluster(s, "add_any_pp", ["ln_bidders"] + CONTROLS_PART, fe)
        a = ols_cluster(s, "z_rebate", ["ln_bidders"] + CONTROLS_PART, fe)
        bdir = ols_cluster(s, "add_any_pp", ["ln_bidders", "z_rebate"] + CONTROLS_PART, fe)
        return (tot.params["ln_bidders"] * LN2, a.params["ln_bidders"] * LN2, bdir.params["z_rebate"],
                bdir.params["ln_bidders"] * LN2)

    tot, a, b, direct = est(sub)
    med = a * b
    rng = np.random.default_rng(SEED)
    groups = sub["ent_g"].unique()
    by = {g: sub.index[sub.ent_g == g].values for g in groups}
    draws = []
    for _ in range(300):
        samp = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([by[g] for g in samp])
        bs = sub.loc[idx].reset_index(drop=True)
        bs["ent_g"] = np.repeat(np.arange(len(samp)), [len(by[g]) for g in samp])
        try:
            t, a_, b_, d_ = est(bs)
            draws.append([t, a_ * b_, d_, a_, b_])
        except Exception:
            continue
    dr = np.array(draws)
    ci = np.percentile(dr, [2.5, 97.5], axis=0)
    rows = [("Total effect of doubling bidders on Pr(addition), pp", tot, ci[0, 0], ci[1, 0]),
            ("Indirect effect via the price rebate ($a \\times b$)", med, ci[0, 1], ci[1, 1]),
            ("Direct effect (holding the rebate fixed)", direct, ci[0, 2], ci[1, 2]),
            ("$a$: doubling bidders $\\to$ rebate (SD units)", a, ci[0, 3], ci[1, 3]),
            ("$b$: rebate (1 SD) $\\to$ Pr(addition), pp", b, ci[0, 4], ci[1, 4])]
    body = [f"{r[0]} & {fnum(r[1], 2)} & [{fnum(r[2], 2)}; {fnum(r[3], 2)}] \\\\" for r in rows]
    pd.DataFrame(rows, columns=["effect", "est", "lo", "hi"]).to_csv(TABLES / "mediation.csv", index=False)
    put_number("med total", tot, ".1")
    put_number("med indirect", med, ".2")
    put_number("med direct", direct, ".1")
    header = "Quantity & Estimate & 95\\% CI"
    notes = ("Scored regimes (menor cuant\\'ia and licitaci\\'on), product-of-coefficients mediation \\citep{baron1986,imai2010} with controls and fixed effects as in "
             "\\cref{tab:main}. Identification of the indirect effect requires sequential ignorability, which we do not claim; the exercise only "
             "asks how much of the participation--renegotiation association is statistically accounted for by the price rebate. "
             "Intervals: cluster bootstrap, 300 replications.")
    write_table("tab_mediation", "\n".join(body), "", "tab:mediation", notes, "lcc", header)


def main() -> None:
    import sys
    only = set(sys.argv[1:])            # optional: run a subset, e.g. `python 08_robustness.py iv mediation`
    df = pd.read_parquet(ANALYSIS_PQ)
    steps = [("grid", r1_grid), ("wild", r2_wild), ("placebo", r3_placebo), ("broad", r4_broad), ("oster", oster),
             ("iv", iv_ar), ("mediation", mediation)]
    for name, fn in steps:
        if not only or name in only:
            fn(df)
    write_numbers_tex()
    print("robustness done")


if __name__ == "__main__":
    main()
