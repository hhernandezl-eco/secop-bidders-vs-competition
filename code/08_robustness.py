"""STEP 08 - Robustness, inference and identification checks for the participation results.

R1  Specification grid (department FE, entity FE, trimming, bidder trimming, vintage, entity order)
R2  Wild-cluster bootstrap-t p-values (Webb weights) for the baseline coefficients
R3  Placebo: merit-based contests (concurso de meritos) where price is not competed
R4  Broad SECOP II works universe (all closed single-award works contracts, not only the frame)
R5  Coefficient stability (Oster 2019) for the scored-regime renegotiation effect
R6  Exploratory IV: calendar exposure of the bidding window; Anderson-Rubin confidence sets
R7  Mediation through price (product of coefficients, cluster bootstrap)

Outputs: paper/tables/tab_robust.tex, tab_wild.tex, tab_placebo.tex, tab_broad.tex, tab_iv.tex
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ANALYSIS_PQ, INTERIM, N_BOOT, SEED, TABLES
from models import CONTROLS_PART, FE_PART, design, ols_cluster, part_mask, prepare_part
from tex import coef_cell, fint, fnum, put_number, star, wild_cluster_p, write_numbers_tex, write_table

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

    def uncap(d):
        """Net saving without the F/B <= 2 cap; additions/budget stays the exact residual R - S."""
        d = d.copy()
        d["net_saving_pp"] = 100 * (1 - d["final_ratio"])
        d["add_base_pp"] = d["rebate_pp"] - d["net_saving_pp"]
        return d
    specs = {
        "Baseline": lambda: (d0, {}),
        "Add department fixed effects": lambda: (d0, dict(extra_fe=["departamento"])),
        "Entity fixed effects": lambda: (d0, dict(fe=["year"], extra_fe=["ent_id"])),
        "No trimming (award/budget $\\geq$ 0.10)": lambda: (prepare_part(df, part_mask(df, 0.10)), {}),
        "Drop top 5\\% bidder counts (by regime)": lambda: (d0[d0.n_bidders <= d0.groupby("regime")["n_bidders"].transform(lambda s: s.quantile(.95))], {}),
        "Signed 2018--2022 only": lambda: (d0[d0.year.astype(int) <= 2022], {}),
        "No cap on final value / budget": lambda: (uncap(d0), {}),
        "Territorial entities only (excl. national order)": lambda: (d0[d0.orden != "A"], dict(fe=["year", "unspsc4"])),
        "National-order entities only": lambda: (d0[d0.orden == "A"], dict(fe=["year", "unspsc4"])),
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
            rows.append(dict(group=g, spec=slab, y=y, b=res.params["ln_bidders"] * LN2, p_cluster=res.pvalues["ln_bidders"],
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
    pub = pd.to_datetime(d["fecha_publicacion"], errors="coerce")
    d["pub_month"] = pub.dt.month.fillna(0).astype(int).astype(str)
    d["pub_week"] = pub.dt.isocalendar().week.astype("float").fillna(0).astype(int).astype(str)
    for g0 in ["minima", "menor", "licitacion"]:
        put_number(f"win median {g0}", float(d.loc[d.regime == g0, "win_cal_days"].median()), ".0f")
    rows, body = [], []
    SPECS = [("", "baseline", []), ("month", "+ publication-month FE", ["pub_month"]), ("week", "+ publication-week FE", ["pub_week"])]
    GR = [("minima", "M\\'inima"), ("scored", "Scored"), ("pooled", "Pooled")]
    for (g, lab0), (stag, slab, extra_fe) in [(a, b) for a in GR for b in SPECS]:
        lab = f"{lab0}, {slab}"
        sub = d if g == "pooled" else sub_group(d, g)
        fe = FE_PART + (["regime"] if g in ("scored", "pooled") else []) + extra_fe
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
        put_number(f"ivF{stag} {g}", rows[-1]["first_stage_F"], ".1")
    pd.DataFrame(rows).to_csv(TABLES / "iv_ar.csv", index=False)
    header = ("& \\multicolumn{2}{c}{Rebate} & \\multicolumn{2}{c}{Any addition} & \\multicolumn{2}{c}{Net saving} & \\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n& 2SLS (s.e.) & AR 95\\% set & 2SLS (s.e.) & AR 95\\% set & 2SLS (s.e.) & AR 95\\% set & First-stage $F$")
    notes = ("2SLS effects (pp) of doubling bidders, instrumenting $\\ln$(bidders) with the number of public holidays falling on weekdays inside the bidding "
             "window (publication to offer deadline), controlling for the window's calendar length and weekend days. Because entities may publish before holidays for reasons of budget closure or administrative haste, which could also relate to incomplete studies, the second and third rows of each group add fixed effects for the month and the week of the year of publication, so that identification uses only movable holidays within the same calendar period. The first-stage $F$ is far below conventional "
             "thresholds, so we do \\emph{not} rely on 2SLS; Anderson--Rubin sets \\citep{anderson1949} are valid under weak instruments and are reported as the "
             "inference of record (``unbounded'' = uninformative).")
    write_table("tab_iv", "\n".join(body), "", "tab:iv", notes, "lcccccc c".replace(" ", ""), header, size="\\scriptsize")


# ---------------------------------------------------------------------------------------------


def scale_check(df: pd.DataFrame) -> None:
    """Is the regime difference an artefact of project scale? Flexible ln(B) by regime, interacted with ln(N).
    Slopes are average marginal effects over the observed distribution of budgets within each regime (not evaluated at ln B = 0)."""
    d = prepare_part(df).copy()
    d["is_min"] = (d.regime == "minima").astype(float)
    d["lnb_min"] = d["ln_bidders"] * d["is_min"]
    d["lnb_sc"] = d["ln_bidders"] * (1 - d["is_min"])
    lo = d.loc[d.regime != "minima", "ln_base"].quantile(0.01)
    hi = d.loc[d.regime == "minima", "ln_base"].quantile(0.99)
    cs_mask = (d.ln_base >= lo) & (d.ln_base <= hi)
    med = d.loc[cs_mask, "ln_base"].median()
    c = d["ln_base"] - med          # centred at the median of the common support
    for k in (1, 2, 3):
        d[f"c{k}"] = c ** k
        d[f"c{k}_min"] = d[f"c{k}"] * d["is_min"]
        d[f"c{k}_Nm"] = d[f"c{k}"] * d["lnb_min"]
        d[f"c{k}_Ns"] = d[f"c{k}"] * d["lnb_sc"]
    cs = d[cs_mask]
    put_number("support n", len(cs), ",d")
    put_number("support minima", int((cs.regime == "minima").sum()), ",d")
    put_number("support scored", int((cs.regime != "minima").sum()), ",d")
    base_ctrl = ["ln_plazo", "plazo_miss"]
    specs = [("(1) Linear $\ln B$ (baseline)", d, ["ln_base"], 0),
             ("(2) Cubic $\ln B$ and $\ln B\times\ln N$, by regime", d,
              ["c1", "c2", "c3", "c1_min", "c2_min", "c3_min", "c1_Nm", "c2_Nm", "c3_Nm", "c1_Ns", "c2_Ns", "c3_Ns"], 3),
             ("(3) Linear $\ln B$ and $\ln B\times\ln N$, by regime, common support", cs,
              ["c1", "c1_min", "c1_Nm", "c1_Ns"], 1)]
    body, rows = [], []
    for lab, dd, xs, deg in specs:
        l1, l2 = [lab], [""]
        for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
            res = ols_cluster(dd, y, ["lnb_min", "lnb_sc", "is_min"] + xs + base_ctrl, ["year", "orden", "unspsc4"])
            names = list(res.params.index)
            def ame(reg_mask, slope, tag):
                v = np.zeros(len(names)); v[names.index(slope)] = 1.0
                for k in range(1, deg + 1):
                    v[names.index(f"c{k}_N{tag}")] = float(dd.loc[reg_mask, f"c{k}"].mean())
                return v
            vm = ame(dd.regime == "minima", "lnb_min", "m")
            vs = ame(dd.regime != "minima", "lnb_sc", "s")
            tm, ts, td = res.t_test(vm), res.t_test(vs), res.t_test(vm - vs)
            b, se, p = float(np.squeeze(tm.effect)) * LN2, float(np.squeeze(tm.sd)) * LN2, float(np.squeeze(tm.pvalue))
            b2, se2, p2 = float(np.squeeze(ts.effect)) * LN2, float(np.squeeze(ts.sd)) * LN2, float(np.squeeze(ts.pvalue))
            dif, pd_ = float(np.squeeze(td.effect)) * LN2, float(np.squeeze(td.pvalue))
            l1.append(f"{fnum(b, 2)}{star(p)} / {fnum(b2, 2)}{star(p2)}")
            l2.append(f"diff. {fnum(dif, 2)} ($p$={fnum(pd_, 3)})")
            rows.append(dict(spec=lab, y=y, minima=b, scored=b2, diff=dif, diff_p=pd_, n=int(res.nobs)))
            if lab.startswith("(3)"):
                put_number(f"scale diff {y}", dif, ".2")
                put_number(f"scale p {y}", pd_, ".3")
                put_number(f"scale minima {y}", b, ".2")
                put_number(f"scale scored {y}", b2, ".2")
            if lab.startswith("(2)"):
                put_number(f"scalecubic diff {y}", dif, ".2")
                put_number(f"scalecubic minima {y}", b, ".2")
                put_number(f"scalecubic scored {y}", b2, ".2")
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    pd.DataFrame(rows).to_csv(TABLES / "scale_check.csv", index=False)
    header = "& Rebate & Any addition & Net saving"
    notes = ("Pooled regressions over the three price-competitive regimes with regime-specific participation slopes: each cell shows the average marginal effect (pp) of doubling bidders in m\'inima cuant\'ia / in the scored regimes, "
             "averaged over the observed distribution of budgets within each regime, and the test of equality of the two effects. All columns include year, UNSPSC-4 and entity-order fixed effects and $\ln$ duration. "
             "In (2)--(3) $\ln B$ is centred at the median of the common support and enters as a polynomial (cubic in (2), linear in (3)) with regime-specific coefficients, interacted with $\ln N$ by regime; the slopes are not evaluated at $\ln B=0$. "
             "Column (3) keeps contracts whose reference budget lies between the 1st percentile of the scored regimes and the 99th percentile of m\'inima cuant\'ia. "
             "Standard errors clustered by entity. * $p<0.10$, ** $p<0.05$, *** $p<0.01$.")
    write_table("tab_scale", chr(10).join(body), "", "tab:scale", notes, "lccc", header)


def broad_sample() -> pd.DataFrame:
    """Broad SECOP II works universe prepared as in r4_broad (re-used by several checks)."""
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
    w["add_any_pp"] = 100 * (w["add_ratio_raw"] > 0.001).astype(float)
    w["add_big_pp"] = 100 * (w["add_ratio_raw"] > 0.10).astype(float)
    w["rebate_pp"] = 100 * (1 - w["award_ratio"].clip(upper=1.0))
    w["final_ratio_w"] = (w["valor_contrato"] / w["precio_base"]).clip(upper=2.0)
    w["net_saving_pp"] = 100 * (1 - w["final_ratio_w"])
    w["ln_base"] = np.log(w["precio_base"])
    dur = w["duracion"].astype(float) * np.select([w.unidad_duracion.fillna("").str.lower().str.startswith("m"),
                                                    w.unidad_duracion.fillna("").str.lower().str.startswith("d"),
                                                    w.unidad_duracion.fillna("").str.lower().str.startswith("a")], [30.0, 1.0, 365.0], default=np.nan)
    w["plazo_dias"] = dur
    w["ln_plazo_raw"] = np.log(dur.where(dur > 0))
    w["plazo_miss"] = w["ln_plazo_raw"].isna().astype(float)
    w["ln_plazo"] = w["ln_plazo_raw"].fillna(w["ln_plazo_raw"].median())
    w["unspsc4"] = w["unspsc4"].astype(str)
    w["orden"] = w["orden_entidad"].fillna("NA")
    w["territorial"] = (w["orden_entidad"].fillna("NA").str.lower() != "nacional").astype(float)
    w["year"] = w["year"].astype(str)
    w["ent_g"] = pd.factorize(w["nit_entidad"])[0]
    return w


def order_het(df: pd.DataFrame) -> None:
    """Award rule x institutional capacity: participation slopes by regime and order of the entity (national vs territorial)."""
    d = prepare_part(df).copy()
    d["territorial"] = (d["orden"] != "A").astype(float)
    w = broad_sample()
    rows, body = [], []
    ctrl = ["ln_base", "ln_plazo", "plazo_miss"]
    for label, data, fe in [("Broad works universe", w, ["year", "unspsc4"]), ("Frame sample (SECOP II works)", d, ["year", "unspsc4"])]:
        plabel = "Panel A. Broad SECOP II works universe" if label.startswith("Broad") else "Panel B. Transport-infrastructure frame (SECOP II works)"
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{plabel}}}}} \\\\")
        for g, gl in [("minima", "M\\'inima cuant\\'ia"), ("scored", "Scored regimes")]:
            sub = data[data.regime == "minima"] if g == "minima" else data[data.regime.isin(["menor", "licitacion"])]
            sub = sub.copy()
            sub["lnb_terr"] = sub["ln_bidders"] * sub["territorial"]
            fe_g = fe + (["regime"] if g == "scored" else [])
            for oname, flag in [("National entities", 0.0), ("Territorial entities", 1.0)]:
                l1, l2 = [f"{gl}: {oname}"], [""]
                for y in ["rebate_pp", "add_any_pp", "net_saving_pp"]:
                    res = ols_cluster(sub, y, ["ln_bidders", "lnb_terr", "territorial"] + ctrl, fe_g)
                    if flag == 0.0:
                        b, se, pv = res.params["ln_bidders"], res.bse["ln_bidders"], res.pvalues["ln_bidders"]
                    else:
                        tt = res.t_test("ln_bidders + lnb_terr = 0")
                        b, se, pv = float(np.squeeze(tt.effect)), float(np.squeeze(tt.sd)), float(tt.pvalue)
                    pdiff = float(res.pvalues["lnb_terr"])
                    c1, c2 = coef_cell(b * LN2, se * LN2, pv, 2)
                    l1.append(c1)
                    l2.append(c2 if flag == 0.0 else c2 + f" [diff. $p$={pdiff:.3f}]")
                    rows.append(dict(sample=label, regime=g, order=oname, y=y, b=b * LN2, se=se * LN2, p=pv, p_diff=pdiff))
                    key = "frame" if label.startswith("Frame") else "broad"
                    put_number(f"ord {key} {g} {'nat' if flag == 0 else 'terr'} {y}", b * LN2, ".2")
                    put_number(f"ordp {key} {g} {'nat' if flag == 0 else 'terr'} {y}", pv, ".3")
                    put_number(f"orddiffp {key} {g} {y}", pdiff, ".3")
                n = int(((sub.territorial == flag)).sum())
                l1.append(fint(n))
                body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    pd.DataFrame(rows).to_csv(TABLES / "order_heterogeneity.csv", index=False)
    header = "& Rebate & Any addition & Net saving & $N$"
    notes = ("Effect (pp) of doubling bidders, estimated within each regime group with a participation slope that is allowed to differ for territorial entities (order other than national); "
             "baseline controls and fixed effects. The bracket gives the $p$-value of the difference between territorial and national slopes. Standard errors clustered by entity. * $p<0.10$, ** $p<0.05$, *** $p<0.01$.")
    write_table("tab_order", "\n".join(body), "", "tab:order", notes, "lcccc", header, size="\\scriptsize")


def threshold_diag(df_unused) -> None:
    """Legal value thresholds (in monthly minimum wages) and manipulation of the running variable around them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import SMMLV, PAPER_FIGURES
    w = broad_sample()
    w["smmlv"] = w["precio_base"] / w["year"].astype(int).map(SMMLV)
    lo, hi = w[(w.smmlv > 85) & (w.smmlv <= 100)], w[(w.smmlv > 100) & (w.smmlv <= 115)]
    below = int(((w.smmlv > 90) & (w.smmlv <= 100)).sum())
    inner = int(((w.smmlv > 80) & (w.smmlv <= 90)).sum())
    above = int(((w.smmlv > 100) & (w.smmlv <= 110)).sum())
    put_number("thr below", below, ",d")
    put_number("thr inner", inner, ",d")
    put_number("thr above", above, ",d")
    put_number("thr minima max", float(w.loc[w.regime == "minima", "smmlv"].quantile(0.995)), ".0f")
    share_min_below = float((w.loc[(w.smmlv > 90) & (w.smmlv <= 100), "regime"] == "minima").mean()) * 100
    share_min_above = float((w.loc[(w.smmlv > 100) & (w.smmlv <= 110), "regime"] == "minima").mean()) * 100
    put_number("thr share below", share_min_below, ".0f")
    put_number("thr share above", share_min_above, ".0f")
    bins = np.concatenate([np.arange(0, 100, 5), np.arange(100, 205, 5)])
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    cols = {"minima": "#0072B2", "menor": "#E69F00", "licitacion": "#009E73"}
    bottoms = np.zeros(len(bins) - 1)
    for g in ["minima", "menor", "licitacion"]:
        h, _ = np.histogram(w.loc[w.regime == g, "smmlv"].clip(upper=199.9), bins=bins)
        ax.bar(bins[:-1], h, width=4.6, bottom=bottoms, align="edge", color=cols[g], label={"minima": "M\u00ednima cuant\u00eda", "menor": "Menor cuant\u00eda", "licitacion": "Licitaci\u00f3n"}[g])
        bottoms += h
    ax.axvline(100, color="black", ls="--", lw=0.9)
    ax.set_xlabel("Reference budget in monthly minimum wages (SMMLV of the signing year)")
    ax.set_ylabel("Number of contracts")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PAPER_FIGURES / "fig_threshold.pdf")
    plt.close(fig)


def indexation_check(df: pd.DataFrame) -> None:
    """Could price adjustments (reajustes) on long contracts mechanically inflate additions in scored regimes?"""
    d = prepare_part(df).copy()
    d["add_big_pp"] = 100 * (d["add_ratio_raw"] > 0.10).astype(float)
    d["plazo_d"] = d["plazo_proc_dias"]
    rows, body = [], []
    specs = [("Baseline", d, FE_PART, "add_any_pp"),
             ("Year $\\times$ $\\ln$ duration interactions", d, FE_PART, "add_any_pp"),
             ("Planned duration $\\leq$ 180 days", d[d.plazo_d <= 180], FE_PART, "add_any_pp"),
             ("Large additions only ($>$10\\% of awarded value)", d, FE_PART, "add_big_pp")]
    d["yd"] = d["year"].astype(str)
    for si, (nm, dd, fe, y) in enumerate(specs):
        l1, l2 = [nm], [""]
        for g in ["minima", "scored"]:
            sub = sub_group(dd, g).copy()
            extra = []
            if nm.startswith("Year"):
                for yy in sorted(sub["year"].unique())[1:]:
                    sub[f"lp_{yy}"] = sub["ln_plazo"] * (sub["year"] == yy).astype(float)
                    extra.append(f"lp_{yy}")
            fe_g = fe + (["regime"] if g == "scored" else [])
            res = ols_cluster(sub, y, ["ln_bidders"] + CONTROLS_PART + extra, fe_g)
            b, se, pv = per_doubling(res)
            c1, c2 = coef_cell(b, se, pv, 2)
            l1.append(c1)
            l2.append(c2)
            rows.append(dict(spec=nm, group=g, y=y, b=b, se=se, p=pv, n=int(res.nobs)))
            put_number(f"idx {g} {['base','inter','short','big'][si]}", b, ".2")
        body += [" & ".join(l1) + " \\\\", " & ".join(l2) + " \\\\"]
    pd.DataFrame(rows).to_csv(TABLES / "indexation_check.csv", index=False)
    header = "& M\\'inima cuant\\'ia & Scored regimes"
    notes = ("Effect (pp) of doubling bidders on the probability of an addition (last row: of an addition larger than 10\\% of the awarded value, which price adjustments on the contract durations observed here would rarely reach). "
             "Baseline controls. Standard errors clustered by entity.")
    write_table("tab_index", "\n".join(body), "", "tab:index", notes, "lcc", header)


def main() -> None:
    import sys
    only = set(sys.argv[1:])            # optional: run a subset, e.g. `python 08_robustness.py iv scale`
    df = pd.read_parquet(ANALYSIS_PQ)
    steps = [("grid", r1_grid), ("wild", r2_wild), ("placebo", r3_placebo), ("broad", r4_broad), ("oster", oster),
             ("iv", iv_ar), ("scale", scale_check), ("order", order_het), ("threshold", threshold_diag), ("index", indexation_check)]
    for name, fn in steps:
        if not only or name in only:
            fn(df)
    write_numbers_tex()
    print("robustness done")


if __name__ == "__main__":
    main()
