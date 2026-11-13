"""STEP 05 - Assemble the analysis dataset, define outcomes and sample flags, write data dictionary.

Output: data/processed/analysis_contracts.parquet (+ .csv)
        docs/data_dictionary.md
        output/tables/sample_flow.csv   (how many contracts survive each filter)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (ANALYSIS_CSV, ANALYSIS_PQ, FRAME_PQ, INTERIM, LOGS, PROJECT_DIR, TABLES,
                    YEAR_MAX, YEAR_MIN)

ADD_THRESHOLD = 0.001
CAP_LO, CAP_HI = 0.49, 0.5001     # window around the legal ceiling (+50 % of the initial value)


def regime_from_modalidad(m: str | None) -> str | float:
    if not isinstance(m, str):
        return np.nan
    s = m.lower()
    if "nima cuant" in s:
        return "minima"
    if "menor cuant" in s:
        return "menor"
    if "concurso" in s:
        return "concurso"
    if "licitaci" in s:
        return "licitacion"
    return "otra"


def duration_days(val, unit) -> pd.Series:
    u = unit.fillna("").str.lower().str.strip()
    mult = np.select([u.str.startswith("m"), u.str.startswith("d"), u.str.startswith("a"), u.str.startswith("s")],
                     [30.0, 1.0, 365.0, 7.0], default=np.nan)
    return val.astype(float) * mult


def main() -> None:
    frame = pd.read_parquet(FRAME_PQ)
    feats = pd.read_parquet(INTERIM / "features_frame.parquet")
    links = pd.read_parquet(INTERIM / "secop2_frame_links.parquet")

    df = frame.merge(feats, on="id_contrato", how="left")

    # ----- SECOP II link columns --------------------------------------------------------------
    lk = links[["id_contrato", "nit_entidad", "proceso_compra", "modalidad", "valor_contrato", "valor_adjudicado",
                "precio_base", "prov_unicos_resp", "prov_invitados", "n_lotes", "n_award_rows", "multi_award",
                "duracion", "unidad_duracion", "fecha_publicacion", "fecha_adjudicacion", "fecha_inicio",
                "fecha_fin", "dias_adicionados", "estado_contrato"]].copy()
    lk = lk.rename(columns={"modalidad": "modalidad_raw", "valor_contrato": "valor_contrato_s2",
                            "duracion": "dur_proc", "unidad_duracion": "dur_unidad",
                            "dias_adicionados": "dias_adic_s2"})
    df = df.merge(lk, on="id_contrato", how="left")

    s2 = df["plataforma"] == "SECOP II"
    s1 = ~s2

    # ----- initial value ---------------------------------------------------------------------
    df["valor_ini"] = np.where(s2 & (df["valor_adjudicado"] > 0), df["valor_adjudicado"], df["valor_inicial"])
    df.loc[df["valor_ini"] <= 0, "valor_ini"] = np.nan
    df["ln_valor"] = np.log(df["valor_ini"])

    # ----- renegotiation outcome (comparable across platforms) -------------------------------
    r1 = df["valor_adicion_s1"] / df["valor_inicial"].where(df["valor_inicial"] > 0)
    r2 = df["valor_contrato_s2"] / df["valor_adjudicado"].where(df["valor_adjudicado"] > 0) - 1
    ok2 = s2 & (df["multi_award"] == 0) & df["valor_adjudicado"].gt(0)
    df["add_ratio_raw"] = np.where(s1, r1, np.where(ok2, r2, np.nan))
    df["add_ratio"] = df["add_ratio_raw"].clip(lower=0, upper=1.0)          # negatives -> 0; >100 % treated as errors
    df["add_any"] = np.where(df["add_ratio_raw"].notna(), (df["add_ratio_raw"] > ADD_THRESHOLD).astype(float), np.nan)
    df["add_cap"] = np.where(df["add_ratio_raw"].notna(),
                             ((df["add_ratio_raw"] >= CAP_LO) & (df["add_ratio_raw"] <= CAP_HI)).astype(float), np.nan)
    df["ln_add"] = np.log1p(df["add_ratio"])
    df["add_value_cop"] = np.where(s1, df["valor_adicion_s1"], np.where(ok2, df["valor_contrato_s2"] - df["valor_adjudicado"], np.nan))

    # ----- participation / competition variables (SECOP II only) -----------------------------
    df["regime"] = df["modalidad_raw"].map(regime_from_modalidad)
    df["n_bidders"] = df["prov_unicos_resp"].where(df["prov_unicos_resp"] > 0)
    df["ln_bidders"] = np.log(df["n_bidders"])
    ratio_ab = df["valor_adjudicado"] / df["precio_base"].where(df["precio_base"] > 0)
    df["award_ratio"] = ratio_ab
    df["rebate"] = 1 - ratio_ab.clip(upper=1.0)
    df["final_ratio"] = df["valor_contrato_s2"] / df["precio_base"].where(df["precio_base"] > 0)
    df["final_ratio_w"] = df["final_ratio"].clip(upper=2.0)
    df["ln_base"] = np.log(df["precio_base"].where(df["precio_base"] > 0))
    df["plazo_proc_dias"] = duration_days(df["dur_proc"], df["dur_unidad"])
    df["ln_dens"] = np.log1p(df["dens_firms_365"])
    df["ln_cong"] = np.log1p(df["congestion_30d"])
    df["ln_vol_exp"] = np.log1p(df["vol_exp"])
    df["ln_spec_exp"] = np.log1p(df["spec_exp"])
    df["ent_soft"] = df["ent_prior_addrate"].where(df["ent_prior_n"] >= 5)
    df["ent_id"] = np.where(s2, df["nit_entidad"].fillna(df["entidad"]), df["entidad"])
    df["firm_id"] = df["nit"]

    # ----- sample flags -----------------------------------------------------------------------
    flow = []

    def step(label: str, mask: pd.Series) -> pd.Series:
        flow.append((label, int(mask.sum())))
        return mask

    m = step("Contracts in the sampling frame", pd.Series(True, index=df.index))
    m = step("  signed 2018-2024", m & df["ano"].between(YEAR_MIN, YEAR_MAX))
    m = step("  works, supervision or consulting contracts (drop service/supply/other)",
             m & df["tipo_contrato"].isin(["OBRA", "INTERVENTORIA", "CONSULTORIA"]))
    m = step("  positive initial value", m & df["valor_ini"].notna())
    m_out = step("  renegotiation outcome measurable (SECOP I reported; SECOP II single-award link)", m & df["add_ratio_raw"].notna())
    df["s_outcome"] = m_out.astype(int)
    m_firm = step("  winner identified (NIT available)  -> firm sample", m_out & df["nit"].notna())
    df["s_firm"] = m_firm.astype(int)

    p = (s2 & df["tipo_contrato"].eq("OBRA") & df["regime"].isin(["minima", "menor", "licitacion"]))
    flow.append(("SECOP II works contracts (price-competitive regimes)", int(p.sum())))
    p = p & df["add_ratio_raw"].notna()
    flow.append(("  single-award link with measurable renegotiation", int(p.sum())))
    p = p & df["n_bidders"].notna() & df["precio_base"].gt(0)
    flow.append(("  bidders observed and reference budget > 0", int(p.sum())))
    p_final = p & df["award_ratio"].between(0.30, 1.0001)
    flow.append(("  award/reference-budget ratio in [0.30, 1.00]  -> participation sample", int(p_final.sum())))
    df["s_part"] = p_final.astype(int)
    c = (s2 & df["regime"].eq("concurso") & df["add_ratio_raw"].notna() & df["n_bidders"].notna() & df["precio_base"].gt(0))
    df["s_concurso"] = c.astype(int)
    flow.append(("SECOP II merit-based contests (placebo sample)", int(c.sum())))

    flow_df = pd.DataFrame(flow, columns=["step", "n"])
    flow_df.to_csv(TABLES / "sample_flow.csv", index=False)

    # time variables
    df["year"] = df["ano"].astype(float)
    df["firma_ym"] = df["fecha_firma"].dt.to_period("M").astype(str)

    keep_cols = [c for c in df.columns if c not in ("objeto",)]
    df[keep_cols].to_parquet(ANALYSIS_PQ, index=False)
    df[keep_cols].to_csv(ANALYSIS_CSV, index=False, encoding="utf-8-sig")
    print(flow_df.to_string(index=False))
    print(df.loc[df.s_part == 1, ["n_bidders", "rebate", "final_ratio_w", "add_ratio", "add_any", "ent_soft"]].describe().T)
    (LOGS / "05_assemble_analysis.log").write_text(flow_df.to_string(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
