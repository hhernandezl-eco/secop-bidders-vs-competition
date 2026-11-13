"""STEP 04 - Firm, entity and market features (all measured strictly *before* the focal contract).

Features
--------
Firm capabilities (focal firm = contract winner, identified by NIT/ID number)
  vol_exp        number of contracts previously signed by the firm (any object) in SECOP II (all
                 4.7 M contracts) plus the SECOP I contracts of the frame             [volume of experience]
  spec_exp       same, restricted to the same UNSPSC 6-digit class                    [specific experience]
  hist_mod       share of the firm's earlier *completed* works contracts (with measurable outcome) that
                 received an addition                                                  [track record, ex ante]
  hist_mod_signed same share over earlier *signed* contracts (the working-paper definition; it uses
                 information that may not yet exist when the focal contract is signed)
Entity enforcement regime (SECOP II sample)
  ent_prior_n, ent_prior_addrate   number / addition rate of the entity's earlier completed works contracts
Market instruments for participation (SECOP II sample)
  dens_firms_365   distinct firms awarded works contracts in the same UNSPSC-4 x department during the
                   365 days before the process was published           (supply-side 'market depth')
  congestion_30d   works processes published by *other* entities in the same department within +-30 days
                   (competing demand for the same contractors' capacity)

Input : frame_marco1.parquet, secop2_*.parquet
Output: data/interim/features_frame.parquet
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from config import (FRAME_PQ, INTERIM, LOGS, SECOP2_CONTRACTS_PQ, SECOP2_PROCESSES_PQ)
from utils import clean_nit

ADD_THRESHOLD = 0.001     # an addition counts when it exceeds 0.1 % of the initial/awarded value


def q(p) -> str:
    return str(p).replace("\\", "/")


# --------------------------------------------------------------------------------------------
def prior_rates(focal: pd.DataFrame, panel: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """For every focal row, statistics over panel rows of the same `key` that happened earlier.

    focal : columns [key, t]  (t = signing date of the focal contract)
    panel : columns [key, sign, end, y]   (y = 1 if the contract had an addition, NaN-free)
    Returns n/rate over contracts *completed* (end <= t) and over contracts *signed* (sign < t).
    """
    out = pd.DataFrame(index=focal.index, columns=[f"{prefix}_n_completed", f"{prefix}_rate_completed",
                                                    f"{prefix}_n_signed", f"{prefix}_rate_signed"], dtype=float)
    pgroups = {k: g for k, g in panel.groupby(key)}
    for k, g in focal.groupby(key):
        if k not in pgroups:
            out.loc[g.index, [f"{prefix}_n_completed", f"{prefix}_n_signed"]] = 0.0
            continue
        p = pgroups[k]
        # completed
        pe = p.sort_values("end")
        ends = pe["end"].values.astype("datetime64[ns]")
        cum_e = np.concatenate([[0.0], np.cumsum(pe["y"].values)])
        # signed
        ps = p.sort_values("sign")
        signs = ps["sign"].values.astype("datetime64[ns]")
        cum_s = np.concatenate([[0.0], np.cumsum(ps["y"].values)])
        t = g["t"].values.astype("datetime64[ns]")
        ie = np.searchsorted(ends, t, side="right")      # end <= t
        is_ = np.searchsorted(signs, t, side="left")     # sign < t
        out.loc[g.index, f"{prefix}_n_completed"] = ie
        out.loc[g.index, f"{prefix}_rate_completed"] = np.where(ie > 0, cum_e[ie] / np.maximum(ie, 1), np.nan)
        out.loc[g.index, f"{prefix}_n_signed"] = is_
        out.loc[g.index, f"{prefix}_rate_signed"] = np.where(is_ > 0, cum_s[is_] / np.maximum(is_, 1), np.nan)
    return out


def main() -> None:
    log: list[str] = []
    frame = pd.read_parquet(FRAME_PQ)
    links = pd.read_parquet(INTERIM / "secop2_frame_links.parquet")
    works = pd.read_parquet(INTERIM / "secop2_works_universe.parquet")

    # ---------------- outcome panel (firm track record) ------------------------------------
    s1 = frame.loc[(frame.plataforma == "SECOP I") & (frame.valor_inicial > 0) & frame.valor_adicion_s1.notna()].copy()
    s1["ratio"] = s1.valor_adicion_s1 / s1.valor_inicial
    plazo = s1["plazo_dias"].fillna(365.0)
    s1["end"] = s1["fecha_firma"] + pd.to_timedelta(plazo + s1["prorroga_total_dias"], unit="D")
    panel_s1 = pd.DataFrame({"nit": s1["nit"], "sign": s1["fecha_firma"], "end": s1["end"],
                             "y": (s1["ratio"] > ADD_THRESHOLD).astype(float), "src": "S1"})

    w = works.loc[(works.outcome_ok == 1) & works.nit_proveedor.notna()].copy()
    w["nit"] = w["nit_proveedor"].map(clean_nit)
    w["ratio"] = w["add_ratio_raw"]
    # planned end: contract end date reported by SECOP II (updated when the term is extended)
    w["end"] = w["fecha_fin"].fillna(w["fecha_firma"] + pd.Timedelta(days=365))
    panel_s2 = pd.DataFrame({"nit": w["nit"], "sign": pd.to_datetime(w["fecha_firma"]),
                             "end": pd.to_datetime(w["end"]), "y": (w["ratio"] > ADD_THRESHOLD).astype(float),
                             "src": "S2"})
    panel = pd.concat([panel_s1, panel_s2], ignore_index=True).dropna(subset=["nit", "sign", "end"])
    panel["end"] = np.maximum(panel["end"].values.astype("datetime64[ns]"),
                              (panel["sign"] + pd.Timedelta(days=1)).values.astype("datetime64[ns]"))
    log.append(f"firm track-record panel: {len(panel):,} contracts ({(panel.src == 'S1').sum():,} SECOP I + "
               f"{(panel.src == 'S2').sum():,} SECOP II works), {panel.nit.nunique():,} firms")

    focal = frame[["id_contrato", "nit", "fecha_firma"]].rename(columns={"fecha_firma": "t"}).copy()
    focal = focal.dropna(subset=["nit", "t"])
    hist = prior_rates(focal.set_index("id_contrato"), panel, "nit", "firm")
    hist = hist.rename(columns={"firm_n_completed": "hist_n_completed", "firm_rate_completed": "hist_mod",
                                "firm_n_signed": "hist_n_signed", "firm_rate_signed": "hist_mod_signed"})
    hist = hist.reset_index().rename(columns={"index": "id_contrato"})

    # ---------------- volume / specific experience via DuckDB -------------------------------
    con = duckdb.connect()
    tmp = INTERIM / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{q(tmp)}'")
    con.execute("SET memory_limit='2500MB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"""
        CREATE TABLE ev2 AS
        SELECT CASE WHEN length(n) = 10 AND substr(n, 1, 1) IN ('8','9') THEN substr(n, 1, 9) ELSE n END AS nit,
               fecha_firma, u6
        FROM (
          SELECT regexp_replace(nit_proveedor, '[^0-9]', '', 'g') AS n, fecha_firma, substr(unspsc_raw, 4, 6) AS u6
          FROM read_parquet('{q(SECOP2_CONTRACTS_PQ)}')
          WHERE nit_proveedor IS NOT NULL AND fecha_firma IS NOT NULL
        ) WHERE n <> ''
    """)
    ev1 = frame.loc[frame.plataforma == "SECOP I", ["nit", "fecha_firma", "unspsc6"]].dropna(subset=["nit", "fecha_firma"])
    ev1 = ev1.rename(columns={"unspsc6": "u6"})
    con.register("ev1", ev1)
    con.execute("INSERT INTO ev2 SELECT nit, CAST(fecha_firma AS DATE), u6 FROM ev1")
    f_in = frame[["id_contrato", "nit", "fecha_firma", "unspsc6"]].dropna(subset=["nit", "fecha_firma"])
    con.register("f_in", f_in)
    exp = con.execute("""
        SELECT f.id_contrato,
               count(e.fecha_firma)                                   AS vol_exp,
               count(e.fecha_firma) FILTER (WHERE e.u6 = f.unspsc6)   AS spec_exp
        FROM f_in f LEFT JOIN ev2 e ON e.nit = f.nit AND e.fecha_firma < CAST(f.fecha_firma AS DATE)
        GROUP BY f.id_contrato
    """).fetchdf()
    log.append(f"experience counts computed for {len(exp):,} focal contracts")

    # ---------------- entity enforcement (SECOP II works) -----------------------------------
    we = works.loc[(works.outcome_ok == 1) & works.nit_entidad.notna()].copy()
    pan_e = pd.DataFrame({"nit_entidad": we["nit_entidad"], "sign": pd.to_datetime(we["fecha_firma"]),
                          "end": pd.to_datetime(we["fecha_fin"].fillna(we["fecha_firma"] + pd.Timedelta(days=365))),
                          "y": (we["add_ratio_raw"] > ADD_THRESHOLD).astype(float)}).dropna()
    pan_e["end"] = np.maximum(pan_e["end"].values.astype("datetime64[ns]"),
                              (pan_e["sign"] + pd.Timedelta(days=1)).values.astype("datetime64[ns]"))
    lk = links.copy()
    lk["t"] = pd.to_datetime(lk["fecha_firma"])
    ent = prior_rates(lk.set_index("id_contrato")[["nit_entidad", "t"]].dropna(), pan_e, "nit_entidad", "ent")
    ent = ent.rename(columns={"ent_n_completed": "ent_prior_n", "ent_rate_completed": "ent_prior_addrate",
                              "ent_n_signed": "ent_prior_n_signed", "ent_rate_signed": "ent_prior_addrate_signed"})
    ent = ent.reset_index().rename(columns={"index": "id_contrato"})
    log.append(f"entity panel: {len(pan_e):,} closed works contracts, {pan_e.nit_entidad.nunique():,} entities")

    # ---------------- market instruments ----------------------------------------------------
    lk2 = lk[["id_contrato", "nit_entidad", "departamento", "unspsc4", "fecha_publicacion", "fecha_firma"]].copy()
    lk2["t0"] = pd.to_datetime(lk2["fecha_publicacion"]).fillna(pd.to_datetime(lk2["fecha_firma"]) - pd.Timedelta(days=45))
    con.register("lk2", lk2.assign(t0=lk2["t0"].dt.date))
    con.execute(f"""
        CREATE TABLE wk AS
        SELECT CASE WHEN length(n) = 10 AND substr(n, 1, 1) IN ('8','9') THEN substr(n, 1, 9) ELSE n END AS nit,
               departamento, substr(unspsc_raw, 4, 4) AS u4, fecha_firma
        FROM (SELECT regexp_replace(nit_proveedor, '[^0-9]', '', 'g') AS n, departamento, unspsc_raw, fecha_firma
              FROM read_parquet('{q(SECOP2_CONTRACTS_PQ)}')
              WHERE substr(unspsc_raw, 4, 2) IN ('72', '95') AND nit_proveedor IS NOT NULL AND fecha_firma IS NOT NULL)
        WHERE n <> ''
    """)
    dens = con.execute("""
        SELECT l.id_contrato, count(DISTINCT w.nit) AS dens_firms_365
        FROM lk2 l LEFT JOIN wk w
          ON w.departamento = l.departamento AND w.u4 = l.unspsc4
         AND w.fecha_firma >= l.t0 - INTERVAL 365 DAY AND w.fecha_firma < l.t0
        GROUP BY l.id_contrato
    """).fetchdf()
    con.execute(f"""
        CREATE TABLE wp AS
        SELECT DISTINCT id_portafolio, nit_entidad, departamento, fecha_publicacion
        FROM read_parquet('{q(SECOP2_PROCESSES_PQ)}')
        WHERE substr(unspsc_raw, 4, 2) IN ('72', '95') AND fecha_publicacion IS NOT NULL
    """)
    cong = con.execute("""
        SELECT l.id_contrato, count(DISTINCT p.id_portafolio) AS congestion_30d
        FROM lk2 l LEFT JOIN wp p
          ON p.departamento = l.departamento
         AND p.fecha_publicacion BETWEEN l.t0 - INTERVAL 30 DAY AND l.t0 + INTERVAL 30 DAY
         AND p.nit_entidad IS DISTINCT FROM l.nit_entidad
        GROUP BY l.id_contrato
    """).fetchdf()
    log.append(f"market instruments computed for {len(dens):,} SECOP II contracts")

    # ---------------- bidding-window calendar (exploratory instrument) -----------------------
    from utils import co_holidays, window_calendar
    hol = set()
    for yy in range(2016, 2026):
        hol |= co_holidays(yy)
    cal = pd.DataFrame(
        [window_calendar(s, e, hol) for s, e in zip(pd.to_datetime(links["fecha_publicacion"]),
                                                    pd.to_datetime(links["fecha_recepcion_resp"]))],
        columns=["win_cal_days", "win_weekend_days", "win_holiday_days"])
    cal["id_contrato"] = links["id_contrato"].values
    log.append(f"bidding-window calendar computed for {cal['win_cal_days'].notna().sum():,} contracts")

    out = (frame[["id_contrato"]].merge(hist, on="id_contrato", how="left")
           .merge(cal, on="id_contrato", how="left")
           .merge(exp, on="id_contrato", how="left")
           .merge(ent, on="id_contrato", how="left")
           .merge(dens, on="id_contrato", how="left")
           .merge(cong, on="id_contrato", how="left"))
    out.to_parquet(INTERIM / "features_frame.parquet", index=False)
    (LOGS / "04_build_features.log").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print(out.describe().T)


if __name__ == "__main__":
    main()
