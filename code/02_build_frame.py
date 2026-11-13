"""STEP 02 - Load and clean the contract frame (marco1.xlsx).

The frame is the stratified sampling frame of road/transport-infrastructure contracts built with
`estratos3.R` (keyword classification + UNSPSC codes, SECOP I and II, 2018-2024, closed contracts).

Output: data/interim/frame_marco1.parquet, output/logs/02_build_frame.log (data-quality audit)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FRAME_PQ, LOGS, raw_frame_xlsx
from utils import clean_nit, norm_text, tipo_cat


def parse_date(s: pd.Series) -> pd.Series:
    a = pd.to_datetime(s, format="%m-%d-%Y", errors="coerce")
    b = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    return a.fillna(b)


def main() -> None:
    raw = pd.read_excel(raw_frame_xlsx(), dtype=str)
    log = [f"rows read from marco1.xlsx: {len(raw):,}"]

    df = pd.DataFrame(index=raw.index)
    df["id_contrato"] = raw["ID_Contrato"].str.strip()
    df["plataforma"] = np.where(raw["Plataforma"].str.contains("1"), "SECOP I", "SECOP II")
    df["fecha_firma"] = parse_date(raw["FechaFirma"])
    df["ano"] = pd.to_numeric(raw["AnoFirma"], errors="coerce").astype("Int64")
    df["departamento"] = raw["Departamento"].map(norm_text)
    df["municipio"] = raw["Municipio"].map(norm_text)
    df["entidad"] = raw["Entidad"].map(norm_text)
    df["tipo_contrato"] = raw["TipoContrato"].map(tipo_cat)
    df["unspsc6"] = raw["UNSPC"].str.strip().str[:6]
    df["unspsc4"] = df["unspsc6"].str[:4]
    df["pyme"] = raw["PYME"].str.strip().str.lower().map({"si": 1, "no": 0})
    df["nit"] = raw["NIT"].map(clean_nit)
    df["proveedor"] = raw["Proveedor"].map(norm_text)
    df["es_consorcio"] = df["proveedor"].fillna("").str.contains(r"\b(CONSORCIO|UNION TEMPORAL|U\.?T\.?|ASOCIACION|PROMESA)\b").astype(int)
    df["estado"] = raw["EsdatoProceso"].str.strip().str.upper()
    df["orden"] = raw["Orden"].str.strip()            # A = national, B = capital-city entity, C = other territorial
    df["modalidad_marco"] = raw["Modalidad"].str.strip()
    df["estrato"] = pd.to_numeric(raw["estrato"], errors="coerce")
    df["en_muestra"] = raw["en_muestra"].str.upper().eq("TRUE").astype(int)
    df["objeto"] = raw["Objeto"].str.strip()

    # money (COP, nominal)
    df["valor_inicial"] = pd.to_numeric(raw["Valor"], errors="coerce")
    df["valor_adicion_s1"] = pd.to_numeric(raw["ValorAdicion"], errors="coerce")   # only SECOP I reports it

    # planned duration in days
    pv = pd.to_numeric(raw["PlazoValor"], errors="coerce")
    unit = raw["PlazoMedida"].fillna("").str.lower()
    mult = np.where(unit.str.startswith("m"), 30.0, np.where(unit.str.startswith("d"), 1.0, np.nan))
    df["plazo_dias"] = pv * mult
    df.loc[df["plazo_dias"] <= 0, "plazo_dias"] = np.nan

    # time extension in days: SECOP I reports days and months separately; SECOP II 'Dias adicionados'
    d = pd.to_numeric(raw["ProrrogaDias"], errors="coerce")
    mth = pd.to_numeric(raw["ProrrogaMeses"], errors="coerce")
    df["prorroga_dias"] = d
    df["prorroga_meses"] = mth
    df["prorroga_total_dias"] = d.fillna(0) + 30.0 * mth.fillna(0)

    # ---- data-quality audit ------------------------------------------------------------
    log.append(f"duplicated id_contrato: {df['id_contrato'].duplicated().sum()}")
    log.append("platform counts: " + df["plataforma"].value_counts().to_dict().__repr__())
    log.append(f"missing NIT: {df['nit'].isna().sum()}  ({df['nit'].isna().mean():.1%})")
    log.append(f"valor_inicial <= 0: {(df['valor_inicial'] <= 0).sum()}")
    log.append(f"year range: {df['ano'].min()}-{df['ano'].max()}")
    log.append("tipo_contrato: " + df["tipo_contrato"].value_counts().to_dict().__repr__())
    log.append(f"SECOP II rows with value additions reported by the platform: "
               f"{df.loc[df.plataforma == 'SECOP II', 'valor_adicion_s1'].notna().sum()}  (all NaN by construction)")

    df.to_parquet(FRAME_PQ, index=False)
    (LOGS / "02_build_frame.log").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
