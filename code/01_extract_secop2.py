"""STEP 01 - Extract the two raw SECOP II CSV files (~8 GB each) into light Parquet tables.

Input : raw SECOP II 'Contratos Electronicos' and 'Procesos de Contratacion' CSV snapshots
Output: data/interim/secop2_contracts_light.parquet
        data/interim/secop2_processes_light.parquet
        output/logs/01_extract_secop2.log

Only the columns used downstream are kept; money fields ('$1,234,567') and dates ('MM/DD/YYYY')
are parsed into numeric / date types. Nothing is filtered here (the full universe is kept so that
firm-experience and market-density variables can be built later).
"""
from __future__ import annotations

import sys
import time

import duckdb

from config import (LOGS, SECOP2_CONTRACTS_PQ, SECOP2_PROCESSES_PQ, raw_contracts_csv,
                    raw_processes_csv)


def money(col: str) -> str:
    return f"try_cast(replace(replace(replace(\"{col}\", '$', ''), ',', ''), ' ', '') as double)"


def date(col: str) -> str:
    return f"try_strptime(\"{col}\", '%m/%d/%Y')::date"


def intc(col: str) -> str:
    return f"try_cast(replace(\"{col}\", ',', '') as bigint)"


def idc(col: str) -> str:
    """Identifier-like field: strip thousands separators and spaces, keep as text."""
    return f"nullif(replace(replace(trim(\"{col}\"), ',', ''), ' ', ''), '')"


CONTRACT_SQL = f"""
SELECT
  "ID Contrato"                                  AS id_contrato,
  "Proceso de Compra"                            AS proceso_compra,       -- joins to processes."ID del Portafolio"
  {idc('Nit Entidad')}                           AS nit_entidad,
  "Nombre Entidad"                               AS entidad,
  "Departamento"                                 AS departamento,
  "Ciudad"                                       AS ciudad,
  "Orden"                                        AS orden_entidad,
  "Sector"                                       AS sector,
  "Estado Contrato"                              AS estado_contrato,
  "Codigo de Categoria Principal"                AS unspsc_raw,
  "Tipo de Contrato"                             AS tipo_contrato,
  "Modalidad de Contratacion"                    AS modalidad,
  {date('Fecha de Firma')}                       AS fecha_firma,
  {date('Fecha de Inicio del Contrato')}         AS fecha_inicio,
  {date('Fecha de Fin del Contrato')}            AS fecha_fin,
  {idc('Documento Proveedor')}                   AS nit_proveedor,
  "Proveedor Adjudicado"                         AS proveedor,
  "Es Pyme"                                      AS es_pyme,
  {money('Valor del Contrato')}                  AS valor_contrato,
  {money('Valor Pagado')}                        AS valor_pagado,
  {intc('Dias adicionados')}                     AS dias_adicionados,
  "Duración del contrato"                        AS duracion_txt,
  "Origen de los Recursos"                       AS origen_recursos,
  "El contrato puede ser prorrogado"             AS puede_prorrogar,
  "URLProceso"                                   AS url_proceso
FROM read_csv('{{path}}', all_varchar=true, header=true, sample_size=50000,
              ignore_errors=true, strict_mode=false, parallel=false)
"""

PROCESS_SQL = f"""
SELECT
  "ID del Proceso"                               AS id_proceso,
  "ID del Portafolio"                            AS id_portafolio,        -- joins to contracts."Proceso de Compra"
  {idc('Nit Entidad')}                           AS nit_entidad,
  "Entidad"                                      AS entidad,
  "Departamento Entidad"                         AS departamento,
  "Ciudad Entidad"                               AS ciudad,
  "OrdenEntidad"                                 AS orden_entidad,
  "Fase"                                         AS fase,
  {date('Fecha de Publicacion del Proceso')}     AS fecha_publicacion,
  {money('Precio Base')}                         AS precio_base,
  "Modalidad de Contratacion"                    AS modalidad,
  {intc('Duracion')}                             AS duracion,
  "Unidad de Duracion"                           AS unidad_duracion,
  {date('Fecha de Recepcion de Respuestas')}     AS fecha_recepcion_resp,
  {date('Fecha de Apertura Efectiva')}           AS fecha_apertura,
  {intc('Proveedores Invitados')}                AS prov_invitados,
  {intc('Proveedores con Invitacion Directa')}   AS prov_invit_directa,
  {intc('Visualizaciones del Procedimiento')}    AS visualizaciones,
  {intc('Proveedores que Manifestaron Interes')} AS prov_interes,
  {intc('Respuestas al Procedimiento')}          AS respuestas,
  {intc('Respuestas Externas')}                  AS respuestas_externas,
  {intc('Conteo de Respuestas a Ofertas')}       AS conteo_resp_ofertas,
  {intc('Proveedores Unicos con Respuestas')}    AS prov_unicos_resp,
  {intc('Numero de Lotes')}                      AS n_lotes,
  "Estado del Procedimiento"                     AS estado_proc,
  "Adjudicado"                                   AS adjudicado,
  {date('Fecha Adjudicacion')}                   AS fecha_adjudicacion,
  {money('Valor Total Adjudicacion')}            AS valor_adjudicado,
  {idc('NIT del Proveedor Adjudicado')}          AS nit_adjudicado,
  "Codigo Principal de Categoria"                AS unspsc_raw,
  "Tipo de Contrato"                             AS tipo_contrato,
  "Estado Resumen"                               AS estado_resumen,
  "URLProceso"                                   AS url_proceso
FROM read_csv('{{path}}', all_varchar=true, header=true, sample_size=50000,
              ignore_errors=true, strict_mode=false, parallel=false)
"""


def run(con: duckdb.DuckDBPyConnection, sql: str, src, dst, log) -> None:
    t0 = time.time()
    path = str(src).replace("\\", "/")
    q = sql.replace("{path}", path)
    con.execute(f"COPY ({q}) TO '{str(dst).replace(chr(92), '/')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{str(dst).replace(chr(92), '/')}')").fetchone()[0]
    msg = f"{src.name} -> {dst.name}: {n:,} rows in {time.time() - t0:,.0f}s"
    print(msg)
    log.write(msg + "\n")
    log.flush()


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")          # machine used for the paper has 8 GB of RAM
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    with open(LOGS / "01_extract_secop2.log", "a", encoding="utf-8") as log:
        which = sys.argv[1] if len(sys.argv) > 1 else "all"
        if which in ("all", "contracts"):
            run(con, CONTRACT_SQL, raw_contracts_csv(), SECOP2_CONTRACTS_PQ, log)
        if which in ("all", "processes"):
            run(con, PROCESS_SQL, raw_processes_csv(), SECOP2_PROCESSES_PQ, log)


if __name__ == "__main__":
    main()
