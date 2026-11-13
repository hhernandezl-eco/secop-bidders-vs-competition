"""STEP 03 - Link SECOP II contracts to their selection process and build renegotiation outcomes.

Why this step exists
--------------------
* SECOP II 'Contratos' reports only 'Dias adicionados' (almost never filled for the contracts in the
  frame) and NO value of additions. We therefore *derive* additions from the contract file:
        addition ratio = (current contract value - awarded value) / awarded value
  where the awarded value comes from the process file (value at award, i.e. before any addition).
  The derived measure is validated in 04/05 by its bunching at the legal ceiling of +50 %.
* The process file contains the number of unique bidders ('Proveedores Unicos con Respuestas'),
  the reference budget ('Precio Base') and the awarded value.

Join keys: contracts."Proceso de Compra" (CO1.BDOS.*) = processes."ID del Portafolio".
A portfolio can have several process rows (phases such as 'Presentacion de observaciones',
'Manifestacion de interes') and several award rows (lots / multiple awardees). We keep only rows with
Adjudicado = 'Si'; contracts whose process has more than one distinct award row are flagged
`multi_award` and excluded from the main sample (the lot-to-contract mapping is ambiguous).

Output: data/interim/secop2_works_universe.parquet  (closed works-type contracts, 2015-2025)
        data/interim/secop2_frame_links.parquet     (frame SECOP II contracts with process info)
"""
from __future__ import annotations

import duckdb
import pandas as pd

from config import (FRAME_PQ, INTERIM, LOGS, SECOP2_CONTRACTS_PQ, SECOP2_PROCESSES_PQ)

WORKS_UNSPSC2 = ("72", "95")            # building/construction services; land & building structures
EXTRA_UNSPSC6 = ("811015",)             # civil engineering services (in the frame definition)


def q(p) -> str:
    return str(p).replace("\\", "/")


def main() -> None:
    con = duckdb.connect()
    tmp = (INTERIM / "duckdb_tmp")
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{q(tmp)}'")
    con.execute("SET memory_limit='2500MB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    log: list[str] = []

    # ---------------- contracts: restrict first (works-type or frame), then de-duplicate -------
    f_ids = pd.read_parquet(FRAME_PQ, columns=["id_contrato", "plataforma"])
    f_ids = f_ids.loc[f_ids.plataforma == "SECOP II", ["id_contrato"]]
    con.register("f_ids", f_ids)
    n_raw = con.execute(f"SELECT count(*) FROM read_parquet('{q(SECOP2_CONTRACTS_PQ)}')").fetchone()[0]
    n_dist = con.execute(f"SELECT count(DISTINCT id_contrato) FROM read_parquet('{q(SECOP2_CONTRACTS_PQ)}')").fetchone()[0]
    log.append(f"SECOP II contracts: raw rows {n_raw:,}; distinct id_contrato {n_dist:,} "
               f"({n_raw - n_dist:,} duplicate rows)")
    con.execute(f"""
        CREATE TABLE c AS
        SELECT * EXCLUDE rn FROM (
          SELECT *, substr(unspsc_raw, 4, 6) AS unspsc6, substr(unspsc_raw, 4, 4) AS unspsc4,
                 row_number() OVER (PARTITION BY id_contrato ORDER BY fecha_firma NULLS LAST) AS rn
          FROM read_parquet('{q(SECOP2_CONTRACTS_PQ)}')
          WHERE (substr(unspsc_raw, 4, 2) IN ('72', '95') OR substr(unspsc_raw, 4, 6) = '811015'
                 OR id_contrato IN (SELECT id_contrato FROM f_ids))
        ) WHERE rn = 1
    """)

    # ---------------- awarded process rows ----------------------------------------------------
    con.execute(f"""
        CREATE TABLE a AS
        SELECT DISTINCT id_portafolio, id_proceso, nit_adjudicado, valor_adjudicado, precio_base,
               prov_unicos_resp, prov_invitados, prov_interes, n_lotes, fecha_publicacion,
               fecha_adjudicacion, fecha_recepcion_resp, modalidad AS modalidad_proc, duracion, unidad_duracion,
               estado_proc, departamento AS depto_proc, orden_entidad AS orden_proc
        FROM read_parquet('{q(SECOP2_PROCESSES_PQ)}')
        WHERE adjudicado = 'Si'
          AND id_portafolio IN (SELECT proceso_compra FROM c WHERE proceso_compra IS NOT NULL)
    """)
    con.execute("""
        CREATE TABLE a1 AS
        SELECT id_portafolio, count(*) AS n_award_rows, count(DISTINCT id_proceso) AS n_proc
        FROM a GROUP BY 1
    """)
    con.execute("""
        CREATE TABLE a_u AS
        SELECT a.*, a1.n_award_rows, a1.n_proc
        FROM a JOIN a1 USING (id_portafolio)
    """)

    # ---------------- link every contract with its (unique) award row -------------------------
    con.execute("""
        CREATE TABLE L AS
        SELECT c.*, a_u.id_proceso, a_u.nit_adjudicado, a_u.valor_adjudicado, a_u.precio_base,
               a_u.prov_unicos_resp, a_u.prov_invitados, a_u.n_lotes, a_u.fecha_publicacion,
               a_u.fecha_adjudicacion, a_u.fecha_recepcion_resp, a_u.modalidad_proc, a_u.duracion, a_u.unidad_duracion,
               a_u.n_award_rows, a_u.n_proc, a_u.depto_proc
        FROM c LEFT JOIN a_u ON a_u.id_portafolio = c.proceso_compra
    """)
    # contracts with more than one distinct award row in their portfolio
    con.execute("""
        CREATE TABLE L2 AS
        SELECT *, CASE WHEN n_award_rows IS NULL THEN NULL
                       WHEN n_award_rows = 1 THEN 0 ELSE 1 END AS multi_award
        FROM L
    """)

    # a contract can still be duplicated when its portfolio has several identical-key award rows; keep first
    con.execute("""
        CREATE TABLE L3 AS
        SELECT * EXCLUDE rn FROM (
          SELECT *, row_number() OVER (PARTITION BY id_contrato ORDER BY id_proceso NULLS LAST) AS rn FROM L2
        ) WHERE rn = 1
    """)

    # ---------------- works universe (closed contracts) --------------------------------------
    works = con.execute(f"""
        SELECT id_contrato, proceso_compra, id_proceso, nit_entidad, entidad, departamento, ciudad,
               orden_entidad, estado_contrato, unspsc6, unspsc4, tipo_contrato, modalidad,
               fecha_firma, fecha_inicio, fecha_fin, nit_proveedor, valor_contrato, dias_adicionados,
               duracion_txt, precio_base, valor_adjudicado, prov_unicos_resp, prov_invitados,
               n_lotes, fecha_publicacion, fecha_adjudicacion, fecha_recepcion_resp, duracion, unidad_duracion,
               multi_award, nit_adjudicado
        FROM L3
        WHERE (substr(unspsc6, 1, 2) IN {WORKS_UNSPSC2} OR unspsc6 IN {EXTRA_UNSPSC6})
          AND estado_contrato IN ('Cerrado', 'terminado')
          AND fecha_firma >= DATE '2015-01-01'
    """).fetchdf()
    log.append(f"closed SECOP II works-type contracts (UNSPSC 72*, 95*, 811015): {len(works):,}")

    works["add_ratio_raw"] = works["valor_contrato"] / works["valor_adjudicado"] - 1
    ok = (works["valor_adjudicado"] > 0) & (works["multi_award"] == 0)
    works["outcome_ok"] = ok.astype(int)
    works.loc[~ok, "add_ratio_raw"] = float("nan")
    log.append(f"  with unique award row and positive award value (outcome measurable): {int(ok.sum()):,}")
    works.to_parquet(INTERIM / "secop2_works_universe.parquet", index=False)

    # ---------------- frame links ------------------------------------------------------------
    frame = pd.read_parquet(FRAME_PQ)
    f2 = frame.loc[frame.plataforma == "SECOP II", ["id_contrato"]]
    con.register("f2", f2)
    links = con.execute("""
        SELECT L3.* FROM L3 JOIN f2 USING (id_contrato)
    """).fetchdf()
    links["add_ratio_raw"] = links["valor_contrato"] / links["valor_adjudicado"] - 1
    log.append(f"frame SECOP II contracts: {len(f2):,}; found in raw contracts: {len(links):,}")
    log.append(f"  linked to an awarded process: {int(links['id_proceso'].notna().sum()):,}")
    log.append(f"  multi-award portfolios (excluded from main sample): {int((links['multi_award'] == 1).sum()):,}")
    links.to_parquet(INTERIM / "secop2_frame_links.parquet", index=False)

    (LOGS / "03_build_secop2_works.log").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
