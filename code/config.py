"""Central configuration: every script imports its paths from here.

Raw data are NOT copied into the project (the two SECOP II files are ~8 GB each).
They are read in place from RAW_DIR. If you move the project, only RAW_DIR needs to change
(or set the environment variable SECOP_RAW_DIR).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("SECOP_RAW_DIR", PROJECT_DIR.parent))

REFERENCE = PROJECT_DIR / "data" / "reference"
INTERIM = PROJECT_DIR / "data" / "interim"
PROCESSED = PROJECT_DIR / "data" / "processed"
TABLES = PROJECT_DIR / "output" / "tables"
FIGURES = PROJECT_DIR / "output" / "figures"
LOGS = PROJECT_DIR / "output" / "logs"
PAPER_TABLES = PROJECT_DIR / "paper" / "tables"
PAPER_FIGURES = PROJECT_DIR / "paper" / "figures"

for _p in (INTERIM, PROCESSED, TABLES, FIGURES, LOGS, PAPER_TABLES, PAPER_FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


def _find(pattern: str) -> Path:
    hits = sorted(glob.glob(str(RAW_DIR / pattern)))
    if not hits:
        raise FileNotFoundError(f"No raw file matching {pattern!r} in {RAW_DIR}")
    return Path(hits[0])


# Raw inputs (read-only)
def raw_contracts_csv() -> Path:
    """SECOP II - Contratos Electronicos (datos.gov.co, snapshot 2025-08-20)."""
    return _find("SECOP_II_-_Contratos*.csv")


def raw_processes_csv() -> Path:
    """SECOP II - Procesos de Contratacion (datos.gov.co, snapshot 2025-08-20)."""
    return _find("SECOP_II_-_Procesos*.csv")


def raw_frame_xlsx() -> Path:
    """Sampling frame of infrastructure contracts (SECOP I + II), built with estratos3.R."""
    return _find("marco1.xlsx")


# Intermediate / processed artefacts
SECOP2_CONTRACTS_PQ = INTERIM / "secop2_contracts_light.parquet"
SECOP2_PROCESSES_PQ = INTERIM / "secop2_processes_light.parquet"
FRAME_PQ = INTERIM / "frame_marco1.parquet"
ANALYSIS_PQ = PROCESSED / "analysis_contracts.parquet"
ANALYSIS_CSV = PROCESSED / "analysis_contracts.csv"

# --------------------------------------------------------------------------------------
# Analysis constants (single source of truth)
# --------------------------------------------------------------------------------------
SEED = 20260513
YEAR_MIN, YEAR_MAX = 2018, 2024
N_BOOT = 500                     # bootstrap replications (clustered)
SMMLV = {                        # Colombian monthly minimum wage (COP), used only for scaling checks
    2018: 781_242, 2019: 828_116, 2020: 877_803, 2021: 908_526,
    2022: 1_000_000, 2023: 1_160_000, 2024: 1_300_000,
}
