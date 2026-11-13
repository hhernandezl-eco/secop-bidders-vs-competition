# Raw inputs (not distributed)

The raw files are not included in the repository (size and redistribution terms). Place them in this folder
(or set the environment variable `SECOP_RAW_DIR` to the folder that contains them).

| File (pattern) | Source | Notes |
|---|---|---|
| `SECOP_II_-_Contratos_Electr*.csv` | datos.gov.co, *SECOP II - Contratos Electrónicos* | Public open data; snapshot of 2025-08-20; ~8 GB |
| `SECOP_II_-_Procesos_de_Contrataci*.csv` | datos.gov.co, *SECOP II - Procesos de Contratación* | Public open data; snapshot of 2025-08-20; ~8 GB |
| `marco1.xlsx` | Contract sampling frame of transport-infrastructure contracts (SECOP I and II) | Available from the authors on request |

Column layouts used by the code are listed in `code/01_extract_secop2.py` and `code/02_build_frame.py`.
