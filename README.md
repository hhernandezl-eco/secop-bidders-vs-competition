# SECOP participation, price pressure and contract renegotiation

Replication code for *More bidders, not more competition: participation, price pressure and contract
renegotiation in Colombian public works* (Albano, Calderón-Ramírez, Hernández-Lasso, Ortiz).

The repository contains everything needed to go from the raw SECOP files to every table, figure and in-text number
of the paper. Data are not included (see `data/raw/README.md`).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the raw files in data/raw/  (or: export SECOP_RAW_DIR=/path/to/folder)
python code/run_all.py                # full pipeline (~25 min, 8 GB RAM is enough)
python code/run_all.py --force        # also redo the raw CSV extraction
python code/run_all.py --from 06      # restart from a given step
```

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 01 | `01_extract_secop2.py` | Raw CSV to Parquet (selected columns; money and dates parsed) |
| 02 | `02_build_frame.py` | Clean the contract frame |
| 03 | `03_build_secop2_works.py` | Link SECOP II contracts to processes; derive additions (current value minus awarded value) |
| 04 | `04_build_features.py` | Firm, entity and market variables, measured before each focal contract |
| 05 | `05_assemble_analysis.py` | Analysis data set, outcomes, sample flags |
| 06 | `06_participation.py` | Main results by award regime |
| 07 | `07_firms.py` | Bidder capabilities and out-of-sample predictive validity |
| 08 | `08_robustness.py` | Robustness, wild-cluster bootstrap, broad universe, placebo, IV, mediation |
| 09 | `09_descriptives_figures.py` | Descriptive tables and figures |
| 11 | `11_data_dictionary.py` | Data dictionary (`docs/data_dictionary.md`) |

Shared helpers: `config.py` (paths, seeds, constants), `utils.py`, `models.py`, `tex.py`.

## Outputs

* `data/interim/`, `data/processed/`: Parquet extracts and the analysis data set (`analysis_contracts.parquet`).
* `output/tables/*.csv`: results behind each table. `output/logs/`: log of each step.
* `output/latex/tables/*.tex`, `output/figures/*.pdf`: tables and figures ready to include in LaTeX.
* `output/numbers.json`, `output/latex/numbers.tex`: every number quoted in the text, as LaTeX macros.

## Notes

* All random procedures use the seed in `code/config.py`.
* Tested with Python 3.9 on Windows 11 with 8 GB of RAM; paths are OS-independent.
* The SECOP II contract file does not report the value of additions; the code derives it from awarded and current values (`03`).
