"""Run the whole pipeline, from the raw SECOP files to tables and figures.

    python code/run_all.py                # full run (skips the slow raw-CSV extraction if Parquet files exist)
    python code/run_all.py --force        # also redo the extraction of the two 8 GB CSV files (~6 min)
    python code/run_all.py     python code/run_all.py --from 06      # restart from a given step

Every step is a stand-alone script that reads from data/ and writes to data/ and output/.
Seeds are fixed in code/config.py, so results are reproducible bit-for-bit given the same inputs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

STEPS = [
    ("01", "01_extract_secop2.py", "raw CSV -> Parquet (SECOP II contracts and processes)"),
    ("02", "02_build_frame.py", "clean the contract frame (marco1.xlsx)"),
    ("03", "03_build_secop2_works.py", "link SECOP II contracts to processes; derive additions"),
    ("04", "04_build_features.py", "firm, entity and market features (ex ante)"),
    ("05", "05_assemble_analysis.py", "analysis data set and sample flags"),
    ("06", "06_participation.py", "main results: participation vs competition"),
    ("07", "07_firms.py", "bidder capabilities and predictive validity"),
    ("08", "08_robustness.py", "robustness, inference, IV, placebo, mediation"),
    ("09", "09_descriptives_figures.py", "descriptive tables and figures"),
    ("11", "11_data_dictionary.py", "data dictionary"),
]


def run(cmd: list[str], cwd: Path) -> None:
    t = time.time()
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)
    print(f"<<< done in {time.time() - t:,.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redo the raw CSV extraction")
    ap.add_argument("--from", dest="start", default="01", help="first step to run (e.g. 06)")
    args = ap.parse_args()

    interim = ROOT / "data" / "interim"
    skip_extract = (interim / "secop2_contracts_light.parquet").exists() and (interim / "secop2_processes_light.parquet").exists() and not args.force
    first = next((i for i, s in enumerate(STEPS) if s[0] == args.start), 0)
    if args.start in ("01", "02", "03", "04", "05", "06"):
        # start the number macros from scratch so that no stale key survives a code change
        nf = ROOT / "output" / "numbers.json"
        if nf.exists():
            nf.unlink()
    for code, script, desc in STEPS[first:]:
        if code == "01" and skip_extract:
            print(f"--- skipping {script} (Parquet extracts exist; use --force to redo)")
            continue
        print(f"\n=== Step {code}: {desc}")
        run([sys.executable, "-W", "ignore", script], HERE)


if __name__ == "__main__":
    main()
