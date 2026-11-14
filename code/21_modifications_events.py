"""STEP 21 - Platform record of contract modifications versus the derived increase (F - P).

The open-data table 'SECOP II - Adiciones' (datos.gov.co, id cb9c-h8sn; one row per modification event, with columns
Identificador, ID_Contrato, Tipo, Descripcion, FechaRegistro) has no amounts, but it types each event
('Adicion en el valor', 'Reduccion en el valor', 'Modificacion General', 'Extension', 'Suspension', ...) and keeps the entity's free-text
justification. It therefore validates the *direction* of the derived measure: do contracts whose current value exceeds the awarded value
carry a registered value-addition event, and contracts without an increase do not? It also gives a rough, text-based split of the
increase into formal additions, larger quantities of work and price adjustments, which the document-level audit of step 20 will refine.

Only events registered on or before the snapshot date of the contract file (2025-08-20) are used, so that events and contract values
refer to the same state of the platform.

Network: the script queries the public Socrata API for the contract ids of the broad works universe (ids are public identifiers; no other
information is sent) and caches the events in data/interim/secop2_modifications_works.parquet. Re-running uses the cache; --refresh re-queries.

Outputs: data/interim/secop2_modifications_works.parquet, data/audit/audit_sample_events.csv (events of the audit sample, to help the coder),
tab_events.tex, numbers
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from config import INTERIM, PROJECT_DIR
from tex import fint, fnum, put_number, write_numbers_tex, write_table

URL = "https://www.datos.gov.co/resource/cb9c-h8sn.json"
CACHE = INTERIM / "secop2_modifications_works.parquet"
AUDIT_DIR = PROJECT_DIR / "data" / "audit"
SNAPSHOT = pd.Timestamp("2025-08-20")
BATCH = 200

KW = {
    "cantidades": r"mayor(es)? cantidad|cantidades (de obra|adicional|mayor)|obras? adicional|items? no previst|obras? complementaria|cantidades de obra",
    "reajuste": r"reajust|ajuste de precio|actualizacion de precio|escalamiento|formula de ajuste|variacion de precio|\bipc\b",
    "otrosi": r"otros[i]|otro si|modificatorio|adicion|adicional|adicionar",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return s.lower()


def fetch(ids: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        where = "id_contrato in (" + ",".join("'" + c + "'" for c in chunk) + ")"
        q = urllib.parse.urlencode({"$where": where, "$limit": 50000})
        for attempt in range(4):
            try:
                req = urllib.request.Request(URL + "?" + q, headers={"User-Agent": "research-client"})
                rows += json.load(urllib.request.urlopen(req, timeout=180))
                break
            except Exception as e:                                         # transient network errors
                if attempt == 3:
                    raise
                print("retry", attempt, e)
                time.sleep(3 * (attempt + 1))
        time.sleep(0.25)
        if (i // BATCH) % 10 == 0:
            print(f"  fetched {min(i + BATCH, len(ids)):,} / {len(ids):,} contracts, {len(rows):,} events", flush=True)
    ev = pd.DataFrame(rows)
    for c in ["identificador", "id_contrato", "tipo", "descripcion", "fecharegistro"]:
        if c not in ev.columns:
            ev[c] = np.nan
    ev["fecharegistro"] = pd.to_datetime(ev["fecharegistro"], errors="coerce")
    return ev[["identificador", "id_contrato", "tipo", "descripcion", "fecharegistro"]]


def load_events(ids: list[str], refresh: bool) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    ev = fetch(ids)
    ev.to_parquet(CACHE, index=False)
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    br = importlib.import_module("08_robustness")
    w = br.broad_sample()
    w["has_add"] = w["add_ratio_raw"] > 0.001
    ev = load_events(w["id_contrato"].unique().tolist(), args.refresh)
    n_all = len(ev)
    ev = ev[ev["fecharegistro"].isna() | (ev["fecharegistro"] <= SNAPSHOT)].copy()
    put_number("events n rows", n_all, ",d")
    put_number("events n rows snapshot", len(ev), ",d")
    ev["tipo_n"] = ev["tipo"].map(norm)
    ev["desc_n"] = ev["descripcion"].map(norm)
    ev["is_addval"] = ev["tipo_n"].eq("adicion en el valor")
    ev["is_redval"] = ev["tipo_n"].eq("reduccion en el valor")
    ev["is_modgen"] = ev["tipo_n"].eq("modificacion general")
    for k, pat in KW.items():
        ev[f"kw_{k}"] = ev["desc_n"].str.contains(pat, regex=True)
    g = ev.groupby("id_contrato").agg(n_events=("identificador", "size"), addval=("is_addval", "any"), redval=("is_redval", "any"),
                                      modgen=("is_modgen", "any"), kw_cantidades=("kw_cantidades", "any"), kw_reajuste=("kw_reajuste", "any"),
                                      kw_otrosi=("kw_otrosi", "any")).reset_index()
    d = w.merge(g, on="id_contrato", how="left")
    for c in ["addval", "redval", "modgen", "kw_cantidades", "kw_reajuste", "kw_otrosi"]:
        d[c] = d[c].fillna(False).astype(bool)
    d["n_events"] = d["n_events"].fillna(0)
    d["no_event"] = d["n_events"].eq(0)
    # classification of the increase from the platform record (hierarchical, a lower bound for each type)
    d["cls"] = np.select([d.addval, d.kw_cantidades, d.kw_reajuste, d.kw_otrosi], ["addval", "cantidades", "reajuste", "otrosi_text"], "none")
    put_number("events n contracts", len(d), ",d")
    put_number("events share with events", 100 * (~d.no_event).mean(), ".0f")      # all contracts carry some event (liquidation record, etc.)

    inc = d[d.has_add]
    noinc = d[~d.has_add]
    cols = [("Increase: all", inc), ("M\\'inima", inc[inc.regime == "minima"]), ("Scored", inc[inc.regime != "minima"]), ("No increase", noinc)]
    labs = [("addval", "Registered event ``Adici\\'on en el valor''"), ("kw_otrosi", "Free text mentions otros\\'i, modificatorio or adici\\'on"),
            ("kw_cantidades", "Free text mentions larger quantities or additional works"), ("kw_reajuste", "Free text mentions price adjustment (reajuste, IPC)")]
    body = []
    for key, lab in labs:
        cells = [fnum(100 * df[key].mean(), 0) for _, df in cols]
        body.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        for (nm, df), tag in zip(cols, ["all", "minima", "scored", "noinc"]):
            put_number(f"events {key} {tag}", 100 * df[key].mean(), ".0")
    body.append("\\midrule")
    body.append("Contracts & " + " & ".join(fint(len(df)) for _, df in cols) + " \\\\")
    # hierarchical split of the contracts with an increase
    body.append("\\midrule")
    body.append("\\multicolumn{5}{l}{\\emph{Contracts with an increase, by the strongest platform record (hierarchical, \\% of contracts)}} \\\\")
    for key, lab in [("addval", "Registered value addition"), ("cantidades", "No such event; text on larger quantities or additional works"),
                     ("reajuste", "Neither; text on price adjustment"), ("otrosi_text", "Neither; text on otros\\'i or modificatorio"),
                     ("none", "No record that points to a value change")]:
        cells = [fnum(100 * (df["cls"] == key).mean(), 0) for _, df in cols[:3]]
        body.append(f"{lab} & " + " & ".join(cells) + " & \\\\")
        put_number(f"events cls {key} all", 100 * (inc["cls"] == key).mean(), ".0")
    write_table("tab_events", "\n".join(body), "", "tab:events",
                "Broad SECOP II works universe. The table of contract modifications of SECOP II (datos.gov.co, ``SECOP II - Adiciones'') types each modification event and keeps the entity's free-text justification, but reports no amounts. "
                "Only events registered on or before 20 August 2025, the snapshot date of the contract file, are used. ``Increase'' means that the current contract value exceeds the awarded value by more than 0.1\\%. "
                "Keyword searches are run on the unaccented, lower-case description. The hierarchical split assigns each contract to the first record that applies, so each category is a lower bound.",
                "lcccc", "& \\multicolumn{3}{c}{With an increase} & Without \\\\\n\\cmidrule(lr){2-4}\n& All & M\\'inima & Scored & increase")
    # ---- contrasts used in the text
    put_number("events n increase", len(inc), ",d")
    put_number("events n noincrease", len(noinc), ",d")
    big = d[d.add_ratio_raw > 0.10]
    put_number("events addval big", 100 * big.addval.mean(), ".0")
    small = d[d.has_add & (d.add_ratio_raw <= 0.10)]
    put_number("events addval small", 100 * small.addval.mean(), ".0")
    put_number("events n small", len(small), ",d")

    # ---- events of the audit sample, to help the coder
    sp = AUDIT_DIR / "audit_sample.csv"
    if sp.exists():
        s = pd.read_csv(sp, encoding="utf-8-sig")[["audit_id", "id_contrato"]]
        e = ev.merge(s, on="id_contrato").sort_values(["audit_id", "fecharegistro"])
        e[["audit_id", "id_contrato", "fecharegistro", "tipo", "descripcion", "identificador"]].to_csv(AUDIT_DIR / "audit_sample_events.csv", index=False, encoding="utf-8-sig")
        put_number("events n audit sample events", len(e), ",d")
    write_numbers_tex()
    print(inc.groupby("regime")[["addval", "kw_cantidades", "kw_reajuste", "kw_otrosi", "no_event"]].mean().round(3))
    print(noinc.groupby("regime")[["addval", "kw_cantidades", "kw_reajuste", "kw_otrosi", "no_event"]].mean().round(3))
    print("cls shares among increases:", inc["cls"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
