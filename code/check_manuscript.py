"""Manuscript integrity checks (run before every submission).

1. Every \\label{fig:*} / \\label{tab:*} / \\label{sec:*} / \\label{app:*} is referenced with \\cref / \\Cref somewhere.
2. Every \\cref target exists.
3. Every \\includegraphics and \\input file exists.
4. Every \\cite key exists in references.bib, and every reference in the bib has a DOI (or is flagged).
5. No LaTeX warnings about undefined references/citations in the last compilation (paper/main.log).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def read_all() -> str:
    parts = [(PAPER / "main.tex").read_text(encoding="utf-8")]
    parts += [p.read_text(encoding="utf-8") for p in sorted((PAPER / "sections").glob("*.tex"))]
    parts += [p.read_text(encoding="utf-8") for p in sorted((PAPER / "tables").glob("*.tex"))]
    return "\n".join(parts)


def main() -> int:
    txt = read_all()
    body = re.sub(r"(?m)(?<!\\)%.*$", "", txt)          # strip comments (an escaped \% is kept)
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs: set[str] = set()
    for m in re.finditer(r"\\(?:[cC]ref|appcref)\{([^}]+)\}", body):
        refs.update(x.strip() for x in m.group(1).split(","))
    problems = 0

    print(f"{len(labels)} labels, {len(refs)} distinct \\cref targets")
    unref = sorted(l for l in labels if l not in refs and l.split(":")[0] in {"fig", "tab", "sec", "app", "eq"})
    for l in unref:
        print(f"  [UNREFERENCED] {l}")
        problems += 1
    for r in sorted(refs - labels):
        print(f"  [UNDEFINED \\cref TARGET] {r}")
        problems += 1

    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        cand = [PAPER / "figures" / f, PAPER / "figures" / (f + ".pdf"), PAPER / f]
        if not any(c.exists() for c in cand):
            print(f"  [MISSING FIGURE] {f}")
            problems += 1
    for f in re.findall(r"\\input\{([^}]+)\}", body):
        if not any((PAPER / (f + ext)).exists() for ext in ("", ".tex")):
            print(f"  [MISSING INPUT] {f}")
            problems += 1

    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", body):
        cited.update(x.strip() for x in m.group(1).split(","))
    for k in sorted(cited - keys):
        print(f"  [CITED BUT NOT IN BIB] {k}")
        problems += 1
    no_doi = []
    for entry in re.split(r"(?m)^@", bib)[1:]:
        key = entry.split("{", 1)[1].split(",", 1)[0]
        if key in cited and "doi" not in entry.lower():
            no_doi.append(key)
    print(f"{len(cited)} cited references; cited without DOI: {no_doi}")

    log = PAPER / "main.log"
    if log.exists():
        lg = log.read_text(encoding="utf-8", errors="ignore")
        bad = re.findall(r"(?m)^.*(?:undefined references|Citation `[^']+' .*undefined|Reference `[^']+' .*undefined).*$", lg)
        for b in bad[:10]:
            print("  [LATEX WARNING]", b.strip())
        problems += len(bad)
        unresolved = re.findall(r"\\R[A-Za-z]+", re.sub(r"\\newcommand.*", "", body))
        defined = set(re.findall(r"\\newcommand\{(\\R[A-Za-z]+)\}", (PAPER / "numbers.tex").read_text(encoding="utf-8")))
        missing = sorted({u for u in unresolved if u not in defined})
        for u in missing:
            print(f"  [NUMBER MACRO NOT DEFINED] {u}")
        problems += len(missing)
    print("OK - no problems" if problems == 0 else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
