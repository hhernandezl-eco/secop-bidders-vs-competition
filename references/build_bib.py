"""Build references.bib from Crossref and verify every reference.

For each entry we give the bibliographic data we *expect* (first author, year, title fragment) and, when we
know it, a DOI hint. The script
  1. fetches the DOI record from Crossref (or, without a hint / on a mismatch, searches Crossref);
  2. accepts the record only if first author, year (+-1) and title similarity match the expectation;
  3. writes BibTeX from the *Crossref metadata* (not from memory) and a verification report.

Run:  python references/build_bib.py          (needs internet; only public bibliographic queries are sent)
Outputs: references/references.bib, references/verification_report.csv, paper/references.bib
"""
from __future__ import annotations

import csv
import difflib
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "replication-package-reference-check/1.0 (public bibliographic lookup)"}

# key, first-author family name, year, expected title (fragment), DOI hint ('' = search)
REFS = [
    # ---- procurement: empirical ----
    ("bajari2009", "Bajari", 2009, "Auctions versus negotiations in procurement", "10.1093/jleo/ewn002"),
    ("bajari2014", "Bajari", 2014, "Bidding for incomplete contracts", "10.1257/aer.104.4.1288"),
    ("decarolis2014", "Decarolis", 2014, "Awarding price, contract performance, and bids screening", "10.1257/app.6.1.108"),
    ("decarolis2015", "Decarolis", 2015, "Renegotiation of public contracts", ""),
    ("decarolis2020", "Decarolis", 2020, "Bureaucratic competence and procurement outcomes", "10.1093/jleo/ewaa004"),
    ("coviello2018", "Coviello", 2018, "The effect of discretion on procurement performance", "10.1287/mnsc.2016.2628"),
    ("szucs2024", "Szucs", 2024, "Discretion and favoritism in public procurement", ""),
    ("kang2022", "Kang", 2022, "Winning by default", ""),
    ("gil2013", "Gil", 2013, "Self-enforcing agreements and relational contracting", ""),
    ("lewisfaupel2016", "Lewis-Faupel", 2016, "Can electronic procurement improve infrastructure provision", "10.1257/pol.20140258"),
    ("bosio2022", "Bosio", 2022, "Public procurement in law and practice", "10.1257/aer.20200738"),
    ("baltrunaite2021", "Baltrunaite", 2021, "Discretion and supplier selection in public procurement", ""),
    ("krasnokutskaya2011", "Krasnokutskaya", 2011, "Bid preference programs and participation in highway procurement auctions", "10.1257/aer.101.6.2653"),
    ("athey2011", "Athey", 2011, "Comparing open and sealed bid auctions", ""),
    ("li2009", "Li", 2009, "Entry and competition effects in first-price auctions", ""),
    ("conley2016", "Conley", 2016, "Detecting bidders groups in collusive auctions", "10.1257/mic.20130254"),
    ("fazekas2020", "Fazekas", 2020, "Uncovering high-level corruption", "10.1017/s0007123417000461"),
    ("auriol2016", "Auriol", 2016, "Public procurement and rent-seeking: the case of Paraguay", ""),
    ("guasch2008", "Guasch", 2008, "Renegotiation of concession contracts in Latin America", ""),
    ("flyvbjerg2002", "Flyvbjerg", 2002, "Underestimating costs in public works projects", "10.1080/01944360208976273"),
    ("flyvbjerg2003", "Flyvbjerg", 2003, "How common and how large are cost overruns in transport infrastructure projects", "10.1080/01441640309904"),
    ("athias2008", "Athias", 2008, "The more the merrier? Number of bidders, information dispersion, renegotiation", "10.2139/ssrn.1269630"),
    ("li2008", "Li", 2008, "Analysis of the impacts of the number of bidders upon bid values", "10.1177/1087724x07312144"),
    ("estache2010", "Estache", 2010, "Bidder asymmetry in infrastructure procurement", "10.1007/s11151-010-9242-z"),
    ("pliatsidis2022", "Pliatsidis", 2022, "Impact of the time limits for the receipt of tenders on the number of bidders", "10.1108/jopp-05-2022-0025"),
    ("christoff2024", "Christoff", 2024, "Colombia", "10.4324/9781003220374-22"),
    ("miranda2016", "Miranda Sarmento", 2016, "Cost overruns in public sector investment projects", "10.1177/1087724x16668357"),
    ("engel2006", "Engel", 2006, "Renegotiation without holdup", "10.3386/w12399"),
    # ---- procurement: theory ----
    ("ganuza2007", "Ganuza", 2007, "Competition and cost overruns in procurement", ""),
    ("spagnolo2012", "Spagnolo", 2012, "Reputation, competition, and entry in procurement", "10.1016/j.ijindorg.2012.01.001"),
    ("albano2009", "Albano", 2009, "Regulating joint bidding in public procurement", ""),
    ("albano2017", "Albano", 2017, "Public procurement with unverifiable quality", "10.1016/j.jpubeco.2016.11.004"),
    ("kagel2009", "Kagel", 2009, "Common Value Auctions and the Winner's Curse", "10.1515/9781400830138"),
    ("thaler1988", "Thaler", 1988, "Anomalies: the winner's curse", "10.1257/jep.2.1.191"),
    ("milgrom1982", "Milgrom", 1982, "A theory of auctions and competitive bidding", "10.2307/1911865"),
    ("tirole1986", "Tirole", 1986, "Procurement and renegotiation", "10.1086/261371"),
    ("hart1988", "Hart", 1988, "Incomplete contracts and renegotiation", "10.2307/1912698"),
    ("bajari2001", "Bajari", 2001, "Incentives versus transaction costs: a theory of procurement contracts", "10.2307/2696361"),
    ("kornai2003", "Kornai", 2003, "Understanding the soft budget constraint", "10.1257/002205103771800004"),
    # ---- econometrics ----
    ("cragg1971", "Cragg", 1971, "Some statistical models for limited dependent variables", "10.2307/1909582"),
    ("heckman1979", "Heckman", 1979, "Sample selection bias as a specification error", "10.2307/1912352"),
    ("cameron2008", "Cameron", 2008, "Bootstrap-based improvements for inference with clustered errors", "10.1162/rest.90.3.414"),
    ("cameron2015", "Cameron", 2015, "A practitioner's guide to cluster-robust inference", "10.3368/jhr.50.2.317"),
    ("andrews2019", "Andrews", 2019, "Weak instruments in instrumental variables regression: theory and practice", "10.1146/annurev-economics-080218-025643"),
    ("anderson1949", "Anderson", 1949, "Estimation of the parameters of a single equation in a complete system of stochastic equations", "10.1214/aoms/1177730090"),
    ("oster2019", "Oster", 2019, "Unobservable selection and coefficient stability", "10.1080/07350015.2016.1227711"),
    ("imai2010", "Imai", 2010, "Identification, inference and sensitivity analysis for causal mediation effects", "10.1214/10-STS321"),
    ("baron1986", "Baron", 1986, "The moderator-mediator variable distinction in social psychological research", "10.1037/0022-3514.51.6.1173"),
    ("nickell1981", "Nickell", 1981, "Biases in dynamic models with fixed effects", "10.2307/1911408"),
    ("lee2022", "Lee", 2022, "Valid t-ratio inference for IV", "10.1257/aer.20211063"),
    ("cinelli2020", "Cinelli", 2020, "Making sense of sensitivity: extending omitted variable bias", "10.1111/rssb.12348"),
]


AUTHOR_OVERRIDE = {"cameron2015": "Cameron, A. Colin and Miller, Douglas L.",
                   "cameron2008": "Cameron, A. Colin and Gelbach, Jonah B. and Miller, Douglas L."}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("‐", "-").replace("–", "-"))


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def ok(item: dict, fam: str, year: int, title: str, tol: int = 1) -> tuple[bool, str]:
    t = (item.get("title") or [""])[0]
    auth = item.get("author") or []
    fam_ok = any(norm(fam) in norm(a.get("family", "")) for a in auth[:3]) if auth else False
    yr = (item.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0]
    yr_ok = yr is not None and abs(yr - year) <= tol
    t_ok = norm(title) in norm(t) or sim(t, title) >= 0.80
    return (fam_ok and yr_ok and t_ok), f"author={fam_ok} year={yr_ok}({yr}) title={t_ok}({sim(t, title):.2f})"


def _get(url, **kw):
    for attempt in range(4):                      # Crossref occasionally times out or rate-limits
        try:
            r = requests.get(url, headers=UA, timeout=45, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def fetch(doi: str):
    r = _get(f"https://api.crossref.org/works/{doi}")
    return r.json()["message"] if r is not None else None


def search(fam: str, year: int, title: str):
    r = _get("https://api.crossref.org/works", params={"query.bibliographic": f"{title} {fam} {year}", "rows": 8})
    if r is None:
        return None
    for it in r.json()["message"]["items"]:
        good, _ = ok(it, fam, year, title)
        if good:
            return it
    return None


def bib_entry(key: str, it: dict) -> str:
    def esc(s):
        return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    import html
    def fix_case(x: str) -> str:
        x = x.replace(chr(0x2010), "-")                 # non-ASCII hyphen
        if not x.isupper():
            return x
        small = {"and", "in", "of", "the", "for", "to", "a", "an", "on", "with", "from", "by", "versus", "vs"}
        words = x.title().split(" ")                    # Crossref sometimes stores ALL-CAPS names/titles
        return " ".join(w.lower() if (i > 0 and w.lower() in small) else w for i, w in enumerate(words))
    authors = " and ".join(f"{fix_case(a.get('family', ''))}, {fix_case(a.get('given', ''))}".strip(", ") for a in it.get("author", []))
    title = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", (it.get("title") or [""])[0]))).strip().rstrip("*")
    title = fix_case(title)
    jr = html.unescape((it.get("container-title") or [""])[0])
    yr = it["issued"]["date-parts"][0][0]
    doi = it["DOI"]
    if it.get("type") == "book-chapter":
        nl = chr(10)
        pub = html.unescape(it.get("publisher", ""))
        return (f"@incollection{{{key},{nl}  author = {{{authors}}},{nl}  title = {{{{{esc(title)}}}}},{nl}"
                f"  booktitle = {{{esc(jr)}}},{nl}  publisher = {{{esc(pub)}}},{nl}  year = {{{yr}}},{nl}  doi = {{{doi}}}{nl}}}{nl}")
    if not jr:                                   # working papers / books
        inst = ("SSRN Working Paper" if doi.startswith("10.2139") else "NBER Working Paper" if doi.startswith("10.3386")
                else "World Bank Policy Research Working Paper" if doi.startswith("10.1596") else "")
        nl = chr(10)
        if it.get("type") == "book" or doi.startswith("10.1515/9781"):
            return (f"@book{{{key},{nl}  author = {{{authors}}},{nl}  title = {{{{{esc(title)}}}}},{nl}"
                    f"  publisher = {{Princeton University Press}},{nl}  year = {{{yr}}},{nl}  doi = {{{doi}}}{nl}}}{nl}")
        return (f"@techreport{{{key},{nl}  author = {{{authors}}},{nl}  title = {{{{{esc(title)}}}}},{nl}"
                f"  institution = {{{inst}}},{nl}  year = {{{yr}}},{nl}  doi = {{{doi}}}{nl}}}{nl}")
    if key in AUTHOR_OVERRIDE:                       # compound given names that Crossref splits wrongly
        authors = AUTHOR_OVERRIDE[key]
    f = [f"  author = {{{authors}}}", f"  title = {{{{{esc(title)}}}}}", f"  journal = {{{esc(jr)}}}", f"  year = {{{yr}}}"]
    for k_src, k_dst in [("volume", "volume"), ("issue", "number"), ("page", "pages")]:
        if it.get(k_src):
            v = it[k_src].replace("-", "--") if k_src == "page" else it[k_src]
            f.append(f"  {k_dst} = {{{v}}}")
    f.append(f"  doi = {{{it['DOI']}}}")
    return f"@article{{{key},\n" + ",\n".join(f) + "\n}\n"


MANUAL = r"""
@misc{gandour2025,
  author = {{Gandour Consultores}},
  title = {Consultor{\'\i}a: an{\'a}lisis del impacto de los documentos normalizados en la contrataci{\'o}n p{\'u}blica},
  year = {2025},
  note = {Informe para el Banco Interamericano de Desarrollo; documento de trabajo no publicado}
}
@misc{ley80,
  author = {{Congreso de la Rep{\'u}blica de Colombia}},
  title = {Ley 80 de 1993, por la cual se expide el Estatuto General de Contrataci{\'o}n de la Administraci{\'o}n P{\'u}blica},
  year = {1993}
}
@misc{ley1150,
  author = {{Congreso de la Rep{\'u}blica de Colombia}},
  title = {Ley 1150 de 2007, por medio de la cual se introducen medidas para la eficiencia y la transparencia en la Ley 80 de 1993},
  year = {2007}
}
@misc{ley1882,
  author = {{Congreso de la Rep{\'u}blica de Colombia}},
  title = {Ley 1882 de 2018, por la cual se adicionan, modifican y dictan disposiciones orientadas a fortalecer la contrataci{\'o}n p{\'u}blica en Colombia},
  year = {2018}
}
@article{bulow1996,
  author = {Bulow, Jeremy and Klemperer, Paul},
  title = {Auctions versus Negotiations},
  journal = {American Economic Review},
  year = {1996},
  volume = {86},
  number = {1},
  pages = {180--194},
}
@misc{cce2024dt,
  author = {{Colombia Compra Eficiente}},
  title = {Documentos Tipo de obra p{\'u}blica de infraestructura de transporte: Resoluciones 463, 464 y 465 de 2024, documentos base de licitaci{\'o}n (versi{\'o}n 4), selecci{\'o}n abreviada de menor cuant{\'\i}a (versi{\'o}n 3) y m{\'\i}nima cuant{\'\i}a, y tabla general de Documentos Tipo},
  year = {2025},
  note = {Agencia Nacional de Contrataci{\'o}n P{\'u}blica}
}
@misc{ancp2024relatoria,
  author = {{Agencia Nacional de Contrataci{\'o}n P{\'u}blica -- Colombia Compra Eficiente}},
  title = {Relator{\'\i}a: conceptos sobre la ponderaci{\'o}n de la propuesta econ{\'o}mica en los Documentos Tipo de obra p{\'u}blica de infraestructura de transporte (Resoluci{\'o}n 1798 de 2019)},
  year = {2024},
  howpublished = {relatoria.colombiacompra.gov.co}
}
@misc{res1798,
  author = {{Agencia Nacional de Contrataci{\'o}n P{\'u}blica -- Colombia Compra Eficiente}},
  title = {Resoluci{\'o}n 1798 de 2019, por la cual se adoptan los Documentos Tipo para la licitaci{\'o}n de obra p{\'u}blica de infraestructura de transporte},
  year = {2019}
}
@book{laffont1993,
  author = {Laffont, Jean-Jacques and Tirole, Jean},
  title = {A Theory of Incentives in Procurement and Regulation},
  publisher = {MIT Press},
  address = {Cambridge, MA},
  year = {1993}
}
@book{wooldridge2010,
  author = {Wooldridge, Jeffrey M.},
  title = {Econometric Analysis of Cross Section and Panel Data},
  edition = {2},
  publisher = {MIT Press},
  address = {Cambridge, MA},
  year = {2010}
}
"""


def main() -> None:
    entries, report = [], []
    for key, fam, year, title, hint in REFS:
        item, how, detail = None, "", ""
        if hint:
            try:
                it = fetch(hint)
            except Exception as e:  # network
                it = None
            if it:
                good, detail = ok(it, fam, year, title, tol=3)
                if good:
                    item, how = it, "doi-hint verified"
        if item is None:
            try:
                it = search(fam, year, title)
            except Exception:
                it = None
            if it:
                item, how = it, "found by Crossref search" + (f" (hint {hint} rejected: {detail})" if hint else "")
        if item is None:
            report.append(dict(key=key, status="NOT VERIFIED - excluded", doi=hint, title="", journal="", year=year, how=detail))
            print(f"[NOT VERIFIED] {key}: {fam} {year} {title}")
        else:
            entries.append(bib_entry(key, item))
            report.append(dict(key=key, status="verified", doi=item["DOI"], title=(item.get("title") or [""])[0],
                               journal=(item.get("container-title") or [""])[0], year=item["issued"]["date-parts"][0][0], how=how))
            print(f"[ok] {key}: {item['DOI']}  {(item.get('title') or [''])[0][:70]}")
        time.sleep(0.25)
    text = "% Generated by references/build_bib.py from Crossref metadata. Do not edit by hand.\n\n" + "\n".join(entries) + "\n" + MANUAL
    (HERE / "references.bib").write_text(text, encoding="utf-8")
    (HERE.parent / "paper" / "references.bib").write_text(text, encoding="utf-8")
    with open(HERE / "verification_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    nv = sum(r["status"] != "verified" for r in report)
    print(f"\n{len(report) - nv} verified, {nv} not verified (excluded from the bibliography)")


if __name__ == "__main__":
    main()
