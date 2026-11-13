"""Small helpers shared by the pipeline scripts."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_text(x) -> str | float:
    """Upper-case, accent-free, single-spaced text. The frame (built in R) has corrupted accents
    such as 'BOYACa' for 'BOYACA'; upper-casing after stripping accents repairs them."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = strip_accents(str(x)).upper().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def clean_nit(x) -> str | float:
    """Digits-only tax/ID number. 10-digit ids starting with 8 or 9 are NITs written with the
    verification digit, which is dropped so the same firm has the same id across platforms."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = re.sub(r"\D", "", str(x))
    if s == "" or set(s) == {"0"}:
        return np.nan
    if len(s) == 10 and s[0] in "89":
        s = s[:-1]
    return s


def tipo_cat(x) -> str:
    s = norm_text(x)
    if not isinstance(s, str):
        return "OTRO"
    if s.startswith("OBRA"):
        return "OBRA"
    if s.startswith("INTERVENTOR"):
        return "INTERVENTORIA"
    if s.startswith("CONSULTOR"):
        return "CONSULTORIA"
    return "OTRO"


def _next_monday(d):
    import datetime as dt
    return d + dt.timedelta(days=(7 - d.weekday()) % 7)


def co_holidays(year: int) -> set:
    """Colombian public holidays (Ley Emiliani, Ley 51 de 1983): fixed dates, dates moved to the following
    Monday, and Easter-based dates."""
    import datetime as dt
    from dateutil.easter import easter
    h = {dt.date(year, m, d) for m, d in [(1, 1), (5, 1), (7, 20), (8, 7), (12, 8), (12, 25)]}
    h |= {_next_monday(dt.date(year, m, d)) for m, d in [(1, 6), (3, 19), (6, 29), (8, 15), (10, 12), (11, 1), (11, 11)]}
    e = easter(year)
    h |= {e - dt.timedelta(days=3), e - dt.timedelta(days=2)}
    h |= {_next_monday(e + dt.timedelta(days=k)) for k in (43, 64, 71)}
    return h


def window_calendar(start, end, holidays: set) -> tuple:
    """Calendar days, weekend days and weekday-holidays between two dates (inclusive)."""
    if pd.isna(start) or pd.isna(end) or end < start:
        return (np.nan, np.nan, np.nan)
    days = pd.date_range(start, end, freq="D")
    wk = int(sum(d.weekday() >= 5 for d in days))
    hol = int(sum((d.date() in holidays) and d.weekday() < 5 for d in days))
    return (len(days), wk, hol)


def winsor(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    a, b = s.quantile([lo, hi])
    return s.clip(a, b)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
