"""Inferenza e descrizione della frequenza di una serie temporale."""

from __future__ import annotations

import pandas as pd
from pandas.tseries.frequencies import to_offset


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Restituisce un alias pandas (es. ``D``, ``MS``, ``h``) per le date date.

    Tollera buchi: se ``pd.infer_freq`` fallisce usa la distanza più frequente.
    """
    idx = pd.DatetimeIndex(index).dropna().sort_values().unique()
    if len(idx) < 3:
        raise ValueError("Servono almeno 3 date distinte per riconoscere la frequenza.")
    with pd.option_context("future.infer_freq_returns_offset", True):
        inferred = pd.infer_freq(idx)
    if inferred is not None:
        return inferred.freqstr if hasattr(inferred, "freqstr") else str(inferred)

    step = pd.Series(idx[1:] - idx[:-1]).mode().iloc[0]
    days = step / pd.Timedelta(days=1)
    if 27 <= days <= 31:
        return "MS" if (idx.day == 1).mean() >= 0.8 else "ME"
    if 88 <= days <= 93:
        return "QS" if (idx.is_quarter_start).mean() >= 0.8 else "QE"
    if 363 <= days <= 367:
        return "YS" if (idx.is_year_start).mean() >= 0.8 else "YE"
    if step == pd.Timedelta(days=7):
        return f"W-{idx[0].day_name()[:3].upper()}"  # settimanale ancorato al giorno dei dati
    for unit, alias in (
        (pd.Timedelta(days=1), "D"),
        (pd.Timedelta(hours=1), "h"),
        (pd.Timedelta(minutes=1), "min"),
    ):
        if step % unit == pd.Timedelta(0):
            n = int(step / unit)
            return alias if n == 1 else f"{n}{alias}"
    return to_offset(step).freqstr


def season_length(freq: str) -> int:
    """Lunghezza tipica del ciclo stagionale (in passi) per una frequenza."""
    off = to_offset(freq)
    n = max(int(getattr(off, "n", 1)), 1)
    if isinstance(off, pd.offsets.Minute):
        return max(1440 // n, 1) if 1440 % n == 0 else 1
    if isinstance(off, pd.offsets.Hour):
        return max(24 // n, 1) if 24 % n == 0 else 1
    if isinstance(off, pd.offsets.BusinessDay):
        return 5
    if isinstance(off, pd.offsets.Day):
        return 7 if n == 1 else 1
    if isinstance(off, pd.offsets.Week):
        return 52 if n == 1 else 1
    if isinstance(off, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
        return 12 if n == 1 else 1
    if isinstance(off, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
        return 4 if n == 1 else 1
    return 1


def describe_frequency(freq: str) -> str:
    off = to_offset(freq)
    n = int(getattr(off, "n", 1))
    table = [
        (pd.offsets.Minute, "ogni {n} minuti", "al minuto"),
        (pd.offsets.Hour, "ogni {n} ore", "oraria"),
        (pd.offsets.BusinessDay, "ogni {n} giorni lavorativi", "giornaliera (lavorativi)"),
        (pd.offsets.Day, "ogni {n} giorni", "giornaliera"),
        (pd.offsets.Week, "ogni {n} settimane", "settimanale"),
        (pd.offsets.MonthBegin, "ogni {n} mesi", "mensile"),
        (pd.offsets.MonthEnd, "ogni {n} mesi", "mensile"),
        (pd.offsets.QuarterBegin, "ogni {n} trimestri", "trimestrale"),
        (pd.offsets.QuarterEnd, "ogni {n} trimestri", "trimestrale"),
        (pd.offsets.YearBegin, "ogni {n} anni", "annuale"),
        (pd.offsets.YearEnd, "ogni {n} anni", "annuale"),
    ]
    for cls, many, one in table:
        if isinstance(off, cls):
            return one if n == 1 else many.format(n=n)
    return freq
