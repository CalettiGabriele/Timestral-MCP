"""Covariate di calendario commerciale (festività, ponti, Black Friday...) come colonne note in anticipo."""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from typing import Any

import holidays
import pandas as pd
from dateutil.easter import easter
from pandas.tseries.frequencies import to_offset

from timestral_mcp.analytics.covariates import fmt_date
from timestral_mcp.data.ingest import Dataset

FEATURES = {
    "holiday": "giorno festivo (festività nazionali del paese)",
    "pre_holiday": "giorno prima di una festività",
    "bridge": "ponte: giorno lavorativo tra una festività e il weekend",
    "black_friday": "Black Friday (venerdì dopo il 4º giovedì di novembre)",
    "christmas": "periodo pre-natalizio (8–24 dicembre)",
    "easter": "settimana di Pasqua (da 7 giorni prima a Pasquetta)",
}
PREFIX = "cal_"


def _country_holidays(country: str, subdiv: str | None, years: range) -> holidays.HolidayBase:
    try:
        return holidays.country_holidays(country.upper(), subdiv=subdiv, years=years)
    except NotImplementedError as exc:
        raise ValueError(
            f"Paese {country!r} non supportato: usa un codice ISO a due lettere come IT, DE, FR, ES, GB, US."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - suddivisione non valida e simili
        raise ValueError(
            f"Impossibile caricare le festività per {country!r} (subdiv={subdiv!r}): {exc}"
        ) from exc


def _black_friday(year: int) -> date:
    d = date(year, 11, 1)
    first_thursday = d + timedelta(days=(3 - d.weekday()) % 7)
    return first_thursday + timedelta(weeks=3, days=1)


def calendar_frame(
    index: pd.DatetimeIndex,
    country: str = "IT",
    subdiv: str | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Colonne 0/1 ``cal_<nome>`` per ogni data dell'indice (si usa la sola parte di data)."""
    wanted = features or list(FEATURES)
    unknown = [f for f in wanted if f not in FEATURES]
    if unknown:
        raise ValueError(f"Feature sconosciute: {', '.join(unknown)}. Disponibili: {', '.join(FEATURES)}.")
    days = [ts.date() for ts in index]
    years = range(min(d.year for d in days) - 1, max(d.year for d in days) + 2)
    hol = _country_holidays(country, subdiv, years)

    def is_off(d: date) -> bool:
        return d.weekday() >= 5 or d in hol

    out: dict[str, list[int]] = {}
    for name in wanted:
        col = []
        for d in days:
            prev, nxt = d - timedelta(days=1), d + timedelta(days=1)
            if name == "holiday":
                v = d in hol
            elif name == "pre_holiday":
                v = nxt in hol and d not in hol
            elif name == "bridge":
                v = not is_off(d) and ((prev in hol and is_off(nxt)) or (nxt in hol and is_off(prev)))
            elif name == "black_friday":
                v = d == _black_friday(d.year)
            elif name == "christmas":
                v = d.month == 12 and 8 <= d.day <= 24
            else:  # easter
                e = easter(d.year)
                v = e - timedelta(days=7) <= d <= e + timedelta(days=1)
            col.append(int(v))
        out[f"{PREFIX}{name}"] = col
    return pd.DataFrame(out, index=index, dtype=float)


def add_calendar_features(
    ds: Dataset,
    *,
    new_id: str,
    country: str = "IT",
    subdiv: str | None = None,
    features: list[str] | None = None,
    extend_future_periods: int = 0,
) -> tuple[Dataset, dict[str, Any]]:
    """Restituisce un nuovo Dataset con le colonne di calendario (anche sulle righe future)."""
    off = to_offset(ds.freq)
    if not isinstance(off, pd.offsets.Day | pd.offsets.Hour | pd.offsets.Minute | pd.offsets.BusinessDay):
        raise ValueError(
            "Il calendario commerciale è disponibile per dati giornalieri o più frequenti; "
            f"questo dataset è {ds.freq_label}."
        )
    if extend_future_periods < 0:
        raise ValueError("extend_future_periods non può essere negativo.")

    new_series: dict[str, pd.DataFrame] = {}
    constant: set[str] = set()
    cols_all: list[str] = []
    old_end = max(df.index.max() for df in ds.series.values())
    for key, df in ds.series.items():
        if extend_future_periods:
            ext = pd.date_range(df.index.max(), periods=extend_future_periods + 1, freq=ds.freq)[1:]
            df = df.reindex(df.index.append(ext))
            df.index.name = "date"
        cal = calendar_frame(df.index, country, subdiv, features)
        df = df.drop(columns=[c for c in cal.columns if c in df.columns]).join(cal)
        new_series[key] = df
        cols_all = list(cal.columns)
        constant |= {c for c in cal.columns if cal[c].nunique() < 2}
    for key, df in new_series.items():
        new_series[key] = df.drop(columns=sorted(constant))
    added = [c for c in cols_all if c not in constant]
    if not added:
        raise ValueError("Le feature richieste sono costanti nel periodo dei dati: nessuna colonna aggiunta.")

    end = max(df.index.max() for df in new_series.values())
    hol = _country_holidays(country, subdiv, range(old_end.year, end.year + 1))
    upcoming = sorted((d, n) for d, n in hol.items() if old_end.date() < d <= end.date())
    new_ds = dataclasses.replace(
        ds,
        id=new_id,
        series=new_series,
        value_columns=[c for c in ds.value_columns if c not in added] + added,
        warnings=list(ds.warnings),
    )
    info = {
        "dataset_id": new_id,
        "derived_from": ds.id,
        "country": country.upper(),
        "added_columns": added,
        "dropped_constant_columns": sorted(constant),
        "descriptions": {f"{PREFIX}{k}": v for k, v in FEATURES.items() if f"{PREFIX}{k}" in added},
        "rows_added_at_end": extend_future_periods,
        "calendar_covers": f"{fmt_date(min(df.index.min() for df in new_series.values()), ds.freq)} → {fmt_date(end, ds.freq)}",
        "holidays_after_original_end": [{"date": f"{d:%Y-%m-%d}", "name": n} for d, n in upcoming[:20]],
        "next_steps": (
            "Usa queste colonne come known_covariates in forecast/backtest (servono righe future: "
            "con extend_future_periods ne vengono create)."
        ),
    }
    return new_ds, info
