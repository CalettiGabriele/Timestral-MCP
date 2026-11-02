"""Lettura di CSV/Excel/testo e costruzione di un :class:`Dataset` di serie regolari."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from timestral_mcp.data.frequency import describe_frequency, infer_frequency, season_length

DATE_HINTS = (
    "date",
    "data",
    "time",
    "ds",
    "timestamp",
    "giorno",
    "mese",
    "periodo",
    "week",
    "settimana",
    "day",
    "month",
    "anno",
)
TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"}
DEFAULT_SERIES_KEY = "serie"

_IT_NUMBER = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")
_COMMA_NUMBER = re.compile(r"^-?\d+,\d+$")
_YEAR_FIRST = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")


@dataclass
class Dataset:
    id: str
    name: str
    source: str
    date_column: str
    series_column: str | None
    freq: str
    series: dict[str, pd.DataFrame]
    value_columns: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def freq_label(self) -> str:
        return describe_frequency(self.freq)

    @property
    def season(self) -> int:
        return season_length(self.freq)

    def get_series(self, key: str | None) -> tuple[str, pd.DataFrame]:
        if key is None:
            if len(self.series) == 1:
                key = next(iter(self.series))
            else:
                raise ValueError(
                    f"Il dataset contiene {len(self.series)} serie: indica quale usare con "
                    f"'series'. Disponibili: {', '.join(list(self.series)[:20])}"
                )
        if key not in self.series:
            raise ValueError(f"Serie {key!r} non trovata. Disponibili: {', '.join(list(self.series)[:20])}")
        return key, self.series[key]


# --------------------------------------------------------------------------- lettura


def read_source(source: str, sheet: str | None = None, max_rows: int = 2_000_000) -> tuple[pd.DataFrame, str]:
    """Legge un file (percorso) oppure dati incollati come testo CSV."""
    text = source.strip()
    if not text:
        raise ValueError("Sorgente dati vuota.")
    looks_like_path = "\n" not in text and len(text) < 1024
    if looks_like_path:
        path = Path(text).expanduser()
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix not in TABLE_SUFFIXES:
                raise ValueError(f"Formato {suffix!r} non supportato. Usa CSV o Excel.")
            if suffix in {".xlsx", ".xlsm", ".xls"}:
                df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
            else:
                df = pd.read_csv(path, sep=None, engine="python")
            origin = str(path)
        elif path.suffix.lower() in TABLE_SUFFIXES or text.startswith(("/", "~", ".")):
            raise FileNotFoundError(
                f"File non trovato: {path}. Indica il percorso completo oppure incolla i dati come testo."
            )
        else:
            df, origin = _read_text(text), "testo incollato"
    else:
        df, origin = _read_text(text), "testo incollato"
    if len(df) > max_rows:
        raise ValueError(f"Troppe righe ({len(df):,}): il limite è {max_rows:,}.")
    df.columns = [str(c).strip() for c in df.columns]
    return df, origin


def _read_text(text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - messaggio comprensibile per l'utente
        raise ValueError(f"Impossibile leggere i dati incollati come tabella: {exc}") from exc
    if df.shape[1] < 2:
        raise ValueError("Servono almeno due colonne (data e valore).")
    return df


# --------------------------------------------------------------------------- tipi


def _to_number(s: pd.Series) -> pd.Series | None:
    """Converte stringhe numeriche (anche formato italiano 1.234,56) in float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.astype("string").str.strip().str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    present = txt.notna() & (txt != "")
    if not present.any():
        return None
    out = txt.copy()
    it = txt.str.match(_IT_NUMBER, na=False)
    out[it] = txt[it].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    comma = txt.str.match(_COMMA_NUMBER, na=False) & ~it
    out[comma] = txt[comma].str.replace(",", ".", regex=False)
    num = pd.to_numeric(out, errors="coerce")
    if num[present].notna().mean() < 0.9:
        return None
    return num.astype(float)


def _parse_dates(s: pd.Series, dayfirst: bool | None) -> tuple[pd.Series, bool | None, bool]:
    """Parsa una colonna di date. Ritorna (date, dayfirst usato, ambiguo)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s), None, False
    txt = s.astype("string").str.strip()
    if txt.str.match(_YEAR_FIRST, na=False).mean() >= 0.9:  # 2024-03-05: anno per primo, inequivocabile
        return pd.to_datetime(txt, errors="coerce", dayfirst=False, format="mixed"), False, False
    a = pd.to_datetime(txt, errors="coerce", dayfirst=False, format="mixed")
    b = pd.to_datetime(txt, errors="coerce", dayfirst=True, format="mixed")
    if dayfirst is True:
        return b, True, False
    if dayfirst is False:
        return a, False, False
    ok_a, ok_b = a.notna().mean(), b.notna().mean()
    if ok_b > ok_a:
        return b, True, False
    if ok_a > ok_b:
        return a, False, False
    if a.equals(b):
        return a, None, False
    return b, True, True  # ambiguo (es. 01/02/2024): si assume giorno/mese


def _looks_like_date_column(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False
    parsed, _, _ = _parse_dates(s, None)
    return parsed.notna().mean() >= 0.9


def detect_date_column(df: pd.DataFrame) -> str:
    hinted = [c for c in df.columns if any(h == c.lower() or h in c.lower() for h in DATE_HINTS)]
    for c in hinted + [c for c in df.columns if c not in hinted]:
        if _looks_like_date_column(df[c]):
            return c
    raise ValueError(
        "Nessuna colonna di date riconosciuta. Colonne disponibili: "
        f"{', '.join(df.columns)}. Indica quale usare con 'date_column'."
    )


def _candidate_series_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    out = []
    for c in df.columns:
        if c in exclude or pd.api.types.is_float_dtype(df[c]):
            continue
        nunique = df[c].nunique(dropna=True)
        textual = not pd.api.types.is_numeric_dtype(df[c])
        if 1 < nunique <= max(len(df) // 2, 2) and (textual or nunique <= 50):
            out.append(c)
    return out


# --------------------------------------------------------------------------- dataset


def build_dataset(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    name: str,
    source: str,
    date_column: str | None = None,
    series_column: str | None = None,
    value_columns: list[str] | None = None,
    dayfirst: bool | None = None,
    freq: str | None = None,
    aggregate_duplicates: Literal["sum", "mean", "last"] | None = None,
) -> Dataset:
    warnings: list[str] = []
    df = df.copy()
    for col in (date_column, series_column, *(value_columns or [])):
        if col is not None and col not in df.columns:
            raise ValueError(f"Colonna {col!r} non trovata. Colonne disponibili: {', '.join(df.columns)}")

    date_column = date_column or detect_date_column(df)
    dates, used_dayfirst, ambiguous = _parse_dates(df[date_column], dayfirst)
    if ambiguous:
        warnings.append(
            f"Il formato delle date in '{date_column}' è ambiguo (es. 01/02/2024): ho assunto giorno/mese. "
            "Se non è corretto, ripeti il caricamento con dayfirst=false."
        )
    bad = int(dates.isna().sum())
    if bad:
        warnings.append(f"{bad} righe con data non valida sono state scartate.")
    df[date_column] = dates
    df = df[dates.notna()]
    if df.empty:
        raise ValueError(f"Nessuna data valida nella colonna {date_column!r}.")

    if series_column is None and df[date_column].duplicated().any() and aggregate_duplicates is None:
        candidates = _candidate_series_columns(df, {date_column})
        hint = f" Possibili colonne identificative: {', '.join(candidates)}." if candidates else ""
        raise ValueError(
            "Ci sono date ripetute: i dati contengono più serie oppure più righe per lo stesso periodo."
            f"{hint} Indica 'series_column' per separarle, oppure 'aggregate_duplicates' "
            "(sum, mean o last) per sommarle/mediarle."
        )

    exclude = {date_column} | ({series_column} if series_column else set())
    candidates = value_columns or [c for c in df.columns if c not in exclude]
    numeric: dict[str, pd.Series] = {}
    for c in candidates:
        conv = _to_number(df[c])
        if conv is not None:
            numeric[c] = conv
        elif value_columns:
            raise ValueError(f"La colonna {c!r} non è numerica.")
    if not numeric:
        raise ValueError("Nessuna colonna numerica trovata da prevedere.")
    skipped = [c for c in candidates if c not in numeric]
    if skipped:
        warnings.append(f"Colonne non numeriche ignorate: {', '.join(skipped)}.")
    values = pd.DataFrame(numeric, index=df.index)
    values.insert(0, date_column, df[date_column])
    if series_column:
        values.insert(1, series_column, df[series_column].astype("string").fillna("(vuoto)"))

    groups = (
        {str(k): g.drop(columns=[series_column]) for k, g in values.groupby(series_column, sort=True)}
        if series_column
        else {DEFAULT_SERIES_KEY: values}
    )

    value_cols = list(numeric)
    raw: dict[str, pd.DataFrame] = {}
    for key, g in groups.items():
        g = g.sort_values(date_column)
        if g[date_column].duplicated().any():
            if aggregate_duplicates is None:
                raise ValueError(
                    f"La serie {key!r} ha date ripetute: usa 'aggregate_duplicates' (sum, mean o last)."
                )
            g = g.groupby(date_column, as_index=False).agg(aggregate_duplicates)
        raw[key] = g.set_index(date_column)[value_cols]

    longest = max(raw.values(), key=len)
    freq = freq or infer_frequency(longest.index)
    series: dict[str, pd.DataFrame] = {}
    off_grid = 0
    for key, g in raw.items():
        full = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        off_grid += int((~g.index.isin(full)).sum())
        series[key] = g.reindex(full)
        series[key].index.name = "date"
    if off_grid:
        warnings.append(
            f"{off_grid} righe non sono allineate alla frequenza {describe_frequency(freq)} e sono state ignorate."
        )

    return Dataset(
        id=dataset_id,
        name=name,
        source=source,
        date_column=date_column,
        series_column=series_column,
        freq=freq,
        series=series,
        value_columns=value_cols,
        warnings=warnings,
    )


def last_observed(s: pd.Series) -> pd.Timestamp | None:
    return s.last_valid_index()


def trailing_future_rows(df: pd.DataFrame, target: str) -> int:
    """Righe dopo l'ultimo valore osservato del target (possono contenere covariate future)."""
    idx = df[target].last_valid_index()
    if idx is None:
        return 0
    return int(len(df) - df.index.get_loc(idx) - 1)


def finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]
