"""Profilo di qualità e statistiche di un dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timestral_mcp.analytics.covariates import fmt_date, future_block
from timestral_mcp.data.ingest import Dataset, trailing_future_rows


def _num(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def _column_profile(df: pd.DataFrame, col: str, season: int, freq: str) -> dict[str, Any]:
    s = df[col]
    last = s.last_valid_index()
    obs = s.loc[:last] if last is not None else s.iloc[0:0]
    vals = obs.dropna().to_numpy(dtype=float)
    out: dict[str, Any] = {
        "points": int(len(obs)),
        "observed": int(len(vals)),
        "missing_inside": int(obs.isna().sum()),
        "missing_pct": round(100 * float(obs.isna().mean()), 2) if len(obs) else None,
        "last_value_date": fmt_date(last, freq) if last is not None else None,
        "is_binary": bool(vals.size and set(np.unique(vals)) <= {0.0, 1.0}),
    }
    if vals.size:
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        outliers = int(((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
        out.update(
            min=_num(vals.min()),
            max=_num(vals.max()),
            mean=_num(vals.mean()),
            std=_num(vals.std()),
            median=_num(np.median(vals)),
            zeros_pct=round(100 * float((vals == 0).mean()), 2),
            negatives=int((vals < 0).sum()),
            outliers_3iqr=outliers,
            last_value=_num(vals[-1]),
        )
        if vals.size > 2 * season and season > 1:
            tail = vals[-season:]
            prev = vals[-2 * season : -season]
            if prev.sum():
                out["last_cycle_vs_previous_pct"] = round(100 * (tail.sum() / prev.sum() - 1), 2)
    return out


def profile_series(ds: Dataset, key: str) -> dict[str, Any]:
    df = ds.series[key]
    cols = {c: _column_profile(df, c, ds.season, ds.freq) for c in ds.value_columns}
    warnings: list[str] = []
    for c, p in cols.items():
        if p["observed"] == 0:
            warnings.append(f"'{c}' non ha valori.")
            continue
        if p["observed"] < 2 * ds.season:
            warnings.append(
                f"'{c}': {p['observed']} osservazioni, meno di due cicli stagionali ({2 * ds.season})."
            )
        if p.get("zeros_pct", 0) >= 30 and not p["is_binary"]:
            warnings.append(
                f"'{c}': {p['zeros_pct']}% di zeri, la domanda potrebbe essere intermittente "
                "o ci sono rotture di stock/dati mancanti codificati come 0."
            )
        if p.get("std") == 0:
            warnings.append(f"'{c}' è costante.")
        if p["missing_pct"] and p["missing_pct"] > 10:
            warnings.append(f"'{c}': {p['missing_pct']}% di valori mancanti.")
        if p.get("outliers_3iqr"):
            warnings.append(f"'{c}': {p['outliers_3iqr']} valori molto anomali (oltre 3 IQR).")
    return {
        "series": key,
        "start": f"{df.index.min():%Y-%m-%d %H:%M}".replace(" 00:00", ""),
        "end": f"{df.index.max():%Y-%m-%d %H:%M}".replace(" 00:00", ""),
        "rows": int(len(df)),
        "columns": cols,
        "rows_without_value_at_end": {c: trailing_future_rows(df, c) for c in ds.value_columns},
        "future_covariates": future_block(df, ds.freq),
        "warnings": warnings,
    }


def profile_dataset(ds: Dataset, series: str | None = None, max_series: int = 20) -> dict[str, Any]:
    keys = [series] if series else list(ds.series)
    if series and series not in ds.series:
        raise ValueError(f"Serie {series!r} non trovata.")
    shown = keys[:max_series]
    return {
        "dataset_id": ds.id,
        "name": ds.name,
        "source": ds.source,
        "date_column": ds.date_column,
        "series_column": ds.series_column,
        "frequency": ds.freq,
        "frequency_label": ds.freq_label,
        "season_length": ds.season,
        "season_length_note": "assunto dalla frequenza dei dati, non rilevato dall'analisi",
        "numeric_columns": ds.value_columns,
        "n_series": len(ds.series),
        "series_shown": len(shown),
        "profiles": [profile_series(ds, k) for k in shown],
        "warnings": ds.warnings,
    }
