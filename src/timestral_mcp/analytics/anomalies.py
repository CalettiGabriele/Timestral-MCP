"""Anomalie: valori reali fuori dall'intervallo atteso, con previsioni fatte solo sui dati precedenti."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from timestral_mcp.analytics.covariates import fmt_date
from timestral_mcp.analytics.forecast import forecast_window, interval_levels
from timestral_mcp.analytics.summary import fmt
from timestral_mcp.config import get_settings
from timestral_mcp.data.ingest import Dataset
from timestral_mcp.models import get_forecaster, normalize_size


@dataclass
class AnomalyPlan:
    ds: Dataset
    series: str
    target: str
    model: str
    interval: float
    block: int
    origins: list[int]
    y: np.ndarray
    index: pd.DatetimeIndex
    past: np.ndarray | None
    known: np.ndarray | None
    flags: pd.DataFrame  # colonne binarie (es. promo) per annotare le anomalie
    warnings: list[str]


@dataclass
class AnomalyResult:
    id: str
    dataset_id: str
    series: str
    target: str
    model: str
    freq: str
    interval: float
    block: int
    windows: int
    points: pd.DataFrame  # date, actual, median, lower, upper, anomaly, score
    anomalies: list[dict[str, Any]]
    expected_by_chance: float
    warnings: list[str] = field(default_factory=list)


def plan_anomalies(
    ds: Dataset,
    *,
    target: str,
    series: str | None = None,
    model: str | None = None,
    interval: float = 0.98,
    lookback: int | None = None,
    block: int | None = None,
    past_covariates: list[str] | None = None,
    known_covariates: list[str] | None = None,
) -> AnomalyPlan:
    key, df = ds.get_series(series)
    if target not in ds.value_columns:
        raise ValueError(f"Target {target!r} non trovato. Colonne numeriche: {', '.join(ds.value_columns)}")
    past_cols, known_cols = list(past_covariates or []), list(known_covariates or [])
    for c in past_cols + known_cols:
        if c not in ds.value_columns or c == target:
            raise ValueError(f"Covariata {c!r} non valida.")
    last = df[target].last_valid_index()
    if last is None:
        raise ValueError(f"La colonna {target!r} non contiene valori.")
    pos = df.index.get_loc(last) + 1
    y = df[target].to_numpy(dtype=float)[:pos]
    block = block or max(ds.season, 14)
    min_context = max(2 * ds.season, 24)
    available = pos - min_context
    if available < block:
        raise ValueError(
            f"Storico troppo corto per cercare anomalie: {pos} punti, ne servono almeno {min_context + block}."
        )
    warnings: list[str] = []
    settings = get_settings()
    lookback = min(lookback or 365, available)
    windows = -(-lookback // block)
    if windows > settings.max_anomaly_windows:
        lookback = settings.max_anomaly_windows * block
        windows = settings.max_anomaly_windows
        warnings.append(
            f"Per contenere i tempi analizzo solo gli ultimi {lookback} periodi ({windows} finestre da {block})."
        )
    start = pos - lookback
    origins = list(range(start, pos, block))
    interval_levels(interval, tuple(np.linspace(0.01, 0.99, 21)))  # valida
    binary = [
        c
        for c in ds.value_columns
        if c != target and set(np.unique(df[c].dropna())) <= {0.0, 1.0} and df[c].notna().any()
    ]
    return AnomalyPlan(
        ds=ds,
        series=key,
        target=target,
        model=normalize_size(model),
        interval=interval,
        block=block,
        origins=origins,
        y=y,
        index=df.index[:pos],
        past=df[past_cols].to_numpy(dtype=float)[:pos].T if past_cols else None,
        known=df[known_cols].to_numpy(dtype=float)[:pos].T if known_cols else None,
        flags=df[binary].iloc[:pos],
        warnings=warnings,
    )


def run_window(plan: AnomalyPlan, origin: int) -> pd.DataFrame:
    h = min(plan.block, len(plan.y) - origin)
    past = plan.past[:, :origin] if plan.past is not None else None
    known = plan.known[:, : origin + h] if plan.known is not None else None
    q, levels = forecast_window(
        get_forecaster(plan.model),
        plan.y[:origin],
        h,
        past=past,
        known=known,
        context_length=get_settings().max_context,
    )
    lo, hi = interval_levels(plan.interval, levels)
    lv = list(levels)
    median, lower, upper = (
        q[:, int(np.argmin(np.abs(np.asarray(lv) - 0.5)))],
        q[:, lv.index(lo)],
        q[:, lv.index(hi)],
    )
    z = NormalDist().inv_cdf(1 - (1 - plan.interval) / 2)
    sigma = np.maximum(upper - lower, 1e-9) / (2 * z)
    actual = plan.y[origin : origin + h]
    score = np.abs(actual - median) / sigma
    out = pd.DataFrame(
        {"actual": actual, "median": median, "lower": lower, "upper": upper, "score": score},
        index=plan.index[origin : origin + h],
    )
    out["anomaly"] = np.isfinite(actual) & ((actual < lower) | (actual > upper))
    return out


def finish_anomalies(
    plan: AnomalyPlan, parts: list[pd.DataFrame], anomaly_id: str, max_report: int = 50
) -> AnomalyResult:
    points = pd.concat(parts)
    points.index.name = "date"
    found = points[points["anomaly"]].sort_values("score", ascending=False)
    freq = plan.ds.freq
    rows = []
    for ts, r in found.head(max_report).iterrows():
        notes = []
        if r["actual"] == 0 and r["median"] > 0:
            notes.append(
                "valore zero con previsione positiva: possibile rottura di stock, chiusura o dato mancante"
            )
        active = [c for c in plan.flags.columns if plan.flags.at[ts, c] == 1]
        if active:
            notes.append(f"indicatori attivi quel giorno: {', '.join(active)}")
        rows.append(
            {
                "date": fmt_date(ts, freq),
                "actual": round(float(r["actual"]), 4),
                "expected_median": round(float(r["median"]), 4),
                "expected_range": [round(float(r["lower"]), 4), round(float(r["upper"]), 4)],
                "direction": "sopra" if r["actual"] > r["upper"] else "sotto",
                "score": round(float(r["score"]), 2),
                "notes": notes,
            }
        )
    n = int(points["actual"].notna().sum())
    warnings = list(plan.warnings)
    if len(found) > max_report:
        warnings.append(f"Trovate {len(found)} anomalie: mostrate le {max_report} più gravi.")
    return AnomalyResult(
        id=anomaly_id,
        dataset_id=plan.ds.id,
        series=plan.series,
        target=plan.target,
        model=plan.model,
        freq=freq,
        interval=plan.interval,
        block=plan.block,
        windows=len(parts),
        points=points,
        anomalies=rows,
        expected_by_chance=round((1 - plan.interval) * n, 1),
        warnings=warnings,
    )


def anomaly_summary(a: AnomalyResult) -> str:
    n = len(a.points)
    total = int(a.points["anomaly"].sum())
    lines = [
        f"Anomalie {a.id} · Timestral-{a.model} · serie '{a.series}' · target '{a.target}'",
        f"Analizzati {n} periodi ({fmt_date(a.points.index[0], a.freq)} → {fmt_date(a.points.index[-1], a.freq)}) in "
        f"{a.windows} finestre da {a.block}: ogni valore è confrontato con una previsione fatta usando solo i dati precedenti. "
        f"Anomalia = fuori dall'intervallo {a.interval:.0%}.",
        f"Trovate {total} anomalie. Per puro caso se ne aspettano circa {a.expected_by_chance:g}: "
        + (
            "il numero è nella norma."
            if total <= max(a.expected_by_chance * 1.5, 1)
            else "ce ne sono più del previsto."
        ),
    ]
    if a.anomalies:
        lines += ["", "data | reale | atteso (mediana) | intervallo | direzione | gravità* | note"]
        for r in a.anomalies[:30]:
            lines.append(
                f"{r['date']} | {fmt(r['actual'])} | {fmt(r['expected_median'])} | "
                f"{fmt(r['expected_range'][0])}–{fmt(r['expected_range'][1])} | {r['direction']} | {r['score']:.1f} | "
                f"{'; '.join(r['notes']) or '-'}"
            )
        lines.append(
            "*gravità = distanza dalla mediana in deviazioni standard equivalenti (stimate dall'ampiezza dell'intervallo)."
        )
    for w in a.warnings:
        lines.append(f"ATTENZIONE: {w}")
    return "\n".join(lines)
