"""Backtest a origine mobile: quanto avrebbe sbagliato il modello sul passato."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from timestral_mcp.analytics.forecast import forecast_window, interval_levels
from timestral_mcp.config import get_settings
from timestral_mcp.data.ingest import Dataset
from timestral_mcp.models import get_forecaster, normalize_size


@dataclass
class Fold:
    origin: pd.Timestamp
    dates: pd.DatetimeIndex
    actual: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mase: float | None
    naive_mase: float | None
    wql: float | None
    coverage: float | None


@dataclass
class BacktestResult:
    id: str
    dataset_id: str
    series: str
    target: str
    model: str
    horizon: int
    interval: float
    season: int
    folds: list[Fold]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass
class BacktestPlan:
    ds: Dataset
    series: str
    target: str
    model: str
    horizon: int
    interval: float
    origins: list[int]
    y: np.ndarray
    index: pd.DatetimeIndex
    past: np.ndarray | None
    known: np.ndarray | None
    past_cols: list[str]
    known_cols: list[str]
    warnings: list[str]


def plan_backtest(
    ds: Dataset,
    *,
    target: str,
    horizon: int,
    series: str | None = None,
    model: str | None = None,
    n_folds: int = 3,
    interval: float = 0.8,
    past_covariates: list[str] | None = None,
    known_covariates: list[str] | None = None,
) -> BacktestPlan:
    key, df = ds.get_series(series)
    if target not in ds.value_columns:
        raise ValueError(f"Target {target!r} non trovato. Colonne numeriche: {', '.join(ds.value_columns)}")
    if horizon < 1 or n_folds < 1:
        raise ValueError("horizon e n_folds devono essere almeno 1.")
    past_cols, known_cols = list(past_covariates or []), list(known_covariates or [])
    for c in past_cols + known_cols:
        if c not in ds.value_columns or c == target:
            raise ValueError(f"Covariata {c!r} non valida.")
    last = df[target].last_valid_index()
    if last is None:
        raise ValueError(f"La colonna {target!r} non contiene valori.")
    pos = df.index.get_loc(last) + 1
    y = df[target].to_numpy(dtype=float)[:pos]
    min_context = max(2 * ds.season, 24)

    warnings: list[str] = []
    folds = n_folds
    while folds > 1 and pos - folds * horizon < min_context:
        folds -= 1
    if pos - folds * horizon < min_context:
        raise ValueError(
            f"Storico troppo corto per un backtest con orizzonte {horizon}: {pos} punti, "
            f"ne servono almeno {min_context + horizon}. Prova un orizzonte più breve."
        )
    if folds < n_folds:
        warnings.append(f"Ridotti i fold da {n_folds} a {folds} per la lunghezza dello storico.")
    origins = [pos - (folds - i) * horizon for i in range(folds)]

    past = df[past_cols].to_numpy(dtype=float)[:pos].T if past_cols else None
    known = df[known_cols].to_numpy(dtype=float)[:pos].T if known_cols else None
    return BacktestPlan(
        ds,
        key,
        target,
        normalize_size(model),
        horizon,
        interval,
        origins,
        y,
        df.index[:pos],
        past,
        known,
        past_cols,
        known_cols,
        warnings,
    )


def _pinball(actual: np.ndarray, q: np.ndarray, levels: tuple[float, ...]) -> float | None:
    mask = np.isfinite(actual)
    if not mask.any():
        return None
    a = actual[mask][:, None]
    qq = q[mask]
    lv = np.asarray(levels)[None, :]
    loss = np.maximum(lv * (a - qq), (lv - 1) * (a - qq)).mean()
    denom = np.abs(actual[mask]).mean()
    return float(2 * loss / denom) if denom > 0 else None  # weighted quantile loss


def _mase(actual: np.ndarray, pred: np.ndarray, train: np.ndarray, m: int) -> float | None:
    mask = np.isfinite(actual)
    t = train[np.isfinite(train)]
    if not mask.any() or t.size <= m:
        return None
    scale = np.abs(t[m:] - t[:-m]).mean()
    if scale == 0:
        return None
    return float(np.abs(actual[mask] - pred[mask]).mean() / scale)


def _seasonal_naive(train: np.ndarray, horizon: int, m: int) -> np.ndarray:
    clean = train.copy()
    if m <= 1 or clean.size < m:
        last = clean[np.isfinite(clean)][-1]
        return np.full(horizon, last)
    cycle = clean[-m:]
    fill = np.nanmedian(cycle) if np.isfinite(cycle).any() else 0.0
    cycle = np.where(np.isfinite(cycle), cycle, fill)
    return np.resize(cycle, horizon)


def run_fold(plan: BacktestPlan, origin: int) -> Fold:
    H = plan.horizon
    m = plan.ds.season if plan.ds.season > 1 else 1
    train = plan.y[:origin]
    actual = plan.y[origin : origin + H]
    past = plan.past[:, :origin] if plan.past is not None else None
    known = plan.known[:, : origin + H] if plan.known is not None else None
    if known is not None and np.isnan(known[:, origin : origin + H]).any():
        raise ValueError("Le covariate note hanno valori mancanti nei periodi del backtest.")
    forecaster = get_forecaster(plan.model)
    q, levels = forecast_window(
        forecaster, train, H, past=past, known=known, context_length=get_settings().max_context
    )
    lo, hi = interval_levels(plan.interval, levels)
    li, hi_i, mi = levels.index(lo), levels.index(hi), int(np.argmin(np.abs(np.asarray(levels) - 0.5)))
    median, lower, upper = q[:, mi], q[:, li], q[:, hi_i]
    naive = _seasonal_naive(train, H, m)
    mask = np.isfinite(actual)
    cov = (
        float(((actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])).mean()) if mask.any() else None
    )
    return Fold(
        origin=plan.index[origin - 1],
        dates=plan.index[origin : origin + H] if origin + H <= len(plan.index) else pd.DatetimeIndex([]),
        actual=actual,
        median=median,
        lower=lower,
        upper=upper,
        mase=_mase(actual, median, train, m),
        naive_mase=_mase(actual, naive, train, m),
        wql=_pinball(actual, q, levels),
        coverage=cov,
    )


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def finish_backtest(plan: BacktestPlan, folds: list[Fold], backtest_id: str) -> BacktestResult:
    mase = _mean([f.mase for f in folds])
    naive = _mean([f.naive_mase for f in folds])
    skill = 1 - mase / naive if mase is not None and naive else None
    summary = {
        "folds": len(folds),
        "mase": mase,
        "naive_mase": naive,
        "skill_vs_seasonal_naive_pct": None if skill is None else 100 * skill,
        "wql": _mean([f.wql for f in folds]),
        "interval_nominal_pct": 100 * plan.interval,
        "interval_coverage_pct": None
        if _mean([f.coverage for f in folds]) is None
        else 100 * _mean([f.coverage for f in folds]),
    }
    warnings = list(plan.warnings)
    cov = summary["interval_coverage_pct"]
    if cov is not None and cov < summary["interval_nominal_pct"] - 15:
        warnings.append(
            f"Gli intervalli all'{summary['interval_nominal_pct']:.0f}% hanno contenuto solo il {cov:.0f}% dei valori reali: "
            "l'incertezza potrebbe essere sottostimata su questi dati."
        )
    if skill is not None and skill < 0:
        warnings.append(
            "Il modello ha fatto peggio di una semplice ripetizione dell'ultimo ciclo stagionale."
        )
    return BacktestResult(
        id=backtest_id,
        dataset_id=plan.ds.id,
        series=plan.series,
        target=plan.target,
        model=plan.model,
        horizon=plan.horizon,
        interval=plan.interval,
        season=plan.ds.season,
        folds=folds,
        summary=summary,
        warnings=warnings,
    )
