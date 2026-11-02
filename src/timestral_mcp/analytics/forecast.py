"""Previsione con Timestral: da DataFrame a quantili con date."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from timestral_mcp.config import get_settings
from timestral_mcp.data.ingest import Dataset, trailing_future_rows
from timestral_mcp.models import get_forecaster, normalize_size


@dataclass
class ForecastResult:
    id: str
    dataset_id: str
    dataset_name: str
    series: str
    target: str
    model: str
    horizon: int
    freq: str
    interval: float
    history: pd.Series
    future_index: pd.DatetimeIndex
    levels: tuple[float, ...]
    quantiles: np.ndarray  # (H, Q)
    past_covariates: list[str] = field(default_factory=list)
    known_covariates: list[str] = field(default_factory=list)
    context_length: int = 0
    warnings: list[str] = field(default_factory=list)
    known_future: pd.DataFrame | None = None  # valori futuri delle covariate note usati dal modello

    # ---- accesso ai quantili
    def level_index(self, level: float) -> int:
        i = int(np.argmin(np.abs(np.asarray(self.levels) - level)))
        if abs(self.levels[i] - level) > 1e-6:
            raise ValueError(
                f"Quantile {level} non disponibile. Disponibili: {', '.join(f'{x:g}' for x in self.levels)}"
            )
        return i

    def q(self, level: float) -> np.ndarray:
        return self.quantiles[:, self.level_index(level)]

    @property
    def median(self) -> np.ndarray:
        return self.q(0.5)

    @property
    def bounds(self) -> tuple[float, float]:
        return interval_levels(self.interval, self.levels)

    @property
    def lower(self) -> np.ndarray:
        return self.q(self.bounds[0])

    @property
    def upper(self) -> np.ndarray:
        return self.q(self.bounds[1])

    def to_frame(self, levels: list[float] | None = None) -> pd.DataFrame:
        cols = levels or list(self.levels)
        data = {f"q{round(level * 100):02d}": self.q(level) for level in cols}
        df = pd.DataFrame(data, index=self.future_index)
        df.insert(0, "median", self.median)
        df.index.name = "date"
        return df


def interval_levels(interval: float, levels: tuple[float, ...]) -> tuple[float, float]:
    """Quantili (basso, alto) che racchiudono un intervallo centrale (es. 0.8 -> 0.1 e 0.9)."""
    if not 0 < interval < 1:
        raise ValueError("interval deve essere tra 0 e 1 (es. 0.8 per l'intervallo all'80%).")
    arr = np.asarray(levels)
    lo = float(arr[np.argmin(np.abs(arr - (1 - interval) / 2))])
    hi = float(arr[np.argmin(np.abs(arr - (1 - (1 - interval) / 2)))])
    return lo, hi


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def forecast_window(
    forecaster: Any,
    y: np.ndarray,
    horizon: int,
    *,
    past: np.ndarray | None = None,
    known: np.ndarray | None = None,
    context_length: int | None = None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Esegue una previsione. ``y``: (T,), ``past``: (C,T), ``known``: (C,T+H)."""
    ctx = np.asarray(y, dtype=np.float32)
    if context_length and ctx.size > context_length:
        cut = ctx.size - context_length
        ctx = ctx[cut:]
        past = past[:, cut:] if past is not None else None
        known = known[:, cut:] if known is not None else None
    kwargs: dict[str, Any] = {}
    if past is not None and past.shape[0]:
        kwargs["past_covariates"] = past[None].astype(np.float32)
    if known is not None and known.shape[0]:
        kwargs["known_covariates"] = known[None].astype(np.float32)
    fc = forecaster.forecast(ctx[None], horizon, **kwargs)
    q = to_numpy(fc.quantiles)[0, 0]
    q = np.maximum.accumulate(q, axis=-1)  # quantili non decrescenti
    levels = tuple(float(x) for x in fc.quantile_levels)
    return q, levels


def _check_columns(ds: Dataset, target: str, cols: list[str], label: str) -> None:
    for c in cols:
        if c not in ds.value_columns:
            raise ValueError(
                f"{label} {c!r} non trovata. Colonne numeriche disponibili: {', '.join(ds.value_columns)}"
            )
        if c == target:
            raise ValueError(f"{c!r} è il target: non può essere anche una covariata.")


def run_forecast(
    ds: Dataset,
    *,
    forecast_id: str,
    target: str,
    horizon: int,
    series: str | None = None,
    model: str | None = None,
    interval: float = 0.8,
    past_covariates: list[str] | None = None,
    known_covariates: list[str] | None = None,
    context_length: int | None = None,
) -> ForecastResult:
    settings = get_settings()
    key, df = ds.get_series(series)
    if target not in ds.value_columns:
        raise ValueError(f"Target {target!r} non trovato. Colonne numeriche: {', '.join(ds.value_columns)}")
    if horizon < 1:
        raise ValueError("horizon deve essere almeno 1.")
    past_cols = list(past_covariates or [])
    known_cols = list(known_covariates or [])
    _check_columns(ds, target, past_cols, "Covariata passata")
    _check_columns(ds, target, known_cols, "Covariata nota")
    overlap = set(past_cols) & set(known_cols)
    if overlap:
        raise ValueError(f"Covariate sia passate che note: {', '.join(overlap)}")

    y_full = df[target]
    last = y_full.last_valid_index()
    if last is None:
        raise ValueError(f"La colonna {target!r} non contiene valori.")
    pos = df.index.get_loc(last)
    context = y_full.iloc[: pos + 1]
    if len(context) < 8:
        raise ValueError(f"Storico troppo corto ({len(context)} punti): servono almeno 8 osservazioni.")

    warnings: list[str] = []
    if horizon > settings.model_horizon_block:
        warnings.append(
            f"Orizzonte di {horizon} passi: oltre {settings.model_horizon_block} il modello procede a blocchi "
            "usando la mediana come contesto, quindi l'incertezza è sottostimata."
        )
    if len(context) < 2 * ds.season:
        warnings.append(
            f"Storico di {len(context)} punti, meno di due cicli stagionali ({2 * ds.season}): "
            "la stagionalità potrebbe non essere catturata."
        )
    n_missing = int(context.isna().sum())
    if n_missing:
        warnings.append(f"{n_missing} valori mancanti nello storico (gestiti dal modello).")

    past = df[past_cols].iloc[: pos + 1].to_numpy(dtype=float).T if past_cols else None
    known = None
    known_future = None
    if known_cols:
        future = df[known_cols].iloc[pos + 1 : pos + 1 + horizon]
        if len(future) < horizon or future.isna().any().any():
            have = trailing_future_rows(df, target)
            raise ValueError(
                f"Le covariate note ({', '.join(known_cols)}) devono avere valori per tutti i {horizon} periodi "
                f"futuri, cioè righe dopo il {last:%Y-%m-%d} con target vuoto. Nel dataset ce ne sono {have}."
            )
        known = df[known_cols].iloc[: pos + 1 + horizon].to_numpy(dtype=float).T
        known_future = future.copy()

    size = normalize_size(model)
    forecaster = get_forecaster(size)
    ctx_len = min(context_length or settings.max_context, settings.max_context)
    q, levels = forecast_window(
        forecaster, context.to_numpy(dtype=float), horizon, past=past, known=known, context_length=ctx_len
    )
    future_index = pd.date_range(last, periods=horizon + 1, freq=ds.freq)[1:]
    interval_levels(interval, levels)  # valida
    return ForecastResult(
        id=forecast_id,
        dataset_id=ds.id,
        dataset_name=ds.name,
        series=key,
        target=target,
        model=size,
        horizon=horizon,
        freq=ds.freq,
        interval=interval,
        history=context,
        future_index=future_index,
        levels=levels,
        quantiles=q,
        past_covariates=past_cols,
        known_covariates=known_cols,
        context_length=min(len(context), ctx_len),
        warnings=warnings,
        known_future=known_future,
    )
