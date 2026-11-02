"""Stato condiviso della sessione: impostazioni e archivi in memoria (usati da server e Studio)."""

from __future__ import annotations

from timestral_mcp.analytics import anomalies as an
from timestral_mcp.analytics import backtest as bt
from timestral_mcp.analytics import batch as bm
from timestral_mcp.analytics import importance as im
from timestral_mcp.analytics.forecast import ForecastResult
from timestral_mcp.config import get_settings
from timestral_mcp.data.ingest import Dataset
from timestral_mcp.data.store import Store

settings = get_settings()
datasets: Store[Dataset] = Store("ds", settings.max_datasets)
forecasts: Store[ForecastResult] = Store("fc", settings.max_forecasts)
backtests: Store[bt.BacktestResult] = Store("bt", settings.max_forecasts)
batches: Store[bm.BatchResult] = Store("bm", settings.max_batches)
anomaly_runs: Store[an.AnomalyResult] = Store("an", settings.max_batches)
importance_runs: Store[im.ImportanceResult] = Store("im", settings.max_batches)


def resolve_forecast(forecast_id: str) -> ForecastResult:
    """Cerca una previsione singola oppure una serie di un batch (id 'bm_xxx::serie')."""
    if bm.SEP in forecast_id:
        batch_id, series = forecast_id.split(bm.SEP, 1)
        batch = batches.get(batch_id)
        if series not in batch.results:
            raise KeyError(
                f"La serie {series!r} non è nel batch {batch_id}. Serie previste: {', '.join(list(batch.results)[:20])}"
            )
        return batch.results[series]
    return forecasts.get(forecast_id)
