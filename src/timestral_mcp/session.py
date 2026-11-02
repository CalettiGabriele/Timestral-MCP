"""Stato condiviso della sessione: impostazioni e archivi in memoria (usati da server e Studio)."""

from __future__ import annotations

from timestral_mcp.analytics import anomalies as an
from timestral_mcp.analytics import backtest as bt
from timestral_mcp.analytics.forecast import ForecastResult
from timestral_mcp.config import get_settings
from timestral_mcp.data.ingest import Dataset
from timestral_mcp.data.store import Store

settings = get_settings()
datasets: Store[Dataset] = Store("ds", settings.max_datasets)
forecasts: Store[ForecastResult] = Store("fc", settings.max_forecasts)
backtests: Store[bt.BacktestResult] = Store("bt", settings.max_forecasts)
anomaly_runs: Store[an.AnomalyResult] = Store("an", settings.max_batches)
