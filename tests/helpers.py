"""Costruisce ForecastResult e Dataset sintetici per i test."""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd

from timestral_mcp.analytics.forecast import ForecastResult

LEVELS = (0.01, 0.05, *[round(0.10 + 0.05 * i, 2) for i in range(17)], 0.95, 0.99)
Z = np.array([statistics.NormalDist().inv_cdf(p) for p in LEVELS])


def normal_forecast(means, sigma=10.0, start="2026-01-01", freq="D", fid="fc_t") -> ForecastResult:
    means = np.asarray(means, dtype=float)
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), means.shape)
    q = means[:, None] + sig[:, None] * Z[None, :]
    idx = pd.date_range(start, periods=len(means), freq=freq)
    hist = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range(end=idx[0] - pd.Timedelta(days=1), periods=50, freq=freq),
    )
    return ForecastResult(
        id=fid,
        dataset_id="ds_t",
        dataset_name="t",
        series="serie",
        target="v",
        model="5M",
        horizon=len(means),
        freq=freq,
        interval=0.8,
        history=hist,
        future_index=idx,
        levels=LEVELS,
        quantiles=q,
    )
