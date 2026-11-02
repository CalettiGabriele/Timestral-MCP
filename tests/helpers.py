"""Costruisce ForecastResult e Dataset sintetici per i test."""

from __future__ import annotations

import statistics

import numpy as np
import pandas as pd

from tests.conftest import FakeForecast  # noqa: F401  (usato da MarkerAwareFake)
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


class MarkerAwareFake:
    """Forecaster finto che reagisce a covariate 'marcate': valore 7 = utile (+30%), valore 9 = fuorviante (x1.6).

    Qualsiasi altro canale (es. rumore 0/1) viene ignorato. Ripete l'ultima settimana come base.
    """

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0

    def forecast(self, context, prediction_length, **kw):
        import time

        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        y = np.asarray(context, dtype=float)[0]
        cycle = np.where(np.isnan(y[-7:]), np.nanmean(y), y[-7:])
        base = np.resize(cycle, prediction_length)
        mult = np.ones(prediction_length)
        known = kw.get("known_covariates")
        if known is not None:
            fut = np.asarray(known)[0, :, -prediction_length:]
            mult = (
                mult * np.where((fut == 7).any(axis=0), 1.3, 1.0) * np.where((fut == 9).any(axis=0), 1.6, 1.0)
            )
        scale = max(np.nanstd(y[-28:]) * 0.2, 1e-3)
        q = (base * mult)[:, None] + scale * Z[None, :]
        return FakeForecast(q[None, None].astype(np.float32))
