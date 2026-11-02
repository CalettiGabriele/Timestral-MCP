from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

os.environ["TIMESTRAL_PRELOAD"] = "0"

LEVELS = (0.01, 0.05, *[round(0.10 + 0.05 * i, 2) for i in range(17)], 0.95, 0.99)
_Z = np.array([statistics.NormalDist().inv_cdf(p) for p in LEVELS])


@dataclass
class FakeForecast:
    quantiles: np.ndarray
    quantile_levels: tuple = LEVELS


@dataclass
class FakeForecaster:
    """Ripete l'ultima settimana con un'incertezza crescente. Registra le chiamate."""

    calls: list = field(default_factory=list)

    def forecast(self, context, prediction_length, **kwargs):
        self.calls.append({"context": np.asarray(context), "H": prediction_length, **kwargs})
        y = np.asarray(context, dtype=float)[0]
        fill = np.nanmean(y)
        cycle = np.where(np.isnan(y[-7:]), fill, y[-7:])
        base = np.resize(cycle, prediction_length)
        scale = max(np.nanstd(y[-28:]) * 0.3, 1e-3) * (1 + 0.05 * np.arange(prediction_length))
        q = base[:, None] + scale[:, None] * _Z[None, :]
        return FakeForecast(q[None, None].astype(np.float32))


@pytest.fixture
def fake():
    from timestral_mcp import models

    f = FakeForecaster()
    models.set_factory(lambda size: f)
    yield f
    models.set_factory(None)


@pytest.fixture
def daily_df():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    sales = 100 + 20 * np.sin(np.arange(120) * 2 * np.pi / 7) + rng.normal(0, 3, 120)
    return pd.DataFrame(
        {
            "data": idx.strftime("%Y-%m-%d"),
            "vendite": sales.round(2),
            "promo": (np.arange(120) % 14 == 0).astype(int),
        }
    )
