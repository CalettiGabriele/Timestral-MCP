import numpy as np
import pandas as pd
import pytest

from timestral_mcp.analytics import anomalies as an
from timestral_mcp.data.ingest import build_dataset


def make_ds(spike_at=None, zero_at=None, n=200, promo_on=()):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    y = 100 + 10 * np.sin(np.arange(n) * 2 * np.pi / 7) + np.random.default_rng(1).normal(0, 1, n)
    for i in promo_on:
        y[i] += 150
    if spike_at is not None:
        y[spike_at] += 200
    if zero_at is not None:
        y[zero_at] = 0
    promo = np.zeros(n)
    for i in promo_on:
        promo[i] = 1
    df = pd.DataFrame({"d": idx.strftime("%Y-%m-%d"), "vendite": y, "promo": promo})
    return build_dataset(df, dataset_id="ds_an", name="an", source="t")


def run(ds, **kw):
    plan = an.plan_anomalies(ds, target="vendite", **kw)
    parts = [an.run_window(plan, o) for o in plan.origins]
    return plan, an.finish_anomalies(plan, parts, "an_1")


def test_spike_and_stockout_are_found_with_context(fake):
    ds = make_ds(spike_at=150, zero_at=120)
    _, res = run(ds, lookback=100, block=14)
    found = {r["date"]: r for r in res.anomalies}
    spike_day = str(ds.series["serie"].index[150].date())
    zero_day = str(ds.series["serie"].index[120].date())
    assert found[spike_day]["direction"] == "sopra" and found[spike_day]["score"] > 3
    assert found[zero_day]["direction"] == "sotto"
    assert any("rottura di stock" in n for n in found[zero_day]["notes"])
    assert res.anomalies == sorted(res.anomalies, key=lambda r: -r["score"])
    assert len(res.points) == 100 and res.windows == 8  # ceil(100/14)
    text = an.anomaly_summary(res)
    assert spike_day in text and "per puro caso" in text.lower()


def test_flat_series_has_no_more_than_chance(fake):
    ds = make_ds()
    _, res = run(ds, lookback=100, block=14)
    assert len(res.anomalies) <= 3  # ~2% atteso su 100 punti


def test_active_indicator_is_reported_in_notes(fake):
    ds = make_ds(promo_on=(150,))
    _, res = run(ds, lookback=100, block=14)  # senza covariata nota la promo sembra un'anomalia
    day = str(ds.series["serie"].index[150].date())
    row = next(r for r in res.anomalies if r["date"] == day)
    assert any("promo" in n for n in row["notes"])


def test_budget_and_short_history(fake, monkeypatch):
    ds = make_ds(n=400)
    plan = an.plan_anomalies(ds, target="vendite", lookback=365, block=14)
    assert len(plan.origins) == 27  # ceil(365/14)
    monkeypatch.setenv("TIMESTRAL_MAX_ANOMALY_WINDOWS", "5")
    capped = an.plan_anomalies(ds, target="vendite", lookback=365, block=14)
    assert len(capped.origins) == 5 and any("Per contenere i tempi" in w for w in capped.warnings)
    short = make_ds(n=30)
    with pytest.raises(ValueError, match="troppo corto"):
        an.plan_anomalies(short, target="vendite")
    with pytest.raises(ValueError, match="non valida"):
        an.plan_anomalies(ds, target="vendite", known_covariates=["vendite"])
