import numpy as np
import pandas as pd
import pytest

from timestral_mcp.analytics import backtest as bt
from timestral_mcp.analytics.forecast import interval_levels, run_forecast
from timestral_mcp.analytics.profile import profile_dataset
from timestral_mcp.analytics.summary import backtest_summary, forecast_details, forecast_summary
from timestral_mcp.data.ingest import build_dataset


@pytest.fixture
def ds(daily_df):
    return build_dataset(daily_df, dataset_id="ds_t", name="t", source="test")


def test_forecast_shapes_and_order(fake, ds):
    r = run_forecast(ds, forecast_id="fc_1", target="vendite", horizon=14)
    assert r.quantiles.shape == (14, 21)
    assert (np.diff(r.quantiles, axis=1) >= 0).all()
    assert r.future_index[0] == ds.series["serie"].index[-1] + pd.Timedelta(days=1)
    assert r.lower.shape == (14,) and (r.lower <= r.median).all() and (r.median <= r.upper).all()
    assert interval_levels(0.8, r.levels) == (0.1, 0.9)
    assert interval_levels(0.98, r.levels) == (0.01, 0.99)


def test_forecast_warnings_for_long_horizon_and_short_history(fake, ds):
    r = run_forecast(ds, forecast_id="fc_1", target="vendite", horizon=1100)
    assert any("blocchi" in w for w in r.warnings)


def test_known_covariates_need_future_rows(fake, ds):
    with pytest.raises(ValueError, match="righe dopo"):
        run_forecast(ds, forecast_id="x", target="vendite", horizon=7, known_covariates=["promo"])


def test_covariates_passed_with_right_shapes(fake, daily_df):
    future = pd.DataFrame(
        {"data": pd.date_range("2024-04-30", periods=7).strftime("%Y-%m-%d"), "vendite": np.nan, "promo": 1}
    )
    ds = build_dataset(pd.concat([daily_df, future]), dataset_id="d", name="t", source="t")
    n = len(daily_df)
    run_forecast(ds, forecast_id="x", target="vendite", horizon=7, known_covariates=["promo"])
    call = fake.calls[-1]
    assert call["context"].shape == (1, n)
    assert call["known_covariates"].shape == (1, 1, n + 7)
    assert (call["known_covariates"][0, 0, -7:] == 1).all()
    run_forecast(ds, forecast_id="x", target="vendite", horizon=7, past_covariates=["promo"])
    assert fake.calls[-1]["past_covariates"].shape == (1, 1, n)


def test_invalid_columns(fake, ds):
    with pytest.raises(ValueError, match="non trovato"):
        run_forecast(ds, forecast_id="x", target="nope", horizon=3)
    with pytest.raises(ValueError, match="target"):
        run_forecast(ds, forecast_id="x", target="vendite", horizon=3, past_covariates=["vendite"])


def test_backtest_metrics(fake, ds):
    plan = bt.plan_backtest(ds, target="vendite", horizon=14, n_folds=3)
    assert len(plan.origins) == 3
    folds = [bt.run_fold(plan, o) for o in plan.origins]
    res = bt.finish_backtest(plan, folds, "bt_1")
    assert res.summary["folds"] == 3
    assert 0 < res.summary["mase"] < 3
    assert 0 <= res.summary["interval_coverage_pct"] <= 100
    assert "MASE" in backtest_summary(res)


def test_backtest_reduces_folds_on_short_history(fake, daily_df):
    short = build_dataset(daily_df.head(50), dataset_id="d", name="t", source="t")
    plan = bt.plan_backtest(short, target="vendite", horizon=14, n_folds=5)
    assert len(plan.origins) < 5 and any("Ridotti" in w for w in plan.warnings)
    with pytest.raises(ValueError, match="troppo corto"):
        bt.plan_backtest(short, target="vendite", horizon=45)


def test_summary_and_details(fake, ds):
    r = run_forecast(ds, forecast_id="fc_1", target="vendite", horizon=45)
    text = forecast_summary(r)
    assert "fc_1" in text and "Mediana prevista" in text and "get_forecast_details" in text
    d = forecast_details(r, aggregate="month", how="sum")
    assert [row["periods"] for row in d["rows"]] and any("approssimata" in n for n in d["notes"])
    d2 = forecast_details(r, start="2024-05-01", end="2024-05-03", quantiles=[0.05, 0.95])
    assert len(d2["rows"]) == 3 and {"q05", "q95", "median"} <= set(d2["rows"][0])
    with pytest.raises(ValueError, match="Nessun periodo"):
        forecast_details(r, start="2030-01-01")


def test_profile_flags_quality_issues(daily_df):
    df = daily_df.copy()
    df.loc[10:60, "vendite"] = 0
    ds = build_dataset(df, dataset_id="d", name="t", source="t")
    p = profile_dataset(ds)
    assert p["frequency"] == "D" and p["profiles"][0]["warnings"]
    assert any("zeri" in w for w in p["profiles"][0]["warnings"])
