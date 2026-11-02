import numpy as np
import pandas as pd
import pytest

from timestral_mcp.analytics.calendar import add_calendar_features, calendar_frame
from timestral_mcp.analytics.forecast import run_forecast
from timestral_mcp.data.ingest import build_dataset


def day(frame, d, col):
    return int(frame.loc[d, f"cal_{col}"])


def test_italian_calendar_features():
    idx = pd.date_range("2026-01-01", "2026-12-31", freq="D")
    f = calendar_frame(idx, "IT")
    assert day(f, "2026-04-06", "holiday") == 1  # Lunedì dell'Angelo
    assert day(f, "2026-12-25", "holiday") == 1 and day(f, "2026-03-10", "holiday") == 0
    assert day(f, "2026-12-24", "pre_holiday") == 1
    assert day(f, "2026-06-01", "bridge") == 1  # lunedì prima del 2 giugno (martedì)
    assert day(f, "2026-06-02", "bridge") == 0  # la festività stessa non è un ponte
    assert day(f, "2026-11-27", "black_friday") == 1 and f["cal_black_friday"].sum() == 1
    assert day(f, "2026-12-08", "christmas") == 1 and day(f, "2026-12-25", "christmas") == 0
    assert day(f, "2026-04-05", "easter") == 1 and day(f, "2026-03-29", "easter") == 1
    assert day(f, "2026-03-28", "easter") == 0


def test_unknown_country_and_feature():
    idx = pd.date_range("2026-01-01", periods=5)
    with pytest.raises(ValueError, match="non supportato"):
        calendar_frame(idx, "ZZ")
    with pytest.raises(ValueError, match="sconosciute"):
        calendar_frame(idx, "IT", features=["nope"])


def _ds(n=420, with_target_future=0):
    idx = pd.date_range("2025-01-01", periods=n + with_target_future, freq="D")
    y = 100 + np.sin(np.arange(len(idx)) * 2 * np.pi / 7) * 10
    y[n:] = np.nan
    return build_dataset(
        pd.DataFrame({"data": idx.strftime("%Y-%m-%d"), "vendite": y}),
        dataset_id="ds_a",
        name="a",
        source="t",
    )


def test_add_calendar_extends_future_and_feeds_forecast(fake):
    ds = _ds()
    new, info = add_calendar_features(ds, new_id="ds_b", country="IT", extend_future_periods=60)
    s = new.series["serie"]
    assert len(s) == len(ds.series["serie"]) + 60 and new.id == "ds_b" and ds.id == "ds_a"
    assert set(info["added_columns"]) <= set(new.value_columns) and "cal_holiday" in info["added_columns"]
    assert info["holidays_after_original_end"] and info["rows_added_at_end"] == 60
    assert "cal_holiday" not in ds.series["serie"].columns  # l'originale non cambia
    r = run_forecast(new, forecast_id="fc", target="vendite", horizon=30, known_covariates=["cal_holiday"])
    assert r.known_future is not None and r.known_future.shape == (30, 1)
    assert fake.calls[-1]["known_covariates"].shape == (1, 1, len(ds.series["serie"]) + 30)


def test_constant_features_are_dropped_and_coarse_frequency_rejected():
    short = _ds(n=20)  # 20 giorni di gennaio: niente Black Friday né Natale né Pasqua
    new, info = add_calendar_features(short, new_id="ds_c", features=["holiday", "black_friday", "christmas"])
    assert "cal_black_friday" in info["dropped_constant_columns"] and "cal_holiday" in info["added_columns"]
    idx = pd.date_range("2024-01-01", periods=24, freq="MS")
    monthly = build_dataset(
        pd.DataFrame({"d": idx.strftime("%Y-%m-%d"), "v": np.arange(24.0)}),
        dataset_id="m",
        name="m",
        source="t",
    )
    with pytest.raises(ValueError, match="giornalieri"):
        add_calendar_features(monthly, new_id="x")
    with pytest.raises(ValueError, match="costanti"):
        add_calendar_features(short, new_id="y", features=["black_friday"])
