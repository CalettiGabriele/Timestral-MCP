"""Regressioni emerse dalla prova reale in Claude Desktop (1 novembre 2026)."""

from pathlib import Path

import pandas as pd
import pytest
from fastmcp import Client

from timestral_mcp import models, server
from timestral_mcp.analytics.covariates import describe_future_values, format_future_values, future_block
from timestral_mcp.analytics.forecast import run_forecast
from timestral_mcp.analytics.profile import profile_dataset
from timestral_mcp.analytics.summary import forecast_details, forecast_summary
from timestral_mcp.data.ingest import build_dataset, read_source

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "vendite_esempio.csv"


@pytest.fixture
def ds():
    df, origin = read_source(str(EXAMPLE))
    return build_dataset(df, dataset_id="ds_ex", name="esempio", source=origin)


def test_future_values_of_binary_covariate_are_described():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    frame = pd.DataFrame({"promo": [0, 0, 1, 1, 0, 0, 1, 0, 0, 0]}, index=idx)
    d = describe_future_values(frame, "D")["promo"]
    assert d["type"] == "binary" and d["active_periods"] == 3
    assert d["active_ranges"] == ["2026-01-03..2026-01-04", "2026-01-07"]
    assert "attiva (=1) in 3 dei 10" in format_future_values({"promo": d})[0]
    zero = describe_future_values(frame * 0, "D")["promo"]
    assert "sempre 0" in format_future_values({"promo": zero})[0]


def test_numeric_future_covariate_is_described():
    frame = pd.DataFrame({"prezzo": [9.9, 10.5, 12.0]}, index=pd.date_range("2026-01-01", periods=3))
    line = format_future_values(describe_future_values(frame, "D"))[0]
    assert "tra 9.9 e 12" in line and "ultimo 12" in line


def test_forecast_summary_states_promo_days(fake, ds):
    """Claude aveva letto 'promo sempre 0': ora il testo dice quando la promo è attiva."""
    r = run_forecast(ds, forecast_id="fc_1", target="vendite", horizon=30, known_covariates=["promo"])
    assert "attiva (=1) in 5 dei 30 periodi futuri: 2026-01-20..2026-01-24" in forecast_summary(r)
    rows = forecast_details(r)["rows"]
    assert {row["date"]: row["promo"] for row in rows}["2026-01-22"] == 1


def test_profile_exposes_future_covariates_and_no_false_alarm(ds):
    p = profile_dataset(ds)
    prof = p["profiles"][0]
    assert prof["future_covariates"]["columns"]["promo"]["active_periods"] == 5
    assert prof["future_covariates"]["from"] == "2025-12-31"
    assert not any("promo" in w and "zeri" in w for w in prof["warnings"])  # colonna 0/1
    assert "future_rows_with_value" not in prof["columns"]["promo"]
    assert prof["columns"]["promo"]["is_binary"] is True
    assert "frequenza" in p["season_length_note"]
    assert future_block(ds.series["serie"][["vendite"]], "D") is None  # nessuna covariata oltre la fine


def test_partial_month_tells_how_far_to_extend(fake, ds):
    r = run_forecast(ds, forecast_id="fc_1", target="vendite", horizon=30)
    d = forecast_details(r, aggregate="month")
    by_period = {p["period"]: p for p in d["partial_periods"]}
    jan = by_period["2026-01-01"]
    assert (jan["covered"], jan["expected"], jan["missing_after"]) == (29, 31, 2)
    assert jan["horizon_for_complete_period"] == 32
    assert by_period["2025-12-01"]["missing_before"] == 30
    assert any("horizon=32" in n for n in d["notes"])
    complete = run_forecast(ds, forecast_id="fc_2", target="vendite", horizon=32)
    d2 = forecast_details(complete, aggregate="month", start="2026-01-01")
    assert d2["partial_periods"] == [] and d2["rows"][0]["periods"] == 31


def test_weekly_partial_periods_are_consistent(fake, ds):
    r = run_forecast(ds, forecast_id="fc_1", target="vendite", horizon=30)
    d = forecast_details(r, aggregate="week")
    assert all(
        p["covered"] + p["missing_before"] + p["missing_after"] == p["expected"] for p in d["partial_periods"]
    )


# ---------------------------------------------------------------- download dei modelli


@pytest.fixture
def no_cache(monkeypatch, fake):
    """Simula: nessun modello sul disco; il 'download' usa il forecaster finto."""
    models.set_factory(None)
    monkeypatch.setattr(models, "is_cached", lambda size: False)
    monkeypatch.setattr(models, "_default_factory", lambda size: fake)
    yield
    models.set_factory(None)


def test_ensure_available_blocks_silent_downloads(no_cache):
    with pytest.raises(models.ModelNotDownloadedError, match=r"837 MB.*allow_download=true"):
        models.ensure_available("209M")
    models.ensure_available("209M", allow_download=True)


async def test_forecast_tool_asks_before_downloading(no_cache):
    server.datasets.clear(), server.forecasts.clear()
    async with Client(server.mcp) as c:
        csv = "date,v\n" + "\n".join(f"2024-01-{d:02d},{d}" for d in range(1, 29))
        ds_id = (await c.call_tool("load_dataset", {"source": csv})).structured_content["dataset_id"]
        with pytest.raises(Exception, match="non è ancora scaricato"):
            await c.call_tool("forecast", {"dataset_id": ds_id, "target": "v", "horizon": 5, "model": "209M"})
        with pytest.raises(Exception, match="non è ancora scaricato"):
            await c.call_tool("backtest", {"dataset_id": ds_id, "target": "v", "horizon": 3, "model": "209M"})
        ok = await c.call_tool(
            "forecast",
            {"dataset_id": ds_id, "target": "v", "horizon": 5, "model": "209M", "allow_download": True},
        )
        assert "Mediana prevista" in ok.content[0].text


def test_preload_skips_uncached_model(monkeypatch):
    calls = []
    monkeypatch.setattr(models, "is_cached", lambda size: False)
    monkeypatch.setattr(models, "get_forecaster", lambda size=None: calls.append(size))
    server._preload()
    assert calls == []


def test_download_command(monkeypatch, capsys):
    got = []
    monkeypatch.setattr(models, "download", lambda size: got.append(size) or f"/cache/{size}")
    monkeypatch.setattr("sys.argv", ["timestral-mcp", "download", "5M", "28M"])
    server.main()
    assert got == ["5M", "28M"]
