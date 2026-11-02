"""End-to-end (client MCP in memoria) dei tool per più serie e per le decisioni."""

import json

import numpy as np
import pandas as pd
import pytest
from fastmcp import Client

from timestral_mcp import models
from timestral_mcp.server import anomaly_runs, backtests, batches, datasets, forecasts, mcp, settings


@pytest.fixture(autouse=True)
def _clean(fake, tmp_path):
    for store in (datasets, forecasts, backtests, batches, anomaly_runs):
        store.clear()
    object.__setattr__(settings, "export_dir", tmp_path / "exports")


def long_csv(n=200, products=("A", "B", "C"), promo=True, spike=None):
    idx = pd.date_range("2025-03-01", periods=n + 30, freq="D")
    rng = np.random.default_rng(5)
    frames = []
    for k, p in enumerate(products):
        y = 40 * (k + 1) + 8 * np.sin(np.arange(len(idx)) * 2 * np.pi / 7) + rng.normal(0, 1, len(idx))
        if spike is not None and p == products[0]:
            y[spike] += 300
        y[n:] = np.nan
        df = pd.DataFrame({"data": idx.strftime("%Y-%m-%d"), "sku": p, "qty": y.round(2)})
        if promo:
            df["promo"] = (np.arange(len(idx)) % 20 == 0).astype(int)
            df.loc[n + 5 : n + 8, "promo"] = 1
        frames.append(df)
    return pd.concat(frames).to_csv(index=False)


async def load(c, text, **kw):
    return (await c.call_tool("load_dataset", {"source": text, **kw})).structured_content


async def test_new_tools_are_listed():
    async with Client(mcp) as c:
        tools = {t.name for t in await c.list_tools()}
        assert {
            "forecast_many",
            "batch_series_forecast",
            "export_batch",
            "add_calendar_features",
            "scenario_forecast",
            "threshold_probability",
            "quantity_for_service_level",
            "detect_anomalies",
        } <= tools


async def test_forecast_many_flow_and_batch_ids(tmp_path):
    async with Client(mcp) as c:
        ld = await load(c, long_csv(), series_column="sku")
        assert ld["n_series"] == 3
        r = await c.call_tool(
            "forecast_many",
            {"dataset_id": ld["dataset_id"], "target": "qty", "horizon": 14, "known_covariates": ["promo"]},
        )
        text = r.content[0].text
        assert "Serie previste: 3" in text and "Covariate note usate: promo" in text
        ui = json.dumps(r.structured_content, default=str)
        assert "DataTable" in ui and "BarChart" in ui
        batch_id = next(w for w in text.split() if w.startswith("bm_"))
        one = await c.call_tool("batch_series_forecast", {"batch_id": batch_id, "series": "B"})
        assert "serie 'B'" in one.content[0].text and "LineChart" in json.dumps(
            one.structured_content, default=str
        )
        det = (
            await c.call_tool("get_forecast_details", {"forecast_id": f"{batch_id}::B", "aggregate": "month"})
        ).structured_content
        assert det["series"] == "B" and det["rows"]
        with pytest.raises(Exception, match="non è nel batch"):
            await c.call_tool("get_forecast_details", {"forecast_id": f"{batch_id}::ZZ"})
        exp = (await c.call_tool("export_batch", {"batch_id": batch_id})).structured_content
        assert exp["series"] == 3 and exp["forecast_rows"] == 3 * 14 and exp["files"][0].endswith(".xlsx")
        exp_csv = (
            await c.call_tool("export_batch", {"batch_id": batch_id, "format": "csv"})
        ).structured_content
        assert len(exp_csv["files"]) == 2
        items = (await c.call_tool("list_session_items", {})).structured_content
        assert items["batches"][0]["series"] == 3


async def test_calendar_then_forecast_with_holidays():
    async with Client(mcp) as c:
        ld = await load(c, long_csv(products=("A",), promo=False), series_column="sku")
        cal = (
            await c.call_tool(
                "add_calendar_features",
                {"dataset_id": ld["dataset_id"], "country": "IT", "extend_future_periods": 60},
            )
        ).structured_content
        assert "cal_holiday" in cal["added_columns"] and cal["dataset_id"] != ld["dataset_id"]
        assert cal["holidays_after_original_end"]
        r = await c.call_tool(
            "forecast",
            {
                "dataset_id": cal["dataset_id"],
                "target": "qty",
                "horizon": 30,
                "known_covariates": ["cal_holiday"],
            },
        )
        assert "Valori futuri usati dal modello — cal_holiday" in r.content[0].text
        with pytest.raises(Exception, match="non supportato"):
            await c.call_tool("add_calendar_features", {"dataset_id": ld["dataset_id"], "country": "ZZ"})


async def test_scenarios_thresholds_and_service_level():
    async with Client(mcp) as c:
        ld = await load(c, long_csv(products=("A",)), series_column="sku")
        ds = ld["dataset_id"]
        sc = await c.call_tool(
            "scenario_forecast",
            {
                "dataset_id": ds,
                "target": "qty",
                "horizon": 20,
                "known_covariates": ["promo"],
                "scenarios": {
                    "promo_lunga": [
                        {"column": "promo", "value": 1, "start": "2025-09-25", "end": "2025-09-30"}
                    ]
                },
            },
        )
        text = sc.content[0].text
        assert (
            "promo_lunga" in text
            and "causalità" in text
            and "LineChart" in json.dumps(sc.structured_content, default=str)
        )
        fc_ids = [w.strip(",.") for w in text.split() if "fc_" in w]
        scen_id = next(w.split("=")[1].strip(".,") for w in text.split() if w.startswith("promo_lunga=fc_"))
        d = (await c.call_tool("get_forecast_details", {"forecast_id": scen_id})).structured_content
        assert d["rows"] and fc_ids

        base = await c.call_tool("forecast", {"dataset_id": ds, "target": "qty", "horizon": 14})
        fid = next(w for w in base.content[0].text.split() if w.startswith("fc_"))
        tp = (
            await c.call_tool(
                "threshold_probability", {"forecast_id": fid, "threshold": 30, "direction": "below"}
            )
        ).structured_content
        assert tp["periods"] == 14 and "at_least_one_period" in tp
        sl = (
            await c.call_tool("quantity_for_service_level", {"forecast_id": fid, "service_level": 0.9})
        ).structured_content
        assert sl["window"]["cautious_total"] >= sl["window"]["median_total"]
        with pytest.raises(Exception, match="0.5"):
            await c.call_tool("quantity_for_service_level", {"forecast_id": fid, "service_level": 0.3})
        with pytest.raises(Exception, match="known_covariates"):
            await c.call_tool(
                "scenario_forecast",
                {
                    "dataset_id": ds,
                    "target": "qty",
                    "horizon": 5,
                    "known_covariates": [],
                    "scenarios": {"x": [{"column": "promo", "value": 1}]},
                },
            )


async def test_detect_anomalies_tool():
    async with Client(mcp) as c:
        ld = await load(c, long_csv(products=("A",), promo=False, spike=150), series_column="sku")
        r = await c.call_tool(
            "detect_anomalies", {"dataset_id": ld["dataset_id"], "target": "qty", "lookback": 90, "block": 14}
        )
        text = r.content[0].text
        assert "Anomalie an_" in text and "sopra" in text and "per puro caso" in text.lower()
        assert "DataTable" in json.dumps(r.structured_content, default=str)
        assert (await c.call_tool("list_session_items", {})).structured_content["anomaly_runs"][0][
            "anomalies"
        ] >= 1


async def test_new_tools_respect_download_guard(monkeypatch):
    async with Client(mcp) as c:
        ld = await load(c, long_csv(products=("A", "B")), series_column="sku")
        models.set_factory(None)
        monkeypatch.setattr(models, "is_cached", lambda size: False)
        for tool, args in (
            ("forecast_many", {"horizon": 7}),
            ("detect_anomalies", {"lookback": 30}),
            (
                "scenario_forecast",
                {
                    "horizon": 7,
                    "known_covariates": ["promo"],
                    "series": "A",
                    "scenarios": {"x": [{"column": "promo", "value": 1}]},
                },
            ),
        ):
            with pytest.raises(Exception, match="non è ancora scaricato"):
                await c.call_tool(tool, {"dataset_id": ld["dataset_id"], "target": "qty", **args})
