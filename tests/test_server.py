import json

import pytest
from fastmcp import Client

from timestral_mcp.server import backtests, datasets, forecasts, mcp, settings


@pytest.fixture(autouse=True)
def _clean(fake, tmp_path):
    datasets.clear(), forecasts.clear(), backtests.clear()
    object.__setattr__(settings, "export_dir", tmp_path / "exports")


CSV = (
    "date,sales\n"
    + "\n".join(f"2024-01-{d:02d},{100 + (d % 7) * 10}" for d in range(1, 29))
    + "\n"
    + "\n".join(f"2024-02-{d:02d},{100 + (d % 7) * 10}" for d in range(1, 29))
)


async def test_tools_resources_prompts_and_skills_are_listed():
    async with Client(mcp) as c:
        tools = {t.name for t in await c.list_tools()}
        assert {
            "load_dataset",
            "profile_dataset",
            "forecast",
            "backtest",
            "get_forecast_details",
            "export_results",
            "forecast_chart_image",
        } <= tools
        uris = {str(r.uri) for r in await c.list_resources()}
        assert "timestral://models" in uris
        assert any(u.startswith("skill://") for u in uris), uris
        assert "previsione_guidata" in {p.name for p in await c.list_prompts()}


async def test_end_to_end_flow(tmp_path):
    async with Client(mcp) as c:
        loaded = (await c.call_tool("load_dataset", {"source": CSV, "name": "demo"})).structured_content
        ds_id = loaded["dataset_id"]
        assert loaded["frequency"] == "D" and loaded["numeric_columns"] == ["sales"]

        profile = (await c.call_tool("profile_dataset", {"dataset_id": ds_id})).structured_content
        assert profile["profiles"][0]["rows"] == 59  # 56 righe + 3 giorni di buco riempiti

        bres = await c.call_tool("backtest", {"dataset_id": ds_id, "target": "sales", "horizon": 7})
        assert "MASE" in bres.content[0].text

        res = await c.call_tool("forecast", {"dataset_id": ds_id, "target": "sales", "horizon": 30})
        text = res.content[0].text
        assert "Mediana prevista" in text
        ui = json.dumps(res.structured_content, default=str)
        assert "AreaChart" in ui and "DataTable" in ui

        fc_id = [w for w in text.split() if w.startswith("fc_")][0]
        det = (
            await c.call_tool("get_forecast_details", {"forecast_id": fc_id, "aggregate": "month"})
        ).structured_content
        assert det["rows"] and det["aggregate"] == "month"

        exp = (
            await c.call_tool(
                "export_results", {"forecast_id": fc_id, "format": "csv", "filename": "../../etc/x"}
            )
        ).structured_content
        assert exp["path"].startswith(str(tmp_path)) and exp["path"].endswith("x.csv")
        png = await c.call_tool("forecast_chart_image", {"forecast_id": fc_id})
        assert png.content[0].type == "image"

        items = (await c.call_tool("list_session_items", {})).structured_content
        assert items["datasets"] and items["forecasts"] and items["backtests"]


async def test_errors_are_readable():
    async with Client(mcp) as c:
        with pytest.raises(Exception, match="File non trovato"):
            await c.call_tool("load_dataset", {"source": "/tmp/non-esiste.csv"})
        with pytest.raises(Exception, match="non trovato"):
            await c.call_tool("forecast", {"dataset_id": "ds_nope", "target": "x", "horizon": 3})
