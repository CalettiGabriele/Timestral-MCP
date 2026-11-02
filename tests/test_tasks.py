"""Task MCP: i tool lunghi girano in background con i client moderni e in modo sincrono con quelli legacy."""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from fastmcp import Client
from fastmcp_tasks.client import call_tool_task

from tests.helpers import MarkerAwareFake
from timestral_mcp import models
from timestral_mcp import server as srv
from timestral_mcp.server import anomaly_runs, backtests, batches, datasets, forecasts, importance_runs, mcp


@pytest.fixture(autouse=True)
def _clean(fake):
    for store in (datasets, forecasts, backtests, batches, anomaly_runs, importance_runs):
        store.clear()


def multi_csv(n=200, products=("A", "B", "C")):
    idx = pd.date_range("2025-01-01", periods=n + 30, freq="D")
    rng = np.random.default_rng(4)
    frames = []
    for k, p in enumerate(products):
        y = 50 * (k + 1) + 8 * np.sin(np.arange(len(idx)) * 2 * np.pi / 7) + rng.normal(0, 1, len(idx))
        y[n:] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "data": idx.strftime("%Y-%m-%d"),
                    "sku": p,
                    "qty": y,
                    "promo": ((np.arange(len(idx)) % 10) < 2).astype(int),
                }
            )
        )
    return pd.concat(frames).to_csv(index=False)


async def load(c, **kw):
    return (
        await c.call_tool("load_dataset", {"source": multi_csv(), "series_column": "sku", **kw})
    ).structured_content


TOOLS = {
    "forecast_many": lambda ds: {"dataset_id": ds, "target": "qty", "horizon": 14},
    "backtest": lambda ds: {"dataset_id": ds, "target": "qty", "horizon": 14, "series": "A"},
    "detect_anomalies": lambda ds: {
        "dataset_id": ds,
        "target": "qty",
        "series": "A",
        "lookback": 60,
        "block": 14,
    },
    "covariate_importance": lambda ds: {
        "dataset_id": ds,
        "target": "qty",
        "horizon": 14,
        "series": "A",
        "n_folds": 3,
    },
}
MARKERS = {
    "forecast_many": "Previsione su più serie",
    "backtest": "MASE",
    "detect_anomalies": "Anomalie an_",
    "covariate_importance": "Importanza delle covariate",
}


@pytest.mark.parametrize("tool", list(TOOLS))
async def test_long_tools_run_as_background_tasks_with_modern_clients(tool):
    async with Client(mcp) as c:
        ds = (await load(c))["dataset_id"]
        task = await call_tool_task(c, tool, TOOLS[tool](ds))
        result = await task.result()
        assert MARKERS[tool] in result.content[0].text
        assert "$prefab" in result.structured_content  # anche come Task restituiscono l'interfaccia
        assert (await task.status()).status == "completed"


@pytest.mark.parametrize("tool", list(TOOLS))
async def test_legacy_clients_get_the_same_result_synchronously(tool):
    async with Client(mcp, mode="legacy") as c:
        ds = (await load(c))["dataset_id"]
        res = await c.call_tool(tool, TOOLS[tool](ds))
        assert MARKERS[tool] in res.content[0].text and "$prefab" in res.structured_content
        with pytest.raises(Exception, match="did not run as a task"):
            await call_tool_task(c, tool, TOOLS[tool](ds))


async def test_time_budget_default_depends_on_execution_mode(monkeypatch):
    seen = []
    original = srv.bm.run_batch

    def spy(*a, **kw):
        seen.append(kw["max_seconds"])
        return original(*a, **kw)

    monkeypatch.setattr(srv.bm, "run_batch", spy)
    args = lambda ds: {"dataset_id": ds, "target": "qty", "horizon": 7}  # noqa: E731
    async with Client(mcp, mode="legacy") as c:
        await c.call_tool("forecast_many", args((await load(c))["dataset_id"]))
    async with Client(mcp) as c:
        task = await call_tool_task(c, "forecast_many", args((await load(c))["dataset_id"]))
        await task.result()
        explicit = await call_tool_task(
            c, "forecast_many", {**args((await load(c))["dataset_id"]), "max_seconds": 30}
        )
        await explicit.result()
    assert seen == [90.0, 900.0, 30.0]


async def test_task_progress_is_observable_while_running():
    slow = MarkerAwareFake(delay=0.05)
    models.set_factory(lambda s: slow)
    async with Client(mcp) as c:
        ds = (await load(c))["dataset_id"]
        task = await call_tool_task(c, "forecast_many", {"dataset_id": ds, "target": "qty", "horizon": 7})
        messages = set()
        for _ in range(200):
            st = await task.status()
            if getattr(st, "status_message", None):
                messages.add(st.status_message)
            if st.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.02)
        result = await task.result()
        assert any(m.startswith("Serie ") for m in messages), messages
        assert "Serie previste: 3" in result.content[0].text
        assert slow.calls == 3


async def test_errors_inside_a_task_are_readable():
    async with Client(mcp) as c:
        ds = (await load(c))["dataset_id"]
        task = await call_tool_task(
            c, "covariate_importance", {"dataset_id": ds, "target": "nope", "horizon": 7, "series": "A"}
        )
        with pytest.raises(Exception, match="non trovato"):
            await task.result()


# ---------------------------------------------------------------------------- covariate_importance via tool


async def test_covariate_importance_tool_flow_and_session_items():
    async with Client(mcp, mode="legacy") as c:
        ds = (await load(c))["dataset_id"]
        r = await c.call_tool(
            "covariate_importance",
            {"dataset_id": ds, "target": "qty", "horizon": 14, "series": "A", "n_folds": 3},
        )
        text = r.content[0].text
        assert "promo | nota in anticipo" in text  # il file ha valori di promo anche per le righe future
        ui = json.dumps(r.structured_content, default=str)
        assert "BarChart" in ui and "DataTable" in ui
        items = (await c.call_tool("list_session_items", {})).structured_content
        assert items["importance_runs"][0]["covariates"] == 1
        with pytest.raises(Exception, match="almeno 1|greater than or equal to 2"):
            await c.call_tool(
                "covariate_importance", {"dataset_id": ds, "target": "qty", "horizon": 7, "n_folds": 1}
            )


async def test_covariate_importance_respects_download_consent(monkeypatch):
    async with Client(mcp, mode="legacy") as c:
        ds = (await load(c))["dataset_id"]
        models.set_factory(None)
        monkeypatch.setattr(models, "is_cached", lambda size: False)
        with pytest.raises(Exception, match="non è ancora scaricato"):
            await c.call_tool(
                "covariate_importance", {"dataset_id": ds, "target": "qty", "horizon": 7, "series": "A"}
            )
