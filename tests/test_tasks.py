"""Task MCP: i tool lunghi girano in background con i client moderni e in modo sincrono con quelli legacy."""

import numpy as np
import pandas as pd
import pytest
from fastmcp import Client
from fastmcp_tasks.client import call_tool_task

from timestral_mcp.server import backtests, datasets, forecasts, mcp


@pytest.fixture(autouse=True)
def _clean(fake):
    for store in (datasets, forecasts, backtests):
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
    "backtest": lambda ds: {"dataset_id": ds, "target": "qty", "horizon": 14, "series": "A"},
}
MARKERS = {
    "backtest": "MASE",
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


# ---------------------------------------------------------------------------- covariate_importance via tool
