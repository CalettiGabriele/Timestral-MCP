"""Forecast Studio e upload: struttura dell'interfaccia e tool di backend (via client MCP in memoria)."""

import base64
import io
import json
import re

import numpy as np
import pandas as pd
import pytest
from fastmcp import Client

from timestral_mcp import models
from timestral_mcp.server import (
    anomaly_runs,
    backtests,
    batches,
    datasets,
    forecasts,
    mcp,
    settings,
    uploads,
)


@pytest.fixture(autouse=True)
def _clean(fake, tmp_path):
    for store in (datasets, forecasts, backtests, batches, anomaly_runs, uploads):
        store.clear()
    object.__setattr__(settings, "export_dir", tmp_path / "exports")


def csv_text(n=200, future=0, promo=True, series=None):
    idx = pd.date_range("2025-03-01", periods=n + future, freq="D")
    rng = np.random.default_rng(2)
    y = 100 + 10 * np.sin(np.arange(len(idx)) * 2 * np.pi / 7) + rng.normal(0, 1, len(idx))
    y[n:] = np.nan
    df = pd.DataFrame({"data": idx.strftime("%Y-%m-%d"), "qty": y.round(2)})
    if promo:
        df["promo"] = (np.arange(len(idx)) % 15 == 0).astype(int)
        df["prezzo"] = 9.9
    if series:
        df["sku"] = series
    return df.to_csv(index=False)


async def load(c, text, **kw):
    return (await c.call_tool("load_dataset", {"source": text, **kw})).structured_content


def backend_tool(payload: dict, suffix: str) -> str:
    names = set(re.findall(r'"tool": "([^"]+)"', json.dumps(payload)))
    return next(n for n in names if n.endswith(suffix))


def walk_strings(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk_strings(v)
    elif isinstance(node, str):
        yield node


def referenced_state_keys(payload: dict) -> set[str]:
    keys: set[str] = set()
    for s in walk_strings(payload["view"]):
        for expr in re.findall(r"\{\{(.*?)\}\}", s):
            expr = re.sub(r"'[^']*'|\"[^\"]*\"", "", expr)
            for ident in re.findall(r"(?<![\w$.])([A-Za-z_]\w*)(?:\.\w+)*", expr):
                if ident not in {"true", "false", "null", "undefined"}:
                    keys.add(ident)
    return keys


async def test_studio_entry_is_for_the_model_and_backend_is_app_only():
    async with Client(mcp) as c:
        tools = {t.name: t for t in await c.list_tools()}
        assert {"forecast_studio", "upload_dataset"} <= set(tools)
        for entry in ("forecast_studio", "upload_dataset"):
            assert (
                tools[entry].meta["ui"]["visibility"] == ["model"]
                and "resourceUri" in tools[entry].meta["ui"]
            )
        # i tool di backend sono elencati ma riservati all'interfaccia: l'host li nasconde al modello
        for backend in ("studio_run", "studio_export", "store_upload"):
            assert tools[backend].meta["ui"]["visibility"] == ["app"]


async def test_studio_without_dataset_explains_what_to_do():
    async with Client(mcp) as c:
        with pytest.raises(Exception, match="Nessun dataset in sessione"):
            await c.call_tool("forecast_studio", {})


async def test_studio_form_reflects_dataset_and_state_is_consistent():
    async with Client(mcp) as c:
        ld = await load(c, csv_text(future=30))
        res = await c.call_tool("forecast_studio", {"dataset_id": ld["dataset_id"]})
        payload = res.structured_content
        state = payload["state"]
        assert state["target"] == "qty" and state["horizon"] == 30 and state["interval"] == "0.8"
        # covariate con valori futuri → preselezionate come "note in anticipo"; le altre no
        assert {state["cov_0"], state["cov_1"]} == {"known"}
        assert state["running"] is False and state["ready"] is False and state["chart"] == []
        text = json.dumps(payload, ensure_ascii=False)
        assert "209M · 837 MB · scaricato" in text and "Calcola previsione" in text
        assert "Invia i risultati a Claude" in text and "Salva in Excel" in text
        # nessuna formula punta a uno stato inesistente (typo in un nome = pulsante che non funziona)
        known_names = set(state) | {"RESULT", "ERROR"}
        assert referenced_state_keys(payload) <= known_names, referenced_state_keys(payload) - known_names


async def test_studio_defaults_follow_frequency_and_series():
    async with Client(mcp) as c:
        idx = pd.date_range("2022-01-01", periods=48, freq="MS")
        frames = [
            pd.DataFrame({"mese": idx.strftime("%Y-%m-%d"), "prodotto": p, "v": np.arange(48.0) + k})
            for k, p in enumerate("AB")
        ]
        ld = await load(c, pd.concat(frames).to_csv(index=False), series_column="prodotto")
        state = (await c.call_tool("forecast_studio", {"dataset_id": ld["dataset_id"]})).structured_content[
            "state"
        ]
        assert state["horizon"] == 12 and state["series"] == "A"
        assert "cov_0" not in state  # una sola colonna: nessuna covariata


async def test_studio_run_end_to_end_via_hidden_backend_tool():
    async with Client(mcp) as c:
        ld = await load(c, csv_text(future=30))
        payload = (await c.call_tool("forecast_studio", {"dataset_id": ld["dataset_id"]})).structured_content
        run_name = backend_tool(payload, "studio_run")
        out = (
            await c.call_tool(
                run_name,
                {
                    "dataset_id": ld["dataset_id"],
                    "target": "qty",
                    "horizon": "30",
                    "series": "serie",
                    "model": "5M",
                    "interval": "0.9",
                    "known": "promo,prezzo,",
                    "past": "",
                    "with_backtest": True,
                    "allow_download": False,
                },
            )
        ).structured_content
        assert out["forecast_id"].startswith("fc_") and len(out["table"]) == 30
        assert {"storico", "mediana", "basso", "alto"} <= set(out["chart"][-1])
        assert out["metrics"]["model"] == "Timestral-5M" and out["metrics"]["interval_label"].startswith(
            "Intervallo 90%"
        )
        assert (
            "promo" in out["covariate_text"]
            and "MASE" in out["backtest_text"]
            and out["backtest_metrics"]["folds"] >= 1
        )
        assert out["forecast_id"] in out["message"] and "get_forecast_details" in out["message"]
        # i risultati restano nella sessione: Claude può interrogarli con i tool normali
        det = (
            await c.call_tool("get_forecast_details", {"forecast_id": out["forecast_id"]})
        ).structured_content
        assert det["rows"] and det["target"] == "qty"
        exp = (
            await c.call_tool(
                backend_tool(payload, "studio_export"), {"forecast_id": out["forecast_id"], "format": "csv"}
            )
        ).structured_content
        assert exp["path"].endswith(".csv") and exp["rows"] == 30


async def test_studio_run_errors_are_readable():
    async with Client(mcp) as c:
        ld = await load(c, csv_text(future=30))
        run = backend_tool(
            (await c.call_tool("forecast_studio", {"dataset_id": ld["dataset_id"]})).structured_content,
            "studio_run",
        )
        base = {"dataset_id": ld["dataset_id"], "target": "qty", "horizon": 10, "model": "5M"}
        with pytest.raises(Exception, match="formula non risolta"):
            await c.call_tool(run, {**base, "known": "{{ (cov_0 == 'known' ? 'promo,' : '') }}"})
        with pytest.raises(Exception, match="devono avere valori"):
            await c.call_tool(run, {**base, "horizon": 60, "known": "promo,"})
        dropped = (await c.call_tool(run, {**base, "known": "qty,promo,"})).structured_content
        assert "tolto dalle covariate" in dropped["warnings_text"]
        with pytest.raises(Exception, match="non trovato"):
            await c.call_tool(run, {**base, "dataset_id": "ds_nope"})


async def test_studio_run_respects_download_consent(monkeypatch):
    async with Client(mcp) as c:
        ld = await load(c, csv_text(future=30))
        run = backend_tool(
            (await c.call_tool("forecast_studio", {"dataset_id": ld["dataset_id"]})).structured_content,
            "studio_run",
        )
        models.set_factory(None)
        monkeypatch.setattr(models, "is_cached", lambda size: False)
        monkeypatch.setattr(
            models,
            "_default_factory",
            lambda size: __import__("tests.conftest", fromlist=["x"]).FakeForecaster(),
        )
        args = {"dataset_id": ld["dataset_id"], "target": "qty", "horizon": 5, "model": "28M"}
        with pytest.raises(Exception, match="non è ancora scaricato"):
            await c.call_tool(run, args)
        ok = (await c.call_tool(run, {**args, "allow_download": True})).structured_content
        assert ok["forecast_id"]
        payload = (await c.call_tool("forecast_studio", {"dataset_id": ld["dataset_id"]})).structured_content
        assert "28M · 111 MB · da scaricare" in json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------------- upload


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


async def test_upload_creates_datasets_and_message_for_the_model():
    async with Client(mcp) as c:
        payload = (await c.call_tool("upload_dataset", {})).structured_content
        store = backend_tool(payload, "store_upload")
        xls = io.BytesIO()
        pd.read_csv(io.StringIO(csv_text(n=60, promo=False))).to_excel(xls, index=False)
        files = [
            {"name": "vendite.csv", "size": 1, "type": "text/csv", "data": b64(csv_text(n=60).encode())},
            {"name": "excel.xlsx", "size": 1, "type": "x", "data": b64(xls.getvalue())},
            {"name": "note.pdf", "size": 1, "type": "application/pdf", "data": b64(b"%PDF")},
        ]
        out = (await c.call_tool(store, {"files": files})).structured_content
        by = {i["name"]: i for i in out["items"]}
        assert by["vendite.csv"]["ok"] and by["excel.xlsx"]["ok"] and not by["note.pdf"]["ok"]
        assert "non supportato" in by["note.pdf"]["info"] and out["loaded"] == 2
        assert by["vendite.csv"]["dataset_id"] in out["message"] and "L'utente ha caricato" in out["message"]
        items = (await c.call_tool("list_session_items", {})).structured_content
        assert len(items["datasets"]) == 2
        # il dataset è subito utilizzabile dallo Studio
        studio = (
            await c.call_tool("forecast_studio", {"dataset_id": by["vendite.csv"]["dataset_id"]})
        ).structured_content
        assert studio["state"]["target"] == "qty"


async def test_unrecognised_upload_can_be_loaded_manually_with_options():
    async with Client(mcp) as c:
        idx = pd.date_range("2025-01-01", periods=40, freq="D")
        long = pd.concat(
            [
                pd.DataFrame({"data": idx.strftime("%Y-%m-%d"), "negozio": n, "qty": np.arange(40.0)})
                for n in ("A", "B")
            ]
        )
        store = backend_tool((await c.call_tool("upload_dataset", {})).structured_content, "store_upload")
        out = (
            await c.call_tool(
                store,
                {
                    "files": [
                        {
                            "name": "negozi.csv",
                            "size": 1,
                            "type": "text/csv",
                            "data": b64(long.to_csv(index=False).encode()),
                        }
                    ]
                },
            )
        ).structured_content
        item = out["items"][0]
        assert not item["ok"] and "upload:negozi.csv" in item["info"] and "date ripetute" in item["info"]
        ld = await load(c, "upload:negozi.csv", series_column="negozio")
        assert ld["n_series"] == 2 and ld["name"] == "negozi"
        with pytest.raises(Exception, match="non trovato"):
            await load(c, "upload:non-esiste.csv")


async def test_upload_limits_and_bad_content(monkeypatch):
    async with Client(mcp) as c:
        store = backend_tool((await c.call_tool("upload_dataset", {})).structured_content, "store_upload")
        object.__setattr__(settings, "max_upload_mb", 0)
        big = (
            await c.call_tool(
                store,
                {
                    "files": [
                        {"name": "a.csv", "size": 1, "type": "text/csv", "data": b64(b"x,y\n1,2\n3,4\n")}
                    ]
                },
            )
        ).structured_content
        assert "supera il limite" in big["items"][0]["info"] and big["loaded"] == 0
        object.__setattr__(settings, "max_upload_mb", 25)
        bad = (
            await c.call_tool(
                store, {"files": [{"name": "b.csv", "size": 1, "type": "text/csv", "data": "@@non-base64@@"}]}
            )
        ).structured_content
        assert "contenuto non valido" in bad["items"][0]["info"]
