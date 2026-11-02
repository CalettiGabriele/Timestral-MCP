import numpy as np
import pandas as pd
import pytest

from tests.conftest import FakeForecast, FakeForecaster
from timestral_mcp import models
from timestral_mcp.analytics.scenarios import run_scenarios, scenario_summary
from timestral_mcp.data.ingest import build_dataset


class PromoAwareFake(FakeForecaster):
    """+30% sul livello nei periodi in cui la prima covariata nota vale 1."""

    def forecast(self, context, prediction_length, **kw):
        fc = super().forecast(context, prediction_length, **kw)
        known = kw.get("known_covariates")
        if known is not None:
            flag = np.asarray(known)[0, 0, -prediction_length:]
            fc.quantiles = fc.quantiles * (1 + 0.3 * flag)[None, None, :, None]
        return FakeForecast(fc.quantiles)


@pytest.fixture
def promo_ds():
    idx = pd.date_range("2025-06-01", periods=200 + 30, freq="D")
    y = 100 + 10 * np.sin(np.arange(len(idx)) * 2 * np.pi / 7)
    y[200:] = np.nan
    promo = np.zeros(len(idx))
    promo[[10, 50, 90, 130]] = 1  # solo 4 attivazioni nello storico
    df = pd.DataFrame({"d": idx.strftime("%Y-%m-%d"), "vendite": y, "promo": promo})
    return build_dataset(df, dataset_id="ds_p", name="p", source="t")


@pytest.fixture
def aware():
    f = PromoAwareFake()
    models.set_factory(lambda s: f)
    yield f
    models.set_factory(None)


def ids():
    n = iter(range(100))
    return lambda: f"fc_{next(n)}"


def test_promo_scenario_increases_total_and_reports_window(aware, promo_ds):
    last = promo_ds.series["serie"]["vendite"].last_valid_index()
    s, e = (
        (last + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        (last + pd.Timedelta(days=11)).strftime("%Y-%m-%d"),
    )
    sr = run_scenarios(
        promo_ds,
        target="vendite",
        horizon=30,
        known_covariates=["promo"],
        id_factory=ids(),
        scenarios={"con_promo": [{"column": "promo", "value": 1, "start": s, "end": e}]},
    )
    base, scen = sr.comparison
    assert base["scenario"] == "base" and scen["delta_vs_base"] > 0
    assert 20 < scen["window_delta_pct"] < 40  # ~ +30% nella settimana di promo
    assert abs(scen["delta_vs_base"] - scen["window_delta"]) < 1e-6 and scen["after_window_delta"] == 0
    assert any("solo 4 volte" in w for w in sr.warnings)  # poche attivazioni nello storico
    text = scenario_summary(sr)
    assert "con_promo" in text and "non sono una prova di causalità" in text
    assert promo_ds.series["serie"]["promo"].iloc[-30:].sum() == 0  # dataset originale intatto


def test_scenario_validation(aware, promo_ds):
    common = dict(target="vendite", horizon=30, id_factory=ids(), known_covariates=["promo"])
    with pytest.raises(ValueError, match="known_covariates"):
        run_scenarios(
            promo_ds, **{**common, "known_covariates": []}, scenarios={"a": [{"column": "promo", "value": 1}]}
        )
    with pytest.raises(ValueError, match="non è tra le known_covariates"):
        run_scenarios(promo_ds, **common, scenarios={"a": [{"column": "vendite", "value": 1}]})
    with pytest.raises(ValueError, match="fuori dall'orizzonte"):
        run_scenarios(
            promo_ds, **common, scenarios={"a": [{"column": "promo", "value": 1, "start": "2030-01-01"}]}
        )
    with pytest.raises(ValueError, match="riservato"):
        run_scenarios(promo_ds, **common, scenarios={"base": [{"column": "promo", "value": 1}]})
    with pytest.raises(ValueError, match="almeno uno scenario"):
        run_scenarios(promo_ds, **common, scenarios={})
    with pytest.raises(ValueError, match="numerico"):
        run_scenarios(promo_ds, **common, scenarios={"a": [{"column": "promo", "value": "molto"}]})
