import numpy as np
import pandas as pd
import pytest

from tests.helpers import MarkerAwareFake
from timestral_mcp import models
from timestral_mcp.analytics import importance as im
from timestral_mcp.data.ingest import build_dataset


@pytest.fixture
def marker_fake():
    f = MarkerAwareFake()
    models.set_factory(lambda s: f)
    yield f
    models.set_factory(None)


def make_ds(n=320, future=30, with_bad=False, seed=11):
    """promo (valore 7, utile): +30% sul target nei giorni attivi; rumore 0/1 casuale; opzionale 'cattiva' (valore 9)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n + future, freq="D")
    t = np.arange(len(idx))
    promo = ((t % 17) < 3).astype(float) * 7
    noise = rng.integers(0, 2, len(idx)).astype(float)
    y = (100 + 8 * np.sin(t * 2 * np.pi / 7)) * (1 + 0.3 * (promo == 7)) + rng.normal(0, 0.5, len(idx))
    y[n:] = np.nan
    cols = {"data": idx.strftime("%Y-%m-%d"), "vendite": y, "promo": promo, "rumore": noise}
    if with_bad:
        cols["fuorviante"] = ((t % 23) < 2).astype(float) * 9
    return build_dataset(pd.DataFrame(cols), dataset_id="ds_im", name="im", source="t")


def run(ds, **kw):
    plan = im.plan_importance(ds, target="vendite", horizon=14, **kw)
    return plan, im.run_importance(plan, "im_1")


def by_cov(res):
    return {r["covariate"]: r for r in res.rows}


def test_useful_noise_and_misleading_covariates_are_ranked(marker_fake):
    ds = make_ds(with_bad=True)
    _, res = run(ds, known_covariates=["promo", "rumore", "fuorviante"], n_folds=4)
    rows = by_cov(res)
    assert rows["promo"]["verdict"] == im.USEFUL and rows["promo"]["delta_wql_pct"] > 5
    assert rows["rumore"]["verdict"] == im.UNCERTAIN and rows["rumore"]["delta_wql_pct"] == 0
    assert rows["fuorviante"]["verdict"] == im.USELESS and rows["fuorviante"]["delta_wql_pct"] < -2
    assert [r["covariate"] for r in res.rows][0] == "promo" and res.rows[-1][
        "covariate"
    ] == "fuorviante"  # ordinate
    assert res.rows[0]["folds_worse_without"].endswith("/4") and res.rows[0]["informative_folds"] >= 2
    assert rows["rumore"]["informative_folds"] == 0  # ignorata dal modello: mai differenza
    assert res.model == "209M" and res.folds == 4


def test_gain_versus_no_covariates_and_summary_text(marker_fake):
    _, res = run(make_ds(), known_covariates=["promo", "rumore"])
    assert res.gain_vs_none_pct["wql"] > 5  # togliendo tutte le covariate l'errore sale
    text = im.importance_summary(res)
    assert "togliendole tutte l'errore (WQL) sale" in text and "non prova che CAUSI" in text
    assert "promo | nota in anticipo" in text and "covariate molto correlate" in text


def test_roles_are_chosen_automatically_from_future_rows(marker_fake):
    ds = make_ds()
    # 'rumore' e 'promo' hanno righe future (30 dopo l'ultimo target) → note in anticipo
    plan = im.plan_importance(ds, target="vendite", horizon=14)
    assert plan.known == ["promo", "rumore"] and plan.past == []
    assert any("scelte in automatico" in w for w in plan.warnings)
    # senza righe future diventano 'solo passate'
    ds2 = make_ds(future=0)
    plan2 = im.plan_importance(ds2, target="vendite", horizon=14)
    assert plan2.past == ["promo", "rumore"] and plan2.known == []
    assert len(plan2.configs) == 2 + 2  # tutte, nessuna, una tolta alla volta


def test_past_covariates_are_dropped_from_the_right_role(marker_fake):
    ds = make_ds(future=0)
    plan = im.plan_importance(
        ds, target="vendite", horizon=14, past_covariates=["promo", "rumore"], known_covariates=[]
    )
    cfg = dict((n, (k, p)) for n, k, p in plan.configs)
    assert cfg["without:promo"] == ([], ["rumore"]) and cfg["all"] == ([], ["promo", "rumore"])


def test_validation_errors(marker_fake):
    ds = make_ds()
    with pytest.raises(ValueError, match="non valida"):
        im.plan_importance(ds, target="vendite", horizon=14, known_covariates=["vendite"])
    with pytest.raises(ValueError, match="sia passate che note"):
        im.plan_importance(
            ds, target="vendite", horizon=14, known_covariates=["promo"], past_covariates=["promo"]
        )
    with pytest.raises(ValueError, match="non trovato"):
        im.plan_importance(ds, target="nope", horizon=14)
    solo = build_dataset(
        pd.DataFrame(
            {"d": pd.date_range("2025-01-01", periods=60).strftime("%Y-%m-%d"), "v": np.arange(60.0)}
        ),
        dataset_id="s",
        name="s",
        source="t",
    )
    with pytest.raises(ValueError, match="Nessuna covariata"):
        im.plan_importance(solo, target="v", horizon=7)
    many = {"data": pd.date_range("2025-01-01", periods=80).strftime("%Y-%m-%d"), "v": np.arange(80.0)}
    many.update({f"c{i}": np.arange(80.0) % (i + 2) for i in range(im.MAX_COVARIATES + 1)})
    with pytest.raises(ValueError, match="Troppe covariate"):
        im.plan_importance(
            build_dataset(pd.DataFrame(many), dataset_id="m", name="m", source="t"), target="v", horizon=7
        )


def test_time_budget_returns_baselines_and_lists_unrun_covariates(marker_fake, monkeypatch):
    import time

    ticks = iter(range(0, 10000, 10))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    ds = make_ds(with_bad=True)
    plan = im.plan_importance(
        ds, target="vendite", horizon=14, known_covariates=["promo", "rumore", "fuorviante"]
    )
    res = im.run_importance(plan, "im_1", max_seconds=15)
    assert (
        res.all_covariates["wql"] is not None and res.no_covariates["wql"] is not None
    )  # le due basi si fanno sempre
    assert res.not_run and any("Tempo massimo" in w for w in res.warnings)
    assert len(res.rows) < 3


def test_progress_callback_reports_each_configuration(marker_fake):
    seen = []
    plan = im.plan_importance(make_ds(), target="vendite", horizon=14, known_covariates=["promo", "rumore"])
    im.run_importance(plan, "im_1", progress=lambda i, n, m: seen.append((i, n, m)))
    assert (
        [s[0] for s in seen] == [0, 1, 2, 3, 4]
        and all(s[1] == 4 for s in seen[:-1])
        and seen[-1][2] == "Completato"
    )
    assert "senza promo" in " ".join(s[2] for s in seen)


def test_rarely_active_covariate_is_not_judged_on_windows_where_it_is_off(marker_fake):
    """Una covariata utile ma attiva in una sola finestra non basta per un giudizio netto."""
    ds = make_ds(n=320, future=30)
    df = ds.series["serie"]
    df["promo"] = 0.0
    df.loc[df.index[300:303], "promo"] = 7.0  # attiva solo nell'ultima finestra di test
    df.loc[df.index[300:303], "vendite"] = df["vendite"].iloc[290] * 1.3
    plan = im.plan_importance(ds, target="vendite", horizon=14, known_covariates=["promo"], n_folds=4)
    res = im.run_importance(plan, "im_1")
    row = res.rows[0]
    assert row["informative_folds"] < im.MIN_INFORMATIVE_FOLDS and row["verdict"] == im.UNCERTAIN
