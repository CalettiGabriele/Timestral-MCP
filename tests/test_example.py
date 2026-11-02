from pathlib import Path

import numpy as np

from timestral_mcp.analytics.forecast import run_forecast
from timestral_mcp.data.ingest import build_dataset, read_source

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "vendite_esempio.csv"


def test_example_file_supports_known_covariates(fake):
    df, origin = read_source(str(EXAMPLE))
    ds = build_dataset(df, dataset_id="ds_ex", name="esempio", source=origin)
    assert ds.freq == "D" and ds.value_columns == ["vendite", "promo"]
    r = run_forecast(ds, forecast_id="fc_ex", target="vendite", horizon=30, known_covariates=["promo"])
    call = fake.calls[-1]
    assert call["known_covariates"].shape == (1, 1, 730 + 30)
    assert np.nansum(call["known_covariates"][0, 0, -30:]) == 5  # 5 giorni di promo pianificati
    assert r.future_index[0].strftime("%Y-%m-%d") == "2025-12-31"
