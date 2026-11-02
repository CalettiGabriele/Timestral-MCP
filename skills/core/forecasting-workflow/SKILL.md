---
name: forecasting-workflow
description: Methodology for going from a data file to a reliable forecast with Timestral. Use whenever the user wants to forecast a time series (sales, demand, consumption, metrics) or asks to "forecast", "predict", "how will it go" with their own data.
---

# Forecasting workflow with Timestral

Follow these steps in order. Do not skip the backtest: it is what lets you tell the user how far to trust the forecast.

## 1. Load the data
- `load_dataset(source=...)` with the file path (CSV/Excel) or the data pasted by the user.
- Read what was detected: date column, frequency, series, numeric columns. **Report it to the user** and ask for confirmation if something looks off (ambiguous day/month dates, wrong column).
- A "long" file with several products/stores in the same table needs `series_column`. If the tool reports repeated dates, ask the user which column identifies the series, or whether the rows should be summed (`aggregate_duplicates="sum"`).
- Do not resample or clean the data yourself before loading: the server detects the frequency and aligns the time grid on its own (gaps become missing values, which the model handles).

## 2. Check quality
- `profile_dataset(dataset_id)`. Report briefly: covered period, number of points, missing values, share of zeros, outliers, warnings.
- If the history is shorter than two seasonal cycles, say so: the forecast will be less reliable.

## 3. Collect the user's choices
Ask (together, in a single question) only what is not already clear:
- column to forecast (`target`) and series, if there is more than one;
- horizon, in periods of the dataset frequency (e.g. 90 for 90 days, 12 for 12 months);
- covariates: columns that may help. *Past* (`past_covariates`) if known only up to today; *known in advance* (`known_covariates`) if they are also known for the future (planned promotions, holidays, decided prices). In that case the file must contain future rows with an empty target.
- model: default 209M (most accurate). 28M for very fast answers. If the chosen model is not downloaded yet, the tool says so: **warn the user** (size and time) and retry with `allow_download=true` only after their consent. Do not switch model on your own between forecasts without saying so.

## 4. Backtest
- `backtest(dataset_id, target, horizon)` with the same horizon as the forecast.
- Explain the result in plain words: MASE below 1 means better than a naive forecast; comparison with the seasonal baseline; interval coverage (if the 80% interval contains only 50% of the actual values, uncertainty is underestimated).
- If the model does worse than the baseline, say so clearly and propose alternatives (shorter horizon, covariates, checking the data).

## 5. Forecast
- `forecast(...)` shows an interactive chart to the user. You receive the numbers in the result text.
- Comment on: expected trend, peaks and troughs with dates, width of the uncertainty and how it grows, any warnings.
- **Known-in-advance covariates**: the text contains a line starting with "Valori futuri usati dal modello" (the future values used by the model, e.g. active promotion days). Base every comment on their effect on those values, **do not assume them**, and do not infer them from other fields. If the line is missing, you do not know what the covariate holds in the future: say so.

## 6. Follow-up questions
- For specific periods, different quantiles or monthly totals: `get_forecast_details` (with `aggregate="month"` etc.). Do not guess numbers from the chart.
- **Incomplete totals**: check `partial_periods` and the notes. If the month (or week) is not fully covered, do not estimate the missing periods by eye: propose rerunning the forecast with `horizon` equal to `horizon_for_complete_period`.
- To save results: `export_results`.
- If the user cannot see the interactive chart: `forecast_chart_image`.
- If you lose an id: `list_session_items`.
