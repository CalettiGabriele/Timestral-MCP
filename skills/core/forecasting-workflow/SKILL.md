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

## Workflow variants
- **Interactive path (Forecast Studio)**: if the user prefers to set things on a screen, or has many options to choose (series, covariates, model, interval), open `forecast_studio` with the `dataset_id` instead of asking each question in chat. **After opening it, stop and wait**: the user sets the options and runs the forecast in the interface. When they press "Invia i risultati a Claude" you receive a message with the `forecast_id` and the full summary: continue from there (comment, `get_forecast_details`, etc.). The results are already stored in the session, so everything works as if you had run `forecast` yourself. The Studio preselects as known-in-advance the covariates that have future rows; do not tell the user what they hold, read it in the summary you receive.
- **Upload from the interface**: if the user has no file path to give (or the file is on another computer), open `upload_dataset`. After the files are processed the dataset ids are added to your context (and shown by `list_session_items`). A file marked "da verificare" was read but not recognised automatically: ask the user which column identifies the series or how to treat duplicate dates, then call `load_dataset(source="upload:<file name>", series_column=..., ...)`.
- **Many series (products, stores)**: after `load_dataset` with `series_column`, use `forecast_many` (not one call per series). Report the total, the series that change the most (in absolute terms), the most uncertain ones and those with data to check; for a single series use `batch_series_forecast`. The work has a time and series limit: if some series are listed as "non eseguite", repeat with `series=[...]`. `with_backtest=true` adds per-series reliability (up to 30 series).
- **Holidays and calendar**: if the data are daily and the calendar matters (sales, consumption), suggest `add_calendar_features` with `extend_future_periods` at least equal to the horizon; then use the `cal_*` columns as `known_covariates`. It creates a **new** dataset (new id). Read `holidays_after_original_end` to tell the user which holidays fall in the period.
- **What-if**: `scenario_forecast` with known covariates (e.g. promotions). Explain the effects as the model's estimate from the historical relationship, not as causality.
- **Thresholds and stock**: `threshold_probability` ("what is the probability of dropping below X?") and `quantity_for_service_level` ("how much is needed to cover 90%?"). Tail probabilities are bounds ("meno del 1%"), not precise numbers.
- **Anomalies**: `detect_anomalies` (see the `anomaly-investigation` skill).
- **Which covariates matter**: `covariate_importance` (see the `covariate-selection` skill) before deciding which covariates to keep. Run it on one series at a time, and say that it measures usefulness for prediction, not causality.
- **Long operations**: `backtest`, `forecast_many`, `detect_anomalies` and `covariate_importance` can take a while. With clients that support MCP Tasks they run in the background and report progress; otherwise they run within a time limit and say what they did not run. When something is listed as not run, offer to repeat only that part instead of repeating everything.

## 6. Follow-up questions
- For specific periods, different quantiles or monthly totals: `get_forecast_details` (with `aggregate="month"` etc.). Do not guess numbers from the chart.
- **Incomplete totals**: check `partial_periods` and the notes. If the month (or week) is not fully covered, do not estimate the missing periods by eye: propose rerunning the forecast with `horizon` equal to `horizon_for_complete_period`.
- To save results: `export_results`.
- If the user cannot see the interactive chart: `forecast_chart_image`.
- If you lose an id: `list_session_items`.
