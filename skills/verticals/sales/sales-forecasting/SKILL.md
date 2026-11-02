---
name: sales-forecasting
description: Best practices specific to forecasting sales (demand, volumes, revenue) with Timestral: commercial calendar, promotions, stock-outs, intermittent products, hierarchies. Use when the data is sales or product demand.
---

# Sales forecasting

The rules of `forecasting-workflow` and `timestral-specifics` apply first. This skill covers the sales-specific points.

## Before forecasting: understand the data
- **What is being forecast**: units sold or revenue? With changing prices these are two different problems; users usually want units (demand) and convert afterwards. Ask if unclear.
- **Granularity**: daily data for weekly/monthly forecasts is fine; for data with a lot of daily noise, forecasting at weekly or monthly level (`aggregate_duplicates="sum"` if rows are transactions) is often more stable.
- **Several products/stores in the same file**: use `series_column`. With many series, forecast the main ones (best sellers) or ask the user which.
- **Transaction rows** (several rows per day): use `aggregate_duplicates="sum"` to get the daily total.

## Stock-outs and zeros
- Zero sales does **not** mean zero demand: it may be an out-of-stock product, a closed store, or missing data coded as 0.
- In `profile_dataset`, a high share of zeros should be investigated with the user. If the zeros are stock-outs, actual demand is underestimated: flag it and, if possible, exclude those periods (missing values in the file) instead of leaving them at 0.
- **Intermittent demand** (many legitimate zeros, sporadic sales): MASE and intervals must be read with caution; uncertainty will be wide. Consider a coarser aggregate (weekly/monthly).

## Promotions, prices and calendar
- Planned promotions, decided prices and holidays are **known-in-advance covariates** (`known_covariates`): the file needs future rows (empty target) with the column value for each future period.
- Before commenting on the effect of a promotion in the forecast, read when it is active in the future: `load_dataset`/`profile_dataset` (`future_covariates` field) and the forecast text (the line starting with "Valori futuri usati dal modello") report the active date ranges. Do not say a promotion is absent or present without reading it there. Peaks matching the active dates are, with good probability, an effect of the covariate.
- History of promotions without a future plan → `past_covariates` (helps understand the past but does not drive the future).
- A 0/1 indicator for promotion/holiday is fine; for several promotion types use separate columns or a categorical code.
- Typical events to consider in the data: Christmas and sales seasons, Black Friday, Easter (moving date), bridge days, closures, paydays. In this version holidays are not generated automatically: if needed, ask the user to add them to the file as a 0/1 column.

## New products or short history
- With less than a year of history, yearly seasonality is not visible; warn the user.
- The model is zero-shot and can still forecast, with larger uncertainty: use wide intervals (0.9) and flag the caution.

## From forecast to decision
- **Total per month/week**: `get_forecast_details(aggregate="month")`. If the month is incomplete, rerun the forecast with the horizon given in `partial_periods` instead of estimating the missing days.
- **Stock and service levels**: high quantiles are the reference. To cover about 90% of cases, look at the 0.90 quantile of the relevant period. Note that summing quantiles across periods overestimates the required quantity; say so and suggest treating the coverage period (lead time) as a single period.
- **Business report**: key message (growth/decline versus the previous period, peaks), interval, reliability (backtest), assumptions (promotions included or not).

## Sanity checks on the result
- Compare the forecast level with the same period last year, if available in the data.
- Negative sales values in the past (returns) should be flagged: they can distort the level.
