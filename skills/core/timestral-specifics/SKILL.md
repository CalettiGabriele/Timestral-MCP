---
name: timestral-specifics
description: Characteristics and limits of Mistral's Timestral model and what NOT to do to the data before using it. Consult when preparing data, choosing the model size, or interpreting limits and warnings.
---

# What to know about Timestral

## It is a zero-shot model
It is not trained on the user's data: it reads the history and forecasts. There are no parameters to tune.

## What NOT to do before using it
- **Do not difference** the series or make it stationary (ARIMA recipes): the model works on the original levels.
- **Do not impute** missing values: the model handles NaNs. Imputing with means or interpolation can hide information.
- **Do not normalize or scale**: it does this internally.
- **Do not remove seasonality** by hand.
If you need to understand the data, use `profile_dataset` instead of transforming it.

## What it returns
- **21 fixed quantiles** for each period: 0.01, 0.05, 0.10, 0.15 … 0.90, 0.95, 0.99. The **median** (0.5) is the point forecast.
- Common central intervals: 50% (q25–q75), 80% (q10–q90), 90% (q05–q95), 98% (q01–q99).
- Quantiles are per single period: they **cannot be summed** to get the interval of a total (monthly, weekly). `get_forecast_details` with `aggregate` sums quantiles as an approximation and says so.

## Limits
- **Maximum context 32,768 points**: beyond that, only the most recent part is used.
- **Horizon 1,024 periods per block**: beyond this threshold the model proceeds in blocks using the median as the new context, so uncertainty is underestimated. Warn the user.
- **Short history**: with fewer than two seasonal cycles (e.g. fewer than 14 days for daily data with a weekly cycle, fewer than 24 months for monthly data) seasonality is not captured well.
- **Covariates** condition the forecast but are not forecast themselves. A "known in advance" covariate must really be known in the future (leakage risk if you use as such a value that is not actually known at forecast time).

## Sizes
| Model | When |
|---|---|
| 209M | maximum accuracy (default) |
| 123M | almost the same as 209M, a bit faster |
| 28M | very fast, excellent trade-off |
| 5M | extremely fast, lower accuracy |
The difference between 123M and 209M is minimal; for many series (hundreds) a smaller size is preferable.

## First use
The first use of a size requires downloading the weights (21 MB for 5M, 111 MB for 28M, 490 MB for 123M, 837 MB for 209M; several minutes for 209M). The server **does not download silently**: `forecast` and `backtest` return an error listing the models already available. In that case:
1. tell the user (size and estimated time);
2. if they agree, retry with `allow_download=true`; otherwise use a model that is already downloaded;
3. alternatively they can download it from a terminal with `uv run timestral-mcp download 209M`.
The `timestral://models` resource lists which models are already on disk.
