---
name: forecast-interpretation
description: How to explain probabilistic forecast results (median, intervals, backtest) to a non-expert and how to answer questions about charts using the numbers. Use when commenting on a forecast or when the user asks about a chart.
---

# Interpreting and explaining a forecast

## Golden rule: use the numbers, not the image
The chart is for the user. Answer from the numbers in the result text or from `get_forecast_details`:
- "What do you forecast for March?" → `get_forecast_details(forecast_id, aggregate="month")` or `start`/`end`.
- "How uncertain is the forecast?" → compare the interval width at the start and at the end of the horizon.
- "How likely is it to fall below X?" → request `quantiles` close to X (e.g. 0.05, 0.1, 0.25) and give a cautious estimate between the quantiles ("between 10% and 25% probability"). Do not invent precise probabilities outside the available quantiles.

## Explaining uncertainty
- The **median** is the central scenario: half of the outcomes above, half below. It is not a certainty.
- An **80% interval** means that 8 times out of 10 the actual value is expected to fall inside it. It does not mean "the value will be between A and B".
- Uncertainty **grows with the horizon**: this is normal. Flag it if it is very wide relative to the level of the series.
- Do not present the median as a target or a guarantee.

## Using the backtest
- **MASE < 1**: better than a naive forecast. Lower is better.
- **Versus the seasonal baseline**: percentage improvement. If negative, the model adds no value on this data.
- **Coverage**: if the 80% interval contained much less than 80% of the actual values, the intervals are too narrow for this data: advise caution and use a wider interval (e.g. 0.9 or 0.98).
- A good backtest on the past does not guarantee the future (regime changes, exceptional events).

## Totals and aggregations
- The sum of the medians is a good estimate of the total; **the interval of the total is approximate** (summed quantiles overestimate it). Say so if the user will rely on that interval for decisions.
- Flag partial periods (incomplete first/last month).

## Structure of a good answer
1. Key message in one sentence (e.g. "a 12% increase is expected by the end of the horizon, peaking in December").
2. Essential numbers (central value, interval, peak dates).
3. How far to trust it (backtest, warnings, data quality).
4. What could change the picture (covariates, unknown events) and how to proceed.
