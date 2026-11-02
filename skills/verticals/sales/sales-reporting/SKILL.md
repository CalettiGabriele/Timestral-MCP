---
name: sales-reporting
description: How to turn sales forecasts into reports and decisions: totals by period, many series, stock by service level, thresholds and targets. Use when the user asks for a report, how much to order, stock levels, targets or a comparison across products/stores.
---

# From forecast to decision (sales)

## Many products or stores
- Use `forecast_many` and read, in this order: total forecast, change versus the latest periods, **series that change the most in absolute terms** (a large change on a small product matters little), uncertain series, data to check.
- The comparison with "the last N periods" **ignores seasonality**: if the forecast period is seasonally different (e.g. January after December), say that part of the drop/rise is expected.
- Focus the commentary on 5–10 relevant series; the rest is in the export (`export_batch`).
- Flag series with short history or many zeros: less reliable forecast, intermittent demand.

## Totals by month or week
- `get_forecast_details(aggregate="month")`. If the month is incomplete, rerun the forecast with the horizon given in `partial_periods`.
- The sum of medians is a good estimate of the total; the interval of the total is approximate (overestimated). Say so if it will be used for decisions.

## Stock and service level
- `quantity_for_service_level(forecast_id, service_level=0.9, start, end)` with `start`/`end` equal to the **period to cover** (lead time + review period).
- The result gives a **range** (`independent_total` – `cautious_total`): the right quantity usually lies between the two. Propose a value inside the range based on the cost of a stock-out: higher if missing a sale is expensive, closer to the independent estimate if holding stock is expensive.
- The **safety stock** is the difference from the median total. Do not choose service levels for the user: show 90%, 95% and the differences.
- A single coverage period (e.g. one weekly order) is more precise than a sum of days.

## Targets and thresholds
- "Will we reach X?": `threshold_probability` with `direction="above"` on a single period. For a total over several days the probability is **not** computable from daily quantiles: report the median total and the approximate interval, not a percentage.
- In the tails the tool only gives bounds ("meno del 1%"): report them as such.

## Structure of a report
1. Key message (one sentence with the change and its main reason).
2. Essential numbers: total, interval, critical periods/series.
3. Reliability: backtest, data quality, warnings.
4. Assumptions (promotions included or not, calendar, model used).
5. What to decide or check.
