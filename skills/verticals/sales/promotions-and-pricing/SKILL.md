---
name: promotions-and-pricing
description: How to use promotions, prices and the commercial calendar in sales forecasts, and how to run what-if scenarios with scenario_forecast and add_calendar_features. Use when the user asks "what if we run a promotion", about the effect of discounts or prices, Black Friday, holidays or sales seasons.
---

# Promotions, prices and calendar

## Preparing the data
- Planned promotions and decided prices are **covariates known in advance**: the file needs future rows (empty target) with the planned value for every future period. If they are missing, ask for the promotion plan, or use `scenario_forecast` on an existing column.
- A 0/1 flag per promotion is fine; for several types (10% off, 2-for-1, flyer) use separate columns or the discount value.
- Holidays and the commercial calendar: `add_calendar_features` (country `IT` by default) creates `cal_holiday`, `cal_pre_holiday`, `cal_bridge`, `cal_black_friday`, `cal_christmas`, `cal_easter`. Use `extend_future_periods` ≥ the horizon. Columns that are constant over the data period are dropped and listed in `dropped_constant_columns`: normal when the history does not include, say, a Christmas.
- Seasonal sales and paydays are not generated: if they matter, they go in the file as 0/1 columns.

## Running a scenario
1. Start from a forecast that already uses the covariate (`known_covariates=["promo"]`).
2. Call `scenario_forecast` with one or more scenarios (max 5), for example:
   `{"promo_10_16_feb": [{"column":"promo","value":1,"start":"2026-02-10","end":"2026-02-16"}]}`.
   For a "no promotion" scenario set the value to 0 over the whole horizon.
3. Read the table: **effect in the window** (the modified week), **after the window** (purchases brought forward or postponed: a dip after a promotion is typical) and the **total** over the horizon.

## How to explain the effect
- It is the effect the model **estimates from the historical relationship** with the covariate: not proof of causality, nor a return on investment. Do not compare it with costs or margins unless the user provides them.
- If the tool warns that the promotion was active only a few times in the history (fewer than 5) or that the value is outside the observed range, **say so before quoting the number**: the effect is unreliable.
- Always report the difference in units and in percent, and the window it refers to.
- Scenarios show the median: uncertainty stays wide. For questions like "will we exceed 1,000 units?" use `threshold_probability` on the scenario's forecast (the ids are in the text).
