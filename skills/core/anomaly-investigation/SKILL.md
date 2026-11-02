---
name: anomaly-investigation
description: How to look for and interpret anomalies in a time series with detect_anomalies, and how to tell data errors from real events. Use when the user asks about anomalies, odd spikes or drops, outliers, or "what happened on day X".
---

# Finding and interpreting anomalies

## How `detect_anomalies` works
Every recent value is compared with a forecast made using **only the data before it** (consecutive windows). A value is anomalous if it falls outside the expected interval (default 98%). Severity is the distance from the median in equivalent standard deviations.

## Before running it
- If covariates explain the peaks (promotions, holidays), pass them in `known_covariates`: a promotion peak then does not show up as an anomaly. Without them, promotions appear as "sopra" (above) anomalies.
- `lookback` (default 365) and `block` (default: seasonal cycle, at least 14) are almost always fine. Short histories: at least `2 × seasonal cycle + block` points are needed.

## Reading the result
- **Compare with the number expected by chance**: with a 98% interval about 2% of points fall outside even when nothing is wrong. Few anomalies, no alarm; many more than expected point to a real issue (regime change, dirty data).
- Prioritise by **severity**, not by date.
- The notes help: "valore zero con previsione positiva" (possible stock-out, closure or missing data coded as 0) and "indicatori attivi quel giorno" (flags active on that day).

## Telling data errors from real events
| Signal | Likely cause | What to ask the user |
|---|---|---|
| isolated zero with high expectation | stock-out, closed store, missing data coded as 0 | was the product available? was the store open? |
| isolated very high spike | loading error, exceptional order, unrecorded promotion | was there a special order or an activity that is not in the file? |
| run of consecutive days above/below | level change (new channel, price, competition) | what changed in that period? |
| repeats on the same days of the year/week | seasonality or recurring event not modelled | add a covariate (calendar, promo) |

## What to do next
- Do not delete or "fix" data on your own. Timestral handles missing values, but data should not be cleaned by guesswork. If the user confirms a point is an error, they can blank it in the file and reload it.
- A good follow-up is to rerun `backtest` or `forecast` after the correction and say how much the result changes.
