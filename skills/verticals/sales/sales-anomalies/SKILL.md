---
name: sales-anomalies
description: Interpreting anomalies in sales data (spikes, drops, zeros) and telling stock-outs, unrecorded promotions and loading errors apart. Use together with detect_anomalies when the data are sales or demand.
---

# Anomalies in sales

The logic of `anomaly-investigation` applies. These are the typical sales cases.

## Stock-outs and zeros
- Zero sales with a high expectation: first hypothesis is **product out of stock** or store closed, not absence of demand.
- Several zeros in a row followed by a spike: often backlog recovery, not extraordinary demand.
- If the user confirms a stock-out, real demand on those days was higher than what was recorded: the data in that period **underestimate** demand. Suggest blanking them in the file before forecasting, instead of leaving them at 0.

## Spikes
- Spike on a day with an active promotion (note "indicatori attivi"): expected, not a problem. If the promotion was not in `known_covariates`, rerun with the covariate.
- Spike with no promotion or holiday: ask about a special order, an event or a loading error (duplicate rows, different units).
- Spikes around holidays, Black Friday, Christmas: add `add_calendar_features` and rerun so they become expected.

## Prolonged drops
- Several consecutive days below the interval: level change (competition, price, range, closed channel). This matters for the forecast: the latest data are more relevant than older ones; consider a shorter `context_length` only if the user confirms a structural change.

## How to present them to the user
A short list (the 5–10 most severe), each with date, actual value, expected value and **one question** about what happened. Group them by type (zeros, spikes, periods) rather than listing everything.
