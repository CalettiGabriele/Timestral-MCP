---
name: covariate-selection
description: How to choose and judge covariates (promotions, prices, holidays, weather...) with covariate_importance and the backtest, and how to avoid leakage and causal claims. Use when the user asks which factors matter, whether a covariate is worth keeping, or what drives the forecast.
---

# Choosing and judging covariates

## Three kinds of covariate
- **Known in advance** (`known_covariates`): also known for the future (planned promotions, holidays, decided prices). The file needs future rows with empty target.
- **Past only** (`past_covariates`): known up to today (observed weather, competitor price). They give the model context but say nothing about the future.
- **Leakage**: never pass as "known in advance" something you will not actually know at forecast time (e.g. actual future weather, final sales of a related product). A leaked covariate makes the backtest look great and the real forecast poor.

## `covariate_importance`
- It runs the same backtest with all covariates, with none, and with each one removed in turn, on the **same windows**. The importance is the % worsening of the probabilistic error (WQL) when a covariate is removed; MASE is shown as a second opinion.
- With no covariate list, the server chooses: columns with future rows are known-in-advance, the others past-only. State this to the user (the result says it).
- Cost: one backtest per covariate (max 10, `n_folds` windows each). With clients that support MCP Tasks it runs in the background; otherwise it runs within a time limit and lists the covariates it did not evaluate: repeat with only those.

## Reading the result
| Verdict | Meaning | What to say |
|---|---|---|
| utile | removing it worsens the forecast in most windows where it matters | keep it; it helps to *predict* |
| inutile o dannosa | removing it improves the forecast | drop it: it adds noise |
| neutra o incerta | small change, windows disagree, or too few windows where it matters | no evidence either way; do not over-read |

- **"Neutra" does not mean "irrelevant to the business"**: a promotion that recurs on the same days every month can already be inferred from the target's history, so the covariate adds nothing to the forecast even if promotions do raise sales.
- **Rare covariates** (a promotion in only one test window) cannot be judged: the tool says so through `informative_folds`.
- **Correlated covariates substitute each other**: removing one barely matters because the other carries the information. If two seem unimportant, test removing both.
- Compare **WQL and MASE**. Opposite signs mean the covariate changes the width of the uncertainty and the median in different ways: report both and do not declare a winner.
- Few windows (≤3) or short history: differences of a few percent are noise.

## What not to claim
- It measures how much a covariate helps to **predict**, not that it **causes** sales. Do not turn it into "promotions bring +X%": use `scenario_forecast` for that, with its own caveats.
- It does not say which *level* of a covariate is best.

## Next steps after the result
1. Keep the useful ones, drop the harmful ones, and rerun `backtest` with the final set to confirm the gain.
2. If a known-in-advance covariate is useful, make sure the future rows are filled for the whole horizon.
3. For promotions and prices continue with `promotions-and-pricing` (scenarios); for calendar effects, `add_calendar_features` and check again.
