# Auto Trader: a rigorous backtesting and execution framework (with honest negative results)

[ภาษาไทย](README.th.md) | English

A Python research project that tries to find a tradable edge in CFD markets (gold, FX, equity indices) and builds the risk and execution plumbing around it. So far the main finding is negative: **none of the 11 hypotheses tested showed an edge that survived the pre-registered criteria.** This repo documents how that conclusion was reached, because the method is the part worth reusing.

> **Disclaimer:** This is a personal learning and research project, not financial advice. The execution module is for **demo accounts only**. Past results, including the negative ones here, do not predict future performance.

## Status

| Area | Status |
|---|---|
| Backtesting (walk-forward, spread-aware costs) | Done |
| Risk engine, position sizing, paper broker, replay | Done, unit-tested |
| MT5 broker adapter (`mt5_broker.py`) | Written, 10 unit tests pass against a fake MT5. Connection, demo check, contract-size check and order submission verified on a real demo account. **Order fills, entry/exit prices, position close and PnL are not yet verified against a live terminal** (pending a smoke test while the market is open) |
| Live trading loop (strategy -> risk -> broker) | Not written yet |
| A strategy that passes the criteria | None so far |

## Research method

The point of the project is to avoid fooling myself:

- **Pre-registration.** Parameters, universe and pass/fail criteria are written down and locked before any result is seen. Each hypothesis is run once. Criteria are never edited after seeing results.
- **Walk-forward evaluation** instead of a single in-sample backtest.
- **Realistic costs.** MT5 history contains bars with `spread = 0`; these are replaced with the median non-zero spread of the same year (`spread.py`), and a stress test doubles the spread. Fixing this barely changed results, so costs were not the cause of the negative findings.
- **Sanity checks.** Tests confirm there is no lookahead (changing future data does not change past signals), and running the strategy on 40 random-walk datasets gives a mean Sharpe of about -0.11 (no phantom profit).
- **Counting trials.** Every hypothesis tried is counted (11 so far), since the more you try, the more likely something looks good by luck. Future hypotheses must clear a stricter bar.
- **No tuning after the fact.** If a result fails, it is accepted rather than "improved".

## Results

### 1. MA + ADX trend following (3 assets x 3 timeframes, 2010-2023)

Pass criteria (locked beforehand): OOS Sharpe > 0.5, positive in at least 2/3 of folds, at least 6 folds. **0 of 9 passed.**

| Symbol | TF | Folds | OOS Sharpe | OOS return % | OOS max DD % | Positive folds |
|---|---|---|---|---|---|---|
| XAUUSD | H4 | 12 | 0.34 | 27.3 | 21.3 | 6 |
| XAUUSD | D1 | 11 | 0.22 | 13.8 | 20.3 | 5 |
| US500 | H4 | 9 | 0.13 | 5.3 | 15.5 | 4 |
| US500 | D1 | 9 | 0.03 | -2.4 | 23.5 | 2 |
| XAUUSD | H1 | 12 | -0.00 | -4.7 | 33.4 | 7 |
| EURUSD | H4 | 12 | -0.24 | -11.2 | 17.4 | 5 |
| EURUSD | D1 | 10 | -0.36 | -5.9 | 6.4 | 0 |
| US500 | H1 | 9 | -0.43 | -33.9 | 40.3 | 3 |
| EURUSD | H1 | 12 | -0.63 | -28.5 | 31.6 | 2 |

The best row (XAUUSD H4, 0.34) is the best of 9 tries, still below the threshold, and positive in only 6 of 12 folds, so it is not a "near miss".

### 2. Time-series momentum (TSMOM), D1, multi-asset

Signal: sign of the 252-day return. Weight: signal x (10% target vol / 60-day vol) / N, monthly rebalance, entry at next open, half-spread costs. Pass criteria: net Sharpe > 0.4, profitable years >= 60%, max DD < 25%, stress (spread x2) Sharpe > 0.3. **Failed both runs.**

| Run | Universe | Period | Net Sharpe | Profitable years | Max DD | Stress Sharpe |
|---|---|---|---|---|---|---|
| 1 | XAUUSD, EURUSD, US500 | 2013-2023 | 0.25 | 64% (pass) | 17.1% (pass) | 0.25 |
| 2 | 8 usable of 12 pre-listed symbols | 2011-2023 | 0.24 | 54% | 13.2% (pass) | 0.23 |

Notes: in run 2, four symbols were dropped by the pre-set rule (data missing or under 5 years), with no substitutes. Lookbacks of 126/189/252 days gave similarly low Sharpe in run 2 (0.21/0.15/0.24), so the result is not about lookback choice. A Sharpe of 0.25 over 11 years has a standard error of roughly 0.3, so it cannot be distinguished from zero.

### 3. What risk management did and did not do

On XAUUSD H4 (2020-2023), the replay with the risk engine (0.5% risk per trade) cut max drawdown from 14.1% to 5.0%, but the return was -1.1% (18 trades) versus +28.8% for buy-and-hold gold. Risk control can limit damage; it cannot create profit from a signal with no edge.

## What I take from this

Retail CFD markets may simply not leave an edge for simple strategies one person can build. That is information, not failure: it argues for choosing hypotheses that have a structural reason to work (who is on the other side of the trade, and why do they keep losing?) rather than testing more indicators on the same price data.

## Repository layout

```
src/project/
  Untility/         data loading, backtesting, walk-forward, spread fix, run_matrix
  Indicator/        indicators
  strategy/         Moving_Avg_Logic.py, tsmom.py
  risk/             position_sizing.py, risk_engine.py
  Execution/        paper_broker.py, replay.py, mt5_broker.py
Test/               unit tests (risk, paper broker, spread, TSMOM, MT5 adapter with a fake MT5)
Test/Notebooks/     experiment notebooks
```

(Folder names follow the code, including its spelling and capitalisation.)

### Risk engine defaults

0.5% risk per trade, 2% daily loss limit (close and block new entries until the next UTC day), 10% drawdown kill switch (latched until manually reset), one open position at a time.

### MT5 adapter safety

Refuses non-demo accounts unless `allow_live=True` is passed explicitly, always sends a stop-loss with the order, checks the contract size against the broker, rounds lots down to the volume step, retries only on clearly retryable return codes, and exposes `sync()` to reconcile with the broker's real state. Windows only; requires the MT5 terminal with Algo Trading enabled.

## Reproducing

Price history is **not** included in this repo. It is broker data pulled from MetaTrader 5 on your own account.

```bash
pip install -r requirements.txt
pytest Test/
```

Then export history from your own MT5 terminal using the loader in `Untility/data_loader.py` and run the notebooks.

## Roadmap

1. Smoke-test the MT5 adapter on a demo account with the market open.
2. Daily live loop on demo, with logging and alerts.
3. 3-4 week forward test on demo, compared against the backtest (fills, costs, swap).
4. New hypotheses only with a structural rationale, a stricter pass threshold that accounts for the 11 trials so far, and locked criteria (candidates: carry, cross-sectional momentum on a broader universe).
5. ML modules only after a non-ML baseline passes, and each must beat that baseline on locked criteria.

Real money is out of scope until a signal passes the locked criteria **and** a demo forward test.
