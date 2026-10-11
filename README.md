# Polymarket Prediction Bot

[![CI](https://github.com/lowriskbtc/polymarket-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/lowriskbtc/polymarket-bot/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A single-process trading bot and Flask dashboard for Polymarket's short-horizon crypto markets (5 minute and 15 minute up/down markets on BTC, ETH, SOL, XRP and others). It combines rule-based strategies, a small machine-learning model, and reinforcement-learning gates that learn which entries to skip.

This repository is the **full-system snapshot from June 2026**. A later refactor stripped the bot down to the ML/RL scanners only. This version keeps the algorithmic strategies alongside them.

> Personal trading software shared as source. It can place real orders with real money once wallet credentials are configured. Nothing here is financial advice and no performance is claimed. Read the code and run it in paper mode first.

## What is inside

All of this lives in `polymarket_bot.py` (about 26,000 lines, deliberately kept as one process so the scanners share one wallet, one order path and one price feed).

**Rule-based strategies**

| Component | What it does |
|---|---|
| `SignalAnalyzer` | Contrarian strategy built on technical indicators, with per-symbol tuned parameters |
| `FadeGigaAnalyzer` and `FadeGiga*5m` | "Fade the giga candle": fades unusually large candles, with variants per asset and timeframe translated from TradingView Pine scripts |
| `HFTScanner` | High-frequency scanner for 5 minute markets, triggered by Binance trade flow |
| `BTCHermesScanner` | Markov-chain persistence trader on 5 minute candle direction (also in `btc_hermes_nightly.py`) |
| `ClaudePicksScanner` | Uses the Anthropic API to look for high expected-value opportunities across several play types |

**Machine learning and RL**

| Component | What it does |
|---|---|
| `BTCMLModel` | Pure-Python logistic regression on 10 hand-built features (no sklearn), retrained every 25 resolved trades |
| `BTCMLScanner` | Symbol-scoped scanner that runs the model against live Polymarket markets (one instance per symbol) |
| `HFTRL`, `BTCMLRl`, `ClaudePicksRL`, `ManipRL` | Online Q-table / bandit gates that learn which setups to take or skip |
| `RLShadowLog` | Records the trades the RL gates blocked and later resolves their outcomes, so the gate can be audited |
| `AutoPauseGate` | Pauses a strategy after 3 consecutive losses or a large loss within a 30 minute window |

**Plumbing and ops**

`CLOBWebSocketFeed`, `BinancePriceWS`, `BinanceFuturesWS`, `OKXLiqWS` supply prices and liquidations. `sizing_agent.py` and `sizing_calibration.py` size positions. `system_monitor.py` is a watchdog, `daily_report.py` and `scorecard.py` produce reports, and `decrypt_keys.py` loads an encrypted env file. `research/` holds the optimizer and walk-forward scripts used to tune the strategies. The `*.html` files are the dashboards for each system.

## Quickstart (paper mode)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python polymarket_bot.py
```

The dashboard serves on http://127.0.0.1:5002.

Without wallet credentials the bot cannot place orders and runs in paper mode. With credentials configured, each scanner has its own `live_mode` setting; a scanner trades live only when that setting is on and the wallet loaded successfully.

## Safety notes

- **Network exposure.** The dashboard has routes that change settings and place orders. It binds to `127.0.0.1` by default (`BOT_HOST` overrides). State-changing requests that come from another machine, or that pass through any proxy or tunnel (anything carrying forwarded-for headers), are rejected unless they carry `BOT_API_TOKEN` as an `X-API-Token` header or `?token=` parameter.
- **Secrets stay out of the code.** Wallet keys, the Telegram token and the Anthropic key are read from the environment. Use a dedicated wallet funded with an amount you can afford to lose.
- **Your wallet is public.** Every trade from a Polymarket address is visible on-chain.

## Tests

```bash
pip install -r requirements-test.txt
pytest tests
```

33 unit tests cover the parts that can be checked offline: Kelly sizing and its caps and kill switch, the calibration-driven stake halving, the encrypted env loader (round trip, wrong password, non-interactive runs), and the Platt and isotonic probability calibration. Every protection was verified by a deliberate mutation that the suite catches. The tests never import `polymarket_bot.py` (it builds exchange clients on import), block the network and strip credentials.

Two sizing behaviors are pinned by tests as they are today and are worth a decision: the daily kill switch uses `abs(daily_pnl)`, so a big winning day also stops trading, and a negative-edge Kelly result is floored up to `min_stake` instead of skipping the bet.

## Known limitations

- One very large file. It works as a single deployable unit but is hard to navigate; the unit tests cover the sizing, key loading and calibration modules around it, not the monolith itself.
- Written for one operator on Windows; paths, process supervision and defaults assume that.
- This snapshot was published without being run, so treat it as source to read and adapt.
- Strategy parameters are tuned defaults from the author's own backtests, not recommendations.
