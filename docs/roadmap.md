# Roadmap

[← back to the README](../readme.md)

## Usability
- [ ] **One-line summary** at the top of every message, e.g. "🟢 Uptrend, strong buying interest – buy signal". The details follow below for anyone who wants them.
- [x] **Easier score**: The −18…+18 score was hard to read. Now a score 1–10 with the assessments as traffic-light confirmation (🟢 🔴 🟡).
- [x] **Simpler score blocks**: one compact numbered list with bars (position within the last 100 days) and plain names in the summary (200d trend, 7d trend, 5d mom.).
- [ ] **`/explain` command**: Explains terms like RSI, SMA200 or neckline in two sentences.

## Setup
- [x] **Raspberry Pi image**: write the image, answer a wizard at the first login; CaSa runs as a service.
- [x] **Windows setup wizard** (`install_casa.bat`): installs Python and CaSa, asks for the keys.
- [x] **Updates in Telegram**: `/update` and a daily check for a new release (CaSa image).
- [x] **Disclaimer in the README**: CaSa provides technical signals only, no investment advice.
- [ ] **Disclaimer in the start message** as well.

## Backtesting
- [ ] **Backtest script in the repo**: The backtest currently lives in a temporary folder. In the repo, every rule change can be checked against historical data.
- [x] **Hit rate in alarm messages**: every alarm and info ends with e.g. "Backtest: price rose in 57 % of cases (after 20 days)". Helps users judge how reliable a signal is.
- [ ] **Sideways market detection (ADX)**: Today only up- and downtrends are detected (SMA50/SMA200). ADX adds a sideways phase, in which many trend signals fail.

## Indicators & charts
- [ ] **ATR-based stop-loss suggestion**, e.g. "Stop around 142 € (2× ATR)". Adapts the stop to how much the asset usually moves.
- [x] **Relative strength vs. index**: every stock message has a line "Market" (leading index and sector ETF today) and the figure vsIndex (5 days vs. the index, bar = position within the last 100 days); the same values go to the AI.
- [ ] **Market overview in the evening summary**: the main indices (DAX, EURO STOXX 50, S&P 500, Nasdaq-100, Nikkei 225, MSCI World) with 1 and 7 days, plus 1–2 theme ETFs (e.g. AI, data centres).
- [ ] **Own benchmark per asset**: an optional field in assets.ini, e.g. `TT = Trane Technologies; stock; 1; USD; SMH`, to compare with a theme (AI chips) instead of the broad sector; European sector ETFs for European stocks.
- [ ] **R2/S2 zones**: A second resistance and support zone, plus an alarm when the price breaks through a zone.
- [ ] **More chart patterns**: Triangles, wedges and flags in addition to the existing patterns.
- [ ] **Pattern detection with a trained model**: A small model trained on the PC with historical charts recognises patterns instead of the fixed rules of today (double top/bottom, head and shoulders); it runs on the Raspberry Pi without an AI accelerator. Only used if it beats the current rules in the backtest.
- [ ] **Fair Value Gaps**: Price gaps between candles that the price may return to. Popular but unproven, so they are tested in the backtest before any alarm is added.

## Alerts & events
- [x] **Own price alarms**: user-defined price levels, independent of the indicators; since 0.1.12 part of the trading plans (`/price_alert` still works as old name of `/plan_alert`).
- [x] **Simple trading plans** (`planalert.ini`, `/plan_alert`): buy zone, target and stop per asset.
- [ ] **Plan conditions**: higher low, close above the first rebound high, above EMA12/SMA200; the AI prompt writes a ready plan line.
- [ ] **Options expiry and market holidays**: The major expiry date (third Friday of March, June, September and December) is calculated automatically. Exchange holidays come from a fixed list.
- [ ] **Crypto events**: Known dates such as Bitcoin halvings in `events.ini`.
- [ ] **Automatic import of economic dates**: Fed, CPI and ECB dates without manual entry. Needs a calendar source.
- [x] **Delete dates directly in Telegram**: Remove entries from `events.ini` via chat.

## Data & sentiment
- [ ] **Fear & Greed index** as a sentiment line in the daily summary. Free for crypto (alternative.me), for stocks only via an unofficial CNN source.
- [ ] **Google Trends**: Search interest as a sentiment indicator, especially for crypto. A `google_trends` function already exists but is not used yet; `pytrends` is no longer installed by default (`python -m pip install pytrends`).
- [ ] **Alarm for unusual news volume**: Triggers when an asset gets far more news than usual (source: yfinance `Ticker.news`). It shows that something is happening, not in which direction.
- [ ] **Insider trades**: Purchases and sales by executives from free official sources: SEC EDGAR (Form 4) for US stocks, Directors' Dealings for German stocks. Large insider purchases are a documented signal.
- [ ] **Regulatory news via RSS**: Press releases from SEC and BaFin, flagged when an asset from the watchlist is mentioned.

## Portfolio
- [ ] **Read-only exchange API for crypto (ccxt)**: Loads holdings directly from the major exchanges. Read-only keys only, never with trading rights.
- [x] **Total value and daily P&L** in the evening summary.
- [ ] **Position size suggestion**: Based on ATR, e.g. max. 2 % of the portfolio at risk per position.

## Reliability
- [ ] **Keep known alarms after a full restart**: They are currently lost, so the same alarm can be sent twice.
- [x] **Health check**: A status line in the log every 6 hours, with a warning when an analysis is overdue or failing.

## Not planned
- **Opening/closing volatility**: Needs minute data, while CaSa works with daily data.
- **X/Twitter tracking**: The API is expensive and the posts are hard to evaluate.
