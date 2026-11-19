# Changelog

All notable changes to CaSa are documented here.
Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

- **PATCH** (0.1.**1**): bug fixes only
- **MINOR** (0.**2**.0): new features; while the version starts with 0, this can also include changes to commands or files
- **MAJOR** (**1**.0.0): first stable release, after that only for changes that require adjustments by the user

## [0.1.12] – 2026-11-19

### Added
- Optional Claude API: `func.claude(prompt, system, max_tokens, effort)` returns Claude's answer as text, only with a key in the new `config.ini` section `[anthropic]` (`api_key`, `model`, default `claude-opus-5-5`); without a key the API is not used. CaSa adds the empty section at the start. Needs the package `anthropic`; the start log line shows "Claude API yes/no", the key is never logged
- Plan alerts replace the price alerts: `planalert.ini` holds one simple trading plan per asset (support with `max_entry` %, target, stop, note). Hourly: buy zone and target; with the daily closes: stop. The message depends on whether you hold the asset (buy zone or add-on zone, take partial profits, setup cancelled or exit; buy zone and target at most once per day and plan); at the stop the plan ends. `/plan_alert` shows plans with status, saves one (`/plan_alert <asset> <support> <max_entry %> <target> <stop> <note>`) or deletes it with 🗑; `/price_alert` still works. `valuealert.ini` is converted at the first start (old file kept as `valuealert.old.ini`)
- `/ai_api_decision3s` and `/ai_api_plan` (Claude API key needed, otherwise a message that no key is set): the same data as the AI prompt, answered by Claude with web search. Decision: only the decision in three sentences. Plan: a short trading plan plus the values for `planalert.ini`; ✅ Save as plan saves it as plan alert, Discard drops it. With a key, buttons for both are below every alarm, info and `/check` (🤖 AI decision (API), 🤖 AI plan (API)); without a key they are not shown
- Claude answers end with tokens in/out, web searches and the cost (list price in USD, roughly in EUR), duration and model; `func.claude(..., with_usage=True)` returns these values
- `func.claude(..., web_search=True)`: Claude may search the web (at most 5 searches); answers are streamed and a paused long search is continued

### Changed
- `/ai` is now `/ai_prompt` (the old name still works)
- `/check`: the AI buttons are directly below the chart, the same as below alarms (no separate question any more)
- `/ai_prompt`, `/ai_api_decision3s` and `/ai_api_plan` are no longer in the Telegram command list (they are buttons); typing them still works
- `/record` replaces `/buy`, `/sell` and `/sellall` in the command list: buttons Buy, Sell and Sell all, or in one line `/record buy|sell|sellall ...`; the old commands still work

### Removed
- `/menu`, `/start` and `/journal`; the alarm journal report still comes on the 1st of each month with the evening summary

### Fixed
- Daily P&L, loss limit alarm and Cross-EMA Sell missed positions you hold when the asset had no `1` in `assets.ini`: assets you hold according to `transactions.csv` are now always analysed. Held assets missing in `assets.ini` (e.g. a different listing symbol) are reported once in Telegram ("Positions without prices") instead of only in the log; the log names inactive transactions once per symbol instead of once per line

## [0.1.11] – 2026-11-19

### Added
- `/menu`: all functions as buttons (Analysis · Portfolio · Alarms & dates · System)
- `/buy` and `/sell` step by step: asset via button (sell: only assets you hold), then amount, quantity and date (today/yesterday/other), confirmation; after saving ↩ Undo. The one-line input still works
- 🤖 AI prompt button below every alarm and info
- `/portfolio`: daily P&L table and 13-week chart at any time
- `/status` with the status of the main program: last analyses, next scheduled runs, alarms, memory, disk space, problems
- Wrong or incomplete input is answered with buttons: unknown asset in `/check`/`/ai`, `/history` and `/sellall` without asset, wrong `/smart_alert`; incomplete `/buy`/`/sell` start the step-by-step input
- Loss limit alarm (`loss_limit`): message when the total P&L of a position you own falls below `total_loss_limit` (new in `[alertsconfig]`, default -5 %); on the day it crosses the limit, again after a recovery above it. Like all signals switchable as alarm, info or off for stocks and crypto
- Alarms and infos separated: 🔔 alarms ("Potential buy/sell", can be important, in the alarm journal) and ℹ️ infos ("Info", never important, not in the alarm journal)
- Infos: trend turns up (Cross-EMA Buy) and sharp drop (Negative-5) are back as infos; new 100-day high and 100-day low of the price (replace the former 100-Minimum/-Maximum of the fast EMA); each info names its backtest result
- Every signal is set to alarm, info or off, separately for stocks and crypto: `config.ini` `[alerts]` with lines `name = <stocks>; <crypto>` or in Telegram with `/smart_alert_config` (two buttons per signal); changes apply from the next analysis on without a restart. The section is required: CaSa creates it (or missing lines) at the start, with an explanation of the format and per line a comment with the meaning and the backtest probability for stocks and crypto. Defaults follow the backtest (e.g. Score>8 and the 100-day high as alarm only for crypto, Oversold only for stocks)
- Alarm journal (`alarm_journal.csv`): every sent alarm is recorded with its price; 20 trading days later (crypto: days) the change and whether it moved in the alarm direction are filled in. `/journal` shows the hit rate per alarm next to the backtest value; monthly report on the 1st with the evening summary. Not recorded in test mode
- Oversold alarm (buy, stocks only): the score falls to -8 or lower for the first time – after a strong sell-off stocks often recover a little (backtest 47 stocks: 58 % hits, +1.3 % vs. an average day after 20 days, positive in all periods); not for crypto, never marked as important
- README: results of the second backtest (54 assets) per alarm
- P&L chart after the daily P&L table: total profit/loss of all positions over the last 13 weeks (Monday to Sunday, current week up to today) as weekly bars with the value on each bar (sales included, not the portfolio value), green above zero, red below; positions without prices are named in the message
- Automatic function tests in `tests/` (pytest, 79 tests, without internet and without touching your own files): files, FIFO, alarm rules, bot commands, images, log; run with `python -m pytest`
- Daily P&L as an extra table image after the daily summary (`My Daily P&L`): your own positions from crypto and stocks together, sorted by 1-day profit in €, with value, bar, 1-day P&L (€ and %), total P&L (€ and %, open position vs. FIFO cost) and a total row; on days without stock analysis only crypto (noted in the message)
- Bot commands accept the symbol or the name from `assets.ini`, e.g. `/buy RHM.DE …` or `/buy Rheinmetall …`; names with spaces and the beginning of a name (at least 3 letters, e.g. `Trane`) work too
- `crypto|stock` no longer needed in the bot commands (`/buy`, `/sell`, `/check`, `/ai`): the type comes from `assets.ini`; the old format still works
- `/transactions` and `/history` as a table image like the daily summary: date, name, BUY/SELL, quantity, amount and price per unit, newest first; totals bought/sold at the top
- Health check every 6 hours in `logfile.txt` only (not sent to Telegram): uptime, bot, last analyses, alarms, price alerts, memory, disk space, log size; WARN with the problems if something looks wrong

### Changed
- Alarm, info and `/check` messages: the figure blocks are collapsed (Telegram expandable quote), score and traffic lights stay visible; if Telegram ever rejects the formatting, the message is sent as plain text instead of being lost
- `/ai` prompt: the fixed rules "buy only after a confirmed bottom" and "after a strong rise wait for a pullback" are replaced by the backtest findings for the asset type (stocks: slight rebound after strong sell-offs, waiting for a bottom did not help; crypto: strength continued, sell-offs often continued); the score gets a short note what it meant for stocks/crypto; new line "Signals today" with the alarms and infos of the day; leading market by the company's home country (US companies on German exchanges: S&P 500 instead of DAX); risk/reward line shortened to the ratio. An unchanged `ai_prompt.txt` from an earlier version is updated automatically, own changes stay
- Faster analysis with identical results: 5-day slope (ROC5/VROC5) for all values in one step instead of one regression per value (about 60x faster), score rank and alarms calculate only the score instead of the whole message, the score of the previous value is calculated at most once
- Exchange rates (e.g. USD/EUR) are fetched once per currency and hour instead of once per stock: fewer Yahoo requests
- Fewer packages to install: `scipy`, `plotly` and `pytrends` are no longer needed (`install_venv.bat` also without `mplfinance` and `ruptures`); `pytrends` is only needed for the unused `google_trends` function
- `config.ini`: section `[alarm]` renamed to `[alertsconfig]` (renamed automatically at the start, comments, token and key stay); `one_day_price_change` and `one_day_profit_limit` are no longer read
- Chart patterns: a breakout counts only when the close is clearly beyond the neckline (at least half an average daily range, 0.5 ATR); weaker breakouts are no longer reported. Backtest 54 assets: half as many messages, crypto +4.7 % instead of +2.8 % vs. an average day after 20 days (all periods positive), stocks unchanged (±0 %). Rule tested with ideas from the open-source projects stock-pattern and chart_patterns (own code)
- Pattern breakout is now an info instead of a buy/sell alarm (no advantage in the backtest with 54 assets)
- `logfile.txt`: clear start line with version, device, operating system, Python version, test mode and CoinGecko key (also for `/restart`)
- `logfile.txt`: short Telegram connection problems as one WARN line instead of an ERROR with traceback (the bot retries on its own); two instances with the same bot as one clear ERROR line

### Fixed
- After `/restart` the daily prices and tables are kept: summary, P&L chart and alarm journal work right away instead of only after the next daily analysis
- AI prompt: stocks from all German exchanges (`.SG`, `.MU`, `.DU`, `.BE`, `.HM`, `.HA`) use the DAX as leading market, not the S&P 500
- Charts are deleted even if sending fails (the `plots` folder no longer fills up during network problems)
- Long HTML messages (over 4,096 visible characters) are sent as plain text in several parts instead of being cut inside a tag and rejected by Telegram
- Chart captions: the 1,024-character limit counts the visible text only, so more charts arrive together with their text

## [0.1.10] – 2026-11-15

### Changed
- Figure blocks: comparison with the last 100 values as `(>73% of days)` = higher than on 73 % of the days
- Figure blocks without the +/- signal marks (the traffic light at the top shows the effect on the score)
- Long-term-Analysis: the distance to the 200-day average is shown at the price (`+4.4 % vs SMA200`), not at the average

### Fixed
- RSI14 line showed the inverted rank (27 % instead of 73 %); now the same direction as all other lines

## [0.1.9] – 2026-11-15

### Changed
- Risk/Reward line at the top only as a very short sentence (e.g. "Less room up than down."); prices and distances stay in the block below
- README: names in `assets.ini` should have at most about 15 characters

### Fixed
- Daily summary: long names (e.g. "Trane Technologies") no longer run into the price; they are shortened with "…" only as far as needed
- Start message without logo on Linux/Raspberry Pi when the file was copied from Windows as `logo.PNG`: the logo is now found regardless of upper/lower case
- Plain-text Telegram messages were not sent since 0.1.7 ("object of type 'generator' has no len()"), e.g. the confirmation after `/smart_alert`, "Transactions were reloaded" and the error summary of an analysis

## [0.1.8] – 2026-11-15

### Added
- Delete events in Telegram: `/event` lists the upcoming dates with a 🗑 button each; deleting asks for confirmation first
- Start message with the CaSa logo (text as caption below it); the "next event" line is gone (list with `/event`); assets shown as used/recommended maximum (crypto 8, with CoinGecko key 12; stocks 30) with a warning above it
- Price alerts (`valuealert.ini`, `/price_alert`): message with your note and the daily analysis when a price reaches your mark; checked every hour, removed after triggering; list with 🗑 buttons to delete

### Changed
- Confirmation lines as traffic light (🟢 🔴 🟡), one line per block below in the same order: new Long-term-Analysis (price vs. 200-day average) and Risk/Reward-Analysis (R1/S1 with ratio, information only ℹ️ – a "good" ratio did not predict better results in the backtest) with their own figure blocks; RSI line from the real RSI value (30/70), the comparison with the last 100 days only in brackets
- Figure blocks show the real value first and the position within the last 100 values in brackets, e.g. `ROC5  +6.1 %  (88 %)`
- `/ai` prompt: same chart pattern as the message (double bottom/top removed from the formation list), risk/reward line, score as 1–10 (no own conversion by the AI); shortened so that it still fits into one Telegram message
- `/alarm` renamed to `/smart_alert` (clearly different from the new `/price_alert`); `/alarm` still works
- Alarm headline "Potential buy" / "Potential sell" instead of "Buy" / "Sell": a signal, not a recommendation
- Score shown as 1–10 (rank vs. the last 100 days of the asset), confirmed by the long-term trend (200-day average), the three assessments (RSI-Analysis, Trend-Analysis, Short-term-Analysis) and the chart pattern (Pattern-Analysis) as short status (full pattern description as its own paragraph below), each line labeled and marked with a traffic light: 🟢 good, 🔴 bad, 🟡 neutral; the figure blocks below without repeating the sentences; also in the daily summary (column SCORE). Internally the score stays at -18 to +18.
- `logfile.txt` reworked: one line per event with level (ERROR, WARN, INFO), Telegram messages only as one short line instead of the full text, a status line after each analysis (ok, alarms, failed, duration), full details for errors; above 1 MB the log continues in a new file (`logfile.old.txt`)

### Fixed
- No more pandas 3.1 deprecation notices ("The inplace keyword … is deprecated") in the console: removed from CaSa's own code, hidden for yfinance
- Failed Telegram commands: the reply names the cause, and the full error is written to `logfile.txt` (before only to the console window, although the message pointed to the log)
- README: `/restart` reloads files but not the program code; after an update CaSa has to be stopped and started again

## [0.1.7] – 2026-11-15

### Added
- After `/check` the bot asks with two buttons whether an AI prompt should be created for the same asset

## [0.1.6] – 2026-11-15

### Added
- Optional free CoinGecko demo key (`config.ini` `[coingecko] api_key`): used if set, otherwise the free public access; an invalid key is reported immediately
- README: roadmap with planned features

### Changed
- Buy alarms (Positive-5, Score>8, pattern breakout up) also for watched assets with `1` that you do not own yet; the message marks them as watched. Sell alarms stay limited to your portfolio.
- Event reminders from `events.ini` now come one day before instead of on the day, with daily analysis; `/event` replies accordingly
- One error message per analysis run instead of one per retry and asset (e.g. when Yahoo blocks requests); retries are only logged
- Quarterly results already in `events.ini` are not reminded twice

### Fixed
- README: hourly stock data in foreign currencies is converted to EUR correctly (outdated limitation removed)

## [0.1.5] – 2026-11-15

### Added
- Chart pattern detection: Double/Triple Top/Bottom and (Inverse) Head and Shoulders, drawn in pink with neckline and target
- "Pattern:" sentence in alarms and `/check` messages
- Pattern-Breakout alarm (Buy 131 / Sell 231) on a fresh close through the neckline, never high priority
- Score>8 alarm (Buy 141): score rises above 8 for the first time
- Reminder one day before quarterly results for stocks in the alarm analysis (date from Yahoo), with daily analysis
- Chart: current price as a black tag on the percentage axis, like TradingView
- Start message with a link to the GitHub repository
- README: recommended number of assets for the free API limits

### Changed
- Alarms reworked after a backtest (9 assets, 7–14 years of daily data)
- Positive-5 only on the first value above the threshold; recommended threshold `seven_day_price_change = 10`
- Trend filter: Positive-5 and Score>8 only when the price is above the SMA200
- Pattern sentence in plain language for users without trading experience
- R1/S1 as tags outside the chart on the left; zones only as a short band at the left edge
- Turning points and max/min are hidden when a pattern is drawn
- Stock prices without dividend adjustment (as on TradingView/Xetra); support/resistance for stocks from daily high/low
- AI prompt shorter so that it fits into one Telegram message; the length check counts the visible text
- Text messages without link preview
- Reminders are not sent again after `/restart`

### Removed
- 100-Minimum / 100-Maximum, Cross-EMA Buy and Negative-5 alarms (no advantage in the backtest)

## [0.1.0] – 2026-11-14

### Added
- Telegram bot for notifications, charts and commands (replaces Pushover)
- `/check` and `/ai` with buttons, `/alarm`, `/event`, `/sellall`
- `assets.ini` as the list of all assets; `; 1` marks assets for the alarm analysis
- `events.ini` with reminders and automatic daily analysis
- AI prompt (`ai_prompt.txt`) with support/resistance, bottom/top formation, market phase and position
- Support/resistance zones and volume profile in the chart
- Daily summary as a ranking by 1-day P&L
- Longer history in the background (250 daily / 450 hourly values)

### Changed
- `watchlist.ini` renamed to `assets.ini`
- Whole program in English; `/termin` is now `/event`, `/alarm` uses `on/off` (old inputs still work)
- Chart: trading-day axis, 200 dpi, percentage axis on the left
- Version number is now kept in `main.py`; the `[version]` section in `config.ini` is no longer needed

### Fixed
- No duplicate daily crypto alarms
- Price no longer cut off at Bollinger breakouts
- Evening summary kept after `/buy` / `/sell`
- `requirements.txt` was missing a line break

## [0.0.3] and earlier

Pushover version with watchlist and basic technical analysis.
