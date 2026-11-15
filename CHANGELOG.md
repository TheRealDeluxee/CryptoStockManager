# Changelog

All notable changes to CaSa are documented here.
Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

- **PATCH** (0.1.**1**): bug fixes only
- **MINOR** (0.**2**.0): new features; while the version starts with 0, this can also include changes to commands or files
- **MAJOR** (**1**.0.0): first stable release, after that only for changes that require adjustments by the user

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
