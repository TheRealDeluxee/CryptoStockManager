# Configuration

[← back to the README](../readme.md)

All files are plain text and can be edited with any editor. After changes, send `/restart`.

## config.ini

```ini
[telegram]
token = YOUR_TELEGRAM_BOT_TOKEN
chat_id = YOUR_CHAT_ID

[alertsconfig]
seven_day_price_change = 10     # threshold for Positive-5 / Negative-5 (%), 10 recommended by the backtest
previous_alarm_change = 3       # minimum change before the same alarm is sent again
test_mode = 0                   # 0 = scheduled analyses, 1 = run all analyses once at start
total_loss_limit = -5           # loss limit alarm: total P&L of an own position in % (e.g. -5 = alarm when it falls below -5 %)
infos_for = all                 # infos (not alarms) for: all = every asset in the analysis, portfolio = only assets you hold (fewer messages)

[coingecko]
# optional: free demo key from coingecko.com/en/developers/dashboard (empty = free public access)
api_key =

[alerts]
# Which signals CaSa sends, separately for stocks and crypto:
#   name = <stocks>; <crypto>      each one of: alarm, info, off
#   alarm = message "Potential buy/sell", can be marked important, counted in the alarm journal
#   info  = message "Info: ...", never important, not counted in the alarm journal
#   off   = no message
# The comment behind each line: what the signal means and how often the price moved in the expected
# direction within 20 days in the backtest (54 assets, up to 14 years); "normal" = the same on an
# average day. Also changeable in Telegram with /smart_alert_config.
#                stocks crypto
positive5      = alarm; alarm # price +10 % in 5 days (uptrend) | rises: stocks 54 % (normal 52 %), crypto 54 % (normal 49 %)
score8         = info;  alarm # score above 8 for the first time (uptrend) | rises: stocks 55 % (normal 52 %, no average gain), crypto 56 % (normal 49 %)
oversold       = alarm; off   # score -8 or lower: strong sell-off | rises: stocks 58 % (normal 52 %), crypto 48 % (normal 49 %)
cross_ema_sell = alarm; alarm # trend turns down (only own positions) | falls: stocks 45 % (normal 48 %, but -0.9 % on average), crypto 55 % (normal 51 %)
loss_limit     = alarm; alarm # total P&L of an own position falls below total_loss_limit in [alertsconfig] | risk alarm for your own limit, no backtest
cross_ema_buy  = info;  info  # trend turns up | rises: stocks 57 % (normal 52 %), crypto 53 % (normal 49 %)
negative5      = info;  info  # price -10 % in 5 days | rises: stocks 53 % (normal 52 %), crypto 48 % (normal 49 %)
high100        = info;  alarm # new highest price of 100 days | rises: stocks 55 % (normal 52 %), crypto 60 % (normal 49 %)
low100         = info;  info  # new lowest price of 100 days | rises: stocks 57 % (normal 52 %), crypto 53 % (normal 49 %)
pattern        = info;  info  # chart pattern breaks out | in breakout direction: stocks 51 % (normal 51 %), crypto 56 % (normal 50 %)

[anthropic]
# optional: key for the Claude API (https://console.anthropic.com), empty = Claude API not used
# the API costs money per request (billed by Anthropic per token)
api_key =
model = claude-opus-5-5
```

**Section names:** `[alertsconfig]` holds the thresholds (it was called `[alarm]` up to 0.1.10 and is renamed automatically at the start; a missing `total_loss_limit` is added with -5). `one_day_price_change` and `one_day_profit_limit` from older versions are no longer used and can be deleted.

**Fewer infos:** `infos_for = portfolio` sends infos (ℹ️) only for assets you hold; for watched assets ("Watched asset – not in your portfolio") only the alarms come. `infos_for = all` (default) sends infos for every asset in the analysis. CaSa adds the line at the start if it is missing. Also switchable in Telegram: the last button of `/smart_alert_config` (ℹ️ Infos for: all assets / only assets you hold), used from the next analysis on.

**Alarms and infos:** the `[alerts]` section is required. Each line is `name = <stocks>; <crypto>`, each one of `alarm`, `info` or `off`; the comment behind it says what the signal means and how often the price moved in the expected direction within 20 days in the backtest, for stocks and crypto, compared with an average day ("normal"). If the section or a line is missing, CaSa adds it at the start with the defaults; lines in the earlier format (one value: all, stock, crypto, off) are converted without changing their effect. `/smart_alert_config` in Telegram shows two buttons per signal (📈 stocks, 🪙 crypto); each tap switches alarm → info → off and rewrites only that line, with its comment. The change applies from the next analysis on, without `/restart`.

**CoinGecko key (optional):** if `api_key` is set, CaSa uses it; otherwise the free public access. With a key, the limit (30 requests per minute, 10,000 per month) applies per key instead of per internet connection – useful for more cryptos or several CaSa instances on one network.

**Claude API key (optional):** with an `api_key` in `[anthropic]`, `/ai_api_decision3s` and `/ai_api_plan` work and the function `func.claude(prompt)` sends a prompt to Claude and returns the answer as text (`web_search=True`: Claude may search the web, at most 5 searches); without a key the Claude API is not used (`func.claude_available()` is False). It needs the Python package `anthropic` (`python -m pip install anthropic`, also on the Raspberry Pi). Each request costs money (billed by Anthropic per token); the answers of `/ai_api_decision3s` and `/ai_api_plan` end with a line showing tokens in/out, web searches, the cost at list price in USD (and roughly in EUR), duration and model, e.g. `12,345 tokens in · 678 out · 3 web searches · $0.062 (≈ 0.053 €) · 41 s · claude-sonnet-5-5`. The log shows the same, never the key. The prices are in `func.CLAUDE_PRICES`; for a model not listed there only the tokens are shown. Refusal fallbacks are on: if the model refuses a request, the API retries it once with a fallback model. CaSa adds the empty section at the start if it is missing.

## assets.ini

All assets known to the program. Crypto symbols are CoinGecko IDs, stock symbols are Yahoo Finance symbols. Names should have at most about 15 characters; longer names are shortened with "…" in the daily summary.

```ini
# Format: symbol = Name; crypto|stock; 1
# 1 at the end = part of the automatic alarm analysis
# without 1    = only available for /check and /ai_prompt
# # at the start hides the entry completely

[crypto]
bitcoin        = Bitcoin; crypto; 1
cardano        = Cardano; crypto

[stocks]
NVD.DE         = Nvidia; stock; 1
APC.DE         = Apple; stock
```

The `1` can also be switched in Telegram with `/smart_alert`.

**Assets you hold are always analysed:** everything you hold according to `transactions.csv` (bought minus sold above zero) is part of the analysis even without `1` – so it is in the daily P&L, gets the loss limit alarm and Cross-EMA Sell. The symbol in `transactions.csv` must be the same as in `assets.ini` (e.g. `TSFA.F`, not `TSFA.DU`). If a held asset is missing in `assets.ini`, CaSa sends one Telegram message "Positions without prices" with the symbols.

**How many assets with `1`?** The free APIs limit the requests:
- **Crypto (CoinGecko): at most about 8, with own key about 12.** Each coin needs about 26 requests a day (hourly analysis); more coins cause "too many requests" errors.
- **Stocks (Yahoo): 30–40 are fine.** About 3 requests per stock and day – the real limit is the number of alarms.
- Prefer liquid listings in EUR (e.g. Xetra `.DE`): no exchange rate request, reliable volume.
- The start message shows how many are in use, e.g. `6/12 crypto, 12/30 stocks`, and warns when there are more than recommended.

## events.ini (optional)

```ini
# Format: name = symbol; DD/MM/YYYY; description
[events]
nvda_20261117  = NVD.DE; 17/11/2026; Q3 earnings after US market close
```

Entries can also be added with `/event` and deleted via the 🗑 buttons in the `/event` list. The reminder comes at 08:00 one day before. Past dates are ignored.

## planalert.ini (optional)

Simple trading plans, one per asset – for example from the trading plan table of an AI analysis. CaSa watches four numbers; everything else (tranche size, "higher low", news) stays in your note, which comes with every message.

```ini
[TSFA.F]
support   = 312.15
max_entry = 2
target    = 358.93
stop      = 312.15
note      = 1/4 of the position after a rebound; also cancel if revenue guidance is cut
start     = 19/11/2026
status    = waiting
```

| Setting | Meaning |
|---|---|
| `support` | Buy zone starts here |
| `max_entry` | Buy only up to this many % above the support (0 = only at or below it) |
| `target` | Take partial profits at or above this price |
| `stop` | The plan fails at a daily close below this price |
| `note` | Your plan in words |
| `start`, `status` | Set by CaSa: daily closes from `start` on count for the stop; `status` = waiting, buy_zone, target or ended |

`support` or `target` is required, the rest is optional. Whether you hold the asset according to `transactions.csv` decides the message:

| Event | Checked | Not held | Held |
|---|---|---|---|
| Price in the buy zone (support up to + max_entry %) | every hour | 🟢 Buy zone: check the conditions in your note | 🟢 Add-on zone (retest) |
| Price at or above the target | every hour | – | 💰 Take partial profits |
| Daily close below the stop | with the daily closes | ❌ Setup cancelled | 🔴 Exit: sell, do not average down |

Each message comes once and again only after the price has left the zone, at most once per day and plan (a price moving around the mark does not send a message every hour). Like an alarm it is one message with chart, score and figures; at the top "📋 Plan alert" with the plan numbers and your note. Buttons: buy zone **🟢 Record buy**, target **🔴 Record sell** (the asset is already chosen, the input starts at the amount), plan ended **🗑 Delete plan** (when you hold the asset also Record sell and Sell all). Date reminders from `events.ini` and the quarterly results come the same way with "📅 Event tomorrow". At the stop the plan ends; it stays in the file and in `/plan_alert` until you delete it. Stocks are checked Monday to Friday, 08:00–22:00. The file is created with explanations by the first `/plan_alert`.

**From valuealert.ini (up to 0.1.11):** at the first start CaSa converts the price alerts into plans – `below` becomes `support` (with `max_entry = 0`, so it triggers as before), `above` becomes `target`, notes are kept. The old file is kept as `valuealert.old.ini`. Converted plans have no stop yet: add it in the file or save the plan again with `/plan_alert`.

## transactions.csv

Written by `/record` (buy, sell, sell all):

```
type;symbol;name;action;amount;qty;date
crypto;bitcoin;Bitcoin;buy;500.0;0.01;01/01/2024
stock;QCOM;Qualcomm;sell;312.16;2.0;14/11/2026
```

## AI prompt (config.ini `[ai_prompt]`)

Template for the `/ai_prompt` prompt (also the data for `/ai_api_decision3s` and `/ai_api_plan`, which add their own answer format at the end). It is the last section of `config.ini`, added automatically at the start, and can be edited freely:

```ini
[ai_prompt]
template =
  Assess {name} ({symbol}, {type}) for a three-month horizon ...

  A. My program's data (daily values, {date})
  ...
```

Every line of the text starts with two spaces; empty lines are allowed. Lines starting with `#` or `;` are treated as comments. Up to 0.1.12 the template was the file `ai_prompt.txt`: at the start CaSa moves its text into `config.ini` (your own changes stay) and keeps the old file as `ai_prompt.old.txt`. Placeholders in curly brackets such as `{name}`, `{price}`, `{support}`, `{bottom}` or `{market}` are filled in by the program. Since 0.1.11 also `{signals}` (alarms and infos the analysis would send today for this asset) and `{evidence}` (the backtest findings for stocks or crypto, e.g. that stocks recovered slightly after strong sell-offs while crypto tended to keep falling). The leading market is chosen by the company's home country (Nvidia on Xetra: S&P 500), not by the exchange. An unchanged template from an earlier version is replaced by the current one automatically; your own changes are never overwritten.

## alarm_journal.csv (created automatically)

Every sent alarm is recorded with date, asset, alarm, price and score. After 20 trading days (crypto: 20 days) CaSa fills in the price, the change in % and whether the price moved in the alarm direction (buy: up, sell: down). This shows over time whether the alarms also work on new data and not only in the backtest. On the 1st of each month the statistics come with the evening summary. Alarms are not recorded in test mode (`test_mode = 1`), because all analyses run at every start there. The file is private and not part of the repository.

## logfile.txt

Written by the program for error analysis and monitoring – one line per event:

```text
2026-11-15 16:39:00  INFO   CaSa 0.1.10 started on raspberrypi (Linux 6.12, Python 3.13.5), test mode off, CoinGecko key yes
2026-11-15 16:39:59  WARN   CoinGecko ripple: too many requests (attempt 1/5), waiting 60 s
2026-11-15 16:40:12  INFO   Analysis crypto 100d done: 6/6 ok, 1 alarms, 0 failed, 23 s
2026-11-15 16:41:02  ERROR  Analysis NVD.DE (100d) failed: no data received after 5 attempts
```

`ERROR` = something failed, `WARN` = problem, the program continues, `INFO` = status (start with version and device, analyses, sent messages as one line). Errors include the full details below the line. Short Telegram connection problems are only one `WARN` line (the bot retries on its own); two instances with the same bot (e.g. PC and Raspberry Pi) are reported as one clear `ERROR` line. Above 1 MB the file is renamed to `logfile.old.txt` and a new log starts.

Every 6 hours a **health check** is written to the log (not sent to Telegram): uptime, Telegram bot, last run of each analysis, alarms since the start, price alerts, memory, free disk space and log size. If something looks wrong (bot stopped, hourly analysis older than 2 h, analyses failing, less than 1 GB disk space), the line is a `WARN` with the problems at the end:

```text
2026-11-19 12:00:00  INFO   Health check: up 3 d 5 h, bot ok, crypto 100h 19/11 11:43 6/6 ok, stocks 100d 19/11 09:30 14/14 ok, alarms since start 2, price alerts 1, memory 172 MB, disk free 21.4 GB, log 42 KB
```

Useful searches, e.g. on a Raspberry Pi: `grep -a "CaSa 0" logfile.txt | tail` (starts and versions), `grep -a "Health check" logfile.txt | tail` (state), `grep -a ERROR logfile.txt | tail` (errors).

`config.ini`, `assets.ini`, `events.ini`, `planalert.ini`, `transactions.csv` and the log files contain personal data and are excluded from Git via `.gitignore`.
