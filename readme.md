![CaSa](logo.png)

# CaSa – Crypto and Stock Alert

CaSa watches your cryptocurrencies and stocks, analyses them with technical indicators and sends alerts, charts and daily summaries to **Telegram**. The same Telegram chat is used to control the program: run analyses on demand, record buys and sells, switch alarms on and off, plan important dates and create prompts for ChatGPT or Claude.

## Contents

- [Why CaSa?](#why-casa)
- [Features](#features)
- [What's New](#whats-new)
- [Quick Start](#quick-start)
- [CaSa Image for the Raspberry Pi](#casa-image-for-the-raspberry-pi)
- [Telegram Commands](#telegram-commands)
- [Daily Schedule](#daily-schedule)
- [Understanding Alerts and Charts](#understanding-alerts-and-charts)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License, Disclaimer and Links](#license)

## Why CaSa?

Casa means “house” or “home” in Spanish. I’m saving to buy a place of my own, and I know many people my age and younger who are trying to do the same. With home prices so high, I started asking myself what I could do today to get a little closer to that goal. That question led me to investing.

As I started learning about stocks and cryptocurrencies, I kept wondering what might happen while I was away from the screen. Crypto prices could change quickly, and I worried about how much I could lose. I wanted a way to follow the markets without checking prices all day. The apps I tried could tell me when a price reached a certain level, but I wanted more control over when I got an alert. That usually meant paying for a subscription. So I built the tool I wanted to use and called it CaSa, after the goal that got me started.

CaSa is free and open source on GitHub. It uses very little computing power, so it can run on a low-cost computer without a cloud server or paid API subscriptions. It explains what you’re seeing in simple words, even if you’re new to investing. I started CaSa with a home of my own in mind. But now I hope it helps others take their first steps in investing, whatever they’re saving for. If you try it, I’d love to hear what would make it more useful to you. You can see what I’m planning next in the [roadmap](https://github.com/TheRealDeluxee/CryptoStockManager#roadmap).

![CaSa alert in Telegram on iPhone and Apple Watch](docs/telegram_phone_watch.png)

## Features

**Analysis**
- Crypto data from CoinGecko, stock data from Yahoo Finance; foreign currencies (USD, GBP, ...) are converted to EUR
- 250 daily / 450 hourly values are loaded, indicators are calculated on the full history and the last 100 values are evaluated
- Scoring system from RSI14, SMA7/VMA7, ROC5/VROC5 with volume support and fake-out detection, shown as an easy score 1–10 with reasons
- EMA 12/26 crossovers, Bollinger Bands (20,2), 7-day trend slope
- Support/resistance zones from pivot points of the full history (stocks: daily high/low, like on TradingView)
- Chart pattern detection: Double/Triple Top/Bottom and (Inverse) Head and Shoulders with neckline, target and invalidation level

**Alarms**
- Automatic hourly and daily analysis of all assets marked with `1` in `assets.ini`
- Triggers chosen by a backtest: strong 5-day rise, score above 8, EMA crossover downwards, chart pattern breakout
- Trend filter: buy alarms only above the 200-day average
- Buy alarms also for watched assets you do not own yet; sell alarms only for your portfolio
- Duplicate filter: the same alarm is not sent twice in a row

**Charts and Messages**
- Chart without weekend gaps: price with price tag, EMAs, Bollinger Bands, signals, support/resistance (R1/S1), chart pattern, volume bars and volume profile
- Clean Telegram messages (bold headers, aligned values)
- Daily ranking of all assets by 1-day P&L
- Daily P&L of your own positions (crypto and stocks together) after the summary: value, 1-day profit/loss in € and %, total profit/loss, total row
- P&L chart of the last 13 weeks as weekly bars: the realised profit/loss of your sales in each week (from `transactions.csv`, FIFO) in full colour, one part per sale (white line between them; gains upwards, losses downwards); for the current week also the unrealised price change of your open positions in light colour, one part per open asset (gains upwards, losses downwards). Values only for the realised parts. At the top the realised total of the 13 weeks, below it in brackets the unrealised gains and losses of this week, e.g. (+400.00 | −700.00) €

![Example summary](docs/example_summary.png)

**Portfolio**
- Buys and sells in `transactions.csv` (FIFO bookkeeping)
- Profit/loss per asset, background colour of the chart shows profit or loss
- Holding period: amount held longer than one year (tax-relevant for crypto in Germany)

**Planning and Research**
- `/ai_prompt` creates a detailed prompt for ChatGPT or Claude: technical data, support/resistance with risk/reward, chart pattern, bottom/top formation, score 1–10, market phase of the asset and its leading market, your position and common investment rules – no AI API needed, just copy and paste
- With a Claude API key: `/ai_api_decision3s` answers directly with the decision in three sentences, `/ai_api_plan` with a trading plan that you save as plan alert with one tap
- Important dates (earnings, central bank meetings, ...) in `events.ini` or via `/event`: reminder one day before, plus daily analysis
- Trading plans in `planalert.ini` or via `/plan_alert`: buy zone, target and stop per asset; messages depend on whether you hold the asset, plus daily analysis
- Quarterly results: automatic reminder one day before for all stocks in the alarm analysis (date from Yahoo), plus daily analysis

## What's New

**Version 0.1.12** – all changes in [CHANGELOG.md](CHANGELOG.md)

- Plan alerts: `planalert.ini` (replaces `valuealert.ini`, converted automatically) with one simple trading plan per asset – buy zone, target, stop; messages depend on whether you hold the asset; `/plan_alert`
- Optional Claude API (`config.ini` `[anthropic]`): buttons 🤖 AI decision (API) and 🤖 AI plan (API) below alarms, infos and `/check`, with web search, cost per answer; the plan can be saved as plan alert with one tap
- `/record` for buy, sell and sell all; `/menu`, `/start` and `/journal` removed; AI buttons directly below `/check`
- Fix: positions you hold are always analysed (daily P&L, loss limit), also without `1` in `assets.ini`

**Version 0.1.11**

- Easier bot: `/menu` with buttons, `/buy`/`/sell` step by step with ↩ Undo, 🤖 AI prompt button below alarms, collapsible figure blocks, `/portfolio`, detailed `/status`
- Alarms and infos separated; every signal per stock/crypto as alarm, info or off (`config.ini` `[alerts]`, `/smart_alert_config`)
- New: Oversold alarm (stocks), loss limit alarm, new 100-day high/low, sharp drop and trend-turn infos
- Second backtest with 54 assets: results per alarm in the README; chart patterns only on clear breakouts
- Daily P&L table and 13-week P&L chart, alarm journal (`/journal`) with the hit rate on new data
- Bot commands with symbol or name, transactions as table images, AI prompt with backtest findings
- Faster analysis, fewer packages

**Version 0.1.10**


- Clearer figure blocks: real value first, then `(>73% of days)` instead of signal marks
- RSI comparison now in the same direction as all other lines
- Distance to the 200-day average shown at the price

**Version 0.1.9**

- Fix: plain-text Telegram messages (e.g. after `/smart_alert`) are sent again
- Fix: logo in the start message also on Linux/Raspberry Pi
- Fix: long names no longer overlap the price in the daily summary
- Shorter Risk/Reward line at the top of the message

**Version 0.1.8**

- Easier messages: score 1–10 with traffic-light lines (🟢 🔴 🟡) and one figure block per line – long-term trend, risk/reward, RSI, trend, short-term, chart pattern
- Price alerts with `/price_alert`: message with your note when a price reaches your mark
- `/smart_alert` (formerly `/alarm`), delete dates with 🗑 in `/event`
- Start message with logo and recommended number of assets
- Clearer `logfile.txt`, errors of Telegram commands shown directly
- AI prompt with the same chart pattern, risk/reward and score as the message

**Version 0.1.7**

- After `/check` the bot asks whether an AI prompt should be created – one tap instead of a second command

**Version 0.1.6**

- Buy alarms also for watched assets you do not own yet
- Reminder one day before every date in `events.ini`
- Optional free CoinGecko demo key for more cryptos or several instances
- One error message per analysis instead of a flood of messages when a data source blocks
- [Roadmap](#roadmap) with planned features

**Version 0.1.5**

- Chart pattern detection (Double/Triple Top/Bottom, Head and Shoulders): drawn in pink, explained in plain language in every message, own breakout alarm
- Alarms reworked after a backtest with 9 assets over 7–14 years: new Score>8 alarm, SMA200 trend filter, weak alarms removed – far fewer but better alarms
- Reminder one day before quarterly results, with daily analysis
- Charts: current price tag, R1/S1 as tags on the left; stock prices and zones like on TradingView
- Shorter AI prompt that fits into one Telegram message

**Version 0.1.0**

- Telegram bot instead of Pushover: notifications, charts and commands in one chat
- Redesigned charts (trading-day axis, 200 dpi, volume profile, support/resistance) and daily ranking image
- New commands: `/check` and `/ai` with buttons, `/alarm`, `/sellall`, `/event`
- `assets.ini` replaces the old watchlist; `; 1` marks assets for the alarm analysis
- AI prompt with bottom/top formation and market phase (SMA50/SMA200, Golden/Death Cross)
- Event reminders with automatic daily analysis
- Bug fixes: no duplicate daily crypto alarms, price no longer cut off at Bollinger breakouts, evening summary kept after `/buy` / `/sell`

## Quick Start

1. **Install Python 3** (tested with Python 3.13)
2. **Install the dependencies**
   - Windows: run `install_venv.bat` (creates `.venv` and installs `requirements.txt`)
   - Linux/macOS: `python3 -m venv .venv` and `.venv/bin/pip install -r requirements.txt`
3. **Create a Telegram bot**
   - Send `/newbot` to [@BotFather](https://t.me/BotFather) and copy the token
   - Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` – the number at `"chat":{"id":...}` is your chat id
4. **Create `config.ini` and `assets.ini`** in the program folder (see [Configuration](#configuration))
5. **Start CaSa**: `python main.py` (Windows: `.venv\Scripts\python main.py`)

After the start, the bot sends the CaSa logo with a short message: the version, the number of monitored assets and the next scheduled analysis. Send `/help` for all commands.

**Tested on**: RockPi3 with Armbian OS and Windows 11.

**Easiest on a Raspberry Pi:** the ready [CaSa image](#casa-image-for-the-raspberry-pi) – no Python installation, a wizard asks for the keys.

## CaSa Image for the Raspberry Pi

The easiest way to run CaSa: a ready **CaSa image** – Raspberry Pi OS Lite 64-bit (Debian 13 Trixie, without desktop) with the Python packages of CaSa, Samba and tmux already installed. Write it with the Raspberry Pi Imager, start the Pi, log in once: a wizard asks for the keyboard layout, whether you want the network drive (optional) and your keys, then downloads the **newest CaSa release from GitHub**. From then on CaSa starts by itself, also after a power cut.

CaSa itself is not in the image: it always comes from GitHub, so the same image works for every CaSa version.

You need: a Raspberry Pi 4 or 5, a microSD card (16 GB or more), a power supply, network (LAN or Wi-Fi) and a PC with an SD card reader.

### Hardware

| | Minimum | Recommended |
|---|---|---|
| Raspberry Pi | Pi 4, 2 GB | Pi 5, 8 GB (room for an own AI model) |
| Storage | microSD 32 GB, A2 | NVMe SSD (M.2 HAT) |
| Extras | official power supply | official 27 W power supply, Active Cooler |

64-bit only (Pi 3, 4, 5, Zero 2 W); the Zero 2 W (512 MB) is too small. An AI model trained on the PC runs on the Pi 5 without an AI accelerator.

### Step 1: Write the image

1. Install and start the **Raspberry Pi Imager** on your PC: [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. **Device**: your Raspberry Pi (4 or 5)
3. **Operating system**: scroll down to **Use custom** and choose `casa.img.xz` (see [Build the image](#build-the-image))
4. **Storage**: your SD card
5. **Next** → **Edit settings**:
   - Tab *General*: user name **`casa-user`** and a password (note it down); Wi-Fi name and password (not needed with a LAN cable). Hostname (`casa-pi`), time zone (Europe/Berlin) and keyboard (German) are already set in the image
   - Tab *Services*: **Enable SSH** → *Use password authentication*
6. **Save** → **Yes** → **Yes**: the card is written (all data on it is deleted)
7. Put the card into the Pi, connect power (and LAN), wait 2–3 minutes

Without the settings in step 5 the Pi asks on its own screen at the first start for the keyboard, a user name and a password (screen and keyboard needed).

### Step 2: Log in once – the wizard starts by itself

On the PC in **PowerShell** (or with screen and keyboard directly on the Pi):

```powershell
ssh casa-user@casa-pi
```

After the password the **CaSa setup wizard** starts by itself (keys: arrow keys and Tab, Enter to confirm):

| Window | What you do |
|---|---|
| Welcome | **Start** |
| Keyboard (1/4) | Your keyboard layout, e.g. **de** German (for a keyboard connected to the Pi) |
| Network drive (2/4) | Optional: **Set up** the drive `\\casa-pi\Python` (then a password, twice) or **No drive** – then you edit your files on the Pi (e.g. `nano ~/Python/assets.ini`). Later again with `casa-setup`: keep, new password or remove |
| Telegram (3/4) | Paste the bot token (from [@BotFather](https://t.me/BotFather), right-click in PowerShell) – it is checked with Telegram. Then send your bot any message: the chat id is found automatically |
| API keys (4/4) | CoinGecko and Claude key (both optional, leave empty for none) and the Claude model |
| Summary | **Apply** – downloads the newest CaSa from GitHub (internet needed) and updates the Python packages (1–3 minutes) |
| Done | CaSa runs, with its version; the address of the network drive |

Your bot sends "CaSa started" within a minute. Keys and passwords appear neither on the screen nor in a log.

### Step 3: Map the network drive in Windows (if set up)

1. Explorer → **This PC** → **Map network drive**
2. Folder: `\\casa-pi\Python` (or with the IP of the Pi: `\\192.168.x.x\Python`)
3. Tick **Connect using different credentials** → **Finish**
4. User `casa-user`, the password from the wizard (step 2/4), tick **Remember my credentials**

The drive shows your CaSa folder (`/home/casa-user/Python` on the Pi): `config.ini`, `assets.ini`, `transactions.csv`, `logfile.txt`, `plots` …

### Step 4: Your assets

Open `assets.ini` on the drive (an example with Bitcoin, Ethereum and Rheinmetall is there), enter your assets (see [assets.ini](#assetsini)), save and send `/restart` to the bot. Moving from an earlier installation: copy your `config.ini`, `assets.ini`, `transactions.csv`, `events.ini`, `planalert.ini` and `alarm_journal.csv` onto the drive and send `/restart` – stop CaSa on the old device first (only one CaSa per Telegram bot).

### Daily use on the Pi

| What | Command (after `ssh casa-user@casa-pi`) |
|---|---|
| Status | `systemctl status casa` |
| Log live | `tail -f ~/Python/logfile.txt` (end with Ctrl+C) – or `logfile.txt` on the drive |
| Stop / start / restart | `sudo systemctl stop casa` / `start casa` / `restart casa` – or `/restart` in Telegram |
| Update to the newest CaSa release | `/update` in Telegram, or the button **⬆ Update now** of the daily message (see below), or `casa-update` on the Pi (a certain version, also back: `casa-update v0.1.12`) |
| Keyboard, drive password, keys | `casa-setup` |
| Run by hand in tmux (as before) | `sudo systemctl stop casa`, then `tmux`, `cd ~/Python`, `.venv/bin/python main.py` |

**Updates:** `/update` in Telegram looks for a new release at once (e.g. right after you published one) and answers in any case – "up to date" or the message below. In addition, every day at 19:00 CaSa checks GitHub by itself. When there is a newer release, the bot sends one message "New CaSa version …" with a link to the changes and the buttons **⬆ Update now** and **Later**. *Update now* gets the release from GitHub, updates the Python packages and restarts CaSa (about 1 minute); your files stay. After the restart the bot confirms with "✅ Update complete: CaSa 0.1.12 → 0.1.13". If the update fails, CaSa keeps running with the old version and says why. The button needs CaSa from the image (Git copy, running as service); elsewhere the message only shows how to update.

CaSa runs as the service `casa`: it starts at every boot and again 30 seconds after an error. `casa-update` gets the newest release from GitHub, updates the Python packages and restarts CaSa; your files (`config.ini`, `assets.ini`, `transactions.csv` …) stay.

### Build the image

The image is built with **pi-gen**, the official tool of Raspberry Pi. It runs in Docker Desktop on the PC (only as build tool – there is no Docker in the image):

```bash
bash image/build_image.sh
```

(Windows: in Git Bash, in the CaSa folder.) It takes about 1–2 hours and about 15 GB of free disk space. Result: `image/deploy/<date>-casa.img.xz`. Neither CaSa itself nor private files (`config.ini`, `transactions.csv` …) go into the image – only the system, the Python packages of `requirements.txt` and the wizard. A new image is only needed for changes of the system, not for a new CaSa version. Settings (hostname, user, time zone, keyboard): `image/config`; what is installed: `image/stage-casa`; the wizard and the update: `image/stage-casa/00-casa/files/casa-setup` and `casa-update`. The Pi downloads what is **released** on GitHub (newest tag such as `v0.1.12`). Afterwards `docker volume rm casa-pigen` frees about 10 GB.

## Telegram Commands

| Command | Function |
|---|---|
| **Analysis** | |
| `/check` | Daily analysis of one asset without alarm (choose via buttons or `/check <symbol or name>`); below the chart the same buttons as below alarms: 🤖 AI prompt, with a Claude API key also 🤖 AI decision (API) and 🤖 AI plan (API) |
| 🤖 AI prompt (button) | AI prompt to copy into ChatGPT or Claude; below every alarm, info and `/check`. Typing `/ai_prompt <symbol or name>` (or the old `/ai`) still works, it is just not in the command list any more |
| 🤖 AI decision (API) (button, `/ai_api_decision3s`) | Claude API (key needed): the same data as `/ai_prompt`, answered directly with only the decision in three sentences (with web search for news) |
| 🤖 AI plan (API) (button, `/ai_api_plan`) | Claude API (key needed): the same data, answered with a short trading plan and the values for `planalert.ini`; **✅ Save as plan** saves it (replaces an existing plan of the asset), **Discard** drops it |
| `/analyse <daily\|hourly\|summary>` | Start an analysis immediately |
| **Alarms and Dates** | |
| `/smart_alert` | Switch assets on/off for the alarm analysis (buttons or `/smart_alert <symbol or name> <on\|off>`); the old `/alarm` still works |
| `/smart_alert_config` | Alarm, info or off per signal, separately for stocks and crypto (buttons); saved in `config.ini` `[alerts]`; the last button switches infos for all assets or only for the assets you hold (`infos_for`) |
| `/event` | List upcoming dates with a 🗑 button to delete each one (with confirmation), or add one: `/event <symbol or name> <DD/MM/YYYY> <text>` |
| `/plan_alert` | Your trading plans with status and the buttons **➕ Add plan** and **🗑 Delete plan**. Add plan guides you step by step: asset (with its current price), support, how many % above it you still buy (0/2/3/5 % or a number), target, stop, note – each price can be skipped – then a summary to save. Delete plan asks which plan, then for confirmation. In one line still possible: `/plan_alert <symbol or name> <support> <max_entry %> <target> <stop> <note>` (`-` for a value you do not use). A new plan replaces the old one of the same asset. The old `/price_alert` still works |
| **Portfolio** | |
| `/portfolio` | Daily P&L table and 13-week P&L chart now (otherwise with the evening summary) |
| `/update` | Look for a new CaSa release on GitHub now (otherwise daily at 19:00); with the CaSa image the answer has the button **⬆ Update now** |
| `/record` | Record a transaction: buttons **🟢 Buy**, **🔴 Sell** and **Sell all**. Buy and sell go step by step: choose the asset with a button (sell: only assets you hold), then amount, quantity and date (today, yesterday or other), confirm; afterwards ↩ Undo. Sell all records the full position at the current price (confirmation required). In one line: `/record buy <symbol or name> <amount €> <qty> <DD/MM/YYYY>`, `/record sell <symbol or name> <amount €> <qty>` (today), `/record sellall <symbol or name> [amount €]`. The old `/buy`, `/sell` and `/sellall` still work |
| `/history` | Transactions of one asset as a table (newest first); asset via button or `/history <symbol or name>` |
| `/transactions` | All transactions as a table like the daily summary (newest first, 40 per image) |
| **System** | |
| `/status` | Status of the main program: last analyses, next scheduled runs, alarms since the start, memory, disk space, problems (during a running analysis it follows afterwards) |
| `/restart` | Restart the main program (reloads all files; after a program update stop and start CaSa completely) |
| `/help` | Overview of all commands |

Only messages from the chat id in `config.ini` are accepted.

## Daily Schedule

| Time | What happens |
|---|---|
| every hour | 100-hour analysis of all crypto assets with `1` |
| 08:00 | Reminder for tomorrow's dates from `events.ini` and tomorrow's quarterly results (stocks with `1`), each plus daily analysis of the asset |
| every hour | Plan alerts from `planalert.ini`: buy zone and target (stocks Monday to Friday, 08:00–22:00), stop with the daily closes |
| 09:30 and 12:30 | 100-day analysis of all assets with `1` (stocks Monday to Friday, 08:00–22:00) |
| 18:30 | Daily ranking (crypto and stocks) |

If the program is started after 12:30, the daily analysis runs immediately; missed date reminders are sent at the start. With `test_mode = 1` there is no schedule – all analyses run once at the start (date reminders still work).

## Understanding Alerts and Charts

CaSa sends two kinds of messages, and you choose for every signal – separately for stocks and crypto – whether it comes as an alarm, as an info or not at all: in `config.ini` `[alerts]` (required, created automatically) or in Telegram with `/smart_alert_config`.

- **🔔 Alarm**: header "Potential buy" / "Potential sell", can be marked as important, recorded in the alarm journal (monthly report)
- **ℹ️ Info**: header "Info: …", never marked as important, not part of the alarm journal

| Signal | Switch | Default stocks; crypto | Meaning |
|---|---|---|---|
| Positive-5 | `positive5` | alarm; alarm | 5-period rate of change rises above `seven_day_price_change` (first value above it); only above the SMA200 |
| Score>8 | `score8` | info; alarm | Internal score (-18 to +18) rises above 8 for the first time (daily only); only above the SMA200 |
| Oversold | `oversold` | alarm; off | The internal score falls to -8 or lower for the first time (strong sell-off, possible rebound; daily only, no SMA200 filter, never marked as important) |
| Trend turns down | `cross_ema_sell` | alarm; alarm | Fast EMA (12) crosses the slow EMA (26) downwards (daily only, only for assets you own) |
| Loss limit | `loss_limit` | alarm; alarm | The total P&L of a position you own (value vs. FIFO buy price) falls below `total_loss_limit` in `[alertsconfig]` (e.g. -5 %); on the day it crosses the limit, again after a recovery above it (daily only). A risk alarm for your own limit, not a forecast |
| Trend turns up | `cross_ema_buy` | info; info | Fast EMA (12) crosses the slow EMA (26) upwards (daily only) |
| Sharp drop | `negative5` | info; info | Price falls more than `seven_day_price_change` in 5 periods (first value below it) |
| New 100-day high | `high100` | info; alarm | Highest closing price of the last 100 days (daily only) |
| New 100-day low | `low100` | info; info | Lowest closing price of the last 100 days (daily only) |
| Pattern breakout | `pattern` | info; info | Clear close through the neckline of a chart pattern (Double/Triple Bottom/Top, (Inverse) Head and Shoulders): at least half an average daily range (0.5 ATR) beyond it; weaker breakouts are not reported |

The defaults follow the backtest: an alarm where the signal showed an advantage for that asset type (for example Score>8 and the 100-day high only for crypto, Oversold only for stocks), otherwise an info.

Buy alarms and infos are sent for all assets with `1` in `assets.ini`, also for watched assets you do not own yet (marked in the message). Sell alarms are only sent for assets you own. Alarm messages with a score above +5 (buy) or below -5 (sell) are marked as important.

**History of the alarms:** versions up to 0.1.0 had six alarms for your own positions: Cross-EMA Buy, Positive-5, 100-Minimum, Cross-EMA Sell, Negative-5 and 100-Maximum. In 0.1.5 Cross-EMA Buy, Negative-5 and 100-Minimum/-Maximum were removed (no advantage in the first backtest), Score>8 and the pattern breakout were added. In 0.1.11 every signal can be an alarm or an info (separately for stocks and crypto); Cross-EMA Buy and Negative-5 came back (default: info), 100-Minimum/-Maximum (turn of the fast EMA after its 100-day low/high) were replaced by the clearer new 100-day high/low of the price, the pattern breakout became an info and Oversold was added.

The selection is based on a backtest with 9 assets (Bitcoin, Ethereum, DAX and six stocks, 7–14 years of daily data): Cross-EMA Buy, Negative-5 and 100-Minimum/-Maximum showed no advantage over an average day and were removed. Positive-5 only works with a high threshold (about 10 %); the score was the strongest signal. Both buy alarms only worked in an uptrend (price above the 200-day average), so they are filtered by it. SMA200 breakout and golden cross were also tested as separate alarms, but added little beyond the existing ones.

A second backtest (version 0.1.11) checked the alarms on 54 other assets (47 stocks, 7 cryptocurrencies, up to 14 years, about 555 asset-years) – assets that were not used to choose the rules. "Excess" is the price change 20 days after the alarm compared with an average day of the same asset; hits = share of buy alarms followed by a rise (sell: a fall) after 20 days, compared with an average day.

| Alarm | Hits | Average day | Excess 20 days | Assets in the right direction | Note |
|---|---|---|---|---|---|
| Positive-5 (buy) | 54 % | 52 % | +2.7 % | 34 of 54 | mainly crypto (+5.3 %); stocks +1.5 % |
| Score>8 (buy) | 55 % | 52 % | +1.2 % | 24 of 54 | crypto +7.6 % (7 of 7), stocks 0.0 % |
| Oversold (buy, stocks) | 58 % | 52 % | +1.3 % | 38 of 47 | positive in 2013–19, 2020–22 and 2023–26 |
| Cross-EMA (sell) | 46 % | 43 % | −1.5 % | 33 of 54 | most stable signal since 2020 |
| Pattern-Breakout (info, since 0.1.11 only clear breakouts) | stocks 51 % / crypto 56 % | 51 % / 50 % | stocks ±0 %, crypto +4.7 % | 22 of 47 / 6 of 7 | only clear breakouts count (≥ 0.5 ATR beyond the neckline); before: +0.3 % / −0.2 %, no advantage |

The infos in the same backtest (excess after 20 days; stocks / crypto): Trend turns up ±0.0 % / +2.2 %, Sharp drop +1.2 % / −1.8 % (afterwards clearly more volatile), New 100-day high −0.1 % / +9.1 % (7 of 7 cryptos), New 100-day low +0.8 % / −1.0 %. They describe what happened; only the 100-day high for crypto points clearly in one direction.

The averages come from a few large moves: for most buy alarms the median is slightly negative, so a single alarm is a hint, not a forecast. Following all buy alarms (hold 60 days) earned less than buying and holding in 36 of 54 assets. Popular TradingView indicators (SuperTrend, Squeeze Momentum, WaveTrend, RSI divergence) showed no advantage on stocks either.

**Message content**

Below every alarm and info a **🤖 AI prompt** button creates the `/ai_prompt` prompt for this asset; with a Claude API key there are also **🤖 AI decision (API)** and **🤖 AI plan (API)** (as `/ai_api_decision3s` and `/ai_api_plan`). The figure blocks are **collapsed** (tap the quote to open them); score and traffic lights stay visible. Wrong or incomplete input (e.g. an unknown name in `/check`, `/history` without an asset) is answered with buttons to choose from instead of only the format.

![Alarm message in Telegram: score 1–10 with traffic-light lines and figure blocks](docs/telegram_message.png)

- **Score 1–10**: today's rating compared with the last 100 days of the same asset (10 = among the strongest 10 %), followed by one traffic-light line per analysis (🟢 good, 🔴 bad, 🟡 neutral or not yet clear) in the same order as the blocks below: **Long-term-Analysis** (price vs. 200-day average), **Risk/Reward-Analysis** (ℹ️ information only: room to resistance R1 and distance to support S1 for stop and target; in the backtest a "good" ratio did not lead to better results), **RSI-Analysis** (real RSI value: below 30 oversold, above 70 overbought), **Trend-Analysis**, **Short-term-Analysis** and **Pattern-Analysis**. Each block below shows the figures behind its line. Internally CaSa still calculates a score from -18 to +18 (used for the Score>8 alarm and the AI prompt).
- **Pattern-Analysis**: in the confirmation only a short status (buy signal, sell signal, not a signal yet, none detected); below it as its own block the detected chart pattern in one sentence (forming or breakout, price that triggers the signal, target, invalidation level); drawn in pink in the chart
- **RSI14**, **SMA7**, **VMA7**, **ROC5**, **VROC5** in the blocks: the real value first (RSI, 7-day average in €, average volume, 5-day change in %), in brackets how it compares with the last 100 values (e.g. `(>88% of days)` = higher than on 88 % of the last 100 days). Values in the top or bottom 30 % raise or lower the score; the traffic light at the top shows the result.
- **Amount >1 year**: holdings older than one year

*Fake-out*: the price moves, but volume does not support the move.

**Chart**
- Black line: price; dotted lines: fast/slow EMA; grey band: Bollinger Bands
- Triangles: EMA crossovers; dots: 100-day high/low of the fast EMA (hidden when a chart pattern is drawn)
- **R1 / S1**: nearest resistance above and support below the price, as a short band at the left edge with a tag outside the chart (stronger colour = at least 2 touches; ▲/▼ = zone outside the visible range); details in the legend at the top right
- **Chart pattern** (pink): pattern line with labels, neckline, star at the breakout and target
- Black tag on the left: current price and its deviation in %
- Grey bars on the right: volume profile of the visible range (darker bar = price level with most volume)
- Left axis: deviation in % from your average buy price (or from the average price if you don't own the asset)
- Background green/red: price above/below your average buy price

## Configuration

All files are plain text and can be edited with any editor. After changes, send `/restart`.

### config.ini

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

### assets.ini

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

### events.ini (optional)

```ini
# Format: name = symbol; DD/MM/YYYY; description
[events]
nvda_20261117  = NVD.DE; 17/11/2026; Q3 earnings after US market close
```

Entries can also be added with `/event` and deleted via the 🗑 buttons in the `/event` list. The reminder comes at 08:00 one day before. Past dates are ignored.

### planalert.ini (optional)

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

### transactions.csv

Written by `/record` (buy, sell, sell all):

```
type;symbol;name;action;amount;qty;date
crypto;bitcoin;Bitcoin;buy;500.0;0.01;01/01/2024
stock;QCOM;Qualcomm;sell;312.16;2.0;14/11/2026
```

### AI prompt (config.ini `[ai_prompt]`)

Template for the `/ai_prompt` prompt (also the data for `/ai_api_decision3s` and `/ai_api_plan`, which add their own answer format at the end). It is the last section of `config.ini`, added automatically at the start, and can be edited freely:

```ini
[ai_prompt]
template =
  Assess {name} ({symbol}, {type}) for a three-month horizon ...

  A. My program's data (daily values, {date})
  ...
```

Every line of the text starts with two spaces; empty lines are allowed. Lines starting with `#` or `;` are treated as comments. Up to 0.1.12 the template was the file `ai_prompt.txt`: at the start CaSa moves its text into `config.ini` (your own changes stay) and keeps the old file as `ai_prompt.old.txt`. Placeholders in curly brackets such as `{name}`, `{price}`, `{support}`, `{bottom}` or `{market}` are filled in by the program. Since 0.1.11 also `{signals}` (alarms and infos the analysis would send today for this asset) and `{evidence}` (the backtest findings for stocks or crypto, e.g. that stocks recovered slightly after strong sell-offs while crypto tended to keep falling). The leading market is chosen by the company's home country (Nvidia on Xetra: S&P 500), not by the exchange. An unchanged template from an earlier version is replaced by the current one automatically; your own changes are never overwritten.

### alarm_journal.csv (created automatically)

Every sent alarm is recorded with date, asset, alarm, price and score. After 20 trading days (crypto: 20 days) CaSa fills in the price, the change in % and whether the price moved in the alarm direction (buy: up, sell: down). This shows over time whether the alarms also work on new data and not only in the backtest. On the 1st of each month the statistics come with the evening summary. Alarms are not recorded in test mode (`test_mode = 1`), because all analyses run at every start there. The file is private and not part of the repository.

### logfile.txt

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

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Main program: loads the files, schedules analyses, handles commands from the bot |
| `classes.py` | `CryptoStock` class (data, transactions, analysis per asset), reading and writing `assets.ini`, `events.ini`, `transactions.csv` |
| `func.py` | Data retrieval, indicators, support/resistance, bottom/top formation, market phase, alarms, charts, summary image |
| `telegram_bot.py` | Telegram bot, runs as a separate process: sends messages, forwards commands to `main.py` |
| `install_venv.bat` | Installation on Windows |
| `image/` | Building the CaSa image for the Raspberry Pi (pi-gen) |
| `requirements.txt` | Python packages |

## Known Limitations

- The free CoinGecko API has rate limits and no daily candles (open/high/low); crypto analysis uses closing prices only.
- The hourly analysis is only scheduled for crypto; stocks are analysed daily.
- `/restart` reloads the files (config, assets, dates, transactions), but not the program code. After an update (new version), stop CaSa and start it again.
- The program remembers sent alarms and date reminders only while it is running; after a full restart (not `/restart`), an alarm or reminder of the same day can be sent again.
- Quarterly results dates come from Yahoo. Right after a report Yahoo often does not know the next date yet; it is picked up automatically as soon as it is announced. Some stocks (e.g. BAE Systems) have no date on Yahoo.
- While an analysis is running, Telegram commands are processed afterwards.

## Roadmap

### Usability
- [ ] **One-line summary** at the top of every message, e.g. "🟢 Uptrend, strong buying interest – buy signal". The details follow below for anyone who wants them.
- [x] **Easier score**: The −18…+18 score was hard to read. Now a score 1–10 with the assessments as traffic-light confirmation (🟢 🔴 🟡).
- [ ] **Simpler score blocks**: Indicator lines such as RSI14 or SMA7 explained in plain language for non-experts.
- [ ] **`/explain` command**: Explains terms like RSI, SMA200 or neckline in two sentences.

### Setup
- [ ] **Docker image**: Simple deployment. Private files (config, portfolio) move to a separate data folder so they stay outside the image.
- [ ] **Setup wizard in Telegram**: New users add assets and settings via chat instead of editing INI files.
- [ ] **Disclaimer** in README and start message: CaSa provides technical signals only, no investment advice.

### Backtesting
- [ ] **Backtest script in the repo**: The backtest currently lives in a temporary folder. In the repo, every rule change can be checked against historical data.
- [ ] **Hit rate in alarm messages**, e.g. "This alarm was right in 61 % of cases (backtest)". Helps users judge how reliable a signal is.
- [ ] **Sideways market detection (ADX)**: Today only up- and downtrends are detected (SMA50/SMA200). ADX adds a sideways phase, in which many trend signals fail.

### Indicators & charts
- [ ] **ATR-based stop-loss suggestion**, e.g. "Stop around 142 € (2× ATR)". Adapts the stop to how much the asset usually moves.
- [ ] **Relative strength vs. index**: Strong stocks in strong markets, e.g. "Nvidia +12 % vs. Nasdaq in 3 months". One of the most robust rules in technical analysis.
- [ ] **R2/S2 zones**: A second resistance and support zone, plus an alarm when the price breaks through a zone.
- [ ] **More chart patterns**: Triangles, wedges and flags in addition to the existing patterns.
- [ ] **Fair Value Gaps**: Price gaps between candles that the price may return to. Popular but unproven, so they are tested in the backtest before any alarm is added.

### Alerts & events
- [x] **Own price alarms**, e.g. `/price_alert NVD.DE below 130`. User-defined price levels, independent of the indicators.
- [x] **Simple trading plans** (`planalert.ini`, `/plan_alert`): buy zone, target and stop per asset.
- [ ] **Plan conditions**: higher low, close above the first rebound high, above EMA12/SMA200; the AI prompt writes a ready plan line.
- [ ] **Options expiry and market holidays**: The major expiry date (third Friday of March, June, September and December) is calculated automatically. Exchange holidays come from a fixed list.
- [ ] **Crypto events**: Known dates such as Bitcoin halvings in `events.ini`.
- [ ] **Automatic import of economic dates**: Fed, CPI and ECB dates without manual entry. Needs a calendar source.
- [x] **Delete dates directly in Telegram**: Remove entries from `events.ini` via chat.

### Data & sentiment
- [ ] **Fear & Greed index** as a sentiment line in the daily summary. Free for crypto (alternative.me), for stocks only via an unofficial CNN source.
- [ ] **Google Trends**: Search interest as a sentiment indicator, especially for crypto. A `google_trends` function already exists but is not used yet; `pytrends` is no longer installed by default (`python -m pip install pytrends`).
- [ ] **Alarm for unusual news volume**: Triggers when an asset gets far more news than usual (source: yfinance `Ticker.news`). It shows that something is happening, not in which direction.
- [ ] **Insider trades**: Purchases and sales by executives from free official sources: SEC EDGAR (Form 4) for US stocks, Directors' Dealings for German stocks. Large insider purchases are a documented signal.
- [ ] **Regulatory news via RSS**: Press releases from SEC and BaFin, flagged when an asset from the watchlist is mentioned.

### Portfolio
- [ ] **Read-only exchange API for crypto (ccxt)**: Loads holdings directly from the major exchanges. Read-only keys only, never with trading rights.
- [x] **Total value and daily P&L** in the evening summary.
- [ ] **Position size suggestion**: Based on ATR, e.g. max. 2 % of the portfolio at risk per position.
- [ ] **Tax-free reminder**: A message on the day a crypto position has been held for more than one year (tax-free under German law).

### Reliability
- [ ] **Keep known alarms after a full restart**: They are currently lost, so the same alarm can be sent twice.
- [x] **Health check**: A status line in the log every 6 hours, with a warning when an analysis is overdue or failing.

### Not planned
- **Opening/closing volatility**: Needs minute data, while CaSa works with daily data.
- **X/Twitter tracking**: The API is expensive and the posts are hard to evaluate.

## Versioning

CaSa uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`). The version is set in `main.py` (`VERSION`) and shown in the start message; each release is tagged in Git (e.g. `v0.1.10`). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the MIT License.

## Disclaimer

This application generates alerts based on technical indicators and market data analysis. All alerts are purely informational and require independent user evaluation. Users are solely responsible for their own investment decisions. CaSa does not recommend buying or selling any assets, nor does it provide personalized investment advice. Answers of AI tools to the `/ai_prompt` prompt, `/ai_api_decision3s` and `/ai_api_plan` are not investment advice either.

## Links

- [Telegram BotFather](https://t.me/BotFather) — create a Telegram bot
- [CoinGecko](https://www.coingecko.com) — cryptocurrency market data
- [Yahoo Finance](https://finance.yahoo.com) — stock market data
- [Rock Pi](https://rockpi.org/) — single-board computer for hosting
- [Armbian OS](https://www.armbian.com/) — lightweight ARM-based OS
