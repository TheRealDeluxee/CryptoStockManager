![CaSa](logo.png)

# CaSa

**C**rypto **a**nd **S**tock **A**lert

CaSa watches your cryptocurrencies and stocks, analyses them with technical indicators and sends alerts, charts and daily summaries to **Telegram**. The same Telegram chat is used to control the program: run analyses on demand, record buys and sells, switch alarms on and off, plan important dates and, if you like, ask ChatGPT or Claude for a second opinion.

## Contents

- [Why CaSa?](#why-casa)
- [How does CaSa work?](#how-does-casa-work)
- [What does CaSa do?](#what-does-casa-do)
- [Latest Version](#latest-version)
- [Learn More](#learn-more)
  - [Features](docs/features.md)
  - [Installation](docs/installation.md)
  - [Telegram commands](docs/commands.md)
  - [Understanding alerts and charts](docs/alerts.md)
  - [Configuration](docs/configuration.md)
  - [Project structure](docs/project-structure.md)
  - [Roadmap](docs/roadmap.md)
- [Known Limitations](#known-limitations)
- [Versioning](#versioning)
- [License, Disclaimer and Links](#license)

## Why CaSa?

**Following your investments should leave room for living.**

I started investing while saving for a home of my own. Before long, I was checking prices throughout the day, wondering what I might miss while I was away. The apps I tried offered price alerts. But getting more control over when and why they arrived often meant paying for a subscription.

So I built CaSa. Its name means “home” in Spanish, after the goal that started it.

CaSa is free and open source. You can inspect the code behind its analyses and alerts, because trust matters when money is involved.

Whatever you are saving for, CaSa helps you stay informed while you get on with your day. Have an idea that would make it more useful? I'd love to hear it. Take a look at the [roadmap](docs/roadmap.md) to see what is next.

![CaSa alert in Telegram on iPhone and Apple Watch](docs/telegram_phone_watch.png)

## How does CaSa work?

**See the move. Understand the signal. Choose what deserves your attention.**

- **Understand why you were notified.** Each signal comes with a chart, a score and a short explanation of what triggered it. For stocks, index and sector comparisons put the move in context.
- **Choose what reaches you.** Pick the assets you follow. Set each signal to an alarm, an info (never marked as important) or off, with separate settings for stocks and crypto.
- **Run it on your own computer.** CaSa runs on a Windows or Linux PC or a Raspberry Pi and sends results to Telegram. It uses market data from Yahoo Finance and CoinGecko. No CaSa account or subscription is required for the core features.

Where available, messages include historical backtest results. Rules without a matching backtest are clearly marked. An alarm journal records alerts and provides monthly summaries of subsequent price changes. See [alerts and backtests](docs/alerts.md) for results and limitations.

## What does CaSa do?

- **Market alerts, explained.** Get a chart and the reason for the message when one of your chosen signals is triggered.
- **Your plan, within reach.** Set buy zones, sell zones, stop levels and position loss limits. CaSa alerts you when a condition is met. It never places trades.
- **Your portfolio at a glance.** Record purchases and sales manually in Telegram. Receive an evening summary of your holdings and profit or loss.
- **Dates worth remembering.** Get reminders for upcoming earnings where dates are available, and add your own events.
- **Optional AI research.** Create prompts for ChatGPT or Claude. With your own Claude API key, Claude combines CaSa’s data with web research on relevant news and business relationships to help you interpret the signals and suggest plans directly in Telegram. Save a suggested plan with one tap. Each API answer shows its usage cost.

Prices are displayed in EUR. Portfolio records are stored on your computer; commands and notifications pass through Telegram. Optional Claude API requests send the asset analysis and your position details to Anthropic. API usage is billed separately by Anthropic.

CaSa provides signals and context, not investment advice. You decide how to act.

**You need:**

- A **Windows or Linux computer**, or for day and night a **Raspberry Pi** 4 or 5 (a small, cheap computer), and internet
- **Telegram** on your phone
- Optional: a **Claude API key** for the AI answers in Telegram (billed by Anthropic)

<p><br>
  <a href="docs/installation.md"><img src="https://img.shields.io/badge/%E2%AC%87%20Install%20now-Windows%20%C2%B7%20Linux-1e8449?style=for-the-badge" alt="Install now – Windows · Linux" height="40"></a>
</p>
Try it out: installed in 2–5 minutes, once Telegram is installed and your bot is created (with @BotFather).

## Latest Version

**Version 0.1.14** – all changes in [CHANGELOG.md](CHANGELOG.md)

- Windows setup wizard `install_casa.bat`: a few questions, CaSa starts with `start_casa.bat`
- Stocks at their home exchange (e.g. NVDA, TT) with real volume, converted to EUR; the home currency in the chart
- Plans with buy-zone, sell-zone, stop-plan and the new message "Turn up" after the buy-zone
- Market line in every stock message: index and sector today, in line with or weaker/stronger than the market
- Guide (API) with the long-term trend, news and customer industries; record buttons below the answer
- Compact messages with a numbered figure list and bars; analysis times changeable in `config.ini`

Earlier versions: [CHANGELOG.md](CHANGELOG.md)

## Learn More

| Page | Content |
|---|---|
| [Features](docs/features.md) | Everything CaSa can do: analysis, alarms, charts and messages, portfolio, planning and research |
| [Installation](docs/installation.md) | Option A: manual installation (2–5 minutes on Windows), to try CaSa out or run it on a computer you already have · Option B: CaSa image for the Raspberry Pi (about 20 minutes; hardware, step by step, daily use) |
| [Telegram commands](docs/commands.md) | Every command with its options |
| [Understanding alerts and charts](docs/alerts.md) | When CaSa sends what, score 1–10, traffic lights, figure list, chart, alarms and infos with their backtest results, evening summary and trading result |
| [Configuration](docs/configuration.md) | `config.ini`, `assets.ini`, `events.ini`, `planalert.ini`, `transactions.csv`, `ai_prompt.txt`, `alarm_journal.csv`, `logfile.txt` |
| [Project structure](docs/project-structure.md) | The program files and folders and what each one does |
| [Roadmap](docs/roadmap.md) | What is planned next, what is done and what is not planned |

## Known Limitations

- The free CoinGecko API has rate limits and no daily candles (open/high/low); crypto analysis uses closing prices only. It also gives only about one year of history: the 50- and 200-week averages for the AI come from Yahoo, so far only for Bitcoin, Ethereum, Solana, XRP, Chainlink and NEAR (other cryptos: "not available").
- The hourly analysis is only scheduled for crypto; stocks are analysed daily.
- Oversold, 100-day low and 100-day high report at the turn since 0.1.14; this rule is not backtested yet and the messages say so.
- `/restart` reloads the files (config, assets, dates, transactions), but not the program code. A new version: with the CaSa image `/update` installs it and restarts CaSa by itself; on Windows start `install_casa.bat` again, then CaSa with `start_casa.bat`.
- The program remembers sent alarms and date reminders only while it is running; after a full restart (not `/restart`), an alarm or reminder of the same day can be sent again.
- Quarterly results dates come from Yahoo. Right after a report Yahoo often does not know the next date yet; it is picked up automatically as soon as it is announced. Some stocks (e.g. BAE Systems) have no date on Yahoo.
- While an analysis is running, Telegram commands are processed afterwards.

## Versioning

CaSa uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`). The version is set in `main.py` (`VERSION`) and shown in the start message; each release is tagged in Git (e.g. `v0.1.10`). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE).

## Disclaimer

This application generates alerts based on technical indicators and market data analysis. All alerts are purely informational and require independent user evaluation. Users are solely responsible for their own investment decisions. CaSa does not recommend buying or selling any assets, nor does it provide personalized investment advice. Answers of AI tools to the 🤖 Prompt (`/ai_prompt`), 🤖 Guide (API) (`/ai_api_guide`) and 🤖 Plan (API) (`/ai_api_plan`) are not investment advice either.

## Links

- [Telegram BotFather](https://t.me/BotFather) — create a Telegram bot
- [CoinGecko](https://www.coingecko.com) — cryptocurrency market data
- [Yahoo Finance](https://finance.yahoo.com) — stock market data
- [Claude Console](https://console.anthropic.com) — optional Claude API key for Guide (API) and Plan (API)
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/) — writes the CaSa image to the microSD card (Option B)
