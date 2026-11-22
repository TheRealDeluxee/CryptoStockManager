![CaSa](logo.png)

# CaSa

**C**rypto **a**nd **S**tock **A**lert

CaSa watches your cryptocurrencies and stocks, analyses them with technical indicators and sends alerts, charts and daily summaries to **Telegram**. The same Telegram chat is used to control the program: run analyses on demand, record buys and sells, switch alarms on and off, plan important dates and create prompts for ChatGPT or Claude.

## Contents

- [Why CaSa?](#why-casa)
- [What does CaSa do?](#what-does-casa-do)
- [What do you need?](#what-do-you-need)
- [What's New?](#whats-new)
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

Casa means “house” or “home” in Spanish. I’m saving to buy a place of my own, and I know many people my age and younger who are trying to do the same. With home prices so high, I started asking myself what I could do today to get a little closer to that goal. That question led me to investing.

As I started learning about stocks and cryptocurrencies, I kept wondering what might happen while I was away from the screen. Crypto prices could change quickly, and I worried about how much I could lose. I wanted a way to follow the markets without checking prices all day. The apps I tried could tell me when a price reached a certain level, but I wanted more control over when I got an alert. That usually meant paying for a subscription. So I built the tool I wanted to use and called it CaSa, after the goal that got me started.

CaSa is free and open source on GitHub. It uses very little computing power, so it can run on a low-cost computer without a cloud server or paid API subscriptions. It explains what you’re seeing in simple words, even if you’re new to investing. I started CaSa with a home of my own in mind. But now I hope it helps others take their first steps in investing, whatever they’re saving for. If you try it, I’d love to hear what would make it more useful to you. You can see what I’m planning next in the [roadmap](docs/roadmap.md).

![CaSa alert in Telegram on iPhone and Apple Watch](docs/telegram_phone_watch.png)

## What does CaSa do?

- **Watches your stocks and cryptos** day and night – you don't have to look at prices all day.
- **Messages you in Telegram** when something important happens, with a chart and a simple score from 1 to 10.
- **Explains in plain words** what is going on – no expert knowledge needed.
- **Knows your portfolio**: what you bought, what you sold, and your profit or loss every evening.
- **Follows your own plan**: set a buy price, a target and a stop for an asset – CaSa tells you when it is reached.
- **Optional AI**: one tap creates a prompt for ChatGPT or Claude – or, with a Claude API key, asks Claude directly for a second opinion.

CaSa gives signals, not investment advice. You decide.

## What do you need?

- A **Windows or Linux computer**, or for day and night a **Raspberry Pi** 4 or 5 (a small, cheap computer), and internet
- **Telegram** on your phone

<p><br>
  <a href="docs/installation.md"><img src="https://img.shields.io/badge/%E2%AC%87%20Install%20now-Windows%20%C2%B7%20Linux-1e8449?style=for-the-badge" alt="Install now – Windows · Linux" height="40"></a>
</p>
Try it out: installed in 2–5 minutes, once Telegram is installed and your bot is created (with @BotFather).

## What's New?

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

- The free CoinGecko API has rate limits and no daily candles (open/high/low); crypto analysis uses closing prices only.
- The hourly analysis is only scheduled for crypto; stocks are analysed daily.
- `/restart` reloads the files (config, assets, dates, transactions), but not the program code. A new version: with the CaSa image `/update` installs it and restarts CaSa by itself; on Windows start `install_casa.bat` again, then CaSa with `start_casa.bat`.
- The program remembers sent alarms and date reminders only while it is running; after a full restart (not `/restart`), an alarm or reminder of the same day can be sent again.
- Quarterly results dates come from Yahoo. Right after a report Yahoo often does not know the next date yet; it is picked up automatically as soon as it is announced. Some stocks (e.g. BAE Systems) have no date on Yahoo.
- While an analysis is running, Telegram commands are processed afterwards.

## Versioning

CaSa uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`). The version is set in `main.py` (`VERSION`) and shown in the start message; each release is tagged in Git (e.g. `v0.1.10`). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE).

## Disclaimer

This application generates alerts based on technical indicators and market data analysis. All alerts are purely informational and require independent user evaluation. Users are solely responsible for their own investment decisions. CaSa does not recommend buying or selling any assets, nor does it provide personalized investment advice. Answers of AI tools to the `/ai_prompt` prompt, `/ai_api_decision3s` and `/ai_api_plan` are not investment advice either.

## Links

- [Telegram BotFather](https://t.me/BotFather) — create a Telegram bot
- [CoinGecko](https://www.coingecko.com) — cryptocurrency market data
- [Yahoo Finance](https://finance.yahoo.com) — stock market data
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/) — writes the CaSa image to the microSD card (Option B)
