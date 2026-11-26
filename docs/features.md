# Features

[← back to the README](../readme.md)

**Analysis**
- Crypto data from CoinGecko, stock data from Yahoo Finance; stocks can be watched at their home exchange (e.g. NVDA, TT, ASML.AS) with real volume and are converted to EUR with the exchange rate of the same day (USD, GBp, JPY, HKD, CAD, NOK, SEK, ...)
- 250 daily / 450 hourly values are loaded, indicators are calculated on the full history and the last 100 values are evaluated
- Scoring system from RSI14, SMA7/VMA7, ROC5/VROC5 with volume support and fake-out detection, shown as an easy score 1–10 with reasons
- EMA 12/26 crossovers, Bollinger Bands (20,2), 7-day trend slope
- Support/resistance zones from pivot points of the full history (stocks: daily high/low, like on TradingView)
- Chart pattern detection: Double/Triple Top/Bottom and (Inverse) Head and Shoulders with neckline, target and invalidation level
- Market comparison for stocks: the leading index of the home country and the US sector ETF today, and whether the stock moved in line with, weaker or stronger than the market (figure vsIndex)

**Alarms**
- Automatic analysis of all assets marked with `1` in `assets.ini`: crypto every hour, stocks and crypto daily at the times in `config.ini` `[schedule]` (default 09:30, 17:45, 22:30)
- Every signal can be an **alarm**, an **info** or **off**, separately for stocks and crypto (`config.ini` `[alerts]` or `/smart_alert_config`): Positive-5, Score>8, Oversold, Cross-EMA Sell, Loss limit, Cross-EMA Buy, Negative-5, 100-day high, 100-day low, pattern breakout
- Each alarm and info ends with its backtest result, e.g. "Backtest: price rose in 57 % of cases (after 20 days)"; Oversold, 100-day low and 100-day high report at the turn of the 12-day average (since 0.1.14, not backtested yet – the message says so)
- Trend filter: Positive-5 and Score>8 only above the 200-day average
- Buy alarms also for watched assets you do not own yet; sell alarms only for your portfolio
- Duplicate filter: the same alarm is not sent twice in a row
- Alarm journal: every alarm is checked 20 days later; on the 1st of each month a report shows how the alarms did compared with the backtest

**Charts and Messages**
- Chart without weekend gaps: price with price tag, EMAs, Bollinger Bands, signals, support/resistance (R1/S1), chart pattern, volume bars and volume profile; values in EUR and in the currency of the home exchange (e.g. "352.14 € (386.29 $ · -11.4 %)")
- Compact Telegram messages for the phone: header in two lines, score 1–10 with traffic-light lines, a short backtest line, a collapsed numbered figure list with bars and the market line
- Buttons below every alarm and info: 🤖 Prompt, and with a Claude API key 🤖 Guide (API) and 🤖 Plan (API)
- Daily ranking of all assets by 1-day P&L
- Daily P&L of your own positions (crypto and stocks together) after the summary: value, 1-day profit/loss in € and %, total profit/loss, total row
- P&L chart of the last 13 weeks as weekly bars: the realised profit/loss of your sales in each week (from `transactions.csv`, FIFO) in full colour, one part per sale (white line between them; gains upwards, losses downwards); for the current week also the profit or loss of your open positions if you sold them today (value minus what you paid, since the buy) in light colour, one part per open asset (gains upwards, losses downwards). Values only for the realised parts. At the top the realised total of the 13 weeks, below it in brackets the gains and losses of the open positions if sold today and their sum, e.g. (+400.00 | −700.00) € open positions if sold today = −300.00 €

**Portfolio**
- Buys and sells with `/record` or the 🟢 Record buy / 🔴 Record sell buttons, saved in `transactions.csv` (FIFO bookkeeping, amounts in EUR)
- Profit/loss per asset, background colour of the chart shows profit or loss
- Holding period: amount held longer than one year (tax-relevant for crypto in Germany)

**Planning and Research**
- `/ai_prompt` creates a detailed prompt for ChatGPT or Claude: technical data, support/resistance with risk/reward, chart pattern, bottom/top formation, score 1–10, market phase of the asset and its leading market, the market move today (index and sector), the long-term trend (50- and 200-week averages), your position, your current trading plan from `planalert.ini` with a plan check (e.g. sell-zone below your average cost), the largest one-day move of the last 10 days, the day change in EUR and in the currency of the exchange, and common investment rules – no AI API needed, just copy and paste
- With a Claude API key (billed by Anthropic; every answer shows its tokens, web searches and cost):
  - **🤖 Guide (API)** (`/ai_api_guide`): the decision in three sentences – the long-term trend from the 50- and 200-week averages, the main reason from current news, the business, its customer industries and similar earlier events (web search, up to 12 searches), and what would change the decision. Buttons below the answer record a buy (and a sale when you hold the asset)
  - **🤖 Plan (API)** (`/ai_api_plan`): a trading plan with buy-zone, sell-zone, stop-plan and a note ("Size | Buy when | Not if"), saved as plan alert with one tap
  - Guide and Plan (API) send the data without the answer format of the copy prompt (shorter, cheaper)
  - Every AI analysis (Prompt, Guide, Plan) gets your current trading plan. The AI keeps it unless something changed; every change to buy-zone, sell-zone, stop-plan or the position building is given as old value → new value with its concrete trigger (e.g. a daily close below a support or a news item with date)
- Important dates (earnings, central bank meetings, ...) in `events.ini` or via `/event`: reminder one day before, plus daily analysis
- Trading plans in `planalert.ini` or via `/plan_alert`: buy-zone, sell-zone, stop-plan and a note per asset, checked every hour; messages "in your buy-zone", "🟢 Turn up" (a daily close above the buy-zone again with a rising 12-day average), sell-zone reached and stop-plan (a daily close below ends the plan); they depend on whether you hold the asset, plus daily analysis
- Quarterly results: automatic reminder one day before for all stocks in the alarm analysis (date from Yahoo), plus daily analysis
