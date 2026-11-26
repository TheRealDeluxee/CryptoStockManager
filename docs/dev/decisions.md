# Decisions

Why things are the way they are. Newest first; add an entry when a choice is not obvious from the code.

- **AI prompt: only values CaSa can compute** (after 0.1.14): plan check, day change in EUR and exchange
  currency, largest one-day move, risk/reward to the stop-plan are computed; share of portfolio, max. risk
  per position and a 60-day backtest were left out because CaSa cannot fill them reliably (an empty or wrong
  value misleads the AI). Horizon stays three months (about 60 trading days) – the backtest measures 20 days
  and the plans work on daily closes; the 100 days are the look-back of the figures, not the horizon.
- **Current trading plan in every AI analysis** (after 0.1.14): `{current_plan}` in part A; a change to
  buy-zone, sell-zone, stop-plan or position building only as "old -> new, because <trigger>". Own templates
  without the placeholder get it appended. Reason: plans changed without a visible reason.
- **Trading result: open positions "if sold today"** (after 0.1.14): value minus FIFO cost of the remaining
  buys instead of the change of the week. Reason: the user wants to see what a sale now would bring, like the
  "return since purchase" of the broker. The fee of the sale is not included.
- **`transactions.csv` like the broker statement** (after 0.1.14): `qty;price;net amount;date;wallet`.
  `net amount` = "Gegenwert" (buy incl. fee, sale after fee), the fee follows from price and net amount.
  Profit/loss keeps using the net amount, so fees are never counted twice. Column 5 is `qty`, not
  "gross amount" (that would be qty × price). Old files are converted at the start, copy in `.bak`.
- **`/ai_api_guide` instead of `/ai_api_decision3s`** (after 0.1.14); the old name stays as alias.
- **50/200-week averages computed by CaSa** (after 0.1.14): one Yahoo request (5 years weekly) only when an
  AI button is used, cached a day. Reason: the AI ran out of web searches for the long-term trend. Cryptos
  only Bitcoin, Ethereum, Solana, XRP, Chainlink, NEAR (CoinGecko free gives about one year).
- **Market move in A, not searched by the AI** (0.1.14): index and US sector ETF from CaSa saves web
  searches and tokens.
- **Turn rule for Oversold / 100-day low / 100-day high** (0.1.14): report at the turn of the 12-day average,
  not at the extreme. Not backtested – the messages say so. Own backtest for stocks: waiting for a confirmed
  bottom did not help, so do not advertise the rule as a strength.
- **Backtest line in every alarm** (0.1.11/0.1.14): honesty about small edges; the numbers are in
  `func.ALERTS` (54 assets, 20 days, "normal" = an average day of the same assets).
- **No Plan (API) button below the Guide answer**: the user decided against it.
- **Telegram is the only interface**: no web UI, no cloud; CaSa runs on the user's own computer.
- **Releases are git tags**: `/update` on the Pi and `install_casa.bat` take the newest `v*` tag; a GitHub
  release page is not needed (only the Pi image is attached to the release `pi-image`).
