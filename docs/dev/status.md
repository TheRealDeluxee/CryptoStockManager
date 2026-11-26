# Status

Update this file when a task ends or the plan changes. Last update: 26/11/2026.

## Released
- Newest tag: `v0.1.14` (22/11/2026). The Raspberry Pi still ran 0.1.13 at that time.

## On `main`, not released yet (goes into the next CHANGELOG entry)
- 50- and 200-week averages for the AI ("Long-term trend" in part A).
- README: Why / How / What, new phone and watch picture, features page brought up to date.
- `/ai_api_decision3s` renamed to `/ai_api_guide` (old name still works).
- `transactions.csv` in the broker format (`qty;price;net amount;date;wallet`), `/record` asks for price
  and wallet, `/transactions` shows real price, wallet and fee.
- Trading result: open positions "if sold today".
- Current trading plan in every AI analysis with the rule "old -> new, because <trigger>".
- AI prompt: horizon "about 60 trading days" / lookback "last 100 trading days" (crypto 90 / 100 days),
  contradictions first, plan check line, day change in EUR and exchange currency with the rate, largest
  one-day move of 10 days, risk/reward to the stop-plan, plain meaning of the score, targeted search for the
  cause of the largest move and monthly sales, backtest marked "20 days – shorter than my horizon";
  Guide/Plan API without part D.
- Developer notes in `docs/dev/`; `CLAUDE.md` is local only (in `.gitignore`).

## Not committed yet
- Nothing.

## Open points / ideas
- Share of the portfolio in the prompt: only when `transactions.csv` is complete.
- Max. risk per position: would need a new setting in `config.ini`.
- Backtest after 60 days with number of cases: needs a new backtest.
- Personal data (holdings, missing transactions) is not noted here: this file is public on GitHub.
- Roadmap: [roadmap.md](../roadmap.md).
