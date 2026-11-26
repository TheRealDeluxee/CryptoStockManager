# Architecture

Developer notes. File overview for users: [project-structure.md](../project-structure.md).

## Processes
- `main.py` `run_program()` starts the main program and `telegram_bot.run_standalone` as a separate process
  (`multiprocessing`). Two queues: `notify_q` (messages and images to send) and `command_q` (commands from
  Telegram, e.g. `{'cmd': 'reload_transactions'}`), handled in `CryptoStockManager.handle_command`.
- Sending: `func.pushover` / `func.pushover_image` put messages into the queue (`func.set_message_queue`).
- Scheduling with `schedule` in `run_program`: hourly crypto analysis, daily analyses at the times of
  `config.ini [schedule]`, evening summary, date reminders, hourly plan check, 6-hourly health check,
  update check.

## Main program (`main.py`, class `CryptoStockManager`)
- Start (`__init__`): bring files up to date (`cl.update_config`, `update_ai_prompt`,
  `cl.update_transactions_header`, `cl.update_plan_header`, `cl.update_plan_keys`, `cl.convert_value_alerts`),
  then `load_config_items`.
- Analyses: `hundred_day_analysis`, `hundred_hour_analysis` → `_analyse_items`; one asset:
  `single_daily_analysis` / `_daily_analysis`.
- AI: `_ai_question` fills `ai_prompt.txt` (placeholders `{...}`); `ai_prompt` (copy text), `ai_api`
  (Claude API with `AI_API_TASKS['decision'|'plan']`), `save_ai_plan`.
- Plans: `plan_alert_check`, `_plan_turn`, `_plan_stop`, `_plan_message`.
- Evening: `send_summary`, `send_pnl`, `send_pnl_chart`, monthly `send_journal_report`.
- Updates: `check_for_update`, `update_casa` (git checkout of the newest tag on the Pi).

## Data and analysis (`func.py`)
- Prices: `get_stock` (Yahoo, converted to EUR with `_fx_rates` per local day), `get_crypto` (CoinGecko).
- Indicators and score: `score_signals`, `calc_stat_limits`, `analysis_details` (score 1–10, traffic
  lights, figure list), `support_resistance`, `chart_pattern`, `trend_phase`, `market_index`/`market_text`,
  `weekly_averages_text` (50/200 weeks for the AI).
- Alarms: `ALERTS` (switches, backtest values), `alarm()`, `backtest_note`.
- Images: charts, `create_pnl_image`, `create_pnl_chart`, `create_transactions_image`, table image.
- Claude API: `claude()`, `claude_usage_text`.

## Files and portfolio (`classes.py`)
- `CryptoStock` (data, FIFO buys/sells, state), `apply_transactions`.
- Readers/writers with checks: `read_assets`, `read_events`, `read_plans`/`add_plan`, `read_transactions`,
  `read_schedule`, `read_alert_settings`, alarm journal (`journal_*`).
- Portfolio maths: `realized_trades`, `open_lots`, `pnl_history`, `weekly_pnl`, `transaction_fee`.
- Migrations at the start: `update_*`, `ensure_alert_section`, `take_ai_prompt_from_config`.

## Telegram (`telegram_bot.py`)
- Commands registered at the end of the file (`app.add_handler(CommandHandler(...))`); old names stay as aliases.
- Guided inputs keep their state in `context.user_data` (`trade`, `plan`); typed answers go to `_on_text`.
- Button callback data: prefixes like `tr|`, `rec|`, `aibtn|`, `aidbtn|`, `aipbtn|`.

## User files (all private, in the program folder)
`config.ini`, `assets.ini`, `events.ini`, `planalert.ini`, `transactions.csv`
(`type;symbol;name;action;qty;price;net amount;date;wallet`), `alarm_journal.csv`, `ai_prompt.txt`,
`logfile.txt`. Formats: [configuration.md](../configuration.md).

## Tests
`tests/` (local only): `conftest.py` with the `sandbox` fixture (own folder, sample `assets.ini` and
`transactions.csv`, messages collected in `sandbox.sent`). Network and API are replaced by monkeypatch.
