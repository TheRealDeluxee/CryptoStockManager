"""CaSa: load the configuration, schedule analyses and handle Telegram commands.

Start: python main.py
The actual calculations stay in classes.py and func.py.
"""
import configparser
import copy
import datetime
import hashlib
import html
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import socket
import time
import traceback
import uuid
from queue import Empty

import pandas as pd
import schedule
import classes as cl
import func
import telegram_bot

# Program version (Semantic Versioning: MAJOR.MINOR.PATCH, see CHANGELOG.md).
# Kept in the code instead of config.ini, because config.ini is private and not part of Git.
VERSION = "0.1.12"
GITHUB_URL = "https://github.com/TheRealDeluxee/CryptoStockManager"
LOGO_FILE = "logo.png"   # shown in the start message (in Git as logo.png)
# Recommended number of assets with 1 for the free APIs (see README "How many assets with 1?")
MAX_CRYPTO, MAX_CRYPTO_WITH_KEY, MAX_STOCKS = 8, 12, 30

MORNING_ANALYSIS = "09:30"
NOON_ANALYSIS = "12:30"
EVENING_SUMMARY = "18:30"
EVENT_REMINDER = "08:00"   # reminder for events from events.ini (with daily analysis)

# Template for /ai. Created as ai_prompt.txt on first use and can be edited there
# with any editor. The program fills in the placeholders in {curly brackets}.
AI_PROMPT_FILE = "ai_prompt.txt"
AI_PROMPT_TEMPLATE = """Assess {name} ({symbol}, {type}) for a three-month horizon (no day trading) using my technical data and current news. {task}

A. My program's data (daily values, {date})
EUR prices from {source}; compare levels only with this source.
- Position: {position}
- Price: {price} · 1 day {day} · 7-day trend {week}
- 250-day high/low: {high} / {low}
- Resistance: {resistance}
- Support: {support}
- Risk/reward: {risk_reward}
- RSI14: {rsi}
- MACD (EMA12/26): {macd}
- Bollinger (20,2): {bollinger}
- Trend: {trend}
- Short term: {short_term}
- Bottom formation: {bottom}
- Top formation: {top}
- Chart pattern: {pattern}
- Market phase {name}: {phase}
- Market phase {market_name}: {market}
- Program score: {score}
- Signals today: {signals}

B. News and dates
Web search (if unavailable, say so): news of the last 14 days, confirmed dates of the next 4 weeks; older figures only dated.
{research}

C. Rules
- Report only what supports, contradicts or changes risk or entry/exit; minor points in one sentence; merge identical signals.
- Technicals only from part A; news with date and source. Missing essentials: write exactly "not verifiable" plus what is missing.
- Price levels critically: a period low or low RSI is only a possible support or recovery.
- Trading plan: staggered tranches; exit below support set in advance (daily close?). Bearish phase: small tranches; trend change only above SMA200. Partial profits at resistance; limit size. {tax}Max. three price zones, few numbers, no disclaimers.
- {evidence}

D. Output
1. Decision in three sentences: (1) {decision}, both market phases, main reason. (2) Support/resistance in euro; bottom/top formation only if decisive. (3) Program score from A; risk and confidence (high/medium/low).
2. Decision drivers: max. four points (technicals, news, valuation): what is proven and what it means for me.
3. Trading plan table (scenario | observable signal | my action):
{plan}
4. Strongest counterargument (max. two sentences) and what would strengthen it.
5. Max. four confirmed dates (next 4 weeks) as table: date and German time, event, relevance, source.
6. Conclusion in max. four simple sentences: what I do now, what I wait for, what triggers or rules out {action}; max. two price levels, no score or abbreviations.
"""
# Earlier default templates (SHA-256 of the text): an unchanged ai_prompt.txt with one of them is replaced by
# the current template. Your own changes in ai_prompt.txt are never overwritten.
OLD_AI_PROMPT_HASHES = {
    'd1e2b611158317a4a773e7cb1731b04a95dc5e4d97f52dbbb17484d1a83e716e',   # 0.1.8 - 0.1.10
    'de1e14d1b2b86e561090bba6a801e5d4191da15ff68d9c23fc050db1d5dc01d6',   # 0.1.5 - 0.1.7
    '6aa0b27ab548496d7690d153bd3c6930b6b36a3042d93b414d9ec2d3c4156856',   # 0.1.0
}
# Backtest findings for the AI (54 own assets, up to 14 years, 20 days after a signal vs. an average day)
AI_EVIDENCE = {
    'stock': ("My backtest, stocks: slight rebound after strong sell-offs (58 % up after 20 days, normal 52 %), "
              "waiting for a confirmed bottom did not help; rises and chart patterns: no follow-through."),
    'crypto': ("My backtest, crypto: strength continued (100-day high: 60 % up after 20 days, normal 49 %); "
               "sell-offs often continued: buy only after an observable bottom."),
}
AI_SCORE_NOTE = {'stock': 'stocks: high = no edge, very low = slight rebound',
                 'crypto': 'crypto: high = strength tended to continue'}


# /ai_api_decision3s and /ai_api_plan: the filled ai_prompt.txt plus one of these tasks (instead of the full answer D)
AI_API_TASKS = {
    'decision': ("\n\nE. Answer for this request (replaces D): reply only with item 1 of D, the decision in exactly "
                 "three sentences. Plain text, no heading, no list, no other items, no sources list."),
    'plan': ("\n\nE. Answer for this request (replaces D): reply only with the trading plan of item 3 of D as at most "
             "three short lines 'scenario: observable signal -> my action' (plain text, no table, no sources list), "
             "then exactly this block for my program:\n"
             "```planalert\n"
             "support = <price in EUR where the buy zone starts, or empty>\n"
             "max_entry = <buy only up to this many % above support, 0 to 10>\n"
             "target = <price in EUR for partial profits, or empty>\n"
             "stop = <the plan fails at a daily close below this price in EUR, or empty>\n"
             "note = <one line: tranche size and the conditions that are not price levels>\n"
             "```\n"
             "Prices from the series in A, two decimals, no currency sign. support or target is required; "
             "target above support + max_entry; stop below the buy zone and below the target."),
}


def split_plan_answer(answer):
    """Claude's /ai_api_plan answer -> (text without the block, content of the ```planalert block or '')."""
    match = re.search(r"```planalert\s*\n(.*?)```", answer, re.DOTALL)
    if not match:
        return answer.strip(), ''
    return (answer[:match.start()] + answer[match.end():]).strip(), match.group(1)


def parse_plan_block(block):
    """'support = 312.15' ... lines from the ```planalert block -> dict for cl.add_plan. ValueError if unusable."""
    if not block.strip():
        raise ValueError('the answer contains no planalert block.')
    values = {}
    for line in block.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip().lower()] = value.strip()
    price = lambda key: cl.parse_price(values[key]) if values.get(key, '').strip('<>- ') else None
    plan = {'support': price('support'), 'target': price('target'), 'stop': price('stop'),
            'max_entry': cl.parse_percent(values.get('max_entry', '').strip() or '0'),
            'note': ' '.join(values.get('note', '').split())}
    cl.check_plan(plan['support'], plan['max_entry'], plan['target'], plan['stop'])
    return plan


def update_ai_prompt_file():
    """Create ai_prompt.txt, or replace an unchanged earlier default template with the current one."""
    if os.path.exists(AI_PROMPT_FILE):
        with open(AI_PROMPT_FILE, encoding='utf-8-sig') as file:
            text = file.read()
        if hashlib.sha256(text.strip().encode('utf-8')).hexdigest() not in OLD_AI_PROMPT_HASHES:
            return   # current template or your own version: keep it
        func.log_print('ai_prompt.txt: unchanged earlier template replaced by the current one')
    with open(AI_PROMPT_FILE, 'w', encoding='utf-8') as file:
        file.write(AI_PROMPT_TEMPLATE)


class CryptoStockManager:
    def __init__(self, config_file_path="config.ini", output_dir="plots"):
        self.output_dir = output_dir
        func.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"{config_file_path} not found.")
        # Bring config.ini up to date first (old section name, new settings), then read it
        for change in cl.update_config(config_file_path):
            func.log_print(f"config.ini: {change}")
        for change in cl.convert_value_alerts():   # price alerts up to 0.1.11 -> plans
            func.log_print(change)
        config = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'))
        config.read(config_file_path, encoding="utf-8-sig")
        self.version = VERSION   # an old [version] section in config.ini is ignored
        self.config_file_path = config_file_path
        func.alert_settings = cl.read_alert_settings(config_file_path)   # invalid values stop the start
        self.test_mode = config.getint('alertsconfig', 'test_mode')
        func.seven_day_price_change = config.getfloat('alertsconfig', 'seven_day_price_change')
        func.previous_alarm_change = config.getfloat('alertsconfig', 'previous_alarm_change')
        func.total_loss_limit = config.getfloat('alertsconfig', 'total_loss_limit')
        limits = (func.seven_day_price_change, func.previous_alarm_change, func.total_loss_limit)
        if not all(math.isfinite(value) for value in limits):
            raise ValueError('Alarm values must be finite numbers.')
        if self.test_mode not in (0, 1):
            raise ValueError('test_mode must be 0 or 1.')
        if func.seven_day_price_change < 0 or func.previous_alarm_change < 0:
            raise ValueError('Alarm thresholds must not be negative.')
        if func.total_loss_limit >= 0:
            raise ValueError('total_loss_limit must be negative, e.g. -5 (= alarm at a total P&L below -5 %).')
        # Optional CoinGecko demo key: used if set, otherwise the free public access
        func.coingecko_api_key = config.get('coingecko', 'api_key', fallback='').strip()
        func.anthropic_api_key = config.get('anthropic', 'api_key', fallback='').strip()
        func.anthropic_model = config.get('anthropic', 'model', fallback='').strip() or 'claude-opus-5-5'
        self.stock_update = False
        # For the health check in the log: start time, last runs, alarms since the start, bot state
        self.started = time.time()
        self.last_runs = {}   # "crypto 100h" -> (time, ok, total)
        self.alarms_sent = 0
        self.bot_alive = lambda: True   # replaced by the bot process check in run_program()
        self.crypto_items_hh = []
        self.crypto_items_hd = []
        self.stock_items_hh = []
        self.stock_items_hd = []
        cols = ['Name', 'Piece [€]', '7d P&L [%]', '1d P&L [%]', 'Rating',
                'Type', 'Value [€]', '1d P&L [€]', 'Total P&L [€]', 'Total P&L [%]']
        self.df_crypto_hd = pd.DataFrame(columns=cols)
        self.df_stock_hd = pd.DataFrame(columns=cols)
        self.df_crypto_hh = pd.DataFrame(columns=cols)
        self.df_stock_hh = pd.DataFrame(columns=cols)
        self.last_daily_update = None
        # Daily closing prices per asset from the last daily analysis, for the P&L chart.
        # Kept when assets are reloaded and on days without stock analysis (weekend).
        self.daily_prices = {}   # (type, symbol lower) -> price per day
        self.events_done = set()   # events already reminded today (in memory only)
        self.missing_held_notified = set()   # held assets missing in assets.ini, already reported in Telegram
        self.plan_stop_checked = {}   # plan -> day the stop was last checked with the daily closes
        self.plan_zone_sent = {}   # (plan, buy_zone/target) -> day of the last message: at most one per day
        self.pending_plans = {}   # /ai_api_plan proposals waiting for the save button (in memory only)

    def load_config_items(self):
        """Build the new lists completely first, then replace the old ones."""
        all_entries = cl.read_assets()
        entries = cl.read_assets(alarm_only=True)   # only assets with 1 are part of the alarm analysis
        rows = cl.read_transactions()
        # Assets you hold are always analysed (P&L, loss limit, Cross-EMA Sell), even without 1.
        held = cl.held_assets(rows)
        added = [key for key in all_entries if key in held and key not in entries]
        entries.update({key: all_entries[key] for key in added})
        crypto_hh, crypto_hd, stock_hh, stock_hd = [], [], [], []
        old_hh = {(item.type, item.symbol.lower()): item
                  for item in self.crypto_items_hh + self.stock_items_hh}
        old_hd = {(item.type, item.symbol.lower()): item
                  for item in self.crypto_items_hd + self.stock_items_hd}
        for (asset_type, symbol), name in entries.items():
            symbol = symbol.upper() if asset_type == 'stock' else symbol
            item_hh = cl.apply_transactions(cl.CryptoStock(symbol, name, asset_type), rows)
            item_hd = copy.deepcopy(item_hh)
            key = (asset_type, symbol.lower())
            # Reloading after /buy must not send known alarms again.
            if key in old_hh:
                item_hh.state.alarm_prev = copy.deepcopy(old_hh[key].state.alarm_prev)
            if key in old_hd:
                item_hd.state.alarm_prev = copy.deepcopy(old_hd[key].state.alarm_prev)
            if asset_type == 'crypto':
                crypto_hh.append(item_hh)
                crypto_hd.append(item_hd)
            else:
                stock_hh.append(item_hh)
                stock_hd.append(item_hd)
        show = lambda key: key[1].upper() if key[0] == 'stock' else key[1]
        if added:
            func.log_print(f"Held without 1 in assets.ini, analysed anyway: {', '.join(show(key) for key in added)}")
        missing_held = sorted(show(key) for key in held if key not in all_entries)
        inactive = sorted({show((row['type'], row['symbol'])) for row in rows
                           if (row['type'], row['symbol']) not in all_entries} - set(missing_held))
        if inactive:
            func.log_print(f"Inactive transactions (not in assets.ini, not held): {', '.join(inactive)}")
        if missing_held:
            func.log_print(f"Held but missing in assets.ini, not analysed: {', '.join(missing_held)}", 'WARN')
            if set(missing_held) != self.missing_held_notified:   # once, not on every reload
                func.pushover("<b>Positions without prices</b>\nYou hold these assets according to transactions.csv, "
                              f"but they are missing in assets.ini: <b>{html.escape(', '.join(missing_held))}</b>\n"
                              "They are not in the daily P&L and get no loss limit alarm. Add the symbol to "
                              "assets.ini or use the same symbol as there in transactions.csv.", html_format=True)
        self.missing_held_notified = set(missing_held)
        self.crypto_items_hh, self.crypto_items_hd = crypto_hh, crypto_hd
        self.stock_items_hh, self.stock_items_hd = stock_hh, stock_hd
        # Daily tables are kept: price, P&L and rating do not depend on the holdings.
        func.log_print(f"Loaded: {len(entries)} of {len(all_entries)} symbols in the alarm analysis, {len(rows)} transactions.")

    def _is_stock_trading_hours(self):
        now = time.localtime()
        # Numbers work independently of the Windows language.
        self.stock_update = self.test_mode == 1 or (now.tm_wday < 5 and 8 <= now.tm_hour < 22)

    def load_alert_settings(self):
        """Alarm/info switches from config.ini [alerts]; read before every analysis so that changes
        from /smart_alert_config apply without a restart. On an error the previous switches stay."""
        try:
            func.alert_settings = cl.read_alert_settings(self.config_file_path)
        except (OSError, ValueError, configparser.Error) as error:
            func.log_print(f"config.ini [alerts] could not be read, previous settings stay: {error}", 'WARN')

    def _analyse_items(self, items, data_type, label=''):
        self.load_alert_settings()
        results, failed, alarm_count, start = [], [], 0, time.time()
        for item in items:
            try:
                df = item.get_API_data(data_type)
                if df is None:
                    raise ValueError('no data received after 5 attempts (source unavailable or too many requests)')
                df, alarms = item.process_data(df, data_type)  # df is cut to 100 values afterwards
                if data_type == '100d':
                    self.daily_prices[(item.type, item.symbol.lower())] = cl.daily_prices(df)
                item.plot_send(df, data_type, alarms)
                alarm_count += len(alarms)
                self._journal_alarms(item, df, data_type, alarms)
                results.append(item.get_summary(df))
            except Exception as error:
                # One failed asset must not block the others.
                func.log_print(f"Analysis {item.symbol} ({data_type}) failed: {error}", 'ERROR')
                failed.append(f"{item.search}: {error}")
        # Status line for monitoring: one per run
        if items:
            self.last_runs[f"{label} {data_type}".strip()] = (time.time(), len(results), len(items))
            self.alarms_sent += alarm_count
            func.log_print(f"Analysis {label} {data_type} done: {len(results)}/{len(items)} ok, "
                           f"{alarm_count} alarms, {len(failed)} failed, {time.time() - start:.0f} s",
                           'WARN' if failed else 'INFO')
        if failed:
            # One message per run instead of one per asset (e.g. when Yahoo blocks requests)
            func.pushover(f"Analysis ({data_type}) failed for {len(failed)} of {len(items)} assets:\n"
                          + "\n".join(failed))
        return results

    def health_check(self):
        """One status line in logfile.txt only (not sent to Telegram): is everything still running?
        WARN if something looks wrong (bot stopped, analysis overdue or failing, little disk space)."""
        parts, problems = self._health()
        func.log_print("Health check: " + ", ".join(parts) + (f" | PROBLEMS: {'; '.join(problems)}" if problems else ''),
                       'WARN' if problems else 'INFO')

    def send_status(self):
        """/status: the health check plus version, device and the next scheduled runs, sent to Telegram."""
        parts, problems = self._health()
        next_runs = {}
        names = {'hundred_hour_analysis': 'hourly analysis', 'hundred_day_analysis': 'daily analysis',
                 'send_summary': 'summary', 'event_reminders': 'reminders', 'plan_alert_check': 'plan alerts'}
        for job in schedule.jobs:
            name = names.get(getattr(getattr(job.job_func, 'func', None), '__name__', ''))
            if name and job.next_run and (name not in next_runs or job.next_run < next_runs[name]):
                next_runs[name] = job.next_run
        try:
            transactions = len(cl.read_transactions())
        except (OSError, UnicodeError, ValueError) as error:
            transactions = f'not readable ({error})'
        lines = [f"<b>CaSa status</b> · version {html.escape(str(self.version))} on {html.escape(socket.gethostname())}",
                 f"Test mode {'on' if self.test_mode == 1 else 'off'} · "
                 f"{len(self.crypto_items_hd) + len(self.stock_items_hd)} assets in the alarm analysis · "
                 f"{transactions} transactions",
                 '', html.escape(', '.join(parts))]
        if next_runs:
            lines += ['', 'Next: ' + ', '.join(f"{name} {when:%d/%m %H:%M}" for name, when in
                                               sorted(next_runs.items(), key=lambda item: item[1]))]
        lines += ['', ('⚠ Problems: ' + html.escape('; '.join(problems))) if problems else '✅ No problems found.']
        func.pushover('\n'.join(lines), html_format=True)

    def send_portfolio(self):
        """/portfolio: daily P&L table and 13-week chart now (from the last daily analysis)."""
        if self.df_crypto_hd.empty and self.df_stock_hd.empty:
            func.pushover('No daily results yet. Please run /analyse daily first.')
            return
        self.send_pnl()
        self.send_pnl_chart()

    def _health(self):
        """(parts, problems) for the health check and /status."""
        now, problems = time.time(), []
        hours = (now - self.started) / 3600
        parts = [f"up {int(hours // 24)} d {int(hours % 24)} h"]
        alive = self.bot_alive()
        parts.append('bot ok' if alive else 'bot stopped')
        if not alive:
            problems.append('Telegram bot stopped')
        # Last run of each analysis; overdue if the hourly one is older than 2 h or the daily one older than 26 h
        limits = {'crypto 100h': 2, 'crypto 100d': 26}
        for name, (when, ok, total) in sorted(self.last_runs.items()):
            parts.append(f"{name} {time.strftime('%d/%m %H:%M', time.localtime(when))} {ok}/{total} ok")
            if ok < total:
                problems.append(f"{name}: {total - ok} failed")
            if self.test_mode != 1 and name in limits and now - when > limits[name] * 3600:
                problems.append(f"{name} overdue")
        if self.test_mode != 1 and self.crypto_items_hh and 'crypto 100h' not in self.last_runs and hours > 2:
            problems.append('hourly analysis has not run yet')
        parts.append(f"alarms since start {self.alarms_sent}")
        try:
            parts.append(f"plan alerts {sum(plan['status'] != 'ended' for plan in cl.read_plans())}")
        except (OSError, ValueError, configparser.Error) as error:
            problems.append(f"planalert.ini: {error}")
        # Memory of the main program (Linux/Raspberry Pi; on Windows only if psutil is installed)
        memory = None
        try:
            with open('/proc/self/status') as file:
                memory = next(int(line.split()[1]) / 1024 for line in file if line.startswith('VmRSS'))
        except (OSError, StopIteration, ValueError):
            try:
                import psutil
                memory = psutil.Process().memory_info().rss / 1024 / 1024
            except Exception:
                pass
        if memory:
            parts.append(f"memory {memory:.0f} MB")
        free = shutil.disk_usage('.').free / 1e9
        parts.append(f"disk free {free:.1f} GB")
        if free < 1:
            problems.append('less than 1 GB disk space')
        if os.path.exists(func.LOG_FILE):
            parts.append(f"log {os.path.getsize(func.LOG_FILE) / 1024:.0f} KB")
        return parts, problems

    def _journal_alarms(self, item, df, data_type, alarms):
        """Record sent alarms (not infos) in alarm_journal.csv (not in test mode: there all analyses run at every start)."""
        if not alarms or self.test_mode == 1:
            return
        for code, info in alarms.items():
            if info.get('kind') == 'info':
                continue   # infos are not alarms: not part of the hit-rate statistics
            try:
                cl.journal_add(item.type, item.symbol, item.search, code, data_type, df['Date'].iloc[-1],
                               df['Price'].iloc[-1], item.state.score, item.state.watch_list)
            except (OSError, ValueError) as error:
                func.log_print(f"Alarm journal: {item.symbol} {code} not recorded: {error}", 'WARN')

    def update_journal(self):
        """Fill in the results of alarms that are 20 trading days (crypto: days) old."""
        try:
            done = cl.journal_update(self.daily_prices)
        except (OSError, ValueError) as error:
            func.log_print(f"Alarm journal could not be updated: {error}", 'WARN')
            return
        if done:
            func.log_print(f"Alarm journal: {done} results filled in")

    def hundred_day_analysis(self):
        self._is_stock_trading_hours()
        cols = list(self.df_crypto_hd.columns)
        self.df_crypto_hd = pd.DataFrame(self._analyse_items(self.crypto_items_hd, '100d', 'crypto'), columns=cols)
        self.df_stock_hd = pd.DataFrame(columns=cols)
        if self.stock_update:
            self.df_stock_hd = pd.DataFrame(self._analyse_items(self.stock_items_hd, '100d', 'stocks'), columns=cols)
        else:
            func.log_print('Analysis stocks 100d skipped: outside trading hours')
        self.last_daily_update = time.strftime('%d/%m/%Y %H:%M:%S')
        self.update_journal()

    def hundred_hour_analysis(self):
        # As before: automatic hourly analysis only for cryptocurrencies.
        cols = list(self.df_crypto_hh.columns)
        self.df_crypto_hh = pd.DataFrame(self._analyse_items(self.crypto_items_hh, '100h', 'crypto'), columns=cols)

    def _daily_analysis(self, symbol, asset_type):
        """Daily analysis of an asset from assets.ini for /check and /ai (without alarm)."""
        entries = cl.read_assets()
        key = (asset_type, symbol.lower())
        if key not in entries:
            raise ValueError(f"{symbol} is not in assets.ini ({asset_type}).")
        name = entries[key]
        symbol = symbol.upper() if asset_type == 'stock' else symbol.lower()
        # Separate object: changes neither analysis nor alarm states.
        item = cl.apply_transactions(cl.CryptoStock(symbol, name, asset_type), cl.read_transactions())
        full = item.get_API_data('100d')
        full = func.calc_indicator_fuctions(full)
        sr_zones = func.support_resistance(full)  # before cutting: needs the long history
        item.state.pattern = func.chart_pattern(full)
        df = func.cut_to_window(full)
        item.state = func.calc_state_fuctions(item.state, df)
        item.state.score_rank = func.score_rank(full)
        details = func.analysis_details(df, item.state.pattern, item.state.score_rank, 'days', sr_zones)
        return name, symbol, item, df, full, sr_zones, details

    def single_daily_analysis(self, symbol, asset_type, ask_ai=False):
        """ask_ai=True (only /check): the AI buttons below the chart, as below alarms and infos."""
        name, symbol, item, df, full, sr_zones, details = self._daily_analysis(symbol, asset_type)
        # Telegram HTML as for the alarms (bold/italic/monospace)
        message = (f"<b>{html.escape(name)} ({html.escape(symbol)}) — daily analysis (100d)</b>\n"
                   f"Manual request without alarm\n\n" + details['message'])
        if not item.state.watch_list:
            message += "\n\n" + html.escape(f"Amount >1 year: {item.state.amount_older_than_one_year:.2f} "
                                           f"({item.state.amount_older_than_one_year_pct:.2f} %)")
        image_name = 'manual_' + uuid.uuid4().hex
        func.plot_and_save(df, image_name, '100d', item.state.zero_line, sr_zones, item.state.pattern)
        func.pushover_image(image_name, message, delete_after_send=True, html_format=True,
                            buttons=func.ai_buttons(asset_type, symbol) if ask_ai else None)

    def _ai_question(self, symbol, asset_type):
        """Fill the prompt from ai_prompt.txt with the daily analysis. Returns (name, symbol, prompt text)."""
        name, symbol, item, df, full, sr_zones, details = self._daily_analysis(symbol, asset_type)
        update_ai_prompt_file()
        with open(AI_PROMPT_FILE, encoding='utf-8-sig') as file:
            question = file.read()

        price = float(df['Price'].iloc[-1])

        def pct(value):
            return f"{value:+.1f} %"

        def zone_text(zone, extreme):
            if zone is None:
                return 'none in the loaded period'
            extra = (f"{zone['touches']} touches" if zone['touches']
                     else f"only the period {extreme}, no confirmed zone")
            return f"{func.format_euro(zone['center'])} ({pct((zone['center'] / price - 1) * 100)}, {extra})"

        resistance, support = sr_zones if sr_zones else (None, None)

        # Position with real values (quantity, cost, value, profit/loss)
        def qty(value):
            # crypto always with 8 decimals (as usual on exchanges), stocks without unnecessary zeros
            text = f"{value:.8f}" if asset_type == 'crypto' else f"{value:.8g}"
            return f"{text} {unit[:-1] if value == 1 else unit}"   # "1 share", "2 shares"

        state = item.state
        unit = 'shares' if asset_type == 'stock' else 'coins'
        if state.watch_list:
            position = 'no holdings'
        else:
            position = (f"{qty(state.amount_crypto_stock)}, avg. cost {func.format_euro(state.zero_line)} "
                        f"value {func.format_euro(state.current_balance_eur)}, "
                        f"P/L {func.format_euro(state.current_profit)} ({pct(state.current_profit_pct)}), "
                        f"held >1 year: {qty(state.amount_older_than_one_year)}")

        # The question and the trading plan depend on whether I hold the asset
        if state.watch_list:
            task = "I have no position: buy or wait? Name the concrete signal that would change this decision."
            decision, action = 'Start with Buy or Wait', 'a buy'
            plan = ("- Wait | which confirmation is missing? | what do I do until then?\n"
                    "- First buy | which price reaction confirms the entry? | when do I buy a small first tranche?\n"
                    "- Plan fails | which price level or event invalidates it? | what do I do before/after a buy?")
        else:
            task = "I hold a position (see A): hold, add or reduce? Name the concrete signal that would change this decision."
            decision, action = 'Start with Hold, Add or Reduce', 'adding or selling'
            plan = ("- Hold | what confirms holding? | what do I watch?\n"
                    "- Add | which price reaction confirms adding? | when and how much?\n"
                    "- Reduce/exit | which price level or event? | how much do I sell?")
        tax = ('Consider the holding period: selling parts held less than 1 year is taxable in Germany. '
               if asset_type == 'crypto' and not state.watch_list else '')
        source = 'CoinGecko, average of several exchanges' if asset_type == 'crypto' else 'Yahoo Finance, converted if needed'

        # MACD line (EMA12 - EMA26) and last crossover in the 100-value window
        diff = df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-1]
        crosses = df.index[df['Position'].fillna(0) != 0]
        since = f"crossover {len(df) - 1 - crosses[-1]} days ago" if len(crosses) else 'no crossover in 100 days'
        macd = f"{'positive (EMA12 > EMA26)' if diff > 0 else 'negative (EMA12 < EMA26)'}, {since}"

        # Position within the Bollinger band: 0 % = lower band, 100 % = upper band
        lower, upper = df['BB_Lower'].iloc[-1], df['BB_Upper'].iloc[-1]
        band = (price - lower) / (upper - lower) * 100 if upper > lower else 50.0
        where = ('below the lower band' if band < 0 else 'near the lower band' if band < 20 else
                 'above the upper band' if band > 100 else 'near the upper band' if band > 80 else 'in the middle range')
        bollinger = f"price {where} ({band:.0f} % of band width)"

        # Bottom/top formation: signs from the long history
        bottom, top = func.formation_signals(full)

        def formation_text(signals, names):
            # grouped by result (yes / no / not assessable) instead of one entry per sign: shorter, same content
            groups = {'yes': [], 'no': [], 'n/a': []}   # n/a = no recent extreme
            for key, label in names:
                result = 'n/a' if signals[key] is None else 'yes' if signals[key] else 'no'
                groups[result].append(label)
            parts = [f"{result}: {', '.join(labels)}" for result, labels in groups.items() if labels]
            return f"{len(groups['yes'])}/{len(names)} – {'; '.join(parts)}"

        # Double bottom/top is not listed here: the chart pattern line (same detection as the Telegram
        # message) covers it, otherwise the AI would see two different pattern names.
        bottom_text = formation_text(bottom, (('folge', 'higher lows'), ('divergenz', 'bullish RSI divergence'),
                                              ('volumen', 'selling climax at low'),
                                              ('bestaetigung', 'price above EMA12')))
        top_text = formation_text(top, (('folge', 'lower highs'), ('divergenz', 'bearish RSI divergence'),
                                        ('volumen', 'buying climax at high'),
                                        ('bestaetigung', 'price below EMA12')))
        rr = func.risk_reward(price, sr_zones)
        rank = details.get('rank')
        score_text = (f"{rank}/10 (vs. the last 100 days; {AI_SCORE_NOTE[asset_type]})"
                      if rank else f"{details['score']} (-18 to +18; {AI_SCORE_NOTE[asset_type]})")

        # Alarms and infos that the analysis would send today for this asset (with your [alerts] settings)
        state = item.state
        labels = {code: label for _codes, label, *_ in func.ALERTS.values() for code in _codes}
        try:
            today_alarms, _, _ = func.alarm(df, name, state.watch_list, state.current_profit_pct,
                                            state.amount_older_than_one_year, state.amount_older_than_one_year_pct,
                                            '100d', state.pattern, state.score_rank, sr_zones, asset_type,
                                            yesterday_profit_pct=state.yesterday_profit_pct)
            signals = ', '.join(f"{labels.get(code, code)} ({info.get('kind', 'alarm')})"
                                for code, info in today_alarms.items()) or 'none'
        except Exception as error:   # extra information: an error must not stop /ai
            signals = f'not verifiable ({str(error)[:60]})'
        # Short forms for the prompt: it has to fit into one Telegram message (4096 characters)
        rsi_label = details['texts']['rsi'].split(': ', 1)[-1].rstrip('.')   # "neutral", "overbought – ..."
        pattern_short = re.sub(r' \([^)]*\)', '', details['texts']['pattern'], count=1)   # without the explanation

        # Market phase: the asset itself and its leading market (Bitcoin or stock index)
        phase = func.trend_phase(full['Price'])
        if asset_type == 'crypto' and symbol == 'bitcoin':
            market_name, market = 'overall crypto market', 'as above, Bitcoin itself is the leader'
        else:
            try:
                market_name, market_prices = func.market_prices(asset_type, symbol)
                market = func.trend_phase(market_prices) if market_prices is not None else 'not verifiable (no data)'
            except Exception as error:   # leading market is extra information: an error must not stop /ai
                market_name, market = 'overall market', f'not verifiable ({str(error)[:60]})'

        # stocks: intraday high/low (as on TradingView), crypto: closing prices
        high = full['PriceHigh'].max() if 'PriceHigh' in full else full['Price'].max()
        low = full['PriceLow'].min() if 'PriceLow' in full else full['Price'].min()
        if asset_type == 'stock':
            research = ("- Earnings/outlook: last revenue, guidance, consensus, next date (German time); "
                        "separate actuals/estimates; what would change the call.\n"
                        "- Analyst targets/valuation: major target changes (original currency), trailing and forward P/E (estimate period).\n"
                        "- Company/industry: management, demand, competition, regulation, exports.")
        else:
            research = ("- Capital flows: ETF in- and outflows, large wallet movements, exchange balances.\n"
                        "- Regulation/macro: laws, proceedings, rate decisions, inflation.\n"
                        "- Project/network: updates, token unlocks, security incidents, competition.")

        values = {
            'name': name, 'symbol': symbol,
            'type': 'cryptocurrency' if asset_type == 'crypto' else 'stock',
            'task': task, 'decision': decision, 'action': action, 'plan': plan,
            'tax': tax, 'source': source,
            'date': time.strftime('%d/%m/%Y %H:%M'),
            'position': position,
            'price': func.format_euro(price),
            'day': pct(item.state.current_one_day_price_change_pct),
            'week': pct(item.state.current_seven_days_slope_pct),
            'high': f"{func.format_euro(high)} ({pct((price / high - 1) * 100)})",
            'low': f"{func.format_euro(low)} ({pct((price / low - 1) * 100)})",
            'resistance': zone_text(resistance, 'high'), 'support': zone_text(support, 'low'),
            'score': score_text,
            'pattern': pattern_short,
            # facts only: a "good" ratio did not predict better results; the levels are in the two lines above
            'risk_reward': (f"ratio {(resistance['center'] / price - 1) / (1 - support['center'] / price):.1f} : 1 "
                            "(room to resistance vs. distance to support)" if resistance and support
                            else rr['short'].rstrip('.')),
            'rsi': f"{df['RSI14'].iloc[-1]:.0f} ({rsi_label})",
            'macd': macd, 'bollinger': bollinger,
            'trend': details['texts']['trend'], 'short_term': details['texts']['short'],
            'bottom': bottom_text, 'top': top_text, 'research': research,
            'phase': phase, 'market_name': market_name, 'market': market,
            'signals': signals, 'evidence': AI_EVIDENCE[asset_type],
        }
        # Simple replace instead of format(): other {brackets} in the file do no harm this way.
        for key, value in values.items():
            question = question.replace('{' + key + '}', value)
        return name, symbol, question.strip()

    def ai_prompt(self, symbol, asset_type):
        """/ai_prompt: the filled prompt from ai_prompt.txt as copyable text for ChatGPT or Claude (no API)."""
        name, symbol, question = self._ai_question(symbol, asset_type)
        message = (f"<b>AI prompt: {html.escape(name)}</b> – copy into ChatGPT or Claude\n"
                   f"<pre>{html.escape(question)}</pre>")
        # Telegram counts only the visible text (max. 4096), not the HTML tags
        if func.telegram_length(message) <= 4096:
            func.pushover(message, html_format=True)
            return
        # Too long for one Telegram message: split into two copyable blocks at a paragraph near the middle
        breaks = [i for i in range(len(question)) if question.startswith('\n\n', i)]
        cut = min(breaks, key=lambda i: abs(i - len(question) / 2)) if breaks else len(question) // 2
        parts = (question[:cut].strip(), question[cut:].strip())
        for number, part in enumerate(parts, 1):
            hint = 'Copy and paste into ChatGPT or Claude:' if number == 1 else 'Paste directly after part 1:'
            text = f"<b>AI prompt: {html.escape(name)} (part {number}/2)</b>\n{hint}\n<pre>{html.escape(part)}</pre>"
            if func.telegram_length(text) <= 4096:
                func.pushover(text, html_format=True)
            else:   # even half is too long (heavily extended template): send without formatting
                func.pushover(part)

    def ai_api(self, symbol, asset_type, mode):
        """/ai_api_decision3s (mode 'decision') and /ai_api_plan (mode 'plan'): the same data as /ai_prompt,
        answered directly by the Claude API (with web search for the news). Without a key: error message."""
        if not func.claude_available():
            func.pushover(func.CLAUDE_NO_KEY)
            return
        name, symbol, question = self._ai_question(symbol, asset_type)
        try:
            answer, usage = func.claude(question + AI_API_TASKS[mode], web_search=True, with_usage=True)
        except (PermissionError, RuntimeError) as error:
            func.pushover(f"Claude API for {name}: {error}")
            return
        footer = f"<i>🧾 {html.escape(func.claude_usage_text(usage))} · {time.strftime('%d/%m/%Y %H:%M')}</i>"
        if mode == 'decision':
            func.pushover(f"<b>🤖 Decision: {html.escape(name)}</b>\n{html.escape(answer[:3500])}\n{footer}",
                          html_format=True)
            return
        text, block = split_plan_answer(answer)
        message = f"<b>🤖 Trading plan: {html.escape(name)}</b>\n{html.escape(text[:2500])}\n{footer}"
        try:
            plan = parse_plan_block(block)
        except ValueError as error:
            func.pushover(message + f"\n\nNo plan to save: {html.escape(str(error))}", html_format=True)
            return
        key = uuid.uuid4().hex[:10]
        self.pending_plans[key] = dict(plan, symbol=symbol, type=asset_type, name=name)
        shown = symbol.upper() if asset_type == 'stock' else symbol
        try:
            replaces = any(p['symbol'] == symbol.lower() for p in cl.read_plans())
        except (OSError, ValueError, configparser.Error):
            replaces = False
        preview = '\n'.join(cl.plan_lines(shown, plan['support'], plan['max_entry'], plan['target'], plan['stop'],
                                          plan['note'], datetime.date.today())[:6])
        func.pushover(message + f"\n\n<b>For planalert.ini:</b>\n<pre>{html.escape(preview)}</pre>"
                      + ("\nSaving replaces your current plan for this asset." if replaces else ''),
                      html_format=True, buttons=[[('✅ Save as plan', f'aiplanok|{key}'),
                                                  ('Discard', f'aiplanno|{key}')]])

    def save_ai_plan(self, key, save):
        """Button below an /ai_api_plan answer: save the proposed plan in planalert.ini or discard it."""
        plan = self.pending_plans.pop(key, None)
        if plan is None:
            func.pushover('This plan proposal is no longer available (already handled or CaSa was restarted).')
            return
        if not save:
            func.pushover(f"Plan proposal for {plan['name']} discarded.")
            return
        try:
            name, _, replaced = cl.add_plan(plan['symbol'], plan['support'], plan['max_entry'], plan['target'],
                                            plan['stop'], plan['note'])
        except (OSError, ValueError, configparser.Error) as error:
            func.pushover(f"Plan for {plan['name']} not saved: {error}")
            return
        func.pushover(f"Plan {'replaced' if replaced else 'saved'} in planalert.ini: {name}. Show it with /plan_alert.")
        self.plan_alert_check()   # zone or target may already be reached

    def send_summary(self):
        if self.df_crypto_hd.empty and self.df_stock_hd.empty:
            func.pushover('No daily results yet. Please run /analyse daily first.')
            return
        for kind, frame in (('crypto', self.df_crypto_hd), ('stock', self.df_stock_hd)):
            if frame.empty:
                continue
            frame = frame.sort_values('1d P&L [%]', ascending=False)
            image_name = f'summary_{kind}_' + uuid.uuid4().hex
            title = 'Crypto · Daily Summary' if kind == 'crypto' else 'Stocks · Daily Summary'
            # The data timestamp is already in the Telegram message, not again in the image
            func.create_table_image(frame, os.path.join(self.output_dir, image_name + '.png'), title=title)
            message = f'Data as of: {self.last_daily_update}'  # title is already in the image
            func.pushover_image(image_name, message, delete_after_send=True)
        self.send_pnl()
        self.send_pnl_chart()
        if datetime.date.today().day == 1:   # monthly: how did the alarms do on new data?
            self.send_journal_report()

    def send_journal_report(self):
        try:
            func.pushover(cl.journal_report(), html_format=True)
        except (OSError, ValueError) as error:
            func.log_print(f"Alarm journal report failed: {error}", 'WARN')

    def send_pnl(self):
        """Own positions (crypto and stocks together): value, 1d P&L and total P&L as a table image."""
        frames = [frame for frame in (self.df_crypto_hd, self.df_stock_hd) if not frame.empty]
        if not frames:
            return
        frame = pd.concat(frames, ignore_index=True)
        frame = frame[frame['Value [€]'] > 0]   # only assets you own
        if frame.empty:
            return
        value, d1 = frame['Value [€]'].sum(), frame['1d P&L [€]'].sum()
        subtitle = f"Portfolio {func.format_euro(value)}"
        image_name = 'pnl_' + uuid.uuid4().hex
        func.create_pnl_image(frame, os.path.join(self.output_dir, image_name + '.png'),
                              title='My Daily P&L', subtitle=subtitle)
        message = f'Data as of: {self.last_daily_update}'
        if self.df_stock_hd.empty:
            message += '\nStocks not included (no stock analysis today, e.g. weekend).'
        func.log_print(f"Daily P&L: {len(frame)} positions, value {value:.2f} €, 1d {d1:+.2f} €")
        func.pushover_image(image_name, message, delete_after_send=True)

    def send_pnl_chart(self, weeks=13):
        """Total profit/loss of the portfolio over the last 13 full weeks (3 months, Monday to Sunday,
        the current week up to today) as weekly bars (not the portfolio value)."""
        today = datetime.date.today()
        days = today.weekday() + 1 + (weeks - 1) * 7   # from Monday 12 weeks ago up to today
        try:
            rows = cl.read_transactions()
        except (OSError, UnicodeError, ValueError) as error:
            func.log_print(f"P&L chart skipped: transactions.csv could not be read: {error}", 'WARN')
            return
        if not any(row['action'] in ('buy', 'sell') for row in rows) or not self.daily_prices:
            return
        history, missing = cl.pnl_history(rows, self.daily_prices, days, today=today)
        if len(history) < 2:
            return
        image_name = 'pnl_chart_' + uuid.uuid4().hex
        func.create_pnl_chart(history, os.path.join(self.output_dir, image_name + '.png'),
                              title=f'P&L · last {weeks} weeks')
        message = 'Profit/loss of all positions (realized + unrealized), not the portfolio value.'
        if missing:
            message += f"\nWithout prices (not in the daily analysis): {', '.join(missing)}"
        func.pushover_image(image_name, message, delete_after_send=True)

    def event_reminders(self):
        """Events from events.ini: one reminder the day before, followed by a daily analysis like /check."""
        try:
            events = cl.read_events()
        except (OSError, ValueError, configparser.Error) as error:
            func.pushover(f'events.ini could not be read: {error}')
            return
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        for event in events:
            key = f"{event['key']}_before"
            if event['date'] != tomorrow or key in self.events_done:
                continue
            self.events_done.add(key)
            text = f"\n{html.escape(event['text'])}" if event['text'] else ''
            func.pushover(f"<b>Event tomorrow: {html.escape(event['name'])}</b>"
                          f" ({event['date'].strftime('%A, %d/%m/%Y')}){text}\nDaily analysis follows.",
                          priority=1, html_format=True)
            try:
                self.single_daily_analysis(event['symbol'], event['type'])
            except Exception as error:
                func.pushover(f"Daily analysis for {event['name']} failed: {error}")
        self.earnings_reminders()

    def plan_alert_check(self):
        """Plans from planalert.ini: buy zone and target every hour with the current price (stocks only Monday
        to Friday 08:00-22:00), the stop with the daily closes. Each message once per change of the status;
        whether you hold the asset (transactions.csv) decides the wording. The plan ends at the stop."""
        try:
            plans = [plan for plan in cl.read_plans() if plan['status'] != 'ended']
            held = cl.held_assets(cl.read_transactions())
        except (OSError, ValueError, configparser.Error) as error:
            func.log_print(f'planalert.ini could not be read: {error}', 'ERROR')
            func.pushover(f'planalert.ini could not be read: {error}')
            return
        if not plans:
            return
        self._is_stock_trading_hours()
        checked = messages = 0
        for plan in plans:
            is_held = (plan['type'], plan['symbol']) in held
            symbol = plan['symbol'].upper() if plan['type'] == 'stock' else plan['symbol']
            try:
                if plan['stop'] is not None and self._plan_stop(plan, symbol, is_held):
                    messages += 1
                    continue
                if plan['type'] == 'stock' and not self.stock_update:
                    continue
                price = func.current_price(symbol, plan['type'])
            except Exception as error:
                func.log_print(f"Plan {plan['name']}: price not available ({error})", 'WARN')
                continue
            checked += 1
            top = cl.plan_zone_top(plan)
            status = 'buy_zone' if top is not None and price <= top else \
                'target' if plan['target'] is not None and price >= plan['target'] else 'waiting'
            if status == plan['status']:
                continue
            cl.set_plan_value(plan['key'], 'status', status)
            if status == 'buy_zone':
                title = '🟢 Add-on zone (retest)' if is_held else '🟢 Buy zone'
                action = ('You already hold it: add only if your note allows a further tranche.' if is_held else
                          'Check the conditions in your note before buying (e.g. rebound, higher low, daily close).')
            elif status == 'target' and is_held:
                title, action = '💰 Take partial profits', 'Target reached: take partial profits as planned.'
            else:
                func.log_print(f"Plan {plan['name']}: status {status}, price {price:.2f}")
                continue
            sent_key = (plan['key'].lower(), status)
            if self.plan_zone_sent.get(sent_key) == datetime.date.today():
                # at most one buy zone / target message per day and plan (price moving around the mark)
                func.log_print(f"Plan {plan['name']}: status {status} again today, no second message")
                continue
            self.plan_zone_sent[sent_key] = datetime.date.today()
            messages += 1
            self._plan_message(plan, title, f"Price {func.format_euro(price)}. {action}", priority=1)
        func.log_print(f"Plan alerts: {checked} of {len(plans)} checked, {messages} messages")

    def _plan_stop(self, plan, symbol, is_held):
        """Stop of a plan: a daily close below it since the start of the plan (only finished days) ends the plan.
        Checked once per day and plan. Returns True when the plan ended."""
        today = datetime.date.today()
        if self.plan_stop_checked.get(plan['key'].lower()) == today:
            return False
        closes = self.daily_prices.get((plan['type'], plan['symbol']))
        if closes is None or closes.empty:
            item = cl.CryptoStock(symbol, plan['name'], plan['type'])
            closes = cl.daily_prices(item.get_API_data('100d'))
        self.plan_stop_checked[plan['key'].lower()] = today
        start = plan['start']
        if start is None:   # plan written by hand without start: closes count from today on
            start = today
            cl.set_plan_value(plan['key'], 'start', f"{today:%d/%m/%Y}")
        dates = closes.index.date
        days = closes[(dates < today) & (dates >= start)]
        below = days[days < plan['stop']]
        if below.empty:
            return False
        day, close = below.index[0].date(), float(below.iloc[0])
        reason = f"daily close {close:.2f} on {day:%d/%m/%Y} below stop {plan['stop']:.2f}"
        cl.set_plan_value(plan['key'], 'status', f"ended {today:%d/%m/%Y}: {reason}")
        if is_held:
            title, action = '🔴 Exit', 'Plan failed: sell as planned, do not average down.'
        else:
            title, action = '❌ Setup cancelled', 'Plan failed before buying: no purchase.'
        self._plan_message(plan, title, f"Daily close {func.format_euro(close)} on {day:%d/%m/%Y} is below "
                                        f"the stop {func.format_euro(plan['stop'])}. {action}\nThe plan has ended "
                                        f"(still listed in /plan_alert until you delete it).", priority=1)
        return True

    def _plan_message(self, plan, title, text, priority=1):
        """Message with plan numbers and note, followed by the daily analysis."""
        note = f"\nYour note: {html.escape(plan['note'])}" if plan['note'] else ''
        func.pushover(f"<b>{title}: {html.escape(plan['name'])}</b>\n{html.escape(text)}\n"
                      f"<i>{html.escape(cl.plan_text(plan))}</i>{note}\nDaily analysis follows.",
                      priority=priority, html_format=True)
        try:
            self.single_daily_analysis(plan['symbol'], plan['type'])
        except Exception as error:
            func.pushover(f"Daily analysis for {plan['name']} failed: {error}")

    def earnings_reminders(self):
        """Stocks in the alarm analysis (1 in assets.ini): reminder one day before the quarterly results
        (date from Yahoo), then a daily analysis like /check. On results days prices often jump strongly."""
        try:
            entries = cl.read_assets(alarm_only=True)
        except (OSError, ValueError, configparser.Error) as error:
            func.pushover(f'assets.ini could not be read: {error}')
            return
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        try:   # dates already in events.ini have their own reminder: no second message
            planned = {(event['symbol'], event['date']) for event in cl.read_events()}
        except (OSError, ValueError, configparser.Error):
            planned = set()
        for (asset_type, symbol), name in entries.items():
            if asset_type != 'stock':
                continue
            date = func.next_earnings_date(symbol.upper())
            key = f'earnings_{symbol}_{date}'
            if date != tomorrow or key in self.events_done or (symbol.lower(), date) in planned:
                continue
            self.events_done.add(key)
            func.pushover(f"<b>Quarterly results tomorrow: {html.escape(name)}</b>\n"
                          f"{date.strftime('%A, %d/%m/%Y')} – prices often move strongly on results days.\n"
                          f"Daily analysis follows.", priority=1, html_format=True)
            try:
                self.single_daily_analysis(symbol, asset_type)
            except Exception as error:
                func.pushover(f"Daily analysis for {name} failed: {error}")

    def startup_message(self):
        """Short start message: version, assets in the alarm analysis, next analysis."""
        crypto, stock = len(self.crypto_items_hd), len(self.stock_items_hd)
        # Recommended maximum for the free APIs: CoinGecko 8 (with own key 12), Yahoo 30
        max_crypto = MAX_CRYPTO_WITH_KEY if func.coingecko_api_key else MAX_CRYPTO
        lines = [f"<b>CaSa started</b> · version {html.escape(str(self.version))}",
                 f"{crypto + stock} assets in the alarm analysis ({crypto}/{max_crypto} crypto, {stock}/{MAX_STOCKS} stocks)"]
        if crypto > max_crypto:
            hint = '' if func.coingecko_api_key else ' (with a free CoinGecko key in config.ini: 12)'
            lines.append(f"⚠ More cryptos than recommended: max. {max_crypto}{hint}, otherwise analyses may fail")
        if stock > MAX_STOCKS:
            lines.append(f"⚠ More stocks than recommended: max. {MAX_STOCKS}")
        if self.test_mode == 1:
            lines.append("Test mode: all analyses run once immediately")
        else:
            now = time.strftime('%H:%M')
            if now < MORNING_ANALYSIS:
                daily = MORNING_ANALYSIS
            elif now < NOON_ANALYSIS:
                daily = NOON_ANALYSIS
            else:
                daily = f"now, then tomorrow {MORNING_ANALYSIS}"   # after 12:30 it runs immediately at start
            summary = EVENING_SUMMARY if now < EVENING_SUMMARY else f"tomorrow {EVENING_SUMMARY}"
            lines.append(f"Next daily analysis: {daily} · summary: {summary}")
        try:   # no "next event" line (list with /event), but a faulty events.ini is reported at the start
            cl.read_events()
        except (OSError, ValueError, configparser.Error) as error:
            lines.append(f"events.ini faulty: {html.escape(str(error))}")
        lines.append("All commands: /help")
        lines.append(f'<a href="{GITHUB_URL}">CaSa on GitHub</a>')
        # With the logo as image, the text below it; without logo file as plain message as before.
        # Name compared without case: Linux (Raspberry Pi) distinguishes logo.png and logo.PNG, Windows does not.
        logo = next((name for name in os.listdir('.') if name.lower() == LOGO_FILE.lower()), None)
        if logo:
            func.pushover_photo(logo, '\n'.join(lines), html_format=True)
        else:
            func.pushover('\n'.join(lines), html_format=True)

    def handle_command(self, command):
        """True means: the main loop should perform a clean restart."""
        action = command.get('cmd')
        if action == 'analyse':
            mode = command.get('mode', 'hourly')
            if mode == 'hourly':
                self.hundred_hour_analysis()
            elif mode == 'daily':
                self.hundred_day_analysis()
            elif mode == 'summary':
                self.send_summary()
            else:
                raise ValueError(f'Unknown analysis: {mode}')
        elif action == 'check':
            self.single_daily_analysis(command['symbol'], command['type'], ask_ai=True)
        elif action == 'ai':
            self.ai_prompt(command['symbol'], command['type'])
        elif action in ('ai_decision', 'ai_plan'):
            self.ai_api(command['symbol'], command['type'], action[3:])
        elif action == 'ai_plan_save':
            self.save_ai_plan(command['key'], command['save'])
        elif action == 'reload_transactions':
            self.load_config_items()
            func.pushover('Transactions were reloaded in the main program.')
        elif action == 'events':
            # after /event with tomorrow's date: reminder and analysis immediately
            self.event_reminders()
        elif action in ('plan_alerts', 'value_alerts'):
            # after /plan_alert: check the new plan right away (zone or target may already be reached)
            self.plan_alert_check()
        elif action == 'reload_assets':
            # after /smart_alert: apply the changed alarm flags from assets.ini
            self.load_config_items()
            names = [item.search for item in self.crypto_items_hd + self.stock_items_hd]
            func.pushover(f"Alarm analysis reloaded ({len(names)} assets): {', '.join(names)}")
        elif action == 'portfolio':
            self.send_portfolio()
        elif action == 'status':
            self.send_status()
        elif action == 'restart_main':
            return True
        else:
            raise ValueError(f'Unknown command: {action}')
        return False


def run_program():
    # Relative paths should also work when started from another folder.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    old_manager = None
    while True:
        bot_process = None
        manager = None
        restart = False
        notify_q, command_q = mp.Queue(), mp.Queue()
        func.set_message_queue(notify_q)
        try:
            manager = CryptoStockManager()
            # Clear start line: which version runs on which device (PC or Raspberry Pi)?
            func.log_print(f"CaSa {VERSION} {'restarted (/restart)' if old_manager else 'started'} on "
                           f"{socket.gethostname()} ({platform.system()} {platform.release()}, "
                           f"Python {platform.python_version()}), test mode {'on' if manager.test_mode == 1 else 'off'}, "
                           f"CoinGecko key {'yes' if func.coingecko_api_key else 'no'}, "
                           f"Claude API {'yes' if func.claude_available() else 'no'}")
            # A restart via /restart must not send known alarms again.
            if old_manager is not None:
                manager.crypto_items_hh, manager.crypto_items_hd = old_manager.crypto_items_hh, old_manager.crypto_items_hd
                manager.stock_items_hh, manager.stock_items_hd = old_manager.stock_items_hh, old_manager.stock_items_hd
                manager.events_done = old_manager.events_done   # no second reminder for the same day
                # Daily results stay: summary, P&L chart and alarm journal work right after the restart
                manager.daily_prices = old_manager.daily_prices
                manager.df_crypto_hd, manager.df_stock_hd = old_manager.df_crypto_hd, old_manager.df_stock_hd
                manager.last_daily_update = old_manager.last_daily_update
                manager.missing_held_notified = old_manager.missing_held_notified   # no second notice
                manager.plan_stop_checked = old_manager.plan_stop_checked
                manager.plan_zone_sent = old_manager.plan_zone_sent
            manager.load_config_items()
            telegram_bot._load_config()  # detect a missing bot configuration before the start
            bot_process = mp.Process(target=telegram_bot.run_standalone,
                                     args=(notify_q, command_q), daemon=True)
            bot_process.start()
            manager.startup_message()
            schedule.clear()
            if manager.test_mode == 1:
                manager.hundred_hour_analysis()
                manager.hundred_day_analysis()
                manager.send_summary()
            else:
                schedule.every().hour.do(manager.hundred_hour_analysis)
                schedule.every().day.at(MORNING_ANALYSIS).do(manager.hundred_day_analysis)
                schedule.every().day.at(NOON_ANALYSIS).do(manager.hundred_day_analysis)
                schedule.every().day.at(EVENING_SUMMARY).do(manager.send_summary)
                if time.strftime('%H:%M') >= NOON_ANALYSIS:
                    manager.hundred_day_analysis()
            # Events from events.ini: checked daily, also in test mode
            schedule.every().day.at(EVENT_REMINDER).do(manager.event_reminders)
            if time.strftime('%H:%M') >= EVENT_REMINDER:
                manager.event_reminders()   # start after 08:00: catch up on today's reminders
            # Plan alerts from planalert.ini: checked every hour and once at the start, also in test mode
            schedule.every().hour.do(manager.plan_alert_check)
            manager.plan_alert_check()
            # Health check every 6 hours, only in logfile.txt (not sent to Telegram)
            manager.bot_alive = bot_process.is_alive
            schedule.every(6).hours.do(manager.health_check)
            while not restart:
                if not bot_process.is_alive():
                    raise RuntimeError(f'Telegram bot has stopped (exit code {bot_process.exitcode}). '
                                       'The reason is in the log lines before.')
                # Queue.empty() is not reliable with several processes.
                for _ in range(20):
                    try:
                        command = command_q.get_nowait()
                    except Empty:
                        break
                    try:
                        restart = manager.handle_command(command)
                    except Exception as error:
                        func.log_print(f"Command {command.get('cmd')} failed:\n{traceback.format_exc()}", 'ERROR')
                        func.pushover(f'Command failed: {error}')
                    if restart:
                        break
                if not restart:
                    schedule.run_pending()
                    time.sleep(0.5)
        except KeyboardInterrupt:
            func.log_print('Program stopped.')
        except Exception as error:
            func.log_print(f'Program error: {error}\n{traceback.format_exc()}', 'ERROR')
        finally:
            schedule.clear()
            if bot_process is not None and bot_process.is_alive():
                notify_q.put(None)
                bot_process.join(timeout=10)
                if bot_process.is_alive():
                    bot_process.terminate()
                    bot_process.join(timeout=5)
            func.set_message_queue(None)
            for queue in (notify_q, command_q):
                queue.cancel_join_thread()
                queue.close()
        if not restart:
            break
        # No recursive subprocess call: the old bot has already stopped.
        old_manager = manager
        func.log_print('Program is being re-initialised.')


if __name__ == '__main__':
    mp.freeze_support()
    run_program()
