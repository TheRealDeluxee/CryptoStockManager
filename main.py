"""CaSa: load the configuration, schedule analyses and handle Telegram commands.

Start: python main.py
The actual calculations stay in classes.py and func.py.
"""
import configparser
import copy
import datetime
import html
import math
import multiprocessing as mp
import os
import time
import uuid
from queue import Empty

import pandas as pd
import schedule
import classes as cl
import func
import telegram_bot

# Program version (Semantic Versioning: MAJOR.MINOR.PATCH, see CHANGELOG.md).
# Kept in the code instead of config.ini, because config.ini is private and not part of Git.
VERSION = "0.1.0"
GITHUB_URL = "https://github.com/TheRealDeluxee/CryptoStockManager"

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
- RSI14: {rsi}
- MACD (EMA12/26): {macd}
- Bollinger (20,2): {bollinger}
- Trend: {trend}
- Short term: {short_term}
- Bottom formation: {bottom}
- Top formation: {top}
- Market phase {name}: {phase}
- Market phase {market_name}: {market}
- Program score: {score} (-18 to +18, positive = rather buy)

B. News and dates
Web search (if unavailable, say so): news of the last 14 days, confirmed dates of the next 4 weeks; older figures only as dated baseline.
{research}

C. Rules
- Check broadly, report selectively: only what supports, contradicts or changes risk or entry/exit; minor but relevant points in one sentence; merge identical signals.
- Technicals only from part A; news with date and source. Missing essentials: write exactly "not verifiable" plus what is missing.
- Price levels critically: a period low is only a possible support, a low RSI only a possible recovery; a bottom needs an observable price reaction.
- Trading plan: with a falling price buy only after a confirmed bottom, staggered near support; set the exit below support in advance (does a daily close count?). Bearish phase (asset or market): small tranches only; trend change only above SMA200.
- After a strong rise wait for a pullback; partial profits at resistance; limit position size. {tax}Max. three price zones, few numbers, no disclaimers.

D. Output
1. Decision in exactly three sentences: (1) {decision}, both market phases, main reason. (2) Support/resistance in euro; bottom confirmed? Top formation only if decisive. (3) Score 0-10 = (program score + 18) / 36 x 10; risk and confidence (high/medium/low).
2. Decision drivers: max. four points (technicals, news, valuation): what is proven, what it means for me; assumptions or data gaps only if decisive.
3. Trading plan table (scenario | observable signal | my action):
{plan}
4. Strongest counterargument (max. two sentences) and what would strengthen it.
5. Max. four confirmed dates (next 4 weeks) as table: date and German time, event, relevance, source.
6. Conclusion in max. four simple sentences: what I do now, what I wait for, what triggers or rules out {action}; max. two price levels, no score or abbreviations.
"""


class CryptoStockManager:
    def __init__(self, config_file_path="config.ini", output_dir="plots"):
        self.output_dir = output_dir
        func.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        config = configparser.ConfigParser()
        if not config.read(config_file_path, encoding="utf-8-sig"):
            raise FileNotFoundError(f"{config_file_path} not found.")
        self.version = VERSION   # an old [version] section in config.ini is ignored
        self.test_mode = config.getint('alarm', 'test_mode')
        func.one_day_price_change = config.getfloat('alarm', 'one_day_price_change')
        func.seven_day_price_change = config.getfloat('alarm', 'seven_day_price_change')
        func.one_day_profit_limit = config.getfloat('alarm', 'one_day_profit_limit')
        func.previous_alarm_change = config.getfloat('alarm', 'previous_alarm_change')
        limits = (func.one_day_price_change, func.seven_day_price_change,
                  func.one_day_profit_limit, func.previous_alarm_change)
        if not all(math.isfinite(value) for value in limits):
            raise ValueError('Alarm values must be finite numbers.')
        if self.test_mode not in (0, 1):
            raise ValueError('test_mode must be 0 or 1.')
        if func.seven_day_price_change < 0 or func.previous_alarm_change < 0:
            raise ValueError('Alarm thresholds must not be negative.')
        self.stock_update = False
        self.crypto_items_hh = []
        self.crypto_items_hd = []
        self.stock_items_hh = []
        self.stock_items_hd = []
        cols = ['Name', 'Piece [€]', '7d P&L [%]', '1d P&L [%]', 'Rating']
        self.df_crypto_hd = pd.DataFrame(columns=cols)
        self.df_stock_hd = pd.DataFrame(columns=cols)
        self.df_crypto_hh = pd.DataFrame(columns=cols)
        self.df_stock_hh = pd.DataFrame(columns=cols)
        self.last_daily_update = None
        self.events_done = set()   # events already reminded today (in memory only)

    def load_config_items(self):
        """Build the new lists completely first, then replace the old ones."""
        all_entries = cl.read_assets()
        entries = cl.read_assets(alarm_only=True)   # only assets with 1 are part of the alarm analysis
        rows = cl.read_transactions()
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
        for row in rows:
            key = (row['type'], row['symbol'])
            if key not in all_entries:
                func.log_print(f"Inactive transaction: {row['symbol']} is missing in assets.ini and is not analysed.")
            elif key not in entries and row['action'] != 'watchlist':
                func.log_print(f"Note: {row['symbol']} is held, but without 1 it is not part of the alarm analysis.")
        self.crypto_items_hh, self.crypto_items_hd = crypto_hh, crypto_hd
        self.stock_items_hh, self.stock_items_hd = stock_hh, stock_hd
        # Daily tables are kept: price, P&L and rating do not depend on the holdings.
        func.log_print(f"Loaded: {len(entries)} of {len(all_entries)} symbols in the alarm analysis, {len(rows)} transactions.")

    def _is_stock_trading_hours(self):
        now = time.localtime()
        # Numbers work independently of the Windows language.
        self.stock_update = self.test_mode == 1 or (now.tm_wday < 5 and 8 <= now.tm_hour < 22)

    def _analyse_items(self, items, data_type):
        results = []
        for item in items:
            try:
                df = item.get_API_data(data_type)
                df, alarms = item.process_data(df, data_type)  # df is cut to 100 values afterwards
                item.plot_send(df, data_type, alarms)
                results.append(item.get_summary(df))
            except Exception as error:
                # One failed asset must not block the others.
                func.pushover(f"Analysis for {item.symbol} ({data_type}) failed: {error}")
        return results

    def hundred_day_analysis(self):
        func.log_print('100-day analysis started')
        self._is_stock_trading_hours()
        cols = list(self.df_crypto_hd.columns)
        self.df_crypto_hd = pd.DataFrame(self._analyse_items(self.crypto_items_hd, '100d'), columns=cols)
        self.df_stock_hd = pd.DataFrame(columns=cols)
        if self.stock_update:
            self.df_stock_hd = pd.DataFrame(self._analyse_items(self.stock_items_hd, '100d'), columns=cols)
        self.last_daily_update = time.strftime('%d/%m/%Y %H:%M:%S')

    def hundred_hour_analysis(self):
        func.log_print('100-hour analysis started')
        # As before: automatic hourly analysis only for cryptocurrencies.
        cols = list(self.df_crypto_hh.columns)
        self.df_crypto_hh = pd.DataFrame(self._analyse_items(self.crypto_items_hh, '100h'), columns=cols)

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
        df = func.cut_to_window(full)
        item.state = func.calc_state_fuctions(item.state, df)
        details = func.analysis_details(df)
        return name, symbol, item, df, full, sr_zones, details

    def single_daily_analysis(self, symbol, asset_type):
        name, symbol, item, df, full, sr_zones, details = self._daily_analysis(symbol, asset_type)
        # Telegram HTML as for the alarms (bold/italic/monospace)
        message = (f"<b>{html.escape(name)} ({html.escape(symbol)}) — daily analysis (100d)</b>\n"
                   f"Manual request without alarm\n\n" + details['message'])
        if not item.state.watch_list:
            message += "\n\n" + html.escape(f"Amount >1 year: {item.state.amount_older_than_one_year:.2f} "
                                           f"({item.state.amount_older_than_one_year_pct:.2f} %)")
        image_name = 'manual_' + uuid.uuid4().hex
        func.plot_and_save(df, image_name, '100d', item.state.zero_line, sr_zones)
        func.pushover_image(image_name, message, delete_after_send=True, html_format=True)

    def ai_prompt(self, symbol, asset_type):
        """/ai: fill the prompt for ChatGPT/Claude from ai_prompt.txt and send it as copyable text."""
        name, symbol, item, df, full, sr_zones, details = self._daily_analysis(symbol, asset_type)
        if not os.path.exists(AI_PROMPT_FILE):
            with open(AI_PROMPT_FILE, 'w', encoding='utf-8') as file:
                file.write(AI_PROMPT_TEMPLATE)
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
            groups = {'yes': [], 'no': [], 'not assessable (no recent extreme)': []}
            for key, label in names:
                result = 'not assessable (no recent extreme)' if signals[key] is None else 'yes' if signals[key] else 'no'
                groups[result].append(label)
            parts = [f"{result}: {', '.join(labels)}" for result, labels in groups.items() if labels]
            return f"{len(groups['yes'])}/{len(names)} – {'; '.join(parts)}"

        bottom_text = formation_text(bottom, (('folge', 'higher lows'), ('divergenz', 'bullish RSI divergence'),
                                              ('doppel', 'double bottom'), ('volumen', 'selling climax at low'),
                                              ('bestaetigung', 'price above EMA12')))
        top_text = formation_text(top, (('folge', 'lower highs'), ('divergenz', 'bearish RSI divergence'),
                                        ('doppel', 'double top'), ('volumen', 'buying climax at high'),
                                        ('bestaetigung', 'price below EMA12')))

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

        high, low = full['Price'].max(), full['Price'].min()
        if asset_type == 'stock':
            research = ("- Earnings/outlook: last revenue, guidance, consensus, next confirmed date (German time); "
                        "separate actuals/guidance/estimates; which outlook statement would change the call.\n"
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
            'score': str(details['score']),
            'rsi': f"{df['RSI14'].iloc[-1]:.0f} ({details['texts']['rsi']})",
            'macd': macd, 'bollinger': bollinger,
            'trend': details['texts']['trend'], 'short_term': details['texts']['short'],
            'bottom': bottom_text, 'top': top_text, 'research': research,
            'phase': phase, 'market_name': market_name, 'market': market,
        }
        # Simple replace instead of format(): other {brackets} in the file do no harm this way.
        for key, value in values.items():
            question = question.replace('{' + key + '}', value)
        question = question.strip()
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

    def event_reminders(self):
        """Today's events from events.ini: send a reminder, then a daily analysis like /check."""
        try:
            events = cl.read_events()
        except (OSError, ValueError, configparser.Error) as error:
            func.pushover(f'events.ini could not be read: {error}')
            return
        today = datetime.date.today()
        for event in events:
            if event['date'] != today or event['key'] in self.events_done:
                continue
            self.events_done.add(event['key'])
            text = f"\n{html.escape(event['text'])}" if event['text'] else ''
            func.pushover(f"<b>Event today: {html.escape(event['name'])}</b>{text}\nDaily analysis follows.",
                          priority=1, html_format=True)
            try:
                self.single_daily_analysis(event['symbol'], event['type'])
            except Exception as error:
                func.pushover(f"Daily analysis for {event['name']} failed: {error}")

    def startup_message(self):
        """Short start message: version, assets in the alarm analysis, next dates."""
        crypto, stock = len(self.crypto_items_hd), len(self.stock_items_hd)
        lines = [f"<b>CaSa started</b> · version {html.escape(str(self.version))}",
                 f"{crypto + stock} assets in the alarm analysis ({crypto} crypto, {stock} stocks)"]
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
        try:
            upcoming = [event for event in cl.read_events() if event['date'] >= datetime.date.today()]
        except (OSError, ValueError, configparser.Error) as error:
            lines.append(f"events.ini faulty: {html.escape(str(error))}")
        else:
            if upcoming:
                event = upcoming[0]
                text = f" – {event['text']}" if event['text'] else ''
                lines.append(html.escape(f"Next event: {event['date'].strftime('%d/%m/%Y')} {event['name']}{text}"))
        lines.append("All commands: /help")
        lines.append(f'<a href="{GITHUB_URL}">CaSa on GitHub</a>')
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
            self.single_daily_analysis(command['symbol'], command['type'])
        elif action == 'ai':
            self.ai_prompt(command['symbol'], command['type'])
        elif action == 'reload_transactions':
            self.load_config_items()
            func.pushover('Transactions were reloaded in the main program.')
        elif action == 'events':
            # after /event with today's date: reminder and analysis immediately
            self.event_reminders()
        elif action == 'reload_assets':
            # after /alarm: apply the changed alarm flags from assets.ini
            self.load_config_items()
            names = [item.search for item in self.crypto_items_hd + self.stock_items_hd]
            func.pushover(f"Alarm analysis reloaded ({len(names)} assets): {', '.join(names)}")
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
            # A restart via /restart must not send known alarms again.
            if old_manager is not None:
                manager.crypto_items_hh, manager.crypto_items_hd = old_manager.crypto_items_hh, old_manager.crypto_items_hd
                manager.stock_items_hh, manager.stock_items_hd = old_manager.stock_items_hh, old_manager.stock_items_hd
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
                manager.event_reminders()   # start after 08:00: catch up on today's events
            while not restart:
                if not bot_process.is_alive():
                    raise RuntimeError('Telegram bot has stopped. Please check the log.')
                # Queue.empty() is not reliable with several processes.
                for _ in range(20):
                    try:
                        command = command_q.get_nowait()
                    except Empty:
                        break
                    try:
                        restart = manager.handle_command(command)
                    except Exception as error:
                        func.pushover(f'Command failed: {error}')
                    if restart:
                        break
                if not restart:
                    schedule.run_pending()
                    time.sleep(0.5)
        except KeyboardInterrupt:
            func.log_print('Program stopped.')
        except Exception as error:
            func.log_print(f'Program error: {error}')
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
