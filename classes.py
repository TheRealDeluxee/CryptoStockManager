import configparser
import csv
import math
import os
import tempfile
import uuid

import func
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class GroupedState:
    # current day values
    current_balance_eur: float = 0
    current_profit_pct: float = 0
    current_profit: float = 0
    current_one_day_price_change_pct: float = 0
    current_seven_days_slope_pct: float = 0

    # yesterday values
    yesterday_balance_eur: float = 0
    yesterday_profit_pct: float = 0
    yesterday_profit: float = 0
    yesterday_one_day_price_change_pct: float = 0
    yesterday_seven_days_slope_pct: float = 0
    yesterday_EMA_diff_pct: float = 0

    # general
    buy_balance_eur: float = 0
    sell_money: float = 0
    amount_crypto_stock: float = 0
    score: float = 0
    tech_indicators: Dict[str, Any] = field(default_factory=dict)

    # EMA / indicators
    current_EMA_diff_pct: float = 0
    fgi: str = ''
    promt: str = ''

    # alarms / bookkeeping
    seven_day_alarms: List[Any] = field(default_factory=list)
    watch_list: bool = True
    alarm_prev: Dict[str, Any] = field(default_factory=dict)
    # buy history and age
    df_buy_history: Any = field(default_factory=lambda: pd.DataFrame(columns=['Date', 'Cost [€]', 'Amount']))
    amount_older_than_one_year: float = 0
    amount_older_than_one_year_pct: float = 0
    # derived values used in plotting
    zero_line: Optional[float] = None
    sr_zones: Any = None  # (resistance, support) from the long history
    pattern: Any = None   # chart pattern (func.chart_pattern) or None
    score_rank: Any = None   # score 1-10 as rank (func.score_rank) or None


class CryptoStock:

    def __init__(self, symbol, search, crypto_or_stock):
        self.symbol = symbol
        self.search = search
        self.type = crypto_or_stock
        self.params_crypto_stock = ''
        self.test_mode = False

        # Grouped state container for many related attributes
        self.state = GroupedState()
        

    def buy(self, fiat_eur, amount, date):
        # Validate first, then book: an error must not change the holdings.
        fiat_eur, amount = valid_numbers(fiat_eur, amount)
        parsed_date = datetime.strptime(str(date).strip(), "%d/%m/%Y")
        self.state.df_buy_history.loc[len(self.state.df_buy_history)] = [parsed_date, fiat_eur, amount]
        self.state.buy_balance_eur += fiat_eur
        self.state.amount_crypto_stock += amount
        self.state.watch_list = False


    def sell(self, fiat_eur, amount, date=None):
        """
        FIFO sell: reduces holdings and cost basis based on the buy history.

        - sell_money += fiat_eur
        - amount_crypto_stock -= amount (sales larger than the holdings are rejected)
        - df_buy_history is reduced FIFO (Amount + Cost [€])
        - buy_balance_eur is reduced by the sold FIFO cost basis
        - avoids the pandas FutureWarning by explicit float casting
        """
        fiat_eur, amount = valid_numbers(fiat_eur, amount)
        available = self.state.amount_crypto_stock
        if amount > available and not math.isclose(amount, available, rel_tol=1e-10, abs_tol=0.0):
            raise ValueError(f"Cannot sell {amount}: holdings are only {available}.")
        amount = min(amount, available)
        if amount <= 0:
            raise ValueError("No holdings to sell.")
        df = self.state.df_buy_history
        if df.empty or not math.isclose(float(df['Amount'].sum()), available, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError("Buy history and holdings do not match.")

        # FIFO: the oldest open buys are sold first.
        df = df.sort_values('Date', kind='stable').reset_index(drop=True)
        self.state.sell_money += fiat_eur
        self.state.amount_crypto_stock -= amount
        self.state.watch_list = self.state.amount_crypto_stock <= 0

        # --- IMPORTANT: fix dtypes (prevents FutureWarning) ---
        # (could also be done once in buy(); here it is robust)
        df = df.copy()
        if "Amount" in df.columns:
            df["Amount"] = df["Amount"].astype("float64")
        if "Cost [€]" in df.columns:
            df["Cost [€]"] = df["Cost [€]"].astype("float64")

        remaining = amount
        rows_to_drop = []

        for i, row in df.iterrows():
            if remaining <= 0:
                break

            row_amount = float(row["Amount"])
            row_cost = float(row["Cost [€]"])

            if row_amount <= 0:
                rows_to_drop.append(i)
                continue

            take = min(row_amount, remaining)
            cost_taken = row_cost * (take / row_amount)

            # reduce open cost basis
            self.state.buy_balance_eur -= cost_taken

            remaining -= take

            leftover_amount = row_amount - take
            leftover_cost = row_cost - cost_taken

            if leftover_amount <= 0 or math.isclose(take, row_amount, rel_tol=1e-12, abs_tol=0.0):
                rows_to_drop.append(i)
            else:
                df.at[i, "Amount"] = float(leftover_amount)
                df.at[i, "Cost [€]"] = float(leftover_cost)

        # clean up history
        if rows_to_drop:
            df = df.drop(index=rows_to_drop).reset_index(drop=True)

        self.state.df_buy_history = df

        # numerical stability
        if self.state.buy_balance_eur < 0 and abs(self.state.buy_balance_eur) < 1e-6:
            self.state.buy_balance_eur = 0.0

    def reset_alarm_duplicates(self):
        self.state.seven_day_alarms.clear()
        self.state.alarm_prev.clear() 

    def activate_test_mode(self):
        self.test_mode = True

    def get_API_data(self, data_type):
        if data_type not in ('100d', '100h') or self.type not in ('crypto', 'stock'):
            raise ValueError('Unknown asset type or analysis period.')

        # Load more history than displayed (indicators, support/resistance).
        # Cutting to 100 values happens only after the indicator calculation.
        # CoinGecko: 2-90 days = hourly, more than 90 days = daily -> resolution stays the same.
        # Request a bit more and then cut to the same number of values:
        # crypto and stocks both get 250 daily or 450 hourly values.
        history = {'100d': 250, '100h': 450}
        if self.type == 'crypto':
            if data_type == '100d':
                self.params_crypto_stock = {'vs_currency': 'eur', 'days': '260'}
            elif data_type == '100h':
                self.params_crypto_stock = {'vs_currency': 'eur', 'days': '20'}
            df = func.get_crypto(self.symbol,self.params_crypto_stock)
        elif self.type == 'stock':
            if data_type == '100d':
                self.params_crypto_stock = "2y"
            elif data_type == '100h':
                self.params_crypto_stock = "6mo"
            df = func.get_stock(self.symbol,self.params_crypto_stock)
        if df is not None:
            df = df.tail(history[data_type]).reset_index(drop=True)
        return df

    def process_data(self, df, data_type):

        if df is None or df.empty or len(df) < 26:
            raise ValueError(f"Not enough price data for {self.symbol} (at least 26 values required).")

        df = func.calc_indicator_fuctions(df)
        self.state.sr_zones = func.support_resistance(df)  # before cutting: needs the long history
        self.state.pattern = func.chart_pattern(df)
        self.state.score_rank = func.score_rank(df)   # 1-10 vs. the last 100 values, before cutting
        df = func.cut_to_window(df)
        self.state = func.calc_state_fuctions(self.state, df)
        alarms, self.state.tech_indicators, self.state.score = func.alarm(df,self.search,self.state.watch_list, self.state.current_profit_pct, self.state.amount_older_than_one_year, self.state.amount_older_than_one_year_pct, data_type, self.state.pattern, self.state.score_rank, self.state.sr_zones, self.type, yesterday_profit_pct=self.state.yesterday_profit_pct)
        alarms_filter = func.filter(alarms, self.state)

        # Return the cut data so that chart and summary use the same 100 values.
        return df, alarms_filter

    def plot_send(self, df, data_type, alarms_filter):
        if not alarms_filter:
            return
        # A separate image per message prevents it from being overwritten later.
        for info in alarms_filter.values():
            image_name = "alarm_" + uuid.uuid4().hex
            func.plot_and_save(df, image_name, data_type, self.state.zero_line, self.state.sr_zones, self.state.pattern)
            # one tap below the alarm/info: AI prompt, with a Claude API key also AI decision (API) and AI plan (API)
            button = func.ai_buttons(self.type, self.symbol)
            func.pushover_image(image_name, info['msg'], info['priority'], delete_after_send=True, html_format=True,
                                buttons=button)

    def get_summary(self, df):
        return {
            'Name': self.search,
            'Piece [€]': round(float(df['Price'].iloc[-1]), 2),
            '7d P&L [%]': round(self.state.current_seven_days_slope_pct, 2),
            '1d P&L [%]': round(self.state.current_one_day_price_change_pct, 2),
            'Rating': self.state.score_rank or 0,   # score 1-10 (0 = not available)
            # own position for the daily P&L message (0 = not held)
            'Type': self.type,
            'Value [€]': round(self.state.current_balance_eur, 2),
            '1d P&L [€]': round(self.state.current_balance_eur - self.state.yesterday_balance_eur, 2),
            'Total P&L [€]': round(self.state.current_profit, 2),
            'Total P&L [%]': round(self.state.current_profit_pct, 2),
        }


# Shared file checks: the same rules apply when entering and when loading.
CSV_HEADER = ["type", "symbol", "name", "action", "amount", "qty", "date"]


def valid_numbers(amount, qty):
    amount = float(str(amount).replace(',', '.'))
    qty = float(str(qty).replace(',', '.'))
    if not math.isfinite(amount) or not math.isfinite(qty) or amount <= 0 or qty <= 0:
        raise ValueError("Amount and quantity must be finite numbers greater than zero.")
    return amount, qty


def read_assets(path="assets.ini", alarm_only=False):
    """Format per line: symbol = Name; crypto|stock; 1
    The 1 at the end means: evaluated in the automatic alarm analysis.
    Without 1 (or with 0) only available for /check and /ai.
    alarm_only=True returns only the assets with 1."""
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path, encoding="utf-8-sig"):
        raise FileNotFoundError(f"{path} not found.")
    entries = {}
    for section, asset_type in (("crypto", "crypto"), ("stocks", "stock")):
        if not config.has_section(section):
            continue
        for symbol, value in config.items(section):
            symbol = symbol.strip().lower()
            if not symbol or any(char in symbol for char in '/\\:*?"<>|'):
                raise ValueError(f"Invalid symbol in [{section}]: {symbol}")
            parts = [part.strip() for part in value.split(';')]
            alarm = parts[2] if len(parts) > 2 else ''
            if alarm not in ('', '0', '1'):
                raise ValueError(f"Invalid alarm flag for {symbol}: '{alarm}' (allowed: 1, 0 or empty)")
            if alarm_only and alarm != '1':
                continue
            entries[(asset_type, symbol)] = parts[0] or symbol
    if not entries and not alarm_only:
        raise ValueError(f"{path} contains no active symbols.")
    return entries


def take_asset(words, assets, types=('crypto', 'stock')):
    """Find an asset at the start of a bot command by symbol or name (e.g. RHM.DE, Rheinmetall, "Trane Technologies").
    Order: symbol or full name; then the beginning of a name ("Trane" -> Trane Technologies).
    The longest fitting word group wins, so names with spaces work too.
    Returns (asset type, symbol, name, remaining words); ValueError if nothing or several assets fit."""
    candidates = [key for key in assets if key[0] in types]
    for exact in (True, False):
        for count in range(len(words), 0, -1):
            text = ' '.join(words[:count]).lower()
            if exact:
                found = [key for key in candidates if text in (key[1], ' '.join(assets[key].lower().split()))]
            elif len(text) >= 3:   # at least 3 letters, so that a typo does not hit a random asset
                found = [key for key in candidates if ' '.join(assets[key].lower().split()).startswith(text)]
            else:
                found = []
            if len(found) > 1:
                names = ', '.join(f"{assets[key]} ({key[1].upper() if key[0] == 'stock' else key[1]})"
                                  for key in found)
                raise ValueError(f"'{' '.join(words[:count])}' fits several assets: {names}. "
                                 'Please use the symbol or add the type (crypto|stock).')
            if found:
                asset_type, symbol = found[0]
                return asset_type, symbol, assets[found[0]], list(words[count:])
    text = words[0] if words else ''
    section = ' or '.join('[crypto]' if t == 'crypto' else '[stocks]' for t in types)
    raise ValueError(f"'{text}' is not in assets.ini {section} (neither as symbol nor as name).")


def read_events(path="events.ini", assets_path="assets.ini"):
    """Events from events.ini: name = symbol; DD/MM/YYYY; description.
    Type (crypto/stock) and name come from assets.ini. Missing file = no events."""
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path, encoding="utf-8-sig") or not config.has_section('events'):
        return []
    assets = read_assets(assets_path)
    events = []
    for key, value in config.items('events'):
        parts = [part.strip() for part in value.split(';', 2)]
        if len(parts) < 2:
            raise ValueError(f"{path}, event '{key}': format is symbol; DD/MM/YYYY; description")
        symbol = parts[0].lower()
        try:
            date = datetime.strptime(parts[1].replace('.', '/'), '%d/%m/%Y').date()
        except ValueError:
            raise ValueError(f"{path}, event '{key}': invalid date '{parts[1]}' (format DD/MM/YYYY)") from None
        matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in assets]
        if len(matches) != 1:
            problem = 'not in assets.ini' if not matches else 'listed as crypto and stock in assets.ini'
            raise ValueError(f"{path}, event '{key}': {parts[0]} is {problem}.")
        events.append({'key': key, 'symbol': symbol, 'type': matches[0], 'name': assets[(matches[0], symbol)],
                       'date': date, 'text': parts[2] if len(parts) > 2 else ''})
    return sorted(events, key=lambda event: event['date'])


def add_event(symbol, date_text, text, path="events.ini", assets_path="assets.ini"):
    """Append a new event at the end of events.ini (for /event). Returns (key, asset name, date)."""
    symbol = symbol.strip().lower()
    assets = read_assets(assets_path)
    matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in assets]
    if len(matches) != 1:
        raise ValueError(f"{symbol} is {'not in assets.ini' if not matches else 'listed as crypto and stock in assets.ini'}.")
    try:
        date = datetime.strptime(date_text.strip().replace('.', '/'), '%d/%m/%Y').date()
    except ValueError:
        raise ValueError(f"Invalid date '{date_text}' (format DD/MM/YYYY).") from None
    if date < datetime.now().date():
        raise ValueError('The date is in the past.')
    text = ' '.join(text.split())   # no line breaks in the ini

    lines = []
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as file:
            lines = file.read().splitlines()
    if not any(line.strip().lower() == '[events]' for line in lines):
        lines.append('[events]')
    existing = {line.split('=', 1)[0].strip().lower() for line in lines if '=' in line and not line.lstrip().startswith('#')}
    key = f"{symbol.replace('.', '_')}_{date:%Y%m%d}"
    number = 2
    while key in existing:   # name must be unique
        key = f"{symbol.replace('.', '_')}_{date:%Y%m%d}_{number}"
        number += 1
    stock_symbol = symbol.upper() if matches[0] == 'stock' else symbol
    lines.append(f"{key:<24} = {stock_symbol}; {date:%d/%m/%Y}; {text}".rstrip('; '))

    # Write a complete new file first, then replace it (as for assets.ini)
    directory = os.path.dirname(os.path.abspath(path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=directory, suffix='.tmp', delete=False) as file:
            temp_path = file.name
            file.write('\n'.join(lines) + '\n')
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    return key, assets[(matches[0], symbol)], date


def _write_lines(path, lines):
    # Write a complete new file first, then replace it: a crash never leaves a half-written file
    directory = os.path.dirname(os.path.abspath(path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=directory, suffix='.tmp', delete=False) as file:
            temp_path = file.name
            file.write('\n'.join(lines) + '\n')
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _delete_ini_entry(path, section_name, key, what):
    """Remove one entry (one line) from a section. Comments and all other entries stay as they are."""
    if not os.path.exists(path):
        raise ValueError(f"{path} not found.")
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    section, index = None, None
    for number, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            section = stripped[1:-1].strip().lower()
            continue
        if section == section_name and '=' in line and not stripped.startswith(('#', ';')) \
                and line.split('=', 1)[0].strip().lower() == key.strip().lower():
            index = number
            break
    if index is None:
        raise ValueError(f"{what} '{key}' is not in {path} (already deleted?).")
    removed = lines.pop(index)
    _write_lines(path, lines)
    return removed


def delete_event(key, path="events.ini"):
    """Remove one event from events.ini (for /event). Returns the removed line."""
    return _delete_ini_entry(path, 'events', key, 'Event')


# ── Plan alerts (planalert.ini, before 0.1.12 valuealert.ini) ──────────────
PLAN_FILE = 'planalert.ini'
PLAN_HEADER = """# CaSa plan alerts: simple trading plans with messages, one plan per asset
#
# [SYMBOL]            symbol as in assets.ini (e.g. TSFA.F or bitcoin)
# support   = 312.15  buy zone: the price falls to the support or up to max_entry % above it
# max_entry = 2       buy only up to 2 % above the support (0 = only at or below the support)
# target    = 358.93  take partial profits at or above this price
# stop      = 312.15  the plan fails at a daily close below this price
# note      = ...     your plan in words (tranche size, conditions, news); shown in every message
# start     = DD/MM/YYYY  set by CaSa: daily closes from this day on count for the stop
# status    = ...     set by CaSa: waiting, buy_zone, target or ended (with date and reason)
# Optional: support or target (at least one of them), stop. Prices in EUR, from the same source as the analysis.
#
# Whether you hold the asset (transactions.csv) decides the message:
#   buy zone: not held = "Buy zone: check the conditions in your note", held = "Add-on zone (retest)"
#   target:   held = "Take partial profits" (not held: no message)
#   stop:     not held = "Setup cancelled", held = "Exit: sell, do not average down"; the plan ends
# Buy zone and target are checked every hour (stocks Monday to Friday 08:00-22:00), each message once until
# the price leaves the zone again, at most once per day and plan. The stop is checked with the daily closing prices.
# Plans can also be shown, added and deleted in Telegram with /plan_alert."""
PLAN_KEYS = ('support', 'max_entry', 'target', 'stop', 'note', 'start', 'status')
PLAN_STATUS = ('waiting', 'buy_zone', 'target', 'ended')


def parse_price(text):
    """'320.82', '320,82', '€320.82' or '1.234,50' -> float."""
    cleaned = text.strip().replace('€', '').replace('EUR', '').strip()
    if ',' in cleaned and '.' in cleaned:   # 1.234,50 or 1,234.50
        cleaned = cleaned.replace('.', '').replace(',', '.') if cleaned.rfind(',') > cleaned.rfind('.') else cleaned.replace(',', '')
    else:
        cleaned = cleaned.replace(',', '.')
    try:
        price = float(cleaned)
    except ValueError:
        raise ValueError(f"Invalid price '{text}' (e.g. 320.82).") from None
    if not math.isfinite(price) or price <= 0:
        raise ValueError('The price must be greater than 0.')
    return price


def parse_percent(text):
    """'2', '2 %', '2,5%' -> float between 0 and 50."""
    cleaned = str(text).strip().replace('%', '').replace(',', '.').strip()
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"Invalid percentage '{text}' (e.g. 2).") from None
    if not math.isfinite(value) or not 0 <= value <= 50:
        raise ValueError('max_entry must be between 0 and 50 (%).')
    return value


def _asset_type(symbol, assets, path):
    matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in assets]
    if len(matches) != 1:
        problem = 'not in assets.ini' if not matches else 'listed as crypto and stock in assets.ini'
        raise ValueError(f"{path}: {symbol} is {problem}.")
    return matches[0]


def check_plan(support, max_entry, target, stop):
    """Plausibility of the four plan numbers (None = not used). ValueError with the reason."""
    if support is None and target is None:
        raise ValueError('A plan needs a support or a target.')
    if support is not None and target is not None and target <= support * (1 + max_entry / 100):
        raise ValueError('The target must be above the buy zone (support + max_entry %).')
    if stop is not None and target is not None and stop >= target:
        raise ValueError('The stop must be below the target.')
    if stop is not None and support is not None and stop > support * (1 + max_entry / 100):
        raise ValueError('The stop must not be above the buy zone.')


def read_plans(path=PLAN_FILE, assets_path="assets.ini"):
    """Plans from planalert.ini, one per asset. Type (crypto/stock) and name come from assets.ini.
    Missing file = no plans."""
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path, encoding="utf-8-sig"):
        return []
    assets = read_assets(assets_path)
    plans = []
    for section in config.sections():
        symbol = section.strip().lower()
        values = config[section]
        try:
            asset_type = _asset_type(symbol, assets, path)
            unknown = [key for key in values if key not in PLAN_KEYS]
            if unknown:
                raise ValueError(f"unknown setting {', '.join(unknown)}")
            number = lambda key: parse_price(values[key]) if values.get(key, '').strip() else None
            support, target, stop = number('support'), number('target'), number('stop')
            max_entry = parse_percent(values.get('max_entry', '').strip() or '0')
            check_plan(support, max_entry, target, stop)
            start = values.get('start', '').strip()
            start = datetime.strptime(start, '%d/%m/%Y').date() if start else None
            status_text = ' '.join(values.get('status', '').split()) or 'waiting'
            status = status_text.split()[0].rstrip(':').lower()
            if status not in PLAN_STATUS:
                raise ValueError(f"unknown status '{status}'")
        except ValueError as error:
            raise ValueError(f"{path}, plan [{section}]: {error}") from None
        plans.append({'key': section, 'symbol': symbol, 'type': asset_type, 'name': assets[(asset_type, symbol)],
                      'support': support, 'max_entry': max_entry, 'target': target, 'stop': stop,
                      'note': ' '.join(values.get('note', '').split()), 'start': start,
                      'status': status, 'status_text': status_text})
    return sorted(plans, key=lambda plan: plan['name'].lower())


def plan_zone_top(plan):
    """Upper end of the buy zone (support + max_entry %), None without support."""
    return None if plan['support'] is None else plan['support'] * (1 + plan['max_entry'] / 100)


def plan_text(plan):
    """One line with the plan numbers, e.g. 'Buy zone 312.15–318.39 € · target 358.93 € · stop 312.15 €'."""
    parts = []
    if plan['support'] is not None:
        top = plan_zone_top(plan)
        parts.append(f"buy zone {func.format_euro(plan['support'])}"
                     + (f"–{func.format_euro(top)}" if top > plan['support'] else ''))
    if plan['target'] is not None:
        parts.append(f"target {func.format_euro(plan['target'])}")
    if plan['stop'] is not None:
        parts.append(f"stop {func.format_euro(plan['stop'])} (daily close)")
    text = ' · '.join(parts)
    return text[:1].upper() + text[1:]


def _plan_sections(lines):
    """{symbol lower: (line of [SYMBOL], line after the section)} for every section in the file lines."""
    starts = [(index, line.strip()[1:-1].strip().lower()) for index, line in enumerate(lines)
              if line.strip().startswith('[') and line.strip().endswith(']')]
    ends = [index for index, _ in starts[1:]] + [len(lines)]
    return {symbol: (start, end) for (start, symbol), end in zip(starts, ends)}


def _plan_number(number):
    return '' if number is None else f"{number:.2f}" if number >= 1 else f"{number:.6g}"


def plan_lines(shown_symbol, support, max_entry, target, stop, note, start, status='waiting'):
    return [f"[{shown_symbol}]",
            f"support   = {_plan_number(support)}",
            f"max_entry = {max_entry:g}",
            f"target    = {_plan_number(target)}",
            f"stop      = {_plan_number(stop)}",
            f"note      = {' '.join(note.split())}",
            f"start     = {start:%d/%m/%Y}",
            f"status    = {status}"]


def add_plan(symbol, support, max_entry, target, stop, note, path=PLAN_FILE, assets_path="assets.ini", today=None):
    """Save a plan (for /plan_alert); an existing plan of the same asset is replaced.
    The file is created with explanations if missing. Returns (asset name, asset type, replaced)."""
    symbol = symbol.strip().lower()
    assets = read_assets(assets_path)
    asset_type = _asset_type(symbol, assets, path)
    check_plan(support, max_entry, target, stop)
    lines = []
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as file:
            lines = file.read().splitlines()
    if not lines:
        lines = PLAN_HEADER.splitlines()
    shown_symbol = symbol.upper() if asset_type == 'stock' else symbol
    new = plan_lines(shown_symbol, support, max_entry, target, stop, note, today or datetime.now().date())
    sections = _plan_sections(lines)
    replaced = symbol in sections
    if replaced:
        start, end = sections[symbol]
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        lines[start:end] = new
    else:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += [''] + new
    _write_lines(path, lines)
    return assets[(asset_type, symbol)], asset_type, replaced


def delete_plan(symbol, path=PLAN_FILE):
    """Remove the plan of one asset (whole section). Returns the removed lines."""
    if not os.path.exists(path):
        raise ValueError(f"{path} not found.")
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    sections = _plan_sections(lines)
    if symbol.lower() not in sections:
        raise ValueError(f"Plan '{symbol}' is not in {path} (already deleted?).")
    start, end = sections[symbol.lower()]
    while start > 0 and not lines[start - 1].strip():   # also the empty line before it
        start -= 1
    removed = lines[start:end]
    del lines[start:end]
    _write_lines(path, lines)
    return removed


def set_plan_value(symbol, key, value, path=PLAN_FILE):
    """Write one line of a plan (e.g. status = buy_zone, status = 'ended DD/MM/YYYY: reason', start = ...)."""
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    sections = _plan_sections(lines)
    if symbol.lower() not in sections:
        raise ValueError(f"Plan '{symbol}' is not in {path} (deleted?).")
    start, end = sections[symbol.lower()]
    line = f"{key:<9} = {' '.join(str(value).split())}"
    for index in range(start + 1, end):
        if '=' in lines[index] and lines[index].split('=', 1)[0].strip().lower() == key:
            lines[index] = line
            break
    else:
        insert = end
        while insert > start + 1 and not lines[insert - 1].strip():
            insert -= 1
        lines.insert(insert, line)
    _write_lines(path, lines)


def convert_value_alerts(old_path='valuealert.ini', path=PLAN_FILE, today=None):
    """Up to 0.1.11 the price alerts were in valuealert.ini. At the start they become plans in planalert.ini:
    'below' = support (buy zone only at or below it, max_entry 0), 'above' = target; notes are kept.
    valuealert.ini is renamed to valuealert.old.ini. Returns a list of what was done."""
    if not os.path.exists(old_path) or os.path.exists(path):
        return []
    config = configparser.ConfigParser(interpolation=None)
    config.read(old_path, encoding='utf-8-sig')
    plans = {}   # symbol as written -> values
    if config.has_section('alerts'):
        for key, value in config.items('alerts'):
            parts = [part.strip() for part in value.split(';', 3)]
            if len(parts) < 3 or parts[1].lower() not in ('below', 'above') or not parts[0]:
                continue
            try:
                price = parse_price(parts[2])
            except ValueError:
                continue
            plan = plans.setdefault(parts[0], {'support': None, 'target': None, 'notes': []})
            field_name = 'support' if parts[1].lower() == 'below' else 'target'
            if plan[field_name] is None:
                plan[field_name] = price
            else:   # second alert of the same kind: keep it as text
                plan['notes'].append(f"also {parts[1].lower()} {price:.2f}")
            if len(parts) > 3 and parts[3]:
                plan['notes'].append(parts[3])
    lines = PLAN_HEADER.splitlines()
    for symbol, plan in plans.items():
        support, target = plan['support'], plan['target']
        if support is not None and target is not None and target <= support:
            plan['notes'].insert(0, f"(old alert above {target:.2f})")
            target = None
        lines += [''] + plan_lines(symbol, support, 0, target, None, ' | '.join(plan['notes']),
                                    today or datetime.now().date())
    _write_lines(path, lines)
    os.replace(old_path, os.path.join(os.path.dirname(old_path), 'valuealert.old.ini'))
    return [f"valuealert.ini converted to planalert.ini ({len(plans)} plans), old file kept as valuealert.old.ini"]


def set_alarm(symbol, asset_type, active, path="assets.ini"):
    """Sets the alarm flag (1 = on, 0 = off) of an asset in assets.ini.
    The file is changed line by line so that comments and alignment are kept."""
    section_name = 'crypto' if asset_type == 'crypto' else 'stocks'
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    section, found = None, False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            section = stripped[1:-1].strip().lower()
            continue
        if section != section_name or stripped.startswith(('#', ';')) or '=' not in line:
            continue
        key, value = line.split('=', 1)
        if key.strip().lower() != symbol.lower():
            continue
        parts = [part.strip() for part in value.split(';')]
        while len(parts) < 3:
            parts.append(asset_type if len(parts) == 1 else '')
        parts[2] = '1' if active else '0'
        lines[index] = f"{key}= {'; '.join(parts)}"
        found = True
        break
    if not found:
        raise ValueError(f"{symbol} is not in {path} in section [{section_name}].")
    # Write a complete new file first, then replace it (as for transactions.csv)
    directory = os.path.dirname(os.path.abspath(path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=directory, suffix='.tmp', delete=False) as file:
            temp_path = file.name
            file.write('\n'.join(lines) + '\n')
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def read_transactions(path="transactions.csv"):
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file, delimiter=';')
            if reader.fieldnames != CSV_HEADER:
                raise ValueError("transactions.csv has a wrong or missing header.")
            rows = list(reader)
    except FileNotFoundError:
        return []
    for line, row in enumerate(rows, 2):
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Wrong number of columns.")
            row['type'] = row['type'].strip().lower()
            row['symbol'] = row['symbol'].strip().lower()
            row['action'] = row['action'].strip().lower()
            if row['type'] not in ('crypto', 'stock') or not row['symbol']:
                raise ValueError("Invalid type or missing symbol.")
            if row['action'] not in ('buy', 'sell', 'watchlist'):
                raise ValueError("Unknown action.")
            if row['action'] == 'watchlist':
                continue
            valid_numbers(row['amount'], row['qty'])
            row['date'] = row['date'].strip().replace('.', '/')
            parsed_date = datetime.strptime(row['date'], '%d/%m/%Y')
            if parsed_date.date() > datetime.now().date():
                raise ValueError("Transactions must not be in the future.")
        except ValueError as error:
            raise ValueError(f"transactions.csv, line {line}: {error}") from error
    return rows


def held_assets(rows):
    """Assets you currently hold according to transactions.csv (bought minus sold quantity above zero),
    as a set of (type, symbol lower)."""
    quantity = {}
    for row in rows:
        if row['action'] not in ('buy', 'sell'):
            continue
        key = (row['type'], row['symbol'].lower())
        qty = valid_numbers(row['amount'], row['qty'])[1]
        quantity[key] = quantity.get(key, 0.0) + (qty if row['action'] == 'buy' else -qty)
    return {key for key, qty in quantity.items() if qty > 1e-9}


def daily_prices(df):
    """Closing price per calendar day from an analysis DataFrame (Date, Price); time zone and time of day removed."""
    dates = pd.to_datetime(df['Date'])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(None)
    series = pd.Series(df['Price'].to_numpy(float), index=dates.dt.normalize())
    return series.groupby(level=0).last()   # CoinGecko: midnight value and the current value on the same day


def pnl_history(rows, prices, days=100, today=None):
    """Total profit/loss of the portfolio per day for the last `days` calendar days:
    value of the holdings + money from sales - money spent on buys (realized and unrealized together).
    rows: transactions (read_transactions); prices: {(type, symbol lower): daily price series}.
    Returns (Series of P&L in € per day, list of held assets without prices)."""
    today = pd.Timestamp(today or datetime.now()).normalize()
    index = pd.date_range(today - pd.Timedelta(days=days - 1), today, freq='D')
    total = pd.Series(0.0, index=index)
    missing = []
    trades = [row for row in rows if row['action'] in ('buy', 'sell')]
    for key in sorted({(row['type'], row['symbol'].lower()) for row in trades}):
        own = [row for row in trades if (row['type'], row['symbol'].lower()) == key]
        dates = pd.to_datetime([row['date'] for row in own], format='%d/%m/%Y')
        signed = [(1 if row['action'] == 'buy' else -1, *valid_numbers(row['amount'], row['qty'])) for row in own]
        trade_df = pd.DataFrame({'qty': [s * q for s, _, q in signed], 'cash': [-s * a for s, a, _ in signed]}, index=dates)
        trade_df = trade_df.groupby(level=0).sum().sort_index()
        # holdings and money flow up to each day (trades before the window count from the first day)
        qty = trade_df['qty'].cumsum().reindex(index.union(trade_df.index)).ffill().fillna(0).reindex(index)
        cash = trade_df['cash'].cumsum().reindex(index.union(trade_df.index)).ffill().fillna(0).reindex(index)
        if key not in prices or prices[key].empty:
            if qty.iloc[-1] > 1e-12:
                missing.append(key[1])
            continue
        # weekends and holidays: last known price (stocks)
        price = prices[key].reindex(index.union(prices[key].index)).ffill().reindex(index)
        value = (qty * price).where(qty.abs() > 1e-12, 0.0)
        total += (value.fillna(0) + cash).where(price.notna() | (qty.abs() <= 1e-12), np.nan)
    return total.dropna(), missing


def realized_trades(rows):
    """Realized profit/loss of every sale from transactions.csv (FIFO: the oldest buys are sold first, as
    in the position overview). Returns a list of dicts: date, type, symbol, name, amount, cost, profit."""
    trades = [row for row in rows if row['action'] in ('buy', 'sell')]
    # by date; on the same date the order in the file
    trades = sorted(trades, key=lambda row: datetime.strptime(row['date'], '%d/%m/%Y'))
    lots, sales = {}, []
    for row in trades:
        key = (row['type'], row['symbol'].lower())
        amount, qty = valid_numbers(row['amount'], row['qty'])
        queue = lots.setdefault(key, [])
        if row['action'] == 'buy':
            queue.append([qty, amount / qty])   # quantity and cost per unit
            continue
        remaining, cost = qty, 0.0
        while remaining > 1e-12 and queue:
            used = min(remaining, queue[0][0])
            cost += used * queue[0][1]
            queue[0][0] -= used
            remaining -= used
            if queue[0][0] <= 1e-12:
                queue.pop(0)
        sold = qty - remaining   # a sale larger than the holdings counts only for the held part
        if sold <= 1e-12:
            continue
        proceeds = amount * sold / qty
        sales.append({'date': pd.Timestamp(datetime.strptime(row['date'], '%d/%m/%Y')), 'type': key[0],
                      'symbol': key[1], 'name': row.get('name') or key[1], 'amount': proceeds, 'cost': cost,
                      'profit': proceeds - cost})
    return sales


def weekly_pnl(rows, prices, weeks=13, today=None):
    """Profit/loss per week (Monday to Sunday) for the last `weeks` weeks, the current week up to today:
    realized = sales of that week (FIFO, from transactions.csv); for the current week also unrealized =
    price change of the open positions since the start of the week (incl. this week's buys).
    Returns (DataFrame with index = week start and columns realized, unrealized, list of held assets
    without prices). frame.attrs['sales'] = {week start: [profit of each sale, ...]} and
    frame.attrs['open'] = [unrealized change of each asset this week, ...] for the bar parts
    (one part per sale or open asset, without names)."""
    today = pd.Timestamp(today or datetime.now()).normalize()
    this_monday = today - pd.Timedelta(days=today.weekday())
    starts = pd.date_range(end=this_monday, periods=weeks, freq='7D')
    frame = pd.DataFrame({'realized': 0.0, 'unrealized': 0.0}, index=starts)
    sales = {}   # week -> realized profit of each sale
    for sale in realized_trades(rows):
        week = sale['date'] - pd.Timedelta(days=sale['date'].weekday())
        if week in frame.index:
            frame.loc[week, 'realized'] += sale['profit']
            sales.setdefault(week, []).append(sale['profit'])
    # current week, per asset: change of its P&L since last Sunday minus its realized part = unrealized change
    sunday = this_monday - pd.Timedelta(days=1)
    days = (today - this_monday).days + 2
    open_changes, missing = [], []
    trades = [row for row in rows if row['action'] in ('buy', 'sell')]
    for key in sorted({(row['type'], row['symbol'].lower()) for row in trades}):
        own = [row for row in trades if (row['type'], row['symbol'].lower()) == key]
        total, lacking = pnl_history(own, prices, days=days, today=today)
        missing += lacking
        if total.empty or sunday not in total.index:
            continue
        realized_now = sum(sale['profit'] for sale in realized_trades(own) if sale['date'] >= this_monday)
        change = total.iloc[-1] - total[sunday] - realized_now
        if abs(change) > 0.005:
            open_changes.append(change)
    frame.loc[this_monday, 'unrealized'] = sum(open_changes)
    frame.attrs['open'] = sorted(open_changes, reverse=True)
    frame.attrs['sales'] = {week: sorted(profits, reverse=True) for week, profits in sales.items()}
    return frame, missing


# ── Alarm and info switches (config.ini [alerts]) ──────────────────────────
# Alarms: signals with an advantage in the backtest ("Potential buy/sell", recorded in the alarm journal).
# Infos: notable events without a proven forecast ("Info", for your overview).
# Value per entry: all | stock | crypto | off
ALERTS = func.ALERTS
ALERT_KINDS = func.ALERT_KINDS
alert_active = func.alert_active

ALERTS_SECTION_HEADER = [
    '[alerts]',
    '# name = <stocks>; <crypto>, each alarm, info or off (explained in the README; also /smart_alert_config)',
    '#                stocks crypto',
]
# The long explanation up to 0.1.13: replaced by the short line above at the start
_OLD_ALERTS_EXPLANATION = [
    '# Which signals CaSa sends, separately for stocks and crypto:',
    '#   name = <stocks>; <crypto>      each one of: alarm, info, off',
    '#   alarm = message "Potential buy/sell", can be marked important, counted in the alarm journal',
    '#   info  = message "Info: ...", never important, not counted in the alarm journal',
    '#   off   = no message',
    '# The comment behind each line: what the signal means and how often the price moved in the expected',
    '# direction within 20 days in the backtest (54 assets, up to 14 years); "normal" = the same on an',
    '# average day. Also changeable in Telegram with /smart_alert_config.',
]
# the same with "/journal" (before /journal became a button)
_OLDER_ALERTS_EXPLANATION = [line.replace('counted in the alarm journal', 'counted in /journal')
                             .replace('not counted in the alarm journal', 'not counted in /journal')
                             for line in _OLD_ALERTS_EXPLANATION]
# Before 0.1.11 final: one value per line (all, stock, crypto, off) and a fixed kind per signal
_OLD_KIND = {'positive5': 'alarm', 'score8': 'alarm', 'oversold': 'alarm', 'cross_ema_sell': 'alarm'}


def alert_line(key, kinds):
    """One line of the [alerts] section: kinds for stocks and crypto (meaning and backtest: README)."""
    stock, crypto = kinds
    return f"{key:<15}= {stock + ';':<6} {crypto}"


def parse_alert_value(key, text):
    """'alarm; info' -> ('alarm', 'info'). Old single values (all, stock, crypto, off, 1, 0) are converted.
    Returns (kinds, old_format)."""
    value = text.split('#', 1)[0].strip().lower()
    parts = [part.strip() for part in value.split(';')]
    if len(parts) == 2 and all(part in ALERT_KINDS for part in parts):
        return (parts[0], parts[1]), False
    old = {'1': 'all', 'on': 'all', '0': 'off', 'stocks': 'stock'}.get(value, value)
    kind = _OLD_KIND.get(key, 'info')
    converted = {'all': (kind, kind), 'stock': (kind, 'off'), 'crypto': ('off', kind), 'off': ('off', 'off')}
    if len(parts) == 1 and old in converted:
        return converted[old], True
    raise ValueError(f"config.ini [alerts] {key} = {value}: expected '<stocks>; <crypto>', "
                     f"each one of {', '.join(ALERT_KINDS)} (e.g. 'alarm; info')")


def _alert_section(lines):
    """(index of the [alerts] header or None, index after the section, {key: (line index, value text)})"""
    section, start, end, found = None, None, len(lines), {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            if section == 'alerts':
                end = index
            section = stripped[1:-1].strip().lower()
            if section == 'alerts':
                start = index
            continue
        if section == 'alerts' and '=' in line and not stripped.startswith(('#', ';')):
            name, value = line.split('=', 1)
            found[name.strip().lower()] = (index, value)
    return start, end, found


def ensure_alert_section(path='config.ini'):
    """The [alerts] section is required. Creates it with the default values and comments, adds missing lines
    and converts lines in the old format (one value per line) without changing their effect.
    Returns a list of what was changed (empty = nothing to do)."""
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    start, end, found = _alert_section(lines)
    kinds, changes = {}, []
    for key in ALERTS:
        if key in found:
            kinds[key], old = parse_alert_value(key, found[key][1])
            if old:
                changes.append(f"{key} converted")
        else:
            kinds[key] = (ALERTS[key][2], ALERTS[key][3])
            changes.append(f"{key} added")
    # comments behind the lines (up to 0.1.13: meaning and backtest) are removed, the values stay
    commented = [key for key in ALERTS if key in found and '#' in found[key][1]]
    for key in commented:
        lines[found[key][0]] = alert_line(key, kinds[key])
    if commented:
        changes.append('comments behind the lines in [alerts] removed')
    if not changes:
        pass
    elif start is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += [''] + ALERTS_SECTION_HEADER + [alert_line(key, kinds[key]) for key in ALERTS]
    elif any(change.endswith('converted') for change in changes):
        # old format: write the whole section new (header with the explanation, one line per signal)
        tail = lines[end:]
        lines = lines[:start] + ALERTS_SECTION_HEADER + [alert_line(key, kinds[key]) for key in ALERTS] \
            + ([''] if tail else []) + tail
    else:
        insert = end   # missing lines at the end of the section (before trailing empty lines)
        while insert > start + 1 and not lines[insert - 1].strip():
            insert -= 1
        lines[insert:insert] = [alert_line(key, kinds[key]) for key in ALERTS if key not in found]
    # the long explanation up to 0.1.13 -> one short line (only when it is unchanged)
    header = next((i for i, line in enumerate(lines) if line.strip().lower() == '[alerts]'), None)
    for old in (_OLD_ALERTS_EXPLANATION, _OLDER_ALERTS_EXPLANATION):
        if header is not None and [line.rstrip() for line in lines[header + 1:header + 1 + len(old)]] == old:
            lines[header + 1:header + 1 + len(old)] = ALERTS_SECTION_HEADER[1:-1]
            changes.append('explanation in [alerts] shortened')
            break
    if changes:
        _write_lines(path, lines)
    return changes


LOSS_LIMIT_LINE = ('total_loss_limit = -5           # loss limit alarm: total P&L of an own position in % '
                   '(e.g. -5 = alarm when it falls below -5 %)')


ANTHROPIC_SECTION = ['[anthropic]',
                     '# optional: key for the Claude API (https://console.anthropic.com), empty = Claude API not used',
                     'api_key =',
                     'model = claude-opus-5-5']


AI_PROMPT_FILE = 'ai_prompt.txt'


def read_ai_prompt(path=AI_PROMPT_FILE):
    """AI prompt template from ai_prompt.txt, None if there is none (or it is empty)."""
    try:
        with open(path, encoding='utf-8-sig') as file:
            return file.read().strip() or None
    except FileNotFoundError:
        return None


def write_ai_prompt(text, path=AI_PROMPT_FILE):
    """Write the template to ai_prompt.txt (replaces the file)."""
    _write_lines(path, text.strip().splitlines())


def take_ai_prompt_from_config(path='config.ini'):
    """0.1.13 had the template in config.ini [ai_prompt]: returns its text and removes the section
    (line by line, all other lines stay). None if there is no such section."""
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().lower() == '[ai_prompt]'), None)
    if start is None:
        return None
    config = configparser.ConfigParser(interpolation=None)   # no inline comments: ";" belongs to the text
    config.read(path, encoding='utf-8-sig')
    text = config.get('ai_prompt', 'template', fallback='').strip()
    # end: the next section header at the start of a line (the text lines are indented)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('[')), len(lines))
    del lines[start:end]
    while lines and not lines[-1].strip():
        lines.pop()
    _write_lines(path, lines)
    return text


INFOS_FOR_LINE = ('infos_for = all                 # infos (not alarms) for: all = every asset in the analysis, '
                  'portfolio = only assets you hold (fewer messages)')
INFOS_FOR = ('all', 'portfolio')


def read_infos_for(path='config.ini'):
    """infos_for from config.ini [alertsconfig]: 'all' or 'portfolio' (missing or unknown: 'all')."""
    config = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'))
    config.read(path, encoding='utf-8-sig')
    value = config.get('alertsconfig', 'infos_for', fallback='all').strip().lower()
    return value if value in INFOS_FOR else 'all'


def set_infos_for(value, path='config.ini'):
    """Change infos_for in config.ini [alertsconfig] (for /smart_alert_config). Only this line is rewritten,
    with its comment; a missing line is added."""
    if value not in INFOS_FOR:
        raise ValueError(f"infos_for must be all or portfolio, not '{value}'.")
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().lower() == '[alertsconfig]'), None)
    if start is None:
        raise ValueError(f"{path}: section [alertsconfig] is missing.")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith('[')), len(lines))
    comment = INFOS_FOR_LINE.index('#')
    new = f'infos_for = {value}'.ljust(comment) + INFOS_FOR_LINE[comment:]   # comment stays in its column
    for index in range(start + 1, end):
        if '=' in lines[index] and not lines[index].lstrip().startswith(('#', ';')) \
                and lines[index].split('=', 1)[0].strip().lower() == 'infos_for':
            lines[index] = new
            break
    else:
        insert = end
        while insert > start + 1 and not lines[insert - 1].strip():
            insert -= 1
        lines.insert(insert, new)
    _write_lines(path, lines)


def update_config(path='config.ini'):
    """Bring an older config.ini up to date (at the start of CaSa), line by line so that the token, the key
    and all comments stay: [alarm] is renamed to [alertsconfig], total_loss_limit is added, [alerts] is
    created or completed, infos_for and an empty [anthropic] section are added. Returns a list of what was changed."""
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    changes = []
    headers = [line.strip()[1:-1].strip().lower() for line in lines if line.strip().startswith('[')]
    if 'alertsconfig' not in headers:
        for index, line in enumerate(lines):
            if line.strip().lower() == '[alarm]':
                lines[index] = '[alertsconfig]'
                changes.append('section [alarm] renamed to [alertsconfig]')
                break
        else:
            raise ValueError(f"{path}: section [alertsconfig] is missing.")
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == '[alertsconfig]')
    end = start + 1
    while end < len(lines) and not lines[end].strip().startswith('['):
        end += 1
    keys = {line.split('=', 1)[0].strip().lower() for line in lines[start + 1:end]
            if '=' in line and not line.strip().startswith(('#', ';'))}
    if 'total_loss_limit' not in keys:
        insert = end
        while insert > start + 1 and not lines[insert - 1].strip():
            insert -= 1
        lines.insert(insert, LOSS_LIMIT_LINE)
        changes.append('total_loss_limit = -5 added to [alertsconfig]')
        end += 1
    if 'infos_for' not in keys:
        insert = end
        while insert > start + 1 and not lines[insert - 1].strip():
            insert -= 1
        lines.insert(insert, INFOS_FOR_LINE)
        changes.append('infos_for = all added to [alertsconfig]')
    if 'anthropic' not in headers:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += [''] + ANTHROPIC_SECTION
        changes.append('empty section [anthropic] added (optional Claude API key)')
    old_comment = '# the API costs money per request (billed by Anthropic per token)'   # up to 0.1.13
    if old_comment in [line.strip() for line in lines]:
        lines = [line for line in lines if line.strip() != old_comment]
        changes.append('comment about the API costs removed from [anthropic]')
    if changes:
        _write_lines(path, lines)
    changes += ensure_alert_section(path)
    return changes


def read_alert_settings(path='config.ini'):
    """Kinds from config.ini [alerts] (required, see ensure_alert_section): {key: (stocks, crypto)}."""
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path, encoding='utf-8-sig'):
        raise FileNotFoundError(f"{path} not found.")
    if not config.has_section('alerts'):
        raise ValueError(f"{path}: section [alerts] is missing (it is created at the start of CaSa).")
    settings = {}
    for key in ALERTS:
        if not config.has_option('alerts', key):
            raise ValueError(f"{path} [alerts]: line '{key}' is missing (it is added at the start of CaSa).")
        settings[key] = parse_alert_value(key, config.get('alerts', key))[0]
    return settings


def set_alert_setting(key, asset_type, kind, path='config.ini'):
    """Change one value in config.ini [alerts] (for /smart_alert_config). Only this line is rewritten, with its
    comment; all other lines (token, key, own comments) stay exactly as they are."""
    if key not in ALERTS or asset_type not in ('stock', 'crypto') or kind not in ALERT_KINDS:
        raise ValueError(f"Unknown alert '{key}', asset type '{asset_type}' or kind '{kind}'.")
    ensure_alert_section(path)
    with open(path, encoding='utf-8-sig') as file:
        lines = file.read().splitlines()
    _start, _end, found = _alert_section(lines)
    index, value = found[key]
    stock, crypto = parse_alert_value(key, value)[0]
    lines[index] = alert_line(key, (kind, crypto) if asset_type == 'stock' else (stock, kind))
    _write_lines(path, lines)


# ── Alarm journal (alarm_journal.csv) ──────────────────────────────────────
# Every sent alarm with its price; 20 values (trading days for stocks, days for crypto) later the result is
# filled in. This shows over time whether the alarms also work on new data, not only in the backtest.
JOURNAL_PATH = 'alarm_journal.csv'
JOURNAL_HEADER = ['date', 'time', 'type', 'symbol', 'name', 'code', 'alarm', 'direction', 'period', 'price',
                  'score', 'watched', 'price_20', 'return_20', 'hit']
JOURNAL_HORIZON = 20
ALARM_NAMES = {'121': 'Positive-5', '141': 'Score>8', '151': 'Oversold', '201': 'Cross-EMA Sell', '211': 'Loss limit',
               # infos can also be set to 'alarm' in config.ini [alerts]
               '301': 'Cross-EMA Buy', '321': 'Negative-5', '361': '100-day high', '362': '100-day low',
               '331': 'Pattern up', '332': 'Pattern down'}
# Backtest 0.1.11 (54 assets, 20 days): share of alarms in the right direction
ALARM_BACKTEST_HITS = {'121': 54, '141': 55, '151': 58, '201': 46}


def read_journal(path=JOURNAL_PATH):
    """All journal rows (missing file = empty journal)."""
    try:
        with open(path, newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file, delimiter=';')
            if reader.fieldnames != JOURNAL_HEADER:
                raise ValueError(f"{path} has a wrong or missing header.")
            return list(reader)
    except FileNotFoundError:
        return []


def _write_journal(rows, path):
    lines = [';'.join(JOURNAL_HEADER)] + [';'.join(str(row.get(column, '')) for column in JOURNAL_HEADER) for row in rows]
    _write_lines(path, lines)


def journal_add(asset_type, symbol, name, code, period, date, price, score, watched, path=JOURNAL_PATH):
    """Record one sent alarm."""
    rows = read_journal(path)
    rows.append({'date': pd.Timestamp(date).strftime('%d/%m/%Y'), 'time': datetime.now().strftime('%H:%M'),
                 'type': asset_type, 'symbol': symbol.lower(), 'name': ' '.join(str(name).replace(';', ',').split()),
                 'code': code, 'alarm': ALARM_NAMES.get(code, code), 'direction': 'buy' if code.startswith('1') else 'sell',
                 'period': period, 'price': f"{float(price):.6g}", 'score': score, 'watched': int(bool(watched)),
                 'price_20': '', 'return_20': '', 'hit': ''})
    _write_journal(rows, path)


def journal_update(prices, path=JOURNAL_PATH, horizon=JOURNAL_HORIZON):
    """Fill in the result of open alarms once `horizon` later prices exist.
    prices: {(type, symbol lower): daily price series}. Returns the number of alarms evaluated now."""
    rows = read_journal(path)
    done = 0
    for row in rows:
        if row['return_20'] or (row['type'], row['symbol']) not in prices:
            continue
        series = prices[(row['type'], row['symbol'])]
        later = series[series.index > pd.to_datetime(row['date'], format='%d/%m/%Y')]
        if len(later) < horizon:
            continue
        end = float(later.iloc[horizon - 1])
        change = (end / float(row['price']) - 1) * 100
        row['price_20'], row['return_20'] = f"{end:.6g}", f"{change:.2f}"
        row['hit'] = int(change > 0 if row['direction'] == 'buy' else change < 0)
        done += 1
    if done:
        _write_journal(rows, path)
    return done


def journal_stats(path=JOURNAL_PATH):
    """Per alarm: count, evaluated, hits in %, average change after 20 values, backtest hits."""
    rows = read_journal(path)
    stats = {}
    for row in rows:
        entry = stats.setdefault(row['code'], {'alarm': row['alarm'], 'count': 0, 'evaluated': 0, 'hits': 0, 'returns': []})
        entry['count'] += 1
        if row['return_20']:
            entry['evaluated'] += 1
            entry['hits'] += int(row['hit'])
            entry['returns'].append(float(row['return_20']))
    for code, entry in stats.items():
        n = entry['evaluated']
        entry['hit_pct'] = 100 * entry['hits'] / n if n else None
        entry['avg_return'] = sum(entry['returns']) / n if n else None
        entry['backtest_hit_pct'] = ALARM_BACKTEST_HITS.get(code)
    first = min((pd.to_datetime(row['date'], format='%d/%m/%Y') for row in rows), default=None)
    return stats, first


def journal_report(path=JOURNAL_PATH):
    """Telegram text (HTML) with the journal statistics."""
    stats, first = journal_stats(path)
    if not stats:
        return '<b>Alarm journal</b>\nNo alarms recorded yet.'
    order = [code for code in ALARM_NAMES if code in stats] + sorted(set(stats) - set(ALARM_NAMES))
    lines = [f"{'Alarm':<15}{'n':>3}{'done':>5}{'hits':>6}{'avg':>8}{'test':>6}"]
    for code in order:
        e = stats[code]
        hit = f"{e['hit_pct']:.0f}%" if e['hit_pct'] is not None else '–'
        avg = f"{e['avg_return']:+.1f}%" if e['avg_return'] is not None else '–'
        test = f"{e['backtest_hit_pct']}%" if e['backtest_hit_pct'] else '–'
        lines.append(f"{e['alarm'][:15]:<15}{e['count']:>3}{e['evaluated']:>5}{hit:>6}{avg:>8}{test:>6}")
    waiting = sum(e['count'] - e['evaluated'] for e in stats.values())
    evaluated = sum(e['evaluated'] for e in stats.values())
    text = (f"<b>Alarm journal</b> since {first:%d/%m/%Y}\n<pre>" + '\n'.join(lines) + "</pre>\n"
            f"done = result after {JOURNAL_HORIZON} trading days (crypto: days) known, hits = price moved in the "
            f"alarm direction, avg = average change (sell: falling is good), test = hits in the backtest.\n"
            f"{waiting} alarms still waiting for their result.")
    if evaluated < 30:
        text += "\n<i>Still few results: wait for about 30 per alarm before drawing conclusions.</i>"
    return text


def apply_transactions(asset, rows):
    matching = [row for row in rows
                if row['type'] == asset.type and row['symbol'].lower() == asset.symbol.lower()
                and row['action'] != 'watchlist']
    # Sort late-entered buys by date; for the same date the CSV order applies.
    matching.sort(key=lambda row: datetime.strptime(row['date'], '%d/%m/%Y'))
    for row in matching:
        amount, qty = valid_numbers(row['amount'], row['qty'])
        if row['action'] == 'buy':
            asset.buy(amount, qty, row['date'])
        else:
            asset.sell(amount, qty, row['date'])
    return asset
