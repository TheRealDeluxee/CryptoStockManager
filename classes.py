import configparser
import csv
import math
import os
import tempfile
import uuid

import func
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
        alarms, self.state.tech_indicators, self.state.score = func.alarm(df,self.search,self.state.watch_list, self.state.current_profit_pct, self.state.amount_older_than_one_year, self.state.amount_older_than_one_year_pct, data_type, self.state.pattern, self.state.score_rank, self.state.sr_zones)
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
            func.pushover_image(image_name, info['msg'], info['priority'], delete_after_send=True, html_format=True)

    def get_summary(self, df):
        return {
            'Name': self.search,
            'Piece [€]': round(float(df['Price'].iloc[-1]), 2),
            '7d P&L [%]': round(self.state.current_seven_days_slope_pct, 2),
            '1d P&L [%]': round(self.state.current_one_day_price_change_pct, 2),
            'Rating': self.state.score_rank or 0   # score 1-10 (0 = not available)
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


# ── Price alerts (valuealert.ini) ──────────────────────────────────────────
VALUE_ALERT_HEADER = """# CaSa price alerts: message when a price reaches your mark
#
# Format: name = symbol; below|above; price in EUR; note
# name   = free choice, must be unique in the file
# symbol = as in assets.ini (e.g. APC.DE or bitcoin)
# below  = alert when the price falls to the mark or lower, above = rises to the mark or higher
# note   = any text, e.g. your plan for this price
#
# Checked every hour (stocks Monday to Friday 08:00-22:00). When the mark is reached you get a message
# with your note and the daily analysis; the alert is then removed.
# Alerts can also be added and deleted in Telegram with /price_alert.

[alerts]"""


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


def read_value_alerts(path="valuealert.ini", assets_path="assets.ini"):
    """Price alerts from valuealert.ini. Type (crypto/stock) and name come from assets.ini. Missing file = no alerts."""
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path, encoding="utf-8-sig") or not config.has_section('alerts'):
        return []
    assets = read_assets(assets_path)
    alerts = []
    for key, value in config.items('alerts'):
        parts = [part.strip() for part in value.split(';', 3)]
        if len(parts) < 3 or parts[1].lower() not in ('below', 'above'):
            raise ValueError(f"{path}, alert '{key}': format is symbol; below|above; price; note")
        symbol = parts[0].lower()
        matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in assets]
        if len(matches) != 1:
            problem = 'not in assets.ini' if not matches else 'listed as crypto and stock in assets.ini'
            raise ValueError(f"{path}, alert '{key}': {parts[0]} is {problem}.")
        alerts.append({'key': key, 'symbol': symbol, 'type': matches[0], 'name': assets[(matches[0], symbol)],
                       'direction': parts[1].lower(), 'price': parse_price(parts[2]),
                       'note': parts[3] if len(parts) > 3 else ''})
    return sorted(alerts, key=lambda alert: (alert['name'], alert['price']))


def add_value_alert(symbol, direction, price, note, path="valuealert.ini", assets_path="assets.ini"):
    """Append a price alert (for /price_alert); the file is created with explanations if missing.
    Returns (key, asset name, asset type)."""
    symbol = symbol.strip().lower()
    assets = read_assets(assets_path)
    matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in assets]
    if len(matches) != 1:
        raise ValueError(f"{symbol} is {'not in assets.ini' if not matches else 'listed as crypto and stock in assets.ini'}.")
    if direction not in ('below', 'above'):
        raise ValueError("Direction must be 'below' or 'above'.")
    note = ' '.join(note.split())   # no line breaks in the ini
    lines = []
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as file:
            lines = file.read().splitlines()
    if not any(line.strip().lower() == '[alerts]' for line in lines):
        lines += VALUE_ALERT_HEADER.splitlines() if not lines else ['', '[alerts]']
    existing = {line.split('=', 1)[0].strip().lower() for line in lines if '=' in line and not line.lstrip().startswith('#')}
    base = f"{symbol.replace('.', '_').replace('-', '_')}_{direction}_{price:.2f}".replace('.', '_')
    key, number = base, 2
    while key in existing:   # name must be unique
        key, number = f"{base}_{number}", number + 1
    shown_symbol = symbol.upper() if matches[0] == 'stock' else symbol
    lines.append(f"{key:<28} = {shown_symbol}; {direction}; {price:.2f}; {note}".rstrip('; '))
    _write_lines(path, lines)
    return key, assets[(matches[0], symbol)], matches[0]


def delete_value_alert(key, path="valuealert.ini"):
    """Remove one price alert (for /price_alert and after it was triggered). Returns the removed line."""
    return _delete_ini_entry(path, 'alerts', key, 'Alert')


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
