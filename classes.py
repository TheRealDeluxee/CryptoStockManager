import configparser
import csv
import math
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
        # Erst prüfen, dann buchen: Ein Fehler darf den Bestand nicht verändern.
        fiat_eur, amount = valid_numbers(fiat_eur, amount)
        parsed_date = datetime.strptime(str(date).strip(), "%d/%m/%Y")
        self.state.df_buy_history.loc[len(self.state.df_buy_history)] = [parsed_date, fiat_eur, amount]
        self.state.buy_balance_eur += fiat_eur
        self.state.amount_crypto_stock += amount
        self.state.watch_list = False


    def sell(self, fiat_eur, amount, date=None):
        """
        FIFO sell: reduziert Bestand und Cost-Basis anhand der Kaufhistorie.

        - sell_money += fiat_eur
        - amount_crypto_stock -= amount (zu große Verkäufe werden abgelehnt)
        - df_buy_history wird FIFO abgebaut (Amount + Cost [€])
        - buy_balance_eur wird um die verkaufte FIFO-Cost-Basis reduziert
        - behebt Pandas FutureWarning durch explizites float-casting
        """
        fiat_eur, amount = valid_numbers(fiat_eur, amount)
        available = self.state.amount_crypto_stock
        if amount > available and not math.isclose(amount, available, rel_tol=1e-10, abs_tol=0.0):
            raise ValueError(f"Verkauf von {amount} nicht möglich: Bestand ist nur {available}.")
        amount = min(amount, available)
        if amount <= 0:
            raise ValueError("Kein Bestand zum Verkaufen vorhanden.")
        df = self.state.df_buy_history
        if df.empty or not math.isclose(float(df['Amount'].sum()), available, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError("Kaufhistorie und Bestand passen nicht zusammen.")

        # FIFO: Die ältesten noch offenen Käufe werden zuerst verkauft.
        df = df.sort_values('Date', kind='stable').reset_index(drop=True)
        self.state.sell_money += fiat_eur
        self.state.amount_crypto_stock -= amount
        self.state.watch_list = self.state.amount_crypto_stock <= 0

        # --- WICHTIG: dtypes fixen (verhindert FutureWarning) ---
        # (kannst du auch einmalig in buy() machen; hier ist es robust)
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

            # offene Cost-Basis reduzieren
            self.state.buy_balance_eur -= cost_taken

            remaining -= take

            leftover_amount = row_amount - take
            leftover_cost = row_cost - cost_taken

            if leftover_amount <= 0 or math.isclose(take, row_amount, rel_tol=1e-12, abs_tol=0.0):
                rows_to_drop.append(i)
            else:
                df.at[i, "Amount"] = float(leftover_amount)
                df.at[i, "Cost [€]"] = float(leftover_cost)

        # Historie bereinigen
        if rows_to_drop:
            df = df.drop(index=rows_to_drop).reset_index(drop=True)

        self.state.df_buy_history = df

        # numerische Stabilität
        if self.state.buy_balance_eur < 0 and abs(self.state.buy_balance_eur) < 1e-6:
            self.state.buy_balance_eur = 0.0

    def reset_alarm_duplicates(self):
        self.state.seven_day_alarms.clear()
        self.state.alarm_prev.clear() 

    def activate_test_mode(self):
        self.test_mode = True

    def get_API_data(self, data_type):
        if data_type not in ('100d', '100h') or self.type not in ('crypto', 'stock'):
            raise ValueError('Unbekannter Anlagetyp oder Analysezeitraum.')

        if self.type == 'crypto':
            if data_type == '100d':
                self.params_crypto_stock = {'vs_currency': 'eur', 'days': '100'}   
            elif data_type == '100h':
                self.params_crypto_stock = {'vs_currency': 'eur', 'days': '4.17'}
            df = func.get_crypto(self.symbol,self.params_crypto_stock)
        elif self.type == 'stock':   
            if data_type == '100d':
                self.params_crypto_stock = "6mo"
            elif data_type == '100h':
                self.params_crypto_stock = "1mo"
            df = func.get_stock(self.symbol,self.params_crypto_stock)
        return df

    def process_data(self, df, data_type):

        if df is None or df.empty or len(df) < 26:
            raise ValueError(f"Zu wenige Kursdaten für {self.symbol} (mindestens 26 benötigt).")

        df = func.calc_indicator_fuctions(df)
        self.state = func.calc_state_fuctions(self.state, df)
        alarms, self.state.tech_indicators, self.state.score = func.alarm(df,self.search,self.state.watch_list, self.state.current_profit_pct, self.state.amount_older_than_one_year, self.state.amount_older_than_one_year_pct, data_type)
        alarms_filter = func.filter(alarms, self.state)
        
        return alarms_filter

    def plot_send(self, df, data_type, alarms_filter):
        if not alarms_filter:
            return
        # Ein eigenes Bild pro Versand verhindert ein späteres Überschreiben.
        for info in alarms_filter.values():
            image_name = "alarm_" + uuid.uuid4().hex
            func.plot_and_save(df, image_name, data_type, self.state.zero_line)
            func.pushover_image(image_name, info['msg'], info['priority'], delete_after_send=True)

    def get_summary(self, df):
        return {
            'Name': self.search,
            'Piece [€]': round(float(df['Price'].iloc[-1]), 2),
            '7d P&L [%]': round(self.state.current_seven_days_slope_pct, 2),
            '1d P&L [%]': round(self.state.current_one_day_price_change_pct, 2),
            'Rating': self.state.score
        }


# Gemeinsame Datei-Prüfungen: Beim Eingeben und Laden gelten dieselben Regeln.
CSV_HEADER = ["type", "symbol", "name", "action", "amount", "qty", "date"]


def valid_numbers(amount, qty):
    amount = float(str(amount).replace(',', '.'))
    qty = float(str(qty).replace(',', '.'))
    if not math.isfinite(amount) or not math.isfinite(qty) or amount <= 0 or qty <= 0:
        raise ValueError("Betrag und Menge müssen endliche Zahlen größer als null sein.")
    return amount, qty


def read_watchlist(path="watchlist.ini"):
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path, encoding="utf-8-sig"):
        raise FileNotFoundError(f"{path} wurde nicht gefunden.")
    entries = {}
    for section, asset_type in (("crypto", "crypto"), ("stocks", "stock")):
        if not config.has_section(section):
            continue
        for symbol, value in config.items(section):
            symbol = symbol.strip().lower()
            if not symbol or any(char in symbol for char in '/\\:*?"<>|'):
                raise ValueError(f"Ungültiges Symbol in [{section}]: {symbol}")
            entries[(asset_type, symbol)] = value.split(';')[0].strip() or symbol
    if not entries:
        raise ValueError("Die Watchlist enthält keine aktiven Symbole.")
    return entries


def read_transactions(path="transactions.csv"):
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file, delimiter=';')
            if reader.fieldnames != CSV_HEADER:
                raise ValueError("transactions.csv hat eine falsche oder fehlende Kopfzeile.")
            rows = list(reader)
    except FileNotFoundError:
        return []
    for line, row in enumerate(rows, 2):
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Falsche Anzahl von Spalten.")
            row['type'] = row['type'].strip().lower()
            row['symbol'] = row['symbol'].strip().lower()
            row['action'] = row['action'].strip().lower()
            if row['type'] not in ('crypto', 'stock') or not row['symbol']:
                raise ValueError("Ungültiger Typ oder fehlendes Symbol.")
            if row['action'] not in ('buy', 'sell', 'watchlist'):
                raise ValueError("Unbekannte Aktion.")
            if row['action'] == 'watchlist':
                continue
            valid_numbers(row['amount'], row['qty'])
            row['date'] = row['date'].strip().replace('.', '/')
            parsed_date = datetime.strptime(row['date'], '%d/%m/%Y')
            if parsed_date.date() > datetime.now().date():
                raise ValueError("Transaktionen dürfen nicht in der Zukunft liegen.")
        except ValueError as error:
            raise ValueError(f"transactions.csv, Zeile {line}: {error}") from error
    return rows


def apply_transactions(asset, rows):
    matching = [row for row in rows
                if row['type'] == asset.type and row['symbol'].lower() == asset.symbol.lower()
                and row['action'] != 'watchlist']
    # Nachgetragene Käufe zeitlich einordnen; bei gleichem Datum gilt CSV-Reihenfolge.
    matching.sort(key=lambda row: datetime.strptime(row['date'], '%d/%m/%Y'))
    for row in matching:
        amount, qty = valid_numbers(row['amount'], row['qty'])
        if row['action'] == 'buy':
            asset.buy(amount, qty, row['date'])
        else:
            asset.sell(amount, qty, row['date'])
    return asset
