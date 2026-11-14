# --- Standard Library ---
import os
import time
import base64, hashlib, hmac
from datetime import datetime, timedelta
from urllib.parse import quote  # only if you need to encode URL parts
import urllib.request, json
import random
import math
import html

# --- External Packages ---
import requests #instal
import pandas as pd #instal
import numpy as np # install
from scipy.stats import linregress #Install
from scipy.stats import zscore
from pytrends.request import TrendReq #Install
import yfinance as yf #Install
import matplotlib #Install
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import multiprocessing as mp

_message_queue = None
one_day_price_change = 0.0
seven_day_price_change = 0.0
one_day_profit_limit = 0.0
output_dir = 'plots'
previous_alarm_change = 0.0

def get_stock(symbol, period):
    df = pd.DataFrame()
    
    if period == '6mo':
        interval = '1h'
    else:
        interval = '1d'
    
    # Exponential backoff: Start with shorter waits, increase over time
    wait_times = [60, 120, 300, 600, 900]  # 1min, 2min, 5min, 10min, 15min
    
    for i in range(5):
        try:
            # Random delay before request to avoid patterns (1-3 seconds)
            time.sleep(random.uniform(1, 3))
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval, timeout=20)
            
            if not df.empty:
                break
            else:
                wait_time = wait_times[i] if i < len(wait_times) else 900
                msg = f"Empty response for {symbol}, waiting {wait_time}s before retry {i+1}"
                pushover(msg)
                if i < 4:
                    time.sleep(wait_time)
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Determine wait time based on error type
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                wait_time = wait_times[i] if i < len(wait_times) else 900
            else:
                wait_time = 30  # Shorter wait time for other errors
            
            infos = {
                'sym': symbol, 
                'tries': int(i + 1), 
                'function': 'get_stock',
                'exception': e,
                'wait_time': wait_time
            }
            msg = "Script try number %(tries)s to %(function)s of symbol %(sym)s\nWaiting %(wait_time)ss\n\n%(exception)s" % infos
            pushover(msg)
            
            if i < 4:  # Do not wait after the last attempt
                time.sleep(wait_time)
    
    if df.empty:
        # Optional: Return None oder raise Exception
        return None
    
    if 'Date' not in df.columns:
        df['Date'] = df.index
    
    df = df.copy()  # gesamte Historie behalten, gekürzt wird in cut_to_window()
    df['Price'] = stock_prices_in_eur(stock, df, period)
    
    df_inverted = df.reset_index(drop=True)
    df_inverted.rename(columns={'index': 'Date'}, inplace=True)
    
    columns_to_drop = ['Close', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']
    df_dropped = df_inverted.drop(columns=[col for col in columns_to_drop if col in df_inverted.columns])
    
    return df_dropped


def stock_prices_in_eur(stock, df, period):
    """Yahoo liefert oft USD/GBP. Unsere Käufe und Diagramme rechnen in EUR."""
    currency = stock.get_history_metadata().get('currency')
    if not currency:
        raise ValueError("Die Kurswährung konnte nicht bestimmt werden.")
    prices = df['Close'].astype(float)
    if currency == 'EUR':
        return prices
    # London notiert teilweise in Pence statt Pfund.
    if currency in ('GBp', 'GBX'):
        currency = 'GBP'
        prices = prices / 100
    if len(currency) != 3 or not currency.isalpha():
        raise ValueError(f"Nicht unterstützte Kurswährung: {currency}")
    fx = yf.Ticker(f"{currency.upper()}EUR=X").history(period=period, interval='1d', timeout=20)
    if fx.empty:
        raise ValueError(f"Kein Wechselkurs für {currency}/EUR verfügbar.")
    # Je Kurstag den zuletzt verfügbaren täglichen Wechselkurs verwenden.
    dates = pd.to_datetime(df.index, utc=True).normalize()
    fx_dates = pd.to_datetime(fx.index, utc=True).normalize()
    rates = pd.DataFrame({'Date': fx_dates, 'Rate': fx['Close'].to_numpy()})
    rates = rates.dropna().sort_values('Date').drop_duplicates('Date', keep='last')
    joined = pd.merge_asof(pd.DataFrame({'Date': dates}).sort_values('Date'), rates,
                           on='Date', direction='backward', tolerance=pd.Timedelta(days=7))
    if joined['Rate'].isna().any() or (joined['Rate'] <= 0).any():
        raise ValueError(f"Unvollständige Wechselkurse für {currency}/EUR.")
    return prices * joined['Rate'].to_numpy()


def get_crypto(symbol, params_historical):
    url_historical = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
    wait_times = [30, 60, 120, 300, 600]
    data = None
    
    for i in range(5):
        try:
            time.sleep(random.uniform(0.5, 2))
            response = requests.get(url_historical, params=params_historical, timeout=30)
            
            # Determine wait time based on status or error type
            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', wait_times[i]))
            elif response.status_code != 200:
                wait_time = wait_times[i]
            else:
                data = response.json()
                if 'prices' in data and data['prices']:
                    break
                wait_time = wait_times[i]
            
            log_print(f"Retry {i+1} for {symbol}, waiting {wait_time}s (Status: {response.status_code})")
            if i < 4:
                time.sleep(wait_time)
                
        except Exception as e:
            wait_time = wait_times[min(i, len(wait_times)-1)]
            log_print(f"Error for {symbol} (try {i+1}): {e}. Waiting {wait_time}s")
            if i < 4:
                time.sleep(wait_time)
    
    if not data or 'prices' not in data or not data['prices']:
        log_print(f"Failed to fetch {symbol} after 5 attempts")
        return None
    
    # Preise und Volumen anhand des Zeitstempels zuordnen (nicht blind zippen).
    prices = pd.DataFrame(data['prices'], columns=['Timestamp', 'Price'])
    volumes = pd.DataFrame(data.get('total_volumes', []), columns=['Timestamp', 'Volume'])
    df = prices.merge(volumes, on='Timestamp', how='left')
    df['Date'] = pd.to_datetime(df.pop('Timestamp'), unit='ms', utc=True).dt.tz_localize(None)
    df['Volume'] = df['Volume'].fillna(0)
    df = df.sort_values('Date').drop_duplicates('Date', keep='last')  # gekürzt wird in cut_to_window()
    return df[['Date', 'Price', 'Volume']].reset_index(drop=True)


def calc_indicator_fuctions(df):
    if df is None or df.empty or len(df) < 26:
        raise ValueError("Mindestens 26 Kurswerte werden benötigt.")
    for column in ('Price', 'Volume'):
        if column not in df or not np.isfinite(df[column].astype(float)).all():
            raise ValueError(f"Fehlende oder ungültige Werte in {column}.")
    if (df['Price'] <= 0).any() or (df['Volume'] < 0).any():
        raise ValueError("Kurse müssen positiv und Volumina mindestens null sein.")
   
    # Calculate Fast EMA and Slow EMA
    df['Fast EMA'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['Slow EMA'] = df['Price'].ewm(span=26, adjust=False).mean()
    df['Signal'] = 0.0
    df['Signal'] = np.where(df['Fast EMA'] > df['Slow EMA'], 1.0, 0.0)
    df['Position'] = df['Signal'].diff()
    df['Diff EMA'] = df['Fast EMA'] - df['Slow EMA']

    # Min und Max werden erst in cut_to_window() auf den 100 Werten berechnet.

    # Calculate the slope of Fast EMA
    df['Fast EMA Slope'] = df['Fast EMA'].diff()

    # Calculate RSI14
    df['RSI14'] = calculate_rsi(df, window=14)

    # Calculate Momentum
    df['Mom10'] = calculate_momentum(df, 'Price', window=10)
    # Calculate Volume Momentum
    df['VMom10'] = calculate_momentum(df, 'Volume', window=10)

    # Calculate VWMA20
    df['VWMA20'] = calculate_vwma(df, window=20)
    # Calculate SMA20
    df['SMA7'] = calculate_ma(df, 'Price', window=7)
    # Calculate VMA20
    df['VMA7'] = calculate_ma(df, 'Volume', window=7)

    # Calculate rolling 5-day slope percentage
    df['ROC5'] = ROC5(df, 'Price', window=5)

    # Calculate rolling 5-day slope percentage
    df['VROC5'] = ROC5(df, 'Volume', window=5)

    # Calculate bollinger bands
    df['BB_Mid'], df['BB_Upper'], df['BB_Lower'] = bollinger_bands(df, 'Price', window=20)

    return df

def cut_to_window(df, window=100):
    # Indikatoren wurden auf der langen Historie berechnet (kein Anlauf mehr am linken Rand).
    # Ausgewertet und gezeichnet werden nur die letzten 100 Werte.
    df = df.tail(window).reset_index(drop=True).copy()

    # Calculate min and max (nach dem Kürzen: bleibt das 100-Tage/-Stunden Minimum/Maximum)
    df['High'] = df['Fast EMA'].max()
    df['Low'] = df['Fast EMA'].min()
    df['HighP'] = np.where(df['Fast EMA'] == df['High'], 1, 0)
    df['LowP'] = np.where(df['Fast EMA'] == df['Low'], 1, 0)

    return df

def support_resistance(df, order=5):
    """Nächste Resistance über und nächster Support unter dem aktuellen Kurs.
    Nur Schlusskurs + Volumen -> gleiches Verfahren für Krypto/Aktie und Tage/Stunden.
    df ist die lange Historie (250 Tage / 450 Stunden), nicht das 100er-Fenster."""
    p = df['Price'].to_numpy(float)
    if len(p) < 2 * order + 2:
        return None, None

    # Umkehrpunkte: höchster/tiefster Wert innerhalb von ±order Werten
    pivots = []
    for i in range(order, len(p) - order):
        window = p[i - order:i + order + 1]
        if p[i] == window.max() or p[i] == window.min():
            pivots.append(p[i])

    # Zonenbreite aus typischer Bewegung pro Wert (Stunden schmaler als Tage)
    width = max(2 * float(np.median(np.abs(np.diff(p) / p[:-1]))), 0.004)

    # Nahe Umkehrpunkte zu Zonen zusammenfassen; eine Zone wächst höchstens auf 2 Breiten
    zones = []
    for pivot in sorted(pivots):
        if zones and pivot <= zones[-1]['low'] * (1 + 2 * width):
            zones[-1]['high'] = pivot
            zones[-1]['touches'] += 1
        else:
            zones.append({'low': pivot, 'high': pivot, 'touches': 1})
    for zone in zones:
        zone['low'], zone['high'] = zone['low'] * (1 - width / 2), zone['high'] * (1 + width / 2)
        zone['center'] = (zone['low'] + zone['high']) / 2

    # Nächste Zone über/unter dem Kurs; die Stärke (Berührungen) steht später in der Legende
    price = p[-1]
    above = [zone for zone in zones if zone['low'] > price]
    below = [zone for zone in zones if zone['high'] < price]
    resistance = min(above, key=lambda zone: zone['low']) if above else None
    support = max(below, key=lambda zone: zone['high']) if below else None

    # Notlösung: Hoch/Tief des geladenen Zeitraums (touches = 0)
    if resistance is None and p.max() > price:
        resistance = {'low': p.max(), 'high': p.max(), 'center': p.max(), 'touches': 0}
    if support is None and p.min() < price:
        support = {'low': p.min(), 'high': p.min(), 'center': p.min(), 'touches': 0}

    return resistance, support

def formation_signals(df, order=5, recent=30):
    """Anzeichen für Boden- und Topbildung (für die KI-Frage bei /ai).
    df ist die lange Historie mit Indikatoren. Nur Schlusskurs, RSI und Volumen."""
    p = df['Price'].to_numpy(float)
    rsi = df['RSI14'].to_numpy(float)
    v = df['Volume'].to_numpy(float)
    ema = df['Fast EMA'].to_numpy(float)
    n = len(p)
    # Umkehrpunkte; die letzten brauchen nur 2 Werte danach, damit frische Tiefs/Hochs zählen
    lows = [i for i in range(order, n - 2) if p[i] == p[i - order:i + order + 1].min()]
    highs = [i for i in range(order, n - 2) if p[i] == p[i - order:i + order + 1].max()]
    tolerance = max(2 * float(np.median(np.abs(np.diff(p) / p[:-1]))), 0.01)   # "gleiche Höhe"
    volume_median = float(np.median(v[-100:])) or 1.0

    def check(points, bottom):
        result = {}
        last_two = points[-2:] if len(points) >= 2 and points[-1] >= n - recent else []
        if last_two:
            a, b = last_two
            between = p[a:b + 1]
            result['folge'] = p[b] > p[a] if bottom else p[b] < p[a]            # höhere Tiefs / tiefere Hochs
            result['divergenz'] = bool(p[b] < p[a] and rsi[b] > rsi[a]) if bottom else bool(p[b] > p[a] and rsi[b] < rsi[a])
            same_level = abs(p[b] / p[a] - 1) <= tolerance
            gap = between.max() > max(p[a], p[b]) * (1 + 2 * tolerance) if bottom else between.min() < min(p[a], p[b]) * (1 - 2 * tolerance)
            result['doppel'] = bool(same_level and gap)                         # Doppelboden / Doppeltop
        else:
            result['folge'] = result['divergenz'] = result['doppel'] = None     # kein aktuelles Tief/Hoch
        # Volumen-Höhepunkt am Extrem der letzten 20 Werte (Kapitulation / Kaufrausch)
        k = n - 20 + int(np.argmax(v[-20:]))
        extreme = n - 20 + int(np.argmin(p[-20:]) if bottom else np.argmax(p[-20:]))
        direction = p[k] < p[k - 1] if bottom else p[k] > p[k - 1]
        result['volumen'] = bool(direction and v[k] >= 1.5 * volume_median and abs(k - extreme) <= 3)
        # Bestätigung: Kurs zurück über (Boden) bzw. unter (Top) dem EMA12
        result['bestaetigung'] = bool(p[-1] > ema[-1]) if bottom else bool(p[-1] < ema[-1])
        return result

    return check(lows, True), check(highs, False)

def trend_phase(prices):
    """Marktphase aus Tageskursen nach der klassischen Trendregel:
    Kurs und SMA50 über SMA200 = bullisch, beide darunter = bärisch, sonst Übergang."""
    p = pd.Series(prices, dtype=float).dropna().reset_index(drop=True)
    if len(p) < 200:
        return 'nicht belegbar (weniger als 200 Tageswerte)'
    sma50, sma200 = p.rolling(50).mean(), p.rolling(200).mean()
    price, s50, s200 = p.iloc[-1], sma50.iloc[-1], sma200.iloc[-1]
    if price > s200 and s50 > s200:
        phase = 'bullisch'
    elif price < s200 and s50 < s200:
        phase = 'bärisch'
    else:
        phase = 'neutral (Übergang)'
    before = sma200.iloc[-21]   # Richtung des SMA200 über 20 Tage
    slope = 'unklar' if pd.isna(before) else 'steigend' if s200 > before else 'fallend'

    # Letzte Kreuzung SMA50/SMA200: Golden Cross (nach oben) / Death Cross (nach unten)
    above = (sma50 > sma200)[sma200.notna()]
    changes = above.index[1:][above.iloc[1:].to_numpy() != above.iloc[:-1].to_numpy()]
    if len(changes):
        last = changes[-1]
        cross = f"{'Golden Cross' if above[last] else 'Death Cross'} vor {len(p) - 1 - last} Tagen"
    else:
        cross = f"keine SMA50/SMA200-Kreuzung in den letzten {len(above)} Tagen"
    distance = f"{(price / s200 - 1) * 100:+.1f} %".replace('.', ',')
    return (f"{phase} (Kurs {distance} zum SMA200, SMA50 {'über' if s50 > s200 else 'unter'} SMA200, "
            f"SMA200 {slope}; {cross})")

def market_prices(asset_type, symbol):
    """Leitmarkt für die Marktphase: Bitcoin für Krypto, Börsenindex je Handelsplatz für Aktien."""
    if asset_type == 'crypto':
        df = get_crypto('bitcoin', {'vs_currency': 'eur', 'days': '260'})
        return 'Bitcoin (Leitwert Krypto)', None if df is None else df['Price']
    suffix = symbol.upper().rsplit('.', 1)[-1] if '.' in symbol else ''
    ticker, name = {'DE': ('^GDAXI', 'DAX'), 'F': ('^GDAXI', 'DAX'),
                    'MI': ('FTSEMIB.MI', 'FTSE MIB')}.get(suffix, ('^GSPC', 'S&P 500'))
    hist = yf.Ticker(ticker).history(period='2y', interval='1d', timeout=20)   # Index in eigener Währung
    return name, None if hist.empty else hist['Close'].tail(250)

def calc_state_fuctions(state,df):
    # update balances and indicators into grouped state

    # Guard: need at least 3 rows for -1, -2, -3 indexing
    if df is None or df.empty or len(df) < 3:
        log_print("calc_state_fuctions: DataFrame too short, skipping calculations")
        return state

    # Safe reads of prices/emas
    try:
        last_price = float(df['Price'].iloc[-1])
        prev_price = float(df['Price'].iloc[-2])
        prev2_price = float(df['Price'].iloc[-3])
    except Exception as e:
        log_print(f"calc_state_fuctions: Price read error: {e}")
        return state

    amount = float(getattr(state, 'amount_crypto_stock', 0) or 0)
    buy_balance = float(getattr(state, 'buy_balance_eur', 0) or 0)

    state.current_balance_eur = amount * last_price
    state.yesterday_balance_eur = amount * prev_price

    state.current_profit = state.current_balance_eur - buy_balance
    state.yesterday_profit = state.yesterday_balance_eur - buy_balance

    # protect division by zero for profit pct
    if buy_balance:
        state.current_profit_pct = state.current_profit / buy_balance * 100
        state.yesterday_profit_pct = state.yesterday_profit / buy_balance * 100
    else:
        state.current_profit_pct = 0.0
        state.yesterday_profit_pct = 0.0

    # price change pct (guard divide by zero)
    state.current_one_day_price_change_pct = (last_price - prev_price) / prev_price * 100 if prev_price else 0.0
    state.yesterday_one_day_price_change_pct = (prev_price - prev2_price) / prev2_price * 100 if prev2_price else 0.0

    # EMA diff pct (guard denom)
    def safe_pct(numer, denom):
        return (numer / denom * 100) if denom else 0.0

    state.current_EMA_diff_pct = safe_pct(df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-1], df['Slow EMA'].iloc[-1])
    state.yesterday_EMA_diff_pct = safe_pct(df['Fast EMA'].iloc[-2] - df['Slow EMA'].iloc[-2], df['Slow EMA'].iloc[-2])

    state.current_seven_days_slope_pct = seven_day_slope_pct(df, True)
    state.yesterday_seven_days_slope_pct = seven_day_slope_pct(df, False)

    state.amount_older_than_one_year = older_than_one_year(state.df_buy_history)

    if amount:
        state.amount_older_than_one_year_pct = state.amount_older_than_one_year / amount * 100
    else:
        state.amount_older_than_one_year_pct = 0.0

    # Eine Watchlist braucht keinen künstlichen Kauf von 1 EUR / 1 Stück.
    state.watch_list = amount <= 0
    state.zero_line = buy_balance / amount if not state.watch_list else None

    return state
        

def bollinger_bands(df, column='Price', window=20, num_std=2):
    sma = df[column].rolling(window=window).mean()
    rstd = df[column].rolling(window=window).std()
    upper_band = sma + num_std * rstd
    lower_band = sma - num_std * rstd
    return sma, upper_band, lower_band


def ROC5(df, column='Price', window=7):
    """
    Calculate rolling slope percentage for a given column over a specified window.
    Returns a Series with the percentage change based on linear regression slope.
    """
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    if len(df) < window:
        return pd.Series(index=df.index, dtype=float)
    
    def calc_slope_pct(series):
        if len(series) < window or series.isna().any():
            return np.nan
        
        try:
            # Create index for regression (0 to window-1)
            x = np.arange(len(series))
            y = series.values
            
            # Perform linear regression
            result = linregress(x, y)
            slope = result.slope
            intercept = result.intercept
            
            # Calculate predicted values at start and end
            y_start = intercept  # when x = 0
            y_end = slope * (len(series) - 1) + intercept  # when x = window-1
            
            # Prevent division by zero
            if y_start == 0:
                return np.nan
            
            # Calculate percentage change
            slope_pct = (y_end / y_start - 1) * 100
            return slope_pct
            
        except Exception:
            return np.nan
    
    # Apply rolling calculation
    rolling_slope = df[column].rolling(window=window).apply(calc_slope_pct, raw=False)
    
    return rolling_slope

def save_csv(df, path=None, index=False, date_cols=()):
    d = df.copy()
    # Datumsspalten ins Format TT.MM.JJJJ konvertieren
    for c in date_cols:
        if c in d:
            d[c] = pd.to_datetime(d[c], errors='coerce').dt.strftime('%d.%m.%Y')
    # Speichern mit deutschem Separator und Encoding
    return d.to_csv(
        path,
        sep=";",
        decimal=",",
        index=index,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

def load_csv(path, index=False):
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        index_col=0 if index else None,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    # erst Date erzeugen
    df["Price"] = df["Close"]
    df["Date"] = pd.to_datetime(df["Close time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # dann droppen und zurückgeben
    columns_to_drop = [
        "Open time", "Open", "High", "Low", "Close", "Close time",
        "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    return df

def calculate_ma(df, column, window=20):
    if column not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return df[column].rolling(window=window).mean()

def calculate_rsi(df, window=14):
    delta = df['Price'].diff()  # Calculate price changes
    gain = delta.where(delta > 0, 0)  # Gains (positive changes)
    loss = -delta.where(delta < 0, 0)  # Losses (negative changes)

    # Moving average for gains and losses
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # RSI calculation
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Ein unveränderter Kurs hat neutralen RSI statt 0/0 (= NaN).
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def calculate_momentum(df, column, window=10):
    momentum = df[column] - df[column].shift(window)
    return momentum

def calculate_vwma(df, window=20):
    if 'Volume' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    pv = df['Price'] * df['Volume']
    return pv.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()

def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'B', 'T'][magnitude])

def human_format_euro(num, pos):
    return human_format(num, pos) + '€'

def plot_and_save(df, symbol, data_type, zero_line=None, sr_zones=None):
    if df is None or df.empty:
        raise ValueError("Kein Diagramm ohne Kursdaten möglich.")
    if df is not None and not df.empty:
    #----------Create
        fig, (ax1, ax3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax2 = ax1.twinx()
        #fig.patch.set_alpha(0)
        # Fortlaufende Nummer statt Kalenderdatum: Wochenenden und Nächte ohne Handel
        # erzeugen so keine Lücken, jeder Balken ist gleich breit.
        x = pd.Series(np.arange(len(df)), index=df.index)
        dates = pd.to_datetime(df['Date']).reset_index(drop=True)

    #-----------Background
        df_min, df_max = df['Price'].min(), df['Price'].max()
        if zero_line is not None and zero_line != 0:
            ax1.set_facecolor('#d0f0d0' if zero_line < df['Price'].iloc[-1] else '#f0d0d0')
            df['Percentage Deviation'] = df['Price']/zero_line * 100 - 100
            if df_min <= zero_line <= df_max:
                ax1.axhline(y=zero_line, linewidth=1, linestyle='-', color='0.5', label=f'Buy at {round(zero_line,2)}€')
        else:
            ax1.set_facecolor('#f0f0f0')
            mean_price = df['Price'].mean()
            df['Percentage Deviation'] = df['Price']/mean_price * 100 - 100

    #---------Axis set-up
        from matplotlib.ticker import MaxNLocator
        # Markierung beim ersten Wert jedes neuen Tages (100h) bzw. Monats (100d).
        period = dates.dt.strftime('%Y-%m-%d' if data_type=='100h' else '%Y-%m')
        ticks = [i for i in range(1, len(dates)) if period.iloc[i] != period.iloc[i-1]]
        # Bei vielen Markierungen (z.B. Aktien-Stundenwerte) nur jede zweite beschriften,
        # sonst überlagern sich die Datumsangaben. Die senkrechten Linien bleiben alle.
        labeled = set(ticks[::2]) if len(ticks) > 8 else set(ticks)
        for ax in [ax1, ax2, ax3]:
            #ax1.set_title('100 Hours' if data_type=='100h' else '100 Days', fontsize=14)
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda value, pos: dates.iloc[int(round(value))].strftime('%d.%m') if int(round(value)) in labeled else ''))
            ax.grid(True, axis='x', color='black', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax3.grid(False)  # Volumenbalken ohne senkrechte Linien besser sichtbar

        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
        # Prozent-Achse links anzeigen, Kurswerte-Achse ausblenden.
        ax2.yaxis.tick_left()
        ax1.tick_params(axis='y', left=False, labelleft=False)
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(human_format))
        
    #--------------Drawing
        ax1.fill_between(
            x,
            df['BB_Lower'], 
            df['BB_Upper'], 
            color='grey', 
            alpha=0.2, 
            label='Bollinger-Bereich'
        )
        # Kurs mit einbeziehen, sonst wird ein Ausbruch aus dem Band abgeschnitten.
        lower = min(df['BB_Lower'].min(), df['Price'].min())
        upper = max(df['BB_Upper'].max(), df['Price'].max())
        if lower == upper:
            padding = max(abs(lower) * 0.01, 0.01)
            lower, upper = lower - padding, upper + padding
        ax1.set_ylim(lower, upper)

        # Rechts 20% der Breite frei lassen, damit das Volume-Profile den Kurs nicht überlagert.
        vp_share = 0.2
        ax1.set_xlim(-1, -1 + (len(df) + 1) / (1 - vp_share - 0.02))

        # Volume-Profile (wie TradingView "sichtbare Range"): Volumen je Preisstufe,
        # rechts im Hintergrund. Breite in Anteil der Diagrammbreite, Höhe in Euro.
        if 'Volume' in df.columns:
            profile, edges = np.histogram(df['Price'], bins=24, range=(lower, upper), weights=df['Volume'])
            if profile.max() > 0:
                trans = mtransforms.blended_transform_factory(ax1.transAxes, ax1.transData)
                widths = profile / profile.max() * vp_share      # längster Balken füllt den freien Bereich
                vp_colors = ['0.45' if i == profile.argmax() else '0.6' for i in range(len(profile))]  # POC etwas dunkler
                ax1.barh(edges[:-1], widths, height=np.diff(edges) * 0.9, left=1 - widths, align='edge',  # am rechten Rand ausgerichtet
                         color=vp_colors, alpha=0.3, linewidth=0, transform=trans, zorder=0)
                # Schwarzer Strich am Ende jedes Balkens: Länge bleibt auch unter Beschriftungen erkennbar.
                filled = profile > 0
                ax1.vlines((1 - widths)[filled], edges[:-1][filled], (edges[:-1] + np.diff(edges) * 0.9)[filled],
                           colors='black', linewidth=0.8, transform=trans, zorder=6)

        # Support/Resistance: Band mit Nummer (R1/S1), ausführlicher Text in der Legende oben rechts.
        if sr_zones is not None:
            handles = []
            for zone, name in zip(sr_zones, ('R1', 'S1')):
                if zone is None:
                    continue
                color = '#c0392b' if name == 'R1' else '#1e8449'
                alpha = 0.22 if zone['touches'] >= 2 else 0.12   # schwache Zone (1 Berührung) heller
                pct = (zone['center'] / df['Price'].iloc[-1] - 1) * 100
                text = f"{name} {human_format_euro(zone['center'], None)} ({pct:+.1f} %)"
                if zone['touches']:
                    text += f" · {zone['touches']}×"
                else:
                    text += " · Hoch Zeitraum" if name == 'R1' else " · Tief Zeitraum"
                if zone['high'] >= lower and zone['low'] <= upper:
                    ax1.axhspan(zone['low'], zone['high'], xmax=1 - vp_share - 0.02, color=color, alpha=alpha, linewidth=0, zorder=0.5)
                    ax1.text(1 - vp_share - 0.015, zone['center'], name, transform=ax1.get_yaxis_transform(), fontsize=7,
                             color=color, fontweight='bold', va='center', ha='left', zorder=5)
                else:
                    # Zone außerhalb des Bildes: Nummer mit Pfeil am oberen/unteren Rand
                    y, arrow = (0.97, '▲') if zone['center'] > upper else (0.03, '▼')
                    ax1.text(1 - vp_share - 0.015, y, f"{arrow} {name}", transform=ax1.transAxes, fontsize=7,
                             color=color, fontweight='bold', va='center', ha='left', zorder=5)
                handles.append(mpatches.Patch(color=color, alpha=alpha + 0.1, label=text))
            if handles:
                # Legende über dem Diagramm: verdeckt weder Kurs noch Volume-Profile.
                ax1.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 1), ncol=len(handles),
                           fontsize=7, frameon=False, handlelength=1, borderaxespad=0.2)

        min_remap = linear_remap(
            lower,
            df['Price'].min(),
            df['Price'].max(),
            df['Percentage Deviation'].min(),
            df['Percentage Deviation'].max()
        )

        max_remap = linear_remap(
            upper,
            df['Price'].min(),
            df['Price'].max(),
            df['Percentage Deviation'].min(),
            df['Percentage Deviation'].max()
        )

        if min_remap == max_remap:
            min_remap, max_remap = min_remap - 1, max_remap + 1
        ax2.set_ylim(min_remap, max_remap)
        
        ax1.plot(x, df['Price'], label='Price', color='k', linewidth=1.5)
        ax1.plot(x, df['Fast EMA'], label='Fast EMA', linestyle=':', color='green')
        ax1.plot(x, df['Slow EMA'], label='Slow EMA', linestyle=':', color='red')
        ax1.plot(x[df['Position']== 1.0], df.loc[df['Position']== 1.0, 'Fast EMA'], '^',  markersize=10, color='green', label='Buy')
        ax1.plot(x[df['Position']==-1.0], df.loc[df['Position']==-1.0, 'Fast EMA'], 'v',  markersize=10, color='red', label='Sell')
        ax1.plot(x[df['HighP']==1],      df.loc[df['HighP']==1, 'Fast EMA'], 'o', markersize=7, label='Max')
        ax1.plot(x[df['LowP']==1],       df.loc[df['LowP']==1,  'Fast EMA'], 'o', markersize=7, label='Min')

        ax2.plot(x, df['Percentage Deviation'], color='k', linewidth=1.0, label='Deviation (%)')

        if 'Volume' in df.columns:
            colors = ['green' if close > open_val else 'red'
                    for open_val, close in zip(df['Price'].shift(1), df['Price'])]
            colors[0] = 'gray'
            # Abstand zwischen zwei Werten ist immer 1 -> Balken 80% davon
            ax3.bar(x, df['Volume'],
                    width=0.8, align='center',
                    color=colors, alpha=0.7, linewidth=0)

            ax3.margins(x=0.01)  # etwas Rand, damit nichts abgeschnitten wird

    #-------------Save
        os.makedirs(output_dir, exist_ok=True)
    # Set figure background to white
    fig.patch.set_facecolor('white')
    # dpi=200 ergibt ca. 1200 Pixel Breite, passend zur Telegram-Maximalgröße von 1280 Pixeln.
    plt.savefig(os.path.join(output_dir, f"{symbol}.png"), dpi=200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close(fig)

def set_message_queue(queue: mp.Queue):
    global _message_queue
    _message_queue = queue

def pushover(message: str, priority: int = 0, html_format=False):
    log_print(message)
    if _message_queue:
        _message_queue.put({'type': 'text', 'message': message, 'priority': priority, 'html': html_format})
    else:
        log_print("No queue – message discarded.")

def pushover_image(symbol: str, message: str, priority: int = 0, delete_after_send=False, html_format=False):
    # html_format=True: Nachricht enthält Telegram-HTML (<b>, <i>, <pre>)
    if _message_queue:
        _message_queue.put({'type': 'image', 'symbol': symbol, 'message': message, 'priority': priority, 'output_dir': output_dir, 'delete_after_send': delete_after_send, 'html': html_format})
    else:
        log_print("No queue – image discarded.")

def google_trends(search):
    
    for i in range(5):
        try:
            pytrends = TrendReq(hl='en-US', tz=360) 
            break
        except Exception as e:
            infos = {'tries': int(i), 'function': __name__,'exception': e}
            msg = "Script try number %(tries)s to %(function)s \n\n%(exception)s" %infos
            pushover(msg)
            if i == 4:
                raise
            time.sleep(300)
    
    kw_list = [search] # List of keywords to get data
    pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m') # Trend for 12 months
    data = pytrends.interest_over_time() 
    data = data.reset_index() 

    df = pd.DataFrame()
    df = pd.DataFrame(columns=['Date', 'Interest'])
    df['Date'] = data['date']
    df['Interest'] = data[search]

    if len(df) < 2:
        raise ValueError('Zu wenige Google-Trends-Daten.')
    return df['Interest'].iloc[-1], df['Interest'].iloc[-2]

def seven_day_slope_pct(df, current):

    if len(df) < 8:
        log_print(f"seven_day_slope_pct: DataFrame has only {len(df)} rows, need at least 8")
        return 0.0
    
    try:
        # ✅ Problem 4a Fix: Explizites iloc
        if current:
            y = df['Price'].iloc[-7:].values
        else:
            y = df['Price'].iloc[-8:-1].values
        
        # Validierung
        if len(y) != 7:
            log_print(f"seven_day_slope_pct: Expected 7 values, got {len(y)}")
            return 0.0
            
        if np.any(np.isnan(y)):
            log_print("seven_day_slope_pct: NaN values in price data")
            return 0.0
        
        # ✅ Problem 4b Fix: Direktes np.arange(7) statt index-Reset
        # x = [0, 1, 2, 3, 4, 5, 6] unabhängig vom DataFrame-Index
        slope, intercept = linregress(np.arange(7), y)[:2]
        
        # Werte an Tag 0 und Tag 6
        y_start = intercept
        y_end = slope * 6 + intercept
        
        if y_start == 0:
            log_print("seven_day_slope_pct: y_start is 0")
            return 0.0
        
        return (y_end / y_start - 1) * 100
        
    except Exception as e:
        # ✅ Problem 4c Fix: Graceful Fehlerbehandlung
        log_print(f"seven_day_slope_pct: Error - {e}")
        return 0.0

def calc_stat_limits(df, column, window=100, invert=False):
    if column not in df.columns:
        return np.nan, np.nan

    if df.empty or not np.isfinite(df[column].iloc[-1]):
        return np.nan, np.nan
    data = df[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()[-window:]
    if data.size < 2:
        return np.nan, np.nan
    
    if np.all(data == data[0]):
        return 0, 50.0
    # Calculate quantile (percentile) of current_value in data
    quantile = float(np.sum(data <= df[column].iloc[-1])) / len(data)
    quantile_pct = round(quantile * 100, 2)

    if invert:
        quantile_pct = 100 - quantile_pct

    if quantile_pct <= 10:
        sig_count = -3
    elif quantile_pct <= 20:
        sig_count = -2
    elif quantile_pct <= 30:
        sig_count = -1
    elif quantile_pct >= 90:
        sig_count = 3
    elif quantile_pct >= 80:
        sig_count = 2
    elif quantile_pct >= 70:
        sig_count = 1
    else:
        sig_count = 0

    return sig_count, quantile_pct


def analysis_details(df):
    """Gemeinsame Kennwerte für Alarm und /check; diese Funktion versendet nichts."""
    signals, quantiles = {}, {}
    for column in ('RSI14', 'SMA7', 'VMA7', 'ROC5', 'VROC5'):
        count, quantile = calc_stat_limits(df, column, window=100, invert=(column == 'RSI14'))
        # Fehlendes Volumen ergibt kein verwertbares Momentum: neutral bewerten.
        # Ein fehlender Preisindikator muss dagegen als Datenfehler auffallen.
        if not math.isfinite(count) or not math.isfinite(quantile):
            if column in ('VMA7', 'VROC5'):
                count, quantile = 0, float('nan')
            else:
                raise ValueError(f"Keine ausreichenden Daten für {column}.")
        signals[column], quantiles[column] = int(count), quantile
    text1, score = tech_analyse1(signals['RSI14'], 0)
    text2, score = tech_analyse2(signals['ROC5'], signals['VROC5'], score)
    text3, score = tech_analyse3(signals['SMA7'], signals['VMA7'], score)
    score += signals['ROC5']

    def line(column):
        count, value = signals[column], quantiles[column]
        if not math.isfinite(value):
            return f"{column:<6} keine Daten (neutral gewertet)"
        marks = '+' * count if count > 0 else '-' * abs(count) if count < 0 else '●'
        return f"{column:<6}{value:>7.2f} %  {marks}"   # feste Breite: Werte stehen untereinander

    def block(lines, text):
        # Telegram-HTML: Werte in Festbreitenschrift, Überschrift fett, Einschätzung kursiv.
        # "Trend-Analysis: ..." wird geteilt, damit der lange Satz in eigener Zeile steht.
        label, sentence = text.split(': ', 1) if ': ' in text else ('', text)
        head = f"<b>{html.escape(label)}</b>\n" if label else ''
        return f"{head}<pre>{html.escape(chr(10).join(lines))}</pre>\n<i>{html.escape(sentence)}</i>"

    message = (f"<b>Score: {score}</b>\n\n"
               f"{block([line('RSI14')], text1)}\n\n"
               f"{block([line('SMA7'), line('VMA7')], text3)}\n\n"
               f"{block([line('ROC5'), line('VROC5')], text2)}")
    # Einschätzungen ohne Überschrift (z.B. für die KI-Frage bei /ai)
    texts = {'rsi': text1, 'trend': text3.split(': ', 1)[-1], 'short': text2.split(': ', 1)[-1]}
    return {'score': score, 'quantiles': quantiles, 'message': message, 'texts': texts}


def alarm(df, symbol, watch_list, current_profit_pct, amount_older_than_one_year, amount_older_than_one_year_pct, data_type):

    alarms = {}
    tech_indicators = None  
    score = 0

    details = analysis_details(df)
    score = details['score']
    tech_indicators = details['quantiles']
    current_EMA_diff_pct = (df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-1]) / df['Slow EMA'].iloc[-1] * 100
    yesterday_EMA_diff_pct = (df['Fast EMA'].iloc[-2] - df['Slow EMA'].iloc[-2]) / df['Slow EMA'].iloc[-2] * 100

    def create_alarm(code, buy_sell, indicator, value):
        priority = 1 if (score > 5 and buy_sell == "Buy") or (score < -5 and buy_sell == "Sell") else 0
        # Telegram-HTML (fett/kursiv/Festbreite), deshalb Texte mit html.escape absichern
        message = (
            f"<b>{html.escape(symbol)} — {buy_sell}</b>\n"
            f"Trigger: {html.escape(indicator)} ({data_type})\n\n"
            + details['message']
        )
        if not watch_list:
            message += "\n\n" + html.escape(f"Amount >1 year: {amount_older_than_one_year:.2f} ({amount_older_than_one_year_pct:.2f} %)")
        alarms[code] = {"value": value, "priority": priority, "msg": message}

    # Nur das Datum: CoinGecko liefert als letzten Wert die aktuelle Uhrzeit.
    value_ts = int(pd.Timestamp(df['Date'].iloc[-1]).normalize().timestamp())

    # Portfolio Alarme (nicht Watchlist)
    if not watch_list:
        # BUY Signals
        if yesterday_EMA_diff_pct < 0 and current_EMA_diff_pct > 0 and data_type == "100d":
            create_alarm("101", "Buy", "Cross-EMA", value_ts)
        
        if df['ROC5'].iloc[-1] > seven_day_price_change:
            create_alarm("121", "Buy", "Positiv-5", df['ROC5'].iloc[-1])
        
        if df['LowP'].iloc[-1] < df['LowP'].iloc[-2] and df['LowP'].iloc[-2] > df['LowP'].iloc[-3] and data_type == "100d":
            create_alarm("122", "Buy", "100-Minimum", value_ts)

        # SELL Signals
        if yesterday_EMA_diff_pct > 0 and current_EMA_diff_pct < 0 and data_type == "100d":
            create_alarm("201", "Sell", "Cross-EMA", value_ts)
        
        if df['ROC5'].iloc[-1] < (seven_day_price_change * -1):
            create_alarm("221", "Sell", "Negative-5", df['ROC5'].iloc[-1])
        
        if df['HighP'].iloc[-1] < df['HighP'].iloc[-2] and df['HighP'].iloc[-2] > df['HighP'].iloc[-3] and data_type == "100d":
            create_alarm("222", "Sell", "100-Maximum", value_ts)

    return alarms, tech_indicators, score

def linear_remap(value, old_min, old_max, new_min, new_max):
    """Mappt value linear von [old_min, old_max] nach [new_min, new_max].
       Wenn old_min == old_max wird der Mittelwert des Zielintervalls zurückgegeben."""
    if old_max == old_min:
        return 0.5 * (new_min + new_max)
    t = (value - old_min) / (old_max - old_min)
    return new_min + t * (new_max - new_min)

    # Analysis 1: RSI14
def tech_analyse1(RSI14, score):
    # Probability Matrix
    PROBABILITY_MATRIX = {
        -3: "Very likely correction to fall.",
        -2: "Possible correction to fall.",
        -1: "Potentially correction to fall.",
         0: "Neutral.",
         1: "Potentially correction to rise.",
         2: "Possible correction to rise.",
         3: "Very likely correction to rise."
    }

    # Werte aus den Matrizen holen
    analysis = PROBABILITY_MATRIX.get(RSI14, "error")
    
    # Score aktualisieren
    score += RSI14
    
    return analysis, score

    # Analysis 2: MOM10 and VMOM10
def tech_analyse2(ROC5, VROC5, score):
    # Volume Support Matrix
    VOLUME_MATRIX = {
        -3: "Very low volume supported", 
        -2: "Low volume supported", 
        -1: "Decreased volume supported",
         0: "Average volume supported",
         1: "Increased volume supported",
         2: "High volume supported",
         3: "Very high volume supported"
    }
    
    # Momentum Movement Matrix
    MOMENTUM_MATRIX = {
        -3: "strong slowdown",
        -2: "slowdown",
        -1: "slight slowdown",
         0: "steady movement",
         1: "slight acceleration",
         2: "acceleration",
         3: "strong acceleration"
    }
    
    # Werte aus den Matrizen holen (mit Fallback)
    volume_text = VOLUME_MATRIX.get(VROC5, "error")
    momentum_text = MOMENTUM_MATRIX.get(ROC5, "error")

    if ROC5 < 0: 
        VROC5_score = VROC5 * -1  
    else:
        VROC5_score = VROC5 

    if  VROC5 < 0:
        Tscore = 0
        fake_out = " (possible fake-out)"
    else:
        Tscore = ROC5 + VROC5_score
        fake_out = ""

    score = score + Tscore


    
    return f"Short-term-Analysis: {volume_text} {momentum_text}{fake_out}.", score

    # Analysis 3: SMA7 and VMA7
def tech_analyse3(SMA7, VMA7, score):
    # Volume Matrix
    VOLUME_MATRIX = {
        -3: "Very low volume",
        -2: "Low volume",
        -1: "Decreased volume",
         0: "Average volume",
         1: "Increased volume",
         2: "High volume",
         3: "Very high volume"
    }
    
    # Trend Strength Matrix (based on absolute value)
    TREND_MATRIX = {
        -3: "with strong negative trend",
        -2: "with negative trend",
        -1: "with slight negative trend",
        0: "without trend",
        1: "with slight trend",
        2: "with trend",
        3: "with strong trend"
    }
    
    # Werte aus den Matrizen holen (mit Fallback)
    volume_text = VOLUME_MATRIX.get(VMA7, "error")
    trend_text = TREND_MATRIX.get(SMA7, "error")

    if SMA7 < 0: 
        VMA7_score = VMA7 * -1  
    else:
        VMA7_score = VMA7 

    if  VMA7 < 0:
        Tscore = 0
        fake_out = " (possible fake-out)"
    else:
        Tscore = SMA7 + VMA7_score
        fake_out = ""
    
    score = score + Tscore

    return f"Trend-Analysis: {volume_text} {trend_text}{fake_out}.", score


def older_than_one_year(df):
    current_date = datetime.now()
    one_year_ago = current_date - timedelta(days=365)
    filtered_df = df[df['Date'] < one_year_ago]
    total_amount = filtered_df['Amount'].sum()
    
    return total_amount

def euro_de(value):
    # deutsches Zahlenformat: 71.660,12 €
    return f"{value:,.2f} €".replace(',', 'X').replace('.', ',').replace('X', '.')

def create_table_image(df, output_path, title='', subtitle='', dpi=200):
    """Tagesübersicht als moderne Rangliste: sortiert nach 1d P&L, beste Zeile hervorgehoben,
    1d P&L zusätzlich als Balken (grün nach rechts, rot nach links)."""
    GREEN, RED, GREY, DARK = '#1e8449', '#c0392b', '#8a8a8a', '#222222'

    def pct_color(value):
        return GREEN if value > 0 else RED if value < 0 else GREY

    df = df.sort_values('1d P&L [%]', ascending=False).reset_index(drop=True)
    rows = len(df)
    fig, ax = plt.subplots(figsize=(6.4, 0.95 + rows * 0.42))
    ax.set_xlim(0, 1)
    ax.set_ylim(rows + 1.6, -0.2)   # y nach unten: Titel oben, dann Kopfzeile, dann Zeilen
    ax.axis('off')

    # Spalten (x-Position, Ausrichtung)
    col = {'rank': (0.03, 'center'), 'name': (0.07, 'left'), 'price': (0.43, 'right'),
           'bar': (0.47, 0.71), 'd1': (0.81, 'right'), 'd7': (0.91, 'right'), 'rating': (0.965, 'center')}

    # Titel
    ax.text(0.0, 0.25, title, fontsize=12, fontweight='bold', color=DARK, va='center')
    ax.text(1.0, 0.25, subtitle, fontsize=7.5, color=GREY, va='center', ha='right')

    # Kopfzeile
    header_y = 1.0
    for key, label in (('rank', '#'), ('name', 'NAME'), ('price', 'KURS'), ('d1', '1D P&L'),
                       ('d7', '7D'), ('rating', 'RATING')):
        x, align = col[key]
        ax.text(x, header_y, label, fontsize=6.5, color=GREY, fontweight='bold', ha=align, va='center')
    ax.plot([0, 1], [header_y + 0.35, header_y + 0.35], color='#dddddd', linewidth=0.8)

    # Balkenlänge: größter Ausschlag füllt die halbe Balkenspalte
    bar_left, bar_right = col['bar']
    bar_mid = (bar_left + bar_right) / 2
    max_abs = max(df['1d P&L [%]'].abs().max(), 0.01)

    for i, row in df.iterrows():
        y = header_y + 1 + i
        d1, d7, rating = row['1d P&L [%]'], row['7d P&L [%]'], int(row['Rating'])

        # Hintergrund: beste Zeile (mit Gewinn) grün getönt, sonst abwechselnd hell
        if i == 0 and d1 > 0:
            background = '#e9f6ec'
        else:
            background = '#f7f7f7' if i % 2 else 'white'
        ax.add_patch(mpatches.FancyBboxPatch((0, y - 0.42), 1, 0.84, boxstyle='round,pad=0,rounding_size=0.01',
                                             facecolor=background, edgecolor='none', zorder=0))

        ax.text(col['rank'][0], y, str(i + 1), fontsize=8, color=GREY, ha='center', va='center')
        ax.text(col['name'][0], y, row['Name'], fontsize=9, color=DARK, fontweight='bold', ha='left', va='center')
        ax.text(col['price'][0], y, euro_de(row['Piece [€]']), fontsize=8.5, color=DARK, ha='right', va='center')

        # 1d P&L als Balken ab der Mitte
        length = d1 / max_abs * (bar_right - bar_left) / 2
        ax.add_patch(mpatches.Rectangle((min(bar_mid, bar_mid + length), y - 0.15), abs(length), 0.3,
                                        facecolor=pct_color(d1), edgecolor='none', alpha=0.85, zorder=1))
        ax.plot([bar_mid, bar_mid], [y - 0.3, y + 0.3], color='#cccccc', linewidth=0.8, zorder=2)

        ax.text(col['d1'][0], y, f"{d1:+.2f} %", fontsize=9, color=pct_color(d1), fontweight='bold', ha='right', va='center')
        ax.text(col['d7'][0], y, f"{d7:+.1f} %", fontsize=7.5, color=pct_color(d7), ha='right', va='center')

        # Rating als kleines farbiges Feld
        ax.text(col['rating'][0], y, f"{rating:+d}" if rating else '0', fontsize=7.5, color='white', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=pct_color(rating), edgecolor='none'))

    fig.patch.set_facecolor('white')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)




def log_print(message: str):
    print(message)
    log_entry = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  " + str(message) + "\n"
    with open('logfile.txt', 'a', encoding='utf-8') as file:
        file.write(log_entry)


def filter(alarm_neu, state):

    change = {}
    for key, new_info in alarm_neu.items():
        new_value = new_info["value"]
        if key in state.alarm_prev:
            prev_value = state.alarm_prev[key]["value"]
            if abs(prev_value - new_value) > previous_alarm_change:
                change[key] = new_info 
                state.alarm_prev[key] = new_info 
        else:
            state.alarm_prev[key] = new_info
            change[key] = new_info
    return change