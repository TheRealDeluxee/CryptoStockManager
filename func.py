# --- Standard Library ---
import os
import time
import base64, hashlib, hmac
from datetime import datetime, timedelta
from urllib.parse import quote  # only if you need to encode URL parts
import urllib.request, json
import random
import math

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
import multiprocessing as mp

_message_queue = None
one_day_price_change = 0.0
seven_day_price_change = 0.0
one_day_profit_limit = 0.0
output_dir = 'plots'
previous_alarm_change = 0.0

def get_stock(symbol, period):
    df = pd.DataFrame()
    
    if period == '1mo':
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
    
    df = df.tail(100).copy()
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
    df = df.sort_values('Date').drop_duplicates('Date', keep='last').tail(100)
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

    # Calculate min and max
    df['High'] = df['Fast EMA'].max()
    df['Low'] = df['Fast EMA'].min()
    df['HighP'] = np.where(df['Fast EMA'] == df['High'], 1, 0)
    df['LowP'] = np.where(df['Fast EMA'] == df['Low'], 1, 0)

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

def plot_and_save(df, symbol, data_type, zero_line=None):
    if df is None or df.empty:
        raise ValueError("Kein Diagramm ohne Kursdaten möglich.")
    if df is not None and not df.empty:
    #----------Create
        fig, (ax1, ax3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax2 = ax1.twinx()
        #fig.patch.set_alpha(0)

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
        for ax in [ax1, ax2, ax3]:
            #ax1.set_title('100 Hours' if data_type=='100h' else '100 Days', fontsize=14)
            ax.xaxis.set_major_locator(mdates.DayLocator() if data_type=='100h' else mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
            ax.grid(True, axis='x', color='black', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(human_format_euro))
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(human_format))
        
    #--------------Drawing
        ax1.fill_between(
            df['Date'], 
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
        
        ax1.plot(df['Date'], df['Price'], label='Price', color='k', linewidth=1.5)
        ax1.plot(df['Date'], df['Fast EMA'], label='Fast EMA', linestyle=':', color='green')
        ax1.plot(df['Date'], df['Slow EMA'], label='Slow EMA', linestyle=':', color='red')
        ax1.plot(df.loc[df['Position']== 1.0, 'Date'], df.loc[df['Position']== 1.0, 'Fast EMA'], '^',  markersize=10, color='green', label='Buy')
        ax1.plot(df.loc[df['Position']==-1.0, 'Date'], df.loc[df['Position']==-1.0, 'Fast EMA'], 'v',  markersize=10, color='red', label='Sell')
        ax1.plot(df.loc[df['HighP']==1, 'Date'],      df.loc[df['HighP']==1, 'Fast EMA'], 'o', markersize=7, label='Max')
        ax1.plot(df.loc[df['LowP']==1,  'Date'],      df.loc[df['LowP']==1,  'Fast EMA'], 'o', markersize=7, label='Min')

        ax2.plot(df['Date'], df['Percentage Deviation'], color='k', linewidth=1.0, label='Deviation (%)')

        if 'Volume' in df.columns:
            colors = ['green' if close > open_val else 'red'
                    for open_val, close in zip(df['Price'].shift(1), df['Price'])]
            colors[0] = 'gray'
            dates_num = mdates.date2num(df['Date'])               # Datum -> float (Tage)
            if len(dates_num) > 1:
                delta_days = float(np.median(np.diff(dates_num))) # typischer Abstand
            else:
                delta_days = 1.0
            bar_width = delta_days * 0.8                          # 80% des Abstands

            ax3.bar(dates_num, df['Volume'],
                    width=bar_width, align='center',
                    color=colors, alpha=0.7, linewidth=0)

            ax3.xaxis_date()  # Achse wieder als Datum formatieren
            ax3.margins(x=0.01)  # etwas Rand, damit nichts abgeschnitten wird 

    #-------------Save
        os.makedirs(output_dir, exist_ok=True)
    # Set figure background to white
    fig.patch.set_facecolor('white')
    plt.savefig(os.path.join(output_dir, f"{symbol}.png"), bbox_inches='tight', facecolor='white', transparent=False)
    plt.close(fig)

def set_message_queue(queue: mp.Queue):
    global _message_queue
    _message_queue = queue

def pushover(message: str, priority: int = 0):
    log_print(message)
    if _message_queue:
        _message_queue.put({'type': 'text', 'message': message, 'priority': priority})
    else:
        log_print("No queue – message discarded.")

def pushover_image(symbol: str, message: str, priority: int = 0, delete_after_send=False):
    if _message_queue:
        _message_queue.put({'type': 'image', 'symbol': symbol, 'message': message, 'priority': priority, 'output_dir': output_dir, 'delete_after_send': delete_after_send})
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
            return f"{column}: keine Daten (neutral gewertet)"
        marks = '+' * count if count > 0 else '-' * abs(count) if count < 0 else '●'
        return f"{column}: {value:.2f} % {marks}"

    message = (f"📊 Score: {score}\n{'─' * 5}\n"
               f"{line('RSI14')}\n{text1}\n\n"
               f"{line('SMA7')}\n{line('VMA7')}\n{text3}\n\n"
               f"{line('ROC5')}\n{line('VROC5')}\n{text2}")
    return {'score': score, 'quantiles': quantiles, 'message': message}


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
        emoji = "🟢" if buy_sell == "Buy" else "🔴"
        priority = 1 if (score > 5 and buy_sell == "Buy") or (score < -5 and buy_sell == "Sell") else 0
        message = (
            f"{emoji} {symbol} — {buy_sell}\n"
            f"🔔 Trigger: {indicator} ({data_type})\n"
            + details['message']
        )
        if not watch_list:
            message += f"\n📅 Amount >1 year: {amount_older_than_one_year:.2f} ({amount_older_than_one_year_pct:.2f} %)"
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

def create_table_image(
    df,
    output_path,
    body_fontsize=7,
    header_fontsize=5.5,
    header_color="#3A778A",
    fig_width=4.5,
    row_height=0.45,
    left_col_width=0.22,
    other_col_width=0.13,
    dpi=200
):
    # Height: Data rows + some space for header
    fig_height = len(df) * row_height + 0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # Color header using colColours (robust with edges='horizontal')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
        edges="horizontal",
        colColours=[header_color] * len(df.columns),
    )

    # Set column widths
    col_widths = [left_col_width] + [other_col_width] * (len(df.columns) - 1)
    total_rows = len(df) + 1  # +1 wegen Header
    for j in range(total_rows):
        for i, w in enumerate(col_widths):
            table[(j, i)].set_width(w)

    # Fonts
    table.auto_set_font_size(False)
    table.set_fontsize(body_fontsize)

    # Format header text
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_text_props(weight="bold", color="white", fontsize=header_fontsize)
        cell.visible_edges = "closed"  # optional: Header komplett umrandet

    # Slight scaling for denser layout
    table.scale(0.55, 1.0)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()




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