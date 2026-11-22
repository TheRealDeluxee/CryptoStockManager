# --- Standard Library ---
import os
import time
from datetime import datetime, timedelta
import random
import math
import html
import re
import warnings

# --- External Packages ---
import requests #instal
import pandas as pd #instal
import numpy as np # install
import yfinance as yf #Install
# yfinance itself still uses "inplace=True", which pandas 3.1 marks as deprecated. Harmless; only this
# one notice is hidden (CaSa's own code does not use inplace), all other warnings stay visible.
# Must come after "import yfinance": yfinance switches its own deprecation warnings on when imported.
warnings.filterwarnings('ignore', message=r'The inplace keyword .* is deprecated')
import matplotlib #Install
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import multiprocessing as mp

_message_queue = None
seven_day_price_change = 0.0
output_dir = 'plots'
previous_alarm_change = 0.0
total_loss_limit = -5.0   # loss limit alarm: total P&L of a position in % (config.ini [alertsconfig])
infos_for = 'all'   # infos for: all = every asset in the analysis, portfolio = only assets you hold
coingecko_api_key = ''   # optional, from config.ini [coingecko] api_key
anthropic_api_key = ''   # optional, from config.ini [anthropic] api_key (empty = Claude API not used)
anthropic_model = 'claude-opus-5-5'   # from config.ini [anthropic] model

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
            # auto_adjust=False: prices only split-adjusted, not dividend-adjusted (same as TradingView/Xetra)
            df = stock.history(period=period, interval=interval, timeout=20, auto_adjust=False)
            
            if not df.empty:
                break
            else:
                wait_time = wait_times[i] if i < len(wait_times) else 900
                # only in the log: a failed analysis is reported once at the end (_analyse_items)
                log_print(f"Yahoo {symbol}: empty response (attempt {i+1}/5)"
                          + (f", waiting {wait_time} s" if i < 4 else ""), 'WARN')
                if i < 4:
                    time.sleep(wait_time)
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Determine wait time based on error type
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                wait_time = wait_times[i] if i < len(wait_times) else 900
            else:
                wait_time = 30  # Shorter wait time for other errors
            
            # only in the log: a failed analysis is reported once at the end (_analyse_items)
            log_print(f"Yahoo {symbol}: {str(e).strip()[:150]} (attempt {i+1}/5)"
                      + (f", waiting {wait_time} s" if i < 4 else ""), 'WARN')
            
            if i < 4:  # Do not wait after the last attempt
                time.sleep(wait_time)
    
    if df.empty:
        # Optional: return None or raise an exception
        return None

    # Yahoo sometimes delivers the last day without price. If that day was already traded, its current
    # price stands in until the closing price comes; otherwise (night, before the opening) the day is skipped.
    df = _fill_today(stock, symbol, df)
    # Days without price are skipped instead of failing the whole analysis with
    # "Missing or invalid values in Price".
    missing = df['Close'].isna() | (df['Close'] <= 0)
    if missing.any():
        days = ', '.join(pd.to_datetime(df.index[missing]).strftime('%d/%m/%Y'))
        log_print(f"Yahoo {symbol}: {int(missing.sum())} day(s) without price skipped ({days})", 'WARN')
        df = df[~missing]
        if df.empty:
            return None
    df = df.assign(Volume=df['Volume'].fillna(0)) if 'Volume' in df else df

    if 'Date' not in df.columns:
        df['Date'] = df.index
    
    df = df.copy()  # keep the full history, cut_to_window() cuts it later
    df['Price'] = stock_prices_in_eur(stock, df, period)
    # Daily high/low in EUR with the same conversion factor (for support/resistance like on TradingView)
    factor = df['Price'] / df['Close']
    df['PriceHigh'] = df['High'] * factor
    df['PriceLow'] = df['Low'] * factor

    df_inverted = df.reset_index(drop=True)
    df_inverted = df_inverted.rename(columns={'index': 'Date'})
    
    columns_to_drop = ['Close', 'Adj Close', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits', 'Capital Gains']
    df_dropped = df_inverted.drop(columns=[col for col in columns_to_drop if col in df_inverted.columns])
    
    return df_dropped


def _fill_today(stock, symbol, df):
    """Last row without closing price: when Yahoo's last trade is from that day, use its current price
    (the closing price follows later). Otherwise the row stays empty and is skipped."""
    if df.empty or not pd.isna(df['Close'].iloc[-1]):
        return df
    try:
        info = stock.info
        price, traded = float(info['regularMarketPrice']), int(info['regularMarketTime'])
    except Exception:
        return df   # no current price: the day is skipped
    day = pd.Timestamp(df.index[-1])
    trade = pd.Timestamp(traded, unit='s', tz='UTC')
    trade = trade.tz_convert(day.tz) if day.tz is not None else trade.tz_localize(None)
    if trade.date() != day.date() or not math.isfinite(price) or price <= 0:
        return df
    df = df.copy()
    for column in ('Close', 'Open', 'High', 'Low', 'Adj Close'):
        if column in df and pd.isna(df[column].iloc[-1]):
            df.loc[df.index[-1], column] = price
    log_print(f"Yahoo {symbol}: no closing price yet for {day:%d/%m/%Y}, current price {price:.2f} used")
    return df


_fx_cache = {}   # (currency, period) -> (time, daily rates): one request per currency and hour, not per stock


def _fx_rates(currency, period):
    """Daily exchange rates currency -> EUR as DataFrame (Date, Rate), kept for one hour."""
    cached = _fx_cache.get((currency, period))
    if cached and time.time() - cached[0] < 3600:
        return cached[1]
    fx = yf.Ticker(f"{currency}EUR=X").history(period=period, interval='1d', timeout=20)
    if fx.empty:
        raise ValueError(f"No exchange rate available for {currency}/EUR.")
    rates = pd.DataFrame({'Date': pd.to_datetime(fx.index, utc=True).normalize(), 'Rate': fx['Close'].to_numpy()})
    rates = rates.dropna().sort_values('Date').drop_duplicates('Date', keep='last')
    _fx_cache[(currency, period)] = (time.time(), rates)
    return rates


def stock_prices_in_eur(stock, df, period):
    """Yahoo often returns USD/GBP. Our buys and charts are in EUR."""
    currency = stock.get_history_metadata().get('currency')
    if not currency:
        raise ValueError("The price currency could not be determined.")
    prices = df['Close'].astype(float)
    if currency == 'EUR':
        return prices
    # London partly quotes in pence instead of pounds.
    if currency in ('GBp', 'GBX'):
        currency = 'GBP'
        prices = prices / 100
    if len(currency) != 3 or not currency.isalpha():
        raise ValueError(f"Unsupported price currency: {currency}")
    rates = _fx_rates(currency.upper(), period)
    # Use the latest available daily exchange rate for each price day.
    dates = pd.to_datetime(df.index, utc=True).normalize()
    joined = pd.merge_asof(pd.DataFrame({'Date': dates}).sort_values('Date'), rates,
                           on='Date', direction='backward', tolerance=pd.Timedelta(days=7))
    if joined['Rate'].isna().any() or (joined['Rate'] <= 0).any():
        raise ValueError(f"Incomplete exchange rates for {currency}/EUR.")
    return prices * joined['Rate'].to_numpy()


def get_crypto(symbol, params_historical):
    url_historical = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
    # Optional free CoinGecko demo key (config.ini [coingecko] api_key): the limit then applies per key
    # (30 requests/minute) instead of per internet connection. Without a key: free public access.
    headers = {'x-cg-demo-api-key': coingecko_api_key} if coingecko_api_key else {}
    wait_times = [30, 60, 120, 300, 600]
    data = None

    for i in range(5):
        try:
            time.sleep(random.uniform(0.5, 2))
            response = requests.get(url_historical, params=params_historical, headers=headers, timeout=30)

            # Invalid key: retrying does not help
            if response.status_code in (401, 403) and coingecko_api_key:
                raise PermissionError("CoinGecko rejected the API key in config.ini ([coingecko] api_key).")
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
            
            reason = 'too many requests' if response.status_code == 429 else f'HTTP {response.status_code}'
            log_print(f"CoinGecko {symbol}: {reason} (attempt {i+1}/5)"
                      + (f", waiting {wait_time} s" if i < 4 else ""), 'WARN')
            if i < 4:
                time.sleep(wait_time)
                
        except PermissionError:
            raise   # wrong key: report immediately instead of waiting through all retries
        except Exception as e:
            wait_time = wait_times[min(i, len(wait_times)-1)]
            log_print(f"CoinGecko {symbol}: {str(e).strip()[:150]} (attempt {i+1}/5)"
                      + (f", waiting {wait_time} s" if i < 4 else ""), 'WARN')
            if i < 4:
                time.sleep(wait_time)
    
    if not data or 'prices' not in data or not data['prices']:
        log_print(f"CoinGecko {symbol}: no data after 5 attempts", 'ERROR')
        return None
    
    # Match prices and volumes by timestamp (do not zip blindly).
    prices = pd.DataFrame(data['prices'], columns=['Timestamp', 'Price'])
    volumes = pd.DataFrame(data.get('total_volumes', []), columns=['Timestamp', 'Volume'])
    df = prices.merge(volumes, on='Timestamp', how='left')
    df['Date'] = pd.to_datetime(df.pop('Timestamp'), unit='ms', utc=True).dt.tz_localize(None)
    df['Volume'] = df['Volume'].fillna(0)
    df = df.sort_values('Date').drop_duplicates('Date', keep='last')  # cut_to_window() cuts it later
    return df[['Date', 'Price', 'Volume']].reset_index(drop=True)


def calc_indicator_fuctions(df):
    if df is None or df.empty or len(df) < 26:
        raise ValueError("At least 26 price values are required.")
    for column in ('Price', 'Volume'):
        if column not in df or not np.isfinite(df[column].astype(float)).all():
            raise ValueError(f"Missing or invalid values in {column}.")
    if (df['Price'] <= 0).any() or (df['Volume'] < 0).any():
        raise ValueError("Prices must be positive and volumes at least zero.")
   
    # Calculate Fast EMA and Slow EMA
    df['Fast EMA'] = df['Price'].ewm(span=12, adjust=False).mean()
    df['Slow EMA'] = df['Price'].ewm(span=26, adjust=False).mean()
    df['Signal'] = 0.0
    df['Signal'] = np.where(df['Fast EMA'] > df['Slow EMA'], 1.0, 0.0)
    df['Position'] = df['Signal'].diff()
    df['Diff EMA'] = df['Fast EMA'] - df['Slow EMA']

    # Min and max are calculated later in cut_to_window() on the 100 values.

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

    # SMA200 as trend filter for the buy alarms (needs the long history, empty with less than 200 values)
    df['SMA200'] = df['Price'].rolling(200).mean()

    return df

def cut_to_window(df, window=100):
    # Indicators were calculated on the long history (no warm-up gap at the left edge).
    # Only the last 100 values are evaluated and plotted.
    df = df.tail(window).reset_index(drop=True).copy()

    # Calculate min and max (after cutting: stays the 100-day/-hour minimum/maximum)
    df['High'] = df['Fast EMA'].max()
    df['Low'] = df['Fast EMA'].min()
    df['HighP'] = np.where(df['Fast EMA'] == df['High'], 1, 0)
    df['LowP'] = np.where(df['Fast EMA'] == df['Low'], 1, 0)

    return df

def support_resistance(df, order=5):
    """Nearest resistance above and nearest support below the current price.
    Stocks: pivots from daily high/low (wicks, as drawn on TradingView); crypto: closing prices
    (CoinGecko has no high/low). df is the long history (250 days / 450 hours), not the 100-value window."""
    p = df['Price'].to_numpy(float)
    if len(p) < 2 * order + 2:
        return None, None
    has_wicks = 'PriceHigh' in df and 'PriceLow' in df and df[['PriceHigh', 'PriceLow']].notna().all().all()
    highs = df['PriceHigh'].to_numpy(float) if has_wicks else p
    lows = df['PriceLow'].to_numpy(float) if has_wicks else p

    # Pivot points: highest high / lowest low within ±order values
    pivots = []
    for i in range(order, len(p) - order):
        if highs[i] == highs[i - order:i + order + 1].max():
            pivots.append(highs[i])
        if lows[i] == lows[i - order:i + order + 1].min():
            pivots.append(lows[i])

    # Zone width from the typical move per value (hours narrower than days)
    width = max(2 * float(np.median(np.abs(np.diff(p) / p[:-1]))), 0.004)

    # Merge nearby pivot points into zones; a zone grows to at most 2 widths
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

    # Nearest zone above/below the price; its strength (touches) is shown in the legend
    price = p[-1]
    above = [zone for zone in zones if zone['low'] > price]
    below = [zone for zone in zones if zone['high'] < price]
    resistance = min(above, key=lambda zone: zone['low']) if above else None
    support = max(below, key=lambda zone: zone['high']) if below else None

    # Fallback: high/low of the loaded period (touches = 0)
    if resistance is None and highs.max() > price:
        resistance = {'low': highs.max(), 'high': highs.max(), 'center': highs.max(), 'touches': 0}
    if support is None and lows.min() < price:
        support = {'low': lows.min(), 'high': lows.min(), 'center': lows.min(), 'touches': 0}

    return resistance, support

def formation_signals(df, order=5, recent=30):
    """Signs of bottom and top formation (for the /ai prompt).
    df is the long history with indicators. Closing price, RSI and volume only."""
    p = df['Price'].to_numpy(float)
    rsi = df['RSI14'].to_numpy(float)
    v = df['Volume'].to_numpy(float)
    ema = df['Fast EMA'].to_numpy(float)
    n = len(p)
    # Pivot points; the latest ones need only 2 values after them so that fresh lows/highs count
    lows = [i for i in range(order, n - 2) if p[i] == p[i - order:i + order + 1].min()]
    highs = [i for i in range(order, n - 2) if p[i] == p[i - order:i + order + 1].max()]
    tolerance = max(2 * float(np.median(np.abs(np.diff(p) / p[:-1]))), 0.01)   # "same level"
    volume_median = float(np.median(v[-100:])) or 1.0

    def check(points, bottom):
        result = {}
        last_two = points[-2:] if len(points) >= 2 and points[-1] >= n - recent else []
        if last_two:
            a, b = last_two
            between = p[a:b + 1]
            result['folge'] = p[b] > p[a] if bottom else p[b] < p[a]            # higher lows / lower highs
            result['divergenz'] = bool(p[b] < p[a] and rsi[b] > rsi[a]) if bottom else bool(p[b] > p[a] and rsi[b] < rsi[a])
            same_level = abs(p[b] / p[a] - 1) <= tolerance
            gap = between.max() > max(p[a], p[b]) * (1 + 2 * tolerance) if bottom else between.min() < min(p[a], p[b]) * (1 - 2 * tolerance)
            result['doppel'] = bool(same_level and gap)                         # double bottom / double top
        else:
            result['folge'] = result['divergenz'] = result['doppel'] = None     # no recent low/high
        # Volume climax at the extreme of the last 20 values (capitulation / buying frenzy)
        k = n - 20 + int(np.argmax(v[-20:]))
        extreme = n - 20 + int(np.argmin(p[-20:]) if bottom else np.argmax(p[-20:]))
        direction = p[k] < p[k - 1] if bottom else p[k] > p[k - 1]
        result['volumen'] = bool(direction and v[k] >= 1.5 * volume_median and abs(k - extreme) <= 3)
        # Confirmation: price back above (bottom) or below (top) the EMA12
        result['bestaetigung'] = bool(p[-1] > ema[-1]) if bottom else bool(p[-1] < ema[-1])
        return result

    return check(lows, True), check(highs, False)

def zigzag(df, order=5):
    """Alternating highs (H) and lows (L) as [index, 'H'/'L', price].
    Stocks: from wicks (high/low), crypto: from closing prices."""
    p = df['Price'].to_numpy(float)
    has_wicks = 'PriceHigh' in df and 'PriceLow' in df and df[['PriceHigh', 'PriceLow']].notna().all().all()
    hi = df['PriceHigh'].to_numpy(float) if has_wicks else p
    lo = df['PriceLow'].to_numpy(float) if has_wicks else p
    raw = []
    for i in range(order, len(p) - 2):   # the latest points need only 2 values after them
        if hi[i] == hi[i - order:i + order + 1].max():
            raw.append([i, 'H', hi[i]])
        if lo[i] == lo[i - order:i + order + 1].min():
            raw.append([i, 'L', lo[i]])
    points = []
    for point in sorted(raw):
        if points and points[-1][1] == point[1]:   # two of the same kind in a row: keep the more extreme one
            if (point[1] == 'H' and point[2] > points[-1][2]) or (point[1] == 'L' and point[2] < points[-1][2]):
                points[-1] = point
        else:
            points.append(point)
    return points

def chart_pattern(df, recent=60):
    """Classic chart patterns like TradingView "Auto Chart Patterns" (stage 1):
    Double/Triple Top/Bottom and (Inverse) Head and Shoulders.
    df is the long history. Returns the youngest valid pattern (forming or fresh breakout) or None.
    Positions are stored as "values ago" so that the chart can use the 100-value window."""
    p = df['Price'].to_numpy(float)
    n = len(p)
    if n < 30:
        return None
    tol = max(2 * float(np.median(np.abs(np.diff(p) / p[:-1]))), 0.015)   # "same level"
    pts = zigzag(df)
    same = lambda a, b: abs(b / a - 1) <= tol
    found = []
    # Average true range (14 values): stocks from daily high/low, crypto from closing prices only
    has_wicks = 'PriceHigh' in df and 'PriceLow' in df and df[['PriceHigh', 'PriceLow']].notna().all().all()
    hi = df['PriceHigh'].to_numpy(float) if has_wicks else p
    lo = df['PriceLow'].to_numpy(float) if has_wicks else p
    true_range = np.maximum(hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - p[:-1]), np.abs(lo[1:] - p[:-1])))
    atr = pd.Series(np.r_[np.nan, true_range]).rolling(14).mean().to_numpy()

    def neck_at(neck, i):   # neckline through two points, value at index i
        (ia, va), (ib, vb) = neck
        return va + (vb - va) * (i - ia) / (ib - ia) if ib != ia else va

    def status(top, neck, extreme, last_point):
        """Breakout = first close beyond the neckline after the last pattern point.
        It counts only as a clear breakout: at least half an average daily range (0.5 ATR) beyond the
        neckline, otherwise 'weak' (not reported). Backtest 54 assets: clear breakouts did better,
        crypto +3.1 % vs. an average day after 20 days (all periods positive), weak ones did not.
        Invalid if the price runs above the high (top) / below the low (bottom) of the pattern.
        Old if the breakout is more than 10 values ago (no fresh signal any more)."""
        breakout = None
        for i in range(last_point + 1, n):
            if (top and p[i] > extreme * (1 + tol)) or (not top and p[i] < extreme * (1 - tol)):
                return 'invalid', None
            if breakout is None and ((top and p[i] < neck_at(neck, i)) or (not top and p[i] > neck_at(neck, i))):
                breakout = i
        if breakout is None:
            return 'forming', None
        margin = (neck_at(neck, breakout) - p[breakout]) if top else (p[breakout] - neck_at(neck, breakout))
        if np.isfinite(atr[breakout]) and margin < 0.5 * atr[breakout]:
            return 'weak', breakout
        return ('breakout', breakout) if breakout >= n - 10 else ('old', breakout)

    def add(name, top, points, labels, neck, extreme, rank):
        st, at = status(top, neck, extreme, points[-1][0])
        height = abs(extreme - neck_at(neck, points[len(points) // 2][0]))
        found.append({'name': name, 'top': top, 'points': points, 'labels': labels, 'neck': neck,
                      'height': height, 'stop': extreme, 'status': st, 'at': at, 'rank': rank})

    for k in range(len(pts)):
        seq = pts[k:]
        top = seq[0][1] == 'H'
        if len(seq) >= 5 and seq[4][0] >= n - recent:
            # Head and Shoulders: shoulder, x, head, x, shoulder
            s1, t1, head, t2, s2 = seq[:5]
            higher = head[2] > max(s1[2], s2[2]) * (1 + tol) if top else head[2] < min(s1[2], s2[2]) * (1 - tol)
            if higher and abs(s2[2] / s1[2] - 1) <= 1.5 * tol:
                add('Head and Shoulders' if top else 'Inverse Head and Shoulders', top, [s1, t1, head, t2, s2],
                    ['LS', '', 'H', '', 'RS'], ((t1[0], t1[2]), (t2[0], t2[2])), head[2], 3)
            # Triple: three equal peaks AND two similar troughs between them (otherwise it is a double top)
            a, t1, b, t2, c = seq[:5]
            if same(a[2], b[2]) and same(a[2], c[2]) and same(b[2], c[2]) and abs(t2[2] / t1[2] - 1) <= 2 * tol:
                level = min(t1[2], t2[2]) if top else max(t1[2], t2[2])
                extreme = max(a[2], b[2], c[2]) if top else min(a[2], b[2], c[2])
                if abs(extreme / level - 1) > 2 * tol:
                    add('Triple Top' if top else 'Triple Bottom', top, [a, t1, b, t2, c], ['1', '', '2', '', '3'],
                        ((t1[0], level), (c[0], level)), extreme, 2)
        if len(seq) >= 3 and seq[2][0] >= n - recent:
            # Double: peak, x, peak with at least 10 values between the peaks
            a, t, b = seq[:3]
            if same(a[2], b[2]) and abs(t[2] / max(a[2], b[2]) - 1) > 2 * tol and b[0] - a[0] >= 10:
                extreme = max(a[2], b[2]) if top else min(a[2], b[2])
                add('Double Top' if top else 'Double Bottom', top, [a, t, b], ['1', '', '2'],
                    ((t[0], t[2]), (b[0], t[2])), extreme, 1)

    found = [f for f in found if f['status'] in ('forming', 'breakout')]
    if not found:
        return None
    f = max(found, key=lambda f: (f['points'][-1][0], f['rank']))   # youngest, then most important

    # Neckline today, target = neckline ± pattern height (classic measuring rule)
    neck_now = neck_at(f['neck'], n - 1)
    target = neck_now - f['height'] if f['top'] else neck_now + f['height']
    dates = pd.to_datetime(df['Date'])
    # Plain language for users without trading experience: what the pattern means and what to watch
    meaning = {
        'Double Top': 'the price failed twice at the same high',
        'Triple Top': 'the price failed three times at the same high',
        'Head and Shoulders': 'the last rise was weaker than the one before',
        'Double Bottom': 'the price held twice at the same low',
        'Triple Bottom': 'the price held three times at the same low',
        'Inverse Head and Shoulders': 'the last drop was weaker than the one before',
    }[f['name']]
    direction = 'fall' if f['top'] else 'rise'
    if f['status'] == 'breakout':
        sentence = (f"{f['name']} ({meaning}). On {dates.iloc[f['at']]:%d/%m} the price broke "
                    f"{'down' if f['top'] else 'up'}. "
                    f"It may {direction} to about {format_euro(target)}. "
                    f"The signal is wrong if the price goes {'above' if f['top'] else 'below'} {format_euro(f['stop'])}.")
    else:
        sentence = (f"{f['name']} forming ({meaning}). If the price closes "
                    f"{'below' if f['top'] else 'above'} {format_euro(neck_now)}, it may {direction} to about {format_euro(target)}.")

    ago = lambda i: n - 1 - i
    return {'name': f['name'], 'top': f['top'], 'status': f['status'], 'sentence': sentence,
            'target': target, 'neck_now': neck_now,
            'points': [(ago(i), float(p[i])) for i, _, _ in f['points']],   # drawn on the closing price line
            'labels': f['labels'],
            'neck': [(ago(i), v) for i, v in f['neck']],
            'at': None if f['at'] is None else (ago(f['at']), float(p[f['at']]))}

def current_price(symbol, asset_type):
    """Latest price in EUR for price alerts (one small request instead of the full history)."""
    if asset_type == 'crypto':
        headers = {'x-cg-demo-api-key': coingecko_api_key} if coingecko_api_key else {}
        response = requests.get('https://api.coingecko.com/api/v3/simple/price', headers=headers, timeout=20,
                                params={'ids': symbol, 'vs_currencies': 'eur'})
        response.raise_for_status()
        price = response.json().get(symbol, {}).get('eur')
        if price is None:
            raise ValueError(f'CoinGecko has no price for {symbol}.')
        return float(price)
    info = yf.Ticker(symbol).fast_info
    price, currency = float(info['last_price']), info['currency']
    if currency in ('GBp', 'GBX'):   # London partly quotes in pence
        price, currency = price / 100, 'GBP'
    if currency != 'EUR':
        price *= float(yf.Ticker(f"{currency.upper()}EUR=X").fast_info['last_price'])
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f'No valid price for {symbol}.')
    return price

def next_earnings_date(symbol):
    """Next quarterly results date of a stock from Yahoo, or None if unknown.
    Yahoo sometimes still returns the last date before the next one is announced: past dates are ignored."""
    try:
        calendar = yf.Ticker(symbol).calendar
    except Exception as error:
        log_print(f"Yahoo {symbol}: earnings date not available ({error})", 'WARN')
        return None
    dates = calendar.get('Earnings Date', []) if isinstance(calendar, dict) else []
    upcoming = [day for day in dates if day >= datetime.now().date()]
    return min(upcoming) if upcoming else None

def trend_phase(prices):
    """Market phase from daily prices using the classic trend rule:
    price and SMA50 above SMA200 = bullish, both below = bearish, otherwise transition."""
    p = pd.Series(prices, dtype=float).dropna().reset_index(drop=True)
    if len(p) < 200:
        return 'not verifiable (less than 200 daily values)'
    sma50, sma200 = p.rolling(50).mean(), p.rolling(200).mean()
    price, s50, s200 = p.iloc[-1], sma50.iloc[-1], sma200.iloc[-1]
    if price > s200 and s50 > s200:
        phase = 'bullish'
    elif price < s200 and s50 < s200:
        phase = 'bearish'
    else:
        phase = 'neutral (transition)'
    before = sma200.iloc[-21]   # direction of the SMA200 over 20 days
    slope = 'unclear' if pd.isna(before) else 'rising' if s200 > before else 'falling'

    # Last SMA50/SMA200 crossover: Golden Cross (upwards) / Death Cross (downwards)
    above = (sma50 > sma200)[sma200.notna()]
    changes = above.index[1:][above.iloc[1:].to_numpy() != above.iloc[:-1].to_numpy()]
    if len(changes):
        last = changes[-1]
        cross = f"{'Golden Cross' if above[last] else 'Death Cross'} {len(p) - 1 - last} days ago"
    else:
        cross = f"no 50/200 cross in {len(above)} days"
    distance = f"{(price / s200 - 1) * 100:+.1f} %"
    # compact wording: the /ai prompt must fit into one Telegram message
    return (f"{phase}: price {distance} vs. SMA200, SMA50 {'>' if s50 > s200 else '<'} SMA200 ({slope}), {cross}")

# Leading stock index per home country of the company (Yahoo "country")
MARKET_BY_COUNTRY = {
    'United States': ('^GSPC', 'S&P 500'), 'Germany': ('^GDAXI', 'DAX'), 'Italy': ('FTSEMIB.MI', 'FTSE MIB'),
    'United Kingdom': ('^FTSE', 'FTSE 100'), 'France': ('^FCHI', 'CAC 40'), 'Netherlands': ('^AEX', 'AEX'),
    'Switzerland': ('^SSMI', 'SMI'), 'Spain': ('^IBEX', 'IBEX 35'), 'Sweden': ('^OMX', 'OMX Stockholm 30'),
    'Norway': ('OBX.OL', 'OBX'), 'Japan': ('^N225', 'Nikkei 225'), 'Taiwan': ('^TWII', 'TAIEX'),
    'China': ('^HSI', 'Hang Seng'), 'Hong Kong': ('^HSI', 'Hang Seng'), 'Canada': ('^GSPTSE', 'S&P/TSX'),
}


def market_prices(asset_type, symbol):
    """Leading market for the market phase: Bitcoin for crypto; for stocks the index of the company's home
    country (Nvidia listed in Frankfurt -> S&P 500), if unknown the index of the exchange."""
    if asset_type == 'crypto':
        df = get_crypto('bitcoin', {'vs_currency': 'eur', 'days': '260'})
        return 'Bitcoin (crypto leader)', None if df is None else df['Price']
    try:
        country = (yf.Ticker(symbol).info or {}).get('country')
    except Exception:
        country = None
    if country in MARKET_BY_COUNTRY:
        ticker, name = MARKET_BY_COUNTRY[country]
    else:
        suffix = symbol.upper().rsplit('.', 1)[-1] if '.' in symbol else ''
        # all German exchanges (Xetra, Frankfurt, Stuttgart, Munich, Duesseldorf, Berlin, Hamburg, Hanover): DAX
        german = {exchange: ('^GDAXI', 'DAX') for exchange in ('DE', 'F', 'SG', 'MU', 'DU', 'BE', 'HM', 'HA')}
        ticker, name = {**german, 'MI': ('FTSEMIB.MI', 'FTSE MIB')}.get(suffix, ('^GSPC', 'S&P 500'))
    hist = yf.Ticker(ticker).history(period='2y', interval='1d', timeout=20)   # index in its own currency
    return name, None if hist.empty else hist['Close'].tail(250)

def calc_state_fuctions(state,df):
    # update balances and indicators into grouped state

    # Guard: need at least 3 rows for -1, -2, -3 indexing
    if df is None or df.empty or len(df) < 3:
        log_print("calc_state_fuctions: DataFrame too short, skipping calculations", 'WARN')
        return state

    # Safe reads of prices/emas
    try:
        last_price = float(df['Price'].iloc[-1])
        prev_price = float(df['Price'].iloc[-2])
        prev2_price = float(df['Price'].iloc[-3])
    except Exception as e:
        log_print(f"calc_state_fuctions: Price read error: {e}", 'WARN')
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

    # A watched asset needs no artificial buy of 1 EUR / 1 piece.
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

    # Linear regression over every window at once (same result as linregress per window, about 50x faster):
    # x = 0 .. window-1, slope = sum((x - mean x) * y) / sum((x - mean x)^2), line through the mean
    y = df[column].to_numpy(float)
    x_centered = np.arange(window) - (window - 1) / 2
    windows = np.lib.stride_tricks.sliding_window_view(y, window)    # one row per window
    slope = windows @ x_centered / (x_centered ** 2).sum()           # a window with NaN gives NaN
    y_start = windows.mean(axis=1) - slope * (window - 1) / 2        # regression line at x = 0
    y_end = y_start + slope * (window - 1)                           # regression line at x = window-1
    with np.errstate(divide='ignore', invalid='ignore'):
        slope_pct = np.where(y_start == 0, np.nan, (y_end / y_start - 1) * 100)

    rolling_slope = np.full(len(y), np.nan)
    rolling_slope[window - 1:] = slope_pct
    return pd.Series(rolling_slope, index=df.index)

def save_csv(df, path=None, index=False, date_cols=()):
    d = df.copy()
    # convert date columns to DD.MM.YYYY
    for c in date_cols:
        if c in d:
            d[c] = pd.to_datetime(d[c], errors='coerce').dt.strftime('%d.%m.%Y')
    # save with German separator and encoding
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

    # create Date first
    df["Price"] = df["Close"]
    df["Date"] = pd.to_datetime(df["Close time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # then drop columns and return
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

    # An unchanged price has a neutral RSI instead of 0/0 (= NaN).
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

def plot_and_save(df, symbol, data_type, zero_line=None, sr_zones=None, pattern=None):
    if df is None or df.empty:
        raise ValueError("No chart possible without price data.")
    if df is not None and not df.empty:
    #----------Create
        fig, (ax1, ax3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax2 = ax1.twinx()
        #fig.patch.set_alpha(0)
        # Running number instead of calendar date: weekends and nights without trading
        # create no gaps, every bar has the same width.
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
        # Tick at the first value of each new day (100h) or month (100d).
        period = dates.dt.strftime('%Y-%m-%d' if data_type=='100h' else '%Y-%m')
        ticks = [i for i in range(1, len(dates)) if period.iloc[i] != period.iloc[i-1]]
        # With many ticks (e.g. hourly stock values) label only every second one,
        # otherwise the dates overlap. All vertical lines are kept.
        labeled = set(ticks[::2]) if len(ticks) > 8 else set(ticks)
        for ax in [ax1, ax2, ax3]:
            #ax1.set_title('100 Hours' if data_type=='100h' else '100 Days', fontsize=14)
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda value, pos: dates.iloc[int(round(value))].strftime('%b %d') if int(round(value)) in labeled else ''))
            ax.grid(True, axis='x', color='black', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax3.grid(False)  # volume bars are easier to see without vertical lines

        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
        # Show the percentage axis on the left, hide the price axis.
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
            label='Bollinger range'
        )
        # Include the price, otherwise a breakout from the band is cut off.
        lower = min(df['BB_Lower'].min(), df['Price'].min())
        upper = max(df['BB_Upper'].max(), df['Price'].max())
        if lower == upper:
            padding = max(abs(lower) * 0.01, 0.01)
            lower, upper = lower - padding, upper + padding
        ax1.set_ylim(lower, upper)

        # Keep 20% of the width free on the right so that the volume profile does not cover the price.
        vp_share = 0.2
        ax1.set_xlim(-1, -1 + (len(df) + 1) / (1 - vp_share - 0.02))

        # Volume profile (like TradingView "visible range"): volume per price level,
        # on the right in the background. Width as share of the chart width, height in euro.
        if 'Volume' in df.columns:
            profile, edges = np.histogram(df['Price'], bins=24, range=(lower, upper), weights=df['Volume'])
            if profile.max() > 0:
                trans = mtransforms.blended_transform_factory(ax1.transAxes, ax1.transData)
                widths = profile / profile.max() * vp_share      # longest bar fills the free area
                vp_colors = ['0.45' if i == profile.argmax() else '0.6' for i in range(len(profile))]  # POC slightly darker
                ax1.barh(edges[:-1], widths, height=np.diff(edges) * 0.9, left=1 - widths, align='edge',  # aligned to the right edge
                         color=vp_colors, alpha=0.3, linewidth=0, transform=trans, zorder=0)
                # Black tick at the end of each bar: length stays visible below labels.
                filled = profile > 0
                ax1.vlines((1 - widths)[filled], edges[:-1][filled], (edges[:-1] + np.diff(edges) * 0.9)[filled],
                           colors='black', linewidth=0.8, transform=trans, zorder=6)

        # Support/resistance: short band (about 1 cm) at the left edge, number (R1/S1) as a tag
        # outside the chart next to the percentage axis, details in the legend at the top right.
        handles = []
        tag_prices = []   # prices of the R1/S1 tags: axis labels next to them are hidden
        ax_width_cm = ax1.get_position().width * fig.get_figwidth() * 2.54
        stub = min(1.0 / ax_width_cm, 0.2)   # 1 cm as share of the chart width
        outside = mtransforms.blended_transform_factory(ax1.transAxes, ax1.transData)
        if sr_zones is not None:
            for zone, name in zip(sr_zones, ('R1', 'S1')):
                if zone is None:
                    continue
                color = '#c0392b' if name == 'R1' else '#1e8449'
                alpha = 0.22 if zone['touches'] >= 2 else 0.12   # weak zone (1 touch) lighter
                pct = (zone['center'] / df['Price'].iloc[-1] - 1) * 100
                text = f"{name} {human_format_euro(zone['center'], None)} ({pct:+.1f} %)"
                if zone['touches']:
                    text += f" · {zone['touches']}×"
                else:
                    text += " · period high" if name == 'R1' else " · period low"
                tag = dict(boxstyle='round,pad=0.25', facecolor=color, edgecolor='none')
                if zone['high'] >= lower and zone['low'] <= upper:
                    ax1.axhspan(zone['low'], zone['high'], xmax=stub, color=color, alpha=alpha + 0.15, linewidth=0, zorder=0.5)
                    y = min(max(zone['center'], lower), upper)
                    ax1.text(-0.01, y, name, transform=outside, fontsize=7, color='white', fontweight='bold',
                             va='center', ha='right', zorder=10, clip_on=False, bbox=tag)
                else:
                    # Zone outside the chart: tag with arrow at the top/bottom edge
                    y, arrow = (upper, '▲') if zone['center'] > upper else (lower, '▼')
                    ax1.text(-0.01, y, f"{arrow} {name}", transform=outside, fontsize=7, color='white', fontweight='bold',
                             va='center', ha='right', zorder=10, clip_on=False, bbox=tag)
                # Too close to the price tag: move the tag up (R1) or down (S1) so that both stay readable
                gap = 0.06 * (upper - lower)
                price_now = df['Price'].iloc[-1]
                if abs(y - price_now) < gap:
                    y = price_now + gap if name == 'R1' else price_now - gap
                    ax1.texts[-1].set_y(y)
                tag_prices.append(y)
                handles.append(mpatches.Patch(color=color, alpha=alpha + 0.1, label=text))

        # Chart pattern (chart_pattern): pink, solid and thick so that it stands out from the dotted EMAs
        if pattern is not None:
            pink = '#e91e8c'
            last = len(df) - 1
            xs = [last - ago for ago, _ in pattern['points']]
            ys = [price for _, price in pattern['points']]
            ax1.plot(xs, ys, color=pink, linewidth=3, alpha=0.9, solid_joinstyle='round', zorder=4)
            for px, py, label in zip(xs, ys, pattern['labels']):
                if label:
                    ax1.annotate(label, (px, py), xytext=(0, 7 if pattern['top'] else -12), textcoords='offset points',
                                 ha='center', fontsize=7, fontweight='bold', color=pink, zorder=5)
            # Neckline from the first neck point to today
            (ago_a, va), (ago_b, vb) = pattern['neck']
            neck_at = lambda px: va + (vb - va) * (px - (last - ago_a)) / max(1, ago_a - ago_b)
            ax1.plot([last - ago_a, last], [va, neck_at(last)], color=pink, linewidth=3, alpha=0.9, solid_capstyle='round', zorder=4)
            # Name tag above (top) or below (bottom) the pattern, like on TradingView
            ax1.annotate(pattern['name'], (xs[len(xs) // 2], max(ys) if pattern['top'] else min(ys)),
                         xytext=(0, 18 if pattern['top'] else -22), textcoords='offset points',
                         ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=6,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=pink, edgecolor='none', alpha=0.9))
            target = pattern['target']
            if lower <= target <= upper:
                ax1.axhline(target, xmax=1 - vp_share - 0.02, color=pink, linewidth=1.2, alpha=0.7, zorder=4)
                ax1.text(1 - vp_share - 0.015, target, 'target', transform=ax1.get_yaxis_transform(), fontsize=7,
                         color=pink, va='center', ha='left', fontweight='bold')
            else:   # target outside the chart: hint at the edge like R1/S1
                y, arrow = (0.90, '▲') if target > upper else (0.10, '▼')
                ax1.text(1 - vp_share - 0.015, y, f'{arrow} target', transform=ax1.transAxes, fontsize=7,
                         color=pink, va='center', ha='left', fontweight='bold')
            if pattern['at'] is not None:   # star at the breakout
                ago, price = pattern['at']
                ax1.plot(last - ago, price, marker='*', markersize=12, color=pink, zorder=6)
            state = 'breakout' if pattern['status'] == 'breakout' else 'forming'
            handles.append(plt.Line2D([], [], color=pink, linewidth=1.2,
                                      label=f"{pattern['name']} ({state}) · target {format_euro(target)}"))

        if handles:
            # Legend above the chart: covers neither price nor volume profile.
            # One column when the pattern is added, otherwise R1/S1 next to each other.
            ax1.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 1),
                       ncol=1 if pattern is not None else len(handles),
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
        if pattern is None:   # with a chart pattern: no turning points and max/min, the pattern is easier to see
            ax1.plot(x[df['Position']== 1.0], df.loc[df['Position']== 1.0, 'Fast EMA'], '^',  markersize=10, color='green', label='Buy')
            ax1.plot(x[df['Position']==-1.0], df.loc[df['Position']==-1.0, 'Fast EMA'], 'v',  markersize=10, color='red', label='Sell')
            ax1.plot(x[df['HighP']==1],      df.loc[df['HighP']==1, 'Fast EMA'], 'o', markersize=7, label='Max')
            ax1.plot(x[df['LowP']==1],       df.loc[df['LowP']==1,  'Fast EMA'], 'o', markersize=7, label='Min')

        ax2.plot(x, df['Percentage Deviation'], color='k', linewidth=1.0, label='Deviation (%)')

        # Current price as a tag on the percentage axis (like TradingView): "765.50 € (-26.0 %)"
        current_pct = df['Percentage Deviation'].iloc[-1]
        ax2.axhline(current_pct, color='black', linewidth=0.7, linestyle=(0, (2, 2)), alpha=0.6, zorder=1)
        ax2.text(-0.01, current_pct, f"{format_euro(df['Price'].iloc[-1])} ({current_pct:+.1f} %)",
                 transform=mtransforms.blended_transform_factory(ax2.transAxes, ax2.transData),
                 ha='right', va='center', fontsize=7.5, color='white', fontweight='bold', zorder=10, clip_on=False,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', edgecolor='none'))
        # hide axis labels right next to the tags (price, R1, S1) so that they do not overlap
        percent = mticker.PercentFormatter()
        percent.set_axis(ax2.yaxis)
        span = max_remap - min_remap
        to_pct = lambda price: linear_remap(price, lower, upper, min_remap, max_remap)
        tag_pcts = [current_pct] + [to_pct(price) for price in tag_prices]
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda value, pos: '' if any(abs(value - t) < 0.06 * span for t in tag_pcts) else percent(value, pos)))

        if 'Volume' in df.columns:
            colors = ['green' if close > open_val else 'red'
                    for open_val, close in zip(df['Price'].shift(1), df['Price'])]
            colors[0] = 'gray'
            # distance between two values is always 1 -> bar 80% of it
            ax3.bar(x, df['Volume'],
                    width=0.8, align='center',
                    color=colors, alpha=0.7, linewidth=0)

            ax3.margins(x=0.01)  # some margin so that nothing is cut off

    #-------------Save
        os.makedirs(output_dir, exist_ok=True)
    # Set figure background to white
    fig.patch.set_facecolor('white')
    # dpi=200 gives about 1200 pixels width, matching Telegram's maximum size of 1280 pixels.
    plt.savefig(os.path.join(output_dir, f"{symbol}.png"), dpi=200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close(fig)

def telegram_length(text, html_format=True):
    """Length as Telegram counts it (limit 4096): visible text after HTML parsing, in UTF-16 units.
    Tags like <pre> and entities like &amp; do not count."""
    if html_format:
        text = html.unescape(re.sub(r'<[^>]+>', '', text))
    return len(text.encode('utf-16-le')) // 2

def set_message_queue(queue: mp.Queue):
    global _message_queue
    _message_queue = queue

def pushover(message: str, priority: int = 0, html_format=False, buttons=None):
    # buttons: optional Telegram buttons below the message, as rows of (label, callback_data)
    log_message('message', message)
    if _message_queue:
        _message_queue.put({'type': 'text', 'message': message, 'priority': priority, 'html': html_format,
                            'buttons': buttons})
    else:
        log_print("No queue – message discarded.", 'WARN')

def pushover_photo(path: str, message: str, html_format=False):
    """Send a fixed image file (e.g. the logo) with the message as caption; the file is never deleted."""
    log_message('photo', message)
    if _message_queue:
        _message_queue.put({'type': 'image', 'path': path, 'message': message, 'priority': 0,
                            'delete_after_send': False, 'html': html_format})
    else:
        log_print("No queue – image discarded.", 'WARN')

def pushover_image(symbol: str, message: str, priority: int = 0, delete_after_send=False, html_format=False, buttons=None):
    # html_format=True: message contains Telegram HTML (<b>, <i>, <pre>)
    # buttons: optional Telegram buttons below the message, as rows of (label, callback_data)
    log_message('chart', message)
    if _message_queue:
        _message_queue.put({'type': 'image', 'symbol': symbol, 'message': message, 'priority': priority, 'output_dir': output_dir,
                            'delete_after_send': delete_after_send, 'html': html_format, 'buttons': buttons})
    else:
        log_print("No queue – image discarded.", 'WARN')

def google_trends(search):
    # currently not used by CaSa: pytrends is imported only here, so it is no longer a required package
    # (install it with "python -m pip install pytrends" before using this function)
    from pytrends.request import TrendReq

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
        raise ValueError('Not enough Google Trends data.')
    return df['Interest'].iloc[-1], df['Interest'].iloc[-2]

def seven_day_slope_pct(df, current):

    if len(df) < 8:
        log_print(f"seven_day_slope_pct: DataFrame has only {len(df)} rows, need at least 8", 'WARN')
        return 0.0
    
    try:
        # ✅ Problem 4a fix: explicit iloc
        if current:
            y = df['Price'].iloc[-7:].values
        else:
            y = df['Price'].iloc[-8:-1].values
        
        # Validation
        if len(y) != 7:
            log_print(f"seven_day_slope_pct: Expected 7 values, got {len(y)}", 'WARN')
            return 0.0
            
        if np.any(np.isnan(y)):
            log_print("seven_day_slope_pct: NaN values in price data", 'WARN')
            return 0.0
        
        # ✅ Problem 4b fix: direct np.arange(7) instead of index reset
        # x = [0, 1, 2, 3, 4, 5, 6] independent of the DataFrame index
        slope, intercept = np.polyfit(np.arange(7), y, 1)   # straight line through the 7 values
        
        # values at day 0 and day 6
        y_start = intercept
        y_end = slope * 6 + intercept
        
        if y_start == 0:
            log_print("seven_day_slope_pct: y_start is 0", 'WARN')
            return 0.0
        
        return (y_end / y_start - 1) * 100
        
    except Exception as e:
        # ✅ Problem 4c fix: graceful error handling
        log_print(f"seven_day_slope_pct: Error - {e}", 'WARN')
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


def risk_reward(price, sr_zones):
    """Room to the nearest resistance vs. distance to the nearest support (R1/S1).
    Information only (ℹ️, no traffic light): in the backtest a "good" ratio did not lead to better results,
    because a price near resistance is usually a strong one. Useful for stop and target, not as a signal.
    Returns a dict: mark, short (sentence for the message), text (details for the AI prompt), lines (block figures)."""
    resistance, support = sr_zones if sr_zones else (None, None)
    up = (resistance['center'] / price - 1) * 100 if resistance else None
    down = (1 - support['center'] / price) * 100 if support else None
    lines = [f"{'R1':<6}{format_euro(resistance['center']):>13}  ({up:+.1f} %)" if resistance
             else f"{'R1':<6}{'none':>13}  (price near its high)",
             f"{'S1':<6}{format_euro(support['center']):>13}  ({-down:+.1f} %)" if support
             else f"{'S1':<6}{'none':>13}  (price near its low)"]
    if up is not None and down:
        lines.append(f"{'Ratio':<6}{f'{up / down:.1f} : 1':>13}")
    if resistance is None and support is None:
        text = 'no resistance or support zone found'
    elif resistance is None:
        text = f"no resistance above (price near its high), {-down:+.1f} % to support ({format_euro(support['center'])})"
    elif support is None:
        text = f"{up:+.1f} % to resistance ({format_euro(resistance['center'])}), no support below (price near its low)"
    else:
        text = (f"{up:+.1f} % to resistance ({format_euro(resistance['center'])}), "
                f"{-down:+.1f} % to support ({format_euro(support['center'])}), ratio {up / down:.1f} : 1")
    # Short line for the top of the message: the figures are in the block below
    if resistance is None and support is None:
        short = 'No zones found.'
    elif resistance is None:
        short = 'No resistance above.'
    elif support is None:
        short = 'No support below.'
    else:
        short = ('Less room up than down.' if up < down else
                 'Twice as much room up as down.' if up >= 2 * down else 'More room up than down.')
    # Position between S1 (0 %) and R1 (100 %) seen from above: the share of the room up to R1 within the whole
    # range S1..R1 (e.g. R1 +6.9 %, S1 -5.4 % -> 56 % of the range is room up)
    room = up / (up + down) * 100 if up is not None and down is not None and up + down > 0 else None
    return {'mark': 'ℹ️', 'short': short, 'text': text, 'lines': lines, 'room': room,
            'ratio': up / down if up is not None and down else None, 'up': up, 'down': down,
            'resistance': resistance['center'] if resistance else None,
            'support': support['center'] if support else None}


def score_rank(full, window=100):
    """Score 1-10 as a rank: today's score compared with the scores of the last `window` values of the
    same asset (10 = among the strongest 10 %). full = long history with indicators."""
    scores = []
    for end in range(max(30, len(full) - window + 1), len(full) + 1):
        try:
            # only the score (no message text); the 100 values before "end" as in cut_to_window()
            scores.append(score_only(full.iloc[max(0, end - 100):end]))
        except ValueError:
            continue
    if len(scores) < 20:
        return None
    today = scores[-1]
    return max(1, math.ceil(sum(s <= today for s in scores) / len(scores) * 10))


def score_signals(df):
    """Position of RSI14, SMA7, VMA7, ROC5, VROC5 within the last 100 values: (signals -3..+3, quantiles)."""
    signals, quantiles = {}, {}
    for column in ('RSI14', 'SMA7', 'VMA7', 'ROC5', 'VROC5'):
        count, quantile = calc_stat_limits(df, column, window=100, invert=(column == 'RSI14'))
        # Missing volume gives no usable momentum: rate it neutral.
        # A missing price indicator, however, must show up as a data error.
        if not math.isfinite(count) or not math.isfinite(quantile):
            if column in ('VMA7', 'VROC5'):
                count, quantile = 0, float('nan')
            else:
                raise ValueError(f"Not enough data for {column}.")
        signals[column], quantiles[column] = int(count), quantile
    return signals, quantiles


def score_only(df):
    """Score -18..+18 like analysis_details(), without building the message (fast, for score_rank and alarms)."""
    signals, _ = score_signals(df)
    _, score = tech_analyse1(signals['RSI14'], 0)
    _, score = tech_analyse2(signals['ROC5'], signals['VROC5'], score)
    _, score = tech_analyse3(signals['SMA7'], signals['VMA7'], score)
    return score + signals['ROC5']


PERCENTILE_STEPS = 8          # characters per bar (one character = 12.5 %)
PERCENTILE_EMPTY = '░'
# Block characters instead of coloured emoji squares: in the monospace block of Telegram emoji are drawn
# smaller or wider than letters, block characters have exactly the width of a letter (aligned bars).


def percentile_bar(percent, label='>{:.0f}%'):
    """Bar for 'higher than on x % of the last 100 values', e.g. '███░░░░░ >40%'; label: text after the bar."""
    filled = min(PERCENTILE_STEPS, max(0, round(percent / 100 * PERCENTILE_STEPS)))
    return f"{'█' * filled}{PERCENTILE_EMPTY * (PERCENTILE_STEPS - filled)} {label.format(percent)}"


def analysis_details(df, pattern=None, rank=None, unit='days', sr_zones=None):
    """Shared figures for alarms and /check; this function sends nothing.
    rank: score 1-10 from score_rank(); unit: 'days' (daily analysis) or 'hours' (hourly analysis);
    sr_zones: (resistance, support) from support_resistance() for the risk/reward line."""
    signals, quantiles = score_signals(df)
    # Score parts: each assessment sentence raises (🟢) or lowers (🔴) the score
    text1, part_rsi = tech_analyse1(signals['RSI14'], 0)
    text2, score = tech_analyse2(signals['ROC5'], signals['VROC5'], part_rsi)
    part_short = score - part_rsi + signals['ROC5']
    text3, score = tech_analyse3(signals['SMA7'], signals['VMA7'], score)
    part_trend = score - part_rsi - (part_short - signals['ROC5'])
    score += signals['ROC5']

    def row(name, shown, higher=None, label='>{:.0f}%'):
        # one line of the figure list: name, real value, bar with the position within the last 100 values
        bar = percentile_bar(higher, label) if higher is not None else f"{PERCENTILE_EMPTY * PERCENTILE_STEPS} neutral"
        return f"{name:<8}{shown:>10} {bar}"   # fixed width: values are aligned

    def line(column):
        # real value first, then a bar with its position within the last 100 values
        quantile = quantiles[column]
        current = df[column].iloc[-1] if column in df else float('nan')
        if not math.isfinite(quantile) or not math.isfinite(current):
            return row(column, 'no data')
        shown = (f"{current:.1f}" if column == 'RSI14' else format_euro(current) if column == 'SMA7' else
                 human_format(current, None) if column == 'VMA7' else f"{current:+.1f} %")
        # ">73%" = higher than on 73 % of the last 100 values. For the score the RSI rank is
        # inverted (high RSI = overbought); here the real position is shown, as for all other lines.
        # No +/- signal marks: the traffic light at the top already shows the effect on the score.
        higher = 100 - quantile if column == 'RSI14' else quantile
        return row(column, shown, higher)

    def block(lines, label):
        # Telegram HTML: header bold, values in monospace (the assessment sentence is in the summary above)
        return f"<b>{html.escape(label)}</b>\n<pre>{html.escape(chr(10).join(lines))}</pre>"

    def mark(part):
        return '🟢' if part > 0 else '🔴' if part < 0 else '🟡'

    # Summary: score 1-10, then one traffic-light line per block below, in the same order
    price = float(df['Price'].iloc[-1])
    sma200 = df['SMA200'].iloc[-1] if 'SMA200' in df else float('nan')
    period = '200-day' if unit == 'days' else '200-hour'
    # One line: how far the price is above/below the SMA200 in %, with a bar like the indicators:
    # this distance is higher than on x % of the last 100 values (price / SMA200 of each of these days)
    if not math.isfinite(sma200):
        long_mark, long_text = '🟡', 'Unknown (less than 200 values).'
        long_line = row('vsSMA200', 'no data')
    else:
        long_mark, long_text = (('🟢', f"Above the {period} average: uptrend.") if price > sma200 else
                                ('🔴', f"Below the {period} average: downtrend."))
        distance = (df['Price'] / df['SMA200']).iloc[-101:]
        previous = distance.iloc[:-1].dropna()
        higher = float((previous < distance.iloc[-1]).mean() * 100) if len(previous) else None
        long_line = row('vsSMA200', f"{(price / sma200 - 1) * 100:+.1f} %", higher)
    # Room to resistance vs. distance to support: is an entry still worth it?
    rr = risk_reward(price, sr_zones) if sr_zones is not None else None

    # RSI from the real value (30/70 limits); the comparison with the last 100 values only in brackets
    rsi_now = float(df['RSI14'].iloc[-1])
    relative = (f" (low compared with the last 100 {unit})" if signals['RSI14'] >= 2 else
                f" (high compared with the last 100 {unit})" if signals['RSI14'] <= -2 else '')
    if rsi_now >= 70:
        rsi_mark, rsi_text = '🔴', f"Overbought – risk of a pullback{relative}."
    elif rsi_now <= 30:
        rsi_mark, rsi_text = '🟢', f"Oversold – rebound possible{relative}."
    else:
        rsi_mark, rsi_text = '🟡', f"Neutral{relative}."

    # Pattern only as a short status here, as information without buy/sell signal (ℹ️ name and state, 🟡 none);
    # the full description follows in the Pattern block.
    pattern_text = pattern['sentence'] if pattern else 'No chart pattern detected.'
    if not pattern:
        pattern_mark, pattern_short = '🟡', 'None detected.'
    elif pattern['status'] != 'breakout':
        pattern_mark, pattern_short = 'ℹ️', f"{pattern['name']} forming."
    else:
        pattern_mark, pattern_short = 'ℹ️', f"{pattern['name']}: price broke {'down' if pattern['top'] else 'up'}."
    # The figures as one numbered list without headers; the summary at the top names their numbers.
    # Each entry: (traffic light, name, short sentence at the top, its lines in the list)
    groups = [(long_mark, 'Long-term', long_text, [long_line])]
    if rr:
        # R1/S1: ratio room up : room down, the bar = share of the room up to R1 within the range S1 (0 %) .. R1 (100 %)
        rr_short = rr['short']
        rr_line = (row('R1/S1', f"{rr['ratio']:.1f} : 1", rr['room'], '{:.0f}%↑') if rr['room'] is not None
                   else row('R1/S1', 'no R1' if rr['resistance'] is None else 'no S1'))
        groups.append((rr['mark'], 'Risk/Reward', rr_short, [rr_line]))
    groups += [
        (rsi_mark, 'RSI', rsi_text, [line('RSI14')]),
        (mark(part_trend), 'Trend', text3.split(': ', 1)[-1], [line('SMA7'), line('VMA7')]),
        (mark(part_short), 'Short-term', text2.split(': ', 1)[-1], [line('ROC5'), line('VROC5')]),
    ]
    numbered, sections = [], []
    for light, name, sentence, rows in groups:
        numbers = list(range(len(numbered) + 1, len(numbered) + len(rows) + 1))
        numbered += [f"{number} {text}" for number, text in zip(numbers, rows)]
        sections.append((light, f"{name} ({', '.join(map(str, numbers))})", sentence, ''))
    # One entry per summary line: (traffic light, name with the numbers of its figures, short sentence, block below)
    sections[0] = sections[0][:3] + (f"<pre>{html.escape(chr(10).join(numbered))}</pre>",)
    sections.append((pattern_mark, 'Pattern', pattern_short,
                     f"<b>Pattern</b>\n{html.escape(pattern_text)}" if pattern else ''))
    if rank is None:
        head = f"<b>Score: {score}</b>"
    elif rank == 1:
        head = f"<b>Score {rank}/10</b> – among the weakest 10 % of the last 100 {unit}"
    else:
        head = f"<b>Score {rank}/10</b> – stronger than {(rank - 1) * 10} % of the last 100 {unit}"
    summary = head + "\n" + "\n".join(f"{light} <b>{name}:</b> {html.escape(sentence)}"
                                      for light, name, sentence, _ in sections)
    # Figure blocks collapsed (Telegram "expandable blockquote", tap to open): the message stays short on the phone
    details = "\n\n".join(content for *_, content in sections if content)
    message = summary + "\n\n<blockquote expandable>" + details + "</blockquote>"
    # assessments without header (e.g. for the /ai prompt)
    texts = {'rsi': rsi_text, 'trend': text3.split(': ', 1)[-1], 'short': text2.split(': ', 1)[-1],
             'pattern': pattern_text}
    return {'score': score, 'rank': rank, 'quantiles': quantiles, 'message': message, 'texts': texts}


# Alarm and info switches (config.ini [alerts], see classes.read_alert_settings)
# Alarms: signals with an advantage in the backtest; infos: notable events without a proven forecast.
# Per line in config.ini: name = <stocks>; <crypto>, each one of alarm | info | off
ALERTS = {
    # key: (codes, label, default for stocks, default for crypto, short description, probability from the
    #       backtest 0.1.11: 54 assets, share of signals with the price moving in the expected direction
    #       after 20 days, "normal" = the same for an average day)
    'positive5':      (('121',), 'Positive-5', 'alarm', 'alarm', 'price +10 % in 5 days (uptrend)',
                       'rises: stocks 54 % (normal 52 %), crypto 54 % (normal 49 %)'),
    'score8':         (('141',), 'Score>8', 'info', 'alarm', 'score above 8 for the first time (uptrend)',
                       'rises: stocks 55 % (normal 52 %, no average gain), crypto 56 % (normal 49 %)'),
    'oversold':       (('151',), 'Oversold', 'alarm', 'off', 'score -8 or lower: strong sell-off',
                       'rises: stocks 58 % (normal 52 %), crypto 48 % (normal 49 %)'),
    'cross_ema_sell': (('201',), 'Cross-EMA Sell', 'alarm', 'alarm', 'trend turns down (only own positions)',
                       'falls: stocks 45 % (normal 48 %, but -0.9 % on average), crypto 55 % (normal 51 %)'),
    'loss_limit':     (('211',), 'Loss limit', 'alarm', 'alarm',
                       'total P&L of an own position falls below total_loss_limit in [alertsconfig]',
                       'risk alarm for your own limit, no backtest'),
    'cross_ema_buy':  (('301',), 'Cross-EMA Buy', 'info', 'info', 'trend turns up',
                       'rises: stocks 57 % (normal 52 %), crypto 53 % (normal 49 %)'),
    'negative5':      (('321',), 'Negative-5', 'info', 'info', 'price -10 % in 5 days',
                       'rises: stocks 53 % (normal 52 %), crypto 48 % (normal 49 %)'),
    'high100':        (('361',), '100-day high', 'info', 'alarm', 'new highest price of 100 days',
                       'rises: stocks 55 % (normal 52 %), crypto 60 % (normal 49 %)'),
    'low100':         (('362',), '100-day low', 'info', 'info', 'new lowest price of 100 days',
                       'rises: stocks 57 % (normal 52 %), crypto 53 % (normal 49 %)'),
    'pattern':        (('331', '332'), 'Pattern breakout', 'info', 'info', 'chart pattern breaks out',
                       'in breakout direction: stocks 51 % (normal 51 %), crypto 56 % (normal 50 %)'),
}
ALERT_KINDS = ('alarm', 'info', 'off')


def backtest_note(key, asset_type):
    """Short backtest line for an alarm or info, e.g. 'Backtest: price rose in 57 % of cases (after 20 days).'
    From the probability in ALERTS for stocks or crypto; '' if the signal has no backtest."""
    odds = ALERTS[key][5]
    found = re.search(rf"{'crypto' if asset_type == 'crypto' else 'stocks'} (\d+) %", odds)
    if not found:
        return ''
    what = ('price rose' if odds.startswith('rises') else 'price fell' if odds.startswith('falls')
            else 'price moved in the breakout direction')
    return f"Backtest: {what} in {found.group(1)} % of cases (after 20 days)."


def alert_active(settings, key, asset_type=None):
    """'alarm', 'info' or None (off). settings: {key: (kind for stocks, kind for crypto)};
    a missing entry uses the default. Without a known asset type the stock setting is used, else crypto."""
    stock, crypto = (settings or {}).get(key, (ALERTS[key][2], ALERTS[key][3]))
    kind = crypto if asset_type == 'crypto' else stock if asset_type == 'stock' else (stock if stock != 'off' else crypto)
    return None if kind == 'off' else kind


alert_settings = {}   # set by main.py from config.ini


def alarm(df, symbol, watch_list, current_profit_pct, amount_older_than_one_year, amount_older_than_one_year_pct, data_type, pattern=None, rank=None, sr_zones=None, asset_type=None,
          yesterday_profit_pct=None):

    alarms = {}
    tech_indicators = None  
    score = 0

    details = analysis_details(df, pattern, rank, 'days' if data_type == '100d' else 'hours', sr_zones)
    score = details['score']
    tech_indicators = details['quantiles']
    current_EMA_diff_pct = (df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-1]) / df['Slow EMA'].iloc[-1] * 100
    yesterday_EMA_diff_pct = (df['Fast EMA'].iloc[-2] - df['Slow EMA'].iloc[-2]) / df['Slow EMA'].iloc[-2] * 100

    def create_alarm(code, buy_sell, indicator, value, important=True):
        # important=False: never high priority (signal without proven advantage in the backtest)
        priority = 1 if important and ((score > 5 and buy_sell == "Buy") or (score < -5 and buy_sell == "Sell")) else 0
        # Telegram HTML (bold/italic/monospace), therefore escape texts with html.escape
        message = (
            # "Potential": a signal, not a recommendation (most alarms are right in about 60 % of cases)
            f"<b>{html.escape(symbol)} — Potential {buy_sell.lower()}</b>\n"
            f"Trigger: {html.escape(indicator)} ({data_type})\n\n"
            + details['message']
        )
        if watch_list:
            message += "\n\n<i>Watched asset – not in your portfolio.</i>"
        else:
            message += "\n\n" + html.escape(f"Amount >1 year: {amount_older_than_one_year:.2f} ({amount_older_than_one_year_pct:.2f} %)")
        alarms[code] = {"value": value, "priority": priority, "msg": message, "kind": "alarm"}

    # Date only: CoinGecko returns the current time as the last value.
    value_ts = int(pd.Timestamp(df['Date'].iloc[-1]).normalize().timestamp())

    def create_info(code, title, value, note):
        # Info: a notable event (no buy/sell wording, never important, not counted in the alarm journal)
        # header: the asset, below it one line with what triggered the info (no second "Trigger:" line)
        message = (f"<b>{html.escape(symbol)}</b>\n"
                   f"ℹ️ Info: {html.escape(title)} ({data_type})\n\n" + details['message']
                   + f"\n\n<i>{html.escape(note)}</i>")
        if watch_list:
            message += "\n<i>Watched asset – not in your portfolio.</i>"
        alarms[code] = {"value": value, "priority": 0, "msg": message, "kind": "info"}

    def emit(key, code, buy_sell, title, value, note, important=True):
        # Alarm or info, as chosen in config.ini [alerts] (or nothing when switched off for this asset type).
        # note None: one short line with the backtest result of this signal for stocks or crypto
        kind = alert_active(alert_settings, key, asset_type)
        note = note or backtest_note(key, asset_type)
        if kind == 'alarm':
            create_alarm(code, buy_sell, title, value, important)
            alarms[code]["msg"] += f"\n\n<i>{html.escape(note)}</i>"
        elif kind == 'info' and not (watch_list and infos_for == 'portfolio'):
            # infos_for = portfolio (config.ini [alertsconfig]): no infos for watched assets you do not hold
            create_info(code, title, value, note)

    # Signals for all assets with 1 in assets.ini, also watched ones you do not own yet (sell: only your own).
    # Trend filter: Positive-5 and Score>8 only above the SMA200 (uptrend); without 200 values it does not block.
    sma200 = df['SMA200'].iloc[-1] if 'SMA200' in df else np.nan
    uptrend = not df['Price'].iloc[-1] <= sma200
    roc_now, roc_before = df['ROC5'].iloc[-1], df['ROC5'].iloc[-2]
    daily = data_type == "100d"
    previous = []   # score of the value before: calculated at most once, only when needed

    def score_before():
        if not previous:
            previous.append(score_only(df.iloc[:-1]))
        return previous[0]

    # Positive-5: only on the first value above the threshold, not on every following value
    if uptrend and roc_now > seven_day_price_change and not roc_before > seven_day_price_change:
        emit('positive5', "121", "Buy", "Positive-5", value_ts,
             None)

    # Score>8: first daily value above 8 (crypto: +7.6 % vs. an average day after 20 days; stocks: no advantage)
    if daily and uptrend and score > 8 and score_before() <= 8:
        emit('score8', "141", "Buy", "Score>8", value_ts,
             None)

    # Oversold: score falls to -8 or lower for the first time. Stocks often recover a little after a strong
    # sell-off (backtest 47 stocks: +1.3 % vs. an average day after 20 days, 58 % hits, all periods positive).
    # Crypto tends to keep falling (-5.5 %), therefore the default is stocks only. No SMA200 filter.
    if daily and score <= -8 and score_before() > -8:
        emit('oversold', "151", "Buy", "Oversold (strong sell-off, possible rebound)", value_ts,
             None, important=False)

    # Trend turns down: only for assets in your portfolio; the most stable signal since 2020
    if daily and not watch_list and yesterday_EMA_diff_pct > 0 and current_EMA_diff_pct < 0:
        emit('cross_ema_sell', "201", "Sell", "Cross-EMA", value_ts,
             None)

    # Loss limit: the total P&L of an own position (value vs. FIFO buy price) falls below your limit
    # (config.ini [alertsconfig] total_loss_limit) - on the day it crosses the limit, again after a recovery above it
    if daily and not watch_list and yesterday_profit_pct is not None \
            and current_profit_pct < total_loss_limit <= yesterday_profit_pct:
        emit('loss_limit', "211", "Sell", f"Total P&L {current_profit_pct:+.1f} % (your limit {total_loss_limit:g} %)",
             value_ts, f"Your position is now more than {abs(total_loss_limit):g} % below your buy price (FIFO). "
             "Check your plan: stop loss, hold or buy more? Risk alarm for your own limit, not a forecast.")

    if daily and yesterday_EMA_diff_pct < 0 and current_EMA_diff_pct > 0:
        emit('cross_ema_buy', "301", "Buy", "Trend turns up (Cross-EMA)", value_ts,
             None, important=False)

    if roc_now < -seven_day_price_change and not roc_before < -seven_day_price_change:
        emit('negative5', "321", "Sell", f"Sharp drop ({roc_now:+.1f} % in 5 days)", value_ts,
             None, important=False)

    if daily and len(df) >= 100:
        price, before = df['Price'].iloc[-1], df['Price'].iloc[:-1]
        if price >= before.max() and not before.iloc[-1] >= before.iloc[:-1].max():
            emit('high100', "361", "Buy", "New 100-day high", value_ts,
                 None, important=False)
        if price <= before.min() and not before.iloc[-1] <= before.iloc[:-1].min():
            emit('low100', "362", "Sell", "New 100-day low", value_ts,
                 None, important=False)

    # Chart pattern: fresh breakout through the neckline (today or the value before); value = date of the
    # breakout, so the same breakout is reported only once.
    if pattern is not None and pattern['status'] == 'breakout' and pattern['at'][0] <= 1:
        breakout_ts = int(pd.Timestamp(df['Date'].iloc[-1 - pattern['at'][0]]).normalize().timestamp())
        direction = 'down' if pattern['top'] else 'up'
        emit('pattern', "332" if pattern['top'] else "331", "Sell" if pattern['top'] else "Buy",
             f"{pattern['name']} breakout {direction}", breakout_ts,
             None, important=False)

    return alarms, tech_indicators, score

def linear_remap(value, old_min, old_max, new_min, new_max):
    """Maps value linearly from [old_min, old_max] to [new_min, new_max].
       If old_min == old_max, the middle of the target interval is returned."""
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

    # get values from the matrices
    analysis = PROBABILITY_MATRIX.get(RSI14, "error")
    
    # update score
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
    
    # get values from the matrices (with fallback)
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
    
    # get values from the matrices (with fallback)
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

def format_euro(value):
    # number format: 71,660.12 €
    return f"{value:,.2f} €"

def create_table_image(df, output_path, title='', subtitle='', dpi=200):
    """Daily summary as a modern ranking: sorted by 1d P&L, best row highlighted,
    1d P&L also shown as a bar (green to the right, red to the left)."""
    GREEN, RED, GREY, DARK = '#1e8449', '#c0392b', '#8a8a8a', '#222222'

    def pct_color(value):
        return GREEN if value > 0 else RED if value < 0 else GREY

    df = df.sort_values('1d P&L [%]', ascending=False).reset_index(drop=True)
    rows = len(df)
    fig, ax = plt.subplots(figsize=(6.4, 0.95 + rows * 0.42))
    ax.set_xlim(0, 1)
    ax.set_ylim(rows + 1.6, -0.2)   # y downwards: title on top, then header, then rows
    ax.axis('off')

    # columns (x position, alignment)
    col = {'rank': (0.03, 'center'), 'name': (0.07, 'left'), 'price': (0.43, 'right'),
           'bar': (0.47, 0.71), 'd1': (0.81, 'right'), 'd7': (0.91, 'right'), 'rating': (0.965, 'center')}

    # title
    ax.text(0.0, 0.25, title, fontsize=12, fontweight='bold', color=DARK, va='center')
    ax.text(1.0, 0.25, subtitle, fontsize=7.5, color=GREY, va='center', ha='right')

    # header
    header_y = 1.0
    for key, label in (('rank', '#'), ('name', 'NAME'), ('price', 'PRICE'), ('d1', '1D P&L'),
                       ('d7', '7D'), ('rating', 'SCORE')):
        x, align = col[key]
        ax.text(x, header_y, label, fontsize=6.5, color=GREY, fontweight='bold', ha=align, va='center')
    ax.plot([0, 1], [header_y + 0.35, header_y + 0.35], color='#dddddd', linewidth=0.8)

    # bar length: the largest move fills half of the bar column
    bar_left, bar_right = col['bar']
    bar_mid = (bar_left + bar_right) / 2
    max_abs = max(df['1d P&L [%]'].abs().max(), 0.01)

    for i, row in df.iterrows():
        y = header_y + 1 + i
        d1, d7, rating = row['1d P&L [%]'], row['7d P&L [%]'], int(row['Rating'])

        # background: best row (with gain) tinted green, otherwise alternating light
        if i == 0 and d1 > 0:
            background = '#e9f6ec'
        else:
            background = '#f7f7f7' if i % 2 else 'white'
        ax.add_patch(mpatches.FancyBboxPatch((0, y - 0.42), 1, 0.84, boxstyle='round,pad=0,rounding_size=0.01',
                                             facecolor=background, edgecolor='none', zorder=0))

        ax.text(col['rank'][0], y, str(i + 1), fontsize=8, color=GREY, ha='center', va='center')
        name_text = ax.text(col['name'][0], y, row['Name'], fontsize=9, color=DARK, fontweight='bold', ha='left', va='center')
        price_text = ax.text(col['price'][0], y, format_euro(row['Piece [€]']), fontsize=8.5, color=DARK, ha='right', va='center')
        # Long names (e.g. "Trane Technologies") must not run into the price: shorten with "…" until they fit
        renderer = fig.canvas.get_renderer()
        gap = price_text.get_window_extent(renderer).x0 - 6
        name = str(row['Name'])
        while len(name) > 4 and name_text.get_window_extent(renderer).x1 > gap:
            name = name[:-1]
            name_text.set_text(name.rstrip() + '…')

        # 1d P&L as a bar from the middle
        length = d1 / max_abs * (bar_right - bar_left) / 2
        ax.add_patch(mpatches.Rectangle((min(bar_mid, bar_mid + length), y - 0.15), abs(length), 0.3,
                                        facecolor=pct_color(d1), edgecolor='none', alpha=0.85, zorder=1))
        ax.plot([bar_mid, bar_mid], [y - 0.3, y + 0.3], color='#cccccc', linewidth=0.8, zorder=2)

        ax.text(col['d1'][0], y, f"{d1:+.2f} %", fontsize=9, color=pct_color(d1), fontweight='bold', ha='right', va='center')
        ax.text(col['d7'][0], y, f"{d7:+.1f} %", fontsize=7.5, color=pct_color(d7), ha='right', va='center')

        # score 1-10 as a small coloured badge: green from 8, red up to 3, grey in between (– = not available)
        badge = GREEN if rating >= 8 else RED if 1 <= rating <= 3 else GREY
        ax.text(col['rating'][0], y, str(rating) if rating else '–', fontsize=7.5, color='white', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=badge, edgecolor='none'))

    fig.patch.set_facecolor('white')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_pnl_image(df, output_path, title='', subtitle='', dpi=200):
    """Daily P&L of the own positions (crypto and stocks together) in the style of the daily summary:
    sorted by 1d P&L in €, bar for the 1d P&L, total row at the bottom.
    df columns: Name, Type, Value [€], 1d P&L [€], 1d P&L [%], Total P&L [€], Total P&L [%]."""
    GREEN, RED, GREY, DARK = '#1e8449', '#c0392b', '#8a8a8a', '#222222'

    def pct_color(value):
        return GREEN if value > 0 else RED if value < 0 else GREY

    def euro_signed(value):
        return ('+' if value > 0 else '−' if value < 0 else '') + format_euro(abs(value))

    df = df.sort_values('1d P&L [€]', ascending=False).reset_index(drop=True)
    rows = len(df)
    fig, ax = plt.subplots(figsize=(6.4, 1.4 + (rows + 1) * 0.42))
    ax.set_xlim(0, 1)
    ax.set_ylim(rows + 2.7, -0.2)   # y downwards: title, header, rows, total row
    ax.axis('off')

    col = {'rank': (0.03, 'center'), 'name': (0.07, 'left'), 'value': (0.42, 'right'),
           'bar': (0.45, 0.63), 'd1': (0.79, 'right'), 'total': (1.0, 'right')}
    ax.text(0.0, 0.25, title, fontsize=12, fontweight='bold', color=DARK, va='center')
    ax.text(1.0, 0.25, subtitle, fontsize=7.5, color=GREY, va='center', ha='right')
    header_y = 1.0
    for key, label in (('rank', '#'), ('name', 'NAME'), ('value', 'VALUE'), ('d1', '1D P&L'), ('total', 'TOTAL P&L')):
        x, align = col[key]
        ax.text(x, header_y, label, fontsize=6.5, color=GREY, fontweight='bold', ha=align, va='center')
    ax.plot([0, 1], [header_y + 0.35, header_y + 0.35], color='#dddddd', linewidth=0.8)

    bar_left, bar_right = col['bar']
    bar_mid = (bar_left + bar_right) / 2
    max_abs = max(df['1d P&L [€]'].abs().max(), 0.01)
    renderer = fig.canvas.get_renderer()

    def money_cells(y, euro, pct, weight='bold'):
        # € value on top, % below in small print (1D and total)
        for key, value, percent in (('d1', euro[0], pct[0]), ('total', euro[1], pct[1])):
            ax.text(col[key][0], y - 0.13, euro_signed(value), fontsize=8.5, color=pct_color(value),
                    fontweight=weight, ha='right', va='center')
            sign = '+' if percent > 0 else '−' if percent < 0 else ''
            ax.text(col[key][0], y + 0.22, f"{sign}{abs(percent):.2f} %", fontsize=6.5, color=pct_color(percent),
                    ha='right', va='center')

    for i, row in df.iterrows():
        y = header_y + 1 + i
        d1 = row['1d P&L [€]']
        background = '#e9f6ec' if i == 0 and d1 > 0 else '#f7f7f7' if i % 2 else 'white'
        ax.add_patch(mpatches.FancyBboxPatch((0, y - 0.42), 1, 0.84, boxstyle='round,pad=0,rounding_size=0.01',
                                             facecolor=background, edgecolor='none', zorder=0))
        ax.text(col['rank'][0], y, str(i + 1), fontsize=8, color=GREY, ha='center', va='center')
        name_text = ax.text(col['name'][0], y - 0.13, str(row['Name']), fontsize=9, color=DARK, fontweight='bold',
                            ha='left', va='center')
        ax.text(col['name'][0], y + 0.22, 'Crypto' if row['Type'] == 'crypto' else 'Stock', fontsize=6.5,
                color=GREY, ha='left', va='center')
        value_text = ax.text(col['value'][0], y, format_euro(row['Value [€]']), fontsize=8.5, color=DARK,
                             ha='right', va='center')
        # Long names must not run into the value: shorten with "…" until they fit
        gap = value_text.get_window_extent(renderer).x0 - 6
        name = str(row['Name'])
        while len(name) > 4 and name_text.get_window_extent(renderer).x1 > gap:
            name = name[:-1]
            name_text.set_text(name.rstrip() + '…')

        length = d1 / max_abs * (bar_right - bar_left) / 2
        ax.add_patch(mpatches.Rectangle((min(bar_mid, bar_mid + length), y - 0.15), abs(length), 0.3,
                                        facecolor=pct_color(d1), edgecolor='none', alpha=0.85, zorder=1))
        ax.plot([bar_mid, bar_mid], [y - 0.3, y + 0.3], color='#cccccc', linewidth=0.8, zorder=2)
        money_cells(y, (d1, row['Total P&L [€]']), (row['1d P&L [%]'], row['Total P&L [%]']))

    # total row: sum of all positions, percentages relative to yesterday's value / cost
    y = header_y + 1 + rows + 0.25
    ax.plot([0, 1], [y - 0.55, y - 0.55], color='#bbbbbb', linewidth=0.8)
    value = df['Value [€]'].sum()
    d1 = df['1d P&L [€]'].sum()
    total = df['Total P&L [€]'].sum()
    yesterday, cost = value - d1, value - total
    ax.text(col['name'][0], y, 'Total', fontsize=9, color=DARK, fontweight='bold', ha='left', va='center')
    ax.text(col['value'][0], y, format_euro(value), fontsize=8.5, color=DARK, fontweight='bold', ha='right', va='center')
    money_cells(y, (d1, total), (d1 / yesterday * 100 if yesterday else 0.0, total / cost * 100 if cost else 0.0))

    fig.patch.set_facecolor('white')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_pnl_chart(weekly, output_path, title='', dpi=200):
    """Profit/loss per week as bars: realized (sales of the week, from transactions.csv) in full colour,
    for the current week also the unrealized price change of the open positions in light colour on top.
    weekly: DataFrame with index = week start (Monday) and columns realized, unrealized;
    weekly.attrs['sales'] = {week: [profit, ...]} splits each bar into one part per sale and
    weekly.attrs['open'] = [change, ...] the light part of the current week into one part per open asset
    (same colour, white line between the parts, no names): gains upwards, losses downwards."""
    GREEN, RED, GREY = '#1e8449', '#c0392b', '#8a8a8a'
    realized = weekly['realized'].to_numpy(float)
    unrealized = weekly['unrealized'].to_numpy(float)
    total_realized, open_change = realized.sum(), unrealized[-1]

    def euro_signed(value):
        return ('+' if value > 0.005 else '−' if value < -0.005 else '') + format_euro(abs(value))

    def color(value):
        return GREEN if value >= 0 else RED

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    # header: realized in the period large; below it in brackets the unrealized gains | losses of this week
    open_values = weekly.attrs.get('open') or ([open_change] if abs(open_change) > 0.005 else [])
    open_up = sum(v for v in open_values if v > 0)
    open_down = sum(v for v in open_values if v < 0)

    fig.text(0.02, 0.97, title, fontsize=9, color=GREY, fontweight='bold', va='top')
    fig.text(0.02, 0.90, euro_signed(total_realized), fontsize=16, color=color(total_realized),
             fontweight='bold', va='top')
    bracket = fig.text(0.02, 0.78, "(", fontsize=9, color=GREY, va='top')
    parts = [(f"+{open_up:,.2f}", GREEN), (" | ", GREY), (f"−{abs(open_down):,.2f}", RED),
             (") € unrealised this week", GREY)]
    renderer = fig.canvas.get_renderer()
    left = bracket.get_window_extent(renderer).x1
    for text, colour in parts:   # coloured pieces one after the other
        piece = fig.text(left / fig.bbox.width, 0.78, text, fontsize=9, color=colour, va='top',
                         alpha=0.6 if colour != GREY else 1)
        left = piece.get_window_extent(renderer).x1
    # right side of the header: period and legend, so that the chart itself gets the full height
    fig.text(0.98, 0.97, f"realised from sales since {weekly.index[0]:%d/%m/%Y}", fontsize=7.5, color=GREY,
             va='top', ha='right')

    x = np.arange(len(weekly))
    sales_by_week = weekly.attrs.get('sales', {})
    for i, week in enumerate(weekly.index):
        parts = sales_by_week.get(week) or ([realized[i]] if abs(realized[i]) > 0.005 else [])
        # one part per sale: gains stacked upwards from 0, losses downwards, a white line between them
        heights = {True: 0.0, False: 0.0}
        for value in parts:
            up = value >= 0
            ax.bar(i, value, bottom=heights[up], width=0.7, color=color(value), alpha=0.9, zorder=3,
                   edgecolor='white', linewidth=0.8)
            heights[up] += value
    # current week: the unrealized part in light colour, at the end of this week's gains (if it is a gain)
    # or of this week's losses (if it is a loss), so that realized and unrealized parts never overlap
    last = len(weekly) - 1
    this_week = sales_by_week.get(weekly.index[last]) or ([realized[last]] if abs(realized[last]) > 0.005 else [])
    gains_end, losses_end = sum(v for v in this_week if v > 0), sum(v for v in this_week if v < 0)
    open_parts = weekly.attrs.get('open') or ([open_change] if abs(open_change) > 0.005 else [])
    # one light part per open asset: unrealized gains on top of the realized gains, losses below the losses
    ends = {True: gains_end, False: losses_end}
    for value in open_parts:
        up = value >= 0
        ax.bar(last, value, bottom=ends[up], width=0.7, color=color(value), alpha=0.3, zorder=3,
               edgecolor='white', linewidth=0.8)
        ends[up] += value
    open_gains_end, open_losses_end = ends[True], ends[False]
    # dashed outline around the light parts: they are not realized yet
    for start, end in ((gains_end, open_gains_end), (losses_end, open_losses_end)):
        if abs(end - start) > 0.005:
            ax.bar(last, end - start, bottom=start, width=0.7, fill=False, zorder=4,
                   edgecolor=color(end - start), linewidth=0.8, linestyle='--', alpha=0.6)
    bottom = gains_end if open_change >= 0 else losses_end
    ax.axhline(0, color=GREY, linewidth=0.8, zorder=4)
    tops = np.concatenate([realized, [open_gains_end, open_losses_end, bottom + open_change]])
    low, high = min(tops.min(), 0), max(tops.max(), 0)
    for parts in sales_by_week.values():   # stacked gains and losses can reach further than the sum
        low = min(low, sum(value for value in parts if value < 0))
        high = max(high, sum(value for value in parts if value > 0))
    pad = (high - low) * 0.18 or 1
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlim(-0.6, len(weekly) - 0.4)
    # value of each week above (gain) or below (loss) its bar; empty weeks without a label
    for i, (week, value) in enumerate(zip(weekly.index, realized)):
        if abs(value) > 0.005:
            # sum of the week at the end of the gains (or losses), not inside the stacked parts
            parts = sales_by_week.get(week) or [value]
            end = sum(v for v in parts if v > 0) if value >= 0 else sum(v for v in parts if v < 0)
            side_has_open = (open_gains_end != gains_end) if value >= 0 else (open_losses_end != losses_end)
            if i == last and side_has_open:
                # the unrealized part sits on top: the realized sum in white inside the realized part
                ax.text(i, end / 2, f"{value:,.0f}", fontsize=5.5, color='white', fontweight='bold',
                        ha='center', va='center', zorder=5)
                continue
            ax.text(i, end + (pad * 0.12 if value >= 0 else -pad * 0.12), f"{value:,.0f}", fontsize=5.5,
                    color=color(value), ha='center', va='bottom' if value >= 0 else 'top')
    labels = [f"{day:%d/%m}" for day in weekly.index]
    labels[-1] = 'this wk'
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f} €"))
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#dddddd')
    ax.tick_params(colors=GREY, labelsize=7, length=0)
    ax.grid(axis='y', color='#eeeeee', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [mpatches.Patch(color=GREY, alpha=0.9, label='realised (sales of the week)'),
               mpatches.Patch(facecolor=GREY, alpha=0.3, edgecolor=GREY, linestyle='--',
                              label='unrealised (open positions, this week)')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.90), ncol=1, fontsize=6.5,
               frameon=False, handlelength=1.6, labelcolor=GREY)
    fig.subplots_adjust(top=0.74, left=0.13, right=0.98, bottom=0.1)
    plt.savefig(output_path, dpi=dpi, facecolor='white')
    plt.close(fig)


def create_transactions_image(rows, output_path, title='', subtitle='', dpi=200):
    """Transactions (/transactions, /history) as a table in the style of the daily summary.
    rows: dicts with date (datetime), name, type ('crypto'/'stock'), action ('buy'/'sell'), amount (€), qty."""
    GREEN, RED, GREY, DARK = '#1e8449', '#c0392b', '#8a8a8a', '#222222'
    rows = sorted(rows, key=lambda row: row['date'], reverse=True)   # newest first
    count = len(rows)
    fig, ax = plt.subplots(figsize=(6.4, 0.95 + count * 0.42))
    ax.set_xlim(0, 1)
    ax.set_ylim(count + 1.6, -0.2)   # y downwards: title on top, then header, then rows
    ax.axis('off')

    # columns (x position, alignment)
    col = {'rank': (0.03, 'center'), 'date': (0.07, 'left'), 'name': (0.225, 'left'), 'action': (0.48, 'center'),
           'qty': (0.65, 'right'), 'amount': (0.83, 'right'), 'unit': (1.0, 'right')}
    ax.text(0.0, 0.25, title, fontsize=12, fontweight='bold', color=DARK, va='center')
    ax.text(1.0, 0.25, subtitle, fontsize=7.5, color=GREY, va='center', ha='right')
    header_y = 1.0
    for key, label in (('rank', '#'), ('date', 'DATE'), ('name', 'NAME'), ('action', 'ACTION'),
                       ('qty', 'QUANTITY'), ('amount', 'AMOUNT'), ('unit', 'PRICE/UNIT')):
        x, align = col[key]
        ax.text(x, header_y, label, fontsize=6.5, color=GREY, fontweight='bold', ha=align, va='center')
    ax.plot([0, 1], [header_y + 0.35, header_y + 0.35], color='#dddddd', linewidth=0.8)

    renderer = fig.canvas.get_renderer()
    for i, row in enumerate(rows):
        y = header_y + 1 + i
        ax.add_patch(mpatches.FancyBboxPatch((0, y - 0.42), 1, 0.84, boxstyle='round,pad=0,rounding_size=0.01',
                                             facecolor='#f7f7f7' if i % 2 else 'white', edgecolor='none', zorder=0))
        ax.text(col['rank'][0], y, str(i + 1), fontsize=8, color=GREY, ha='center', va='center')
        ax.text(col['date'][0], y, row['date'].strftime('%d/%m/%Y'), fontsize=7.5, color=GREY, ha='left', va='center')
        name_text = ax.text(col['name'][0], y, str(row['name']), fontsize=9, color=DARK,
                            fontweight='bold', ha='left', va='center')
        buy = row['action'] == 'buy'
        badge = ax.text(col['action'][0], y, 'BUY' if buy else 'SELL', fontsize=7, color='white', fontweight='bold',
                        ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor=GREEN if buy else RED,
                                                            edgecolor='none'))
        # Long names must not run into the action badge: shorten with "…" until they fit
        gap = badge.get_window_extent(renderer).x0 - 10
        name = str(row['name'])
        while len(name) > 4 and name_text.get_window_extent(renderer).x1 > gap:
            name = name[:-1]
            name_text.set_text(name.rstrip() + '…')
        qty = f"{row['qty']:.8f}".rstrip('0').rstrip('.') if row['type'] == 'crypto' else f"{row['qty']:g}"
        ax.text(col['qty'][0], y, qty, fontsize=8, color=DARK, ha='right', va='center')
        ax.text(col['amount'][0], y, format_euro(row['amount']), fontsize=8.5, color=DARK, fontweight='bold',
                ha='right', va='center')
        unit_price = row['amount'] / row['qty'] if row['qty'] else 0.0
        ax.text(col['unit'][0], y, format_euro(unit_price), fontsize=7.5, color=GREY, ha='right', va='center')

    fig.patch.set_facecolor('white')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


LOG_FILE = 'logfile.txt'
LOG_MAX_BYTES = 1_000_000   # larger: renamed to logfile.old.txt, a new log starts

def log_print(message, level='INFO'):
    """One clear line per event for error analysis and monitoring:
    2026-11-15 16:38:37  WARN   CoinGecko near: too many requests, retry 1 in 60 s
    level: ERROR (something failed), WARN (problem, program continues), INFO (status).
    HTML tags are removed; further lines are indented under the message."""
    text = _strip_telegram_html(message)
    lines = text.splitlines() or ['']
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    entry = f"{stamp}  {level:<5}  {lines[0]}\n" + ''.join(f"{'':28}{line}\n" for line in lines[1:])
    print(entry, end='')
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > LOG_MAX_BYTES:
            os.replace(LOG_FILE, 'logfile.old.txt')
        with open(LOG_FILE, 'a', encoding='utf-8') as file:
            file.write(entry)
    except OSError:
        pass   # the log must never stop the program


def _strip_telegram_html(message):
    # Only the Telegram tags (<b>, <i>, <pre>, <a href=...> ...), not e.g. "<module>" in a traceback
    text = re.sub(r'</?(?:b|i|u|s|pre|code|a|blockquote)(?:\s[^>]*)?>', '', str(message))
    return html.unescape(text).strip()


def log_message(kind, message):
    """Telegram message as a short status line (first visible line only, not the whole text)."""
    first = _strip_telegram_html(message).split('\n', 1)[0]
    log_print(f"Telegram {kind}: {first[:100]}")


CLAUDE_NO_KEY = ('No Claude API key: /ai_api_decision3s and /ai_api_plan need an api_key in config.ini '
                 '[anthropic] (console.anthropic.com, costs money per request). /ai_prompt works without a key.')


# Anthropic list prices in USD per million tokens (input, output) and per web search; billed in USD.
# Models not listed here: tokens are shown, the cost as "not known".
CLAUDE_PRICES = {'claude-opus-5-5': (4.0, 20.0), 'claude-sonnet-5-5': (2.0, 10.0),
                 'claude-haiku-5-5': (0.10, 0.50), 'claude-fable-5-2': (8.0, 40.0)}
CLAUDE_HAIKU_LONG = (0.50, 2.50)   # Haiku 5.5 with more than 100,000 input tokens in one request
CLAUDE_WEB_SEARCH_PRICE = 0.01    # USD per search ($10 per 1,000)


def claude_cost(model, tokens_in, tokens_out):
    """List price of one request in USD, None for an unknown model."""
    prices = CLAUDE_PRICES.get(model)
    if prices is None:
        return None
    if model == 'claude-haiku-5-5' and tokens_in > 100_000:
        prices = CLAUDE_HAIKU_LONG
    return (tokens_in * prices[0] + tokens_out * prices[1]) / 1_000_000


def claude_usage_text(usage):
    """One line for Telegram, e.g. '12,345 tokens in · 678 out · 3 web searches · $0.06 (≈ 0.05 €) · 41 s · claude-sonnet-5-5'."""
    parts = [f"{usage['tokens_in']:,} tokens in", f"{usage['tokens_out']:,} out"]
    if usage['searches']:
        parts.append(f"{usage['searches']} web search{'es' if usage['searches'] != 1 else ''}")
    if usage['cost'] is None:
        parts.append('cost not known for this model')
    else:
        cost = f"${usage['cost']:.3f}"
        try:   # rough EUR value with the exchange rate CaSa uses for the US stocks
            cost += f" (≈ {usage['cost'] * float(_fx_rates('USD', '5d')['Rate'].iloc[-1]):.3f} €)"
        except Exception:
            pass
        parts.append(cost)
    parts += [f"{usage['seconds']:.0f} s", ', '.join(usage['models'])]
    return ' · '.join(parts)


def ai_buttons(asset_type, symbol):
    """Buttons below alarms, infos and /check: 🤖 AI prompt (as /ai_prompt); with a Claude API key also
    🤖 AI decision (API) and 🤖 AI plan (API) (as /ai_api_decision3s and /ai_api_plan)."""
    asset = f"{asset_type}|{symbol.lower()}"
    row = [('🤖 AI prompt', f"aibtn|{asset}")]
    if claude_available():
        row += [('🤖 AI decision (API)', f"aidbtn|{asset}"), ('🤖 AI plan (API)', f"aipbtn|{asset}")]
    return [row]


def claude_available():
    """True when a Claude API key is set in config.ini [anthropic] api_key."""
    return bool(anthropic_api_key)


def claude(prompt, system=None, max_tokens=16000, effort='medium', web_search=False, with_usage=False):
    """Sends a prompt to the Claude API and returns the answer as text.
    Only works with a key in config.ini [anthropic] api_key, otherwise RuntimeError.
    effort: low, medium, high (more effort = better answer, but slower and more expensive).
    web_search=True: Claude may search the web (at most 5 searches, each search costs extra).
    with_usage=True: returns (text, usage) with tokens, web searches, list price in USD, seconds and model(s).
    Refusal fallbacks are on (not for Haiku): if the model refuses, the API retries once with a fallback model.
    Errors: PermissionError (key wrong or not allowed), RuntimeError (everything else)."""
    if not anthropic_api_key:
        raise RuntimeError("Claude API not configured: no api_key in config.ini [anthropic]")
    try:
        import anthropic   # only needed when the API is used
    except ImportError:
        raise RuntimeError("Python package 'anthropic' missing: python -m pip install anthropic")

    messages = [{"role": "user", "content": prompt}]
    request = dict(model=anthropic_model, max_tokens=max_tokens, messages=messages,
                   output_config={"effort": effort})
    if 'haiku' not in anthropic_model:   # Haiku has no server-side fallback (400)
        request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if web_search:   # Haiku: older search tool without dynamic filtering
        version = 'web_search_20250305' if 'haiku' in anthropic_model else 'web_search_20260209'
        request['tools'] = [{"type": version, "name": "web_search", "max_uses": 5}]
    if system:
        request['system'] = system
    start = time.time()
    texts, tokens_in, tokens_out, searches, cost, models = [], 0, 0, 0, 0.0, []
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)   # retries 429/5xx itself
        for _ in range(4):   # a long web search can pause the answer: send it back so that Claude continues
            # streaming: no timeout on long answers; only the finished message is used
            with client.beta.messages.stream(**request) as stream:
                response = stream.get_final_message()
            texts += [block.text for block in response.content if block.type == 'text']
            usage = response.usage
            tokens_in, tokens_out = tokens_in + usage.input_tokens, tokens_out + usage.output_tokens
            found = getattr(getattr(usage, 'server_tool_use', None), 'web_search_requests', 0) or 0
            searches += found
            # the model that answered (with a refusal fallback it can differ from config.ini)
            model = getattr(response, 'model', None) or anthropic_model
            models += [model] if model not in models else []
            part = claude_cost(model, usage.input_tokens, usage.output_tokens)
            cost = None if cost is None or part is None else cost + part + found * CLAUDE_WEB_SEARCH_PRICE
            if response.stop_reason != 'pause_turn':
                break
            messages.append({"role": "assistant", "content": response.content})
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        log_print(f"Claude API: key rejected (HTTP {e.status_code})", 'ERROR')
        raise PermissionError("Claude API key rejected, check config.ini [anthropic] api_key") from None
    except anthropic.NotFoundError:
        log_print(f"Claude API: model {anthropic_model} not found", 'ERROR')
        raise RuntimeError(f"Claude model '{anthropic_model}' not found, check config.ini [anthropic] model") from None
    except anthropic.RateLimitError:
        log_print("Claude API: too many requests", 'WARN')
        raise RuntimeError("Claude API: too many requests, try again later") from None
    except anthropic.APIStatusError as e:
        log_print(f"Claude API: error HTTP {e.status_code}", 'ERROR')
        raise RuntimeError(f"Claude API error HTTP {e.status_code}") from None
    except anthropic.APIConnectionError:
        log_print("Claude API: no connection", 'WARN')
        raise RuntimeError("Claude API: no connection") from None

    if response.stop_reason == 'refusal':
        details = getattr(response, 'stop_details', None)
        category = getattr(details, 'category', None) or 'unknown'
        log_print(f"Claude API: request refused ({category})", 'WARN')
        raise RuntimeError(f"Claude refused the request ({category})")
    log_print(f"Claude API: answer in {time.time() - start:.1f} s, {tokens_in} tokens in, {tokens_out} tokens out"
              + (f", {searches} web searches" if searches else '')
              + (f", about ${cost:.3f}" if cost is not None else '')
              + (" (cut off at max_tokens)" if response.stop_reason == 'max_tokens' else '')
              + (" (still paused after 4 parts)" if response.stop_reason == 'pause_turn' else ''))
    text = ''.join(texts).strip()
    if with_usage:
        return text, {'tokens_in': tokens_in, 'tokens_out': tokens_out, 'searches': searches, 'cost': cost,
                      'seconds': time.time() - start, 'models': models}
    return text


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