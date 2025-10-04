# Modified func.py with volume bars added to the plot

# --- Standard Library ---
import os
import time
import base64, hashlib, hmac
from datetime import datetime, timedelta
from urllib.parse import quote  # only if you need to encode URL parts
import urllib.request, json

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

token_pushover  = ""
user_pushover = ""
one_day_price_change = ""
seven_day_price_change = ""
one_day_profit_limit = ""
output_dir = 'plots'

def get_stock(symbol,period):
    df = pd.DataFrame()

    if period =='1mo':
        interval='1h'
    else:
        interval='1d'

    for i in range(5):
        try:
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)  # Getting the last 100 days of stock data
            if not df.empty:  # Check whether the retrieval was successful
                break
            else:
                time.sleep(300)
        except Exception as e:
            infos = {'sym': symbol, 'tries': int(i), 'function': 'get_stock','exception': e}
            msg = "Script try number %(tries)s to %(function)s of symbol %(sym)s \n\n%(exception)s" %infos
            pushover(msg) 
            time.sleep(300)
   

    if 'Date' not in df.columns:
        df['Date'] = df.index  # Convert the DatetimeIndex into a 'Date' column

    df = df.tail(100)
    df['Price'] = df['Close']

    df_inverted = df.reset_index(drop=True)

    df_inverted.rename(columns={'index': 'Date'}, inplace=True)

    columns_to_drop = ['Close', 'Open', 'High', 'Low', 'Dividends', 'Stock Splits']
    df_dropped = df_inverted.drop(columns=[col for col in columns_to_drop if col in df_inverted.columns])

    return df_dropped


def get_crypto(symbol,params_historical):
    # Set the API endpoint and parameters for historical data
    url_historical = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'

    for i in range(5):
        try:
            response_historical = requests.get(url_historical, params=params_historical)
            data = response_historical.json()

            if 'prices' in data and data['prices']:
                break
            else:
                time.sleep(300)
        except Exception as e:
            infos = {'sym': symbol, 'tries': i+1, 'function': 'get_crypto', 'exception': e}
            msg = "Script try number %(tries)s to %(function)s of symbol %(sym)s \n\n%(exception)s" % infos
            print(msg)
            time.sleep(300)

    # Extract the data for the last 100 days
    prices = data['prices']
    volumes = data['total_volumes']
    last_100_days_data = {}

    for price, volume in zip(prices, volumes):  # excluding the latest price from the loop
        # Convert timestamp to datetime and shift by one day
        date = pd.Timestamp(price[0], unit='ms')
        last_100_days_data[date.strftime('%Y-%m-%d %H:%M:%S')] = {'Price': price[1], 'Volume': volume[1]}

    # Convert the data to pandas DataFrame for easier analysis
    df = pd.DataFrame(last_100_days_data).T.reset_index()
    df.columns = ['Date', 'Price', 'Volume']

    if len(df) > 100:
        df = df.iloc[:-1]  # Remove the last row to ensure we have only 100 rows

    df['Date'] = pd.to_datetime(df['Date'])
    return df

def calc_indicator_fuctions(df):
   
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

    #save_csv(df,"test.csv") #For debugging
    return df


def save_csv(df, path=None, index=False, date_cols=()):
    d = df.copy()
    for c in date_cols:
        if c in d: d[c] = pd.to_datetime(d[c], errors='coerce').dt.strftime('%d.%m.%Y')
    return d.to_csv(path, sep=";", decimal=",", index=index, encoding="utf-8-sig", lineterminator="\n")

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

    # Return RSI
    return rsi


def calculate_momentum(df, column, window=10):
    momentum = df[column] - df[column].shift(window)
    return momentum

def calculate_vwma(df, window=20):
    if 'Volume' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    pv = df['Price'] * df['Volume']
    return pv.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()

def human_format(x, pos):
    if x >= 1_000_000_000:
        return f"{x*1.0/1_000_000_000:.0f}B"
    elif x >= 1_000_000:
        return f"{x*1.0/1_000_000:.0f}M"
    elif x >= 1_000:
        return f"{x*1.0/1_000:.0f}k"
    else:
        return f"{int(x)}"
    
def human_format_euro(x, pos):
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.0f}B €"
    elif x >= 1_000_000:
        return f"{x/1_000_000:.0f}M €"
    elif x >= 1_000:
        return f"{x/1_000:.0f}k €"
    else:
        return f"{int(x)} €"

def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'B', 'T'][magnitude])

def human_format_euro(num, pos):
    return human_format(num, pos) + '€'

def plot_and_save(df, symbol, data_type, zero_line=None):
    if df is not None and not df.empty:
        #----------Create
        fig, (ax1, ax3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax2 = ax1.twinx()

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
        ax1.plot(df['Date'], df['Price'], label='Price', color='k', linewidth=1.5)
        ax1.plot(df['Date'], df['Fast EMA'], label='Fast EMA', linestyle=':', color='green')
        ax1.plot(df['Date'], df['Slow EMA'], label='Slow EMA', linestyle=':', color='red')
        ax1.plot(df.loc[df['Position']== 1.0, 'Date'], df.loc[df['Position']== 1.0, 'Fast EMA'], '^',  markersize=10, color='green', label='Buy')
        ax1.plot(df.loc[df['Position']==-1.0, 'Date'], df.loc[df['Position']==-1.0, 'Fast EMA'], 'v',  markersize=10, color='red', label='Sell')
        ax1.plot(df.loc[df['HighP']==1, 'Date'],      df.loc[df['HighP']==1, 'Fast EMA'], 'o', markersize=7, label='Max')
        ax1.plot(df.loc[df['LowP']==1,  'Date'],      df.loc[df['LowP']==1,  'Fast EMA'], 'o', markersize=7, label='Min')
        ax2.plot(df['Date'], df['Percentage Deviation'], color='k', linewidth=1.0, label='Deviation (%)')

        if 'Volume' in df.columns:
            colors = ['green' if close > open_val else 'red' for open_val, close in zip(df['Price'].shift(1), df['Price'])]
            colors[0] = 'gray'  # First bar neutral, since no previous day
            ax3.bar(df['Date'], df['Volume'], color=colors, alpha=0.7, label='Volume')  # Increased alpha for better visibility
            
        #-------------Save
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{symbol}.png"), bbox_inches='tight')
        plt.close(fig)

def pushover(message: str):
    # Log file
    with open('logfile.txt', 'a') as f:
        f.write(time.strftime("%x %X", time.localtime()) + '  \n' + message + '\n\n')

    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token_pushover,
                "user": user_pushover,
                "message": message,
                "url_title": "Link",
                "html": 1,
            },
            timeout=10,
        )
    except Exception as e:
        print(f"Pushover error: {e}")

def pushover_image(symbol: str, message: str):
    with open('logfile.txt', 'a') as f:
        f.write(time.strftime("%x %X", time.localtime()) + '  \n' + message + '\n\n')

    file_name = os.path.join(output_dir, symbol + ".png")
    try:
        with open(file_name, "rb") as fh:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": token_pushover,
                    "user": user_pushover,
                    "url_title": "Link",
                    "message": message,
                    "html": 1,
                },
                files={"attachment": ("image.jpg", fh, "image/jpeg")},
                timeout=30,
            )
    except Exception as e:
        print(f"Pushover image error: {e}")


def google_trends(search):
    
    for i in range(5):
        try:
            pytrends = TrendReq(hl='en-US', tz=360) 
            break
        except Exception as e:
            infos = {'tries': int(i), 'function': __name__,'exception': e}
            msg = "Script try number %(tries)s to %(function)s \n\n%(exception)s" %infos
            pushover(msg)
            time.sleep(300)
    
    kw_list = [search] # List of keywords to get data
    pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m') # Trend for 12 months
    data = pytrends.interest_over_time() 
    data = data.reset_index() 

    df = pd.DataFrame()
    df = pd.DataFrame(columns=['Datum', 'Interesse'])
    df['Datum'] = data['date']
    df['Interesse'] = data[search]

    return df.tail(1)['Interesse'].values[0], df.tail(2)['Interesse'].values[0]

def seven_day_slope_pct(df, current):
    # Ensure that the DataFrame has at least 8 rows to view both current and previous 7 days
    if len(df) < 8:
        raise ValueError("The DataFrame must contain at least 8 lines.")
    
    y = []

    if current:
        subset_df = df.tail(7)
    else:
        subset_df = df[-8:-1]

    subset_df.reset_index(inplace=True)  # Includes the index in a column called 'index'

    # Linear regression
    try:
        result = linregress(subset_df['index'], subset_df['Price'])
    except Exception as e:
        raise RuntimeError(f"Error in linear regression: {e}")

    steigung = result.slope
    y_achsenabschnitt = result.intercept

    # Calculate the predicted y-values for day 1 and day 7
    y.append(steigung * subset_df['index'].iloc[0] + y_achsenabschnitt)
    y.append(steigung * subset_df['index'].iloc[6] + y_achsenabschnitt)

    # Ensure that y[0] is not 0 to prevent division from zero
    if y[0] == 0:
        raise ValueError("Calculation of the gradient is not possible as y[0] is 0.")

    slope_pct = (y[1] / y[0] - 1)* 100

    return slope_pct

def calc_stat_limits(df, column, window=100, invert=False):
    if column not in df.columns:
        return np.nan, np.nan

    data = df[column].dropna().to_numpy()[-window:]
    if data.size < 2:
        return np.nan, np.nan
    
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
        sig_count = 3
    else:
        sig_count = 0

    return sig_count, quantile_pct


def alarm(df,symbol,watch_list, current_profit_pct, amount_older_than_one_year, amount_older_than_one_year_pct, link, data_type):

    alarms = {}
    alarm_indicator = ""
    score = 0
    alarm_message_add = ""

    #Indikatoren
    RSI14_signal_count, RSI14_quantile_pct = calc_stat_limits(df, 'RSI14', window=100, invert=True) 
    MOM10_signal_count, MOM10_quantile_pct = calc_stat_limits(df, 'Mom10', window=100, invert=False) 
    VMOM10_signal_count, VMOM10_quantile_pct = calc_stat_limits(df, 'VMom10', window=100, invert=False)
    SMA7_signal_count, SMA7_quantile_pct = calc_stat_limits(df, 'SMA7', window=100, invert=False)
    VMA7_signal_count, VMA7_quantile_pct = calc_stat_limits(df, 'VMA7', window=100, invert=False)

    
    #dEMA_pct, signal_count = signal_slope((df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-2]) / df['Slow EMA'].iloc[-2] * 100, signal_count)
    #dVWMA20_pct, signal_count = signal_slope((df['VWMA20'].iloc[-1] - df['VWMA20'].iloc[-2])/df['VWMA20'].iloc[-2]*100,signal_count) #VolRatio20 check before VWMA20

    #current_one_day_price_change_pct = (df['Price'].iloc[-1] - df['Price'].iloc[-2]) / df['Price'].iloc[-2] * 100
    current_EMA_diff_pct = (df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-1]) / df['Slow EMA'].iloc[-1] * 100
    current_seven_days_slope_pct = seven_day_slope_pct(df, True)
    yesterday_EMA_diff_pct = (df['Fast EMA'].iloc[-2] - df['Slow EMA'].iloc[-2]) / df['Slow EMA'].iloc[-2] * 100
    alarm_buy_sell = "Hold"

    # Watch-list & Portfolio-list

    if yesterday_EMA_diff_pct < 0 and current_EMA_diff_pct > 0 and data_type == "100d":
        alarm_buy_sell = "Buy"
        alarm_indicator = "Cross-EMA"
        alarm_code = "101"
        alarm_value = 1

    if current_seven_days_slope_pct > seven_day_price_change:
        alarm_buy_sell = "Buy"
        alarm_indicator = "Positiv-7"
        alarm_code = "121"
        alarm_value = current_seven_days_slope_pct

    if df['LowP'].iloc[-1] < df['LowP'].iloc[-2] and df['LowP'].iloc[-2] > df['LowP'].iloc[-3] and data_type == "100d":
        alarm_buy_sell = "Buy"
        alarm_indicator = "100-Minimum"
        alarm_code = "122"
        alarm_value = 1
        
    # Portfolio-list
    if not watch_list: 
        
        alarm_message_add = "Amount >1 year: {} ({} %)".format(round(amount_older_than_one_year,2),round(amount_older_than_one_year_pct,2))

        if yesterday_EMA_diff_pct > 0 and current_EMA_diff_pct < 0 and data_type == "100d":
            alarm_buy_sell = "Sell"
            alarm_indicator = "Cross-EMA"
            alarm_code = "201"
            alarm_value = 1      
        
        if current_seven_days_slope_pct < (seven_day_price_change * -1):
            alarm_buy_sell = "Sell"
            alarm_indicator = "Negative-7"
            alarm_code = "221"
            alarm_value = current_seven_days_slope_pct

        if df['HighP'].iloc[-1] < df['HighP'].iloc[-2] and df['HighP'].iloc[-2] > df['HighP'].iloc[-3] and data_type == "100d":
            alarm_buy_sell = "Sell"
            alarm_indicator = "100-Maximum"
            alarm_code = "222"
            alarm_value = 1

    if alarm_buy_sell == "Sell": 
        alarm_symbol = "&#9660;"
        alarm_symbol_color = "red"
    if alarm_buy_sell == "Buy":
        alarm_symbol = "&#9650;"
        alarm_symbol_color = "green"

    if alarm_buy_sell != "Hold":
    # Headline (only symbol)
        alarm_headline = f"{symbol}"

    # Helper to build arrow string for signal count
        def arrow_string(count):
            if np.isnan(count):
                return ""
            arrow_up = "+"  # Up
            arrow_down = "-"  # Down
            cycle = "&#8226;"      # Neutral
            if count > 0:
                return f" {arrow_up * int(count)}"
            elif count < 0:
                return f" {arrow_down * abs(int(count))}"
            return f" {cycle}"

        alarm_analysis1, score = tech_analyse1(RSI14_signal_count, score)
        alarm_analysis2, score = tech_analyse2(MOM10_signal_count,VMOM10_signal_count, score)
        alarm_analysis3, score = tech_analyse3(SMA7_signal_count,VMA7_signal_count, score)

        alarm_message = (
            f"Trigger: {alarm_indicator}"
            f"Score: {score}\n"
            f" ({data_type})\n"
            f"RSI14: {round(RSI14_quantile_pct,2)} %{arrow_string(RSI14_signal_count)}\n"
            f"{alarm_analysis1}\n\n"
            f"SMA7: {SMA7_quantile_pct} %{arrow_string(SMA7_signal_count)}\n"
            f"VMA7: {VMA7_quantile_pct} %{arrow_string(VMA7_signal_count)}\n"
            f"{alarm_analysis3}\n\n"
            f"MOM10: {MOM10_quantile_pct} %{arrow_string(MOM10_signal_count)}\n"
            f"VMOM10: {VMOM10_quantile_pct} %{arrow_string(VMOM10_signal_count)}\n"
            f"{alarm_analysis2}\n\n"
            f"{alarm_message_add}\n"
        )

        score = RSI14_signal_count

    # Build HTML – headline separate, body black
        alarms[alarm_code] = {
                "value": alarm_value,
                "msg": (
                    f"<html><body>"
                    # Headline (separate element, 28px, colored)
                    f"<div style='font-size:28px; font-weight:700; color:{alarm_symbol_color};'>"
                    f"{alarm_headline} {alarm_symbol}</div>"
                    # Body (black; \n remains due to white-space:pre-line)
                    f"<div style='color:black; white-space:pre-line;'>{alarm_message}</div>"
                    f"</body></html>"
                )
            }

    return alarms, score


# Analyse 1: Nur RSI
def tech_analyse1(RSI14, score):
    parts = []
    prob_map = {1: "Potentially", 2: "Possible", 3: "Very likely"}
    if RSI14 != 0:
        direction = "correction to rise." if RSI14 > 0 else "correction to fall."
        parts.append(f"{prob_map.get(abs(RSI14), 'Potentially')} {direction}")
    analysis = "Analysis: " + " ".join(s.strip() for s in parts if s)
    score += RSI14
    return analysis, score

# Analyse 2: MOM10 und VMOM10
def tech_analyse2(MOM10, VMOM10, score):
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
    volume_text = VOLUME_MATRIX.get(VMOM10, "Volume support anomaly")
    momentum_text = MOMENTUM_MATRIX.get(MOM10, "movement")

    score += MOM10 + VMOM10

    if MOM10 > 0 and VMOM10 < 0:
        fake_out = " (possible fake-out)"
        score += 0
    else:
        fake_out = ""
    
    return f"Short-term-Analysis: {volume_text} {momentum_text}{fake_out}.", score

# Analyse 3: SMA7 und VMA7
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
    
    # Trend Strength Matrix (basierend auf Absolutwert)
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
    volume_text = VOLUME_MATRIX.get(VMA7, "Volume anomaly")
    trend_text = TREND_MATRIX.get(abs(SMA7), "with trend")

    score += SMA7 + VMA7

    if SMA7 > 0 and VMA7 < 0:
        fake_out = " (possible fake-out)"
        score += 0
    else:
        fake_out = ""

    return f"Trend-Analysis: {volume_text} {trend_text}{fake_out}.", score


def older_than_one_year(df):
    current_date = datetime.now()
    one_year_ago = current_date - timedelta(days=365)
    filtered_df = df[df['Date'] < one_year_ago]
    total_amount = filtered_df['Amount'].sum()
    
    return total_amount

def create_table_image(df, output_path, title):
    
    # Figure erstellen
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Tabelle erstellen
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header-Zeile hervorheben
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Total-Zeile hervorheben (letzte Zeile)
    for i in range(len(df.columns)):
        cell = table[(len(df), i)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold')
    
    # Alternierende Zeilenfarben
    for i in range(1, len(df)):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()