# Modified func.py with volume bars added to the plot

# --- Standardbibliothek ---
import os
import time
import base64, hashlib, hmac
from datetime import datetime, timedelta
from urllib.parse import quote  # nur falls du URL-Teile kodieren musst
import urllib.request, json

# --- Externe Pakete ---
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

token_pushover  = ""
user_pushover = ""

one_day_price_change = ""
seven_day_price_change = ""
one_day_profit_limit = ""
output_dir = 'plots'
av_key = ""

def get_stock(symbol, period):
    # Determine interval and Alpha Vantage function
    if period == '1mo':
        function = 'TIME_SERIES_INTRADAY'
        interval = '60min'
    else:
        function = 'TIME_SERIES_DAILY'

    # Construct API URL
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': av_key
    }

    if function == 'TIME_SERIES_INTRADAY':
        params['interval'] = interval
        params['outputsize'] = 'compact'
    else:
        params['outputsize'] = 'compact'

    df = pd.DataFrame()

    for i in range(5):
        try:
            response = requests.get(base_url, params=params)
            data = response.json()

            key = next(k for k in data if "Time Series" in k)
            df = pd.DataFrame.from_dict(data[key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            if not df.empty:
                break
            else:
                time.sleep(300)
        except Exception as e:
            print(f"Attempt {i+1} failed for {symbol}: {e}")
            time.sleep(300)

    df = df.tail(100)
    df['Date'] = df.index
    df['Price'] = df['4. close'].astype(float)
    df['Volume'] = df['5. volume'].astype(float)  # Keep volume

    df = df.reset_index(drop=True)
    df_dropped = df.drop(columns=[col for col in df.columns if 'open' in col.lower()
                                                       or 'high' in col.lower()
                                                       or 'low' in col.lower()
                                                       or col not in ['Date', 'Price', 'Volume']])  # Keep Volume

    return df_dropped


def get_stock_yf_old(symbol,period):
    # Fetch historical stock data from Yahoo Finance
    # Attempts multiple tries if data retrieval fails
    # Drops unneeded columns before returning the DataFrame
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

    # Calculate Min Max
    df['High'] = df['Fast EMA'].max()
    df['Low'] = df['Fast EMA'].min()
    df['HighP'] = np.where(df['Fast EMA'] == df['High'], 1, 0)
    df['LowP'] = np.where(df['Fast EMA'] == df['Low'], 1, 0)

    # Calculate the slope of Fast EMA
    df['Fast EMA Slope'] = df['Fast EMA'].diff()

    # Calculate RSI14
    df['RSI14'] = calculate_rsi(df, window=14)
    # Calculate Momentum
    df['Mom10'] = calculate_momentum(df, window=10)
    # Calculate VWMA20
    df['VWMA20'] = calculate_vwma(df, window=20)
    # Calculate SMA20
    df['SMA20'] = calculate_sma(df, window=20)

    # Calculate VolMA20
    if 'Volume' in df.columns:
        df['VolMA20'] = df['Volume'].rolling(20, min_periods=10).mean()
        df['VolRatio20'] = df['Volume'] / df['VolMA20']
    save_csv(df,"test.csv")
    return df


def save_csv(df, path=None, index=False, date_cols=()):
    d = df.copy()
    for c in date_cols:
        if c in d: d[c] = pd.to_datetime(d[c], errors='coerce').dt.strftime('%d.%m.%Y')
    return d.to_csv(path, sep=";", decimal=",", index=index, encoding="utf-8-sig", lineterminator="\n")

def calculate_stabw(a: pd.Series, window=20):
    mean = a.tail(window).mean()
    std = a.tail(window).std(ddof=0)
    
    return std/mean*100


def calculate_sma(df, window=20):
    if 'Price' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return df['Price'].rolling(window=window).mean()

def calculate_rsi(df, window=14):
    delta = df['Price'].diff()  # Preisänderungen berechnen
    gain = delta.where(delta > 0, 0)  # Gewinne (positive Änderungen)
    loss = -delta.where(delta < 0, 0)  # Verluste (negative Änderungen)

    # Gleitender Durchschnitt für Gewinne und Verluste
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Berechnung von RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Rückgabe von RSI
    return rsi


def calculate_momentum(df, window=10):
    momentum = df['Price'] - df['Price'].shift(window)
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
        for ax in [ax1, ax3]:
            #ax1.set_title('100 Hours' if data_type=='100h' else '100 Days', fontsize=14)
            ax.xaxis.set_major_locator(mdates.DayLocator() if data_type=='100h' else mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
            ax.grid(True, axis='x', color='black', alpha=0.4)
        
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
            colors[0] = 'gray'  # Erster Balken neutral, da kein Vortag
            ax3.bar(df['Date'], df['Volume'], color=colors, alpha=0.7, label='Volume')  # Alpha erhöht für bessere Sichtbarkeit
            
        #-------------Save
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{symbol}.png"), bbox_inches='tight')
        plt.close(fig)



def get_crypto_price(coin): #in Euro
    url = 'https://api.coingecko.com/api/v3/simple/price?ids={}&vs_currencies=eur'.format(coin)
    
    for i in range(5):
        try:
            response = requests.get(url)
            break
        except Exception as e:
            infos = {'tries': int(i), 'function': 'get_crypto_price','exception': e}
            msg = "Script try number %(tries)s to %(function)s \n\n%(exception)s" %infos
            pushover(msg)
            time.sleep(300)
    
    if response.status_code == 200:
        json_data = response.json()
        price = json_data[coin]['eur']
        return price
    else:
        return -1

def pushover(message: str):
    # Logfile
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
    
    kw_list = [search] # list of keywords to get data 
    pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m') #Trend for 12 month
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

    # Ensure that y[0] is not 0 to prevent division by zero
    if y[0] == 0:
        raise ValueError("Calculation of the gradient is not possible as y[0] is 0.")

    slope_pct = (y[1] / y[0] - 1)* 100

    return slope_pct


def RSI_check(value, sig_count):
    sig = str(round(value, 1))
    if value < 30: 
        sig += " < 30 (buy-oversold)"
        sig_count += 1
    elif value > 70: 
        sig += " > 70 (sell-overbought)"
        sig_count -= 1  
    elif 30 <= value <= 50:
        sig += " (neutral-bearish)"
    else:  # 50 < value <= 70
        sig += " (neutral-bullish)"  # Fixed typo: "bullisch" -> "bullish"

    return sig, sig_count

def signal_slope(slope_pct,sig_count):
    
    sig = str(round(slope_pct,1))
    if slope_pct > 2: 
        sig += "% (buy)"
        sig_count += 1
    elif slope_pct < -2: 
        sig += "% (sell)"
        sig_count += -1
    else: 
        sig += "% (neutral)"
        sig_count += 0

    return sig, sig_count




def VWMAvSMA_check(vwma_value, sma_value, sig_count):
    if pd.isna(vwma_value) or pd.isna(sma_value) or sma_value == 0:
        return "N/A", sig_count

    diff_pct = (vwma_value - sma_value) / sma_value * 100

    if diff_pct > 2:
        sig = f"Strong bullish"
        sig_count += 2
    elif diff_pct > 0:
        sig = f"Slightly bullish"
        sig_count += 1
    elif diff_pct < -2:
        sig = f"Strong bearish"
        sig_count -= 2
    elif diff_pct < 0:
        sig = f"Slightly bearish"
        sig_count -= 1
    else:
        sig = "VWMA = SMA → neutral"
        sig_count = 0

    return sig, sig_count


def alarm(df,symbol,watch_list, current_profit_pct, amount_older_than_one_year, amount_older_than_one_year_pct, link, data_type):

    alarms = {}
    signal_count = 0

    RSI14, signal_count= RSI_check(df['RSI14'].iloc[-1],signal_count)

    dEMA_pct, signal_count = signal_slope((df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-2])/df['Slow EMA'].iloc[-2]*100,signal_count)

    if df['Mom10'].iloc[-2] != 0:
        dMom10_pct, signal_count = signal_slope((df['Mom10'].iloc[-1] - df['Mom10'].iloc[-2]) / df['Mom10'].iloc[-2] * 100, signal_count)
    else:
        dMom10_pct, signal_count = signal_slope(0, signal_count)  

    #VolRatio20 check before VWMA20

    dVWMAvSMA20, signal_count = VWMAvSMA_check(df["VWMA20"].iloc[-1], df["SMA20"].iloc[-1], signal_count)

    
    current_one_day_price_change_pct = (df['Price'].iloc[-1] - df['Price'].iloc[-2]) / df['Price'].iloc[-2] * 100
    current_EMA_diff_pct = (df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-1]) / df['Slow EMA'].iloc[-1] * 100
    current_seven_days_slope_pct = seven_day_slope_pct(df,True)

    yesterday_EMA_diff_pct = (df['Fast EMA'].iloc[-2] - df['Slow EMA'].iloc[-2]) / df['Slow EMA'].iloc[-2] * 100

    #Watch-list & Portfolio-list
    if yesterday_EMA_diff_pct < 0 and current_EMA_diff_pct > 0 and data_type == "100d":
        alarm_message = "Potential buy {} \nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n#Cross-EMA ".format(symbol,dEMA_pct, RSI14, dMom10_pct, dVWMAvSMA20)
        alarm_symbol = ">&#9650;"
        alarm_symbol_color = "green"
        alarms["101"] = {
            "value": 1,
            "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'>{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
        }

    if df['LowP'].iloc[-1] < df['LowP'].iloc[-2] and df['LowP'].iloc[-2] > df['LowP'].iloc[-3] and data_type == "100d":
        alarm_message = "Potential buy {} \nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n#100-Minimum ".format(symbol,dEMA_pct, RSI14, dMom10_pct, dVWMAvSMA20)
        alarm_symbol = ">&#9679;"
        alarm_symbol_color = "green"
        alarms["122"] = {
            "value": 1,
            "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'>{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
        }
        
    #Portfolio-list
    if not watch_list: 
        
        alarm_message_add = "Amount >1 year: {} ({} %)".format(round(amount_older_than_one_year,2),round(amount_older_than_one_year_pct,2))

        if yesterday_EMA_diff_pct > 0 and current_EMA_diff_pct < 0 and data_type == "100d":
            alarm_message = "Potential sell {} \nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n{} \n#Cross-EMA ".format(symbol,dEMA_pct, RSI14, dMom10_pct, dVWMAvSMA20,alarm_message_add)
            alarm_symbol = ">&#9660;"
            alarm_symbol_color = "red"
            alarms["201"] = {
                "value": 1,
                "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
            }     

       # if current_profit_pct < one_day_profit_limit and data_type == "100d":
       #     alarm_message = "Potential sell {} \nProfit: {} %\nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n{} \n#Under buy rate ".format(symbol,round(current_profit_pct,2),dEMA_pct, RSI14, dMom10_pct, dVWMA20_pct,alarm_message_add)
       #     alarm_symbol = ">&#9679;"
       #     alarm_symbol_color = "red"
       #     alarms["202"] = {
       #         "value": current_profit_pct ,
       #         "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
       #     }  
        
       # if current_one_day_price_change_pct < (one_day_price_change * -1) and data_type == "100d":
       #     alarm_message = "Potential sell {} \nPrice': {} % \nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n{} \n#Negative-1 ".format(symbol,round(current_one_day_price_change_pct,2),dEMA_pct, RSI14, dMom10_pct, dVWMA20_pct,alarm_message_add)
       #     alarm_symbol = ">&#9679;"
       #     alarm_symbol_color = "black"
       #     alarms["211"] = {
       #         "value": current_one_day_price_change_pct,
       #         "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
       #     }    
        
        if current_seven_days_slope_pct < (seven_day_price_change * -1):
            alarm_message = "Potential sell {} \nPrice7': {} % \nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n{} \n#Negative-7 ".format(symbol,round(current_seven_days_slope_pct,2),dEMA_pct, RSI14, dMom10_pct, dVWMAvSMA20,alarm_message_add)
            #alarm_message2 = ", ".join(df['Price'].tail(7).round(1).astype(str))
            #alarm_message =  alarm_message1 + '\n' + alarm_message2 #Only for test
            alarm_symbol = ">&#9679;"
            alarm_symbol_color = "black"
            alarms["221"] = {
                "value": current_seven_days_slope_pct,
                "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
            }  

        if df['HighP'].iloc[-1] < df['HighP'].iloc[-2] and df['HighP'].iloc[-2] > df['HighP'].iloc[-3] and data_type == "100d":
            alarm_message = "Potential sell {} \nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n{} \n#100-Maximum ".format(symbol,dEMA_pct, RSI14, dMom10_pct, dVWMAvSMA20,alarm_message_add)
            alarm_symbol = ">&#9679;"
            alarm_symbol_color = "red"
            alarms["222"] = {
                "value": 1,
                "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
            }

        #if current_one_day_price_change_pct > one_day_price_change and data_type == "100d":
        #    alarm_message = "Potential buy {} \nPrice': {} % \nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n#Positive-1 ".format(symbol,round(current_one_day_price_change_pct,2),dEMA_pct, RSI14, dMom10_pct, dVWMA20_pct)
        #    alarm_symbol = ">&#9679;"
        #    alarm_symbol_color = "black"
        #    alarms["111"] = {
        #        "value": current_one_day_price_change_pct,
        #        "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
        #    }

        if current_seven_days_slope_pct > seven_day_price_change:
            alarm_message = "Potential buy {} \nPrice7': {} % \nEMA': {} \nRSI14: {} \nMOM10': {} \nPrice-Volume-push: {} \n#Positiv-7 ".format(symbol,round(current_seven_days_slope_pct,2),dEMA_pct, RSI14, dMom10_pct, dVWMAvSMA20)
            #alarm_message2 = ", ".join(df['Price'].tail(7).round(1).astype(str))
            #alarm_message =  alarm_message1 + '\n' + alarm_message2 #Only for test
            alarm_symbol = ">&#9679;"
            alarm_symbol_color = "black"
            alarms["121"] = {
                "value": current_seven_days_slope_pct,
                "msg": f"<html><body><span style='color: {alarm_symbol_color}; font-size: 24px;'{alarm_symbol}</span><span style='color: black;'>{alarm_message}</span><a href='{link}'>#Technical analysis</a></body></html>"
            }


    return alarms, signal_count

def older_than_one_year(df):
    current_date = datetime.now()
    one_year_ago = current_date - timedelta(days=365)
    filtered_df = df[df['Date'] < one_year_ago]
    total_amount = filtered_df['Amount'].sum()
    
    return total_amount