
# Crypto and Stock Alert - CaSa 

## Overview

This project provides a Python-based system for analyzing and monitoring cryptocurrencies and stocks. It uses indicators like Moving Averages, RSI, and EMA to evaluate trends and alert users on significant market conditions. The system can track and visualize data, calculate profits, and send notifications using Pushover.

## Why do I need CaSa?

I’ve always wanted to invest in stocks and cryptocurrencies, but I was held back by the fear of missing rapid market changes and, especially with crypto, the risk of losing significant money. Constantly monitoring prices felt overwhelming and stressful.

Moreover, I was never fully satisfied with the existing apps and their features for tracking price changes. Most of them offer only basic alert thresholds, while more advanced functionalities often come with a price tag.

That’s why I decided to take matters into my own hands and developed CaSa a free, open-source program now available on GitHub. CaSa is reliable and already includes several essential features, making it an excellent starting point for anyone facing the same challenges I did.

While it’s functional and stable, there’s still plenty of room for growth. Check out the "New Feature Idea List" for planned improvements and let’s build something great together!

![Project Overview](alert_watch_phone.png)

## Features

- **Automatic Data Retrieval**: Fetches crypto data from CoinGecko and stock data from Yahoo Finance.
- **Indicator Calculations**: Calculates indicators such as Fast EMA, Slow EMA, RSI, and VWMA for analysis.
- **Alarms**: Sends buy/sell signals based on custom thresholds and market trends.
- **Scheduling**: Configurable scheduling for running hourly and daily analyses.
- **Portfolio-list**: Tracks trends and potential buy and sell signals based on configured indicators.
- **Watch-List**: Tracks trends and potential buy signals based on configured indicators.

## Project Structure

- `classes.py`: Contains the `crypto_stock` class, which manages individual crypto or stock entities, stores transaction, and alarm data.
- `func.py`: Utility functions for data retrieval, indicator calculations, alarm handling, and image generation.
- `main.py`: The main script that orchestrates the data analysis process, schedules tasks, and manages notifications.
- `config.ini`: Contains a crypto and stock list, alert limits, and the API key. This file is not included in the GitHub structure and must be created manually with the configuration details below.

## Installation

1. Install Python 3.
2. Install Dependencies: `pip3 install pandas numpy yfinance scipy matplotlib plotly schedule requests pytrends configparser http.client urllib3`
3. Create `config.ini` in the root directory (see **Configuration File** section).
4. Run the script: `python3 main.py`

**Note**: This project is tested on a RockPi3 with Armbian OS, initiated over SSH and using a mapped folder for editing the `config.ini`.

## Get a Pushover Message

- **Status message**: Provides information about the program start or errors.
- **Potential buy/sell alert**: Alerts for review based on indicators; decision-making is up to the user.
  - **Price'**: 7-day price slope for context.
  - **EMA'**: Slope of the Exponential Moving Average indicator.
  - **RSI14**: 14-day Relative Strength Index, indicating if an asset is overbought or oversold.
  - **MOM10'**: 10-day Momentum indicator slope, reflecting price change speed.
  - **VWMA20'**: 20-day Volume Weighted Moving Average slope, adjusting price trends by trading volume.
  - **Amount >1 year**: Asset quantity held for over a year, relevant for tax free crypto transactions (like in Germany, Portugal, Belgium and Luxembourg).
  - **# [Alert type]**: Specifies alert type, like “#Cross-EMA” for an EMA crossover or “#100-Minimum” for a 100-day low.
  - **# Technical analysis**: Provides additional references or links for deeper insights.
- **Daily summary**: Summary of all crypto and stock items, sorted by rating (indicator count suggesting a buy).

![Project Overview](example_msg.png)

## Configuration File (config.ini)

The `config.ini` file should contain sections like `pushover`, `alarm`, `crypto`, and `stocks`. Here’s an example configuration:

```ini
[pushover]
token = YOUR_PUSHOVER_TOKEN
user = YOUR_PUSHOVER_USER_KEY

[alarm]
one_day_price_change = 10       # Threshold for a one-day price change alarm (%).
seven_day_price_change = 7      # Threshold for a seven-day price change alarm (%).
previous_alarm_change = 3       # Minimum change required between consecutive alarms (%).
one_day_profit_limit = -2       # Profitability threshold for a one-day period (%). #currently inactive
test_mode = 0                   # Test mode

[crypto]
bitcoin = Bitcoin; 1000,0.05,01/01/2022; 1000,0.05,2/01/2022   # * 2 x Portfolio buy: [API ID (https://www.coingecko.com/)] = [Display name], [Investment in €], [Quantity], [Buy date in DD/MM/YYYY]
matic-network = Polygon; 1 , 1, 1                              # Watch list

[stocks]
APC.DE = Apple; 1000,10,01/01/2022       # 1 x Portfolio buy: [EURO-Symbol (https://finance.yahoo.com/)] = [Display name], [Investment in €], [Quantity], [Buy date in DD/MM/YYYY]
LHA.DE = Lufthansa; 1, 1, 1            # Watch list 
```

**Note**: Use only stock symbols with the Euro currently (e.g. APC.DE and AMZ.DE).

## Dependencies

- **Python Libraries**: `pandas`, `numpy`, `yfinance`, `scipy`, `matplotlib`, `plotly`, `schedule`, `requests`, `pytrends`, `configparser`, `http.client`, `urllib3`
- **External APIs**: Requires access to Pushover (free account needed), CoinGecko, and Yahoo Finance APIs.

## License

This project is licensed under the MIT License.

## Links

- [Pushover](https://pushover.net/) – Notification service for sending alerts directly to your devices. Available on Apple App Store and Google Play Store.
- [CoinGecko](https://www.coingecko.com) – A comprehensive cryptocurrency market tracker used for data retrieval.
- [Yahoo Finance](https://finance.yahoo.com) – Provides stock market data, trends, and financial news.
- [Rock Pi](https://rockpi.org/) – A robust single-board computer for running this project efficiently.
- [Armbian OS](https://www.armbian.com/) – A lightweight operating system for ARM-based devices, perfect for deploying CaSa.