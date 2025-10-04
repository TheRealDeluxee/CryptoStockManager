import func
import pandas as pd
from datetime import datetime

class crypto_stock:

    def __init__(self, symbol, search, crypto_or_stock, previous_alarm_change):
        self.symbol = symbol
        self.search = search
        self.type = crypto_or_stock
        self.params_crypto_stock = ''

        self.buy_balance_eur = 0

        self.current_balance_eur = 0
        self.current_profit_pct = 0
        self.current_profit = 0
        self.current_price = 0
        self.current_one_day_price_change_pct = 0
        self.current_seven_days_slope_pct = 0

        self.yesterday_balance_eur = 0
        self.yesterday_profit_pct = 0
        self.yesterday_profit = 0
        self.yesterday_price = 0
        self.yesterday_one_day_price_change_pct = 0
        self.yesterday_seven_days_slope_pct = 0
        self.yesterday_EMA_diff_pct = 0

        self.price_change_to_last_full_hour = 0
        self.sell_money = 0
        self.amount_crypto_stock = 0
        self.data_type = ''
        self.score = 0
        self.tech_indicators = {}

        self.test_mode = False
        self.current_EMA_diff_pct = 0
        self.EMA_now_diff = 0
        
        self.EMA_last_full_hour_diff = 0
        self.fgi = ''
        self.promt = ''
        self.std7 = 0
        self.google_trends = 0
        self.google_trends_yesterday = 0
        self.seven_day_alarms = []
        self.watch_list = False
        self.alarm_prev = {}
        self.previous_alarm_change = previous_alarm_change

        self.df_buy_history = pd.DataFrame(columns=['Date', 'Cost [€]', 'Amount'])
        self.amount_older_than_one_year = 0

    def buy(self, fiat_eur, amount, date):
        # Record a purchase transaction
        self.buy_balance_eur += fiat_eur
        self.amount_crypto_stock += amount
    
        if  fiat_eur  != 1 and amount != 1 and date != 1: 
            parsed_date = datetime.strptime(date.strip(), "%d/%m/%Y")
            new_row = [parsed_date, fiat_eur, amount]
            self.df_buy_history.loc[len(self.df_buy_history)] = new_row

    def reset_alarm_duplicates(self):
        self.seven_day_alarms.clear() 

    def activate_test_mode(self):
        self.test_mode = True

    def refresh(self, data_type):
        # Update the data based on the data_type ('100d' or '100h')
        # and calculate updated metrics like profit and EMA
        self.data_type = data_type

        if self.type == 'crypto':
            if self.data_type == '100d':
                self.params_crypto_stock = {'vs_currency': 'eur', 'days': '100'}   
            elif self.data_type == '100h':
                self.params_crypto_stock = {'vs_currency': 'eur', 'days': '4.17'}
            df = func.calc_indicator_fuctions(func.get_crypto(self.symbol,self.params_crypto_stock))
        elif self.type == 'stock':   
            if self.data_type == '100d':
                self.params_crypto_stock = "6mo"
            elif self.data_type == '100h':
                self.params_crypto_stock = "1mo"
            df = func.calc_indicator_fuctions(func.get_stock(self.symbol,self.params_crypto_stock))


        link = "https://de.tradingview.com/symbols/"

        if self.buy_balance_eur  == 1 and self.amount_crypto_stock == 1 : 
            self.watch_list = True
            func.plot_and_save(df,self.symbol, self.data_type)
            self.amount_older_than_one_year = func.older_than_one_year(self.df_buy_history)
            self.amount_older_than_one_year_pct = self.amount_older_than_one_year / self.amount_crypto_stock * 100
        else:
            func.plot_and_save(df,self.symbol, self.data_type, self.buy_balance_eur/self.amount_crypto_stock)

        self.current_balance_eur = self.amount_crypto_stock * df.tail(1)['Price'].values[0]
        self.yesterday_balance_eur = self.amount_crypto_stock * df.tail(2)['Price'].values[0]

        self.current_profit = self.current_balance_eur - self.buy_balance_eur
        self.yesterday_profit = self.yesterday_balance_eur - self.buy_balance_eur

        self.current_profit_pct = (self.current_balance_eur - self.buy_balance_eur)/self.buy_balance_eur*100
        self.yesterday_profit_pct = (self.yesterday_balance_eur - self.buy_balance_eur )/self.buy_balance_eur*100

        self.current_one_day_price_change_pct = (df['Price'].iloc[-1] - df['Price'].iloc[-2]) / df['Price'].iloc[-2] * 100
        self.yesterday_one_day_price_change_pct = (df['Price'].iloc[-2] - df['Price'].iloc[-3]) / df['Price'].iloc[-3] * 100
        
        self.current_EMA_diff_pct = (df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-1]) / df['Slow EMA'].iloc[-1] * 100
        self.yesterday_EMA_diff_pct = (df['Fast EMA'].iloc[-2] - df['Slow EMA'].iloc[-2]) / df['Slow EMA'].iloc[-2] * 100

        self.current_seven_days_slope_pct = func.seven_day_slope_pct(df,True)
        self.yesterday_seven_days_slope_pct = func.seven_day_slope_pct(df,False)

        self.amount_older_than_one_year = func.older_than_one_year(self.df_buy_history)
        self.amount_older_than_one_year_pct = self.amount_older_than_one_year / self.amount_crypto_stock * 100
        
        alarms, self.tech_indicators, self.score = func.alarm(df,self.search,self.watch_list, self.current_profit_pct, self.amount_older_than_one_year, self.amount_older_than_one_year_pct, link, self.data_type)

        alarms_filter = self.filter(alarms)
        for key, info in alarms_filter.items():
            func.pushover_image(self.symbol, info['msg'])

        return {
            'Name': self.search,
            'RSI14': self.tech_indicators.get('RSI14', None),
            'MOM10': self.tech_indicators.get('MOM10', None),
            'VMOM10': self.tech_indicators.get('VMOM10', None),
            'SMA7': self.tech_indicators.get('SMA7', None),
            'VMA7': self.tech_indicators.get('VMA7', None),
            'Rating': self.score
        }


    def filter(self,alarm_neu):
        change = {}
        for key, new_info in alarm_neu.items():
            new_value = new_info["value"]
            if key in self.alarm_prev:
                prev_value = self.alarm_prev[key]["value"]
                if abs(prev_value - new_value) > self.previous_alarm_change:
                    self.alarm_prev[key] = new_info 
                    change[key] = new_info 
            else:
                self.alarm_prev[key] = new_info
                change[key] = new_info
        return change
    

        
