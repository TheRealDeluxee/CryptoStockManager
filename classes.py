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
    watch_list: bool = False
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
        # Record a purchase transaction
        self.state.buy_balance_eur += fiat_eur
        self.state.amount_crypto_stock += amount
    
        if  fiat_eur  != 1 and amount != 1 and date != 1: 
            parsed_date = datetime.strptime(date.strip(), "%d/%m/%Y")
            new_row = [parsed_date, fiat_eur, amount]
            # append to state-managed buy history
            self.state.df_buy_history.loc[len(self.state.df_buy_history)] = new_row

    def reset_alarm_duplicates(self):
        self.state.seven_day_alarms.clear() 

    def activate_test_mode(self):
        self.test_mode = True

    def get_API_data(self, data_type):

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

        if df is None or df.empty:
            func.log_print(f"process_data: No data available for {self.symbol}")
            return {
                'Name': self.search,
                'Piece [€]': 0.0,
                '7d P&L [%]': 0.0,
                '1d P&L [%]': 0.0,
                'Rating': 0
            }

        link = "https://de.tradingview.com/symbols/"

        df = func.calc_indicator_fuctions(df)
        self.state = func.calc_state_fuctions(self.state, df)
        
        func.plot_and_save(df,self.symbol, data_type, self.state.zero_line)
        
        alarms, self.state.tech_indicators, self.state.score = func.alarm(df,self.search,self.state.watch_list, self.state.current_profit_pct, self.state.amount_older_than_one_year, self.state.amount_older_than_one_year_pct, data_type)

        alarms_filter = func.filter(alarms, self.state)
        for key, info in alarms_filter.items():
            func.pushover_image(self.symbol, info['msg'], info['priority'])

        # Watch-list items
        if self.state.watch_list:
            return {
                'Name': self.search,
                'Piece [€]': round(df.tail(1)['Price'].values[0], 2),
                '7d P&L [%]': round(self.state.current_seven_days_slope_pct, 2),
                '1d P&L [%]': round(self.state.current_one_day_price_change_pct, 2),
                'Rating': self.state.score
            }
        else:
            return {
                'Name': self.search,
                'Piece [€]': round(df.tail(1)['Price'].values[0], 2),
                '7d P&L [%]': round(self.state.current_seven_days_slope_pct, 2),
                '1d P&L [%]': round(self.state.current_one_day_price_change_pct, 2),
                'Rating': self.state.score
            }
    
