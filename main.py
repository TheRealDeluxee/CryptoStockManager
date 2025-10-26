import configparser
import classes as cl
import pandas as pd
import plotly.figure_factory as ff
import os
import time
import func
import schedule
import sys
import subprocess

morning_analysis =  "09:30"
noon_analysis = "12:30"
midnight_config = "00:30"
evening_summary = "18:30"

class CryptoStockManager:
    def __init__(self, config_file_path, output_dir='plots'):
        # Initialize main attributes for each crypto or stock item
        self.config_file_path = config_file_path
        self.test_mode = 0
        self.output_dir = output_dir
        self.df_crypto_hd = pd.DataFrame(columns=['Name', 'Piece [€]', '7d P&L [%]', '1d P&L [%]', 'Rating'])
        self.df_stock_hd = pd.DataFrame(columns=['Name', 'Piece [€]', '7d P&L [%]', '1d P&L [%]', 'Rating'])
        self.df_crypto_hh = pd.DataFrame(columns=['Name', 'Piece [€]', '7d P&L [%]', '1d P&L [%]', 'Rating'])
        self.df_stock_hh = pd.DataFrame(columns=['Name', 'Piece [€]', '7d P&L [%]', '1d P&L [%]', 'Rating'])
        self.version = "0.0.0"
        self.crypto_items_hh = []
        self.crypto_items_hd = []
        self.stock_items_hh = []
        self.stock_items_hd = []
        self.stock_update = False
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        config = configparser.ConfigParser()
        if not config.read(self.config_file_path):
            raise FileNotFoundError(f"The INI file '{self.config_file_path}' could not be found or loaded.")

        try:
            self.version = config.get('version', 'number')
            func.token_pushover = config.get('pushover', 'token')
            func.user_pushover = config.get('pushover', 'user')
            self.test_mode = int(config.get('alarm', 'test_mode'))
            func.one_day_price_change = int(config.get('alarm', 'one_day_price_change'))
            func.seven_day_price_change = int(config.get('alarm', 'seven_day_price_change'))
            func.one_day_profit_limit = int(config.get('alarm', 'one_day_profit_limit'))
            func.previous_alarm_change = int(config.get('alarm', 'previous_alarm_change'))
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            func.log_print(f"Error in the configuration: {e}")

        if "crypto" in config.sections():
            crypto_config_items = config.items("crypto")

            for crypto, transactions in crypto_config_items:
                try:
                    name, transaction_data = transactions.split(";", 1)
                    crypto_objhh = cl.CryptoStock(crypto, name.strip(), 'crypto')
                    crypto_objhd = cl.CryptoStock(crypto, name.strip(), 'crypto')
                    self.crypto_items_hh.append(crypto_objhh)
                    self.crypto_items_hd.append(crypto_objhd)
                
                    for transaction in transaction_data.split(";"):
                        try:
                            spend_str, quantity_str, date_str = transaction.strip().split(",")
                            spend = float(spend_str)
                            quantity = float(quantity_str)
                            date = date_str.strip()
                            crypto_objhh.buy(spend, quantity, date)
                            crypto_objhd.buy(spend, quantity, date)
                        except ValueError:
                            func.log_print(f"Error when parsing spend or quantity for {name}")
                except ValueError:
                    func.log_print(f"Error when parsing spend or quantity for {crypto}")

        else:
            func.log_print("Error: Section 'crypto' is missing in the INI file.")

        if "stocks" in config.sections():
            stock_config_items = config.items("stocks")

            for stock, transactions in stock_config_items:
                try:
                    name, transaction_data = transactions.split(";", 1)
                    stock_objhh = cl.CryptoStock(stock, name.strip(), 'stock')
                    stock_objhd = cl.CryptoStock(stock, name.strip(), 'stock')
                    self.stock_items_hh.append(stock_objhh)
                    self.stock_items_hd.append(stock_objhd)

                    for transaction in transaction_data.split(";"):
                        try:
                            spend_str, quantity_str, date_str = transaction.strip().split(",")
                            spend = float(spend_str)
                            quantity = float(quantity_str)
                            date = date_str.strip()
                            stock_objhh.buy(spend, quantity, date)
                            stock_objhd.buy(spend, quantity, date)
                        except ValueError:
                            func.log_print(f"Error when parsing spend or quantity for {name}")
                except ValueError:
                    func.log_print(f"Error when parsing spend or quantity for {stock}")
        else:
            func.log_print("Error: Section 'stocks' is missing in the INI file.")
        
    def restart_program(self):
        python = sys.executable
        return_code = subprocess.call([python] + sys.argv)
        if return_code != 0:
            func.log_print(f"Warning: Restart failed with return code {return_code}")

    def schedule_on_weekdays(self):
        now = time.localtime()
        current_day = time.strftime("%A", now)
        current_hour = now.tm_hour
        self.stock_update = False

        if current_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"] or self.test_mode == 1: #Test_mode
            if (current_hour >= 8 and current_hour < 22) or self.test_mode == 1: #Test_mode
                self.stock_update = True

    def hundred_day_analysis(self):
        now = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", now)

        func.log_print(f"Hundred day analysis started")

        self.df_crypto_hd = self.df_crypto_hd[0:0]
        self.df_stock_hd = self.df_stock_hd[0:0]

        self.schedule_on_weekdays()
        if self.stock_update:
            for item in self.stock_items_hd:
                self.df_stock_hd.loc[len(self.df_stock_hd)] = item.process_data(item.get_API_data('100d'), '100d')

        for item in self.crypto_items_hd:
            self.df_crypto_hd.loc[len(self.df_crypto_hd)] = item.process_data(item.get_API_data('100d'), '100d')

    def hundred_hour_analysis(self): 
        now = time.localtime()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", now)

        func.log_print(f"Hundred hour analysis started")

        self.df_crypto_hh = self.df_crypto_hh[0:0]
        self.df_stock_hh = self.df_stock_hh[0:0]

        self.schedule_on_weekdays()
        #if self.stock_update:
        #    for item in self.stock_items_hh:
        #        df_stock_hh = item.get_API_data('100h')
        #        self.df_stock_hh.loc[len(self.df_stock_hh)] = item.process_data(df_stock_hh)

        for item in self.crypto_items_hh:
            self.df_crypto_hh.loc[len(self.df_crypto_hh)] = item.process_data(item.get_API_data('100h'), '100h')

    def send_summary(self):
        now = time.localtime()

        try:
            # Sort and prepare the tables
            self.df_crypto_hd = self.df_crypto_hd.sort_values(by=['Rating'], ascending=False)
            self.df_stock_hd = self.df_stock_hd.sort_values(by=['Rating'], ascending=False)

            func.log_print(f"\nLast refresh {time.asctime(now)}")
            func.log_print(self.df_crypto_hd.round(2))

            if self.stock_update and not self.df_stock_hd.empty:
                func.log_print('\n')
                func.log_print(self.df_stock_hd.round(2))

            # Create and send summaries using matplotlib
            try:
                output_path = os.path.join(self.output_dir, 'summary_crypto.png')
                if not self.df_crypto_hd.empty:
                    func.create_table_image(self.df_crypto_hd.round(2), output_path)
                    func.pushover_image('summary_crypto', 'Daily summary crypto')
            except Exception as e:
                func.log_print(f"Error while processing crypto summary: {e}")

            if self.stock_update:
                try:
                    output_path = os.path.join(self.output_dir, 'summary_stock.png')
                    if not self.df_stock_hd.empty:
                        func.create_table_image(self.df_stock_hd.round(2), output_path)
                        func.pushover_image('summary_stock', 'Daily summary stock')
                except Exception as e:
                    func.log_print(f"Error while processing stock summary: {e}")

        except Exception as e:
            func.log_print(f"Unexpected error in send_summary: {e}")

    def backtest(self):
        test_obj = cl.CryptoStock("test", "test", 'crypto')
        test_obj.buy(1000, 0.5, "01/01/2022")

        test_data = func.load_csv("test.csv", index=False, date_cols=['Date'])
        test_obj.process_data(test_data, '100d')

# Initialize the manager
manager = CryptoStockManager(config_file_path="config.ini")
func.pushover(f"Initialization complete - Version {manager.version}")
start_time = time.time()  
now = time.localtime()
current_hour = now.tm_hour
current_minute = now.tm_min
noon_hour, noon_minute = map(int, noon_analysis.split(":"))

if manager.test_mode == 1: #Test_mode in config

    #manager.backtest()
    manager.hundred_hour_analysis() 
    manager.hundred_day_analysis() 
    manager.send_summary()
else:
    
    if current_hour > noon_hour or (current_hour == noon_hour and current_minute >= noon_minute):
        manager.hundred_day_analysis() 

    # Schedule tasks
    schedule.every().hour.do(manager.hundred_hour_analysis)          
    schedule.every().day.at(morning_analysis).do(manager.hundred_day_analysis)
    schedule.every().day.at(noon_analysis).do(manager.hundred_day_analysis)

    #schedule.every().day.at(midnight_config).do(lambda: manager.load_config()) # Reload config at midnight
    schedule.every().day.at(evening_summary).do(manager.send_summary) # Daily summary

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except KeyboardInterrupt:
        break
    except configparser.NoSectionError as e:
        func.pushover(f"Configuration error: {e}")
        break
    except configparser.NoOptionError as e:
        func.pushover(f"Missing option in the configuration: {e}")
        break
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        infos = {'hours': int(hours), 'minutes': int(minutes), 'seconds': int(seconds), 'exception': e}
        msg = "Script stopped because of an exception \nRuntime: %(hours)s h %(minutes)s min %(seconds)s sec \n\n%(exception)s" % infos
        func.pushover(msg)
        func.log_print(e)
        time.sleep(600)  # Waiting time of 10 minutes before restarting
        manager.restart_program()

