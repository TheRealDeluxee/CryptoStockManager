"""CaSa: Konfiguration laden, Analysen planen und Telegram-Befehle bearbeiten.

Start: python main.py
Die eigentlichen Berechnungen bleiben in classes.py und func.py.
"""
import configparser
import copy
import math
import multiprocessing as mp
import os
import time
import uuid
from queue import Empty

import pandas as pd
import schedule
import classes as cl
import func
import telegram_bot

MORNING_ANALYSIS = "09:30"
NOON_ANALYSIS = "12:30"
EVENING_SUMMARY = "18:30"


class CryptoStockManager:
    def __init__(self, config_file_path="config.ini", output_dir="plots"):
        self.output_dir = output_dir
        func.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        config = configparser.ConfigParser()
        if not config.read(config_file_path, encoding="utf-8-sig"):
            raise FileNotFoundError(f"{config_file_path} wurde nicht gefunden.")
        self.version = config.get('version', 'number')
        self.test_mode = config.getint('alarm', 'test_mode')
        func.one_day_price_change = config.getfloat('alarm', 'one_day_price_change')
        func.seven_day_price_change = config.getfloat('alarm', 'seven_day_price_change')
        func.one_day_profit_limit = config.getfloat('alarm', 'one_day_profit_limit')
        func.previous_alarm_change = config.getfloat('alarm', 'previous_alarm_change')
        limits = (func.one_day_price_change, func.seven_day_price_change,
                  func.one_day_profit_limit, func.previous_alarm_change)
        if not all(math.isfinite(value) for value in limits):
            raise ValueError('Alarmwerte müssen endliche Zahlen sein.')
        if self.test_mode not in (0, 1):
            raise ValueError('test_mode muss 0 oder 1 sein.')
        if func.seven_day_price_change < 0 or func.previous_alarm_change < 0:
            raise ValueError('Alarm-Abstände dürfen nicht negativ sein.')
        self.stock_update = False
        self.crypto_items_hh = []
        self.crypto_items_hd = []
        self.stock_items_hh = []
        self.stock_items_hd = []
        cols = ['Name', 'Piece [€]', '7d P&L [%]', '1d P&L [%]', 'Rating']
        self.df_crypto_hd = pd.DataFrame(columns=cols)
        self.df_stock_hd = pd.DataFrame(columns=cols)
        self.df_crypto_hh = pd.DataFrame(columns=cols)
        self.df_stock_hh = pd.DataFrame(columns=cols)
        self.last_daily_update = None

    def load_config_items(self):
        """Neue Listen erst vollständig aufbauen, dann die alten ersetzen."""
        entries = cl.read_watchlist()
        rows = cl.read_transactions()
        crypto_hh, crypto_hd, stock_hh, stock_hd = [], [], [], []
        old_hh = {(item.type, item.symbol.lower()): item
                  for item in self.crypto_items_hh + self.stock_items_hh}
        old_hd = {(item.type, item.symbol.lower()): item
                  for item in self.crypto_items_hd + self.stock_items_hd}
        for (asset_type, symbol), name in entries.items():
            symbol = symbol.upper() if asset_type == 'stock' else symbol
            item_hh = cl.apply_transactions(cl.CryptoStock(symbol, name, asset_type), rows)
            item_hd = copy.deepcopy(item_hh)
            key = (asset_type, symbol.lower())
            # Neuladen nach /buy darf bekannte Alarme nicht erneut versenden.
            if key in old_hh:
                item_hh.state.alarm_prev = copy.deepcopy(old_hh[key].state.alarm_prev)
            if key in old_hd:
                item_hd.state.alarm_prev = copy.deepcopy(old_hd[key].state.alarm_prev)
            if asset_type == 'crypto':
                crypto_hh.append(item_hh)
                crypto_hd.append(item_hd)
            else:
                stock_hh.append(item_hh)
                stock_hd.append(item_hd)
        for row in rows:
            if (row['type'], row['symbol']) not in entries:
                func.log_print(f"Inaktive Transaktion: {row['symbol']} fehlt in der Watchlist und wird nicht analysiert.")
        self.crypto_items_hh, self.crypto_items_hd = crypto_hh, crypto_hd
        self.stock_items_hh, self.stock_items_hd = stock_hh, stock_hd
        # Tagestabellen bleiben erhalten: Kurs, P&L und Rating hängen nicht vom Bestand ab.
        func.log_print(f"Geladen: {len(entries)} Symbole, {len(rows)} Transaktionen.")

    def _is_stock_trading_hours(self):
        now = time.localtime()
        # Zahlen funktionieren unabhängig von der Windows-Sprache.
        self.stock_update = self.test_mode == 1 or (now.tm_wday < 5 and 8 <= now.tm_hour < 22)

    def _analyse_items(self, items, data_type):
        results = []
        for item in items:
            try:
                df = item.get_API_data(data_type)
                alarms = item.process_data(df, data_type)
                item.plot_send(df, data_type, alarms)
                results.append(item.get_summary(df))
            except Exception as error:
                # Eine fehlgeschlagene Aktie darf die anderen nicht blockieren.
                func.pushover(f"Analyse für {item.symbol} ({data_type}) fehlgeschlagen: {error}")
        return results

    def hundred_day_analysis(self):
        func.log_print('100-Tage-Analyse gestartet')
        self._is_stock_trading_hours()
        cols = list(self.df_crypto_hd.columns)
        self.df_crypto_hd = pd.DataFrame(self._analyse_items(self.crypto_items_hd, '100d'), columns=cols)
        self.df_stock_hd = pd.DataFrame(columns=cols)
        if self.stock_update:
            self.df_stock_hd = pd.DataFrame(self._analyse_items(self.stock_items_hd, '100d'), columns=cols)
        self.last_daily_update = time.strftime('%d.%m.%Y %H:%M:%S')

    def hundred_hour_analysis(self):
        func.log_print('100-Stunden-Analyse gestartet')
        # Wie bisher: automatische Stundenanalyse nur für Kryptowährungen.
        cols = list(self.df_crypto_hh.columns)
        self.df_crypto_hh = pd.DataFrame(self._analyse_items(self.crypto_items_hh, '100h'), columns=cols)

    def single_daily_analysis(self, symbol, asset_type):
        entries = cl.read_watchlist()
        key = (asset_type, symbol.lower())
        if key not in entries:
            raise ValueError(f"{symbol} ist nicht in der Watchlist ({asset_type}).")
        name = entries[key]
        symbol = symbol.upper() if asset_type == 'stock' else symbol.lower()
        # Eigenes Objekt: /check verändert weder Analyse- noch Alarmzustände.
        item = cl.apply_transactions(cl.CryptoStock(symbol, name, asset_type), cl.read_transactions())
        df = item.get_API_data('100d')
        df = func.calc_indicator_fuctions(df)
        item.state = func.calc_state_fuctions(item.state, df)
        details = func.analysis_details(df)
        message = f"📊 {name} ({symbol}) — Tagesanalyse (100d)\nManueller Abruf ohne Alarm\n" + details['message']
        if not item.state.watch_list:
            message += (f"\n📅 Amount >1 year: {item.state.amount_older_than_one_year:.2f} "
                        f"({item.state.amount_older_than_one_year_pct:.2f} %)")
        image_name = 'manual_' + uuid.uuid4().hex
        func.plot_and_save(df, image_name, '100d', item.state.zero_line)
        func.pushover_image(image_name, message, delete_after_send=True)

    def send_summary(self):
        if self.df_crypto_hd.empty and self.df_stock_hd.empty:
            func.pushover('Noch keine Tagesergebnisse vorhanden. Bitte zuerst /analyse daily ausführen.')
            return
        for kind, frame in (('crypto', self.df_crypto_hd), ('stock', self.df_stock_hd)):
            if frame.empty:
                continue
            frame = frame.sort_values('1d P&L [%]', ascending=False)
            image_name = f'summary_{kind}_' + uuid.uuid4().hex
            func.create_table_image(frame, os.path.join(self.output_dir, image_name + '.png'))
            message = f'Tagesübersicht {kind} — Datenstand: {self.last_daily_update}'
            func.pushover_image(image_name, message, delete_after_send=True)

    def handle_command(self, command):
        """True bedeutet: Hauptschleife soll einen sauberen Neustart ausführen."""
        action = command.get('cmd')
        if action == 'analyse':
            mode = command.get('mode', 'hourly')
            if mode == 'hourly':
                self.hundred_hour_analysis()
            elif mode == 'daily':
                self.hundred_day_analysis()
            elif mode == 'summary':
                self.send_summary()
            else:
                raise ValueError(f'Unbekannte Analyse: {mode}')
        elif action == 'check':
            self.single_daily_analysis(command['symbol'], command['type'])
        elif action == 'reload_transactions':
            self.load_config_items()
            func.pushover('Transaktionen wurden im Hauptprogramm neu geladen.')
        elif action == 'restart_main':
            return True
        else:
            raise ValueError(f'Unbekannter Befehl: {action}')
        return False


def run_program():
    # Relative Pfade sollen auch beim Start aus einem anderen Ordner stimmen.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    old_manager = None
    while True:
        bot_process = None
        manager = None
        restart = False
        notify_q, command_q = mp.Queue(), mp.Queue()
        func.set_message_queue(notify_q)
        try:
            manager = CryptoStockManager()
            # Neustart per /restart darf bekannte Alarme nicht erneut versenden.
            if old_manager is not None:
                manager.crypto_items_hh, manager.crypto_items_hd = old_manager.crypto_items_hh, old_manager.crypto_items_hd
                manager.stock_items_hh, manager.stock_items_hd = old_manager.stock_items_hh, old_manager.stock_items_hd
            manager.load_config_items()
            telegram_bot._load_config()  # Fehlende Bot-Konfiguration vor dem Start erkennen.
            bot_process = mp.Process(target=telegram_bot.run_standalone,
                                     args=(notify_q, command_q), daemon=True)
            bot_process.start()
            func.pushover(f'Initialisierung abgeschlossen — Version {manager.version}')
            schedule.clear()
            if manager.test_mode == 1:
                manager.hundred_hour_analysis()
                manager.hundred_day_analysis()
                manager.send_summary()
            else:
                schedule.every().hour.do(manager.hundred_hour_analysis)
                schedule.every().day.at(MORNING_ANALYSIS).do(manager.hundred_day_analysis)
                schedule.every().day.at(NOON_ANALYSIS).do(manager.hundred_day_analysis)
                schedule.every().day.at(EVENING_SUMMARY).do(manager.send_summary)
                if time.strftime('%H:%M') >= NOON_ANALYSIS:
                    manager.hundred_day_analysis()
            while not restart:
                if not bot_process.is_alive():
                    raise RuntimeError('Telegram-Bot ist beendet. Bitte das Protokoll prüfen.')
                # Queue.empty() ist bei mehreren Prozessen nicht zuverlässig.
                for _ in range(20):
                    try:
                        command = command_q.get_nowait()
                    except Empty:
                        break
                    try:
                        restart = manager.handle_command(command)
                    except Exception as error:
                        func.pushover(f'Befehl fehlgeschlagen: {error}')
                    if restart:
                        break
                if not restart:
                    schedule.run_pending()
                    time.sleep(0.5)
        except KeyboardInterrupt:
            func.log_print('Programm beendet.')
        except Exception as error:
            func.log_print(f'Programmfehler: {error}')
        finally:
            schedule.clear()
            if bot_process is not None and bot_process.is_alive():
                notify_q.put(None)
                bot_process.join(timeout=10)
                if bot_process.is_alive():
                    bot_process.terminate()
                    bot_process.join(timeout=5)
            func.set_message_queue(None)
            for queue in (notify_q, command_q):
                queue.cancel_join_thread()
                queue.close()
        if not restart:
            break
        # Kein rekursiver subprocess-Aufruf: Der alte Bot ist bereits beendet.
        old_manager = manager
        func.log_print('Programm wird neu initialisiert.')


if __name__ == '__main__':
    mp.freeze_support()
    run_program()
