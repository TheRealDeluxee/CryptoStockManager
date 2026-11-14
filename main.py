"""CaSa: Konfiguration laden, Analysen planen und Telegram-Befehle bearbeiten.

Start: python main.py
Die eigentlichen Berechnungen bleiben in classes.py und func.py.
"""
import configparser
import copy
import datetime
import html
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
EVENT_REMINDER = "08:00"   # Erinnerung an Termine aus events.ini (mit Tagesanalyse)

# Vorlage für /ai. Wird beim ersten Aufruf als ki_frage.txt angelegt und kann dort
# mit dem Editor geändert werden. Platzhalter in {Klammern} füllt das Programm.
KI_FRAGE_DATEI = "ki_frage.txt"
KI_FRAGE_VORLAGE = """{name} ({symbol}) – Analyse für drei Monate

Bewerte {name} ({typ}) anhand meiner technischen Daten und aktueller Nachrichten. Anlagehorizont: drei Monate, kein Daytrading. {aufgabe}

A. Technische Daten aus meinem Programm (Tageswerte, Stand {datum})
Kurse in EUR, Quelle: {quelle}. Vergleiche Kursmarken nur mit dieser Quelle.
- Ausgangslage: {position}
- Kurs: {kurs} · 1 Tag {tag} · 7-Tage-Trend {woche}
- Hoch/Tief der letzten 250 Tageswerte: {hoch} / {tief}
- Widerstand: {resistance}
- Unterstützung: {support}
- RSI14: {rsi}
- MACD (EMA12/EMA26): {macd}
- Bollinger (20,2): {bollinger}
- Trend: {trend}
- Kurzfristig: {kurzfristig}
- Bodenbildung: {boden}
- Topbildung: {top}
- Marktphase {name}: {phase}
- Marktphase {markt_name}: {markt}
- Programm-Score: {score} (Skala -18 bis +18, positiv = eher Kauf)

B. Nachrichten und Termine
Nutze die Websuche; ohne Webzugriff sage das ausdrücklich. Nachrichten der letzten 14 Tage, bestätigte Termine der nächsten 4 Wochen. Ältere Zahlen nur als datierte Vergleichsbasis.
{recherche}

C. Regeln
- Prüfe breit, berichte gezielt: nur was die Empfehlung begründet, ihr widerspricht, das Risiko oder den Ein-/Ausstieg verändert. Gleiche Signale zusammenfassen.
- Technik nur aus Teil A, Nachrichten mit Datum und Quelle. Fehlt Wesentliches: exakt "nicht belegbar" und was fehlt.
- Kursmarken kritisch: Ein Periodentief ist nur eine mögliche Unterstützung, ein niedriger RSI zeigt nur, dass eine Erholung möglich ist. Ein Boden gilt erst nach beobachtbarer Kursreaktion als bestätigt.
- Handelsplan: Bei fallendem Kurs erst nach bestätigtem Boden kaufen, gestaffelt nahe der Unterstützung. Ausstieg unter der Unterstützung vorher festlegen und sagen, ob ein Tagesschluss zählt. Bei bärischer Phase (Wert oder Gesamtmarkt) nur kleine Tranchen, Trendwechsel erst über dem SMA200.
- Nach starkem Anstieg auf Rücksetzer warten, am Widerstand Teilgewinne erwägen, Position begrenzen. {steuer}Höchstens drei Preiszonen, wenige Zahlen, keine Disclaimer.

D. Ausgabe
1. Entscheidung in genau drei Sätzen: (1) {entscheidung}, Marktphase von {name} und {markt_name}, wichtigster Grund. (2) Unterstützung und Widerstand in Euro, ist ein Boden bestätigt? Topbildung nur, wenn entscheidend. (3) Score 0-10 = (Programm-Score + 18) / 36 x 10, dazu Risiko und Sicherheit je hoch/mittel/niedrig.
2. Was die Entscheidung trägt: höchstens vier Punkte aus Technik, Nachrichten und Bewertung - was belegt ist und was es für mich bedeutet.
3. Handelsplan als Tabelle (Szenario | beobachtbares Signal | meine Handlung):
{plan}
4. Stärkstes Gegenargument in höchstens zwei Sätzen, und welche Entwicklung es stärken würde.
5. Höchstens vier bestätigte Termine der nächsten 4 Wochen als Tabelle (Datum und deutsche Uhrzeit, Ereignis, Bedeutung, Quelle).
6. Fazit in höchstens vier einfachen Sätzen: was ich jetzt tue, worauf ich warte, was {aktion} auslöst oder ausschließt. Höchstens zwei Kursmarken, ohne Score und Fachkürzel.
"""


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
        self.events_done = set()   # heute schon erinnerte Termine (nur im Speicher)

    def load_config_items(self):
        """Neue Listen erst vollständig aufbauen, dann die alten ersetzen."""
        all_entries = cl.read_assets()
        entries = cl.read_assets(alarm_only=True)   # nur Werte mit 1 kommen in die Alarm-Analyse
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
            key = (row['type'], row['symbol'])
            if key not in all_entries:
                func.log_print(f"Inaktive Transaktion: {row['symbol']} fehlt in assets.ini und wird nicht analysiert.")
            elif key not in entries and row['action'] != 'watchlist':
                func.log_print(f"Hinweis: {row['symbol']} ist im Bestand, aber ohne 1 nicht in der Alarm-Analyse.")
        self.crypto_items_hh, self.crypto_items_hd = crypto_hh, crypto_hd
        self.stock_items_hh, self.stock_items_hd = stock_hh, stock_hd
        # Tagestabellen bleiben erhalten: Kurs, P&L und Rating hängen nicht vom Bestand ab.
        func.log_print(f"Geladen: {len(entries)} von {len(all_entries)} Symbolen in der Alarm-Analyse, {len(rows)} Transaktionen.")

    def _is_stock_trading_hours(self):
        now = time.localtime()
        # Zahlen funktionieren unabhängig von der Windows-Sprache.
        self.stock_update = self.test_mode == 1 or (now.tm_wday < 5 and 8 <= now.tm_hour < 22)

    def _analyse_items(self, items, data_type):
        results = []
        for item in items:
            try:
                df = item.get_API_data(data_type)
                df, alarms = item.process_data(df, data_type)  # df ist danach auf 100 Werte gekürzt
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

    def _daily_analysis(self, symbol, asset_type):
        """Tagesanalyse eines Werts aus assets.ini für /check und /ai (ohne Alarm)."""
        entries = cl.read_assets()
        key = (asset_type, symbol.lower())
        if key not in entries:
            raise ValueError(f"{symbol} ist nicht in assets.ini ({asset_type}).")
        name = entries[key]
        symbol = symbol.upper() if asset_type == 'stock' else symbol.lower()
        # Eigenes Objekt: verändert weder Analyse- noch Alarmzustände.
        item = cl.apply_transactions(cl.CryptoStock(symbol, name, asset_type), cl.read_transactions())
        full = item.get_API_data('100d')
        full = func.calc_indicator_fuctions(full)
        sr_zones = func.support_resistance(full)  # vor dem Kürzen: braucht die lange Historie
        df = func.cut_to_window(full)
        item.state = func.calc_state_fuctions(item.state, df)
        details = func.analysis_details(df)
        return name, symbol, item, df, full, sr_zones, details

    def single_daily_analysis(self, symbol, asset_type):
        name, symbol, item, df, full, sr_zones, details = self._daily_analysis(symbol, asset_type)
        # Telegram-HTML wie bei den Alarmen (fett/kursiv/Festbreite)
        message = (f"<b>{html.escape(name)} ({html.escape(symbol)}) — Tagesanalyse (100d)</b>\n"
                   f"Manueller Abruf ohne Alarm\n\n" + details['message'])
        if not item.state.watch_list:
            message += "\n\n" + html.escape(f"Amount >1 year: {item.state.amount_older_than_one_year:.2f} "
                                           f"({item.state.amount_older_than_one_year_pct:.2f} %)")
        image_name = 'manual_' + uuid.uuid4().hex
        func.plot_and_save(df, image_name, '100d', item.state.zero_line, sr_zones)
        func.pushover_image(image_name, message, delete_after_send=True, html_format=True)

    def ai_prompt(self, symbol, asset_type):
        """/ai: Frage für ChatGPT/Claude aus ki_frage.txt füllen und als kopierbaren Text senden."""
        name, symbol, item, df, full, sr_zones, details = self._daily_analysis(symbol, asset_type)
        if not os.path.exists(KI_FRAGE_DATEI):
            with open(KI_FRAGE_DATEI, 'w', encoding='utf-8') as file:
                file.write(KI_FRAGE_VORLAGE)
        with open(KI_FRAGE_DATEI, encoding='utf-8-sig') as file:
            question = file.read()

        price = float(df['Price'].iloc[-1])

        def pct(value):
            return f"{value:+.1f} %".replace('.', ',')

        def zone_text(zone, extreme):
            if zone is None:
                return 'keine im geladenen Zeitraum'
            extra = (f"{zone['touches']} Berührungen laut Programm" if zone['touches']
                     else f"nur {extreme} des Zeitraums, keine bestätigte Zone")
            return f"{func.euro_de(zone['center'])} ({pct((zone['center'] / price - 1) * 100)}, {extra})"

        resistance, support = sr_zones if sr_zones else (None, None)

        # Ausgangslage mit echten Werten (Menge, Einstand, Wert, Gewinn/Verlust)
        def qty(value):
            # Krypto immer mit 8 Nachkommastellen (wie bei Börsen üblich), Aktien ohne unnötige Nullen
            return (f"{value:.8f}" if asset_type == 'crypto' else f"{value:.8g}").replace('.', ',')

        state = item.state
        unit = 'Stück' if asset_type == 'stock' else 'Coins'
        if state.watch_list:
            position = 'kein Bestand'
        else:
            position = (f"{qty(state.amount_crypto_stock)} {unit}, Einstand {func.euro_de(state.zero_line)} je {unit[:-1] if unit == 'Coins' else unit} "
                        f"(investiert {func.euro_de(state.buy_balance_eur)}) · Wert {func.euro_de(state.current_balance_eur)} · "
                        f"Gewinn/Verlust {func.euro_de(state.current_profit)} ({pct(state.current_profit_pct)}) · "
                        f"länger als 1 Jahr gehalten: {qty(state.amount_older_than_one_year)} {unit}")

        # Fragestellung und Handelsplan hängen davon ab, ob ich den Wert besitze
        if state.watch_list:
            aufgabe = (f"Ich habe keine Position in {name}. Entscheide, ob ich kaufen oder abwarten sollte, "
                       f"und nenne das konkrete Signal, das diese Entscheidung ändern würde.")
            entscheidung, aktion = 'Beginne mit Kaufen oder Abwarten', 'einen Kauf'
            plan = ("- Abwarten | welche Bestätigung fehlt noch? | was tue ich bis dahin?\n"
                    "- Erster Kauf | welche Kursreaktion bestätigt den Einstieg? | wann kaufe ich eine kleine erste Tranche?\n"
                    "- Plan fällt aus | welche Kursmarke oder welches Ereignis macht ihn hinfällig? | was tue ich vor bzw. nach einem Kauf?")
        else:
            aufgabe = (f"Ich besitze {name} (siehe Ausgangslage). Entscheide, ob ich halten, aufstocken oder reduzieren sollte, "
                       f"und nenne das konkrete Signal, das diese Entscheidung ändern würde.")
            entscheidung, aktion = 'Beginne mit Halten, Aufstocken oder Reduzieren', 'einen Zukauf oder Verkauf'
            plan = ("- Halten | welche Lage bestätigt das Halten? | was beobachte ich?\n"
                    "- Aufstocken | welche Kursreaktion bestätigt einen Zukauf? | wann und wie viel kaufe ich nach?\n"
                    "- Reduzieren/Ausstieg | welche Kursmarke oder welches Ereignis? | wie viel verkaufe ich?")
        steuer = ('Haltedauer beachten: Ein Verkauf von Teilen unter 1 Jahr Haltedauer ist steuerpflichtig. '
                  if asset_type == 'crypto' and not state.watch_list else '')
        quelle = 'CoinGecko (Durchschnitt mehrerer Börsen)' if asset_type == 'crypto' else 'Yahoo Finance, Kurse ggf. in EUR umgerechnet'

        # MACD-Linie (EMA12 - EMA26) und letzte Kreuzung im 100er-Fenster
        diff = df['Fast EMA'].iloc[-1] - df['Slow EMA'].iloc[-1]
        crosses = df.index[df['Position'].fillna(0) != 0]
        since = f"Kreuzung vor {len(df) - 1 - crosses[-1]} Tagen" if len(crosses) else 'keine Kreuzung in 100 Tagen'
        macd = f"{'positiv (EMA12 über EMA26)' if diff > 0 else 'negativ (EMA12 unter EMA26)'}, {since}"

        # Lage im Bollinger-Band: 0 % = unteres Band, 100 % = oberes Band
        lower, upper = df['BB_Lower'].iloc[-1], df['BB_Upper'].iloc[-1]
        band = (price - lower) / (upper - lower) * 100 if upper > lower else 50.0
        where = ('unter dem unteren Band' if band < 0 else 'nahe unterem Band' if band < 20 else
                 'über dem oberen Band' if band > 100 else 'nahe oberem Band' if band > 80 else 'im mittleren Bereich')
        bollinger = f"Kurs {where} ({band:.0f} % der Bandbreite)"

        # Boden-/Topbildung: Anzeichen aus der langen Historie
        bottom, top = func.formation_signals(full)

        def formation_text(signals, names):
            parts = [f"{label}: {'kein aktuelles Extrem' if signals[key] is None else 'ja' if signals[key] else 'nein'}"
                     for key, label in names]
            count = sum(1 for key, _ in names if signals[key])
            return f"{count} von {len(names)} Anzeichen ({', '.join(parts)})"

        boden = formation_text(bottom, (('folge', 'höhere Tiefs'), ('divergenz', 'bullische RSI-Divergenz'),
                                        ('doppel', 'Doppelboden'), ('volumen', 'Verkaufs-Höhepunkt beim Tief'),
                                        ('bestaetigung', 'Kurs über EMA12')))
        top_text = formation_text(top, (('folge', 'tiefere Hochs'), ('divergenz', 'bärische RSI-Divergenz'),
                                        ('doppel', 'Doppeltop'), ('volumen', 'Kauf-Höhepunkt beim Hoch'),
                                        ('bestaetigung', 'Kurs unter EMA12')))

        # Marktphase: eigener Wert und Leitmarkt (Bitcoin bzw. Börsenindex)
        phase = func.trend_phase(full['Price'])
        if asset_type == 'crypto' and symbol == 'bitcoin':
            market_name, market = 'Krypto-Gesamtmarkt', 'wie oben, Bitcoin ist selbst der Leitwert'
        else:
            try:
                market_name, market_prices = func.market_prices(asset_type, symbol)
                market = func.trend_phase(market_prices) if market_prices is not None else 'nicht belegbar (keine Daten)'
            except Exception as error:   # Leitmarkt ist Zusatzinfo: Fehler darf /ai nicht stoppen
                market_name, market = 'Gesamtmarkt', f'nicht belegbar ({error})'

        high, low = full['Price'].max(), full['Price'].min()
        if asset_type == 'stock':
            recherche = ("- Quartalszahlen und Ausblick: letzter Umsatz, letzte Unternehmensprognose, Analystenkonsens, nächster "
                         "bestätigter Termin (deutsche Uhrzeit). Ist, Prognose und Schätzung trennen; welche Aussage im nächsten "
                         "Ausblick die Empfehlung ändern würde.\n"
                         "- Analystenziele und Bewertung: wesentliche Kursziel-Änderungen (Originalwährung), KGV rückblickend "
                         "und erwartet (mit Zeitraum der Schätzung).\n"
                         "- Unternehmen und Branche: Management, Nachfrage, Wettbewerb, Regulierung und Exporte.")
        else:
            recherche = ("- Kapitalflüsse: ETF-Zu- und -Abflüsse, große Wallet-Bewegungen, Bestände auf Börsen.\n"
                         "- Regulierung und Makro: Gesetze, Verfahren, Zinsentscheide, Inflation.\n"
                         "- Projekt und Netzwerk: Updates, Token-Freigaben (Unlocks), Sicherheitsvorfälle, Konkurrenz.")

        values = {
            'name': name, 'symbol': symbol,
            'typ': 'Kryptowährung' if asset_type == 'crypto' else 'Aktie',
            'aufgabe': aufgabe, 'entscheidung': entscheidung, 'aktion': aktion, 'plan': plan,
            'steuer': steuer, 'quelle': quelle,
            'datum': time.strftime('%d.%m.%Y %H:%M'),
            'position': position,
            'kurs': func.euro_de(price),
            'tag': pct(item.state.current_one_day_price_change_pct),
            'woche': pct(item.state.current_seven_days_slope_pct),
            'hoch': f"{func.euro_de(high)} ({pct((price / high - 1) * 100)})",
            'tief': f"{func.euro_de(low)} ({pct((price / low - 1) * 100)})",
            'resistance': zone_text(resistance, 'Hoch'), 'support': zone_text(support, 'Tief'),
            'score': str(details['score']),
            'rsi': f"{df['RSI14'].iloc[-1]:.0f} ({details['texts']['rsi']})",
            'macd': macd, 'bollinger': bollinger,
            'trend': details['texts']['trend'], 'kurzfristig': details['texts']['short'],
            'boden': boden, 'top': top_text, 'recherche': recherche,
            'phase': phase, 'markt_name': market_name, 'markt': market,
        }
        # Einfaches Ersetzen statt format(): weitere {Klammern} in der Datei stören so nicht.
        for key, value in values.items():
            question = question.replace('{' + key + '}', value)
        question = question.strip()
        message = (f"<b>KI-Frage: {html.escape(name)}</b>\n"
                   f"Kopieren und in ChatGPT oder Claude einfügen:\n"
                   f"<pre>{html.escape(question)}</pre>")
        if len(message) <= 4000:
            func.pushover(message, html_format=True)
            return
        # Zu lang für eine Telegram-Nachricht: an einem Absatz nahe der Mitte in zwei kopierbare Blöcke teilen
        breaks = [i for i in range(len(question)) if question.startswith('\n\n', i)]
        cut = min(breaks, key=lambda i: abs(i - len(question) / 2)) if breaks else len(question) // 2
        parts = (question[:cut].strip(), question[cut:].strip())
        for number, part in enumerate(parts, 1):
            hint = 'Kopieren und in ChatGPT oder Claude einfügen:' if number == 1 else 'Direkt im Anschluss an Teil 1 einfügen:'
            text = f"<b>KI-Frage: {html.escape(name)} (Teil {number}/2)</b>\n{hint}\n<pre>{html.escape(part)}</pre>"
            if len(text) <= 4000:
                func.pushover(text, html_format=True)
            else:   # auch die Hälfte ist zu lang (stark erweiterte Vorlage): ohne Formatierung senden
                func.pushover(part)

    def send_summary(self):
        if self.df_crypto_hd.empty and self.df_stock_hd.empty:
            func.pushover('Noch keine Tagesergebnisse vorhanden. Bitte zuerst /analyse daily ausführen.')
            return
        for kind, frame in (('crypto', self.df_crypto_hd), ('stock', self.df_stock_hd)):
            if frame.empty:
                continue
            frame = frame.sort_values('1d P&L [%]', ascending=False)
            image_name = f'summary_{kind}_' + uuid.uuid4().hex
            title = 'Krypto · Tagesübersicht' if kind == 'crypto' else 'Aktien · Tagesübersicht'
            # Datenstand steht schon in der Telegram-Nachricht, nicht noch einmal im Bild
            func.create_table_image(frame, os.path.join(self.output_dir, image_name + '.png'), title=title)
            message = f'Datenstand: {self.last_daily_update}'  # Titel steht schon im Bild
            func.pushover_image(image_name, message, delete_after_send=True)

    def event_reminders(self):
        """Termine von heute aus events.ini: Erinnerung senden, danach Tagesanalyse wie /check."""
        try:
            events = cl.read_events()
        except (OSError, ValueError, configparser.Error) as error:
            func.pushover(f'events.ini konnte nicht gelesen werden: {error}')
            return
        today = datetime.date.today()
        for event in events:
            if event['date'] != today or event['key'] in self.events_done:
                continue
            self.events_done.add(event['key'])
            text = f"\n{html.escape(event['text'])}" if event['text'] else ''
            func.pushover(f"<b>Termin heute: {html.escape(event['name'])}</b>{text}\nTagesanalyse folgt.",
                          priority=1, html_format=True)
            try:
                self.single_daily_analysis(event['symbol'], event['type'])
            except Exception as error:
                func.pushover(f"Tagesanalyse für {event['name']} fehlgeschlagen: {error}")

    def startup_message(self):
        """Kurze Startnachricht: Version, Werte in der Alarm-Analyse, nächste Termine."""
        crypto, stock = len(self.crypto_items_hd), len(self.stock_items_hd)
        lines = [f"<b>CaSa gestartet</b> · Version {html.escape(str(self.version))}",
                 f"{crypto + stock} Werte in der Alarm-Analyse ({crypto} Krypto, {stock} Aktien)"]
        if self.test_mode == 1:
            lines.append("Testmodus: alle Analysen laufen einmal sofort")
        else:
            now = time.strftime('%H:%M')
            if now < MORNING_ANALYSIS:
                daily = MORNING_ANALYSIS
            elif now < NOON_ANALYSIS:
                daily = NOON_ANALYSIS
            else:
                daily = f"jetzt, danach morgen {MORNING_ANALYSIS}"   # nach 12:30 läuft sie beim Start sofort
            summary = EVENING_SUMMARY if now < EVENING_SUMMARY else f"morgen {EVENING_SUMMARY}"
            lines.append(f"Nächste Tagesanalyse: {daily} · Übersicht: {summary}")
        try:
            upcoming = [event for event in cl.read_events() if event['date'] >= datetime.date.today()]
        except (OSError, ValueError, configparser.Error) as error:
            lines.append(f"events.ini fehlerhaft: {html.escape(str(error))}")
        else:
            if upcoming:
                event = upcoming[0]
                text = f" – {event['text']}" if event['text'] else ''
                lines.append(html.escape(f"Nächster Termin: {event['date'].strftime('%d.%m.%Y')} {event['name']}{text}"))
        lines.append("Alle Befehle: /help")
        func.pushover('\n'.join(lines), html_format=True)

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
        elif action == 'ai':
            self.ai_prompt(command['symbol'], command['type'])
        elif action == 'reload_transactions':
            self.load_config_items()
            func.pushover('Transaktionen wurden im Hauptprogramm neu geladen.')
        elif action == 'events':
            # nach /termin mit heutigem Datum: Erinnerung und Analyse sofort
            self.event_reminders()
        elif action == 'reload_assets':
            # nach /alarm: geänderte Alarm-Kennzeichen aus assets.ini übernehmen
            self.load_config_items()
            names = [item.search for item in self.crypto_items_hd + self.stock_items_hd]
            func.pushover(f"Alarm-Analyse neu geladen ({len(names)} Werte): {', '.join(names)}")
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
            manager.startup_message()
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
            # Termine aus events.ini: auch im Testmodus täglich prüfen
            schedule.every().day.at(EVENT_REMINDER).do(manager.event_reminders)
            if time.strftime('%H:%M') >= EVENT_REMINDER:
                manager.event_reminders()   # Start nach 08:00: heutige Termine nachholen
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
