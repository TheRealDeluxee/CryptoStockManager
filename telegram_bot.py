"""
telegram_bot.py
- Always-on Telegram Bot
- Sendet Notifications aus notify_queue (Text/Bild) an Telegram
- Nimmt Commands entgegen (/help /status /analyse /check /ai /alarm /buy /sell /sellall /history /restart)
  und schickt sie als dict in command_queue an main

Wichtig:
- Bot ist entkoppelt von main. Kein direkter Zugriff auf CryptoStockManager.
"""

import asyncio
import configparser
import csv
import logging
import os
import tempfile
from queue import Empty
import classes as cl
from datetime import datetime
from multiprocessing import Queue
from typing import Dict, List, Optional

from telegram import Update, BotCommand, BotCommandScopeChat, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOT] %(message)s",
)

_TRANSACTIONS_PATH = "transactions.csv"
CSV_HEADER = ["type", "symbol", "name", "action", "amount", "qty", "date"]


def _load_config() -> Dict[str, str]:
    cfg = configparser.ConfigParser()
    if not cfg.read("config.ini", encoding="utf-8-sig"):
        raise FileNotFoundError("config.ini nicht gefunden")
    token = cfg.get("telegram", "token")
    chat_id = cfg.get("telegram", "chat_id")
    return {"token": token, "chat_id": str(chat_id)}


def _authorized(update: Update, chat_id: str) -> bool:
    return update.effective_chat is not None and str(update.effective_chat.id) == str(chat_id)


def _priority_prefix(priority: int) -> str:
    # Text statt Symbol; leise Nachrichten (-1/-2) ohne Zusatz
    return {1: "WICHTIG · ", 2: "DRINGEND · "}.get(int(priority), "")


# ────────────────────────────────────────────────
# CSV helpers (wie früher)
# ────────────────────────────────────────────────
def _ensure_csv_exists() -> None:
    if not os.path.exists(_TRANSACTIONS_PATH):
        with open(_TRANSACTIONS_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADER, delimiter=";")
            w.writeheader()


def _append_transaction(asset_type, symbol, name, action, amount, qty, date):
    # Erst eine vollständige neue Datei schreiben, dann atomar austauschen.
    # main.py kann dadurch keine halb geschriebene CSV-Zeile lesen.
    rows = cl.read_transactions(_TRANSACTIONS_PATH)
    row = {"type": asset_type, "symbol": symbol, "name": name,
           "action": action, "amount": str(amount), "qty": str(qty), "date": date}
    asset = cl.CryptoStock(symbol, name, asset_type)
    cl.apply_transactions(asset, rows + [row])
    rows.append(row)
    directory = os.path.dirname(os.path.abspath(_TRANSACTIONS_PATH))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                         dir=directory, suffix='.tmp', delete=False) as file:
            temp_path = file.name
            writer = csv.DictWriter(file, fieldnames=CSV_HEADER, delimiter=';')
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, _TRANSACTIONS_PATH)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _get_history(symbol):
    return [row for row in cl.read_transactions(_TRANSACTIONS_PATH)
            if row['symbol'].lower() == symbol.lower()]


def _count_transactions():
    return len(cl.read_transactions(_TRANSACTIONS_PATH))


def _asset_name(symbol, asset_type):
    entries = cl.read_assets()
    key = (asset_type, symbol.lower())
    if key not in entries:
        section = 'crypto' if asset_type == 'crypto' else 'stocks'
        raise ValueError(f"Symbol {symbol} ist nicht in assets.ini im Abschnitt [{section}] vorhanden.")
    return entries[key]


def _parse_floats(amount_str, qty_str):
    try:
        amount, qty = cl.valid_numbers(amount_str, qty_str)
        return amount, qty, None
    except ValueError as error:
        return 0.0, 0.0, str(error)


def _valid_date(s: str) -> bool:
    try:
        return datetime.strptime(s, "%d/%m/%Y").date() <= datetime.now().date()
    except ValueError:
        return False


# ────────────────────────────────────────────────
# Queue workers
# ────────────────────────────────────────────────
def _text_chunks(text, limit=4000):
    # Telegram begrenzt die Länge; auch Emojis zählen zuverlässig mit.
    part, units = '', 0
    for char in str(text):
        size = len(char.encode('utf-16-le')) // 2
        if units + size > limit:
            yield part
            part, units = '', 0
        part += char
        units += size
    if part:
        yield part


async def _reply_long(update, text):
    for part in _text_chunks(text):
        await update.message.reply_text(part, parse_mode=None)


async def _notify_worker(app: Application) -> None:
    notify_q = app.bot_data['notify_q']
    chat_id = app.bot_data['chat_id']
    while True:
        try:
            item = notify_q.get_nowait()
        except Empty:
            await asyncio.sleep(0.2)
            continue
        if item is None:
            # stop() allein beendet run_polling() nicht.
            while not app.running:
                await asyncio.sleep(0.1)
            app.stop_running()
            return
        if not isinstance(item, dict):
            continue
        try:
            text = _priority_prefix(item.get('priority', 0)) + str(item.get('message', ''))
            parse_mode = 'HTML' if item.get('html') else None  # Analyse-Nachrichten mit fett/kursiv/Festbreite
            if item.get('type') == 'image':
                path = os.path.join(item.get('output_dir', 'plots'), item.get('symbol', 'plot') + '.png')
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Diagramm fehlt: {path}")
                # Längere Kennwerte als eigene Nachricht senden, nichts abschneiden.
                caption_fits = len(text.encode('utf-16-le')) // 2 <= 1000
                with open(path, 'rb') as photo:
                    await app.bot.send_photo(chat_id=chat_id, photo=photo,
                                             caption=text if caption_fits else None, parse_mode=parse_mode)
                if not caption_fits:
                    for part in _text_chunks(text):
                        await app.bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode)
                if item.get('delete_after_send'):
                    os.remove(path)
            elif item.get('type') == 'text':
                for part in _text_chunks(text):
                    await app.bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode)
        except Exception:
            logging.exception('Telegram-Nachricht konnte nicht gesendet werden')
        await asyncio.sleep(0)


# ────────────────────────────────────────────────
# Commands -> command_queue
# ────────────────────────────────────────────────
def _cmdq(app: Application) -> Queue:
    return app.bot_data["command_q"]  # type: ignore[return-value]


HELP_TEXT = (
        "🤖 *CaSa Bot*\n"
        "────────────────────────\n"
        "`/analyse <hourly|daily|summary>`\n"
        "`/check`  (Auswahl per Schaltfläche, daily ohne Alarm)\n"
        "`/check <symbol> <crypto|stock>`\n"
        "`/ai`  (KI-Frage zum Kopieren in ChatGPT/Claude)\n"
        "`/ai <symbol>`\n"
        "`/alarm`  (Werte in Alarm-Analyse an/aus schalten)\n"
        "`/alarm <symbol> <an|aus>`\n"
        "`/buy <symbol> <crypto|stock> <amount> <qty> <DD/MM/YYYY>`\n"
        "`/sell <symbol> <crypto|stock> <amount> <qty>`\n"
        "`/sellall <symbol> [betrag]`  (ganzen Bestand verkaufen)\n"
        "`/history <symbol>`\n"
        "`/transactions`  (alle Transaktionen anzeigen)\n"
        "`/status`\n"
        "`/restart`  (main neu starten + reload transactions)\n"
        "`/start`  (Hilfe anzeigen)\n"
        "`/help`"
)


async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")

async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _cmd_help(update, context)


async def _cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    try:
        tx_count = _count_transactions()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await _reply_long(update, f'Bot läuft, aber Transaktionen sind nicht lesbar: {error}')
        return
    await update.message.reply_text(
        "✅ *Bot running*\n"
        f"Time: `{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}`\n"
        f"Transactions saved: `{tx_count}`\nNur Bot-Status; laufende Analysen können noch warten.",
        parse_mode="Markdown"
    )


async def _cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    # Ohne Angaben: Werte aus assets.ini als Schaltflächen anzeigen, Antippen startet die Analyse.
    if not context.args:
        await _reply_asset_buttons(update, 'check', 'Welche Tagesanalyse (ohne Alarm)?')
        return
    if len(context.args) != 2 or context.args[1].lower() not in ('crypto', 'stock'):
        await update.message.reply_text('Format: /check (Auswahl per Schaltfläche) oder /check <symbol> <crypto|stock>')
        return
    symbol, asset_type = context.args[0].lower(), context.args[1].lower()
    try:
        _asset_name(symbol, asset_type)
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Fehler: {error}')
        return
    # Wie /analyse: Das Hauptprogramm übernimmt die Berechnung.
    _cmdq(app).put({'cmd': 'check', 'symbol': symbol, 'type': asset_type})
    await update.message.reply_text(f'Tagesanalyse für {symbol} vorgemerkt (ohne Alarm).')


async def _reply_asset_buttons(update, cmd, question):
    # Werte aus assets.ini als Schaltflächen; callback_data = "<cmd>|<typ>|<symbol>" (für /check und /ai)
    try:
        entries = cl.read_assets()
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Fehler: {error}')
        return
    buttons = [InlineKeyboardButton(('🪙 ' if asset_type == 'crypto' else '📈 ') + name,
                                    callback_data=f'{cmd}|{asset_type}|{symbol}')
               for (asset_type, symbol), name in entries.items()]
    keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]  # 2 Schaltflächen pro Zeile
    await update.message.reply_text(question, reply_markup=InlineKeyboardMarkup(keyboard))


_BUTTON_REPLIES = {'check': 'Tagesanalyse für {name} vorgemerkt (ohne Alarm).',
                   'ai': 'KI-Frage für {name} wird erstellt …'}


async def _cb_assets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Klick auf eine Schaltfläche aus /check oder /ai
    query = update.callback_query
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        await query.answer()
        return
    await query.answer()
    cmd, asset_type, symbol = query.data.split('|', 2)
    try:
        name = _asset_name(symbol, asset_type)  # assets.ini kann sich inzwischen geändert haben
    except (OSError, ValueError, configparser.Error) as error:
        await query.edit_message_text(f'Fehler: {error}')
        return
    _cmdq(app).put({'cmd': cmd, 'symbol': symbol, 'type': asset_type})
    # Schaltflächen entfernen, damit nicht versehentlich doppelt gestartet wird
    await query.edit_message_text(_BUTTON_REPLIES[cmd].format(name=name))


async def _cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /ai: Frage für ChatGPT/Claude erzeugen (Text zum Kopieren, keine API)
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    if not context.args:
        await _reply_asset_buttons(update, 'ai', 'KI-Frage zu welchem Wert?')
        return
    if len(context.args) > 2 or (len(context.args) == 2 and context.args[1].lower() not in ('crypto', 'stock')):
        await update.message.reply_text('Format: /ai (Auswahl per Schaltfläche) oder /ai <symbol> [crypto|stock]')
        return
    symbol = context.args[0].lower()
    try:
        entries = cl.read_assets()
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Fehler: {error}')
        return
    # Typ darf fehlen: wird in assets.ini gesucht
    types = [context.args[1].lower()] if len(context.args) == 2 else ['crypto', 'stock']
    matches = [asset_type for asset_type in types if (asset_type, symbol) in entries]
    if len(matches) != 1:
        problem = 'nicht in assets.ini' if not matches else 'als crypto und stock vorhanden, bitte Typ angeben'
        await update.message.reply_text(f'Fehler: {symbol} ist {problem}.')
        return
    _cmdq(app).put({'cmd': 'ai', 'symbol': symbol, 'type': matches[0]})
    await update.message.reply_text(_BUTTON_REPLIES['ai'].format(name=entries[(matches[0], symbol)]))


def _alarm_keyboard():
    # Alle Werte aus assets.ini mit aktuellem Alarm-Zustand; Antippen schaltet um
    entries = cl.read_assets()
    active = cl.read_assets(alarm_only=True)
    buttons = [InlineKeyboardButton(f"{name}: {'an' if key in active else 'aus'}",
                                    callback_data=f'alarm|{key[0]}|{key[1]}')
               for key, name in entries.items()]
    return InlineKeyboardMarkup([buttons[i:i + 2] for i in range(0, len(buttons), 2)]), len(active)


async def _cmd_alarm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /alarm: Wert in die automatische Alarm-Analyse aufnehmen (1) oder herausnehmen (0)
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    try:
        if not context.args:
            keyboard, count = _alarm_keyboard()
            await update.message.reply_text(f'Alarm-Analyse: {count} Werte an. Antippen schaltet um.',
                                            reply_markup=keyboard)
            return
        switch = {'an': True, 'on': True, '1': True, 'aus': False, 'off': False, '0': False}
        if len(context.args) != 2 or context.args[1].lower() not in switch:
            await update.message.reply_text('Format: /alarm (Auswahl per Schaltfläche) oder /alarm <symbol> <an|aus>')
            return
        symbol, active = context.args[0].lower(), switch[context.args[1].lower()]
        entries = cl.read_assets()
        matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in entries]
        if len(matches) != 1:
            problem = 'nicht in assets.ini' if not matches else 'als crypto und stock vorhanden, bitte über die Schaltflächen wählen'
            await update.message.reply_text(f'Fehler: {symbol} ist {problem}.')
            return
        cl.set_alarm(symbol, matches[0], active)
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Fehler: {error}\nNicht geändert.')
        return
    _cmdq(app).put({'cmd': 'reload_assets'})
    name = entries[(matches[0], symbol)]
    await update.message.reply_text(f"{name}: Alarm {'an' if active else 'aus'}. Das Hauptprogramm lädt neu.")


async def _cb_alarm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Klick auf eine Schaltfläche aus /alarm: Zustand umschalten, Schaltflächen bleiben für weitere Änderungen
    query = update.callback_query
    app = context.application
    await query.answer()
    if not _authorized(update, app.bot_data['chat_id']):
        return
    _, asset_type, symbol = query.data.split('|', 2)
    try:
        name = _asset_name(symbol, asset_type)
        active = (asset_type, symbol) not in cl.read_assets(alarm_only=True)   # umschalten
        cl.set_alarm(symbol, asset_type, active)
        keyboard, count = _alarm_keyboard()
    except (OSError, ValueError, configparser.Error) as error:
        await query.edit_message_text(f'Fehler: {error}\nNicht geändert.')
        return
    _cmdq(app).put({'cmd': 'reload_assets'})
    await query.edit_message_text(f"{name}: Alarm {'an' if active else 'aus'}.\n"
                                  f"Alarm-Analyse: {count} Werte an. Antippen schaltet um.",
                                  reply_markup=keyboard)


async def _cmd_analyse(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    arg = context.args[0].lower() if context.args else "hourly"
    if arg not in ("hourly", "daily", "summary"):
        await update.message.reply_text(
            "❌ Unknown argument.\nUse `/analyse hourly|daily|summary`",
            parse_mode="Markdown",
        )
        return

    _cmdq(app).put({"cmd": "analyse", "mode": arg})
    await update.message.reply_text(f"Analyse vorgemerkt: {arg}")


async def _cmd_restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    _cmdq(app).put({"cmd": "restart_main"})
    await update.message.reply_text("🔄 main wird neu gestartet…", parse_mode="Markdown")


async def _save_trade(update, context, action):
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    expected = 5 if action == 'buy' else 4
    if len(context.args) != expected:
        date_help = ' <DD/MM/YYYY>' if action == 'buy' else ''
        await update.message.reply_text(f'Format: /{action} <symbol> <crypto|stock> <betrag> <menge>{date_help}')
        return
    symbol, asset_type, amount_text, qty_text = context.args[:4]
    asset_type = asset_type.lower()
    if asset_type not in ('crypto', 'stock'):
        await update.message.reply_text('Fehler: Typ muss crypto oder stock sein. Nicht gespeichert.')
        return
    symbol = symbol.lower() if asset_type == 'crypto' else symbol.upper()
    date = context.args[4] if action == 'buy' else datetime.now().strftime('%d/%m/%Y')
    try:
        if not _valid_date(date):
            raise ValueError('Datum ungültig oder in der Zukunft. Format: DD/MM/YYYY.')
        amount, qty = cl.valid_numbers(amount_text, qty_text)
        name = _asset_name(symbol, asset_type)
        _append_transaction(asset_type, symbol, name, action, amount, qty, date)
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
        await _reply_long(update, f'Fehler: {error}\nTransaktion nicht gespeichert.')
        return
    _cmdq(app).put({'cmd': 'reload_transactions'})
    label = 'Kauf' if action == 'buy' else 'Verkauf'
    await update.message.reply_text(
        f'{label} gespeichert: {symbol}\nBetrag: {amount:.2f} €\nMenge: {qty}\nDatum: {date}\n'
        'Das Hauptprogramm lädt die Transaktionen neu.', parse_mode=None)


async def _cmd_buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _save_trade(update, context, 'buy')


async def _cmd_sell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _save_trade(update, context, 'sell')


def _holding(symbol, asset_type, name):
    # Aktueller Bestand laut transactions.csv (FIFO wie im Hauptprogramm)
    asset = cl.apply_transactions(cl.CryptoStock(symbol, name, asset_type), cl.read_transactions(_TRANSACTIONS_PATH))
    return asset.state.amount_crypto_stock


def _current_price(symbol, asset_type, name):
    # Letzter Kurs in EUR; blockiert bei API-Wiederholungen -> nur über asyncio.to_thread aufrufen
    df = cl.CryptoStock(symbol, name, asset_type).get_API_data('100d')
    if df is None or df.empty:
        raise ValueError('Kein aktueller Kurs verfügbar. Bitte Betrag angeben: /sellall <symbol> <betrag>')
    return float(df['Price'].iloc[-1])


async def _cmd_sellall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /sellall <symbol> [betrag]: kompletten Bestand als Verkauf eintragen (nach Bestätigung)
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    if len(context.args) not in (1, 2):
        await update.message.reply_text('Format: /sellall <symbol> [betrag in €]\nOhne Betrag wird der aktuelle Kurs verwendet.')
        return
    symbol = context.args[0].lower()
    try:
        entries = cl.read_assets()
        matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in entries]
        if len(matches) != 1:
            raise ValueError(f"{symbol} ist {'nicht in assets.ini' if not matches else 'als crypto und stock vorhanden'}.")
        asset_type = matches[0]
        name = entries[(asset_type, symbol)]
        symbol = symbol.lower() if asset_type == 'crypto' else symbol.upper()
        qty = _holding(symbol, asset_type, name)
        if qty <= 0:
            raise ValueError(f'Kein Bestand von {name} vorhanden.')
        if len(context.args) == 2:
            amount, _ = cl.valid_numbers(context.args[1], qty)
            source = 'dein Betrag'
        else:
            await update.message.reply_text(f'Aktueller Kurs für {name} wird abgefragt …')
            price = await asyncio.to_thread(_current_price, symbol, asset_type, name)
            amount = round(price * qty, 2)
            source = f'aktueller Kurs {price:.2f} €'
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
        await _reply_long(update, f'Fehler: {error}\nNichts gespeichert.')
        return

    # Erst nach Bestätigung speichern; Auftrag wird bis dahin im Bot gemerkt
    pending_id = os.urandom(4).hex()
    app.bot_data.setdefault('sellall', {})[pending_id] = (asset_type, symbol, name, amount, qty)
    unit = 'Coins' if asset_type == 'crypto' else 'Stück'
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton('Verkauf speichern', callback_data=f'sellall|{pending_id}|ja'),
                                      InlineKeyboardButton('Abbrechen', callback_data=f'sellall|{pending_id}|nein')]])
    await update.message.reply_text(
        f"Gesamten Bestand von {name} als Verkauf eintragen?\n"
        f"Menge: {qty:.8g} {unit}\nBetrag: {amount:.2f} € ({source})\n"
        f"Datum: {datetime.now().strftime('%d/%m/%Y')}\n"
        f"Wenn du einen anderen Erlös hattest: /sellall {symbol} <betrag>",
        reply_markup=keyboard)


async def _cb_sellall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Bestätigung aus /sellall
    query = update.callback_query
    app = context.application
    await query.answer()
    if not _authorized(update, app.bot_data['chat_id']):
        return
    _, pending_id, answer = query.data.split('|', 2)
    order = app.bot_data.get('sellall', {}).pop(pending_id, None)
    if order is None:
        await query.edit_message_text('Auftrag nicht mehr vorhanden (bereits erledigt oder Bot neu gestartet). Nichts gespeichert.')
        return
    if answer != 'ja':
        await query.edit_message_text('Abgebrochen. Nichts gespeichert.')
        return
    asset_type, symbol, name, amount, qty = order
    date = datetime.now().strftime('%d/%m/%Y')
    try:
        _append_transaction(asset_type, symbol, name, 'sell', amount, qty, date)
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
        await query.edit_message_text(f'Fehler: {error}\nNichts gespeichert.')
        return
    _cmdq(app).put({'cmd': 'reload_transactions'})
    await query.edit_message_text(f'Verkauf gespeichert: {name}\nBetrag: {amount:.2f} €\nMenge: {qty:.8g}\nDatum: {date}\n'
                                  'Das Hauptprogramm lädt die Transaktionen neu.')


async def _cmd_transactions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):
        return

    try:
        rows = cl.read_transactions(_TRANSACTIONS_PATH)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await _reply_long(update, f'Transaktionen konnten nicht gelesen werden: {error}')
        return

    if not rows:
        await update.message.reply_text("Noch keine Transaktionen gespeichert.")
        return

    lines = [f"Transaktionen: {len(rows)}"]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"{index}. {row.get('date', '')} | {row.get('symbol', '')} "
            f"({row.get('type', '')}) | {row.get('name', '')} | "
            f"{row.get('action', '').upper()} | "
            f"{row.get('amount', '')} EUR | Menge: {row.get('qty', '')}"
        )

    await _reply_long(update, "\n".join(lines))


async def _cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    if not context.args:
        await update.message.reply_text("❌ Symbol missing. Example: `/history bitcoin`", parse_mode="Markdown")
        return

    symbol = context.args[0].strip()
    try:
        rows = _get_history(symbol)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await _reply_long(update, f'Historie konnte nicht gelesen werden: {error}')
        return
    if not rows:
        await _reply_long(update, f'Keine Transaktionen für {symbol} gefunden.')
        return
    lines = [f'Historie: {symbol.upper()}']
    for row in rows:
        lines.append(f"{row['date']} | {row['type']} | {row['action'].upper()} | "
                     f"{row['amount']} EUR | Menge: {row['qty']}")
    await _reply_long(update, '\n'.join(lines))


# ────────────────────────────────────────────────
# Entry
# ────────────────────────────────────────────────
async def _post_init(app: Application) -> None:
    # In post_init läuft bereits ein Eventloop -> kein "no running loop"
    loop = asyncio.get_running_loop()
    app.bot_data["_notify_task"] = loop.create_task(_notify_worker(app))
    logging.info("Notify-Worker task gestartet")

    # Register the menu only for the configured chat.
    commands = [
        BotCommand("start", "Hilfe anzeigen"),
        BotCommand("help", "Alle Befehle und ihre Parameter anzeigen"),
        BotCommand("status", "Bot-Status und Anzahl der Transaktionen"),
        BotCommand("check", "Tagesanalyse ohne Alarm: Wert per Schaltfläche wählen"),
        BotCommand("ai", "KI-Frage zum Kopieren in ChatGPT oder Claude"),
        BotCommand("alarm", "Werte in die Alarm-Analyse aufnehmen oder entfernen"),
        BotCommand("analyse", "Analyse starten: hourly, daily oder summary"),
        BotCommand("buy", "Kauf speichern: Symbol Typ Betrag Menge Datum"),
        BotCommand("sell", "Verkauf speichern: Symbol Typ Betrag Menge"),
        BotCommand("sellall", "Ganzen Bestand verkaufen: Symbol [Betrag]"),
        BotCommand("history", "Transaktionen eines Symbols anzeigen"),
        BotCommand("transactions", "Alle gespeicherten Transaktionen anzeigen"),
        BotCommand("restart", "Programm neu starten"),
    ]
    try:
        await app.bot.set_my_commands(
            commands, scope=BotCommandScopeChat(chat_id=int(app.bot_data["chat_id"]))
        )
    except Exception:
        logging.exception("Telegram-Befehlsmenue konnte nicht aktualisiert werden")

    try:
        await app.bot.send_message(
            chat_id=app.bot_data["chat_id"],
            text="Programm gestartet.\n\n" + HELP_TEXT,
            parse_mode="Markdown",
        )
    except Exception:
        logging.exception("Telegram-Starthilfe konnte nicht gesendet werden")



async def _post_stop(app):
    task = app.bot_data.get('_notify_task')
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _on_error(update, context):
    logging.error('Fehler bei Telegram-Befehl', exc_info=context.error)
    if isinstance(update, Update) and _authorized(update, context.application.bot_data['chat_id']):
        await update.effective_message.reply_text('Der Befehl ist fehlgeschlagen. Bitte das Programmprotokoll prüfen.')


def run_standalone(notify_q: Queue, command_q: Queue) -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = _load_config()
    token = cfg["token"]
    chat_id = cfg["chat_id"]

    _ensure_csv_exists()

    builder = Application.builder().token(token).post_init(_post_init).post_stop(_post_stop)
    app = builder.build()

    app.bot_data["notify_q"] = notify_q
    app.bot_data["command_q"] = command_q
    app.bot_data["chat_id"] = chat_id

    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CommandHandler("status", _cmd_status))
    app.add_handler(CommandHandler("analyse", _cmd_analyse))
    app.add_handler(CommandHandler("check", _cmd_check))
    app.add_handler(CommandHandler("ai", _cmd_ai))
    app.add_handler(CommandHandler("alarm", _cmd_alarm))
    app.add_handler(CallbackQueryHandler(_cb_alarm, pattern=r"^alarm\|"))
    app.add_handler(CallbackQueryHandler(_cb_assets, pattern=r"^(check|ai)\|"))
    app.add_handler(CommandHandler("buy", _cmd_buy))
    app.add_handler(CommandHandler("sell", _cmd_sell))
    app.add_handler(CommandHandler("sellall", _cmd_sellall))
    app.add_handler(CallbackQueryHandler(_cb_sellall, pattern=r"^sellall\|"))
    app.add_handler(CommandHandler("history", _cmd_history))
    app.add_handler(CommandHandler("transactions", _cmd_transactions))
    app.add_handler(CommandHandler("restart", _cmd_restart))

    app.add_error_handler(_on_error)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info(f"Telegram Bot gestartet (chat_id={chat_id})")
    app.run_polling(stop_signals=None)
    logging.info("Telegram Bot beendet")


if __name__ == '__main__':
    print('Bitte main.py starten. telegram_bot.py wird vom Hauptprogramm aufgerufen.')
