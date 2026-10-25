"""
telegram_bot.py
- Always-on Telegram Bot
- Sendet Notifications aus notify_queue (Text/Bild) an Telegram
- Nimmt Commands entgegen (/help /status /analyse /buy /sell /history /restart)
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

from telegram import Update, BotCommand, BotCommandScopeChat
from telegram.ext import Application, CommandHandler, ContextTypes

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
    return {-2: "🔕 ", -1: "🔕 ", 0: "", 1: "⚠️ ", 2: "🚨 "}.get(int(priority), "")


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


def _watchlist_name(symbol, asset_type):
    entries = cl.read_watchlist()
    key = (asset_type, symbol.lower())
    if key not in entries:
        section = 'crypto' if asset_type == 'crypto' else 'stocks'
        raise ValueError(f"Symbol {symbol} ist nicht in watchlist.ini im Abschnitt [{section}] vorhanden.")
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
            if item.get('type') == 'image':
                path = os.path.join(item.get('output_dir', 'plots'), item.get('symbol', 'plot') + '.png')
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Diagramm fehlt: {path}")
                # Längere Kennwerte als eigene Nachricht senden, nichts abschneiden.
                caption_fits = len(text.encode('utf-16-le')) // 2 <= 1000
                with open(path, 'rb') as photo:
                    await app.bot.send_photo(chat_id=chat_id, photo=photo,
                                             caption=text if caption_fits else None, parse_mode=None)
                if not caption_fits:
                    for part in _text_chunks(text):
                        await app.bot.send_message(chat_id=chat_id, text=part, parse_mode=None)
                if item.get('delete_after_send'):
                    os.remove(path)
            elif item.get('type') == 'text':
                for part in _text_chunks(text):
                    await app.bot.send_message(chat_id=chat_id, text=part, parse_mode=None)
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
        "`/check <symbol> <crypto|stock>`  (daily ohne Alarm)\n"
        "`/buy <symbol> <crypto|stock> <amount> <qty> <DD/MM/YYYY>`\n"
        "`/sell <symbol> <crypto|stock> <amount> <qty>`\n"
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
    if len(context.args) != 2 or context.args[1].lower() not in ('crypto', 'stock'):
        await update.message.reply_text('Format: /check <symbol> <crypto|stock> (nur daily)')
        return
    symbol, asset_type = context.args[0].lower(), context.args[1].lower()
    try:
        _watchlist_name(symbol, asset_type)
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Fehler: {error}')
        return
    # Wie /analyse: Das Hauptprogramm übernimmt die Berechnung.
    _cmdq(app).put({'cmd': 'check', 'symbol': symbol, 'type': asset_type})
    await update.message.reply_text(f'Tagesanalyse für {symbol} vorgemerkt (ohne Alarm).')


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
        name = _watchlist_name(symbol, asset_type)
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
        BotCommand("check", "Einzelne Tagesanalyse ohne Alarm: Symbol crypto|stock"),
        BotCommand("analyse", "Analyse starten: hourly, daily oder summary"),
        BotCommand("buy", "Kauf speichern: Symbol Typ Betrag Menge Datum"),
        BotCommand("sell", "Verkauf speichern: Symbol Typ Betrag Menge"),
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
    app.add_handler(CommandHandler("buy", _cmd_buy))
    app.add_handler(CommandHandler("sell", _cmd_sell))
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
