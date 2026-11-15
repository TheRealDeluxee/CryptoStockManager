"""
telegram_bot.py
- Always-on Telegram bot
- Sends notifications from notify_queue (text/image) to Telegram
- Accepts commands (/help /status /analyse /check /ai /smart_alert /event /price_alert /buy /sell /sellall /history /restart)
  and sends them as dict via command_queue to main

Important:
- The bot is decoupled from main. No direct access to CryptoStockManager.
"""

import asyncio
import configparser
import csv
import logging
import os
import tempfile
import traceback
from queue import Empty
import classes as cl
import func
from datetime import datetime
from multiprocessing import Queue
from typing import Dict, List, Optional

from telegram import Update, BotCommand, BotCommandScopeChat, InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions
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
        raise FileNotFoundError("config.ini not found")
    token = cfg.get("telegram", "token")
    chat_id = cfg.get("telegram", "chat_id")
    return {"token": token, "chat_id": str(chat_id)}


def _authorized(update: Update, chat_id: str) -> bool:
    return update.effective_chat is not None and str(update.effective_chat.id) == str(chat_id)


def _priority_prefix(priority: int) -> str:
    # text instead of symbols; silent messages (-1/-2) without prefix
    return {1: "IMPORTANT · ", 2: "URGENT · "}.get(int(priority), "")


# ────────────────────────────────────────────────
# CSV helpers (as before)
# ────────────────────────────────────────────────
def _ensure_csv_exists() -> None:
    if not os.path.exists(_TRANSACTIONS_PATH):
        with open(_TRANSACTIONS_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_HEADER, delimiter=";")
            w.writeheader()


def _append_transaction(asset_type, symbol, name, action, amount, qty, date):
    # Write a complete new file first, then replace it atomically.
    # This way main.py can never read a half-written CSV line.
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
        raise ValueError(f"Symbol {symbol} is not in assets.ini in section [{section}].")
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
    # Telegram limits the length; emojis are counted correctly as well.
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
            # stop() alone does not end run_polling().
            while not app.running:
                await asyncio.sleep(0.1)
            app.stop_running()
            return
        if not isinstance(item, dict):
            continue
        try:
            text = _priority_prefix(item.get('priority', 0)) + str(item.get('message', ''))
            parse_mode = 'HTML' if item.get('html') else None  # analysis messages with bold/italic/monospace
            if item.get('type') == 'image':
                # fixed file (e.g. logo in the start message) or a chart from the plots folder
                path = item.get('path') or os.path.join(item.get('output_dir', 'plots'), item.get('symbol', 'plot') + '.png')
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Chart missing: {path}")
                # Send longer figures as a separate message, cut nothing off.
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
                # HTML that fits as a whole (visible text <= 4096) is never split: a cut <pre> block would be invalid
                fits = parse_mode == 'HTML' and func.telegram_length(text) <= 4096
                parts = [text] if fits else list(_text_chunks(text))   # list: number of parts is needed below
                # optional buttons (e.g. "AI prompt?" after /check) below the last part
                markup = None
                if item.get('buttons'):
                    markup = InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in row]
                                                   for row in item['buttons']])
                for number, part in enumerate(parts, 1):
                    # no link preview: keeps e.g. the start message with the GitHub link compact
                    await app.bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode,
                                               link_preview_options=LinkPreviewOptions(is_disabled=True),
                                               reply_markup=markup if number == len(parts) else None)
        except Exception as error:
            logging.exception('Telegram message could not be sent')
            func.log_print(f"Telegram message could not be sent: {type(error).__name__}: {error}", 'ERROR')
        await asyncio.sleep(0)


# ────────────────────────────────────────────────
# Commands -> command_queue
# ────────────────────────────────────────────────
def _cmdq(app: Application) -> Queue:
    return app.bot_data["command_q"]  # type: ignore[return-value]


# Telegram HTML: commands as plain text are tappable, examples in monospace.
HELP_TEXT = (
        "<b>Analysis</b>\n"
        "/check – daily analysis of one asset (choose via buttons)\n"
        "/ai – AI prompt to copy into ChatGPT or Claude\n"
        "/analyse – start daily, hourly or summary now\n"
        "\n"
        "<b>Alarms and dates</b>\n"
        "/smart_alert – switch assets on/off for the alarm analysis\n"
        "/event – show upcoming dates (with 🗑 to delete) or add a new one\n"
        "/price_alert – price alerts: show (with 🗑 to delete) or add one\n"
        "\n"
        "<b>Portfolio</b>\n"
        "/buy – record a buy\n"
        "/sell – record a sell\n"
        "/sellall – record the sale of the full position\n"
        "/history – transactions of one asset\n"
        "/transactions – all transactions\n"
        "\n"
        "<b>System</b>\n"
        "/status – bot running? number of transactions\n"
        "/restart – restart the program\n"
        "\n"
        "<b>Examples</b>\n"
        "<code>/buy QCOM stock 300 2 14/11/2026</code>\n"
        "<code>/sell bitcoin crypto 650 0.01</code>\n"
        "<code>/sellall qcom 312</code>\n"
        "<code>/check qcom stock</code>\n"
        "<code>/ai bitcoin</code>\n"
        "<code>/smart_alert goog on</code>\n"
        "<code>/event NVD.DE 17/11/2026 Q3 earnings</code>\n"
        "<code>/price_alert APC.DE 320.82 wait for pullback</code>\n"
        "<code>/analyse daily</code>"
)


async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="HTML")

async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _cmd_help(update, context)


async def _cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    try:
        tx_count = _count_transactions()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await _reply_long(update, f'Bot is running, but transactions cannot be read: {error}')
        return
    await update.message.reply_text(
        "<b>Bot running</b>\n"
        f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"Transactions saved: {tx_count}\nBot status only; running analyses may still be in progress.",
        parse_mode="HTML"
    )


async def _cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    # Without arguments: show the assets from assets.ini as buttons, tapping starts the analysis.
    if not context.args:
        await _reply_asset_buttons(update, 'check', 'Which daily analysis (without alarm)?')
        return
    if len(context.args) != 2 or context.args[1].lower() not in ('crypto', 'stock'):
        await update.message.reply_text('Format: /check (choose via buttons) or /check <symbol> <crypto|stock>')
        return
    symbol, asset_type = context.args[0].lower(), context.args[1].lower()
    try:
        _asset_name(symbol, asset_type)
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error: {error}')
        return
    # Like /analyse: the main program does the calculation.
    _cmdq(app).put({'cmd': 'check', 'symbol': symbol, 'type': asset_type})
    await update.message.reply_text(f'Daily analysis for {symbol} queued (without alarm).')


async def _reply_asset_buttons(update, cmd, question):
    # Assets from assets.ini as buttons; callback_data = "<cmd>|<type>|<symbol>" (for /check and /ai)
    try:
        entries = cl.read_assets()
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error: {error}')
        return
    buttons = [InlineKeyboardButton(('🪙 ' if asset_type == 'crypto' else '📈 ') + name,
                                    callback_data=f'{cmd}|{asset_type}|{symbol}')
               for (asset_type, symbol), name in entries.items()]
    keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]  # 2 buttons per row
    await update.message.reply_text(question, reply_markup=InlineKeyboardMarkup(keyboard))


_BUTTON_REPLIES = {'check': 'Daily analysis for {name} queued (without alarm).',
                   'ai': 'Creating AI prompt for {name} …'}


async def _cb_assets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Tap on a button from /check or /ai
    query = update.callback_query
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        await query.answer()
        return
    await query.answer()
    cmd, asset_type, symbol = query.data.split('|', 2)
    try:
        name = _asset_name(symbol, asset_type)  # assets.ini may have changed in the meantime
    except (OSError, ValueError, configparser.Error) as error:
        await query.edit_message_text(f'Error: {error}')
        return
    _cmdq(app).put({'cmd': cmd, 'symbol': symbol, 'type': asset_type})
    # Remove the buttons so that nothing is started twice by accident
    await query.edit_message_text(_BUTTON_REPLIES[cmd].format(name=name))


async def _cb_noai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # "No" on the question after /check: only remove the buttons
    query = update.callback_query
    await query.answer()
    if _authorized(update, context.application.bot_data['chat_id']):
        await query.edit_message_text('No AI prompt.')


async def _cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /ai: create a prompt for ChatGPT/Claude (text to copy, no API)
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    if not context.args:
        await _reply_asset_buttons(update, 'ai', 'AI prompt for which asset?')
        return
    if len(context.args) > 2 or (len(context.args) == 2 and context.args[1].lower() not in ('crypto', 'stock')):
        await update.message.reply_text('Format: /ai (choose via buttons) or /ai <symbol> [crypto|stock]')
        return
    symbol = context.args[0].lower()
    try:
        entries = cl.read_assets()
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error: {error}')
        return
    # The type may be omitted: it is looked up in assets.ini
    types = [context.args[1].lower()] if len(context.args) == 2 else ['crypto', 'stock']
    matches = [asset_type for asset_type in types if (asset_type, symbol) in entries]
    if len(matches) != 1:
        problem = 'not in assets.ini' if not matches else 'listed as crypto and stock, please add the type'
        await update.message.reply_text(f'Error: {symbol} is {problem}.')
        return
    _cmdq(app).put({'cmd': 'ai', 'symbol': symbol, 'type': matches[0]})
    await update.message.reply_text(_BUTTON_REPLIES['ai'].format(name=entries[(matches[0], symbol)]))


def _alarm_keyboard():
    # All assets from assets.ini with their current alarm state; tapping toggles
    entries = cl.read_assets()
    active = cl.read_assets(alarm_only=True)
    buttons = [InlineKeyboardButton(f"{name}: {'on' if key in active else 'off'}",
                                    callback_data=f'alarm|{key[0]}|{key[1]}')
               for key, name in entries.items()]
    return InlineKeyboardMarkup([buttons[i:i + 2] for i in range(0, len(buttons), 2)]), len(active)


async def _cmd_alarm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /smart_alert: add an asset to the automatic alarm analysis (1) or remove it (0)
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    try:
        if not context.args:
            keyboard, count = _alarm_keyboard()
            await update.message.reply_text(f'Alarm analysis: {count} assets on. Tap to toggle.',
                                            reply_markup=keyboard)
            return
        switch = {'on': True, '1': True, 'an': True, 'off': False, '0': False, 'aus': False}
        if len(context.args) != 2 or context.args[1].lower() not in switch:
            await update.message.reply_text('Format: /smart_alert (choose via buttons) or /smart_alert <symbol> <on|off>')
            return
        symbol, active = context.args[0].lower(), switch[context.args[1].lower()]
        entries = cl.read_assets()
        matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in entries]
        if len(matches) != 1:
            problem = 'not in assets.ini' if not matches else 'listed as crypto and stock, please use the buttons'
            await update.message.reply_text(f'Error: {symbol} is {problem}.')
            return
        cl.set_alarm(symbol, matches[0], active)
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error: {error}\nNot changed.')
        return
    _cmdq(app).put({'cmd': 'reload_assets'})
    name = entries[(matches[0], symbol)]
    await update.message.reply_text(f"{name}: alarm {'on' if active else 'off'}. The main program reloads.")


async def _cb_alarm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Tap on a button from /smart_alert: toggle the state, buttons stay for further changes
    query = update.callback_query
    app = context.application
    await query.answer()
    if not _authorized(update, app.bot_data['chat_id']):
        return
    _, asset_type, symbol = query.data.split('|', 2)
    try:
        name = _asset_name(symbol, asset_type)
        active = (asset_type, symbol) not in cl.read_assets(alarm_only=True)   # toggle
        cl.set_alarm(symbol, asset_type, active)
        keyboard, count = _alarm_keyboard()
    except (OSError, ValueError, configparser.Error) as error:
        await query.edit_message_text(f'Error: {error}\nNot changed.')
        return
    _cmdq(app).put({'cmd': 'reload_assets'})
    await query.edit_message_text(f"{name}: alarm {'on' if active else 'off'}.\n"
                                  f"Alarm analysis: {count} assets on. Tap to toggle.",
                                  reply_markup=keyboard)


async def _cmd_event(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /event: show upcoming dates; /event <symbol> <DD/MM/YYYY> <description>: add a date
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    try:
        if not context.args:
            today = datetime.now().date()
            upcoming = [event for event in cl.read_events() if event['date'] >= today]
            lines = ['Upcoming events:'] if upcoming else ['No upcoming events.']
            lines += [f"{event['date']:%d/%m/%Y}  {event['name']}" + (f" – {event['text']}" if event['text'] else '')
                      for event in upcoming]
            lines.append('\nAdd: /event <symbol> <DD/MM/YYYY> <description>')
            if upcoming:
                lines.append('Delete: tap 🗑 below')
            # One delete button per event (Telegram allows at most 64 bytes of button data)
            buttons = [[InlineKeyboardButton(f"🗑 {event['date']:%d/%m} {event['name']}",
                                             callback_data=f"evdel|{event['key']}")]
                       for event in upcoming if len(f"evdelok|{event['key']}".encode()) <= 64]
            await update.message.reply_text('\n'.join(lines)[:4000],
                                            reply_markup=InlineKeyboardMarkup(buttons) if buttons else None)
            return
        if len(context.args) < 2:
            await update.message.reply_text('Format: /event <symbol> <DD/MM/YYYY> <description>\n'
                                            'Example: /event NVD.DE 17/11/2026 Q3 earnings after market close')
            return
        key, name, date = cl.add_event(context.args[0], context.args[1], ' '.join(context.args[2:]))
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error: {error}\nNot saved.')
        return
    days_left = (date - datetime.now().date()).days
    if days_left == 1:
        _cmdq(app).put({'cmd': 'events'})   # event tomorrow: reminder and analysis immediately
        when = 'The event is tomorrow: reminder and daily analysis follow shortly.'
    elif days_left == 0:
        when = 'The event is today: no reminder (reminders come one day before).'
    else:
        when = 'Reminder with daily analysis one day before the event at 08:00.'
    await update.message.reply_text(f"Event saved: {date:%d/%m/%Y} {name}\n{when}")


async def _cb_event_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # 🗑 in the /event list: first ask, then delete (protects against an accidental tap)
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    action, _, key = query.data.partition('|')
    if action == 'evdelno':
        await query.edit_message_text('Not deleted.')
        return
    try:
        event = next((e for e in cl.read_events() if e['key'] == key), None)
        if action == 'evdel':
            if event is None:
                await query.edit_message_text('This event no longer exists.')
                return
            text = f" – {event['text']}" if event['text'] else ''
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton('Yes, delete', callback_data=f'evdelok|{key}'),
                                              InlineKeyboardButton('Cancel', callback_data='evdelno|')]])
            await query.edit_message_text(f"Delete this event?\n{event['date']:%d/%m/%Y} {event['name']}{text}",
                                          reply_markup=keyboard)
            return
        cl.delete_event(key)   # action == 'evdelok'
    except (OSError, ValueError, configparser.Error) as error:
        await query.edit_message_text(f'Error: {error}\nNot deleted.')
        return
    name = f"{event['date']:%d/%m/%Y} {event['name']}" if event else key
    await query.edit_message_text(f'Event deleted: {name}')


async def _cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /price_alert: show price alerts (with 🗑 to delete); /price_alert <symbol> [below|above] <price> [note]: add one
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    args = context.args or []
    try:
        if not args:
            alerts = cl.read_value_alerts()
            lines = ['Price alerts:'] if alerts else ['No price alerts.']
            lines += [f"{alert['name']} {alert['direction']} {func.format_euro(alert['price'])}"
                      + (f" – {alert['note']}" if alert['note'] else '') for alert in alerts]
            lines.append('\nAdd: /price_alert <symbol> <price> <note>')
            if alerts:
                lines.append('Delete: tap 🗑 below')
            buttons = [[InlineKeyboardButton(f"🗑 {alert['name']} {alert['direction']} {alert['price']:.2f} €",
                                             callback_data=f"vadel|{alert['key']}")]
                       for alert in alerts if len(f"vadelok|{alert['key']}".encode()) <= 64]
            await update.message.reply_text('\n'.join(lines)[:4000],
                                            reply_markup=InlineKeyboardMarkup(buttons) if buttons else None)
            return
        direction = args[1].lower() if len(args) > 1 and args[1].lower() in ('below', 'above') else None
        rest = args[2:] if direction else args[1:]
        if not rest:
            await update.message.reply_text('Format: /price_alert <symbol> [below|above] <price> <note>\n'
                                            'Example: /price_alert APC.DE 320.82 Wait for pullback, buy small amount')
            return
        symbol, price = args[0].lower(), cl.parse_price(rest[0])
        entries = cl.read_assets()
        matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in entries]
        if len(matches) != 1:
            raise ValueError(f"{symbol} is {'not in assets.ini' if not matches else 'listed as crypto and stock'}.")
        asset_type = matches[0]
        # Current price: decides "below"/"above" if not given and shows the distance to the mark
        try:
            now = await asyncio.to_thread(func.current_price, symbol.upper() if asset_type == 'stock' else symbol,
                                          asset_type)
        except Exception:
            now = None
        if direction is None:
            if now is None:
                await update.message.reply_text('The current price is not available right now. '
                                                'Please add below or above, e.g. /price_alert APC.DE below 320.82')
                return
            direction = 'below' if price <= now else 'above'
        key, name, _ = cl.add_value_alert(symbol, direction, price, ' '.join(rest[1:]))
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error: {error}\nNot saved.')
        return
    text = f"Price alert saved: {name} {direction} {func.format_euro(price)}"
    if now is not None:
        text += f"\nCurrent price {func.format_euro(now)} ({(price / now - 1) * 100:+.1f} % to the mark)."
        if (direction == 'below' and now <= price) or (direction == 'above' and now >= price):
            _cmdq(app).put({'cmd': 'value_alerts'})   # mark already reached: report right away
            text += '\nThe mark is already reached: the alert follows shortly.'
    text += '\nChecked every hour (stocks Mon–Fri 08:00–22:00).'
    await update.message.reply_text(text)


async def _cb_value_alert_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # 🗑 in the /price_alert list: first ask, then delete
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    action, _, key = query.data.partition('|')
    if action == 'vadelno':
        await query.edit_message_text('Not deleted.')
        return
    try:
        alert = next((a for a in cl.read_value_alerts() if a['key'] == key), None)
        if action == 'vadel':
            if alert is None:
                await query.edit_message_text('This price alert no longer exists (already triggered or deleted).')
                return
            note = f" – {alert['note']}" if alert['note'] else ''
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton('Yes, delete', callback_data=f'vadelok|{key}'),
                                              InlineKeyboardButton('Cancel', callback_data='vadelno|')]])
            await query.edit_message_text(f"Delete this price alert?\n{alert['name']} {alert['direction']} "
                                          f"{func.format_euro(alert['price'])}{note}", reply_markup=keyboard)
            return
        cl.delete_value_alert(key)   # action == 'vadelok'
    except (OSError, ValueError, configparser.Error) as error:
        await query.edit_message_text(f'Error: {error}\nNot deleted.')
        return
    label = f"{alert['name']} {alert['direction']} {func.format_euro(alert['price'])}" if alert else key
    await query.edit_message_text(f'Price alert deleted: {label}')


async def _cmd_analyse(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    arg = context.args[0].lower() if context.args else "hourly"
    if arg not in ("hourly", "daily", "summary"):
        await update.message.reply_text("Unknown argument. Use: /analyse hourly|daily|summary")
        return

    _cmdq(app).put({"cmd": "analyse", "mode": arg})
    await update.message.reply_text(f"Analysis queued: {arg}")


async def _cmd_restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    _cmdq(app).put({"cmd": "restart_main"})
    await update.message.reply_text("Restarting the main program …")


async def _save_trade(update, context, action):
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    expected = 5 if action == 'buy' else 4
    if len(context.args) != expected:
        date_help = ' <DD/MM/YYYY>' if action == 'buy' else ''
        await update.message.reply_text(f'Format: /{action} <symbol> <crypto|stock> <amount> <quantity>{date_help}')
        return
    symbol, asset_type, amount_text, qty_text = context.args[:4]
    asset_type = asset_type.lower()
    if asset_type not in ('crypto', 'stock'):
        await update.message.reply_text('Error: type must be crypto or stock. Not saved.')
        return
    symbol = symbol.lower() if asset_type == 'crypto' else symbol.upper()
    date = context.args[4] if action == 'buy' else datetime.now().strftime('%d/%m/%Y')
    try:
        if not _valid_date(date):
            raise ValueError('Invalid date or date in the future. Format: DD/MM/YYYY.')
        amount, qty = cl.valid_numbers(amount_text, qty_text)
        name = _asset_name(symbol, asset_type)
        _append_transaction(asset_type, symbol, name, action, amount, qty, date)
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
        await _reply_long(update, f'Error: {error}\nTransaction not saved.')
        return
    _cmdq(app).put({'cmd': 'reload_transactions'})
    label = 'Buy' if action == 'buy' else 'Sell'
    await update.message.reply_text(
        f'{label} saved: {symbol}\nAmount: {amount:.2f} €\nQuantity: {qty}\nDate: {date}\n'
        'The main program reloads the transactions.', parse_mode=None)


async def _cmd_buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _save_trade(update, context, 'buy')


async def _cmd_sell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _save_trade(update, context, 'sell')


def _holding(symbol, asset_type, name):
    # Current holdings according to transactions.csv (FIFO as in the main program)
    asset = cl.apply_transactions(cl.CryptoStock(symbol, name, asset_type), cl.read_transactions(_TRANSACTIONS_PATH))
    return asset.state.amount_crypto_stock


def _current_price(symbol, asset_type, name):
    # Latest price in EUR; blocks during API retries -> only call via asyncio.to_thread
    df = cl.CryptoStock(symbol, name, asset_type).get_API_data('100d')
    if df is None or df.empty:
        raise ValueError('No current price available. Please add the amount: /sellall <symbol> <amount>')
    return float(df['Price'].iloc[-1])


async def _cmd_sellall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /sellall <symbol> [amount]: record the full position as a sale (after confirmation)
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    if len(context.args) not in (1, 2):
        await update.message.reply_text('Format: /sellall <symbol> [amount in €]\nWithout an amount the current price is used.')
        return
    symbol = context.args[0].lower()
    try:
        entries = cl.read_assets()
        matches = [asset_type for asset_type in ('crypto', 'stock') if (asset_type, symbol) in entries]
        if len(matches) != 1:
            raise ValueError(f"{symbol} is {'not in assets.ini' if not matches else 'listed as crypto and stock'}.")
        asset_type = matches[0]
        name = entries[(asset_type, symbol)]
        symbol = symbol.lower() if asset_type == 'crypto' else symbol.upper()
        qty = _holding(symbol, asset_type, name)
        if qty <= 0:
            raise ValueError(f'No holdings of {name}.')
        if len(context.args) == 2:
            amount, _ = cl.valid_numbers(context.args[1], qty)
            source = 'your amount'
        else:
            await update.message.reply_text(f'Fetching the current price for {name} …')
            price = await asyncio.to_thread(_current_price, symbol, asset_type, name)
            amount = round(price * qty, 2)
            source = f'current price {price:.2f} €'
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
        await _reply_long(update, f'Error: {error}\nNothing saved.')
        return

    # Save only after confirmation; until then the order is kept in the bot
    pending_id = os.urandom(4).hex()
    app.bot_data.setdefault('sellall', {})[pending_id] = (asset_type, symbol, name, amount, qty)
    unit = 'coins' if asset_type == 'crypto' else 'shares'
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton('Save sale', callback_data=f'sellall|{pending_id}|ja'),
                                      InlineKeyboardButton('Cancel', callback_data=f'sellall|{pending_id}|nein')]])
    await update.message.reply_text(
        f"Record the full position of {name} as a sale?\n"
        f"Quantity: {qty:.8g} {unit}\nAmount: {amount:.2f} € ({source})\n"
        f"Date: {datetime.now().strftime('%d/%m/%Y')}\n"
        f"If your proceeds were different: /sellall {symbol} <amount>",
        reply_markup=keyboard)


async def _cb_sellall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Confirmation from /sellall
    query = update.callback_query
    app = context.application
    await query.answer()
    if not _authorized(update, app.bot_data['chat_id']):
        return
    _, pending_id, answer = query.data.split('|', 2)
    order = app.bot_data.get('sellall', {}).pop(pending_id, None)
    if order is None:
        await query.edit_message_text('Order no longer available (already done or bot restarted). Nothing saved.')
        return
    if answer != 'ja':
        await query.edit_message_text('Cancelled. Nothing saved.')
        return
    asset_type, symbol, name, amount, qty = order
    date = datetime.now().strftime('%d/%m/%Y')
    try:
        _append_transaction(asset_type, symbol, name, 'sell', amount, qty, date)
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
        await query.edit_message_text(f'Error: {error}\nNothing saved.')
        return
    _cmdq(app).put({'cmd': 'reload_transactions'})
    await query.edit_message_text(f'Sale saved: {name}\nAmount: {amount:.2f} €\nQuantity: {qty:.8g}\nDate: {date}\n'
                                  'The main program reloads the transactions.')


async def _cmd_transactions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):
        return

    try:
        rows = cl.read_transactions(_TRANSACTIONS_PATH)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await _reply_long(update, f'Transactions could not be read: {error}')
        return

    if not rows:
        await update.message.reply_text("No transactions saved yet.")
        return

    lines = [f"Transactions: {len(rows)}"]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"{index}. {row.get('date', '')} | {row.get('symbol', '')} "
            f"({row.get('type', '')}) | {row.get('name', '')} | "
            f"{row.get('action', '').upper()} | "
            f"{row.get('amount', '')} EUR | quantity: {row.get('qty', '')}"
        )

    await _reply_long(update, "\n".join(lines))


async def _cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    if not context.args:
        await update.message.reply_text("Symbol missing. Example: /history bitcoin")
        return

    symbol = context.args[0].strip()
    try:
        rows = _get_history(symbol)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await _reply_long(update, f'History could not be read: {error}')
        return
    if not rows:
        await _reply_long(update, f'No transactions found for {symbol}.')
        return
    lines = [f'History: {symbol.upper()}']
    for row in rows:
        lines.append(f"{row['date']} | {row['type']} | {row['action'].upper()} | "
                     f"{row['amount']} EUR | quantity: {row['qty']}")
    await _reply_long(update, '\n'.join(lines))


# ────────────────────────────────────────────────
# Entry
# ────────────────────────────────────────────────
async def _post_init(app: Application) -> None:
    # An event loop is already running in post_init -> no "no running loop"
    loop = asyncio.get_running_loop()
    app.bot_data["_notify_task"] = loop.create_task(_notify_worker(app))
    logging.info("Notify worker task started")

    # Register the menu only for the configured chat.
    commands = [
        BotCommand("start", "Show help"),
        BotCommand("help", "Show all commands and their parameters"),
        BotCommand("status", "Bot status and number of transactions"),
        BotCommand("check", "Daily analysis without alarm: choose asset via buttons"),
        BotCommand("ai", "AI prompt to copy into ChatGPT or Claude"),
        BotCommand("smart_alert", "Add assets to or remove them from the alarm analysis"),
        BotCommand("event", "Show or add dates: symbol date text"),
        BotCommand("price_alert", "Price alerts: show, or add: symbol price note"),
        BotCommand("analyse", "Start analysis: hourly, daily or summary"),
        BotCommand("buy", "Record a buy: symbol type amount quantity date"),
        BotCommand("sell", "Record a sell: symbol type amount quantity"),
        BotCommand("sellall", "Sell the full position: symbol [amount]"),
        BotCommand("history", "Show transactions of a symbol"),
        BotCommand("transactions", "Show all saved transactions"),
        BotCommand("restart", "Restart the program"),
    ]
    try:
        await app.bot.set_my_commands(
            commands, scope=BotCommandScopeChat(chat_id=int(app.bot_data["chat_id"]))
        )
    except Exception:
        logging.exception("Telegram command menu could not be updated")
    # No separate start message: the main program announces the start (version, assets, dates).



async def _post_stop(app):
    task = app.bot_data.get('_notify_task')
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _on_error(update, context):
    logging.error('Error in Telegram command', exc_info=context.error)
    error = context.error
    # Also in logfile.txt (the console window is often not visible), with the full traceback
    details = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    command = update.effective_message.text if isinstance(update, Update) and update.effective_message else ''
    try:
        func.log_print(f"Telegram command {command!r} failed:\n{details}", 'ERROR')
    except OSError:
        pass
    if isinstance(update, Update) and update.effective_message and _authorized(update, context.application.bot_data['chat_id']):
        # Name the cause directly so that it can be fixed without looking into the log
        await update.effective_message.reply_text(
            f'The command failed: {type(error).__name__}: {str(error)[:300]}\nDetails in logfile.txt.')


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
    app.add_handler(CommandHandler(["smart_alert", "alarm"], _cmd_alarm))   # /alarm still works as before
    app.add_handler(CommandHandler(["event", "termin"], _cmd_event))   # /termin still works as before
    app.add_handler(CommandHandler("price_alert", _cmd_price))
    app.add_handler(CallbackQueryHandler(_cb_alarm, pattern=r"^alarm\|"))
    app.add_handler(CallbackQueryHandler(_cb_assets, pattern=r"^(check|ai)\|"))
    app.add_handler(CallbackQueryHandler(_cb_noai, pattern=r"^noai$"))
    app.add_handler(CallbackQueryHandler(_cb_event_delete, pattern=r"^evdel(ok|no)?\|"))
    app.add_handler(CallbackQueryHandler(_cb_value_alert_delete, pattern=r"^vadel(ok|no)?\|"))
    app.add_handler(CommandHandler("buy", _cmd_buy))
    app.add_handler(CommandHandler("sell", _cmd_sell))
    app.add_handler(CommandHandler("sellall", _cmd_sellall))
    app.add_handler(CallbackQueryHandler(_cb_sellall, pattern=r"^sellall\|"))
    app.add_handler(CommandHandler("history", _cmd_history))
    app.add_handler(CommandHandler("transactions", _cmd_transactions))
    app.add_handler(CommandHandler("restart", _cmd_restart))

    app.add_error_handler(_on_error)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info(f"Telegram bot started (chat_id={chat_id})")
    app.run_polling(stop_signals=None)
    logging.info("Telegram bot stopped")


if __name__ == '__main__':
    print('Please start main.py. telegram_bot.py is started by the main program.')
