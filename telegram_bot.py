"""
telegram_bot.py
- Always-on Telegram bot
- Sends notifications from notify_queue (text/image) to Telegram
- Accepts commands (/help /status /analyse /check /ai_prompt /ai_api_decision3s /ai_api_plan /smart_alert /event /plan_alert /record /history /restart)
  and sends them as dict via command_queue to main

Important:
- The bot is decoupled from main. No direct access to CryptoStockManager.
"""

import asyncio
import configparser
import csv
import html
import logging
import os
import tempfile
import traceback
import types
from queue import Empty
import classes as cl
import func
from datetime import datetime, timedelta
from multiprocessing import Queue
from typing import Dict

from telegram import Update, BotCommand, BotCommandScopeChat, InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.error import BadRequest, Conflict, NetworkError

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
    _write_transactions(rows)
    return row


def _undo_transaction(row):
    """Remove the last transaction again (↩ Undo after /buy or /sell), only if it is still the given one."""
    rows = cl.read_transactions(_TRANSACTIONS_PATH)
    if not rows:
        raise ValueError('No transactions saved.')
    last = rows[-1]
    same = (last['type'] == row['type'] and last['symbol'].lower() == str(row['symbol']).lower()
            and last['action'] == row['action'] and last['date'] == row['date']
            and cl.valid_numbers(last['amount'], last['qty']) == cl.valid_numbers(row['amount'], row['qty']))
    if not same:
        raise ValueError('The last transaction is a different one (saved later?). Nothing undone.')
    _write_transactions(rows[:-1])


def _write_transactions(rows):
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


def _get_history(text):
    # By symbol or name; also for assets that are no longer in assets.ini (name as saved in transactions.csv)
    rows = cl.read_transactions(_TRANSACTIONS_PATH)
    wanted = ' '.join(text.lower().split())
    found = [row for row in rows if wanted in (row['symbol'].lower(), ' '.join((row.get('name') or '').lower().split()))]
    if found:
        return found
    try:
        _, symbol, _, _ = _find_asset(text.split())   # e.g. only the beginning of the name: "Trane"
    except (OSError, ValueError, configparser.Error):
        return []
    return [row for row in rows if row['symbol'].lower() == symbol]


def _asset_name(symbol, asset_type):
    entries = cl.read_assets()
    key = (asset_type, symbol.lower())
    if key not in entries:
        section = 'crypto' if asset_type == 'crypto' else 'stocks'
        raise ValueError(f"Symbol {symbol} is not in assets.ini in section [{section}].")
    return entries[key]


_TYPES = ('crypto', 'stock')


def _find_asset(words, types=_TYPES, allow_rest=False):
    # Asset by symbol or name from assets.ini (e.g. RHM.DE or Rheinmetall); returns (type, symbol, name, rest)
    asset_type, symbol, name, rest = cl.take_asset(words, cl.read_assets(), types)
    if rest and not allow_rest:
        raise ValueError(f"Unexpected text after {name}: {' '.join(rest)}")
    return asset_type, symbol, name, rest


def _split_type(words):
    # Optional crypto|stock as the last word: returns (words without it, allowed types)
    if words and words[-1].lower() in _TYPES:
        return words[:-1], (words[-1].lower(),)
    return words, _TYPES


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


async def _send_text(app, chat_id, text, parse_mode, markup=None):
    """Text message; HTML that fits (visible text <= 4096) is sent whole. Longer HTML cannot be split safely
    (a cut <pre> or <b> would be rejected by Telegram): it is sent as plain text in several parts."""
    if parse_mode == 'HTML' and func.telegram_length(text) > 4096:
        text, parse_mode = func._strip_telegram_html(text), None
    parts = [text] if parse_mode == 'HTML' else list(_text_chunks(text))   # list: number of parts is needed below
    for number, part in enumerate(parts, 1):
        # no link preview: keeps e.g. the start message with the GitHub link compact
        options = dict(chat_id=chat_id, link_preview_options=LinkPreviewOptions(is_disabled=True),
                       reply_markup=markup if number == len(parts) else None)
        try:
            await app.bot.send_message(text=part, parse_mode=parse_mode, **options)
        except BadRequest as error:
            if parse_mode != 'HTML':
                raise
            # Telegram rejected the formatting: send the same text without it instead of losing it
            func.log_print(f"Telegram HTML rejected ({error}), sent as plain text", 'WARN')
            for plain in _text_chunks(func._strip_telegram_html(part)):
                await app.bot.send_message(text=plain, **options)


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
            # optional buttons (e.g. "AI prompt?" after /check, "🤖 AI prompt" under alarms) below the message
            markup = None
            if item.get('buttons'):
                markup = InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in row]
                                               for row in item['buttons']])
            if item.get('type') == 'image':
                # fixed file (e.g. logo in the start message) or a chart from the plots folder
                path = item.get('path') or os.path.join(item.get('output_dir', 'plots'), item.get('symbol', 'plot') + '.png')
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Chart missing: {path}")
                try:
                    # Telegram allows 1024 visible characters as caption (HTML tags do not count);
                    # longer texts follow as a separate message, nothing is cut off.
                    caption_fits = func.telegram_length(text, parse_mode == 'HTML') <= 1024
                    try:
                        with open(path, 'rb') as photo:
                            await app.bot.send_photo(chat_id=chat_id, photo=photo, caption=text if caption_fits else None,
                                                     parse_mode=parse_mode, reply_markup=markup if caption_fits else None)
                    except BadRequest as error:
                        if not (caption_fits and parse_mode == 'HTML'):
                            raise
                        # formatting of the caption rejected: chart without caption, text follows as plain text
                        func.log_print(f"Telegram HTML caption rejected ({error}), sent as plain text", 'WARN')
                        with open(path, 'rb') as photo:
                            await app.bot.send_photo(chat_id=chat_id, photo=photo)
                        caption_fits, text, parse_mode = False, func._strip_telegram_html(text), None
                    if not caption_fits:
                        await _send_text(app, chat_id, text, parse_mode, markup)
                finally:
                    # also when sending failed: otherwise the plots folder slowly fills up
                    if item.get('delete_after_send') and os.path.exists(path):
                        os.remove(path)
            elif item.get('type') == 'text':
                await _send_text(app, chat_id, text, parse_mode, markup)
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
        "🤖 AI prompt, AI decision (API), AI plan (API) – buttons below every alarm, info and /check "
        "(AI decision and AI plan with a Claude API key)\n"
        "/analyse – start daily, hourly or summary now\n"
        "\n"
        "<b>Alarms and dates</b>\n"
        "/smart_alert – switch assets on/off for the alarm analysis\n"
        "/smart_alert_config – alarm, info or off per signal (stocks, crypto)\n"
        "/event – show upcoming dates (with 🗑 to delete) or add a new one\n"
        "/plan_alert – trading plans (buy zone, target, stop): show (with 🗑 to delete) or add one\n"
        "\n"
        "<b>Portfolio</b>\n"
        "/portfolio – daily P&amp;L table and 13-week chart now\n"
        "/record – record a buy, sell or sale of the full position (buttons, or in one line)\n"
        "/history – transactions of one asset\n"
        "/transactions – all transactions\n"
        "\n"
        "<b>System</b>\n"
        "/status – analyses, next runs, problems\n"
        "/update – look for a new CaSa version now (otherwise daily at 19:00) and install it\n"
        "/restart – restart the program\n"
        "\n"
        "<b>Examples</b> (symbol or name from assets.ini)\n"
        "<code>/record buy Qualcomm 300 2 14/11/2026</code>\n"
        "<code>/record sell bitcoin 650 0.01</code>\n"
        "<code>/record sellall Qualcomm 312</code>\n"
        "<code>/check Rheinmetall</code>\n"

        "<code>/smart_alert goog on</code>\n"
        "<code>/event Nvidia 17/11/2026 Q3 earnings</code>\n"
        "<code>/plan_alert TSMC 312.15 2 358.93 312.15 1/4 after a rebound</code>\n"
        "<code>/analyse daily</code>"
)


async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="HTML")


async def _cmd_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /update: look for a newer CaSa release on GitHub now; the answer with the button comes from the main program
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):
        return
    _cmdq(app).put({'cmd': 'check_update'})
    await update.message.reply_text('Looking for a new CaSa version on GitHub …')


async def _cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    # The bot answers at once; the details (analyses, next runs, problems) come from the main program,
    # during a running analysis only after it.
    _cmdq(app).put({'cmd': 'status'})
    await update.message.reply_text(f"Bot running ({datetime.now():%d/%m/%Y %H:%M}). "
                                    "Status of the main program follows (after a running analysis).")


async def _cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /portfolio: daily P&L table and 13-week chart now instead of only with the evening summary
    if not _authorized(update, context.application.bot_data["chat_id"]):
        return
    _cmdq(context.application).put({'cmd': 'portfolio'})
    await update.message.reply_text('Creating P&L table and chart …')


async def _cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    # Without arguments: show the assets from assets.ini as buttons, tapping starts the analysis.
    if not context.args:
        await _reply_asset_buttons(update, 'check', 'Which daily analysis (without alarm)?')
        return
    try:
        asset_type, symbol, name, _ = _find_asset(*_split_type(list(context.args)))
    except (OSError, ValueError, configparser.Error) as error:
        # unknown asset: offer the assets as buttons instead of only the format
        await _reply_asset_buttons(update, 'check', f'{error}\nWhich daily analysis (without alarm)?')
        return
    # Like /analyse: the main program does the calculation.
    _cmdq(app).put({'cmd': 'check', 'symbol': symbol, 'type': asset_type})
    await update.message.reply_text(_BUTTON_REPLIES['check'].format(name=name))


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
                   'ai': 'Creating AI prompt for {name} …',
                   'ai_decision': 'Asking Claude for a decision on {name} … (about 1 minute)',
                   'ai_plan': 'Asking Claude for a trading plan for {name} … (about 1 minute)'}
_AI_QUESTIONS = {'ai': 'AI prompt for which asset?', 'ai_decision': 'Claude decision for which asset?',
                 'ai_plan': 'Claude trading plan for which asset?'}


def _claude_key_set():
    # The Claude API runs in the main program; the bot only checks that a key is there
    cfg = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'))
    cfg.read('config.ini', encoding='utf-8-sig')
    return bool(cfg.get('anthropic', 'api_key', fallback='').strip())


async def _cb_assets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Tap on a button from /check or /ai
    query = update.callback_query
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        await query.answer()
        return
    await query.answer()
    cmd, asset_type, symbol = query.data.split('|', 2)
    if cmd in ('ai_decision', 'ai_plan') and not _claude_key_set():
        await query.edit_message_text(func.CLAUDE_NO_KEY)
        return
    try:
        name = _asset_name(symbol, asset_type)  # assets.ini may have changed in the meantime
    except (OSError, ValueError, configparser.Error) as error:
        await query.edit_message_text(f'Error: {error}')
        return
    _cmdq(app).put({'cmd': cmd, 'symbol': symbol, 'type': asset_type})
    # Remove the buttons so that nothing is started twice by accident
    await query.edit_message_text(_BUTTON_REPLIES[cmd].format(name=name))


_AI_BUTTONS = {'aibtn': 'ai', 'aidbtn': 'ai_decision', 'aipbtn': 'ai_plan'}


async def _cb_ai_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # "🤖 AI prompt", "🤖 AI decision (API)", "🤖 AI plan (API)" below an alarm or info: the alarm message itself stays unchanged
    query = update.callback_query
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        await query.answer()
        return
    button, asset_type, symbol = query.data.split('|', 2)
    cmd = _AI_BUTTONS[button]
    if cmd != 'ai' and not _claude_key_set():   # key removed since the alarm was sent
        await query.answer(func.CLAUDE_NO_KEY[:200], show_alert=True)
        return
    try:
        name = _asset_name(symbol, asset_type)
    except (OSError, ValueError, configparser.Error) as error:
        await query.answer(f'Error: {error}'[:200], show_alert=True)
        return
    await query.answer('Creating AI prompt …' if cmd == 'ai' else 'Asking Claude …')
    _cmdq(app).put({'cmd': cmd, 'symbol': symbol, 'type': asset_type})
    await query.message.reply_text(_BUTTON_REPLIES[cmd].format(name=name))


async def _cb_noai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # "No" on the AI question after /check in messages from before 0.1.12: only remove the buttons
    query = update.callback_query
    await query.answer()
    if _authorized(update, context.application.bot_data['chat_id']):
        await query.edit_message_text('No AI prompt.')


async def _run_ai(update: Update, context: ContextTypes.DEFAULT_TYPE, cmd) -> None:
    # /ai_prompt (cmd 'ai'): prompt to copy, no API; /ai_api_decision3s and /ai_api_plan: answer from the Claude API
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    if cmd != 'ai' and not _claude_key_set():
        await update.message.reply_text(func.CLAUDE_NO_KEY)
        return
    if not context.args:
        await _reply_asset_buttons(update, cmd, _AI_QUESTIONS[cmd])
        return
    try:
        # The type may be omitted: it is looked up in assets.ini
        asset_type, symbol, name, _ = _find_asset(*_split_type(list(context.args)))
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_asset_buttons(update, cmd, f'{error}\n{_AI_QUESTIONS[cmd]}')
        return
    _cmdq(app).put({'cmd': cmd, 'symbol': symbol, 'type': asset_type})
    await update.message.reply_text(_BUTTON_REPLIES[cmd].format(name=name))


async def _cmd_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_ai(update, context, 'ai')


async def _cmd_ai_decision(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_ai(update, context, 'ai_decision')


async def _cmd_ai_plan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _run_ai(update, context, 'ai_plan')


async def _cb_ai_plan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # "✅ Save as plan" / "Discard" below an /ai_api_plan answer: the main program keeps the proposal
    query = update.callback_query
    app = context.application
    await query.answer()
    if not _authorized(update, app.bot_data['chat_id']):
        return
    action, _, key = query.data.partition('|')
    _cmdq(app).put({'cmd': 'ai_plan_save', 'key': key, 'save': action == 'aiplanok'})
    await query.edit_message_reply_markup(reply_markup=None)   # no second tap


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
        try:
            if len(context.args) < 2 or context.args[-1].lower() not in switch:
                raise ValueError('Format: /smart_alert <symbol or name> <on|off>')
            active = switch[context.args[-1].lower()]
            asset_type, symbol, name, _ = _find_asset(*_split_type(list(context.args[:-1])))
        except ValueError as error:
            # wrong input: show the buttons to toggle instead
            keyboard, count = _alarm_keyboard()
            await update.message.reply_text(f'{error}\nAlarm analysis: {count} assets on. Tap to toggle.',
                                            reply_markup=keyboard)
            return
        cl.set_alarm(symbol, asset_type, active)
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error: {error}\nNot changed.')
        return
    _cmdq(app).put({'cmd': 'reload_assets'})
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


_KIND_ICON = {'alarm': '🔔', 'info': 'ℹ️', 'off': '–'}
_ALERT_CONFIG_TEXT = (
    "<b>Alarm and info settings</b> (📈 stocks, 🪙 crypto)\n"
    "🔔 <b>alarm</b>: \"Potential buy/sell\", can be important, counted in the alarm journal\n"
    "ℹ️ <b>info</b>: \"Info: …\", never important, not in the alarm journal\n"
    "– <b>off</b>: no message\n\n"
    "Tap a button to switch: alarm → info → off. Saved in config.ini [alerts], "
    "used from the next analysis on.\n"
    "The last button: infos for all assets or only for the assets you hold (config.ini infos_for).")


def _alert_config_keyboard():
    # One row per signal: button for stocks and button for crypto
    settings = cl.read_alert_settings()
    rows = []
    for key, (_codes, label, *_rest) in cl.ALERTS.items():
        stock, crypto = settings[key]
        rows.append([InlineKeyboardButton(f"📈 {label}: {_KIND_ICON[stock]} {stock}", callback_data=f"acfg|{key}|stock"),
                     InlineKeyboardButton(f"🪙 {_KIND_ICON[crypto]} {crypto}", callback_data=f"acfg|{key}|crypto")])
    # infos for all assets or only for your portfolio (config.ini [alertsconfig] infos_for)
    scope = 'all assets' if cl.read_infos_for() == 'all' else 'only assets you hold'
    rows.append([InlineKeyboardButton(f"ℹ️ Infos for: {scope}", callback_data='acfg|infos_for|toggle')])
    return InlineKeyboardMarkup(rows)


def _alert_config_lines():
    settings = cl.read_alert_settings()
    lines = []
    for key, (_codes, label, _ds, _dc, desc, odds) in cl.ALERTS.items():
        stock, crypto = settings[key]
        lines.append(f"\n<b>{html.escape(label)}</b>: 📈 {stock}, 🪙 {crypto}\n"
                     f"<i>{html.escape(desc)} – {html.escape(odds)}</i>")
    return '\n'.join(lines)


async def _cmd_alert_config(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /smart_alert_config: alarm, info or off per signal, separately for stocks and crypto (config.ini [alerts])
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    try:
        cl.ensure_alert_section()   # normally done at the start of CaSa
        text, keyboard = _ALERT_CONFIG_TEXT + '\n' + _alert_config_lines(), _alert_config_keyboard()
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error in config.ini [alerts]: {error}')
        return
    await _reply_long_markup(update, text, keyboard)


async def _reply_long_markup(update, text, keyboard):
    # The explanation can exceed one message: buttons go with the last part
    parts = list(_text_chunks(text, 3800))
    for part in parts[:-1]:
        await update.message.reply_text(part, parse_mode='HTML')
    await update.message.reply_text(parts[-1], parse_mode='HTML', reply_markup=keyboard)


async def _cb_alert_config(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Tap in /smart_alert_config: next kind alarm -> info -> off -> alarm for stocks or crypto
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    _, key, asset_type = query.data.split('|', 2)
    try:
        if key == 'infos_for':   # all assets <-> only assets you hold
            cl.set_infos_for('portfolio' if cl.read_infos_for() == 'all' else 'all')
            await query.edit_message_reply_markup(reply_markup=_alert_config_keyboard())
            return
        stock, crypto = cl.read_alert_settings()[key]
        current = stock if asset_type == 'stock' else crypto
        kinds = cl.ALERT_KINDS
        cl.set_alert_setting(key, asset_type, kinds[(kinds.index(current) + 1) % len(kinds)])
        keyboard = _alert_config_keyboard()
    except (OSError, ValueError, KeyError, configparser.Error) as error:
        await query.edit_message_text(f'Error: {error}\nNot changed.')
        return
    # Only the buttons change (the text above them stays the same)
    await query.edit_message_reply_markup(reply_markup=keyboard)


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
        _, symbol, _, rest = _find_asset(list(context.args), allow_rest=True)
        if not rest:
            await update.message.reply_text('Format: /event <symbol or name> <DD/MM/YYYY> <description>\n'
                                            'Example: /event Nvidia 17/11/2026 Q3 earnings after market close')
            return
        key, name, date = cl.add_event(symbol, rest[0], ' '.join(rest[1:]))
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


PLAN_FORMAT = ('Format: /plan_alert <symbol or name> <support> <max_entry %> <target> <stop> <note>\n'
               '- for a value you do not use, e.g. no stop: ... 312.15 2 358.93 - note\n'
               'Example: /plan_alert TSMC 312.15 2 358.93 312.15 1/4 of the position after a rebound')
_PLAN_STATUS_TEXT = {'waiting': '⏳ waiting', 'buy_zone': '🟢 in the buy zone', 'target': '💰 at the target',
                     'ended': '⛔ ended'}


def _plan_line(plan):
    status = _PLAN_STATUS_TEXT[plan['status']]
    if plan['status'] == 'ended':
        status += plan['status_text'][len('ended'):]
    note = f"\n   {plan['note']}" if plan['note'] else ''
    return f"• {plan['name']} – {status}\n   {cl.plan_text(plan)}{note}"


async def _cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /plan_alert: show plans (with 🗑 to delete); /plan_alert <asset> <support> <max %> <target> <stop> <note>: save one
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    args = context.args or []
    try:
        if not args:
            plans = cl.read_plans()
            lines = ['<b>Plan alerts</b>'] if plans else ['No plan alerts yet.']
            lines += [html.escape(_plan_line(plan)) for plan in plans]
            row = [InlineKeyboardButton('➕ Add plan', callback_data='plg|add')]
            if plans:
                row.append(InlineKeyboardButton('🗑 Delete plan', callback_data='plg|del'))
            await update.message.reply_text('\n'.join(lines)[:4000], parse_mode='HTML',
                                            reply_markup=InlineKeyboardMarkup([row]))
            return
        asset_type, symbol, _, rest = _find_asset(list(args), allow_rest=True)
        if len(rest) < 4:
            await update.message.reply_text(PLAN_FORMAT)
            return
        value = lambda text: None if text.strip() in ('-', '') else cl.parse_price(text)
        support, max_entry, target, stop = value(rest[0]), cl.parse_percent(rest[1]), value(rest[2]), value(rest[3])
        name, _, replaced = cl.add_plan(symbol, support, max_entry, target, stop, ' '.join(rest[4:]))
    except (OSError, ValueError, configparser.Error) as error:
        await _reply_long(update, f'Error: {error}\nNot saved.\n\n{PLAN_FORMAT}')
        return
    plan = next(p for p in cl.read_plans() if p['symbol'] == symbol.lower())
    text = f"Plan {'replaced' if replaced else 'saved'}: {name}\n{cl.plan_text(plan)}"
    try:
        now = await asyncio.to_thread(func.current_price, symbol.upper() if asset_type == 'stock' else symbol,
                                      asset_type)
        text += f"\nCurrent price {func.format_euro(now)}."
    except Exception:
        pass
    text += '\nBuy zone and target are checked every hour (stocks Mon–Fri 08:00–22:00), the stop with the daily close.'
    _cmdq(app).put({'cmd': 'plan_alerts'})   # zone or target may already be reached: check right away
    await update.message.reply_text(text)


async def _cb_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # "⬆ Update now" / "Later" below the message about a new CaSa version
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    action, _, version = query.data.partition('|')
    await query.edit_message_reply_markup(reply_markup=None)   # no second tap
    if action == 'updno':
        await query.message.reply_text('Not updated. Later on the Pi: casa-update')
        return
    _cmdq(context.application).put({'cmd': 'update', 'version': version})
    await query.message.reply_text(f'Updating to {version} … (1–3 minutes)')


# ── Guided plan: asset -> support -> max_entry -> target -> stop -> note -> save (state in context.user_data) ──
_PLAN_CANCEL = [InlineKeyboardButton('✖ Cancel', callback_data='plg|cancel')]
_PLAN_QUESTIONS = {
    'support': 'Support: the price in € where the buy zone starts (e.g. 312.15)? Send it as a message, '
               'or Skip for a plan with a target only.',
    'target': 'Target: price in € for partial profits? Send it as a message, or Skip.',
    'stop': 'Stop: the plan fails at a daily close below this price in €? Send it as a message, or Skip.',
    'note': 'Your note (tranche size, conditions, e.g. "1/4 after a rebound")? Send it as a message, or Skip.',
}


def _plan_step_keyboard(step):
    skip = [InlineKeyboardButton('Skip', callback_data=f'plg|skip|{step}')]
    return InlineKeyboardMarkup([skip, _PLAN_CANCEL])


async def _plan_ask(message, plan, step):
    """Next question of the guided plan."""
    plan['step'] = step
    if step == 'max_entry':
        buttons = [[InlineKeyboardButton(f'{value} %', callback_data=f'plg|max|{value}') for value in (0, 2, 3, 5)],
                   _PLAN_CANCEL]
        await message.reply_text(f"Buy zone: buy only up to how many % above the support {plan['support']:.2f} €?\n"
                                  "Tap a button or send a number (0 to 50).", reply_markup=InlineKeyboardMarkup(buttons))
    elif step == 'confirm':
        try:
            cl.check_plan(plan['support'], plan['max_entry'], plan['target'], plan['stop'])
        except ValueError as error:
            plan.update(support=None, max_entry=0.0, target=None, stop=None)
            await message.reply_text(f'{error}\nPlease enter the prices again.')
            await _plan_ask(message, plan, 'support')
            return
        shown = {'support': plan['support'], 'max_entry': plan['max_entry'], 'target': plan['target'],
                 'stop': plan['stop'], 'note': plan['note']}
        text = f"Plan for {plan['name']}:\n{cl.plan_text(shown)}"
        if plan['note']:
            text += f"\nNote: {plan['note']}"
        if plan.get('replaces'):
            text += '\n\nThis replaces the current plan of this asset.'
        await message.reply_text(text + '\n\nSave?', reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton('✔ Save', callback_data='plg|save')], _PLAN_CANCEL]))
    else:
        await message.reply_text(_PLAN_QUESTIONS[step], reply_markup=_plan_step_keyboard(step))


_PLAN_NEXT = {'support': 'max_entry', 'max_entry': 'target', 'target': 'stop', 'stop': 'note', 'note': 'confirm'}


async def _cb_plan_guided(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Buttons of /plan_alert: ➕ Add plan (guided), 🗑 Delete plan, and the steps of the guided input
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    parts = query.data.split('|')
    action = parts[1]
    plan = context.user_data.get('plan')
    try:
        if action == 'add':
            context.user_data.pop('trade', None)   # only one guided input at a time
            await query.edit_message_text('New plan: for which asset?',
                                          reply_markup=_asset_keyboard(cl.read_assets(), 'plg|asset', cancel=False))
            await query.message.reply_text('Or cancel:', reply_markup=InlineKeyboardMarkup([_PLAN_CANCEL]))
        elif action == 'del':
            plans = cl.read_plans()
            buttons = [[InlineKeyboardButton(f"🗑 {p['name']}", callback_data=f"pldel|{p['key']}")] for p in plans
                       if len(f"pldelok|{p['key']}".encode()) <= 64]
            await query.edit_message_text('Delete which plan?' if buttons else 'No plans to delete.',
                                          reply_markup=InlineKeyboardMarkup(buttons) if buttons else None)
        elif action == 'cancel':
            context.user_data.pop('plan', None)
            await query.edit_message_text('Cancelled. No plan saved.')
        elif action == 'asset':
            asset_type, symbol = parts[2], parts[3]
            name = cl.read_assets().get((asset_type, symbol), symbol)
            plan = {'type': asset_type, 'symbol': symbol, 'name': name, 'support': None, 'max_entry': 0.0,
                    'target': None, 'stop': None, 'note': ''}
            plan['replaces'] = any(p['symbol'] == symbol for p in cl.read_plans())
            context.user_data['plan'] = plan
            try:
                now = await asyncio.to_thread(func.current_price, symbol.upper() if asset_type == 'stock' else symbol,
                                              asset_type)
                price = f"Current price: {func.format_euro(now)}"
            except Exception:
                price = 'Current price not available right now.'
            await query.edit_message_text(f"New plan: {name}\n{price}")
            await _plan_ask(query.message, plan, 'support')
        elif not plan:
            await query.edit_message_text('This input is no longer active. Start again with /plan_alert.')
        elif action == 'skip':
            await query.edit_message_reply_markup(reply_markup=None)
            step = parts[2]
            # without support there is no buy zone: max_entry is not asked
            await _plan_ask(query.message, plan, 'target' if step == 'support' else _PLAN_NEXT[step])
        elif action == 'max':
            await query.edit_message_reply_markup(reply_markup=None)
            plan['max_entry'] = cl.parse_percent(parts[2])
            await _plan_ask(query.message, plan, 'target')
        elif action == 'save':
            name, _, replaced = cl.add_plan(plan['symbol'], plan['support'], plan['max_entry'], plan['target'],
                                            plan['stop'], plan['note'])
            context.user_data.pop('plan', None)
            _cmdq(context.application).put({'cmd': 'plan_alerts'})   # zone or target may already be reached
            summary = cl.plan_text({key: plan[key] for key in ('support', 'max_entry', 'target', 'stop')})
            await query.edit_message_text(f"Plan {'replaced' if replaced else 'saved'}: {name}\n{summary}\n"
                                          "Checked every hour (stocks Mon–Fri 08:00–22:00), the stop with the daily close.")
    except (OSError, ValueError, configparser.Error) as error:
        await query.message.reply_text(f'Error: {error}')


async def _on_plan_text(update, context, plan):
    """Typed answers during the guided plan: prices, max_entry in %, note."""
    text = update.message.text.strip()
    step = plan['step']
    try:
        if step == 'note':
            plan['note'] = ' '.join(text.split())
        elif step == 'max_entry':
            plan['max_entry'] = cl.parse_percent(text)
        elif step in ('support', 'target', 'stop'):
            plan[step] = cl.parse_price(text)
        else:
            return
    except ValueError as error:
        await update.message.reply_text(f'{error}\nPlease try again.', reply_markup=_plan_step_keyboard(step))
        return
    await _plan_ask(update.message, plan, _PLAN_NEXT[step])


async def _show(query, text, reply_markup=None):
    """Answer to a button: a text message is changed; below a chart (photo) a new message follows and
    the chart with its analysis stays."""
    message = getattr(query, 'message', None)
    if message is not None and getattr(message, 'photo', None):
        await message.reply_text(text, reply_markup=reply_markup)
    else:
        await query.edit_message_text(text, reply_markup=reply_markup)


async def _cb_plan_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # 🗑 in the /plan_alert list or below a plan alert: first ask, then delete
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    action, _, key = query.data.partition('|')
    if action == 'pldelno':
        await query.edit_message_text('Not deleted.')
        return
    try:
        plan = next((p for p in cl.read_plans() if p['key'].lower() == key.lower()), None)
        if action == 'pldel':
            if plan is None:
                await _show(query, 'This plan no longer exists (already deleted).')
                return
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton('Yes, delete', callback_data=f'pldelok|{key}'),
                                              InlineKeyboardButton('Cancel', callback_data='pldelno|')]])
            await _show(query, f"Delete this plan?\n{_plan_line(plan)}", reply_markup=keyboard)
            return
        cl.delete_plan(key)   # action == 'pldelok'
    except (OSError, ValueError, configparser.Error) as error:
        await _show(query, f'Error: {error}\nNot deleted.')
        return
    await _show(query, f"Plan deleted: {plan['name'] if plan else key}")


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
    # The name may have several words ("Trane Technologies"): the fixed values are counted from the end
    # crypto|stock is optional (taken from assets.ini, still accepted for the old format)
    tail = 3 if action == 'buy' else 2   # amount, quantity [, date]
    args = list(context.args or [])
    if len(args) <= tail:
        # no or incomplete input: ask step by step with buttons instead of showing the format
        await _start_guided_trade(update.message, context, action)
        return
    words, (amount_text, qty_text, *date_arg) = args[:-tail], args[-tail:]
    words, types = _split_type(words)
    try:
        asset_type, symbol, name, _ = _find_asset(words, types)
    except (OSError, configparser.Error, ValueError) as error:
        await _start_guided_trade(update.message, context, action, note=f'{error}\n')
        return
    date = date_arg[0] if action == 'buy' else datetime.now().strftime('%d/%m/%Y')
    try:
        if not _valid_date(date):
            raise ValueError('Invalid date or date in the future. Format: DD/MM/YYYY.')
        amount, qty = cl.valid_numbers(amount_text, qty_text)
        symbol = symbol.lower() if asset_type == 'crypto' else symbol.upper()
        row = _append_transaction(asset_type, symbol, name, action, amount, qty, date)
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
        await _reply_long(update, f'Error: {error}\nTransaction not saved.')
        return
    await _trade_saved(update.message, context, row)


async def _cmd_buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _save_trade(update, context, 'buy')


async def _cmd_sell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _save_trade(update, context, 'sell')


_RECORD = {'buy': _cmd_buy, 'sell': _cmd_sell}   # sellall is added below its function


async def _cmd_record(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /record: buttons Buy / Sell / Sell all; /record buy|sell|sellall <...>: directly as before /buy, /sell, /sellall
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    args = list(context.args or [])
    action = args[0].lower() if args else ''
    if action in _RECORD:
        inner = types.SimpleNamespace(args=args[1:], application=context.application,
                                      user_data=context.user_data, bot_data=context.bot_data)
        await _RECORD[action](update, inner)
        return
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton('🟢 Buy', callback_data='rec|buy'),
                                      InlineKeyboardButton('🔴 Sell', callback_data='rec|sell'),
                                      InlineKeyboardButton('Sell all', callback_data='rec|sellall')],
                                     [InlineKeyboardButton('✖ Cancel', callback_data='rec|cancel')]])
    text = 'What do you want to record?'
    if action:
        text = f"Unknown: {args[0]}. Use buy, sell or sellall, e.g. /record buy Qualcomm 300 2\n{text}"
    await update.message.reply_text(text, reply_markup=keyboard)


async def _cb_record(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Button from /record: start the chosen input like /record buy, /record sell or /record sellall
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    parts = query.data.split('|')
    action = parts[1]
    if action not in _RECORD:
        await query.edit_message_text('Nothing recorded.')
        return
    if len(parts) == 4:
        # below a plan alert: the asset is already known, the chart message stays
        asset_type, symbol = parts[2], parts[3]
        if action == 'sellall':
            await _run_command(_cmd_sellall, update, context, [symbol])
        else:
            await _start_trade_for_asset(query.message, context, action, asset_type, symbol)
        return
    await query.edit_message_text({'buy': 'Record a buy', 'sell': 'Record a sell',
                                   'sellall': 'Sell the full position'}[action])
    await _run_command(_RECORD[action], update, context)


async def _start_trade_for_asset(message, context, action, asset_type, symbol):
    """Guided buy/sell with the asset already chosen (button below a plan alert): starts at the amount."""
    try:
        name = cl.read_assets().get((asset_type, symbol)) or _held_assets().get((asset_type, symbol), symbol)
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError):
        name = symbol
    context.user_data.pop('plan', None)
    context.user_data['trade'] = {'action': action, 'step': 'amount', 'type': asset_type, 'name': name,
                                  'symbol': symbol if asset_type == 'crypto' else symbol.upper()}
    verb = 'paid' if action == 'buy' else 'received'
    label = 'Record a buy' if action == 'buy' else 'Record a sell'
    await message.reply_text(f"{label}: {name}\nAmount in € {verb} (e.g. 1200 or 1200,50)? Send it as a message.",
                             reply_markup=InlineKeyboardMarkup([_CANCEL_ROW]))


# ── Guided /buy and /sell: asset -> amount -> quantity -> date -> confirm (state in context.user_data) ──
_CANCEL_ROW = [InlineKeyboardButton('✖ Cancel', callback_data='tr|cancel')]


def _held_assets():
    """Assets with holdings (for /sell and /sellall buttons): {(type, symbol): name}."""
    rows = cl.read_transactions(_TRANSACTIONS_PATH)
    held = {}
    for key in {(row['type'], row['symbol'].lower()) for row in rows if row['action'] in ('buy', 'sell')}:
        name = next((row.get('name') or key[1] for row in rows if (row['type'], row['symbol'].lower()) == key), key[1])
        symbol = key[1] if key[0] == 'crypto' else key[1].upper()
        asset = cl.apply_transactions(cl.CryptoStock(symbol, name, key[0]), rows)
        if asset.state.amount_crypto_stock > 1e-12:
            held[key] = name
    return dict(sorted(held.items(), key=lambda item: item[1].lower()))


def _asset_keyboard(entries, prefix, cancel=True):
    # two buttons per row: 🪙 crypto / 📈 stock with the name
    buttons = [InlineKeyboardButton(('🪙 ' if asset_type == 'crypto' else '📈 ') + name,
                                    callback_data=f'{prefix}|{asset_type}|{symbol}')
               for (asset_type, symbol), name in entries.items()]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows + ([_CANCEL_ROW] if cancel else []))


async def _start_guided_trade(message, context, action, note=''):
    """Step 1: choose the asset (buy: all assets, sell: only assets you hold)."""
    try:
        entries = cl.read_assets() if action == 'buy' else _held_assets()
    except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
        await message.reply_text(f'Error: {error}')
        return
    if not entries:
        await message.reply_text('No holdings to sell.' if action == 'sell' else 'No assets in assets.ini.')
        return
    context.user_data.pop('plan', None)   # only one guided input at a time
    context.user_data['trade'] = {'action': action, 'step': 'asset'}
    question = 'Which asset did you buy?' if action == 'buy' else 'Which asset did you sell?'
    await message.reply_text(note + question, reply_markup=_asset_keyboard(entries, 'tr|asset'))


def _date_keyboard():
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    return InlineKeyboardMarkup([[InlineKeyboardButton(f"Today ({today:%d/%m})", callback_data=f"tr|date|{today:%d/%m/%Y}"),
                                  InlineKeyboardButton(f"Yesterday ({yesterday:%d/%m})", callback_data=f"tr|date|{yesterday:%d/%m/%Y}")],
                                 [InlineKeyboardButton('Other date', callback_data='tr|date|other')], _CANCEL_ROW])


def _trade_summary(trade):
    label = 'Buy' if trade['action'] == 'buy' else 'Sell'
    unit = trade['amount'] / trade['qty'] if trade.get('qty') else 0
    return (f"{label}: {trade['name']} ({trade['symbol']})\nAmount: {func.format_euro(trade['amount'])}\n"
            f"Quantity: {trade['qty']:g}\nPrice per unit: {func.format_euro(unit)}\nDate: {trade['date']}")


async def _cb_trade(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Buttons of the guided /buy and /sell and ↩ Undo
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    parts = query.data.split('|')
    trade = context.user_data.get('trade')
    if parts[1] == 'cancel':
        context.user_data.pop('trade', None)
        await query.edit_message_text('Cancelled. Nothing saved.')
        return
    if parts[1] == 'undo':
        row = context.bot_data.get('last_trade')
        try:
            if not row or row.get('id') != parts[2]:
                raise ValueError('This transaction can no longer be undone.')
            _undo_transaction(row)
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            await query.edit_message_text(f'{query.message.text}\n\n{error}')
            return
        context.bot_data.pop('last_trade', None)
        _cmdq(context.application).put({'cmd': 'reload_transactions'})
        await query.edit_message_text(f'{query.message.text}\n\n↩ Undone: the transaction was removed again.')
        return
    if not trade:
        await query.edit_message_text('This input is no longer active. Start again with /record.')
        return
    if parts[1] == 'asset':
        asset_type, symbol = parts[2], parts[3]
        try:
            name = cl.read_assets().get((asset_type, symbol)) or _held_assets().get((asset_type, symbol), symbol)
        except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError):
            name = symbol
        trade.update(type=asset_type, symbol=symbol if asset_type == 'crypto' else symbol.upper(), name=name,
                     step='amount')
        verb = 'paid' if trade['action'] == 'buy' else 'received'
        await query.edit_message_text(f"{name}: amount in € {verb} (e.g. 1200 or 1200,50)? Send it as a message.",
                                      reply_markup=InlineKeyboardMarkup([_CANCEL_ROW]))
    elif parts[1] == 'date':
        if parts[2] == 'other':
            trade['step'] = 'date'
            await query.edit_message_text('Date of the transaction (DD/MM/YYYY)? Send it as a message.',
                                          reply_markup=InlineKeyboardMarkup([_CANCEL_ROW]))
            return
        trade.update(date=parts[2], step='confirm')
        await query.edit_message_text(_trade_summary(trade) + '\n\nSave?', reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton('✔ Save', callback_data='tr|save')], _CANCEL_ROW]))
    elif parts[1] == 'save' and trade.get('step') == 'confirm':
        context.user_data.pop('trade', None)
        try:
            row = _append_transaction(trade['type'], trade['symbol'], trade['name'], trade['action'],
                                      trade['amount'], trade['qty'], trade['date'])
        except (OSError, UnicodeError, csv.Error, configparser.Error, ValueError) as error:
            await query.edit_message_text(f'Error: {error}\nTransaction not saved.')
            return
        await query.edit_message_text(_trade_summary(trade))
        await _trade_saved(query.message, context, row)


async def _on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Typed answers during the guided /buy or /sell (amount, quantity, other date) or the guided plan;
    # other texts are ignored
    plan = context.user_data.get('plan')
    if plan and update.message is not None and _authorized(update, context.application.bot_data['chat_id']):
        await _on_plan_text(update, context, plan)
        return
    trade = context.user_data.get('trade')
    if not trade or trade.get('step') not in ('amount', 'qty', 'date') or update.message is None:
        return
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    text = update.message.text.strip()
    cancel = InlineKeyboardMarkup([_CANCEL_ROW])
    try:
        if trade['step'] == 'amount':
            trade['amount'] = cl.parse_price(text)
            trade['step'] = 'qty'
            unit = 'coins' if trade['type'] == 'crypto' else 'shares'
            await update.message.reply_text(f"Quantity ({unit}, e.g. 2 or 0,015)?", reply_markup=cancel)
        elif trade['step'] == 'qty':
            trade['qty'] = cl.valid_numbers(trade['amount'], text)[1]
            trade['step'] = 'date_buttons'
            await update.message.reply_text('Date of the transaction?', reply_markup=_date_keyboard())
        else:
            date = text.replace('.', '/')
            if not _valid_date(date):
                raise ValueError('Invalid date or date in the future (format DD/MM/YYYY).')
            trade.update(date=datetime.strptime(date, '%d/%m/%Y').strftime('%d/%m/%Y'), step='confirm')
            await update.message.reply_text(_trade_summary(trade) + '\n\nSave?', reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton('✔ Save', callback_data='tr|save')], _CANCEL_ROW]))
    except ValueError as error:
        await update.message.reply_text(f'{error}\nPlease try again.', reply_markup=cancel)


async def _trade_saved(message, context, row):
    """Confirmation after saving, with ↩ Undo (removes exactly this transaction if it is still the last one)."""
    row = dict(row, id=os.urandom(3).hex())
    context.bot_data['last_trade'] = row
    _cmdq(context.application).put({'cmd': 'reload_transactions'})
    label = 'Buy' if row['action'] == 'buy' else 'Sell'
    symbol = row['symbol'] if row['type'] == 'crypto' else row['symbol'].upper()
    await message.reply_text(
        f"{label} saved: {row['name']} ({symbol})\nAmount: {float(row['amount']):.2f} €\nQuantity: {float(row['qty']):g}\n"
        f"Date: {row['date']}\nThe main program reloads the transactions.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton('↩ Undo', callback_data=f"tr|undo|{row['id']}")]]))


def _holding(symbol, asset_type, name):
    # Current holdings according to transactions.csv (FIFO as in the main program)
    asset = cl.apply_transactions(cl.CryptoStock(symbol, name, asset_type), cl.read_transactions(_TRANSACTIONS_PATH))
    return asset.state.amount_crypto_stock


def _current_price(symbol, asset_type, name):
    # Latest price in EUR; blocks during API retries -> only call via asyncio.to_thread
    df = cl.CryptoStock(symbol, name, asset_type).get_API_data('100d')
    if df is None or df.empty:
        raise ValueError('No current price available. Please add the amount: /record sellall <symbol> <amount>')
    return float(df['Price'].iloc[-1])


async def _cmd_sellall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /sellall <symbol> [amount]: record the full position as a sale (after confirmation)
    app = context.application
    if not _authorized(update, app.bot_data['chat_id']):
        return
    if not context.args:
        await _reply_held_buttons(update.message, 'sapick', 'Sell the full position of which asset? '
                                  '(the current price is used; with your own amount: /record sellall <name> <amount>)')
        return
    try:
        asset_type, symbol, name, rest = _find_asset(list(context.args), allow_rest=True)
    except (OSError, configparser.Error, ValueError) as error:
        # unknown asset: offer the assets you hold as buttons
        await _reply_held_buttons(update.message, 'sapick', f'{error}\nSell the full position of which asset?')
        return
    try:
        if len(rest) > 1:
            raise ValueError(f"Unexpected text after {name}: {' '.join(rest)}")
        symbol = symbol.lower() if asset_type == 'crypto' else symbol.upper()
        qty = _holding(symbol, asset_type, name)
        if qty <= 0:
            raise ValueError(f'No holdings of {name}.')
        if rest:
            amount, _ = cl.valid_numbers(rest[0], qty)
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
        f"If your proceeds were different: /record sellall {symbol} <amount>",
        reply_markup=keyboard)


_RECORD['sellall'] = _cmd_sellall


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
    await _reply_transactions_table(update, rows, 'Transactions')


_TABLE_ROWS_PER_IMAGE = 40   # Telegram limits the image size: more rows are sent as several images


async def _reply_transactions_table(update, rows, title):
    """Transactions as a table image (style of the daily summary), newest first."""
    table = []
    for row in rows:
        amount, qty = cl.valid_numbers(row['amount'], row['qty'])
        table.append({'date': datetime.strptime(row['date'], '%d/%m/%Y'), 'name': row.get('name') or row['symbol'],
                      'type': row['type'], 'action': row['action'], 'amount': amount, 'qty': qty})
    table.sort(key=lambda row: row['date'], reverse=True)
    bought = sum(row['amount'] for row in table if row['action'] == 'buy')
    sold = sum(row['amount'] for row in table if row['action'] == 'sell')
    subtitle = f"{len(table)} transactions · bought {func.format_euro(bought)} · sold {func.format_euro(sold)}"
    parts = [table[i:i + _TABLE_ROWS_PER_IMAGE] for i in range(0, len(table), _TABLE_ROWS_PER_IMAGE)]
    for number, part in enumerate(parts, 1):
        name = f"{title} ({number}/{len(parts)})" if len(parts) > 1 else title
        path = os.path.join(tempfile.gettempdir(), f"casa_transactions_{os.getpid()}_{number}.png")
        try:
            # drawing takes a moment: in a thread, so that the bot stays responsive
            await asyncio.to_thread(func.create_transactions_image, part, path, name, subtitle)
            with open(path, 'rb') as photo:
                await update.message.reply_photo(photo=photo)
        finally:
            if os.path.exists(path):
                os.remove(path)


async def _cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    if not _authorized(update, app.bot_data["chat_id"]):  # type: ignore[index]
        return

    if not context.args:
        await _reply_traded_buttons(update.message, 'History of which asset?')
        return

    symbol = ' '.join(context.args).strip()
    try:
        rows = _get_history(symbol)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await _reply_long(update, f'History could not be read: {error}')
        return
    if not rows:
        await _reply_traded_buttons(update.message, f'No transactions found for {symbol}. History of which asset?')
        return
    await _reply_transactions_table(update, rows, f"History · {rows[0].get('name') or symbol.upper()}")


def _traded_assets():
    """All assets with transactions (also ones no longer in assets.ini): {(type, symbol): name}."""
    names = {}
    for row in cl.read_transactions(_TRANSACTIONS_PATH):
        if row['action'] in ('buy', 'sell'):
            names[(row['type'], row['symbol'].lower())] = row.get('name') or row['symbol']
    return dict(sorted(names.items(), key=lambda item: item[1].lower()))


async def _reply_traded_buttons(message, question):
    try:
        entries = _traded_assets()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await message.reply_text(f'Transactions could not be read: {error}')
        return
    if not entries:
        await message.reply_text('No transactions saved yet.')
        return
    await message.reply_text(question, reply_markup=_asset_keyboard(entries, 'hist', cancel=False))


async def _reply_held_buttons(message, prefix, question):
    try:
        entries = _held_assets()
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        await message.reply_text(f'Transactions could not be read: {error}')
        return
    if not entries:
        await message.reply_text('No holdings.')
        return
    await message.reply_text(question, reply_markup=_asset_keyboard(entries, prefix, cancel=False))


async def _cb_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Asset chosen with a button: /history (hist|type|symbol) or /sellall (sapick|type|symbol)
    query = update.callback_query
    await query.answer()
    if not _authorized(update, context.application.bot_data['chat_id']):
        return
    prefix, asset_type, symbol = query.data.split('|', 2)
    await query.edit_message_reply_markup(reply_markup=None)   # the choice is made: remove the buttons
    command = _cmd_history if prefix == 'hist' else _cmd_sellall
    await _run_command(command, update, context, [symbol])


async def _run_command(command, update, context, args=()):
    """Run a command handler from a button (asset choice, /record): replies go to the chat of the button."""
    fake_update = types.SimpleNamespace(message=update.callback_query.message, effective_chat=update.effective_chat)
    fake_context = types.SimpleNamespace(args=list(args), application=context.application,
                                         user_data=context.user_data, bot_data=context.bot_data)
    await command(fake_update, fake_context)


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
        BotCommand("help", "Show all commands and their parameters"),
        BotCommand("status", "Analyses, next runs, problems"),
        BotCommand("check", "Daily analysis without alarm: choose asset via buttons"),
        BotCommand("smart_alert", "Add assets to or remove them from the alarm analysis"),
        BotCommand("smart_alert_config", "Alarm, info or off per signal (stocks, crypto)"),
        BotCommand("event", "Show or add dates: symbol date text"),
        BotCommand("plan_alert", "Trading plans: show, or add: asset support max% target stop note"),
        BotCommand("analyse", "Start analysis: hourly, daily or summary"),
        BotCommand("portfolio", "Daily P&L table and chart now"),
        BotCommand("record", "Record a buy, sell or sale of the full position"),
        BotCommand("history", "Transactions of one asset (choose via buttons)"),
        BotCommand("transactions", "Show all saved transactions"),
        BotCommand("update", "Look for a new CaSa version and install it"),
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
    error = context.error
    # Two CaSa instances with the same bot (e.g. PC and Raspberry Pi): one clear line instead of a traceback
    if isinstance(error, Conflict):
        func.log_print("Telegram: another CaSa instance is running with the same bot (Conflict). "
                       "Stop one of them (e.g. PC or Raspberry Pi).", 'ERROR')
        return
    # Short connection problems while asking Telegram for new commands: the bot retries on its own
    if isinstance(error, NetworkError) and not (isinstance(update, Update) and update.effective_message):
        func.log_print(f"Telegram connection problem, retrying automatically: {type(error).__name__}: "
                       f"{str(error).strip()[:120]}", 'WARN')
        return
    logging.error('Error in Telegram command', exc_info=error)
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
    """Bot process. An error that stops the bot is written to logfile.txt (otherwise it is only in the console)."""
    try:
        _run_bot(notify_q, command_q)
    except BaseException:
        func.log_print(f"Telegram bot stopped with an error:\n{traceback.format_exc()}", 'ERROR')
        raise


def _run_bot(notify_q: Queue, command_q: Queue) -> None:
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

    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CommandHandler("portfolio", _cmd_portfolio))
    app.add_handler(CommandHandler("status", _cmd_status))
    app.add_handler(CommandHandler("analyse", _cmd_analyse))
    app.add_handler(CommandHandler("check", _cmd_check))
    app.add_handler(CommandHandler(["ai_prompt", "ai"], _cmd_ai))   # /ai: old name
    app.add_handler(CommandHandler("ai_api_decision3s", _cmd_ai_decision))
    app.add_handler(CommandHandler("ai_api_plan", _cmd_ai_plan))
    app.add_handler(CommandHandler(["smart_alert", "alarm"], _cmd_alarm))   # /alarm still works as before
    app.add_handler(CommandHandler(["event", "termin"], _cmd_event))   # /termin still works as before
    app.add_handler(CommandHandler(["plan_alert", "price_alert"], _cmd_plan))   # /price_alert: old name
    app.add_handler(CallbackQueryHandler(_cb_alarm, pattern=r"^alarm\|"))
    app.add_handler(CommandHandler("smart_alert_config", _cmd_alert_config))
    app.add_handler(CallbackQueryHandler(_cb_alert_config, pattern=r"^acfg\|"))
    app.add_handler(CallbackQueryHandler(_cb_assets, pattern=r"^(check|ai|ai_decision|ai_plan)\|"))
    app.add_handler(CallbackQueryHandler(_cb_ai_plan, pattern=r"^aiplan(ok|no)\|"))
    app.add_handler(CallbackQueryHandler(_cb_noai, pattern=r"^noai$"))
    app.add_handler(CallbackQueryHandler(_cb_ai_button, pattern=r"^ai[dp]?btn\|"))
    app.add_handler(CallbackQueryHandler(_cb_event_delete, pattern=r"^evdel(ok|no)?\|"))
    app.add_handler(CallbackQueryHandler(_cb_plan_delete, pattern=r"^pldel(ok|no)?\|"))
    app.add_handler(CallbackQueryHandler(_cb_plan_guided, pattern=r"^plg\|"))
    app.add_handler(CallbackQueryHandler(_cb_update, pattern=r"^upd(no)?\|"))
    app.add_handler(CommandHandler("record", _cmd_record))
    app.add_handler(CallbackQueryHandler(_cb_record, pattern=r"^rec\|"))
    app.add_handler(CommandHandler("buy", _cmd_buy))     # old names, not in the command list any more
    app.add_handler(CommandHandler("sell", _cmd_sell))
    app.add_handler(CallbackQueryHandler(_cb_trade, pattern=r"^tr\|"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))   # answers of the guided /buy, /sell
    app.add_handler(CommandHandler("sellall", _cmd_sellall))
    app.add_handler(CallbackQueryHandler(_cb_sellall, pattern=r"^sellall\|"))
    app.add_handler(CallbackQueryHandler(_cb_pick, pattern=r"^(hist|sapick)\|"))
    app.add_handler(CommandHandler("history", _cmd_history))
    app.add_handler(CommandHandler("transactions", _cmd_transactions))
    app.add_handler(CommandHandler("update", _cmd_update))
    app.add_handler(CommandHandler("restart", _cmd_restart))

    app.add_error_handler(_on_error)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info(f"Telegram bot started (chat_id={chat_id})")
    app.run_polling(stop_signals=None)
    logging.info("Telegram bot stopped")


if __name__ == '__main__':
    print('Please start main.py. telegram_bot.py is started by the main program.')
