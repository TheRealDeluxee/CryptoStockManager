"""Telegram bot commands (without Telegram): /buy, /sell, /check, /ai, /smart_alert, /event, /history, /transactions."""
import asyncio
import datetime
import types

import pytest

import classes as cl
import telegram_bot as tb


@pytest.fixture
def bot(monkeypatch):
    """Fake bot: replies (with their buttons) and commands to the main program are collected.
    user_data/bot_data stay between calls, like in the real bot (needed for the guided /buy and /sell)."""
    queue, replies, photos, markups, edits = [], [], [], [], []
    monkeypatch.setattr(tb, '_authorized', lambda *a: True)
    monkeypatch.setattr(tb, '_cmdq', lambda app: types.SimpleNamespace(put=queue.append))
    app = types.SimpleNamespace(bot_data={'chat_id': 1})
    user_data = {}

    async def reply_text(text, **kwargs):
        replies.append(text)
        markups.append(kwargs.get('reply_markup'))

    async def reply_photo(photo, **kwargs):
        photos.append(photo.read())

    message = types.SimpleNamespace(reply_text=reply_text, reply_photo=reply_photo, text='')

    def context(args=()):
        return types.SimpleNamespace(args=list(args), application=app, user_data=user_data, bot_data=app.bot_data)

    def run(command, *args):
        replies.clear()
        asyncio.run(command(types.SimpleNamespace(message=message), context(args)))
        return chr(10).join(replies)

    def type_text(text):
        # the user sends a normal message (answer in the guided /buy or /sell)
        replies.clear()
        message.text = text
        asyncio.run(tb._on_text(types.SimpleNamespace(message=message), context()))
        return chr(10).join(replies)

    def tap(data):
        # the user taps a button with this callback data
        async def answer(*a, **k):
            pass

        async def edit_message_text(text, **kwargs):
            edits.append(text)
            markups.append(kwargs.get('reply_markup'))
        query = types.SimpleNamespace(data=data, answer=answer, edit_message_text=edit_message_text,
                                      message=types.SimpleNamespace(text=edits[-1] if edits else '', reply_text=reply_text))
        asyncio.run(tb._cb_trade(types.SimpleNamespace(callback_query=query), context()))
        return edits[-1] if edits else ''

    def buttons():
        markup = markups[-1]
        return [button.callback_data for row in markup.inline_keyboard for button in row] if markup else []

    return types.SimpleNamespace(run=run, queue=queue, photos=photos, type_text=type_text, tap=tap,
                                 buttons=buttons, replies=replies, edits=edits, user_data=user_data)


def last_transaction():
    return cl.read_transactions()[-1]


# ── /buy and /sell ─────────────────────────────────────────────────────────
@pytest.mark.parametrize('args', [
    ('Rheinmetall', '100', '1', '01/10/2025'),
    ('RHM.DE', '100', '1', '01/10/2025'),
    ('rhm.de', 'stock', '100', '1', '01/10/2025'),     # old format with type still works
])
def test_buy_by_symbol_or_name(bot, args):
    reply = bot.run(tb._cmd_buy, *args)
    assert 'Buy saved: Rheinmetall (RHM.DE)' in reply
    row = last_transaction()
    assert (row['symbol'], row['action'], row['date']) == ('rhm.de', 'buy', '01/10/2025')
    assert {'cmd': 'reload_transactions'} in bot.queue


def test_buy_name_with_spaces(bot):
    assert 'Trane Technologies (2IS.F)' in bot.run(tb._cmd_buy, 'Trane', 'Technologies', '760', '2', '14/07/2025')


def test_sell_crypto_without_type(bot):
    assert 'Sell saved: Bitcoin (bitcoin)' in bot.run(tb._cmd_sell, 'bitcoin', '300', '0.005')
    assert last_transaction()['date'] == datetime.date.today().strftime('%d/%m/%Y')


@pytest.mark.parametrize('args, error', [
    (('Bitcoin', 'stock', '1', '1', '01/01/2025'), 'Which asset did you buy?'),   # wrong type: choose instead
    (('Rheinmetall', '100', '1', '01/01/2099'), 'future'),                  # date in the future
    (('Rheinmetall', '0', '1', '01/01/2025'), 'greater than zero'),         # invalid amount
    (('Unknown', '100', '1', '01/01/2025'), 'Which asset did you buy?'),
])
def test_buy_errors_save_nothing(bot, args, error):
    before = len(cl.read_transactions())
    assert error in bot.run(tb._cmd_buy, *args)
    assert len(cl.read_transactions()) == before


def test_sell_more_than_owned_is_rejected(bot):
    reply = bot.run(tb._cmd_sell, 'bitcoin', '5000', '1')
    assert 'not saved' in reply
    assert len(cl.read_transactions()) == 4


def test_incomplete_buy_starts_guided_input(bot):
    assert 'Which asset did you buy?' in bot.run(tb._cmd_buy, 'Rheinmetall')
    assert 'tr|asset|stock|rhm.de' in bot.buttons()


def test_guided_buy_step_by_step_and_undo(bot):
    bot.run(tb._cmd_buy)
    assert 'Rheinmetall' in bot.tap('tr|asset|stock|rhm.de')
    assert 'Please try again' in bot.type_text('abc')            # invalid amount: asked again
    assert 'Quantity' in bot.type_text('1.234,50')
    assert 'Please try again' in bot.type_text('-2')             # invalid quantity
    assert 'Date of the transaction?' in bot.type_text('2')
    assert 'tr|date|other' in bot.buttons()
    bot.tap('tr|date|other')
    assert 'Please try again' in bot.type_text('01/01/2099')     # future date
    summary = bot.type_text('15.10.2025')
    assert 'Amount: 1,234.50 €' in summary and 'Price per unit: 617.25 €' in summary and 'Date: 15/10/2025' in summary
    before = len(cl.read_transactions())
    bot.tap('tr|save')
    rows = cl.read_transactions()
    assert len(rows) == before + 1 and rows[-1]['date'] == '15/10/2025' and float(rows[-1]['amount']) == 1234.5
    assert 'Buy saved: Rheinmetall (RHM.DE)' in bot.replies[-1]
    undo = [b for b in bot.buttons() if b.startswith('tr|undo|')][0]
    assert {'cmd': 'reload_transactions'} in bot.queue
    assert 'Undone' in bot.tap(undo)
    assert len(cl.read_transactions()) == before
    assert 'no longer be undone' in bot.tap(undo)                 # only once


def test_guided_sell_offers_only_held_assets(bot):
    bot.run(tb._cmd_sell)
    assert set(bot.buttons()) == {'tr|asset|crypto|bitcoin', 'tr|asset|stock|rhm.de', 'tr|cancel'}
    assert 'Cancelled' in bot.tap('tr|cancel')
    assert bot.user_data.get('trade') is None


def test_guided_sell_more_than_held_is_rejected(bot):
    bot.run(tb._cmd_sell)
    bot.tap('tr|asset|crypto|bitcoin')
    bot.type_text('500')
    bot.type_text('1')
    bot.tap('tr|date|' + datetime.date.today().strftime('%d/%m/%Y'))
    assert 'not saved' in bot.tap('tr|save')
    assert len(cl.read_transactions()) == 4


def test_undo_after_direct_buy(bot):
    bot.run(tb._cmd_buy, 'Nvidia', '450', '3', '05/01/2025')
    undo = [b for b in bot.buttons() if b.startswith('tr|undo|')][0]
    bot.tap(undo)
    assert len(cl.read_transactions()) == 4


# ── analysis commands ──────────────────────────────────────────────────────
def test_check_and_ai_by_name(bot):
    assert 'Rheinmetall' in bot.run(tb._cmd_check, 'Rheinmetall')
    assert 'Trane Technologies' in bot.run(tb._cmd_ai, 'trane')
    assert bot.queue == [{'cmd': 'check', 'symbol': 'rhm.de', 'type': 'stock'},
                         {'cmd': 'ai', 'symbol': '2is.f', 'type': 'stock'}]


def test_check_unknown_asset_shows_buttons(bot):
    reply = bot.run(tb._cmd_check, 'foo')
    assert 'not in assets.ini' in reply and 'Which daily analysis' in reply
    assert 'check|stock|rhm.de' in bot.buttons()
    assert bot.queue == []


def test_history_and_sellall_without_asset_show_buttons(bot):
    assert 'History of which asset?' in bot.run(tb._cmd_history)
    assert set(bot.buttons()) == {'hist|crypto|bitcoin', 'hist|stock|rhm.de'}
    assert 'Sell the full position' in bot.run(tb._cmd_sellall)
    assert set(bot.buttons()) == {'sapick|crypto|bitcoin', 'sapick|stock|rhm.de'}   # only assets you hold
    assert 'Sell the full position' in bot.run(tb._cmd_sellall, 'xyz')                 # unknown -> buttons
    assert 'No holdings of Apple' in bot.run(tb._cmd_sellall, 'Apple')                  # known, not held
    assert 'Alarm analysis' in bot.run(tb._cmd_alarm, 'Rheinmetall', 'maybe')         # wrong switch -> buttons


def test_smart_alert_on_off(bot):
    assert 'alarm on' in bot.run(tb._cmd_alarm, 'Trane', 'on')
    assert ('stock', '2is.f') in cl.read_assets(alarm_only=True)
    assert 'alarm off' in bot.run(tb._cmd_alarm, 'Trane', 'off')
    assert ('stock', '2is.f') not in cl.read_assets(alarm_only=True)


def test_event_by_name(bot):
    date = (datetime.date.today() + datetime.timedelta(days=30)).strftime('%d/%m/%Y')
    assert 'Event saved' in bot.run(tb._cmd_event, 'Nvidia', date, 'Q4', 'earnings')
    assert cl.read_events()[0]['text'] == 'Q4 earnings'


# ── history and transactions as table images ──────────────────────────────
@pytest.mark.parametrize('text', ['rhm.de', 'Rheinmetall', 'rhein'])
def test_history_by_symbol_or_name(text):
    assert len(tb._get_history(text)) == 3


def test_history_unknown():
    assert tb._get_history('nothing') == []


def test_transactions_table_image(bot):
    bot.run(tb._cmd_transactions)
    assert len(bot.photos) == 1
    assert bot.photos[0][:8] == b'\x89PNG\r\n\x1a\n'


def test_many_transactions_are_split_into_several_images(bot, monkeypatch):
    monkeypatch.setattr(tb, '_TABLE_ROWS_PER_IMAGE', 2)
    bot.run(tb._cmd_transactions)
    assert len(bot.photos) == 2                     # 4 transactions, 2 per image


# ── long texts ─────────────────────────────────────────────────────────────
def test_text_chunks_respect_the_limit():
    text = 'a' * 9000 + '😀' * 10
    parts = list(tb._text_chunks(text, limit=4000))
    assert ''.join(parts) == text
    assert all(len(part.encode('utf-16-le')) // 2 <= 4000 for part in parts)
