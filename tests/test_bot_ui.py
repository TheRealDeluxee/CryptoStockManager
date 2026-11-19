"""Bot usability: 🤖 AI prompt button below alarms, collapsible details, /menu, /portfolio, /status, plain-text fallback."""
import asyncio
import types

import pandas as pd
import pytest

import classes as cl
import func
import telegram_bot as tb
from conftest import make_prices


@pytest.fixture
def ui(monkeypatch):
    """Fake chat: messages, edits and commands to the main program are collected."""
    queue, sent, edits = [], [], []
    monkeypatch.setattr(tb, '_authorized', lambda *a: True)
    monkeypatch.setattr(tb, '_cmdq', lambda app: types.SimpleNamespace(put=queue.append))
    app = types.SimpleNamespace(bot_data={'chat_id': 1})

    async def reply_text(text, **kwargs):
        sent.append((text, kwargs.get('reply_markup')))

    message = types.SimpleNamespace(reply_text=reply_text, text='')

    def context(args=()):
        return types.SimpleNamespace(args=list(args), application=app, user_data={}, bot_data=app.bot_data)

    def tap(handler, data):
        async def answer(*a, **k):
            pass

        async def edit_message_text(text, **kwargs):
            edits.append((text, kwargs.get('reply_markup')))

        async def edit_message_reply_markup(reply_markup=None):
            edits.append(('', reply_markup))
        query = types.SimpleNamespace(data=data, answer=answer, edit_message_text=edit_message_text,
                                      edit_message_reply_markup=edit_message_reply_markup, message=message)
        asyncio.run(handler(types.SimpleNamespace(callback_query=query, effective_chat=None), context()))

    def run(command, *args):
        asyncio.run(command(types.SimpleNamespace(message=message), context(args)))

    return types.SimpleNamespace(queue=queue, sent=sent, edits=edits, tap=tap, run=run)


def callbacks(markup):
    return [button.callback_data for row in markup.inline_keyboard for button in row]


# ── 🤖 AI prompt below alarms and infos ────────────────────────────────────
def test_alarm_messages_carry_the_ai_button(sandbox, monkeypatch):
    images = []
    monkeypatch.setattr(func, 'plot_and_save', lambda *a, **k: None)
    monkeypatch.setattr(func, 'pushover_image', lambda name, msg, *a, **k: images.append(k.get('buttons')))
    asset = cl.CryptoStock('RHM.DE', 'Rheinmetall', 'stock')
    asset.plot_send(pd.DataFrame(), '100d', {'121': {'msg': 'x', 'priority': 0}})
    assert images == [[[('🤖 AI prompt', 'aibtn|stock|rhm.de')]]]   # one message, one row, one button


def test_ai_button_queues_prompt_and_keeps_the_alarm(ui):
    ui.tap(tb._cb_ai_button, 'aibtn|stock|rhm.de')
    assert ui.queue == [{'cmd': 'ai', 'symbol': 'rhm.de', 'type': 'stock'}]
    assert ui.edits == []                                    # the alarm message is not changed
    assert 'Creating AI prompt for Rheinmetall' in ui.sent[-1][0]


# ── collapsible details ───────────────────────────────────────────────────
def test_figure_blocks_are_collapsible():
    df = func.cut_to_window(func.calc_indicator_fuctions(make_prices()))
    message = func.analysis_details(df, rank=5)['message']
    summary, details = message.split('<blockquote expandable>')
    assert 'Score 5/10' in summary and 'Long-term-Analysis' in summary          # traffic lights stay visible
    assert details.endswith('</blockquote>') and '<pre>' in details             # figures collapsed
    assert '<blockquote' not in func._strip_telegram_html(message)


def test_rejected_html_is_sent_as_plain_text():
    sent = []

    async def send_message(chat_id, text, parse_mode=None, **kwargs):
        if parse_mode == 'HTML':
            raise tb.BadRequest("Can't parse entities")
        sent.append(text)
    app = types.SimpleNamespace(bot=types.SimpleNamespace(send_message=send_message))
    asyncio.run(tb._send_text(app, 1, '<b>Title</b>\n<blockquote expandable>details</blockquote>', 'HTML'))
    assert sent == ['Title\ndetails']


# ── /record: buy, sell and sell all in one command ─────────────────────────
def test_record_buttons(ui):
    ui.run(tb._cmd_record)
    text, markup = ui.sent[-1]
    assert text == 'What do you want to record?'
    assert callbacks(markup) == ['rec|buy', 'rec|sell', 'rec|sellall', 'rec|cancel']


@pytest.mark.parametrize('data, expected', [('rec|buy', 'Which asset did you buy?'),
                                            ('rec|sell', 'Which asset did you sell?'),
                                            ('rec|sellall', 'Sell the full position of which asset?')])
def test_record_button_starts_input(ui, data, expected):
    ui.tap(tb._cb_record, data)
    assert expected in ui.sent[-1][0]


def test_record_cancel_and_unknown(ui):
    ui.tap(tb._cb_record, 'rec|cancel')
    assert ui.edits[-1][0] == 'Nothing recorded.'
    ui.run(tb._cmd_record, 'kaufen')
    assert ui.sent[-1][0].startswith('Unknown: kaufen')


def test_record_in_one_line(ui, sandbox):
    ui.run(tb._cmd_record, 'buy', 'Nvidia', '450', '3', '05/01/2025')
    assert 'Buy saved: Nvidia' in ui.sent[-1][0]
    ui.run(tb._cmd_record, 'SELL', 'Nvidia', '200', '1')
    assert 'Sell saved: Nvidia' in ui.sent[-1][0]


# ── /status and /portfolio in the main program ────────────────────────────
def fake_manager(sandbox):
    import main
    manager = types.SimpleNamespace(
        started=0, bot_alive=lambda: True, last_runs={}, test_mode=0, crypto_items_hh=[], crypto_items_hd=[1],
        stock_items_hd=[1, 2], alarms_sent=3, version='0.1.11', df_crypto_hd=pd.DataFrame(), df_stock_hd=pd.DataFrame())
    manager._health = lambda: main.CryptoStockManager._health(manager)
    return main, manager


def test_status_message(sandbox):
    main, manager = fake_manager(sandbox)
    main.CryptoStockManager.send_status(manager)
    text = sandbox.sent[-1]
    assert 'CaSa status' in text and 'version 0.1.11' in text and '3 assets in the alarm analysis' in text
    assert '4 transactions' in text and 'alarms since start 3' in text


def test_portfolio_without_daily_results(sandbox):
    main, manager = fake_manager(sandbox)
    main.CryptoStockManager.send_portfolio(manager)
    assert 'No daily results yet' in sandbox.sent[-1]
