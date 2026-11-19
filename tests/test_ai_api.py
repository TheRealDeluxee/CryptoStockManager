"""/ai_api_decision3s and /ai_api_plan: Claude answers from the /ai_prompt data; the plan can be saved
in planalert.ini with a button. No real API request: func.claude is replaced."""
import asyncio
import types

import pytest

import classes as cl
import func
import main
import telegram_bot as tb

PLAN_ANSWER = """Wait: price well above 312.15 -> stay out.
First buy: rebound at 312.15 and close above it -> buy 1/4.
Plan fails: daily close below 310 -> cancel.
```planalert
support = 312.15
max_entry = 2
target = 358.93
stop = 310.00
note = 1/4 after a rebound, add 1/4 after a retest
```"""


@pytest.fixture
def manager(sandbox, monkeypatch):
    m = main.CryptoStockManager.__new__(main.CryptoStockManager)
    m.pending_plans, m.daily_prices, m.plan_stop_checked, m.stock_update = {}, {}, {}, True
    m.plan_zone_sent = {}
    m._is_stock_trading_hours = lambda: None
    m._ai_question = lambda symbol, asset_type: ('Apple', symbol.upper(), 'PROMPT')
    m.asked = []

    def fake_claude(prompt, **kwargs):
        m.asked.append((prompt, kwargs))
        usage = {'tokens_in': 12345, 'tokens_out': 678, 'searches': 3, 'cost': 0.0617, 'seconds': 41.2,
                 'models': ['claude-sonnet-5-5']}
        return (m.answer, usage) if kwargs.get('with_usage') else m.answer
    monkeypatch.setattr(func, 'claude', fake_claude)
    monkeypatch.setattr(func, 'anthropic_api_key', 'sk-ant-TEST')
    monkeypatch.setattr(func, 'current_price', lambda symbol, asset_type: 400.0)
    return m


def test_parse_plan_block():
    text, block = main.split_plan_answer(PLAN_ANSWER)
    assert '```' not in text and text.startswith('Wait:')
    plan = main.parse_plan_block(block)
    assert plan == {'support': 312.15, 'max_entry': 2.0, 'target': 358.93, 'stop': 310.0,
                    'note': '1/4 after a rebound, add 1/4 after a retest'}
    assert main.parse_plan_block('support = 100\nmax_entry = 0\ntarget =\nstop = -\nnote = x')['target'] is None
    with pytest.raises(ValueError, match='no planalert block'):
        main.parse_plan_block(main.split_plan_answer('only text')[1])
    with pytest.raises(ValueError, match='stop must be below the target'):
        main.parse_plan_block('target = 100\nstop = 120')


def test_no_key(sandbox, manager, monkeypatch):
    monkeypatch.setattr(func, 'anthropic_api_key', '')
    manager.ai_api('apc.de', 'stock', 'decision')
    assert manager.asked == [] and 'No Claude API key' in sandbox.sent[-1]


def test_decision(sandbox, manager):
    manager.answer = 'Wait. Support 312 €, resistance 359 €. Score 4/10, risk medium.'
    manager.ai_api('apc.de', 'stock', 'decision')
    prompt, kwargs = manager.asked[0]
    assert prompt.startswith('PROMPT') and 'exactly three sentences' in prompt and kwargs['web_search']
    assert '🤖 Decision: Apple' in sandbox.sent[-1] and 'Score 4/10' in sandbox.sent[-1]
    assert '12,345 tokens in · 678 out · 3 web searches · $0.062' in sandbox.sent[-1]
    assert 'claude-sonnet-5-5' in sandbox.sent[-1]


def test_plan_save_with_button(sandbox, manager, monkeypatch):
    buttons = []
    monkeypatch.setattr(func, 'pushover', lambda message, *a, **k: (sandbox.sent.append(message),
                                                                     buttons.append(k.get('buttons'))))
    manager.answer = PLAN_ANSWER
    manager.ai_api('apc.de', 'stock', 'plan')
    assert '```planalert' in manager.asked[0][0]
    message = sandbox.sent[-1]
    assert '🤖 Trading plan: Apple' in message and 'support   = 312.15' in message and '```' not in message
    assert '🧾 12,345 tokens in' in message
    (save, discard), = buttons[-1]
    assert save[0] == '✅ Save as plan' and save[1].startswith('aiplanok|') and discard[1].startswith('aiplanno|')
    assert cl.read_plans() == []                                   # nothing saved before the tap
    manager.save_ai_plan(save[1].split('|')[1], True)
    plan, = cl.read_plans()
    assert (plan['symbol'], plan['support'], plan['target'], plan['stop']) == ('apc.de', 312.15, 358.93, 310.0)
    assert any('Plan saved in planalert.ini: Apple' in str(text) for text in sandbox.sent)
    manager.save_ai_plan(save[1].split('|')[1], True)              # second tap: already handled
    assert 'no longer available' in sandbox.sent[-1]


def test_plan_discard_and_unusable_block(sandbox, manager):
    manager.answer = PLAN_ANSWER
    manager.ai_api('apc.de', 'stock', 'plan')
    key, = manager.pending_plans
    manager.save_ai_plan(key, False)
    assert 'discarded' in sandbox.sent[-1] and cl.read_plans() == []
    manager.answer = 'Some plan without block'
    manager.ai_api('apc.de', 'stock', 'plan')
    assert 'No plan to save' in sandbox.sent[-1] and manager.pending_plans == {}


def test_api_error_reported(sandbox, manager, monkeypatch):
    def failing(prompt, **kwargs):
        raise RuntimeError('Claude API: too many requests, try again later')
    monkeypatch.setattr(func, 'claude', failing)
    manager.ai_api('apc.de', 'stock', 'decision')
    assert 'too many requests' in sandbox.sent[-1]


# ── Telegram ─────────────────────────────────────────────────────────────────
@pytest.fixture
def chat(monkeypatch):
    queue, sent = [], []
    monkeypatch.setattr(tb, '_authorized', lambda *a: True)
    monkeypatch.setattr(tb, '_cmdq', lambda app: types.SimpleNamespace(put=queue.append))
    app = types.SimpleNamespace(bot_data={'chat_id': 1})

    async def reply_text(text, **kwargs):
        sent.append(text)
    message = types.SimpleNamespace(reply_text=reply_text, text='')

    def run(command, *args):
        context = types.SimpleNamespace(args=list(args), application=app, user_data={}, bot_data=app.bot_data)
        asyncio.run(command(types.SimpleNamespace(message=message), context))
    return types.SimpleNamespace(queue=queue, sent=sent, run=run, app=app)


def write_key(sandbox, key):
    (sandbox.path / 'config.ini').write_text(f'[anthropic]\napi_key = {key}\nmodel = claude-sonnet-5-5\n',
                                              encoding='utf-8')


def test_bot_without_key(sandbox, chat):
    write_key(sandbox, '')
    chat.run(tb._cmd_ai_plan, 'Apple')
    assert chat.queue == [] and 'No Claude API key' in chat.sent[-1]
    chat.run(tb._cmd_ai, 'Apple')                                  # /ai_prompt needs no key
    assert chat.queue == [{'cmd': 'ai', 'symbol': 'apc.de', 'type': 'stock'}]


def test_bot_with_key_and_save_button(sandbox, chat):
    write_key(sandbox, 'sk-ant-TEST')
    chat.run(tb._cmd_ai_decision, 'Apple')
    chat.run(tb._cmd_ai_plan, 'Apple')
    assert chat.queue == [{'cmd': 'ai_decision', 'symbol': 'apc.de', 'type': 'stock'},
                          {'cmd': 'ai_plan', 'symbol': 'apc.de', 'type': 'stock'}]
    assert 'about 1 minute' in chat.sent[-1]
    edits = []

    async def answer(*a, **k):
        pass

    async def edit_message_reply_markup(reply_markup=None):
        edits.append(reply_markup)
    query = types.SimpleNamespace(data='aiplanok|abc123', answer=answer,
                                  edit_message_reply_markup=edit_message_reply_markup)
    asyncio.run(tb._cb_ai_plan(types.SimpleNamespace(callback_query=query, effective_chat=None),
                               types.SimpleNamespace(application=chat.app)))
    assert chat.queue[-1] == {'cmd': 'ai_plan_save', 'key': 'abc123', 'save': True} and edits == [None]


# ── buttons at the same places as the AI prompt, only with a key ─────────────
def alarm_buttons(monkeypatch):
    import pandas as pd
    images = []
    monkeypatch.setattr(func, 'plot_and_save', lambda *a, **k: None)
    monkeypatch.setattr(func, 'pushover_image', lambda name, msg, *a, **k: images.append(k.get('buttons')))
    cl.CryptoStock('RHM.DE', 'Rheinmetall', 'stock').plot_send(pd.DataFrame(), '100d', {'121': {'msg': 'x', 'priority': 0}})
    return images[0]


def test_alarm_buttons_with_and_without_key(sandbox, monkeypatch):
    monkeypatch.setattr(func, 'anthropic_api_key', '')
    assert alarm_buttons(monkeypatch) == [[('🤖 AI prompt', 'aibtn|stock|rhm.de')]]
    monkeypatch.setattr(func, 'anthropic_api_key', 'sk-ant-TEST')
    assert alarm_buttons(monkeypatch) == [[('🤖 AI prompt', 'aibtn|stock|rhm.de'), ('🤖 AI decision (API)', 'aidbtn|stock|rhm.de'),
                                           ('🤖 AI plan (API)', 'aipbtn|stock|rhm.de')]]


@pytest.mark.parametrize('key', ['', 'sk-ant-TEST'])
def test_check_has_the_alarm_buttons(sandbox, monkeypatch, key):
    images, texts = [], []
    monkeypatch.setattr(func, 'pushover', lambda message, *a, **k: texts.append(message))
    monkeypatch.setattr(func, 'pushover_image', lambda name, msg, *a, **k: images.append(k.get('buttons')))
    monkeypatch.setattr(func, 'plot_and_save', lambda *a, **k: None)
    monkeypatch.setattr(func, 'anthropic_api_key', key)
    m = main.CryptoStockManager.__new__(main.CryptoStockManager)
    item = cl.CryptoStock('APC.DE', 'Apple', 'stock')
    item.state.watch_list = True
    m._daily_analysis = lambda symbol, asset_type: ('Apple', 'APC.DE', item, None, None, None, {'message': ''})
    m.single_daily_analysis('apc.de', 'stock', ask_ai=True)
    assert texts == []                                             # no separate question any more
    assert images == [func.ai_buttons('stock', 'APC.DE')]          # the same buttons as below alarms
    data = [d for row in images[0] for _, d in row]
    assert 'aibtn|stock|apc.de' in data
    assert ('aidbtn|stock|apc.de' in data and 'aipbtn|stock|apc.de' in data) == bool(key)
    m.single_daily_analysis('apc.de', 'stock')                     # e.g. after a plan alert: no buttons
    assert images[-1] is None


def test_ai_commands_not_in_the_command_list():
    import inspect
    source = inspect.getsource(tb._post_init)
    assert 'BotCommand("ai_prompt"' not in source and 'BotCommand("ai_api_plan"' not in source
    assert 'BotCommand("ai_api_decision3s"' not in source and 'BotCommand("record"' in source


def test_alarm_api_button(sandbox, chat):
    replies = []

    async def answer(*a, **k):
        replies.append(a[0] if a else '')

    async def reply_text(text, **k):
        replies.append(text)

    def tap(data):
        query = types.SimpleNamespace(data=data, answer=answer, message=types.SimpleNamespace(reply_text=reply_text))
        asyncio.run(tb._cb_ai_button(types.SimpleNamespace(callback_query=query, effective_chat=None),
                                     types.SimpleNamespace(application=chat.app)))
    write_key(sandbox, '')
    tap('aipbtn|stock|apc.de')
    assert chat.queue == [] and 'No Claude API key' in replies[-1]
    write_key(sandbox, 'sk-ant-TEST')
    tap('aipbtn|stock|apc.de')
    tap('aidbtn|stock|apc.de')
    assert chat.queue == [{'cmd': 'ai_plan', 'symbol': 'apc.de', 'type': 'stock'},
                          {'cmd': 'ai_decision', 'symbol': 'apc.de', 'type': 'stock'}]
