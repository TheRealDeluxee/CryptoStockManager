"""Plan alerts (planalert.ini): file, conversion from valuealert.ini and the hourly/daily check.
No internet: current price and daily closes are given by the test."""
import datetime

import pandas as pd
import pytest

import classes as cl
import func

TODAY = datetime.date.today()


def plan_file(sandbox):
    return (sandbox.path / 'planalert.ini').read_text(encoding='utf-8')


# ── file ─────────────────────────────────────────────────────────────────────
def test_add_read_replace_delete(sandbox):
    name, asset_type, replaced = cl.add_plan('APC.DE', 312.15, 2, 358.93, 310, 'quarter position')
    assert (name, asset_type, replaced) == ('Apple', 'stock', False)
    assert plan_file(sandbox).startswith('# CaSa plan alerts')          # new file with explanations
    plan, = cl.read_plans()
    assert (plan['support'], plan['max_entry'], plan['target'], plan['stop']) == (312.15, 2, 358.93, 310)
    assert plan['status'] == 'waiting' and plan['start'] == TODAY and plan['note'] == 'quarter position'
    assert cl.plan_zone_top(plan) == pytest.approx(318.393)
    assert 'Buy zone' in cl.plan_text(plan) and 'stop' in cl.plan_text(plan)
    cl.add_plan('bitcoin', 60000, 5, None, None, '')
    _, _, replaced = cl.add_plan('APC.DE', 300, 0, 350, None, 'new plan')   # one plan per asset
    assert replaced
    plans = {p['symbol']: p for p in cl.read_plans()}
    assert set(plans) == {'apc.de', 'bitcoin'} and plans['apc.de']['note'] == 'new plan'
    assert plans['apc.de']['stop'] is None and plans['bitcoin']['target'] is None
    cl.delete_plan('APC.DE')
    assert [p['symbol'] for p in cl.read_plans()] == ['bitcoin']
    assert '[APC.DE]' not in plan_file(sandbox)


@pytest.mark.parametrize('support, max_entry, target, stop, message', [
    (None, 0, None, None, 'support or a target'),
    (100, 2, 101, None, 'target must be above'),
    (100, 2, 120, 130, 'stop must be below the target'),
    (100, 2, 120, 105, 'stop must not be above'),
])
def test_invalid_plans(support, max_entry, target, stop, message):
    with pytest.raises(ValueError, match=message):
        cl.add_plan('APC.DE', support, max_entry, target, stop, '')


def test_unknown_asset_and_bad_file(sandbox):
    with pytest.raises(ValueError, match='not in assets.ini'):
        cl.add_plan('XYZ', 100, 2, 120, 90, '')
    (sandbox.path / 'planalert.ini').write_text('[APC.DE]\nsupport = 100\nprice = 3\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unknown setting price'):
        cl.read_plans()


def test_status_line_written(sandbox):
    cl.add_plan('APC.DE', 100, 2, 120, 95, '')
    cl.set_plan_value('APC.DE', 'status', 'ended 20/11/2026: daily close below stop')
    plan, = cl.read_plans()
    assert plan['status'] == 'ended' and 'daily close below stop' in plan['status_text']


def test_convert_value_alerts(sandbox):
    (sandbox.path / 'valuealert.ini').write_text(
        "# old\n[alerts]\n"
        "a = APC.DE; below; 312.15; wait for rebound\n"
        "b = APC.DE; above; 358.93; take profits\n"
        "c = bitcoin; below; 50000; buy small\n"
        "d = bitcoin; below; 45000; second\n", encoding='utf-8')
    changes = cl.convert_value_alerts()
    assert '2 plans' in changes[0]
    assert not (sandbox.path / 'valuealert.ini').exists() and (sandbox.path / 'valuealert.old.ini').exists()
    plans = {p['symbol']: p for p in cl.read_plans()}
    assert plans['apc.de']['support'] == 312.15 and plans['apc.de']['target'] == 358.93
    assert plans['apc.de']['max_entry'] == 0                       # alert only at or below the mark, as before
    assert plans['apc.de']['note'] == 'wait for rebound | take profits'
    assert plans['bitcoin']['support'] == 50000 and 'also below 45000.00' in plans['bitcoin']['note']
    assert cl.convert_value_alerts() == []                         # second start: nothing to do


# ── check in the main program ───────────────────────────────────────────────
def closes(*values):
    days = pd.date_range(end=pd.Timestamp(TODAY), periods=len(values), freq='D')
    return pd.Series(values, index=days, dtype=float)


@pytest.fixture
def manager(sandbox, monkeypatch):
    import main
    m = main.CryptoStockManager.__new__(main.CryptoStockManager)
    m.daily_prices, m.plan_stop_checked, m.plan_zone_sent, m.stock_update = {}, {}, {}, True
    m._is_stock_trading_hours = lambda: None
    m.analysed = []
    m.single_daily_analysis = lambda symbol, asset_type: m.analysed.append(symbol)
    m.price = 0.0
    monkeypatch.setattr(func, 'current_price', lambda symbol, asset_type: m.price)
    return m


def messages(sandbox):
    return [text for text in sandbox.sent if isinstance(text, str)]


def start_plan(symbol='APC.DE', start=None):
    cl.add_plan(symbol, 100, 2, 120, 95, 'quarter only', today=start or TODAY - datetime.timedelta(days=10))


def test_buy_zone_once_not_held(sandbox, manager):
    start_plan()
    manager.daily_prices[('stock', 'apc.de')] = closes(110, 105, 101)
    manager.price = 101.5                                          # inside 100–102
    manager.plan_alert_check()
    sent = messages(sandbox)
    assert len(sent) == 1 and '🟢 Buy zone: Apple' in sent[0] and 'quarter only' in sent[0]
    assert manager.analysed == ['apc.de']
    assert cl.read_plans()[0]['status'] == 'buy_zone'
    manager.plan_alert_check()                                     # still in the zone: no second message
    assert len(messages(sandbox)) == 1
    manager.price = 110                                            # leaves the zone, comes back the same day:
    manager.plan_alert_check()                                     # no second message (at most one per day)
    manager.price = 100
    manager.plan_alert_check()
    assert len(messages(sandbox)) == 1 and cl.read_plans()[0]['status'] == 'buy_zone'
    yesterday = TODAY - datetime.timedelta(days=1)                 # next day: message again
    manager.plan_zone_sent = {key: yesterday for key in manager.plan_zone_sent}
    manager.price = 110
    manager.plan_alert_check()
    manager.price = 100
    manager.plan_alert_check()
    assert len(messages(sandbox)) == 2


def test_held_add_on_zone_and_target(sandbox, manager):
    start_plan('RHM.DE')                                           # RHM.DE is held in the sample transactions
    manager.daily_prices[('stock', 'rhm.de')] = closes(110, 105)
    manager.price = 100
    manager.plan_alert_check()
    assert 'Add-on zone' in messages(sandbox)[-1]
    manager.price = 121
    manager.plan_alert_check()
    assert '💰 Take partial profits: Rheinmetall' in messages(sandbox)[-1]


def test_target_not_held_no_message(sandbox, manager):
    start_plan()
    manager.daily_prices[('stock', 'apc.de')] = closes(110, 105)
    manager.price = 125
    manager.plan_alert_check()
    assert messages(sandbox) == [] and cl.read_plans()[0]['status'] == 'target'


@pytest.mark.parametrize('symbol, title', [('APC.DE', '❌ Setup cancelled'), ('RHM.DE', '🔴 Exit')])
def test_stop_on_daily_close_ends_plan(sandbox, manager, symbol, title):
    start_plan(symbol)
    manager.daily_prices[('stock', symbol.lower())] = closes(110, 94, 99)   # yesterday 94 < stop 95
    manager.price = 99
    manager.plan_alert_check()
    sent = messages(sandbox)
    assert len(sent) == 1 and title in sent[0] and '94.00' in sent[0]
    plan, = cl.read_plans()
    assert plan['status'] == 'ended' and 'below stop 95.00' in plan['status_text']
    manager.plan_alert_check()                                     # ended plans are not checked any more
    assert len(messages(sandbox)) == 1


def test_stop_ignores_today_and_days_before_start(sandbox, manager):
    start_plan(start=TODAY - datetime.timedelta(days=1))
    # 90 before the start, 96 yesterday, 80 today (not a finished day yet)
    manager.daily_prices[('stock', 'apc.de')] = closes(90, 96, 80)
    manager.price = 110
    manager.plan_alert_check()
    assert messages(sandbox) == [] and cl.read_plans()[0]['status'] == 'waiting'


def test_stock_outside_trading_hours_still_checks_stop(sandbox, manager):
    start_plan()
    manager.stock_update = False
    manager.daily_prices[('stock', 'apc.de')] = closes(110, 94, 99)
    manager.plan_alert_check()
    assert 'Setup cancelled' in messages(sandbox)[0]


# ── /plan_alert in Telegram ──────────────────────────────────────────────────
def test_bot_save_list_delete(sandbox, monkeypatch):
    import asyncio
    import types
    import telegram_bot as tb
    queue, sent, edits = [], [], []
    monkeypatch.setattr(tb, '_authorized', lambda *a: True)
    monkeypatch.setattr(tb, '_cmdq', lambda app: types.SimpleNamespace(put=queue.append))
    monkeypatch.setattr(func, 'current_price', lambda symbol, asset_type: 330.0)
    app = types.SimpleNamespace(bot_data={'chat_id': 1})

    async def reply_text(text, **kwargs):
        sent.append((text, kwargs.get('reply_markup')))
    message = types.SimpleNamespace(reply_text=reply_text, text='')

    def run(*args):
        context = types.SimpleNamespace(args=list(args), application=app, user_data={}, bot_data=app.bot_data)
        asyncio.run(tb._cmd_plan(types.SimpleNamespace(message=message), context))

    def tap(data):
        async def answer(*a, **k):
            pass

        async def edit_message_text(text, **kwargs):
            edits.append(text)
        query = types.SimpleNamespace(data=data, answer=answer, edit_message_text=edit_message_text)
        context = types.SimpleNamespace(application=app)
        asyncio.run(tb._cb_plan_delete(types.SimpleNamespace(callback_query=query, effective_chat=None), context))

    run('Apple', '312,15', '2%', '358.93', '-', 'quarter', 'after', 'rebound')
    assert sent[-1][0].startswith('Plan saved: Apple') and 'Current price 330.00' in sent[-1][0].replace(',', '')
    assert queue == [{'cmd': 'plan_alerts'}]
    plan, = cl.read_plans()
    assert plan['support'] == 312.15 and plan['stop'] is None and plan['note'] == 'quarter after rebound'
    run('Apple', '312')                                            # too few numbers: format help
    assert sent[-1][0].startswith('Format: /plan_alert')
    run()
    text, markup = sent[-1]
    assert 'Apple' in text and 'waiting' in text
    data = markup.inline_keyboard[0][0].callback_data
    assert data == 'pldel|APC.DE'
    tap(data)
    assert edits[-1].startswith('Delete this plan?')
    tap('pldelok|APC.DE')
    assert edits[-1] == 'Plan deleted: Apple' and cl.read_plans() == []
