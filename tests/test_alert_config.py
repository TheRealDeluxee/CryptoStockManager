"""Alarms vs. infos: config.ini [alerts] with one column for stocks and one for crypto (also via /smart_alert_config)."""
import asyncio
import types

import numpy as np
import pytest

import classes as cl
import func
import telegram_bot as tb
from conftest import make_prices
from test_analysis import analyse

CONFIG = """[alertsconfig]
seven_day_price_change = 10
test_mode = 0

[telegram]
token = SECRET-TEST-TOKEN
chat_id = 1
"""


@pytest.fixture
def config(sandbox):
    (sandbox.path / 'config.ini').write_text(CONFIG, encoding='utf-8')
    return sandbox.path / 'config.ini'


def line_of(path, key):
    return next(l for l in path.read_text(encoding='utf-8').splitlines() if l.startswith(key + ' '))


# ── config.ini [alerts] ────────────────────────────────────────────────────
def test_section_is_required(config):
    with pytest.raises(ValueError, match=r'\[alerts\] is missing'):
        cl.read_alert_settings()


def test_ensure_creates_section_with_explanation_and_probabilities(config):
    changes = cl.ensure_alert_section()
    assert len(changes) == len(cl.ALERTS)
    text = config.read_text(encoding='utf-8')
    assert 'token = SECRET-TEST-TOKEN' in text and 'seven_day_price_change = 10' in text
    assert 'name = <stocks>; <crypto>' in text and 'alarm = message "Potential buy/sell"' in text
    assert line_of(config, 'oversold').startswith('oversold       = alarm; off   # ')
    assert 'stocks 58 % (normal 52 %)' in line_of(config, 'oversold')
    assert line_of(config, 'high100').startswith('high100        = info;  alarm # ')
    settings = cl.read_alert_settings()
    assert settings['score8'] == ('info', 'alarm') and settings['positive5'] == ('alarm', 'alarm')
    assert cl.ensure_alert_section() == []                 # second call changes nothing
    assert config.read_text(encoding='utf-8') == text


def test_old_format_is_converted_without_changing_the_effect(config):
    with open(config, 'a', encoding='utf-8') as file:
        file.write('\n[alerts]\n# old comment\npositive5 = all   # x\nscore8 = crypto\noversold = stock\n'
                   'high100 = off\nnegative5 = 1\n\n[other]\nx = 1\n')
    changes = cl.ensure_alert_section()
    assert 'positive5 converted' in changes and 'pattern added' in changes
    settings = cl.read_alert_settings()
    assert settings['positive5'] == ('alarm', 'alarm')
    assert settings['score8'] == ('off', 'alarm')
    assert settings['oversold'] == ('alarm', 'off')
    assert settings['high100'] == ('off', 'off')
    assert settings['negative5'] == ('info', 'info')
    assert settings['pattern'] == ('info', 'info')          # missing line: default
    text = config.read_text(encoding='utf-8')
    assert text.count('[alerts]') == 1 and '[other]\nx = 1' in text and 'token = SECRET-TEST-TOKEN' in text
    assert text.index('pattern') < text.index('[other]')


def test_missing_line_is_added_and_own_values_stay(config):
    cl.ensure_alert_section()
    text = config.read_text(encoding='utf-8')
    config.write_text(text.replace(line_of(config, 'low100') + '\n', '')
                      .replace(line_of(config, 'score8'), 'score8 = alarm; off   # my comment'), encoding='utf-8')
    assert cl.ensure_alert_section() == ['low100 added']
    assert 'score8 = alarm; off   # my comment' in config.read_text(encoding='utf-8')
    assert cl.read_alert_settings()['score8'] == ('alarm', 'off')


def test_invalid_value(config):
    cl.ensure_alert_section()
    config.write_text(config.read_text(encoding='utf-8').replace(line_of(config, 'pattern'), 'pattern = alarm; maybe'),
                      encoding='utf-8')
    with pytest.raises(ValueError, match='pattern'):
        cl.read_alert_settings()


def test_set_changes_one_column_and_keeps_the_rest(config):
    cl.ensure_alert_section()
    cl.set_alert_setting('score8', 'stock', 'off')
    cl.set_alert_setting('high100', 'crypto', 'info')
    assert cl.read_alert_settings()['score8'] == ('off', 'alarm')
    assert cl.read_alert_settings()['high100'] == ('info', 'info')
    assert 'crypto 56 % (normal 49 %)' in line_of(config, 'score8')
    assert 'token = SECRET-TEST-TOKEN' in config.read_text(encoding='utf-8')


@pytest.mark.parametrize('kinds, stock, crypto', [(('alarm', 'info'), 'alarm', 'info'), (('off', 'alarm'), None, 'alarm'),
                                                  (('info', 'off'), 'info', None), (('off', 'off'), None, None)])
def test_alert_active(kinds, stock, crypto):
    settings = {'score8': kinds}
    assert func.alert_active(settings, 'score8', 'stock') == stock
    assert func.alert_active(settings, 'score8', 'crypto') == crypto


# ── alarm or info in the analysis ──────────────────────────────────────────
def rising_then_jump():
    data = make_prices(days=300, drift=0.002, volatility=0.003, seed=3)
    data.loc[data.index[-1], 'Price'] *= 1.13
    data.loc[data.index[-1], ['PriceHigh', 'PriceLow']] = data['Price'].iloc[-1] * np.array([1.01, 0.99])
    return data


def test_kind_follows_the_setting(monkeypatch):
    assert analyse(rising_then_jump())[2]['121']['kind'] == 'alarm'
    monkeypatch.setattr(func, 'alert_settings', {'positive5': ('info', 'off')})
    alarms = analyse(rising_then_jump(), asset_type='stock')[2]
    assert alarms['121']['kind'] == 'info' and 'Info: Positive-5' in alarms['121']['msg']
    assert '121' not in analyse(rising_then_jump(), asset_type='crypto')[2]


def test_info_can_become_an_alarm(monkeypatch):
    info = analyse(rising_then_jump(), asset_type='stock')[2]['361']    # default stocks: info
    assert info['kind'] == 'info' and 'Info: New 100-day high' in info['msg'] and info['priority'] == 0
    crypto = analyse(rising_then_jump(), asset_type='crypto')[2]['361']  # default crypto: alarm
    assert crypto['kind'] == 'alarm' and 'Potential buy' in crypto['msg'] and 'crypto rose further' in crypto['msg']


def test_sharp_drop_and_100_day_low_are_infos():
    data = make_prices(days=300, drift=0.0005, volatility=0.003, seed=5)
    data.loc[data.index[-1], 'Price'] *= 0.85              # -15 % on the last day
    alarms = analyse(data, watch_list=True)[2]
    assert alarms['321']['kind'] == 'info' and 'Sharp drop' in alarms['321']['msg']
    assert alarms['362']['kind'] == 'info'
    assert 'Watched asset' in alarms['321']['msg']


def test_journal_records_alarms_but_not_infos(sandbox):
    import main
    import pandas as pd
    item = types.SimpleNamespace(type='crypto', symbol='bitcoin', search='Bitcoin',
                                 state=types.SimpleNamespace(score=9, watch_list=False))
    df = pd.DataFrame({'Date': pd.to_datetime(['2026-11-18']), 'Price': [80000.0]})
    manager = types.SimpleNamespace(test_mode=0)
    main.CryptoStockManager._journal_alarms(manager, item, df, '100d',
                                            {'361': {'kind': 'alarm'}, '301': {'kind': 'info'}})
    rows = cl.read_journal()
    assert [(row['code'], row['alarm']) for row in rows] == [('361', '100-day high')]


# ── /smart_alert_config ────────────────────────────────────────────────────
def test_bot_alert_config(config, monkeypatch):
    monkeypatch.setattr(tb, '_authorized', lambda *a: True)
    sent, markups = [], []

    async def reply_text(text, **kwargs):
        sent.append((text, kwargs.get('reply_markup')))

    async def edit_message_reply_markup(reply_markup=None):
        markups.append(reply_markup)

    async def answer():
        pass
    app = types.SimpleNamespace(bot_data={'chat_id': 1})
    update = types.SimpleNamespace(message=types.SimpleNamespace(reply_text=reply_text))
    asyncio.run(tb._cmd_alert_config(update, types.SimpleNamespace(args=[], application=app)))
    text, keyboard = sent[-1]
    first_row = [button.text for button in keyboard.inline_keyboard[0]]
    assert first_row == ['📈 Positive-5: 🔔 alarm', '🪙 🔔 alarm']
    assert 'alarm → info → off' in ''.join(t for t, _ in sent)
    for expected in ('off', 'alarm', 'info'):           # stocks column of score8 starts at 'info'
        query = types.SimpleNamespace(data='acfg|score8|stock', answer=answer,
                                      edit_message_reply_markup=edit_message_reply_markup)
        asyncio.run(tb._cb_alert_config(types.SimpleNamespace(callback_query=query),
                                        types.SimpleNamespace(application=app)))
        assert cl.read_alert_settings()['score8'] == (expected, 'alarm')
    assert 'token = SECRET-TEST-TOKEN' in config.read_text(encoding='utf-8')
