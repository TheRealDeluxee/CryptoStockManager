"""Loss limit alarm (total P&L of an own position below total_loss_limit) and the config.ini update
([alarm] -> [alertsconfig], new line total_loss_limit)."""
import configparser

import pytest

import classes as cl
import func
from conftest import make_prices

OLD_CONFIG = """# my own comment
[alarm]
one_day_price_change = 10
seven_day_price_change = 10
previous_alarm_change = 1
one_day_profit_limit = -2
test_mode = 0

[coingecko]
api_key = SECRET-KEY

[telegram]
token = SECRET-TEST-TOKEN
chat_id = 1
"""


def run_alarm(current, yesterday, watch_list=False, data_type='100d'):
    df = func.cut_to_window(func.calc_indicator_fuctions(make_prices()))
    alarms, _, _ = func.alarm(df, 'Test', watch_list, current, 0, 0, data_type, asset_type='stock',
                              yesterday_profit_pct=yesterday)
    return alarms


def test_alarm_on_the_day_the_limit_is_crossed(monkeypatch):
    monkeypatch.setattr(func, 'total_loss_limit', -5.0)
    alarms = run_alarm(-6.2, -4.1)
    assert alarms['211']['kind'] == 'alarm'
    assert 'Total P&amp;L -6.2 % (your limit -5 %)' in alarms['211']['msg']
    assert 'Potential sell' in alarms['211']['msg']


@pytest.mark.parametrize('current, yesterday, watch_list, data_type', [
    (-6.2, -5.5, False, '100d'),    # already below yesterday: no new alarm
    (-4.0, -3.0, False, '100d'),    # still above the limit
    (-6.2, -4.1, True, '100d'),     # watched asset (not owned)
    (-6.2, -4.1, False, '100h'),    # hourly analysis
    (-6.2, None, False, '100d'),    # no value for yesterday
])
def test_no_alarm(monkeypatch, current, yesterday, watch_list, data_type):
    monkeypatch.setattr(func, 'total_loss_limit', -5.0)
    assert '211' not in run_alarm(current, yesterday, watch_list, data_type)


def test_loss_limit_can_be_an_info_or_off(monkeypatch):
    monkeypatch.setattr(func, 'total_loss_limit', -5.0)
    monkeypatch.setattr(func, 'alert_settings', {'loss_limit': ('info', 'alarm')})
    assert run_alarm(-6.2, -4.1)['211']['kind'] == 'info'
    monkeypatch.setattr(func, 'alert_settings', {'loss_limit': ('off', 'alarm')})
    assert '211' not in run_alarm(-6.2, -4.1)


def test_update_config_renames_section_and_adds_settings(sandbox):
    path = sandbox.path / 'config.ini'
    path.write_text(OLD_CONFIG, encoding='utf-8')
    changes = cl.update_config(str(path))
    assert 'section [alarm] renamed to [alertsconfig]' in changes
    assert 'total_loss_limit = -5 added to [alertsconfig]' in changes
    text = path.read_text(encoding='utf-8')
    assert '[alarm]' not in text and '[alertsconfig]' in text
    assert '# my own comment' in text and 'token = SECRET-TEST-TOKEN' in text and 'api_key = SECRET-KEY' in text
    assert text.index('total_loss_limit') < text.index('[coingecko]')     # inside [alertsconfig]
    assert '[alerts]' in text and 'loss_limit     = alarm; alarm' in text
    assert 'empty section [anthropic] added (optional Claude API key)' in changes
    assert chr(10).join(cl.ANTHROPIC_SECTION) in text
    config = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'))
    config.read(path, encoding='utf-8')
    assert config.getfloat('alertsconfig', 'total_loss_limit') == -5
    assert config.getint('alertsconfig', 'test_mode') == 0
    assert config.get('anthropic', 'api_key') == ''                        # empty = Claude API not used
    assert cl.update_config(str(path)) == []                               # second start: nothing to do
    assert path.read_text(encoding='utf-8') == text


def test_update_config_keeps_own_limit(sandbox):
    path = sandbox.path / 'config.ini'
    path.write_text(OLD_CONFIG.replace('[alarm]', '[alertsconfig]\ntotal_loss_limit = -12'), encoding='utf-8')
    changes = cl.update_config(str(path))
    assert not any('total_loss_limit' in change or 'renamed' in change for change in changes)
    assert 'total_loss_limit = -12' in path.read_text(encoding='utf-8')
