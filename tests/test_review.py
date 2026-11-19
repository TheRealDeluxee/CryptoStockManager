"""Checks from the code review: fast calculations give the same results, exchange-rate cache,
leading market for German exchanges, sending of long messages and charts."""
import asyncio
import os
import types

import numpy as np
import pandas as pd
import pytest

import func
import telegram_bot as tb
from conftest import make_prices


def roc_reference(series, window):
    # the former calculation: straight line per window, % change from its start to its end
    out = []
    for end in range(len(series)):
        y = series[max(0, end - window + 1):end + 1]
        if len(y) < window or np.isnan(y).any():
            out.append(np.nan)
            continue
        slope, intercept = np.polyfit(np.arange(window), y, 1)
        out.append((slope * (window - 1) + intercept) / intercept * 100 - 100)
    return np.array(out)


@pytest.mark.parametrize('column', ['Price', 'Volume'])
def test_fast_roc_equals_line_per_window(column):
    df = make_prices(days=200, seed=4)
    df.loc[50, column] = np.nan                 # a gap gives NaN for the windows containing it
    fast = func.ROC5(df, column, window=5).to_numpy()
    reference = roc_reference(df[column].to_numpy(float), 5)
    assert np.array_equal(np.isnan(fast), np.isnan(reference))
    assert np.nanmax(np.abs(fast - reference)) < 1e-9


def test_score_only_equals_analysis_details():
    full = func.calc_indicator_fuctions(make_prices(days=300, seed=9))
    for end in range(150, 301, 10):
        window = func.cut_to_window(full.iloc[:end])
        assert func.score_only(window) == func.analysis_details(window)['score']


def test_score_rank_equals_the_slow_way():
    full = func.calc_indicator_fuctions(make_prices(days=300, seed=12))
    scores = [func.analysis_details(func.cut_to_window(full.iloc[:end]))['score'] for end in range(201, 301)]
    slow = max(1, int(np.ceil(sum(s <= scores[-1] for s in scores) / len(scores) * 10)))
    assert func.score_rank(full) == slow


def test_exchange_rates_are_fetched_once_per_hour(monkeypatch):
    calls = []

    class Ticker:
        def __init__(self, symbol):
            calls.append(symbol)

        def history(self, **kwargs):
            return pd.DataFrame({'Close': [0.9, 0.91]}, index=pd.date_range('2026-11-17', periods=2, tz='UTC'))
    monkeypatch.setattr(func.yf, 'Ticker', Ticker)
    monkeypatch.setattr(func, '_fx_cache', {})
    for _ in range(3):
        func._fx_rates('USD', '2y')
    assert calls == ['USDEUR=X']
    func._fx_rates('GBP', '2y')
    assert calls == ['USDEUR=X', 'GBPEUR=X']


@pytest.mark.parametrize('symbol, market', [('RHM.DE', 'DAX'), ('D7G.SG', 'DAX'), ('NTH.F', 'DAX'), ('X.MU', 'DAX'),
                                            ('FMNB.DU', 'DAX'), ('LDO.MI', 'FTSE MIB'), ('VFF', 'S&P 500')])
def test_leading_market(monkeypatch, symbol, market):
    class Ticker:
        def __init__(self, ticker):
            pass

        def history(self, **kwargs):
            return pd.DataFrame({'Close': [1.0]})
    monkeypatch.setattr(func.yf, 'Ticker', Ticker)
    assert func.market_prices('stock', symbol)[0] == market


# ── sending (Telegram is replaced by a fake bot) ───────────────────────────
def fake_app(fail_photo=False):
    sent = []

    async def send_message(chat_id, text, parse_mode=None, **kwargs):
        sent.append(('text', text, parse_mode))

    async def send_photo(chat_id, photo, caption=None, parse_mode=None, **kwargs):
        if fail_photo:
            raise ConnectionError('network down')
        sent.append(('photo', caption, parse_mode))
    return types.SimpleNamespace(bot=types.SimpleNamespace(send_message=send_message, send_photo=send_photo)), sent


def test_long_html_is_sent_as_plain_text():
    app, sent = fake_app()
    text = '<b>Title</b>\n<pre>' + 'x' * 5000 + '</pre>'
    asyncio.run(tb._send_text(app, 1, text, 'HTML'))
    assert len(sent) == 2 and all(mode is None for _, _, mode in sent)
    assert '<b>' not in sent[0][1] and sent[0][1].startswith('Title')


def test_short_html_stays_html_in_one_message():
    app, sent = fake_app()
    asyncio.run(tb._send_text(app, 1, '<b>Title</b> ' + 'y' * 4000, 'HTML'))   # raw > 4000, visible < 4096
    assert sent == [('text', '<b>Title</b> ' + 'y' * 4000, 'HTML')]


def run_worker(app, item):
    queue = types.SimpleNamespace(items=[item, None])
    queue.get_nowait = lambda: queue.items.pop(0)
    app.bot_data = {'notify_q': queue, 'chat_id': 1}
    app.running = True
    app.stop_running = lambda: None
    asyncio.run(tb._notify_worker(app))


def test_chart_is_deleted_even_when_sending_fails(sandbox):
    app, sent = fake_app(fail_photo=True)
    os.makedirs('plots', exist_ok=True)
    open('plots/chart_x.png', 'wb').write(b'png')
    run_worker(app, {'type': 'image', 'symbol': 'chart_x', 'message': 'hi', 'output_dir': 'plots',
                     'delete_after_send': True, 'html': False})
    assert not os.path.exists('plots/chart_x.png')


def test_caption_counts_visible_text_only(sandbox):
    app, sent = fake_app()
    os.makedirs('plots', exist_ok=True)
    open('plots/chart_y.png', 'wb').write(b'png')
    caption = '<b>' + 'z' * 1000 + '</b><i>!</i>'                 # 1001 visible characters, raw much longer
    run_worker(app, {'type': 'image', 'symbol': 'chart_y', 'message': caption, 'output_dir': 'plots',
                     'delete_after_send': True, 'html': True})
    assert sent == [('photo', caption, 'HTML')]                   # one message: chart with its text


# ── chart pattern: only clear breakouts (at least 0.5 ATR beyond the neckline) ──
def double_bottom(last):
    p = (list(np.linspace(130, 90, 41)) + list(np.linspace(91, 100, 10)) + list(np.linspace(99, 90, 10))
         + list(np.linspace(91, 99.6, 10)) + [last])
    return pd.DataFrame({'Date': pd.bdate_range('2025-01-01', periods=len(p)), 'Price': p, 'Volume': 1000.0})


def test_weak_breakout_is_not_reported():
    assert func.chart_pattern(double_bottom(100.2)) is None          # 0.2 above the neckline: weak


def test_clear_breakout_is_reported():
    pattern = func.chart_pattern(double_bottom(101.5))
    assert (pattern['name'], pattern['status'], pattern['at'][0]) == ('Double Bottom', 'breakout', 0)


# ── /ai prompt ─────────────────────────────────────────────────────────────
def test_leading_market_by_home_country(monkeypatch):
    class Ticker:
        def __init__(self, ticker):
            self.info = {'country': 'United States'} if ticker == 'NVD.DE' else {}

        def history(self, **kwargs):
            return pd.DataFrame({'Close': [1.0]})
    monkeypatch.setattr(func.yf, 'Ticker', Ticker)
    assert func.market_prices('stock', 'NVD.DE')[0] == 'S&P 500'      # US company on Xetra
    assert func.market_prices('stock', 'RHM.DE')[0] == 'DAX'          # no country: exchange decides


def test_old_default_prompt_is_replaced_own_version_kept(sandbox):
    import main
    # an unchanged earlier default (identified by its fingerprint) is replaced
    main.OLD_AI_PROMPT_HASHES.add(__import__('hashlib').sha256(b'old default').hexdigest())
    with open('ai_prompt.txt', 'w', encoding='utf-8') as file:
        file.write('old default\n')
    main.update_ai_prompt_file()
    assert open('ai_prompt.txt', encoding='utf-8').read() == main.AI_PROMPT_TEMPLATE
    # your own version stays
    with open('ai_prompt.txt', 'w', encoding='utf-8') as file:
        file.write('my own prompt {name}\n')
    main.update_ai_prompt_file()
    assert open('ai_prompt.txt', encoding='utf-8').read() == 'my own prompt {name}\n'


def test_prompt_template_has_evidence_and_signals():
    import main
    assert '{evidence}' in main.AI_PROMPT_TEMPLATE and '{signals}' in main.AI_PROMPT_TEMPLATE
    assert 'confirmed bottom' not in main.AI_PROMPT_TEMPLATE.split('{evidence}')[0]   # no fixed rule against the backtest
    assert set(main.AI_EVIDENCE) == set(main.AI_SCORE_NOTE) == {'stock', 'crypto'}
