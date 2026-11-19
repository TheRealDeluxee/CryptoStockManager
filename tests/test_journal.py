"""Alarm journal: record sent alarms, fill in the result 20 trading days later, statistics and the report."""
import types

import pandas as pd
import pytest

import classes as cl


def prices(start, values):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_add_and_read():
    cl.journal_add('stock', 'RHM.DE', 'Rheinmetall', '151', '100d', '2026-10-01', 1500.0, -9, False)
    rows = cl.read_journal()
    assert len(rows) == 1
    row = rows[0]
    assert (row['date'], row['symbol'], row['alarm'], row['direction'], row['return_20']) == \
           ('01/10/2026', 'rhm.de', 'Oversold', 'buy', '')


def test_result_after_20_trading_days():
    cl.journal_add('stock', 'RHM.DE', 'Rheinmetall', '151', '100d', '2026-10-01', 100.0, -9, False)
    cl.journal_add('crypto', 'bitcoin', 'Bitcoin', '201', '100d', '2026-10-01', 100.0, -2, False)
    # only 10 later prices: not yet evaluated
    assert cl.journal_update({('stock', 'rhm.de'): prices('2026-10-01', [100] * 11)}) == 0
    series = prices('2026-10-01', [100.0] + [101.0] * 19 + [110.0] + [120.0] * 5)   # 20th value after: 110
    done = cl.journal_update({('stock', 'rhm.de'): series, ('crypto', 'bitcoin'): series})
    assert done == 2
    rhm, btc = cl.read_journal()
    assert (rhm['price_20'], rhm['return_20'], rhm['hit']) == ('110', '10.00', '1')   # buy and price up: hit
    assert btc['hit'] == '0'                                                        # sell and price up: miss
    assert cl.journal_update({('stock', 'rhm.de'): series}) == 0                    # never twice


def test_stats_and_report():
    for price_20, code in ((110, '151'), (95, '151'), (90, '201')):
        cl.journal_add('stock', 'RHM.DE', 'Rheinmetall', code, '100d', '2026-10-01', 100.0, 0, False)
    cl.journal_update({('stock', 'rhm.de'): prices('2026-10-01', [100.0] * 20 + [110.0])})
    stats, first = cl.journal_stats()
    assert stats['151']['count'] == 2 and stats['151']['evaluated'] == 2 and stats['151']['hit_pct'] == 100
    assert stats['201']['hit_pct'] == 0          # sell alarm, price rose
    assert stats['151']['backtest_hit_pct'] == 58
    report = cl.journal_report()
    assert report.startswith('<b>Alarm journal</b> since 01/10/2026')
    assert 'Oversold' in report and 'Cross-EMA Sell' in report and 'few results' in report


def test_empty_journal_report():
    assert 'No alarms recorded yet' in cl.journal_report()


def test_journal_report_text(sandbox):
    cl.journal_add('crypto', 'bitcoin', 'Bitcoin', '121', '100h', '2026-11-01 14:00', 70000, 9, True)
    text = cl.journal_report()
    assert 'Positive-5' in text and '1 alarms still waiting' in text


def test_main_records_alarms_but_not_in_test_mode(sandbox):
    import main
    item = types.SimpleNamespace(type='stock', symbol='RHM.DE', search='Rheinmetall',
                                 state=types.SimpleNamespace(score=-9, watch_list=False))
    df = pd.DataFrame({'Date': pd.to_datetime(['2026-11-18']), 'Price': [1700.0]})
    manager = types.SimpleNamespace(test_mode=1)
    main.CryptoStockManager._journal_alarms(manager, item, df, '100d', {'151': {}})
    assert cl.read_journal() == []
    manager.test_mode = 0
    main.CryptoStockManager._journal_alarms(manager, item, df, '100d', {'151': {}, '201': {}})
    assert [row['code'] for row in cl.read_journal()] == ['151', '201']
