"""P&L chart: profit/loss per day from transactions and daily prices (realized + unrealized, not the value)."""
import types

import pandas as pd
import pytest

import classes as cl
import func

TODAY = '2026-11-19'


def rows(*trades):
    return [dict(zip(['type', 'symbol', 'name', 'action', 'amount', 'qty', 'date'], t)) for t in trades]


def flat_prices(price, days=120):
    return pd.Series(price, index=pd.date_range(end=TODAY, periods=days, freq='D'))


def test_pnl_counts_buys_sells_and_price():
    trades = rows(('stock', 'rhm.de', 'Rheinmetall', 'buy', '1200', '2', '01/10/2026'),
                  ('stock', 'rhm.de', 'Rheinmetall', 'sell', '700', '1', '01/11/2026'))
    pnl, missing = cl.pnl_history(trades, {('stock', 'rhm.de'): flat_prices(650.0)}, days=100, today=TODAY)
    assert len(pnl) == 100 and missing == []
    assert pnl[pd.Timestamp('2026-09-30')] == pytest.approx(0)       # before the buy: nothing
    assert pnl[pd.Timestamp('2026-10-15')] == pytest.approx(100)     # 2 x 650 - 1200
    assert pnl[pd.Timestamp('2026-11-19')] == pytest.approx(150)     # 1 x 650 + 700 - 1200 (sale is realized)


def test_pnl_buy_before_the_window_and_weekend_prices():
    trades = rows(('crypto', 'bitcoin', 'Bitcoin', 'buy', '500', '0.01', '15/01/2025'))
    prices = pd.Series([60000.0, 70000.0], index=pd.to_datetime(['2026-08-01', '2026-11-10']))
    pnl, _ = cl.pnl_history(trades, {('crypto', 'bitcoin'): prices}, days=100, today=TODAY)
    assert pnl.iloc[0] == pytest.approx(0.01 * 60000 - 500)           # bought long before the window
    assert pnl.iloc[-1] == pytest.approx(0.01 * 70000 - 500)          # last price carried forward


def test_pnl_reports_held_assets_without_prices():
    trades = rows(('stock', 'xyz', 'Unknown', 'buy', '100', '1', '01/01/2026'),
                  ('stock', 'rhm.de', 'Rheinmetall', 'buy', '600', '1', '01/01/2026'))
    pnl, missing = cl.pnl_history(trades, {('stock', 'rhm.de'): flat_prices(650.0)}, today=TODAY)
    assert missing == ['xyz']
    assert pnl.iloc[-1] == pytest.approx(50)


def test_daily_prices_removes_time_zone_and_duplicates():
    df = pd.DataFrame({'Date': pd.to_datetime(['2026-11-18 00:00', '2026-11-19 00:00', '2026-11-19 14:30']),
                       'Price': [1.0, 2.0, 3.0]})
    series = cl.daily_prices(df)
    assert list(series) == [1.0, 3.0]
    stock = pd.DataFrame({'Date': pd.date_range('2026-11-17', periods=2, tz='Europe/Berlin'), 'Price': [5.0, 6.0]})
    assert cl.daily_prices(stock).index.tz is None


def test_pnl_chart_image(sandbox):
    pnl = pd.Series([156.0, 120.0, 30.0, -11.22], index=pd.date_range(end=TODAY, periods=4))
    func.create_pnl_chart(pnl, 'chart.png', title='P&L · last 100 days')
    with open('chart.png', 'rb') as file:
        assert file.read(8) == b'\x89PNG\r\n\x1a\n'


def test_send_pnl_chart_13_full_weeks(sandbox, monkeypatch):
    import main
    charts = []
    real_chart = func.create_pnl_chart
    monkeypatch.setattr(func, 'create_pnl_chart', lambda pnl, path, **k: (charts.append(pnl), real_chart(pnl, path, **k)))
    today = pd.Timestamp.today().normalize()
    prices = pd.Series(650.0, index=pd.date_range(end=today, periods=120, freq='D'))
    fake = types.SimpleNamespace(daily_prices={('stock', 'rhm.de'): prices, ('crypto', 'bitcoin'): prices * 100},
                                 output_dir=str(sandbox.path))
    main.CryptoStockManager.send_pnl_chart(fake)
    pnl = charts[0]
    assert pnl.index[0].dayofweek == 0                      # starts on a Monday
    assert pnl.index[-1] == today                           # ends today
    assert len(pnl.resample('W-SUN').last()) == 13          # 13 weekly bars
    name, message = sandbox.sent[-1]
    assert name.startswith('pnl_chart_')
    assert 'not the portfolio value' in message
