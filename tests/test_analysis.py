"""Analysis and alarms: indicators, score, traffic lights, alarm rules (Positive-5, SMA200 filter, sell only when owned)."""
import math

import numpy as np
import pandas as pd
import pytest

import classes as cl
import func
from conftest import make_prices


def analyse(data, watch_list=False, asset_type='stock'):
    """Like CryptoStock.process_data(): indicators on the long history, then 100 values, then alarm()."""
    full = func.calc_indicator_fuctions(data.copy())
    pattern = func.chart_pattern(full)
    df = func.cut_to_window(full)
    alarms, quantiles, score = func.alarm(df, 'Test', watch_list, 0, 0, 0, '100d', pattern, asset_type=asset_type)
    return full, df, alarms, score


def test_indicators_are_calculated(prices):
    full = func.calc_indicator_fuctions(prices.copy())
    for column in ('Fast EMA', 'Slow EMA', 'RSI14', 'ROC5', 'SMA7', 'VMA7', 'SMA200'):
        assert column in full, column
    assert math.isfinite(full['SMA200'].iloc[-1])
    assert 0 <= full['RSI14'].iloc[-1] <= 100


def test_cut_to_window_keeps_100_values(prices):
    df = func.cut_to_window(func.calc_indicator_fuctions(prices.copy()))
    assert len(df) == 100
    assert df['Date'].iloc[-1] == prices['Date'].iloc[-1]


def test_analysis_details_score_and_message(prices):
    df = func.cut_to_window(func.calc_indicator_fuctions(prices.copy()))
    details = func.analysis_details(df, rank=7)
    assert -18 <= details['score'] <= 18
    message = details['message']
    assert message.startswith('<b>Score 7/10</b>')
    for block in ('Long-term-Analysis', 'RSI-Analysis', 'Trend-Analysis', 'Short-term-Analysis'):
        assert block in message
    assert any(light in message for light in ('🟢', '🔴', '🟡'))   # traffic lights


def test_analysis_details_without_enough_data():
    with pytest.raises(ValueError):
        func.analysis_details(func.cut_to_window(func.calc_indicator_fuctions(make_prices(days=5))))


def test_score_rank_is_between_1_and_10(prices):
    rank = func.score_rank(func.calc_indicator_fuctions(prices.copy()))
    assert rank in range(1, 11)


@pytest.mark.parametrize('value, quantile', [(10, 10.0), (55, 55.0), (95, 95.0)])
def test_calc_stat_limits_quantile(value, quantile):
    df = pd.DataFrame({'X': list(range(1, 100)) + [value]})
    count, pct = func.calc_stat_limits(df, 'X')
    assert pct == pytest.approx(quantile, abs=1.5)
    assert count in (-3, -2, -1, 0, 1, 2, 3)


def test_positive5_alarm_on_first_strong_rise():
    data = make_prices(days=300, drift=0.002, volatility=0.003, seed=3)
    data.loc[data.index[-1], 'Price'] *= 1.13            # +13 % on the last day
    data.loc[data.index[-1], ['PriceHigh', 'PriceLow']] = data['Price'].iloc[-1] * np.array([1.01, 0.99])
    full, _, alarms, _ = analyse(data)
    assert full['ROC5'].iloc[-1] > func.seven_day_price_change >= full['ROC5'].iloc[-2]
    assert '121' in alarms
    assert 'Positive-5' in alarms['121']['msg']
    assert 'Potential buy' in alarms['121']['msg']


def test_positive5_blocked_below_sma200():
    data = make_prices(days=300, drift=-0.003, volatility=0.003, seed=3)   # long downtrend
    data.loc[data.index[-1], 'Price'] *= 1.20
    full, _, alarms, _ = analyse(data)
    # precondition: the rise alone would trigger Positive-5, only the SMA200 filter may block it
    assert full['ROC5'].iloc[-1] > func.seven_day_price_change >= full['ROC5'].iloc[-2]
    assert full['Price'].iloc[-1] < full['SMA200'].iloc[-1]
    assert '121' not in alarms and '141' not in alarms


def test_sell_alarms_only_for_owned_assets():
    data = make_prices(days=400, volatility=0.02, seed=7)
    found = False
    for end in range(260, 400):               # find a day with a Cross-EMA sell
        _, _, owned, _ = analyse(data.iloc[:end].reset_index(drop=True), watch_list=False)
        if '201' in owned:
            _, _, watched, _ = analyse(data.iloc[:end].reset_index(drop=True), watch_list=True)
            assert not any(code.startswith('2') for code in watched)
            found = True
            break
    assert found, 'no Cross-EMA sell in the sample data'


def test_oversold_alarm_only_for_stocks_on_the_first_day():
    data = make_prices(days=400, volatility=0.025, seed=11)
    found = False
    for end in range(260, 400):               # find a day where the score falls to -8 or lower
        part = data.iloc[:end].reset_index(drop=True)
        _, df, alarms, score = analyse(part, asset_type='stock')
        if '151' in alarms:
            assert score <= -8
            assert func.analysis_details(df.iloc[:-1])['score'] > -8      # first day only
            assert 'Oversold' in alarms['151']['msg'] and alarms['151']['priority'] == 0
            _, _, crypto_alarms, _ = analyse(part, asset_type='crypto')
            assert '151' not in crypto_alarms                              # never for crypto
            _, _, watched, _ = analyse(part, watch_list=True)
            assert '151' in watched                                        # also for watched stocks
            found = True
            break
    assert found, 'no oversold day in the sample data'


def test_filter_reports_same_alarm_only_once():
    state = cl.GroupedState()
    alarms = {'121': {'value': 1700000000, 'priority': 0, 'msg': 'x'}}
    assert func.filter(alarms, state) == alarms
    assert func.filter(alarms, state) == {}                               # same day: not again
    later = {'121': {'value': 1700086400, 'priority': 0, 'msg': 'x'}}
    assert func.filter(later, state) == later                             # new day: again


def test_calc_state_balances_and_profit(prices):
    asset = cl.CryptoStock('RHM.DE', 'Rheinmetall', 'stock')
    asset.buy(1000, 10, '01/01/2024')
    df = func.cut_to_window(func.calc_indicator_fuctions(prices.copy()))
    state = func.calc_state_fuctions(asset.state, df)
    price, before = df['Price'].iloc[-1], df['Price'].iloc[-2]
    assert state.current_balance_eur == pytest.approx(10 * price)
    assert state.yesterday_balance_eur == pytest.approx(10 * before)
    assert state.current_profit == pytest.approx(10 * price - 1000)
    assert state.current_one_day_price_change_pct == pytest.approx((price / before - 1) * 100)


def test_get_summary_contains_daily_pnl(prices):
    asset = cl.CryptoStock('RHM.DE', 'Rheinmetall', 'stock')
    asset.buy(1000, 10, '01/01/2024')
    df, _ = asset.process_data(prices.copy(), '100d')
    row = asset.get_summary(df)
    assert row['Value [€]'] == pytest.approx(10 * df['Price'].iloc[-1], abs=0.01)
    assert row['1d P&L [€]'] == pytest.approx(asset.state.current_balance_eur - asset.state.yesterday_balance_eur, abs=0.01)
    assert row['Rating'] in range(0, 11)


def test_risk_reward_sentences():
    zones = ({'center': 110.0}, {'center': 95.0})
    assert func.risk_reward(100, zones)['short'] == 'Twice as much room up as down.'
    assert func.risk_reward(100, ({'center': 103.0}, {'center': 95.0}))['short'] == 'Less room up than down.'
    assert func.risk_reward(100, (None, {'center': 95.0}))['short'] == 'No resistance above.'
    assert func.risk_reward(100, None)['short'] == 'No zones found.'


def test_chart_pattern_and_support_resistance_do_not_crash(prices):
    full = func.calc_indicator_fuctions(prices.copy())
    pattern = func.chart_pattern(full)
    assert pattern is None or {'name', 'top', 'target', 'status'} <= set(pattern)
    resistance, support = func.support_resistance(full)
    price = full['Price'].iloc[-1]
    assert resistance is None or resistance['center'] >= price * 0.98
    assert support is None or support['center'] <= price * 1.02
