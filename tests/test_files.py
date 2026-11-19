"""Files and inputs: assets.ini, transactions.csv (FIFO), events.ini, numbers and prices."""
import datetime

import pytest

import classes as cl


# ── assets.ini ─────────────────────────────────────────────────────────────
def test_read_assets_all_and_alarm_only():
    assets = cl.read_assets()
    assert assets[('stock', 'rhm.de')] == 'Rheinmetall'
    assert assets[('crypto', 'bitcoin')] == 'Bitcoin'
    assert len(assets) == 6
    assert set(cl.read_assets(alarm_only=True)) == {('crypto', 'bitcoin'), ('stock', 'rhm.de'), ('stock', 'nvd.de')}


def test_read_assets_rejects_invalid_alarm_flag(sandbox):
    (sandbox.path / 'assets.ini').write_text('[stocks]\nRHM.DE = Rheinmetall; stock; 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='alarm flag'):
        cl.read_assets()


def test_read_assets_missing_file(sandbox):
    (sandbox.path / 'assets.ini').unlink()
    with pytest.raises(FileNotFoundError):
        cl.read_assets()


def test_set_alarm_keeps_other_lines(sandbox):
    cl.set_alarm('2IS.F', 'stock', True)
    assert ('stock', '2is.f') in cl.read_assets(alarm_only=True)
    cl.set_alarm('rhm.de', 'stock', False)
    assert ('stock', 'rhm.de') not in cl.read_assets(alarm_only=True)
    assert len(cl.read_assets()) == 6
    with pytest.raises(ValueError):
        cl.set_alarm('XYZ', 'stock', True)


# ── symbol or name (bot commands) ─────────────────────────────────────────
@pytest.mark.parametrize('words, symbol', [
    (['RHM.DE'], 'rhm.de'),
    (['rheinmetall'], 'rhm.de'),
    (['Trane', 'Technologies'], '2is.f'),
    (['trane'], '2is.f'),            # beginning of the name (at least 3 letters)
    (['BITCOIN'], 'bitcoin'),
])
def test_take_asset_by_symbol_or_name(words, symbol):
    assert cl.take_asset(words, cl.read_assets())[1] == symbol


def test_take_asset_returns_the_rest():
    asset_type, symbol, name, rest = cl.take_asset(['Nvidia', '17/11/2026', 'Q3'], cl.read_assets())
    assert (asset_type, symbol, name, rest) == ('stock', 'nvd.de', 'Nvidia', ['17/11/2026', 'Q3'])


def test_take_asset_errors():
    assets = cl.read_assets()
    with pytest.raises(ValueError, match='not in assets.ini'):
        cl.take_asset(['xyz'], assets)
    with pytest.raises(ValueError):
        cl.take_asset(['ap'], assets)                         # too short for a name beginning
    with pytest.raises(ValueError):
        cl.take_asset(['Bitcoin'], assets, ('stock',))        # wrong type


# ── numbers and prices ─────────────────────────────────────────────────────
def test_valid_numbers():
    assert cl.valid_numbers('1200,50', '0.5') == (1200.5, 0.5)
    for amount, qty in (('0', '1'), ('-5', '1'), ('nan', '1'), ('10', 'inf')):
        with pytest.raises(ValueError):
            cl.valid_numbers(amount, qty)


@pytest.mark.parametrize('text, value', [('320.82', 320.82), ('320,82', 320.82), ('€320.82', 320.82),
                                         ('1.234,50', 1234.5), ('1,234.50', 1234.5)])
def test_parse_price(text, value):
    assert cl.parse_price(text) == pytest.approx(value)


@pytest.mark.parametrize('text', ['abc', '0', '-3'])
def test_parse_price_invalid(text):
    with pytest.raises(ValueError):
        cl.parse_price(text)


# ── transactions.csv and FIFO ──────────────────────────────────────────────
def test_read_transactions():
    rows = cl.read_transactions()
    assert len(rows) == 4
    assert rows[1]['symbol'] == 'rhm.de'   # symbols are lower case after reading


def test_read_transactions_rejects_future_date(sandbox):
    future = (datetime.date.today() + datetime.timedelta(days=5)).strftime('%d/%m/%Y')
    with open('transactions.csv', 'a', encoding='utf-8') as file:
        file.write(f'stock;RHM.DE;Rheinmetall;buy;100;1;{future}\n')
    with pytest.raises(ValueError, match='future'):
        cl.read_transactions()


def test_read_transactions_wrong_header(sandbox):
    (sandbox.path / 'transactions.csv').write_text('a;b;c\n', encoding='utf-8')
    with pytest.raises(ValueError, match='header'):
        cl.read_transactions()


def test_fifo_sell_reduces_oldest_buy_first():
    asset = cl.apply_transactions(cl.CryptoStock('RHM.DE', 'Rheinmetall', 'stock'), cl.read_transactions())
    # bought 2 for 1200 (600 each) and 1 for 1500; sold 1 -> one of the 600 € shares leaves first
    assert asset.state.amount_crypto_stock == pytest.approx(2)
    assert asset.state.buy_balance_eur == pytest.approx(600 + 1500)
    assert asset.state.sell_money == pytest.approx(1700)
    assert not asset.state.watch_list


def test_sell_more_than_holdings_is_rejected():
    asset = cl.CryptoStock('bitcoin', 'Bitcoin', 'crypto')
    asset.buy(500, 0.01, '15/01/2025')
    with pytest.raises(ValueError):
        asset.sell(1000, 0.02)
    assert asset.state.amount_crypto_stock == pytest.approx(0.01)   # nothing changed


# ── events.ini ─────────────────────────────────────────────────────────────
def test_events_add_read_delete():
    date = (datetime.date.today() + datetime.timedelta(days=10)).strftime('%d/%m/%Y')
    key, name, _ = cl.add_event('NVD.DE', date, 'Q3 earnings')
    events = cl.read_events()
    assert [(e['key'], e['name'], e['text']) for e in events] == [(key, 'Nvidia', 'Q3 earnings')]
    cl.delete_event(key)
    assert cl.read_events() == []
    with pytest.raises(ValueError):
        cl.delete_event(key)


def test_event_in_the_past_or_unknown_asset():
    with pytest.raises(ValueError, match='past'):
        cl.add_event('NVD.DE', '01/01/2020', 'old')
    with pytest.raises(ValueError, match='not in assets.ini'):
        cl.add_event('XYZ', '01/01/2099', 'unknown')


