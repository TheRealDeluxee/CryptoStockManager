"""Assets you hold according to transactions.csv are always analysed (daily P&L, loss limit),
even without the 1 in assets.ini; held assets missing in assets.ini are reported once in Telegram."""
import classes as cl

EXTRA = """stock;2IS.F;Trane;buy;4000.00;10;02/01/2025
stock;APC.DE;Apple;buy;900.00;5;02/01/2025
stock;APC.DE;Apple;sell;1000.00;5;02/02/2025
stock;TSFA.DU;TSMC;buy;2000.00;17;02/01/2025
stock;OLD.DE;Old;buy;100.00;1;02/01/2025
stock;OLD.DE;Old;sell;120.00;1;03/01/2025
"""


def manager():
    import main
    m = main.CryptoStockManager.__new__(main.CryptoStockManager)
    m.crypto_items_hh, m.crypto_items_hd, m.stock_items_hh, m.stock_items_hd = [], [], [], []
    m.missing_held_notified = set()
    return m


def add_rows(sandbox):
    path = sandbox.path / 'transactions.csv'
    path.write_text(path.read_text(encoding='utf-8') + EXTRA, encoding='utf-8')


def test_held_assets(sandbox):
    add_rows(sandbox)
    held = cl.held_assets(cl.read_transactions())
    assert ('stock', '2is.f') in held and ('stock', 'rhm.de') in held and ('crypto', 'bitcoin') in held
    assert ('stock', 'tsfa.du') in held
    assert ('stock', 'apc.de') not in held and ('stock', 'old.de') not in held   # completely sold


def test_held_without_1_is_analysed(sandbox):
    add_rows(sandbox)
    m = manager()
    m.load_config_items()
    stocks = {item.symbol for item in m.stock_items_hd}
    assert stocks == {'RHM.DE', 'NVD.DE', '2IS.F'}           # 2IS.F held without 1, APC.DE sold
    assert {item.symbol for item in m.stock_items_hh} == stocks
    trane = next(item for item in m.stock_items_hd if item.symbol == '2IS.F')
    assert trane.state.amount_crypto_stock == 10
    log = (sandbox.path / 'logfile.txt').read_text(encoding='utf-8')
    assert 'analysed anyway: 2IS.F' in log
    assert 'Inactive transactions (not in assets.ini, not held): OLD.DE' in log


def test_held_but_missing_reported_once(sandbox):
    add_rows(sandbox)
    m = manager()
    m.load_config_items()
    notices = [text for text in sandbox.sent if 'Positions without prices' in str(text)]
    assert len(notices) == 1 and 'TSFA.DU' in notices[0] and 'OLD.DE' not in notices[0]
    m.load_config_items()                                      # reload (/reload_assets, /buy): no second notice
    assert len([text for text in sandbox.sent if 'Positions without prices' in str(text)]) == 1


def test_nothing_missing_no_notice(sandbox):
    m = manager()
    m.load_config_items()
    assert not any('Positions without prices' in str(text) for text in sandbox.sent)
