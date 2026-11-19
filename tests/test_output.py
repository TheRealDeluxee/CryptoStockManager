"""Messages, images and log: daily summary, daily P&L, transactions table, logfile.txt."""
import os
import types

import pandas as pd
import pytest

import func


def is_png(path):
    with open(path, 'rb') as file:
        return file.read(8) == b'\x89PNG\r\n\x1a\n'


SUMMARY = pd.DataFrame([
    {'Name': 'Rheinmetall', 'Piece [€]': 1700.0, '7d P&L [%]': 4.2, '1d P&L [%]': 1.5, 'Rating': 8,
     'Type': 'stock', 'Value [€]': 3400.0, '1d P&L [€]': 50.2, 'Total P&L [€]': 700.0, 'Total P&L [%]': 25.9},
    {'Name': 'Trane Technologies', 'Piece [€]': 380.0, '7d P&L [%]': -1.0, '1d P&L [%]': -0.4, 'Rating': 3,
     'Type': 'stock', 'Value [€]': 0.0, '1d P&L [€]': 0.0, 'Total P&L [€]': 0.0, 'Total P&L [%]': 0.0},
    {'Name': 'Bitcoin', 'Piece [€]': 80000.0, '7d P&L [%]': 2.0, '1d P&L [%]': -2.1, 'Rating': 0,
     'Type': 'crypto', 'Value [€]': 800.0, '1d P&L [€]': -17.2, 'Total P&L [€]': 300.0, 'Total P&L [%]': 60.0},
])


def test_format_euro():
    assert func.format_euro(71660.123) == '71,660.12 €'


def test_telegram_length_counts_visible_text_only():
    assert func.telegram_length('<b>Hi</b> &amp; <pre>x</pre>') == len('Hi & x')
    assert func.telegram_length('😀') == 2   # emojis count double at Telegram


def test_daily_summary_image(sandbox):
    func.create_table_image(SUMMARY, 'summary.png', title='Stocks · Daily Summary')
    assert is_png('summary.png')


def test_pnl_image(sandbox):
    func.create_pnl_image(SUMMARY[SUMMARY['Value [€]'] > 0], 'pnl.png', title='My Daily P&L')
    assert is_png('pnl.png')


def test_transactions_image(sandbox):
    rows = [{'date': pd.Timestamp('2025-01-05'), 'name': 'A very long company name Inc.', 'type': 'stock',
             'action': 'buy', 'amount': 450.0, 'qty': 3.0},
            {'date': pd.Timestamp('2026-11-02'), 'name': 'Bitcoin', 'type': 'crypto',
             'action': 'sell', 'amount': 320.5, 'qty': 0.004}]
    func.create_transactions_image(rows, 'tx.png', title='Transactions')
    assert is_png('tx.png')


def test_send_pnl_only_owned_positions(sandbox, monkeypatch):
    import main
    images = []
    monkeypatch.setattr(func, 'create_pnl_image', lambda df, path, **k: images.append(df))
    fake = types.SimpleNamespace(df_crypto_hd=SUMMARY[SUMMARY['Type'] == 'crypto'],
                                 df_stock_hd=SUMMARY[SUMMARY['Type'] == 'stock'],
                                 output_dir=str(sandbox.path), last_daily_update='19/11/2026 18:30:00')
    main.CryptoStockManager.send_pnl(fake)
    assert list(images[0]['Name']) == ['Bitcoin', 'Rheinmetall']        # Trane (not owned) is left out
    assert sandbox.sent[-1][1].startswith('Data as of: 19/11/2026')


def test_send_pnl_notes_missing_stocks(sandbox, monkeypatch):
    import main
    monkeypatch.setattr(func, 'create_pnl_image', lambda *a, **k: None)
    fake = types.SimpleNamespace(df_crypto_hd=SUMMARY[SUMMARY['Type'] == 'crypto'], df_stock_hd=SUMMARY.iloc[0:0],
                                 output_dir=str(sandbox.path), last_daily_update='22/11/2026 18:30:00')
    main.CryptoStockManager.send_pnl(fake)
    assert 'Stocks not included' in sandbox.sent[-1][1]


def test_log_print_writes_one_line_per_event(sandbox):
    func.log_print('<b>Hello</b> world\nsecond line', 'WARN')
    lines = (sandbox.path / 'logfile.txt').read_text(encoding='utf-8').splitlines()
    assert 'WARN' in lines[0] and lines[0].endswith('Hello world')     # HTML tags removed
    assert lines[1].strip() == 'second line'


def test_log_rollover_above_limit(sandbox, monkeypatch):
    monkeypatch.setattr(func, 'LOG_MAX_BYTES', 100)
    (sandbox.path / 'logfile.txt').write_text('x' * 200, encoding='utf-8')
    func.log_print('new start')
    assert os.path.exists('logfile.old.txt')
    assert 'new start' in (sandbox.path / 'logfile.txt').read_text(encoding='utf-8')
