"""Shared test setup for CaSa.

Every test runs in its own empty temporary folder with sample files: your real assets.ini,
transactions.csv, events.ini, valuealert.ini and logfile.txt are never read or changed.
No internet access is needed (no prices are downloaded).

Run all tests (in the CaSa folder):  python -m pytest
"""
import os
import sys
import types
import warnings

import matplotlib
matplotlib.use('Agg')   # images without a window
import numpy as np
import pandas as pd
import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
warnings.filterwarnings('ignore')

import func      # noqa: E402  (after the path is set)
import classes   # noqa: E402

ASSETS_INI = """[crypto]
bitcoin        = Bitcoin; crypto; 1
ethereum       = Ethereum; crypto

[stocks]
RHM.DE         = Rheinmetall; stock; 1
2IS.F          = Trane Technologies; stock
NVD.DE         = Nvidia; stock; 1
APC.DE         = Apple; stock
"""

TRANSACTIONS_CSV = """type;symbol;name;action;amount;qty;date
crypto;bitcoin;Bitcoin;buy;500.00;0.01;15/01/2025
stock;RHM.DE;Rheinmetall;buy;1200.00;2;27/09/2024
stock;RHM.DE;Rheinmetall;buy;1500.00;1;05/03/2025
stock;RHM.DE;Rheinmetall;sell;1700.00;1;10/06/2025
"""


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Own empty folder per test with sample assets.ini and transactions.csv."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'assets.ini').write_text(ASSETS_INI, encoding='utf-8')
    (tmp_path / 'transactions.csv').write_text(TRANSACTIONS_CSV, encoding='utf-8')
    # Alarm settings as in config.ini (recommended values)
    monkeypatch.setattr(func, 'seven_day_price_change', 10.0)
    monkeypatch.setattr(func, 'previous_alarm_change', 1.0)
    monkeypatch.setattr(func, 'alert_settings', {})   # all alarms and infos with their default
    # Telegram messages are collected instead of sent
    sent = []
    monkeypatch.setattr(func, 'pushover', lambda message, *a, **k: sent.append(message))
    monkeypatch.setattr(func, 'pushover_image', lambda name, message, *a, **k: sent.append((name, message)))
    return types.SimpleNamespace(path=tmp_path, sent=sent)


def make_prices(days=300, start=100.0, drift=0.001, volatility=0.02, seed=1, crypto=False):
    """Synthetic daily prices in the format of get_stock()/get_crypto(): Date, Price, Volume (+ high/low for stocks)."""
    rng = np.random.default_rng(seed)
    price = start * np.exp(np.cumsum(drift + volatility * rng.standard_normal(days)))
    df = pd.DataFrame({'Date': pd.bdate_range('2024-01-01', periods=days),
                       'Price': price,
                       'Volume': rng.integers(1_000, 10_000, days).astype(float)})
    if not crypto:
        df['PriceHigh'] = price * 1.01
        df['PriceLow'] = price * 0.99
    return df


@pytest.fixture
def prices():
    return make_prices()
