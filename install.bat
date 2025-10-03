@echo off
echo Installing required Python packages for CryptoStockManager...

REM Upgrade pip to latest version first
python -m pip install --upgrade pip

REM Install necessary packages
pip install pandas numpy scipy requests matplotlib plotly pytrends yfinance schedule mplfinance ruptures

echo Installation completed.
pause