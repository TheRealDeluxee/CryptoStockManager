@echo off
echo Checking Python installation...

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python found. Checking for updates...
    winget upgrade Python.Python.3 --silent
) else (
    echo Installing Python...
    winget install Python.Python.3 --silent
    echo Restart terminal and run script again.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing/Updating packages...
pip install --upgrade pandas numpy scipy requests matplotlib plotly pytrends yfinance schedule mplfinance ruptures

echo Done!
pause