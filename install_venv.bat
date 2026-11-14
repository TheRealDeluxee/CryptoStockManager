@echo off
setlocal

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

echo.
echo Creating virtual environment...
if not exist .venv (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Upgrading pip inside venv...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing/Updating packages inside venv...
python -m pip install --upgrade ^
    pandas ^
    numpy ^
    scipy ^
    requests ^
    matplotlib ^
    plotly ^
    pytrends ^
    yfinance ^
    schedule ^
    mplfinance ^
    ruptures ^
    psutil ^
    python-telegram-bot
if %errorlevel% neq 0 (
    echo Package installation failed.
    pause
    exit /b 1
)

echo.
echo Done!
echo Virtual environment: .venv
pause
