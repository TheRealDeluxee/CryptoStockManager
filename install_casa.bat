@echo off
rem CaSa setup for Windows - double-click this file.
rem It installs Python 3 (if needed), downloads the newest CaSa release from GitHub, installs the Python packages,
rem asks for your Telegram bot and API keys, writes config.ini and assets.ini. CaSa itself: start_casa.bat
rem Started again: change the settings or update CaSa; your own files stay.
rem The wizard itself is the Python part at the end of this file.
setlocal EnableExtensions
chcp 65001 >nul
title CaSa setup
cd /d "%~dp0"

call :find_python
if defined PY goto wizard
echo Python 3 is not installed yet. Installing Python 3.13 (about 2 minutes) ...
where winget >nul 2>&1
if errorlevel 1 goto no_python
winget install -e --id Python.Python.3.13 --scope user --silent --accept-package-agreements --accept-source-agreements
call :find_python
if not defined PY goto no_python

:wizard
set "WIZARD=%TEMP%\casa_setup_wizard.py"
"%PY%" -c "import sys; s = open(sys.argv[1], encoding='utf-8').read(); open(sys.argv[2], 'w', encoding='utf-8').write(s.split('\n:: ==PYTHON==\n', 1)[1])" "%~f0" "%WIZARD%"
if errorlevel 1 goto failed
"%PY%" "%WIZARD%" "%~dp0." "%~f0"
set "RC=%errorlevel%"
del "%WIZARD%" >nul 2>&1
echo.
pause
exit /b %RC%

:no_python
echo.
echo Python could not be installed automatically.
echo Please install Python 3 from https://www.python.org/downloads/
echo - tick "Add python.exe to PATH" - and start install_casa.bat again.
echo.
pause
exit /b 1

:failed
echo.
echo The setup could not be started.
pause
exit /b 1

:find_python
rem Python 3.10 or newer: on the PATH, via the launcher py, or where winget installs it
set "PY="
for /f "delims=" %%I in ('python -c "import sys; print(sys.executable * (sys.version_info >= (3, 10)))" 2^>nul') do set "PY=%%I"
if not defined PY for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable * (sys.version_info >= (3, 10)))" 2^>nul') do set "PY=%%I"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
exit /b 0

:: ==PYTHON==
"""CaSa setup wizard for Windows, started by install_casa.bat (which installs Python before, if needed).

Like the wizard of the CaSa image: asks for the folder, the Telegram bot and the API keys, then downloads the newest
CaSa release from GitHub, installs the Python packages, writes config.ini, assets.ini and start_casa.bat.
CaSa is not started: only by hand with start_casa.bat.
Started again, it keeps your keys (if you want) and updates the program files; your own files always stay.
Keys are never shown, logged or passed as command arguments.
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

REPO = 'TheRealDeluxee/CryptoStockManager'
# your own files: never overwritten by a download
PRIVATE = {'config.ini', 'assets.ini', 'events.ini', 'planalert.ini', 'valuealert.ini', 'transactions.csv',
           'alarm_journal.csv', 'logfile.txt'}
MODELS = (('claude-sonnet-5-5', 'recommended: good and affordable'),
          ('claude-opus-5-5', 'strongest, about twice the price'),
          ('claude-haiku-5-5', 'fastest and cheapest'))
LINE = '-' * 72

CONFIG_TEMPLATE = """[telegram]
# from @BotFather (/newbot); chat_id: your chat with the bot
token =
chat_id =

[alertsconfig]
seven_day_price_change = 10     # threshold for Positive-5 / Negative-5 (%), 10 recommended by the backtest
previous_alarm_change = 3       # minimum change before the same alarm is sent again
test_mode = 0                   # 0 = scheduled analyses, 1 = run all analyses once at start
total_loss_limit = -5           # loss limit alarm: total P&L of an own position in % (e.g. -5 = alarm when it falls below -5 %)

[coingecko]
# optional: free demo key from coingecko.com/en/developers/dashboard (empty = free public access)
api_key =

[anthropic]
# optional: key for the Claude API (https://console.anthropic.com), empty = Claude API not used
api_key =
model = claude-sonnet-5-5
"""

ASSETS_EXAMPLE = """# CaSa assets. Format: symbol = Name; crypto|stock; 1; currency
# 1 = watch with alarms, also when you don't hold it; 0 or empty = only for /check; currency only for your overview
# Crypto: CoinGecko IDs (e.g. bitcoin), stocks: Yahoo Finance symbols at the home exchange (e.g. RHM.DE, NVDA)

[crypto]
bitcoin        = Bitcoin; crypto; 1
ethereum       = Ethereum; crypto; 1

[stocks]
RHM.DE         = Rheinmetall; stock; 1; EUR
"""

START_BAT = """@echo off
rem Starts CaSa (written by install_casa.bat). Closing this window stops CaSa.
cd /d "%~dp0"
title CaSa
".venv\\Scripts\\python.exe" main.py
echo.
echo CaSa has stopped.
pause
"""


class Cancel(Exception):
    pass


class SetupError(Exception):
    pass


# ── questions ────────────────────────────────────────────────────────────────
def header(title):
    print()
    print(LINE)
    print('  ' + title)
    print(LINE)


def ask(prompt, default=''):
    try:
        answer = input(f'{prompt}{f" [{default}]" if default else ""}: ').strip()
    except EOFError:
        raise Cancel
    return answer or default


def ask_secret(prompt):
    """Normal input (paste with Ctrl+V or right-click); after Enter the key on the screen is replaced by *."""
    try:
        answer = input(prompt + ': ').strip()
    except EOFError:
        raise Cancel
    if answer and sys.stdout.isatty():
        # one line up, clear it, write it again with * instead of the key
        print(f'\x1b[1A\x1b[2K{prompt}: {"*" * min(len(answer), 20)}', flush=True)
    return answer


def yes(prompt, default=True):
    while True:
        answer = ask(prompt + (' (Y/n)' if default else ' (y/N)')).lower()
        if not answer:
            return default
        if answer in ('y', 'yes', 'j', 'ja'):
            return True
        if answer in ('n', 'no', 'nein'):
            return False


def wait_enter(text):
    try:
        input(text)
    except EOFError:
        raise Cancel


# ── internet ─────────────────────────────────────────────────────────────────
def fetch(url, timeout=30):
    request = urllib.request.Request(url, headers={'User-Agent': 'CaSa-setup'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def telegram(token, method):
    """Telegram API answer as dict ({} without internet or with a wrong token). The token is never printed."""
    try:
        return json.loads(fetch(f'https://api.telegram.org/bot{token}/{method}', timeout=20))
    except Exception:
        return {}


def last_chat_id(token):
    updates = telegram(token, 'getUpdates').get('result', [])
    chats = [u['message']['chat']['id'] for u in updates if 'message' in u]
    return str(chats[-1]) if chats else ''


def newest_release():
    tags = json.loads(fetch(f'https://api.github.com/repos/{REPO}/tags?per_page=100'))
    versions = [t['name'] for t in tags if re.fullmatch(r'v\d+(\.\d+)*', t['name'])]
    if not versions:
        raise SetupError('No CaSa release found on GitHub.')
    return max(versions, key=lambda v: tuple(int(x) for x in v[1:].split('.')))


def download(tag, folder):
    """Program files of the release into the folder; your own files stay."""
    data = fetch(f'https://github.com/{REPO}/archive/refs/tags/{tag}.zip', timeout=180)
    root = os.path.normpath(folder)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            parts = info.filename.split('/', 1)          # first part: CryptoStockManager-<version>/
            if len(parts) < 2 or not parts[1] or info.is_dir():
                continue
            target = os.path.normpath(os.path.join(root, parts[1]))
            if not target.startswith(root + os.sep):
                continue
            if parts[1] in PRIVATE and os.path.exists(target):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.open(info) as source, open(target, 'wb') as file:
                shutil.copyfileobj(source, file)


# ── config.ini ───────────────────────────────────────────────────────────────
def config_value(path, section, key):
    current = None
    try:
        with open(path, encoding='utf-8-sig') as file:
            for line in file:
                line = line.strip()
                if line.startswith('['):
                    current = line.strip('[]').lower()
                elif current == section and '=' in line and not line.startswith(('#', ';')):
                    name, value = line.split('=', 1)
                    if name.strip().lower() == key:
                        return value.split('#')[0].strip()
    except OSError:
        pass
    return ''


def set_value(lines, section, key, value):
    """key = value in [section]; comments and other lines stay. Missing: added."""
    start = next((i for i, line in enumerate(lines) if line.strip().lower() == f'[{section}]'), None)
    if start is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ['', f'[{section}]', f'{key} = {value}']
        return
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('[')), len(lines))
    for i in range(start + 1, end):
        if '=' in lines[i] and not lines[i].lstrip().startswith(('#', ';')) \
                and lines[i].split('=', 1)[0].strip().lower() == key:
            lines[i] = f'{key} = {value}'
            return
    insert = end
    while insert > start + 1 and not lines[insert - 1].strip():
        insert -= 1
    lines.insert(insert, f'{key} = {value}')


def write_config(path, values):
    if os.path.exists(path):
        shutil.copy2(path, path + '.bak')          # backup of the old config.ini
        with open(path, encoding='utf-8-sig') as file:
            lines = file.read().splitlines()
    else:
        lines = CONFIG_TEMPLATE.splitlines()
    for (section, key), value in values.items():
        set_value(lines, section, key, value)
    # write a complete new file first, then replace: never a half-written config.ini
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=os.path.dirname(path), suffix='.tmp',
                                     delete=False) as file:
        file.write('\n'.join(lines) + '\n')
    os.replace(file.name, path)


# ── the steps ────────────────────────────────────────────────────────────────
def choose_folder(here):
    """(folder, what to do with the program files: 'download' / 'keep')"""
    header('Step 1/3: Folder for CaSa')
    if os.path.exists(os.path.join(here, 'main.py')):
        folder = here
        print(f'CaSa is in this folder: {folder}')
    else:
        print('CaSa and your files (config.ini, assets.ini, transactions.csv ...) go into this folder.')
        folder = os.path.abspath(os.path.expandvars(ask('Folder (Enter = the folder of install_casa.bat)', here)))
    if not os.path.exists(os.path.join(folder, 'main.py')):
        return folder, 'download'
    if os.path.isdir(os.path.join(folder, '.git')):
        print('This is a Git copy of CaSa: the program files are kept as they are.')
        return folder, 'keep'
    if yes('Update the program files to the newest CaSa release? Your own files stay.'):
        return folder, 'download'
    return folder, 'keep'


def ask_telegram(config):
    """(token, chat_id, bot name) or None to keep the current ones."""
    header('Step 2/3: Your Telegram bot')
    if config_value(config, 'telegram', 'token') and config_value(config, 'telegram', 'chat_id'):
        if yes('Your Telegram bot is already set. Keep it?'):
            return None
    print('New bot: in Telegram send /newbot to @BotFather and copy the token.')
    print('Paste it here with a right-click or Ctrl+V, then press Enter.')
    while True:
        token = re.sub(r'\s', '', ask_secret('Bot token'))
        if re.fullmatch(r'\d+:[\w-]+', token):
            bot = telegram(token, 'getMe').get('result', {}).get('username')
            if bot:
                break
        print('Telegram does not accept this token (or there is no internet). Please check it and enter it again.')
    print(f'\nToken OK: your bot is @{bot}')
    wait_enter(f'Now open Telegram, send @{bot} any message (e.g. hello), then press Enter here ... ')
    chat_id = last_chat_id(token)
    while not chat_id:
        answer = ask(f'No message to @{bot} found. Send the bot a message and press Enter to try again, '
                     f'or type your chat id')
        chat_id = answer if re.fullmatch(r'-?\d+', answer) else last_chat_id(token)
    print(f'Chat id found: {chat_id}')
    return token, chat_id, bot


def ask_keys(config, keep_telegram):
    """{'coingecko': ..., 'claude': ..., 'model': ...} or None to keep the current ones."""
    header('Step 3/3: API keys (optional)')
    if keep_telegram and (config_value(config, 'coingecko', 'api_key') or config_value(config, 'anthropic', 'api_key')):
        if yes('Your API keys are already set. Keep them?'):
            return None
    print('CoinGecko demo key (coingecko.com/en/developers/dashboard): CaSa can watch more cryptos with it.')
    coingecko = ask_secret('CoinGecko key, Enter = none')
    print('\nClaude API key (console.anthropic.com): for the buttons AI decision (API) and AI plan (API);')
    print('each request costs a few cents.')
    claude = ask_secret('Claude key, Enter = none')
    model = MODELS[0][0]
    if claude:
        print('\nWhich Claude model should answer?')
        for number, (name, text) in enumerate(MODELS, 1):
            print(f'  {number}  {name:20} {text}')
        choice = ask('Model (1-3)', '1')
        model = MODELS[int(choice) - 1][0] if choice in ('1', '2', '3') else model
    return {'coingecko': coingecko, 'claude': claude, 'model': model}


def install_packages(folder):
    python = os.path.join(folder, '.venv', 'Scripts', 'python.exe')
    if not os.path.exists(python):
        if subprocess.call([sys.executable, '-m', 'venv', '.venv'], cwd=folder):
            raise SetupError('The Python environment .venv could not be created.')
    subprocess.call([python, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', '--upgrade', 'pip'],
                    cwd=folder)
    if subprocess.call([python, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', '--upgrade',
                        '-r', 'requirements.txt'], cwd=folder):
        raise SetupError('The Python packages could not be installed (internet?).')
    return python


def write_start_file(folder):
    """start_casa.bat: starts CaSa by hand (never automatically)."""
    with open(os.path.join(folder, 'start_casa.bat'), 'w', encoding='utf-8', newline='\r\n') as file:
        file.write(START_BAT)


def main():
    os.system('')     # switches on the screen control codes of the Windows console (for the *)
    here = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
    this_file = sys.argv[2] if len(sys.argv) > 2 else ''
    print(LINE)
    print('  Welcome to the CaSa setup')
    print(LINE)
    print('This wizard sets up CaSa on this computer. It asks for:')
    print('  1. The folder for CaSa')
    print('  2. Your Telegram bot (token from @BotFather)')
    print('  3. Optional: CoinGecko and Claude API key')
    print('Then it downloads the newest CaSa from GitHub, installs the Python packages (2-5 minutes)')
    print('and writes start_casa.bat. Your own files (config.ini, assets.ini, transactions.csv ...) always stay.')
    wait_enter('\nPress Enter to start, or close this window to cancel ... ')

    folder, program = choose_folder(here)
    tag = ''
    if program == 'download':
        try:
            tag = newest_release()
        except SetupError:
            raise
        except Exception:
            raise SetupError('GitHub cannot be reached. Please check the internet connection.')
        print(f'Newest CaSa release: {tag}')
    config = os.path.join(folder, 'config.ini')
    bot = ask_telegram(config)
    keys = ask_keys(config, bot is None)

    header('Summary')
    print(f'Folder:              {folder}')
    print(f'CaSa:                {f"download release {tag}" if tag else "keep the program files"}')
    print(f'Telegram bot:        {f"@{bot[2]} (chat id {bot[1]})" if bot else "unchanged"}')
    if keys:
        print(f'CoinGecko key:       {"set" if keys["coingecko"] else "none"}')
        claude = 'set, model ' + keys['model'] if keys['claude'] else 'none'
        print(f'Claude key:          {claude}')
    else:
        print('API keys:            unchanged')
    print()
    if not yes('Install CaSa?'):
        raise Cancel

    header('Installing')
    os.makedirs(folder, exist_ok=True)
    if tag:
        print(f'[1/4] Downloading CaSa {tag} from GitHub ...')
        try:
            download(tag, folder)
        except Exception:
            raise SetupError('CaSa could not be downloaded from GitHub (internet?).')
    else:
        print('[1/4] Program files: kept')
    if this_file and os.path.exists(this_file) and not os.path.exists(os.path.join(folder, 'install_casa.bat')):
        shutil.copy2(this_file, os.path.join(folder, 'install_casa.bat'))   # for later: settings and updates
    print('[2/4] Installing the Python packages (2-5 minutes) ...')
    install_packages(folder)
    print('[3/4] Saving your settings ...')
    values = {}
    if bot:
        values[('telegram', 'token')], values[('telegram', 'chat_id')] = bot[0], bot[1]
    if keys:
        values[('coingecko', 'api_key')] = keys['coingecko']
        values[('anthropic', 'api_key')] = keys['claude']
        values[('anthropic', 'model')] = keys['model']
    if values or not os.path.exists(config):
        write_config(config, values)
    assets = os.path.join(folder, 'assets.ini')
    if not os.path.exists(assets):
        with open(assets, 'w', encoding='utf-8') as file:
            file.write(ASSETS_EXAMPLE)
    print('[4/4] Start file start_casa.bat ...')
    write_start_file(folder)

    header('Done - CaSa is set up!')
    print(f'Your files are in {folder}')
    print('  1. Enter your assets in assets.ini (an example with Bitcoin, Ethereum and Rheinmetall is there).')
    print('  2. Start CaSa: double-click start_casa.bat in this folder. Your bot sends "CaSa started"')
    print('     within a minute; closing that window stops CaSa. CaSa never starts by itself.')
    print('  Settings or a new CaSa version: install_casa.bat in this folder')
    print('  Only one CaSa per Telegram bot: stop CaSa on other devices first (e.g. a Raspberry Pi).')
    print(LINE)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Cancel, KeyboardInterrupt):
        print('\n\nSetup cancelled. Start install_casa.bat again at any time.')
        sys.exit(1)
    except SetupError as error:
        print(f'\n\nError: {error}\nNothing is lost - start install_casa.bat again when the problem is solved.')
        sys.exit(1)
