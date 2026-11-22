# Project Structure

[← back to the README](../readme.md)

| File | Purpose |
|---|---|
| `main.py` | Main program: loads the files, schedules analyses, handles commands from the bot |
| `classes.py` | `CryptoStock` class (data, transactions, analysis per asset), reading and writing `assets.ini`, `events.ini`, `transactions.csv` |
| `func.py` | Data retrieval, indicators, support/resistance, bottom/top formation, market phase, alarms, charts, summary image |
| `telegram_bot.py` | Telegram bot, runs as a separate process: sends messages, forwards commands to `main.py` |
| `install_casa.bat` | Setup wizard for Windows: installs Python and CaSa, asks for the keys, writes `start_casa.bat` |
| `image/` | Building the CaSa image for the Raspberry Pi (pi-gen) |
| `docs/` | Manual pages and images of this README |
| `requirements.txt` | Python packages |
