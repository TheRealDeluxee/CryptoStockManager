# Installation

[← back to the README](../readme.md)

## Option A: Manual Installation (2–5 minutes on Windows)

**Good for:** trying CaSa out, or running it on a computer you already have (Windows, Linux, macOS, a home server). CaSa only runs while that computer is on.

### Windows: the setup wizard

1. **Create your Telegram bot**: in Telegram, send `/newbot` to [@BotFather](https://t.me/BotFather) and copy the token
2. **Download [`install_casa.bat`](https://github.com/TheRealDeluxee/CryptoStockManager/releases/latest)** (in the release under *Source code (zip)*) and **double-click** it. If Windows warns "Windows protected your PC": *More info* → *Run anyway*
3. **Answer the questions** in the window
4. **Your assets**: enter them in `assets.ini` (see [assets.ini](configuration.md#assetsini))
5. **Start CaSa**: double-click `start_casa.bat` in your CaSa folder. Your bot sends "CaSa started" within a minute; closing the window stops CaSa. CaSa never starts by itself (not after the setup, not with Windows, not as a service)

Later: after changes to `assets.ini` send `/restart` to the bot. To change the keys or get a new CaSa version, start `install_casa.bat` in your CaSa folder again – your files stay.

### Linux, macOS or by hand

1. **Install Python 3** (tested with Python 3.13)
2. **Install the dependencies**: `python3 -m venv .venv` and `.venv/bin/pip install -r requirements.txt`
3. **Create a Telegram bot**
   - Send `/newbot` to [@BotFather](https://t.me/BotFather) and copy the token
   - Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` – the number at `"chat":{"id":...}` is your chat id
4. **Create `config.ini` and `assets.ini`** in the program folder (see [Configuration](configuration.md#configuration))
5. **Start CaSa**: `.venv/bin/python main.py`

After the start, the bot sends the CaSa logo with a short message: the version, the number of monitored assets and the computer CaSa runs on (network name and operating system in two lines, e.g. `casa-pi` and `Debian GNU/Linux 13 (trixie)`). Send `/help` for all commands.

For permanent use on a Raspberry Pi, [Option B](#option-b-casa-image-for-the-raspberry-pi-about-20-minutes) is easier: no Python installation, automatic start and updates in Telegram.

## Option B: CaSa Image for the Raspberry Pi (about 20 minutes)

**Good for:** a small computer of its own that runs CaSa day and night – it starts by itself, also after a power cut, and updates with one tap in Telegram.

The easiest way to run CaSa: a ready **CaSa image** – Raspberry Pi OS Lite 64-bit (Debian 13 Trixie, without desktop) with the Python packages of CaSa, Samba and tmux already installed. Write it with the Raspberry Pi Imager, start the Pi, log in once: a wizard asks for the keyboard layout, whether you want the network drive (optional) and your keys, then downloads the **newest CaSa release from GitHub**. From then on CaSa starts by itself, also after a power cut.

CaSa itself is not in the image: it always comes from GitHub, so the same image works for every CaSa version.

You need: a Raspberry Pi 4 or 5, a microSD card (16 GB or more), a power supply, network (LAN or Wi-Fi) and a PC with an SD card reader.

### Hardware

| | Minimum | Recommended |
|---|---|---|
| Raspberry Pi | Pi 4, 2 GB | Pi 5, 8 GB (room for an own AI model) |
| Storage | microSD 32 GB, A2 | NVMe SSD (M.2 HAT) |
| Extras | official power supply | official 27 W power supply, Active Cooler |

64-bit only (Pi 3, 4, 5, Zero 2 W); the Zero 2 W (512 MB) is too small. An AI model trained on the PC runs on the Pi 5 without an AI accelerator.

### Step 1: Write the image

1. Install and start the **Raspberry Pi Imager** on your PC: [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. **Device**: your Raspberry Pi (4 or 5)
3. **Operating system**: scroll down to **Use custom** and choose `casa.img.xz`
4. **Storage**: your SD card
5. **Next** → **Edit settings**:
   - Tab *General*: user name **`casa-user`** and a password (note it down); Wi-Fi name and password (not needed with a LAN cable). Hostname (`casa-pi`), time zone (Europe/Berlin) and keyboard (German) are already set in the image
   - Tab *Services*: **Enable SSH** → *Use password authentication*
6. **Save** → **Yes** → **Yes**: the card is written (all data on it is deleted)
7. Put the card into the Pi, connect power (and LAN), wait 2–3 minutes

Without the settings in step 5 the Pi asks on its own screen at the first start for the keyboard, a user name and a password (screen and keyboard needed).

### Step 2: Log in once – the wizard starts by itself

On the PC in **PowerShell** (or with screen and keyboard directly on the Pi):

```powershell
ssh casa-user@casa-pi
```

After the password the **CaSa setup wizard** starts by itself (keys: arrow keys and Tab, Enter to confirm):

| Window | What you do |
|---|---|
| Welcome | **Start** |
| Keyboard (1/4) | Your keyboard layout, e.g. **de** German (for a keyboard connected to the Pi) |
| Network drive (2/4) | Optional: **Set up** the drive `\\casa-pi\Python` (then a password, twice) or **No drive** – then you edit your files on the Pi (e.g. `nano ~/Python/assets.ini`). Later again with `casa-setup`: keep, new password or remove |
| Telegram (3/4) | Paste the bot token (from [@BotFather](https://t.me/BotFather), right-click in PowerShell) – it is checked with Telegram. Then send your bot any message: the chat id is found automatically |
| API keys (4/4) | CoinGecko and Claude key (both optional, leave empty for none) and the Claude model |
| Summary | **Apply** – downloads the newest CaSa from GitHub (internet needed) and updates the Python packages (1–3 minutes) |
| Done | CaSa runs, with its version; the address of the network drive |

Your bot sends "CaSa started" within a minute. Keys and passwords appear neither on the screen nor in a log.

### Step 3: Map the network drive in Windows (if set up)

1. Explorer → **This PC** → **Map network drive**
2. Folder: `\\casa-pi\Python` (or with the IP of the Pi: `\\192.168.x.x\Python`)
3. Tick **Connect using different credentials** → **Finish**
4. User `casa-user`, the password from the wizard (step 2/4), tick **Remember my credentials**

The drive shows your CaSa folder (`/home/casa-user/Python` on the Pi): `config.ini`, `assets.ini`, `transactions.csv`, `logfile.txt`, `plots` …

### Step 4: Your assets

Open `assets.ini` on the drive (an example with Bitcoin, Ethereum and Rheinmetall is there), enter your assets (see [assets.ini](configuration.md#assetsini)), save and send `/restart` to the bot. Moving from an earlier installation: copy your `config.ini`, `assets.ini`, `transactions.csv`, `events.ini`, `planalert.ini` and `alarm_journal.csv` onto the drive and send `/restart` – stop CaSa on the old device first (only one CaSa per Telegram bot).

### Daily use on the Pi

| What | Command (after `ssh casa-user@casa-pi`) |
|---|---|
| Status | `systemctl status casa` |
| Log | `logfile.txt` in your CaSa folder `~/Python` (also on the network drive) |
| Stop / start / restart | `sudo systemctl stop casa` / `start casa` / `restart casa` – or `/restart` in Telegram |
| Update to the newest CaSa release | `/update` in Telegram, or the button **⬆ Update now** of the daily message (see below), or `casa-update` on the Pi (a certain version, also back: `casa-update v0.1.12`) |
| Keyboard, drive password, keys | `casa-setup` |

**Updates:** CaSa checks for a new release every day at 19:00, or at once with `/update`. If there is one, tap **⬆ Update now**: CaSa updates itself in about a minute, your files stay, and the bot confirms with "✅ Update complete". If something goes wrong, the old version keeps running. This works with the CaSa image; elsewhere the message shows how to update by hand.

CaSa runs as the service `casa`: it starts at every boot and again 30 seconds after an error. `casa-update` gets the newest release from GitHub, updates the Python packages and restarts CaSa; your files (`config.ini`, `assets.ini`, `transactions.csv` …) stay.
