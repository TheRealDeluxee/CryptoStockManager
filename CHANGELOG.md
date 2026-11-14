# Changelog

All notable changes to CaSa are documented here.
Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

- **PATCH** (0.1.**1**): bug fixes only
- **MINOR** (0.**2**.0): new features; while the version starts with 0, this can also include changes to commands or files
- **MAJOR** (**1**.0.0): first stable release, after that only for changes that require adjustments by the user

## [0.1.0] – 2026-11-14

### Added
- Telegram bot for notifications, charts and commands (replaces Pushover)
- `/check` and `/ai` with buttons, `/alarm`, `/event`, `/sellall`
- `assets.ini` as the list of all assets; `; 1` marks assets for the alarm analysis
- `events.ini` with reminders and automatic daily analysis
- AI prompt (`ai_prompt.txt`) with support/resistance, bottom/top formation, market phase and position
- Support/resistance zones and volume profile in the chart
- Daily summary as a ranking by 1-day P&L
- Longer history in the background (250 daily / 450 hourly values)

### Changed
- `watchlist.ini` renamed to `assets.ini`
- Whole program in English; `/termin` is now `/event`, `/alarm` uses `on/off` (old inputs still work)
- Chart: trading-day axis, 200 dpi, percentage axis on the left
- Version number is now kept in `main.py`; the `[version]` section in `config.ini` is no longer needed

### Fixed
- No duplicate daily crypto alarms
- Price no longer cut off at Bollinger breakouts
- Evening summary kept after `/buy` / `/sell`
- `requirements.txt` was missing a line break

## [0.0.3] and earlier

Pushover version with watchlist and basic technical analysis.
