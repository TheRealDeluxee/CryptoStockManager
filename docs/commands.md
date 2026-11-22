# Telegram Commands

[← back to the README](../readme.md)

| Command | Function |
|---|---|
| **Analysis** | |
| `/check` | Daily analysis of one asset without alarm (choose via buttons or `/check <symbol or name>`); below the chart the same buttons as below alarms: 🤖 Prompt, with a Claude API key also 🤖 Guide (API) and 🤖 Plan (API) |
| 🤖 Prompt (button) | AI prompt to copy into ChatGPT or Claude; below every alarm, info and `/check`. Typing `/ai_prompt <symbol or name>` (or the old `/ai`) still works, it is just not in the command list any more |
| 🤖 Guide (API) (button, `/ai_api_decision3s`) | Claude API (key needed): the same data as `/ai_prompt`, answered directly with a short guide in three sentences (with web search for news) |
| 🤖 Plan (API) (button, `/ai_api_plan`) | Claude API (key needed): the same data, answered with a short trading plan and the values for `planalert.ini`; **✅ Save as plan** saves it (replaces an existing plan of the asset), **Discard** drops it |
| `/analyse <daily\|hourly\|summary>` | Start an analysis immediately |
| **Alarms and Dates** | |
| `/smart_alert` | Switch assets on/off for the alarm analysis (buttons or `/smart_alert <symbol or name> <on\|off>`); the old `/alarm` still works |
| `/smart_alert_config` | Alarm, info or off per signal, separately for stocks and crypto (buttons); saved in `config.ini` `[alerts]`; the last button switches infos for all assets or only for the assets you hold (`infos_for`) |
| `/event` | List upcoming dates with a 🗑 button to delete each one (with confirmation), or add one: `/event <symbol or name> <DD/MM/YYYY> <text>` |
| `/plan_alert` | Your trading plans with status and the buttons **➕ Add plan** and **🗑 Delete plan**. Add plan guides you step by step: asset (with its current price), buy-zone, sell-zone, stop-plan, note – each price can be skipped – then a summary to save. Delete plan asks which plan, then for confirmation. In one line still possible: `/plan_alert <symbol or name> <buy-zone> <sell-zone> <stop-plan> <note>` (`-` for a value you do not use). A new plan replaces the old one of the same asset. The old `/price_alert` still works |
| **Portfolio** | |
| `/portfolio` | Daily P&L table and 13-week P&L chart now (otherwise with the evening summary) |
| `/update` | Look for a new CaSa release on GitHub now (otherwise daily at 19:30); with the CaSa image the answer has the button **⬆ Update now** |
| `/record` | Record a transaction: buttons **🟢 Buy**, **🔴 Sell** and **Sell all**. Buy and sell go step by step: choose the asset with a button (sell: only assets you hold), then amount, quantity and date (today, yesterday or other), confirm; afterwards ↩ Undo. Sell all records the full position at the current price (confirmation required). In one line: `/record buy <symbol or name> <amount €> <qty> <DD/MM/YYYY>`, `/record sell <symbol or name> <amount €> <qty>` (today), `/record sellall <symbol or name> [amount €]`. The old `/buy`, `/sell` and `/sellall` still work |
| `/history` | Transactions of one asset as a table (newest first); asset via button or `/history <symbol or name>` |
| `/transactions` | All transactions as a table like the daily summary (newest first, 40 per image) |
| **System** | |
| `/status` | Status of the main program: last analyses, next scheduled runs, alarms since the start, memory, disk space, problems (during a running analysis it follows afterwards) |
| `/restart` | Restart the main program (reloads all files; after a program update stop and start CaSa completely) |
| `/help` | Overview of all commands |

Only messages from the chat id in `config.ini` are accepted.
