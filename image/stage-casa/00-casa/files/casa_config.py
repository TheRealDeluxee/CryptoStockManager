"""CaSa image: writes the keys from the setup wizard into config.ini.

    printf 'token=...\\nchat_id=...\\ncoingecko=...\\nclaude=...\\nmodel=...\\n' | python3 casa_config.py ~/Python/config.ini

The keys come via stdin (never as arguments: not visible in the process list). A new config.ini gets all
settings with comments; in an existing one only these values change, comments and other lines stay.
CaSa adds the remaining sections ([alerts], [ai_prompt]) itself at the start.
"""
import os
import sys
import tempfile

TEMPLATE = """[telegram]
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
# the API costs money per request (billed by Anthropic per token)
api_key =
model = claude-sonnet-5-5
"""

FIELDS = (('telegram', 'token', 'token'), ('telegram', 'chat_id', 'chat_id'),
          ('coingecko', 'api_key', 'coingecko'), ('anthropic', 'api_key', 'claude'), ('anthropic', 'model', 'model'))


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


def main(path):
    answers = {}
    for line in sys.stdin:
        if '=' in line:
            key, value = line.split('=', 1)
            answers[key.strip()] = value.strip()
    if not answers.get('token') or not answers.get('chat_id'):
        sys.exit('casa_config.py: token and chat_id are required.')
    answers['model'] = answers.get('model') or 'claude-sonnet-5-5'
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as file:
            lines = file.read().splitlines()
    else:
        lines = TEMPLATE.splitlines()
    for section, key, name in FIELDS:
        set_value(lines, section, key, answers.get(name, ''))
    # write a complete new file first, then replace: never a half-written config.ini
    folder = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=folder, suffix='.tmp', delete=False) as file:
        file.write('\n'.join(lines) + '\n')
    os.chmod(file.name, 0o600)   # the keys: only readable for you
    os.replace(file.name, path)
    print('config.ini saved (the keys are not shown).')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'config.ini')
