"""Claude API function in func.py: without a key it cannot be used, the key never lands in the log.
No real request is sent, the Anthropic client is replaced by a fake."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import func

anthropic = pytest.importorskip('anthropic')
httpx2 = pytest.importorskip('httpx2')

SECRET = 'sk-ant-TEST-SECRET-KEY'


class FakeClient:
    """Stands in for anthropic.Anthropic: remembers the requests, returns fixed answers (one per request,
    the last one again) or raises."""
    last = {}

    def __init__(self, api_key):
        FakeClient.last = {'api_key': api_key, 'calls': 0}
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    @contextmanager
    def _stream(self, **request):
        FakeClient.last['request'] = dict(request, messages=list(request['messages']))
        FakeClient.last['calls'] += 1
        results = FakeClient.result if isinstance(FakeClient.result, list) else [FakeClient.result]
        result = results[min(FakeClient.last['calls'], len(results)) - 1]
        if isinstance(result, Exception):
            raise result
        yield SimpleNamespace(get_final_message=lambda: result)


def answer(text='Hello', stop_reason='end_turn', category=None, searches=0, model='claude-opus-5-5'):
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type='thinking', thinking='...'), SimpleNamespace(type='text', text=text)],
        stop_reason=stop_reason, stop_details=SimpleNamespace(category=category),
        usage=SimpleNamespace(input_tokens=12, output_tokens=34,
                              server_tool_use=SimpleNamespace(web_search_requests=searches)))


def api_error(cls, status):
    response = httpx2.Response(status, request=httpx2.Request('POST', 'https://api.anthropic.com/v1/messages'))
    return cls('error', response=response, body=None)


@pytest.fixture
def claude_api(sandbox, monkeypatch):
    monkeypatch.setattr(func, 'anthropic_api_key', SECRET)
    monkeypatch.setattr(func, 'anthropic_model', 'claude-opus-5-5')
    monkeypatch.setattr(anthropic, 'Anthropic', FakeClient)
    FakeClient.result = answer()
    return sandbox


def log_text(sandbox):
    path = sandbox.path / func.LOG_FILE
    return path.read_text(encoding='utf-8') if path.exists() else ''


def test_without_key_not_usable(sandbox, monkeypatch):
    monkeypatch.setattr(func, 'anthropic_api_key', '')
    monkeypatch.setattr(anthropic, 'Anthropic', FakeClient)
    FakeClient.last = {}
    assert func.claude_available() is False
    with pytest.raises(RuntimeError, match='not configured'):
        func.claude('Hi')
    assert FakeClient.last == {}                                    # no request was sent


def test_answer_text_and_request(claude_api):
    assert func.claude_available() is True
    assert func.claude('How is Bitcoin?', system='You are short.', effort='low') == 'Hello'
    request = FakeClient.last['request']
    assert FakeClient.last['api_key'] == SECRET
    assert request['model'] == 'claude-opus-5-5'
    assert request['messages'] == [{'role': 'user', 'content': 'How is Bitcoin?'}]
    assert request['system'] == 'You are short.'
    assert request['output_config'] == {'effort': 'low'}
    assert request['fallbacks'] == 'default' and 'server-side-fallback-2026-07-01' in request['betas']
    assert 'thinking' not in request                               # not allowed on Opus 5.5
    log = log_text(claude_api)
    assert 'Claude API: answer' in log and '12 tokens in' in log
    assert SECRET not in log


def test_no_system_prompt_not_sent(claude_api):
    func.claude('Hi')
    assert 'system' not in FakeClient.last['request'] and 'tools' not in FakeClient.last['request']


def test_web_search_and_paused_answer(claude_api):
    FakeClient.result = [answer('Part 1. ', stop_reason='pause_turn', searches=2), answer('Part 2.', searches=1)]
    assert func.claude('News?', web_search=True) == 'Part 1. Part 2.'
    request = FakeClient.last['request']
    assert request['tools'] == [{'type': 'web_search_20260209', 'name': 'web_search', 'max_uses': 5}]
    assert FakeClient.last['calls'] == 2
    assert request['messages'][-1]['role'] == 'assistant'           # paused part sent back to continue
    assert '3 web searches' in log_text(claude_api)


def test_usage_and_cost(claude_api):
    FakeClient.result = [answer('A ', stop_reason='pause_turn', searches=2, model='claude-sonnet-5-5'),
                         answer('B', searches=1, model='claude-sonnet-5-5')]
    text, usage = func.claude('News?', web_search=True, with_usage=True)
    assert text == 'A B'
    assert (usage['tokens_in'], usage['tokens_out'], usage['searches']) == (24, 68, 3)
    # 24 tokens in at $2/M, 68 out at $10/M, 3 searches at $0.01
    assert usage['cost'] == pytest.approx(24 * 2e-6 + 68 * 10e-6 + 0.03)
    assert usage['models'] == ['claude-sonnet-5-5']
    assert 'about $0.031' in log_text(claude_api)


def test_usage_text_and_prices(monkeypatch):
    import pandas as pd
    assert func.claude_cost('claude-opus-5-5', 1_000_000, 100_000) == pytest.approx(4 + 2)
    assert func.claude_cost('claude-haiku-5-5', 200_000, 0) == pytest.approx(0.10)    # long prompt: $0.50/M
    assert func.claude_cost('claude-unknown', 10, 10) is None
    monkeypatch.setattr(func, '_fx_rates', lambda currency, period: pd.DataFrame({'Rate': [0.9]}))
    usage = {'tokens_in': 12345, 'tokens_out': 678, 'searches': 1, 'cost': 0.05, 'seconds': 40.6,
             'models': ['claude-sonnet-5-5']}
    assert func.claude_usage_text(usage) == ('12,345 tokens in · 678 out · 1 web search · $0.050 (≈ 0.045 €) · '
                                             '41 s · claude-sonnet-5-5')
    assert 'cost not known' in func.claude_usage_text(dict(usage, cost=None))


def test_web_search_haiku_older_tool(claude_api, monkeypatch):
    monkeypatch.setattr(func, 'anthropic_model', 'claude-haiku-5-5')
    func.claude('News?', web_search=True)
    assert FakeClient.last['request']['tools'][0]['type'] == 'web_search_20250305'


def test_haiku_without_fallback(claude_api, monkeypatch):
    monkeypatch.setattr(func, 'anthropic_model', 'claude-haiku-5-5')
    func.claude('Hi')
    request = FakeClient.last['request']
    assert request['model'] == 'claude-haiku-5-5'
    assert 'fallbacks' not in request and 'betas' not in request   # Haiku answers fallbacks with 400


def test_refusal(claude_api):
    FakeClient.result = answer(text='', stop_reason='refusal', category='cyber')
    with pytest.raises(RuntimeError, match='refused.*cyber'):
        func.claude('Hi')


@pytest.mark.parametrize('error, expected, text', [
    (lambda: api_error(anthropic.AuthenticationError, 401), PermissionError, 'rejected'),
    (lambda: api_error(anthropic.PermissionDeniedError, 403), PermissionError, 'rejected'),
    (lambda: api_error(anthropic.NotFoundError, 404), RuntimeError, 'not found'),
    (lambda: api_error(anthropic.RateLimitError, 429), RuntimeError, 'too many requests'),
    (lambda: api_error(anthropic.InternalServerError, 500), RuntimeError, 'HTTP 500'),
    (lambda: anthropic.APIConnectionError(request=httpx2.Request('POST', 'https://api.anthropic.com')),
     RuntimeError, 'no connection'),
])
def test_errors(claude_api, error, expected, text):
    FakeClient.result = error()
    with pytest.raises(expected, match=text):
        func.claude('Hi')
    assert SECRET not in log_text(claude_api)
