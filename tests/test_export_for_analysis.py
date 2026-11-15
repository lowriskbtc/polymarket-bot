import zipfile

from export_for_analysis import fetch_klines, select_files


def test_select_files_includes_logs_and_never_secrets(tmp_path):
    files = {
        'claude_paper_trades.json': '[{"condition_id": "0x' + 'ab' * 32 + '"}]',
        'hft_rl_blocked.weekend.json': '[]',
        'mispriced_regime_log.jsonl': '{}\n',
        'bot_predictions.json': '[]',
        '.env': 'PRIVATE_KEY=abc',
        'keys.enc': 'xxx',
        'hft_settings.json': '{}',
        'bot_config.json': '{}',
        'tg_inbox.json': '[]',
        'btc_ml_trades.json': '{"note": "private_key leaked here"}',
        'eth_ml_trades.json': '{"tg": "123456789:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}',
        'random_notes.txt': 'hi',
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    (tmp_path / 'subdir').mkdir()

    included, skipped = select_files(str(tmp_path))
    names = {n for n, _ in included}
    assert names == {'claude_paper_trades.json', 'hft_rl_blocked.weekend.json',
                     'mispriced_regime_log.jsonl', 'bot_predictions.json'}
    reasons = dict(skipped)
    assert reasons['btc_ml_trades.json'] == 'looks like it contains a secret'
    assert reasons['eth_ml_trades.json'] == 'looks like it contains a secret'
    for secret in ('.env', 'keys.enc', 'hft_settings.json', 'bot_config.json', 'tg_inbox.json'):
        assert secret not in names


class _Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def raise_for_status(self):
        assert self.status_code == 200

    def json(self):
        return self._data


class _Session:
    """Serves 1m candles from 0 to 2500 minutes, 1000 per page."""
    def __init__(self, known=True):
        self.known, self.calls = known, 0

    def get(self, url, params, timeout):
        self.calls += 1
        if not self.known:
            return _Resp(400, {})
        t, end = params['startTime'], min(params['endTime'] + 1, 2500 * 60_000)
        rows = [[ms, '1', '2', '0.5', '1.5', '10'] for ms in range(t, end, 60_000)][:1000]
        return _Resp(200, rows)


def test_fetch_klines_paginates(monkeypatch):
    monkeypatch.setattr('export_for_analysis.time.sleep', lambda s: None)
    s = _Session()
    rows = fetch_klines(s, 'x', 'BTCUSDT', 0, 2500 * 60_000)
    assert len(rows) == 2500 and s.calls == 3
    assert rows[0][0] == 0 and rows[-1][0] == 2499 * 60_000
    assert fetch_klines(_Session(known=False), 'x', 'FOOUSDT', 0, 60_000) is None
