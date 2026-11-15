#!/usr/bin/env python
"""
Bundle the bot's logs plus Binance price history into one zip for offline analysis.

Run it from the bot folder (where polymarket_bot.py and its .json logs live):

    python export_for_analysis.py              # last 90 days of Binance 1m prices
    python export_for_analysis.py --days 30    # smaller file
    python export_for_analysis.py --no-prices  # logs only
    python export_for_analysis.py --binance https://api.binance.us/api/v3

Writes analysis_export_<date>.zip (gitignored). It contains:
  logs/          trade logs, RL-blocked shadow logs, prediction/signal logs, *.jsonl
  binance_1m/    <PAIR>.csv  open_time_ms,open,high,low,close,volume
  manifest.json  what was included and skipped

Never included: .env files, encrypted keys, salts, settings/config files,
Telegram inbox, or any file whose text looks like it holds a secret.
"""
import argparse
import csv
import fnmatch
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone

INCLUDE = [
    '*_trades*.json', '*_rl_blocked*.json', '*.jsonl',
    'bot_predictions.json', 'signals_log.json', 'claude_paper_log.json',
    'max_backtest_results.json', 'btc_hermes_journal*.json', 'btc_hermes_matrix.json',
    'algos.json', 'algo_assignments.json', 'claude_strategy_modes.json',
]
NEVER = [
    '.env*', '*.env', '*.enc', '*.salt', '*settings*.json', 'bot_config*.json',
    'sizing_config.json', 'tg_inbox*.json', '*key*', '*secret*', '*wallet*',
    'analysis_export_*.zip',
]
SECRET_RE = re.compile(
    r'(private[_-]?key|api[_-]?secret|passphrase|mnemonic|seed[_ -]?phrase|'
    r'sk-ant-[A-Za-z0-9_-]{10,}|\b\d{8,10}:[A-Za-z0-9_-]{35}\b)',
    re.IGNORECASE)
PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'BNBUSDT', 'HYPEUSDT']


def select_files(folder: str):
    """Return (included, skipped) lists of (name, reason) for files in `folder`."""
    included, skipped = [], []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if any(fnmatch.fnmatch(name, p) for p in NEVER):
            if any(fnmatch.fnmatch(name, p) for p in INCLUDE):
                skipped.append((name, 'excluded name'))
            continue
        if not any(fnmatch.fnmatch(name, p) for p in INCLUDE):
            continue
        try:
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except OSError as e:
            skipped.append((name, f'unreadable: {e}'))
            continue
        if SECRET_RE.search(text):
            skipped.append((name, 'looks like it contains a secret'))
            continue
        included.append((name, os.path.getsize(path)))
    return included, skipped


def fetch_klines(session, base: str, pair: str, start_ms: int, end_ms: int):
    """All 1m candles for pair in [start_ms, end_ms), paginated 1000 at a time."""
    rows, t = [], start_ms
    while t < end_ms:
        r = session.get(f"{base}/klines", params={'symbol': pair, 'interval': '1m',
                                                  'startTime': t, 'endTime': end_ms - 1,
                                                  'limit': 1000}, timeout=15)
        if r.status_code == 400:            # unknown pair on this exchange
            return None
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend((k[0], k[1], k[2], k[3], k[4], k[5]) for k in batch)
        t = int(batch[-1][0]) + 60_000
        time.sleep(0.15)                    # stay well under rate limits
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--folder', default='.', help='bot folder (default: current)')
    ap.add_argument('--days', type=int, default=90)
    ap.add_argument('--no-prices', action='store_true')
    ap.add_argument('--binance', default='https://api.binance.com/api/v3')
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    included, skipped = select_files(folder)
    if not included:
        sys.exit(f"No log files found in {folder}. Run this from the bot folder.")

    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    out = os.path.join(folder, f'analysis_export_{stamp}.zip')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'logs': [{'name': n, 'bytes': b} for n, b in included],
                'skipped': [{'name': n, 'reason': r} for n, r in skipped],
                'prices': {}}

    print(f"Logs to include ({len(included)}):")
    for n, b in included:
        print(f"  {n:<45} {b / 1e6:7.2f} MB")
    for n, r in skipped:
        print(f"  SKIPPED {n}: {r}")

    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n, _ in included:
            z.write(os.path.join(folder, n), f'logs/{n}')

        if not args.no_prices:
            import requests
            s = requests.Session()
            end_ms = int(time.time() // 60 * 60 * 1000)
            start_ms = end_ms - args.days * 86_400_000
            print(f"\nDownloading {args.days}d of Binance 1m candles (a few minutes)...")
            for pair in PAIRS:
                try:
                    rows = fetch_klines(s, args.binance.rstrip('/'), pair, start_ms, end_ms)
                except Exception as e:
                    print(f"  {pair}: failed ({e})")
                    manifest['prices'][pair] = f'failed: {e}'
                    continue
                if rows is None:
                    print(f"  {pair}: not listed on this exchange, skipped")
                    manifest['prices'][pair] = 'not listed'
                    continue
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(['open_time_ms', 'open', 'high', 'low', 'close', 'volume'])
                w.writerows(rows)
                z.writestr(f'binance_1m/{pair}.csv', buf.getvalue())
                manifest['prices'][pair] = len(rows)
                print(f"  {pair}: {len(rows):,} candles")

        z.writestr('manifest.json', json.dumps(manifest, indent=2))

    print(f"\nDone: {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    print("Upload this zip in the Claude session.")


if __name__ == '__main__':
    main()
