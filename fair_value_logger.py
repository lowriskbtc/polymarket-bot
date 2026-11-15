#!/usr/bin/env python
"""
Fair-value logger for Polymarket crypto Up/Down markets. LOGGING ONLY: places no orders.

Every few seconds, for each symbol and timeframe, it records:
  - Binance spot, the window's start price and recent realized volatility
  - the model's fair P(Up) (see fair_value.py)
  - the real Polymarket order book (best bid/ask and size for Up and Down)
After each window closes it records how the market actually resolved.

Then run fair_value_report.py to see whether the model is calibrated and
whether buying when fair value beats the ask would have made money.

Usage:
    python fair_value_logger.py                                 # BTC,ETH,SOL,XRP x 5m,15m
    python fair_value_logger.py --symbols BTC,ETH --tfs 15m --interval 3
    python fair_value_logger.py --binance https://api.binance.us/api/v3   # if Binance.com is blocked

Output: fair_value_log.jsonl (gitignored). Safe to stop and restart at any time;
on restart it picks up unresolved markets from the existing file.
"""
import argparse
import json
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import requests

from fair_value import (TF_SECONDS, best_bid_ask, fair_prob_up, market_slug,
                        realized_vol_per_sec, window_start)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BINANCE = "https://api.binance.com/api/v3"
PAIRS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT',
         'DOGE': 'DOGEUSDT', 'BNB': 'BNBUSDT', 'HYPE': 'HYPEUSDT'}

VOL_REFRESH_S = 60        # recompute realized vol once a minute
VOL_BARS = 60             # from the last 60 completed 1m candles
RESOLVE_DELAY_S = 60      # wait this long after a window ends before asking Gamma
RESOLVE_GIVEUP_S = 3600   # after an hour, fall back to Binance for the outcome

log = logging.getLogger("fv")


class FairValueLogger:
    def __init__(self, symbols, tfs, interval, out_path, binance):
        self.symbols = symbols
        self.tfs = tfs
        self.interval = interval
        self.out_path = out_path
        self.binance = binance.rstrip('/')
        self.http = requests.Session()
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.running = True
        self.markets: Dict[str, dict] = {}       # slug -> market info (current windows)
        self.vol: Dict[str, tuple] = {}          # symbol -> (sigma_per_sec, fetched_at)
        self.pending: Dict[str, dict] = {}       # slug -> {symbol, tf, start_ts, end_ts}
        self.resolved = set()
        self._last_resolve_check = 0.0
        self._load_existing()

    # ── persistence ─────────────────────────────────────────────────────────
    def _write(self, row: dict):
        with open(self.out_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')

    def _load_existing(self):
        """Re-queue markets that were logged but never resolved (after a restart)."""
        if not os.path.exists(self.out_path):
            return
        seen = {}
        with open(self.out_path, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('type') == 'resolution':
                    self.resolved.add(r['slug'])
                elif r.get('type') == 'tick':
                    seen[r['slug']] = r
        for slug, r in seen.items():
            if slug not in self.resolved:
                self.pending[slug] = {'symbol': r['symbol'], 'tf': r['tf'],
                                      'start_ts': r['start_ts'],
                                      'end_ts': r['start_ts'] + TF_SECONDS[r['tf']]}
        if self.pending:
            log.info("Re-queued %d unresolved markets from %s", len(self.pending), self.out_path)

    # ── data sources ────────────────────────────────────────────────────────
    def _get(self, url, **params):
        r = self.http.get(url, params=params, timeout=5)
        r.raise_for_status()
        return r.json()

    def spot_prices(self) -> Dict[str, float]:
        pairs = [PAIRS[s] for s in self.symbols]
        data = self._get(f"{self.binance}/ticker/price",
                         symbols=json.dumps(pairs, separators=(',', ':')))
        by_pair = {d['symbol']: float(d['price']) for d in data}
        return {s: by_pair[PAIRS[s]] for s in self.symbols if PAIRS[s] in by_pair}

    def sigma(self, symbol: str) -> Optional[float]:
        cached = self.vol.get(symbol)
        if cached and time.time() - cached[1] < VOL_REFRESH_S:
            return cached[0]
        raw = self._get(f"{self.binance}/klines", symbol=PAIRS[symbol],
                        interval='1m', limit=VOL_BARS + 1)
        closes = [float(k[4]) for k in raw[:-1]]       # drop the still-open candle
        s = realized_vol_per_sec(closes, 60)
        self.vol[symbol] = (s, time.time())
        return s

    def binance_price_at(self, symbol: str, ts: int) -> Optional[float]:
        """Open of the 1m Binance candle starting at ts (the price at that moment)."""
        raw = self._get(f"{self.binance}/klines", symbol=PAIRS[symbol], interval='1m',
                        startTime=ts * 1000, limit=1)
        if raw and int(raw[0][0]) == ts * 1000:
            return float(raw[0][1])
        return None

    def gamma_market(self, slug: str) -> Optional[dict]:
        data = self._get(f"{GAMMA}/events", slug=slug)
        if not data:
            return None
        mkt = (data[0].get('markets') or [{}])[0]

        def _list(v):
            return json.loads(v) if isinstance(v, str) else (v or [])
        return {'outcomes': _list(mkt.get('outcomes')),
                'token_ids': _list(mkt.get('clobTokenIds')),
                'outcome_prices': _list(mkt.get('outcomePrices')),
                'closed': bool(mkt.get('closed'))}

    def book(self, token_id: str) -> dict:
        return self._get(f"{CLOB}/book", token_id=token_id)

    # ── per-window setup ────────────────────────────────────────────────────
    def ensure_market(self, symbol: str, tf: str, start_ts: int) -> Optional[dict]:
        slug = market_slug(symbol, tf, start_ts)
        if slug in self.markets:
            return self.markets[slug]
        g = self.gamma_market(slug)
        if not g or len(g['token_ids']) < 2:
            return None
        start_price = self.binance_price_at(symbol, start_ts)
        if start_price is None:
            return None
        idx = {o.lower(): i for i, o in enumerate(g['outcomes'])}
        if 'up' not in idx or 'down' not in idx:
            return None
        info = {'slug': slug, 'symbol': symbol, 'tf': tf, 'start_ts': start_ts,
                'end_ts': start_ts + TF_SECONDS[tf], 'start_price': start_price,
                'up_token': g['token_ids'][idx['up']],
                'down_token': g['token_ids'][idx['down']]}
        # Drop finished windows for this symbol/tf and queue them for resolution.
        for old_slug, old in list(self.markets.items()):
            if old['symbol'] == symbol and old['tf'] == tf:
                del self.markets[old_slug]
                if old_slug not in self.resolved:
                    self.pending[old_slug] = old
        self.markets[slug] = info
        log.info("New window %s  start=%.6g", slug, start_price)
        return info

    # ── main loop ───────────────────────────────────────────────────────────
    def tick(self):
        now = time.time()
        spots = self.spot_prices()
        for sym in self.symbols:
            if sym not in spots:
                continue
            try:
                sig = self.sigma(sym)
            except Exception as e:
                log.warning("vol %s: %s", sym, e)
                continue
            if not sig:
                continue
            for tf in self.tfs:
                try:
                    m = self.ensure_market(sym, tf, window_start(now, tf))
                except Exception as e:
                    log.warning("market %s %s: %s", sym, tf, e)
                    continue
                if not m:
                    continue
                secs_left = m['end_ts'] - now
                if secs_left <= 0:
                    continue
                up_f = self.pool.submit(self.book, m['up_token'])
                dn_f = self.pool.submit(self.book, m['down_token'])
                try:
                    up_b, dn_b = up_f.result(), dn_f.result()
                except Exception as e:
                    log.warning("book %s: %s", m['slug'], e)
                    continue
                book_ts = time.time()
                ub, ubs, ua, uas = best_bid_ask(up_b)
                db, dbs, da, das = best_bid_ask(dn_b)
                self._write({
                    'type': 'tick', 'ts': round(now, 3), 'book_ts': round(book_ts, 3),
                    'symbol': sym, 'tf': tf, 'slug': m['slug'], 'start_ts': m['start_ts'],
                    'secs_left': round(secs_left, 2), 'spot': spots[sym],
                    'start_price': m['start_price'], 'sigma': sig,
                    'fair_up': round(fair_prob_up(spots[sym], m['start_price'], secs_left, sig), 5),
                    'up_bid': ub, 'up_bid_size': ubs, 'up_ask': ua, 'up_ask_size': uas,
                    'down_bid': db, 'down_bid_size': dbs, 'down_ask': da, 'down_ask_size': das,
                })

    def resolve_pending(self):
        now = time.time()
        if now - self._last_resolve_check < 30:
            return
        self._last_resolve_check = now
        for slug, m in list(self.pending.items()):
            if now < m['end_ts'] + RESOLVE_DELAY_S:
                continue
            winner, source = None, None
            try:
                g = self.gamma_market(slug)
                if g and g['closed'] and g['outcome_prices']:
                    prices = [float(p) for p in g['outcome_prices']]
                    i = max(range(len(prices)), key=lambda k: prices[k])
                    if prices[i] >= 0.99:
                        winner, source = g['outcomes'][i].capitalize(), 'polymarket'
                if winner is None and now > m['end_ts'] + RESOLVE_GIVEUP_S:
                    start = m.get('start_price') or self.binance_price_at(m['symbol'], m['start_ts'])
                    end = self.binance_price_at(m['symbol'], m['end_ts'])
                    if start and end:
                        winner, source = ('Up' if end >= start else 'Down'), 'binance'
            except Exception as e:
                log.warning("resolve %s: %s", slug, e)
                continue
            if winner:
                self._write({'type': 'resolution', 'ts': round(now, 3), 'slug': slug,
                             'symbol': m['symbol'], 'tf': m['tf'], 'winner': winner,
                             'source': source})
                self.resolved.add(slug)
                del self.pending[slug]
                log.info("Resolved %s -> %s (%s)", slug, winner, source)

    def run(self, duration: Optional[float] = None):
        t_end = time.time() + duration if duration else None
        log.info("Logging %s x %s every %ss -> %s  (no orders are placed)",
                 ','.join(self.symbols), ','.join(self.tfs), self.interval, self.out_path)
        while self.running and (t_end is None or time.time() < t_end):
            t0 = time.time()
            try:
                self.tick()
            except Exception as e:
                log.warning("tick: %s", e)
            try:
                self.resolve_pending()
            except Exception as e:
                log.warning("resolve: %s", e)
            time.sleep(max(0.0, self.interval - (time.time() - t0)))
        self.pool.shutdown(wait=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--symbols', default='BTC,ETH,SOL,XRP')
    ap.add_argument('--tfs', default='5m,15m')
    ap.add_argument('--interval', type=float, default=5.0, help='seconds between snapshots')
    ap.add_argument('--out', default='fair_value_log.jsonl')
    ap.add_argument('--binance', default=BINANCE, help='Binance REST base URL')
    ap.add_argument('--duration', type=float, default=None, help='stop after N seconds')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    bad = [s for s in symbols if s not in PAIRS]
    if bad:
        ap.error(f"unknown symbols: {bad}")
    tfs = [t.strip() for t in args.tfs.split(',') if t.strip()]
    if any(t not in TF_SECONDS for t in tfs):
        ap.error("--tfs must be 5m and/or 15m")

    fvl = FairValueLogger(symbols, tfs, args.interval, args.out, args.binance)

    def _stop(*_):
        fvl.running = False
        log.info("Stopping...")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    fvl.run(args.duration)


if __name__ == '__main__':
    main()
