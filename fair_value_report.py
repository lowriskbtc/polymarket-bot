#!/usr/bin/env python
"""
Report on fair_value_log.jsonl (written by fair_value_logger.py). Stdlib only.

Answers, in order:
  1. CALIBRATION  When the model says 70%, does Up win about 70% of the time?
  2. MODEL vs MARKET  Brier score of the model vs the Polymarket mid price.
     If the market is already more accurate, the model adds nothing.
  3. TRADING  Buy the first time per market that fair value beats the ask by
     at least --min-edge. One trade per market, paid at the logged ask.
     Shown for several edge thresholds and time-left buckets, and split into
     the earlier and later half of the data. Treat a result as real only if it
     is positive in BOTH halves with t >= 2 and 200+ trades.

Usage:
    python fair_value_report.py
    python fair_value_report.py --log fair_value_log.jsonl --fee-rate 0.0 --min-size 5
"""
import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fair_value import edge

EDGE_GRID = (0.03, 0.05, 0.08, 0.12)
TIME_BUCKETS = (            # (label, min secs_left, max secs_left)
    ('all', 0, 10 ** 9),
    ('>10m', 600, 10 ** 9),
    ('2-10m', 120, 600),
    ('30s-2m', 30, 120),
    ('<30s', 0, 30),
)


# ── loading ──────────────────────────────────────────────────────────────────
def load(path: str) -> Tuple[Dict[str, List[dict]], Dict[str, str]]:
    """Return (ticks grouped by slug, sorted by time) and {slug: 'Up'|'Down'}."""
    ticks: Dict[str, List[dict]] = defaultdict(list)
    winners: Dict[str, str] = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('type') == 'tick':
                ticks[r['slug']].append(r)
            elif r.get('type') == 'resolution' and r.get('winner') in ('Up', 'Down'):
                winners[r['slug']] = r['winner']
    for rows in ticks.values():
        rows.sort(key=lambda r: r['ts'])
    return dict(ticks), winners


def market_mid_up(t: dict) -> Optional[float]:
    b, a = t.get('up_bid'), t.get('up_ask')
    if b is not None and a is not None and 0 < b <= a < 1:
        return (a + b) / 2
    return None


# ── 1 + 2: calibration and accuracy ─────────────────────────────────────────
def calibration(ticks: Dict[str, List[dict]], winners: Dict[str, str],
                key=lambda t: t['fair_up'], n_buckets: int = 10) -> List[dict]:
    """Bucket predicted P(Up) and compare with the actual Up rate."""
    b = defaultdict(lambda: {'n': 0, 'p_sum': 0.0, 'ups': 0, 'markets': set()})
    for slug, rows in ticks.items():
        if slug not in winners:
            continue
        up = winners[slug] == 'Up'
        for t in rows:
            p = key(t)
            if p is None:
                continue
            i = min(int(p * n_buckets), n_buckets - 1)
            b[i]['n'] += 1
            b[i]['p_sum'] += p
            b[i]['ups'] += up
            b[i]['markets'].add(slug)
    out = []
    for i in sorted(b):
        x = b[i]
        out.append({'lo': i / n_buckets, 'hi': (i + 1) / n_buckets, 'n': x['n'],
                    'markets': len(x['markets']), 'pred': x['p_sum'] / x['n'],
                    'actual': x['ups'] / x['n']})
    return out


def brier(ticks: Dict[str, List[dict]], winners: Dict[str, str]) -> Tuple[float, float, int]:
    """(model Brier, market-mid Brier, n) on ticks where both exist. Lower is better."""
    sm = sk = 0.0
    n = 0
    for slug, rows in ticks.items():
        if slug not in winners:
            continue
        y = 1.0 if winners[slug] == 'Up' else 0.0
        for t in rows:
            mid = market_mid_up(t)
            if mid is None:
                continue
            sm += (t['fair_up'] - y) ** 2
            sk += (mid - y) ** 2
            n += 1
    return (sm / n, sk / n, n) if n else (float('nan'), float('nan'), 0)


# ── 3: trading simulation ───────────────────────────────────────────────────
def simulate(ticks: Dict[str, List[dict]], winners: Dict[str, str], min_edge: float,
             fee_rate: float = 0.0, secs_lo: float = 0, secs_hi: float = 10 ** 9,
             min_size: float = 0.0, slugs=None) -> List[dict]:
    """First qualifying entry per resolved market. Pays the logged ask (+fee)."""
    trades = []
    for slug, rows in ticks.items():
        if slug not in winners or (slugs is not None and slug not in slugs):
            continue
        for t in rows:
            if not (secs_lo <= t['secs_left'] < secs_hi):
                continue
            best = None
            for side, p, ask, size in (
                    ('Up', t['fair_up'], t.get('up_ask'), t.get('up_ask_size')),
                    ('Down', 1.0 - t['fair_up'], t.get('down_ask'), t.get('down_ask_size'))):
                e = edge(p, ask, fee_rate)
                if e is None or e < min_edge or (size or 0) < min_size:
                    continue
                if best is None or e > best[1]:
                    best = (side, e, ask)
            if best:
                side, e, ask = best
                cost = ask * (1 + fee_rate)
                win = winners[slug] == side
                trades.append({'slug': slug, 'symbol': t['symbol'], 'tf': t['tf'],
                               'side': side, 'ask': ask, 'edge': e, 'win': win,
                               'pnl': (1.0 if win else 0.0) - cost, 'cost': cost,
                               'secs_left': t['secs_left'], 'start_ts': t['start_ts']})
                break
    return trades


def summarize(trades: List[dict]) -> dict:
    n = len(trades)
    if not n:
        return {'n': 0}
    pnls = [t['pnl'] for t in trades]
    mean = sum(pnls) / n
    sd = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1)) if n > 1 else 0.0
    cost = sum(t['cost'] for t in trades)
    return {'n': n, 'wr': sum(t['win'] for t in trades) / n,
            'avg_ask': sum(t['ask'] for t in trades) / n,
            'pnl': sum(pnls), 'roi': sum(pnls) / cost if cost else 0.0,
            't': mean / (sd / math.sqrt(n)) if sd > 0 else 0.0}


def split_halves(ticks: Dict[str, List[dict]], winners: Dict[str, str]):
    """Resolved slugs split by window start time into earlier / later halves."""
    resolved = sorted((rows[0]['start_ts'], s) for s, rows in ticks.items()
                      if s in winners and rows)
    mid = len(resolved) // 2
    return {s for _, s in resolved[:mid]}, {s for _, s in resolved[mid:]}


# ── output ───────────────────────────────────────────────────────────────────
def _fmt(s: dict) -> str:
    if not s.get('n'):
        return f"{'0':>5}  {'—':>6}  {'—':>6}  {'—':>8}  {'—':>7}  {'—':>6}"
    return (f"{s['n']:>5}  {s['wr'] * 100:>5.1f}%  {s['avg_ask']:>6.3f}  "
            f"{s['pnl']:>+8.2f}  {s['roi'] * 100:>+6.1f}%  {s['t']:>+6.2f}")


def report(path: str, fee_rate: float, min_size: float) -> str:
    ticks, winners = load(path)
    out = []
    p = out.append
    resolved = [s for s in ticks if s in winners]
    n_ticks = sum(len(ticks[s]) for s in resolved)
    p(f"FAIR VALUE REPORT  ({path})")
    p(f"Markets logged: {len(ticks)}   resolved: {len(resolved)}   ticks on resolved: {n_ticks}")
    p(f"Fee rate: {fee_rate:.3f}   min ask size: {min_size:g} shares")
    if len(resolved) < 200:
        p(f"\n!! Only {len(resolved)} resolved markets. Numbers below are noise until ~200+;"
          " keep the logger running.")
    if not resolved:
        return '\n'.join(out)

    p("\n1. CALIBRATION (model)            want: actual ≈ predicted")
    p(f"   {'bucket':>11}  {'ticks':>7}  {'mkts':>5}  {'pred':>6}  {'actual':>6}")
    for r in calibration(ticks, winners):
        p(f"   {r['lo']:.1f} – {r['hi']:.1f}  {r['n']:>7}  {r['markets']:>5}  "
          f"{r['pred'] * 100:>5.1f}%  {r['actual'] * 100:>5.1f}%")

    bm, bk, bn = brier(ticks, winners)
    p("\n2. MODEL vs MARKET (Brier score, lower = more accurate)")
    p(f"   model {bm:.4f}   market mid {bk:.4f}   on {bn} ticks")
    p("   " + ("Model is MORE accurate than the market." if bm < bk else
               "Market is at least as accurate as the model: no edge from the model alone."))

    hdr = f"   {'':>8}  {'N':>5}  {'WR':>6}  {'ask':>6}  {'PnL/sh':>8}  {'ROI':>7}  {'t':>6}"
    p("\n3. TRADING: first entry per market where fair - ask >= edge (one trade per market)")
    for label, lo, hi in TIME_BUCKETS:
        p(f"\n   time left: {label}")
        p(hdr)
        for me in EDGE_GRID:
            s = summarize(simulate(ticks, winners, me, fee_rate, lo, hi, min_size))
            p(f"   edge>={me:.2f}  {_fmt(s)}")

    first, second = split_halves(ticks, winners)
    p("\n   Stability check (time left: all): earlier half vs later half")
    p(f"   {'':>8}  {'earlier half':^48}  |  {'later half':^48}")
    for me in EDGE_GRID:
        a = summarize(simulate(ticks, winners, me, fee_rate, min_size=min_size, slugs=first))
        b = summarize(simulate(ticks, winners, me, fee_rate, min_size=min_size, slugs=second))
        p(f"   edge>={me:.2f}  {_fmt(a)}  |  {_fmt(b)}")

    p("\n   By symbol / timeframe (edge>=0.05, time left: all)")
    p(hdr)
    groups = defaultdict(list)
    for t in simulate(ticks, winners, 0.05, fee_rate, min_size=min_size):
        groups[f"{t['symbol']} {t['tf']}"].append(t)
    for k in sorted(groups):
        p(f"   {k:>8}  {_fmt(summarize(groups[k]))}")

    p("\nHow to read this: a setting is worth paper-trading only if it is positive in BOTH"
      "\nhalves, has t >= 2, and has 200+ trades. Many settings are shown, so expect one or"
      "\ntwo to look good by chance. Fills are at the logged ask; real fills will be a bit worse.")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--log', default='fair_value_log.jsonl')
    ap.add_argument('--fee-rate', type=float, default=0.0,
                    help='cost = ask * (1 + fee_rate); matches the bot convention')
    ap.add_argument('--min-size', type=float, default=5.0,
                    help='ignore asks with fewer shares than this')
    args = ap.parse_args()
    print(report(args.log, args.fee_rate, args.min_size))


if __name__ == '__main__':
    main()
