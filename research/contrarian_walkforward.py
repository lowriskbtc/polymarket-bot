"""
Walk-forward validator for Contrarian V9 configs promoted 2026-05-19.

Background: contrarian_optimize.py used a single 70/30 IS/OOS split on 90d of
15m data and promoted BTC/ETH/XRP/BNB cfgs whose OOS WR exceeded IS WR — the
inverse of the usual overfit signal. That asymmetry is suspicious: either the
27d OOS window happened to be regime-favorable, or the optimizer picked configs
that benefit from the specific regime at the tail of the data. Either way the
forward-edge estimate is closer to IS WR, not OOS WR.

This script re-tests the LIVE configs (as actually written into ASSET_CONFIGS
in polymarket_bot.py on 2026-05-19) on 3 non-overlapping 30d windows
and reports the WR/EV distribution. If a cfg only fires +EV in the latest
window, it is regime-fitted and should be reverted; if it fires +EV in 2/3
windows it is treated as robust.

VERIFICATION ONLY. Does not modify ASSET_CONFIGS or bot_config.json.

Output:
  contrarian_walkforward_results.json  (per-symbol per-window detail)
  printed verdict table
"""
import json
import time
from datetime import datetime
from typing import Dict

import numpy as np

# Reuse data fetcher, indicator precompute and backtest core from the optimizer
from contrarian_optimize import (
    fetch,
    Precomp,
    backtest,
    N_BARS,
    fmt_cfg,
    cfg_dict_to_tuple,
)


# ---------- LIVE CONFIGS (as written into ASSET_CONFIGS 2026-05-19) ----------
#
# Lifted verbatim from polymarket_bot.py:1131-1220. These are the
# REFRESHED configs we're verifying, NOT the optimizer's raw picks (BTC and
# XRP were softened from the picker's choice per advisor — see comments).

LIVE_CONFIG_15M_REFRESHED = {
    'BTC': dict(
        score=4, rsi_ext=(25, 75), rsi_mod=(40, 60), stoch=(20, 80),
        use_macd=False, use_ctr=False, invert=False, cd=0,
    ),
    'ETH': dict(
        score=5, rsi_ext=(20, 80), rsi_mod=(35, 65), stoch=(15, 85),
        use_macd=False, use_ctr=False, invert=False, cd=3,
    ),
    'XRP': dict(
        score=5, rsi_ext=(25, 75), rsi_mod=(35, 65), stoch=(15, 85),
        use_macd=False, use_ctr=True, invert=False, cd=2,
    ),
    'BNB': dict(
        score=6, rsi_ext=(30, 70), rsi_mod=(35, 65), stoch=(25, 75),
        use_macd=False, use_ctr=True, invert=False, cd=3,
    ),
}

# Reported IS/OOS from the 2026-05-19 promotion run (for sanity check).
REPORTED_PROMOTION_NUMBERS = {
    'BTC': {'is_n': 460, 'is_wr': 55.0, 'oos_n': 176, 'oos_wr': 59.1},
    'ETH': {'is_n': 115, 'is_wr': 62.6, 'oos_n':  49, 'oos_wr': 77.6},
    'XRP': {'is_n': 148, 'is_wr': 55.4, 'oos_n':  62, 'oos_wr': 62.9},
    'BNB': {'is_n': 126, 'is_wr': 61.1, 'oos_n':  56, 'oos_wr': 67.9},
}

N_WINDOWS = 3
MIN_N_PER_WINDOW = 10            # user gate: window must fire >=10 to count
EV_BREAKEVEN_WR = 100.0 / 1.75   # 57.14% at 0.75/1.00 Polymarket payoff

SYMBOLS = ['BTC', 'ETH', 'XRP', 'BNB']


def fmt_window_dates(candles, lo_i, hi_i):
    t0 = candles[lo_i]['t']
    t1 = candles[min(hi_i, len(candles) - 1)]['t']
    return (datetime.utcfromtimestamp(t0 / 1000).strftime('%Y-%m-%d'),
            datetime.utcfromtimestamp(t1 / 1000).strftime('%Y-%m-%d'))


def walkforward_one(symbol: str, cfg_dict: dict, candles: list) -> Dict:
    P = Precomp(candles)
    cfg_tup = cfg_dict_to_tuple(cfg_dict)

    total = P.n
    # Split into N_WINDOWS equal slices; backtest() handles warmup internally
    # (lo_i = max(start_i, 50)) so window 0 effectively loses 50 bars but that
    # is <0.5d on 15m and inconsequential for distribution analysis.
    chunk = total // N_WINDOWS
    windows = []
    for w in range(N_WINDOWS):
        lo = w * chunk
        hi = total if w == N_WINDOWS - 1 else (w + 1) * chunk
        r = backtest(
            P,
            cfg_dict['score'],
            cfg_dict['rsi_ext'][0], cfg_dict['rsi_ext'][1],
            cfg_dict['rsi_mod'][0], cfg_dict['rsi_mod'][1],
            cfg_dict['stoch'][0], cfg_dict['stoch'][1],
            cfg_dict['use_macd'], cfg_dict['use_ctr'], cfg_dict['invert'],
            cfg_dict['cd'],
            lo, hi,
        )
        date_lo, date_hi = fmt_window_dates(candles, max(lo, 50), hi - 1)
        windows.append({
            'window':  w + 1,
            'bars':    [lo, hi],
            'dates':   [date_lo, date_hi],
            'n':       r['n'],
            'n_bull':  r['n_bull'],
            'n_bear':  r['n_bear'],
            'wins':    r['wins'],
            'wr':      r['wr'],
            'ev':      r['ev'],
        })

    # Also full-period sanity number
    full = backtest(
        P,
        cfg_dict['score'],
        cfg_dict['rsi_ext'][0], cfg_dict['rsi_ext'][1],
        cfg_dict['rsi_mod'][0], cfg_dict['rsi_mod'][1],
        cfg_dict['stoch'][0], cfg_dict['stoch'][1],
        cfg_dict['use_macd'], cfg_dict['use_ctr'], cfg_dict['invert'],
        cfg_dict['cd'],
        0, total,
    )

    # Verdict logic
    wr_positive   = [w for w in windows if w['wr'] > 50.0 and w['n'] >= MIN_N_PER_WINDOW]
    ev_positive   = [w for w in windows if w['ev'] > 0   and w['n'] >= MIN_N_PER_WINDOW]
    under_min_n   = [w for w in windows if w['n'] < MIN_N_PER_WINDOW]

    latest_positive_only = (
        len(ev_positive) == 1 and ev_positive[0]['window'] == N_WINDOWS
    )

    if under_min_n:
        verdict = 'INSUFFICIENT_SIGNALS'
        verdict_detail = (
            f"window(s) {[w['window'] for w in under_min_n]} have N<{MIN_N_PER_WINDOW}; "
            f"cfg does not fire enough across regimes to validate"
        )
    elif len(ev_positive) >= 2:
        verdict = 'ROBUST'
        verdict_detail = f"{len(ev_positive)}/3 windows EV-positive at Polymarket payoff"
    elif latest_positive_only:
        verdict = 'REGIME_FITTED'
        verdict_detail = (
            "only the most-recent window is EV-positive — the optimizer's OOS slice "
            "was riding this regime, not finding a durable edge. Revert or pick a more "
            "conservative cfg."
        )
    elif len(ev_positive) == 1:
        verdict = 'MARGINAL'
        verdict_detail = (
            f"only window {ev_positive[0]['window']}/3 EV-positive — edge is concentrated "
            f"in one regime; trade with reduced size or revert"
        )
    elif len(wr_positive) >= 2:
        verdict = 'COINFLIP'
        verdict_detail = (
            f"{len(wr_positive)}/3 windows above 50% WR but none clear the "
            f"{EV_BREAKEVEN_WR:.1f}% Polymarket breakeven — net unprofitable"
        )
    else:
        verdict = 'REVERT'
        verdict_detail = "0/3 windows EV-positive — cfg is not viable"

    return {
        'cfg':            cfg_dict,
        'cfg_summary':    fmt_cfg(cfg_tup),
        'reported':       REPORTED_PROMOTION_NUMBERS.get(symbol),
        'windows':        windows,
        'full_period':    {'n': full['n'], 'wr': full['wr'], 'ev': full['ev']},
        'wr_positive_count': len(wr_positive),
        'ev_positive_count': len(ev_positive),
        'ev_positive_windows': [w['window'] for w in ev_positive],
        'verdict':        verdict,
        'verdict_detail': verdict_detail,
    }


def main():
    print("=" * 100)
    print(f"CONTRARIAN V9 WALK-FORWARD VALIDATION  ({datetime.now():%Y-%m-%d %H:%M})")
    print("=" * 100)
    print(f"Mode: {N_WINDOWS} non-overlapping windows on ~90d 15m candles")
    print(f"Robust gate: EV-positive in 2+ windows AND N>={MIN_N_PER_WINDOW} per window")
    print(f"Polymarket EV breakeven WR: {EV_BREAKEVEN_WR:.2f}%")
    print()

    all_results = {}
    for sym in SYMBOLS:
        cfg = LIVE_CONFIG_15M_REFRESHED[sym]
        print(f"--- {sym} 15m -------------------------------------------------------------")
        print(f"  cfg: {fmt_cfg(cfg_dict_to_tuple(cfg))}")

        candles = fetch(sym, '15m', N_BARS['15m'])
        if len(candles) < 6000:
            print(f"  INSUFFICIENT DATA: only {len(candles)} bars")
            all_results[sym] = {'error': f'only {len(candles)} bars'}
            continue
        print(f"  fetched {len(candles)} bars")

        t0 = time.time()
        r = walkforward_one(sym, cfg, candles)
        elapsed = time.time() - t0

        for w in r['windows']:
            ev_marker = '+' if w['ev'] > 0 else ' '
            wr_marker = '+' if w['wr'] > 50.0 else ' '
            print(f"  W{w['window']} {w['dates'][0]}->{w['dates'][1]}  "
                  f"N={w['n']:>4}  WR={w['wr']:>5.1f}%{wr_marker}  "
                  f"EV={w['ev']:>+6.2f}{ev_marker}  (B/S {w['n_bull']}/{w['n_bear']})")
        f = r['full_period']
        print(f"  FULL              "
              f"                   N={f['n']:>4}  WR={f['wr']:>5.1f}%   EV={f['ev']:>+6.2f}")
        rep = r['reported']
        if rep:
            print(f"  reported promotion: IS N={rep['is_n']} {rep['is_wr']:.1f}% / "
                  f"OOS N={rep['oos_n']} {rep['oos_wr']:.1f}%")
        print(f"  VERDICT: {r['verdict']}  ({r['verdict_detail']})")
        print(f"  ({elapsed:.1f}s)")
        print()

        all_results[sym] = r

    # ---- SUMMARY TABLE ----
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Sym':>4} | {'W1 WR':>7} {'W2 WR':>7} {'W3 WR':>7} | "
          f"{'W1 EV':>7} {'W2 EV':>7} {'W3 EV':>7} | {'EV+':>5} | {'VERDICT':<22}")
    print("-" * 100)
    for sym in SYMBOLS:
        r = all_results.get(sym, {})
        if 'error' in r:
            print(f"{sym:>4} | ERROR: {r['error']}")
            continue
        w1, w2, w3 = r['windows']
        print(f"{sym:>4} | "
              f"{w1['wr']:>6.1f}% {w2['wr']:>6.1f}% {w3['wr']:>6.1f}% | "
              f"{w1['ev']:>+6.2f} {w2['ev']:>+6.2f} {w3['ev']:>+6.2f} | "
              f"{r['ev_positive_count']:>2}/3  | {r['verdict']:<22}")
    print()

    # Persist
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(x) for x in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    payload = {
        'run_at':        datetime.now().isoformat(timespec='seconds'),
        'mode':          f'{N_WINDOWS} non-overlapping windows on 90d 15m',
        'min_n_per_window': MIN_N_PER_WINDOW,
        'ev_breakeven_wr': EV_BREAKEVEN_WR,
        'results':       all_results,
    }
    with open('contrarian_walkforward_results.json', 'w', encoding='utf-8') as fh:
        json.dump(_clean(payload), fh, indent=2, default=str)
    print("Saved -> contrarian_walkforward_results.json")


if __name__ == '__main__':
    main()
