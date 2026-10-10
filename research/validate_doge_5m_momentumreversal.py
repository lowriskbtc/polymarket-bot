"""validate_doge_5m_momentumreversal.py
90-day IS/OOS validation for `BTC MomentumReversal 15m Polymarket` applied to DOGE 5m.

Uses the EXACT same engine (`_run_momentum_reversal` at polymarket_bot.py:19176)
and config-parse (`_parse_momentum_reversal` at :19160) so results are directly comparable
to the in-app /api/backtest-max output.

Methodology — per 2026-05-16 strategy refinement (journal #45):
  90 days of candles, 70/30 IS/OOS split
  Deploy: winner OOS N >= 25 AND OOS EV > 0
  Quarantine: both fail
  IS→OOS WR gap < 5pp = stable (not overfit)

This validates the cross-window finding from the 2026-05-19 daily_notes:
  Stale (2026-05-14):  n=32  WR=65.6%  EV=+$14.80
  Fresh (2026-05-19):  n=31  WR=67.7%  EV=+$18.48
Two-window agreement is suggestive but not formal IS/OOS validation —
this script does that formal split.
"""
import json, time, math, sys
import urllib.request, urllib.parse

# ────────────────────────────────────────────────────────────────────────────
# 1. Fetch 90d DOGE 5m candles from Binance
# ────────────────────────────────────────────────────────────────────────────
SYMBOL = 'DOGEUSDT'
INTERVAL = '5m'
DAYS = 90
CANDLES_PER_DAY = 288  # 5m: 24h * 60min / 5min = 288
NUM = DAYS * CANDLES_PER_DAY  # 25,920

def fetch_candles():
    print(f'Fetching {DAYS}d × {INTERVAL} candles for {SYMBOL} (target n={NUM:,})...')
    all_c = []
    end_t = int(time.time() * 1000)
    rem = NUM
    while rem > 0:
        batch = min(rem, 1000)
        params = urllib.parse.urlencode({
            'symbol': SYMBOL, 'interval': INTERVAL,
            'limit': batch, 'endTime': end_t,
        })
        url = f'https://api.binance.com/api/v3/klines?{params}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'validator/1.0'})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.load(r)
        except Exception as e:
            print(f'  fetch err: {e}')
            break
        if not data:
            break
        batch_c = [{
            'timestamp': c[0],
            'open':  float(c[1]),
            'high':  float(c[2]),
            'low':   float(c[3]),
            'close': float(c[4]),
            'volume': float(c[5]),
        } for c in data]
        all_c = batch_c + all_c
        end_t = data[0][0] - 1
        rem -= len(data)
        if len(data) < batch:
            break
        # Don't hammer the API
        time.sleep(0.1)
    print(f'  fetched {len(all_c):,} candles')
    return all_c

# ────────────────────────────────────────────────────────────────────────────
# 2. Indicator helpers (mirror polymarket_bot.py:18813, :18834)
# ────────────────────────────────────────────────────────────────────────────
def wilder_rsi(prices, period=7):
    """Same as _wilder_rsi_arr in the main file."""
    import numpy as np
    n = len(prices)
    rsi = np.full(n, 50.0)
    if n < period + 2:
        return rsi
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    alpha  = 1.0 / period
    ag = np.zeros(n - 1)
    al = np.zeros(n - 1)
    ag[0] = gains[0]; al[0] = losses[0]
    for k in range(1, n - 1):
        ag[k] = alpha * gains[k] + (1 - alpha) * ag[k - 1]
        al[k] = alpha * losses[k] + (1 - alpha) * al[k - 1]
    safe_al = np.where(al > 0, al, np.nan)
    rs = np.where(al > 0, ag / safe_al, np.where(ag > 0, 1e9, 1.0))
    rsi[1:] = 100 - 100 / (1 + rs)
    return rsi

# ────────────────────────────────────────────────────────────────────────────
# 3. MomentumReversal engine — exact mirror of _run_momentum_reversal at :19176
# ────────────────────────────────────────────────────────────────────────────
CFG = {
    'rsi_period':   7,
    'rsi_ob':       60,
    'rsi_os':       40,
    'body_min':     0.4,
    'range_pos_t':  85.0,
}

def run_mr(candles, days_for_freq):
    """Returns dict with n / wins / losses / wr / ev / signals_per_day.
    Mirrors polymarket_bot.py:19176 exactly."""
    import numpy as np
    n = len(candles)
    closes = np.array([c['close'] for c in candles])
    opens  = np.array([c['open']  for c in candles])
    highs  = np.array([c['high']  for c in candles])
    lows   = np.array([c['low']   for c in candles])

    rsi_a = wilder_rsi(closes, CFG['rsi_period'])
    results = []

    for i in range(10, n - 1):
        rsi = float(rsi_a[i])
        cls = float(closes[i]); opn = float(opens[i])
        body_pct = abs(cls - opn) / cls * 100 if cls > 0 else 0

        hi5 = float(highs[max(0, i-4):i+1].max())
        lo5 = float(lows[max(0, i-4):i+1].min())
        rng5 = hi5 - lo5
        range_pos = (cls - lo5) / rng5 * 100 if rng5 > 0 else 50.0

        four_up = (closes[i] > closes[i-1] > closes[i-2] > closes[i-3] > closes[i-4])
        four_dn = (closes[i] < closes[i-1] < closes[i-2] < closes[i-3] < closes[i-4])

        obought = rsi > CFG['rsi_ob']  and body_pct > CFG['body_min'] and range_pos > CFG['range_pos_t']        and four_up
        osold   = rsi < CFG['rsi_os']  and body_pct > CFG['body_min'] and range_pos < (100 - CFG['range_pos_t']) and four_dn

        if osold:    # predict UP
            ok = candles[i+1]['close'] > cls
            results.append({'correct': ok, 'signal': 'UP', 'i': i})
        if obought:  # predict DOWN
            ok = candles[i+1]['close'] < cls
            results.append({'correct': ok, 'signal': 'DOWN', 'i': i})

    tot  = len(results)
    wins = sum(1 for r in results if r['correct'])
    wr   = round(wins / tot * 100, 1) if tot > 0 else 0
    ev   = round(wr / 100 * 75 - (100 - wr) / 100 * 100, 2) if tot > 0 else 0
    return {
        'total': tot, 'wins': wins, 'losses': tot - wins,
        'win_rate': wr, 'ev': ev,
        'signals_per_day': round(tot / max(days_for_freq, 1), 1),
        'results': results,
    }

# ────────────────────────────────────────────────────────────────────────────
# 4. 70/30 IS/OOS split + decision
# ────────────────────────────────────────────────────────────────────────────
def main():
    candles = fetch_candles()
    if len(candles) < 1000:
        print(f'ERROR: only got {len(candles)} candles, need more for valid backtest')
        return 1
    actual_days = (candles[-1]['timestamp'] - candles[0]['timestamp']) / 86400000.0
    print(f'Actual time span: {actual_days:.1f} days')

    # 70/30 split — by INDEX (matches 2026-05-16 methodology)
    cut = int(len(candles) * 0.70)
    is_candles  = candles[:cut]
    oos_candles = candles[cut:]
    is_days  = (is_candles[-1]['timestamp']  - is_candles[0]['timestamp'])  / 86400000.0
    oos_days = (oos_candles[-1]['timestamp'] - oos_candles[0]['timestamp']) / 86400000.0
    print(f'\nIS  window: {len(is_candles):,} candles ≈ {is_days:.1f}d  ({iso_of_ts(is_candles[0]["timestamp"])} → {iso_of_ts(is_candles[-1]["timestamp"])})')
    print(f'OOS window: {len(oos_candles):,} candles ≈ {oos_days:.1f}d  ({iso_of_ts(oos_candles[0]["timestamp"])} → {iso_of_ts(oos_candles[-1]["timestamp"])})')

    print('\n--- IS results ---')
    is_  = run_mr(is_candles,  is_days)
    print(f'  n={is_["total"]}  wins={is_["wins"]}  WR={is_["win_rate"]}%  EV=${is_["ev"]:+.2f}  ({is_["signals_per_day"]}/day)')

    print('--- OOS results ---')
    oos = run_mr(oos_candles, oos_days)
    print(f'  n={oos["total"]}  wins={oos["wins"]}  WR={oos["win_rate"]}%  EV=${oos["ev"]:+.2f}  ({oos["signals_per_day"]}/day)')

    # Stability check
    print('\n--- Overfit / stability ---')
    if is_['total'] >= 5 and oos['total'] >= 5:
        gap = is_['win_rate'] - oos['win_rate']
        stability = 'STABLE (gap < 5pp)' if abs(gap) < 5 else 'WATCH (gap >= 5pp)' if abs(gap) < 10 else 'OVERFIT (gap >= 10pp)'
        print(f'  IS WR={is_["win_rate"]}%  OOS WR={oos["win_rate"]}%  gap={gap:+.1f}pp — {stability}')
    else:
        print('  Insufficient samples for stability check.')

    # Deploy / quarantine decision per journal #45 methodology
    print('\n--- DEPLOY / QUARANTINE DECISION ---')
    print(f'Rule: deploy when winner OOS N>=25 AND OOS EV>0; quarantine when both fail.\n')
    if oos['total'] >= 25 and oos['ev'] > 0:
        print(f'  ✅ DEPLOY: OOS n={oos["total"]} >= 25 AND OOS EV=${oos["ev"]:+.2f} > 0')
        print(f'     Wire BTC MomentumReversal as DOGE 5m algo.')
    elif oos['total'] < 25 and oos['ev'] <= 0:
        print(f'  ❌ QUARANTINE: OOS n={oos["total"]} < 25 AND OOS EV=${oos["ev"]:+.2f} <= 0')
        print(f'     Do not deploy. Stays in paper observation.')
    elif oos['total'] >= 25:
        print(f'  ⚠️ DEFER: OOS n={oos["total"]} >= 25 but OOS EV=${oos["ev"]:+.2f} <= 0')
        print(f'     Enough samples but no edge. Do not deploy.')
    else:
        print(f'  ⏳ COLLECT: OOS n={oos["total"]} < 25 (need more samples)')
        print(f'     EV=${oos["ev"]:+.2f} {">" if oos["ev"]>0 else "<="} 0 — trending {"positive" if oos["ev"]>0 else "negative"}')

    # Compare to recent max-backtest fresh window for sanity
    print('\n--- Cross-check vs single-window max_backtest_results.json ---')
    try:
        with open('max_backtest_results.json', encoding='utf-8') as f:
            d = json.load(f)
        for r in d.get('all', []):
            if 'momentum' in (r.get('algo_name') or '').lower() and 'reversal' in (r.get('algo_name') or '').lower() \
               and r.get('symbol') == 'DOGE' and r.get('tf') == '5m':
                print(f'  Single-window 30d (max_backtest): n={r["total"]} WR={r["win_rate"]}% EV=${r["ev"]:+.2f}')
                break
    except Exception as e:
        print(f'  (could not read max_backtest_results.json: {e})')

    # Save full output JSON for follow-up scripts
    out = {
        'algo': 'BTC MomentumReversal 15m Polymarket',
        'target_sym': 'DOGE', 'target_tf': '5m',
        'config': CFG,
        'fetched_at': time.time(),
        'candles_total': len(candles),
        'actual_days': round(actual_days, 1),
        'is_candles': len(is_candles), 'is_days': round(is_days, 1),
        'oos_candles': len(oos_candles), 'oos_days': round(oos_days, 1),
        'is':  {k: v for k, v in is_.items() if k != 'results'},
        'oos': {k: v for k, v in oos.items() if k != 'results'},
    }
    with open('validate_doge_5m_momentumreversal.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\n(detail written to validate_doge_5m_momentumreversal.json)')
    return 0


def iso_of_ts(ms):
    import datetime as dt
    return dt.datetime.fromtimestamp(ms / 1000).strftime('%Y-%m-%d')


if __name__ == '__main__':
    sys.exit(main())
