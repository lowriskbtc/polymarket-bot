"""
BB-Extreme parameter optimizer — sweeps BB period / std-mult / min-distance / cooldown
per (symbol, timeframe), with 70/30 in-sample / out-of-sample split.

Outcome model matches live system: predict close[i] vs upper/lower BB[i],
resolve on close[i+1] vs open[i+1] (the same "did target bar close above its
open" model used by check_market_outcome in polymarket_bot.py).

Grid: BB period x std x min-dist (fraction of band width) x cooldown.
Reports winner on OOS — refuses to promote winners that don't beat the
BB(20, 2.0, cd=2, no-min-dist) live baseline.
"""
import requests, time, json, sys
import numpy as np
from typing import List, Dict, Optional

# ---------- DATA FETCH ----------

BINANCE_URL = "https://api.binance.com/api/v3/klines"
HYPER_URL   = "https://api.hyperliquid.xyz/info"

BINANCE_MAP = {'BTC':'BTCUSDT','ETH':'ETHUSDT','SOL':'SOLUSDT',
               'XRP':'XRPUSDT','DOGE':'DOGEUSDT','BNB':'BNBUSDT'}

def fetch_binance(symbol: str, interval: str, n: int) -> List[Dict]:
    """Paginate backwards from now for `n` candles. interval in {'5m','15m'}."""
    sym = BINANCE_MAP[symbol]
    bar_ms = {'5m':5*60*1000, '15m':15*60*1000}[interval]
    out, end_ms = [], int(time.time() * 1000)
    while len(out) < n:
        want = min(1000, n - len(out))
        start_ms = end_ms - want * bar_ms
        try:
            r = requests.get(BINANCE_URL,
                             params={'symbol':sym, 'interval':interval,
                                     'startTime':start_ms, 'endTime':end_ms,
                                     'limit':want},
                             timeout=20)
            batch = r.json()
        except Exception as e:
            print(f"  {symbol} {interval}: fetch error {e}")
            break
        if not batch or not isinstance(batch, list):
            break
        # batch is oldest -> newest
        out = batch + out
        end_ms = batch[0][0] - 1
        if len(batch) < want:
            break
        time.sleep(0.15)  # polite
    return [{'t':c[0], 'o':float(c[1]), 'h':float(c[2]),
             'l':float(c[3]), 'c':float(c[4])} for c in out]

def fetch_hyperliquid(symbol: str, interval: str, n: int) -> List[Dict]:
    bar_ms = {'5m':5*60*1000, '15m':15*60*1000}[interval]
    out, end_ms = [], int(time.time() * 1000)
    while len(out) < n:
        want = min(500, n - len(out))
        start_ms = end_ms - want * bar_ms
        try:
            r = requests.post(HYPER_URL,
                              json={"type":"candleSnapshot",
                                    "req":{"coin":symbol, "interval":interval,
                                           "startTime":start_ms, "endTime":end_ms}},
                              timeout=20)
            batch = r.json()
        except Exception as e:
            print(f"  {symbol} {interval}: hyperliquid err {e}")
            break
        if not batch:
            break
        out = batch + out
        end_ms = batch[0]['t'] - 1
        if len(batch) < want:
            break
        time.sleep(0.15)
    return [{'t':c['t'], 'o':float(c['o']), 'h':float(c['h']),
             'l':float(c['l']), 'c':float(c['c'])} for c in out]

def fetch(symbol: str, interval: str, n: int) -> List[Dict]:
    if symbol == 'HYPE':
        return fetch_hyperliquid(symbol, interval, n)
    return fetch_binance(symbol, interval, n)

# ---------- BB COMPUTATION ----------

def rolling_bb(closes: np.ndarray, period: int) -> tuple:
    """Rolling mean & std (population, matching np.std with default ddof=0
    — same as TechnicalIndicators.calculate_bollinger_bands in the live code)."""
    n = len(closes)
    means = np.full(n, np.nan)
    stds  = np.full(n, np.nan)
    # Use cumulative sums for O(n) rolling stats
    csum  = np.concatenate(([0.0], np.cumsum(closes)))
    csum2 = np.concatenate(([0.0], np.cumsum(closes*closes)))
    for i in range(period-1, n):
        s  = csum[i+1]  - csum[i+1-period]
        s2 = csum2[i+1] - csum2[i+1-period]
        m = s / period
        var = max(s2/period - m*m, 0.0)
        means[i] = m
        stds[i]  = np.sqrt(var)
    return means, stds

# ---------- BACKTEST ----------

def backtest(opens: np.ndarray, closes: np.ndarray,
             period: int, std_mult: float, min_dist: float, cooldown: int,
             start_i: int, end_i: int) -> Dict:
    """Run a single config over candles[start_i:end_i].

    Outcome model: predict close[i] vs BB[i]; resolve on close[i+1] vs open[i+1].
    Mirrors live's check_market_outcome.
    """
    means, stds = rolling_bb(closes, period)
    upper = means + std_mult * stds
    lower = means - std_mult * stds
    width = np.maximum(upper - lower, 1e-9)

    wins = losses = 0
    n_bull = n_bear = 0
    bars_since = 999
    for i in range(start_i, min(end_i, len(closes)-1)):
        bars_since += 1
        if np.isnan(upper[i]):
            continue
        if bars_since < cooldown:
            continue
        dist_up   = (closes[i] - upper[i]) / width[i]   # >0 if above upper band
        dist_down = (lower[i] - closes[i]) / width[i]   # >0 if below lower band
        signal = 0
        if dist_up > min_dist:
            signal = -1   # BEARISH: bet next bar closes red
        elif dist_down > min_dist:
            signal = +1   # BULLISH: bet next bar closes green
        if signal == 0:
            continue
        bars_since = 0
        # Outcome on the NEXT bar — open->close
        win = (closes[i+1] > opens[i+1]) if signal == +1 else (closes[i+1] < opens[i+1])
        if signal == +1: n_bull += 1
        else:            n_bear += 1
        if win: wins += 1
        else:   losses += 1

    n = wins + losses
    wr = (wins / n * 100) if n else 0.0
    # EV @ Polymarket 0.57 entry cost, $1 payoff
    ev = (wr/100 * (1.0 - 0.57)) - ((100-wr)/100 * 0.57)
    return {'n':n, 'wins':wins, 'wr':wr, 'ev_57':ev*100,
            'n_bull':n_bull, 'n_bear':n_bear}

# ---------- DRIVER ----------

SYMBOLS = ['BTC','ETH','SOL','XRP','DOGE','BNB','HYPE']
TIMEFRAMES = ['15m', '5m']

# 90 days 15m = 8640 bars ; 30 days 5m = 8640 bars
N_BARS = {'15m': 8640, '5m': 8640}
DAYS   = {'15m': 90,   '5m': 30}

# Parameter grid
GRID_PERIOD   = [10, 14, 20, 25, 30]
GRID_STD      = [1.5, 2.0, 2.5, 3.0]
GRID_MIN_DIST = [0.0, 0.05, 0.10, 0.20, 0.30]
GRID_COOLDOWN = [1, 2, 3]

# Baseline = live default (BB(20,2), cd=2, no min-dist)
BASELINE = (20, 2.0, 0.0, 2)

# Min sample sizes
MIN_IS  = 30
MIN_OOS = 12

def run_one(symbol: str, tf: str, results: Dict):
    print(f"\n=== {symbol} {tf} ===")
    candles = fetch(symbol, tf, N_BARS[tf])
    if len(candles) < 1000:
        print(f"  Insufficient data ({len(candles)} bars), skipping")
        results[(symbol, tf)] = {'error': f'only {len(candles)} bars'}
        return
    print(f"  Fetched {len(candles)} bars ({len(candles)*(5 if tf=='5m' else 15)/60/24:.1f} days)")

    opens  = np.array([c['o'] for c in candles])
    closes = np.array([c['c'] for c in candles])

    # 70/30 IS/OOS split. Start at idx >= max(period)+5 so BB is valid.
    valid_start = max(GRID_PERIOD) + 5
    total = len(closes) - 1
    is_end  = valid_start + int((total - valid_start) * 0.7)
    oos_end = total

    print(f"  IS bars: {is_end - valid_start}  OOS bars: {oos_end - is_end}")

    # Baseline on OOS
    b_period, b_std, b_min, b_cd = BASELINE
    base_oos = backtest(opens, closes, b_period, b_std, b_min, b_cd, is_end, oos_end)
    base_is  = backtest(opens, closes, b_period, b_std, b_min, b_cd, valid_start, is_end)
    print(f"  BASELINE BB({b_period},{b_std},cd{b_cd}): "
          f"IS N={base_is['n']} WR={base_is['wr']:.1f}% | "
          f"OOS N={base_oos['n']} WR={base_oos['wr']:.1f}% EV={base_oos['ev_57']:+.2f}")

    # Sweep on IS, evaluate winners on OOS
    is_results = []
    for p in GRID_PERIOD:
        for s in GRID_STD:
            for md in GRID_MIN_DIST:
                for cd in GRID_COOLDOWN:
                    r = backtest(opens, closes, p, s, md, cd, valid_start, is_end)
                    if r['n'] >= MIN_IS:
                        r['cfg'] = (p, s, md, cd)
                        is_results.append(r)
    if not is_results:
        print(f"  No config met MIN_IS={MIN_IS}, skipping")
        results[(symbol, tf)] = {'error': 'no IS config met min sample size'}
        return

    # Rank by IS WR (then by N as tiebreaker — prefer more samples)
    is_results.sort(key=lambda x: (-x['wr'], -x['n']))

    # Evaluate top IS candidates on OOS, pick the one that wins on OOS
    top_n = min(20, len(is_results))
    candidates = []
    for r in is_results[:top_n]:
        p, s, md, cd = r['cfg']
        oos = backtest(opens, closes, p, s, md, cd, is_end, oos_end)
        if oos['n'] >= MIN_OOS:
            candidates.append({
                'cfg': r['cfg'],
                'is_n': r['n'], 'is_wr': r['wr'],
                'oos_n': oos['n'], 'oos_wr': oos['wr'], 'oos_ev': oos['ev_57'],
                'oos_bull': oos['n_bull'], 'oos_bear': oos['n_bear'],
            })

    if not candidates:
        print(f"  No top-IS config met MIN_OOS={MIN_OOS}, skipping")
        results[(symbol, tf)] = {'baseline_oos': base_oos, 'winner': None,
                                  'reason': 'no OOS sample large enough'}
        return

    # Rank by OOS WR (then OOS EV)
    candidates.sort(key=lambda x: (-x['oos_wr'], -x['oos_ev']))
    winner = candidates[0]

    print(f"  WINNER: BB({winner['cfg'][0]},{winner['cfg'][1]},md={winner['cfg'][2]},cd{winner['cfg'][3]})")
    print(f"          IS  N={winner['is_n']} WR={winner['is_wr']:.1f}%")
    print(f"          OOS N={winner['oos_n']} WR={winner['oos_wr']:.1f}% EV={winner['oos_ev']:+.2f} (B/S {winner['oos_bull']}/{winner['oos_bear']})")
    lift_wr = winner['oos_wr'] - base_oos['wr']
    lift_ev = winner['oos_ev'] - base_oos['ev_57']
    print(f"          LIFT vs baseline: WR {lift_wr:+.1f}pp  EV {lift_ev:+.2f}pp")

    results[(symbol, tf)] = {
        'bars': len(candles),
        'baseline_oos': base_oos,
        'baseline_is':  base_is,
        'winner': winner,
        'lift_wr': lift_wr,
        'lift_ev': lift_ev,
        'beats_baseline': lift_wr > 0,
    }

def main():
    results = {}
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                run_one(sym, tf, results)
            except Exception as e:
                import traceback
                print(f"  ERROR {sym} {tf}: {e}")
                traceback.print_exc()
                results[(sym, tf)] = {'error': str(e)}

    # ---- FINAL TABLE ----
    print("\n" + "="*100)
    print("BB-EXTREME OPTIMIZATION — OUT-OF-SAMPLE WINNERS")
    print("="*100)
    print(f"{'Sym':>4} {'TF':>4} | {'Best Cfg':<24} | {'IS':>14} | {'OOS Winner':>21} | {'Baseline OOS':>17} | {'Lift':>10}")
    print("-"*120)

    table_rows = []
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            r = results.get((sym, tf), {})
            if 'error' in r:
                row = f"{sym:>4} {tf:>4} | ERROR: {r['error']}"
                print(row)
                continue
            w = r.get('winner')
            b = r.get('baseline_oos', {})
            if not w:
                row = (f"{sym:>4} {tf:>4} | {'no edge':<24} | "
                       f"{'-':>14} | {'-':>21} | "
                       f"N={b.get('n','-'):>3} WR={b.get('wr',0):.1f}% | {'-':>10}")
                print(row)
                continue
            cfg = w['cfg']
            cfg_str = f"BB({cfg[0]},{cfg[1]}) md={cfg[2]} cd{cfg[3]}"
            print(f"{sym:>4} {tf:>4} | {cfg_str:<24} | "
                  f"N={w['is_n']:>3} WR={w['is_wr']:>5.1f}% | "
                  f"N={w['oos_n']:>3} WR={w['oos_wr']:>5.1f}% EV={w['oos_ev']:+5.2f} | "
                  f"N={b['n']:>3} WR={b['wr']:>5.1f}% | "
                  f"{r['lift_wr']:+5.1f}pp WR")

    # Save JSON
    json_out = {}
    for (sym, tf), r in results.items():
        key = f"{sym}_{tf}"
        json_out[key] = r
    # Convert tuples to lists for JSON
    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k,v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serialize(x) for x in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return obj
    with open('bb_extreme_optimization_results.json', 'w') as f:
        json.dump(serialize(json_out), f, indent=2)
    print("\nResults saved to bb_extreme_optimization_results.json")

if __name__ == '__main__':
    main()
