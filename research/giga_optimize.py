"""
FadeGiga (Giga) parameter optimizer — sweeps per-asset configs per (symbol, TF).

Replays the live FadeGigaAnalyzer signal logic on historical candles with a
vectorized numpy implementation. Outcome model matches live system:
predict at close[i] (signal bar), resolve on close[i+1] vs open[i+1]
(open-to-close of the target bar — same as check_market_outcome).

Grid sweep (per cell):
    percentile      [80, 85, 90, 93]
    min_body_pct    5m: [0.05, 0.10, 0.20, 0.30]  15m: [0.30, 0.40, 0.50]
    min_move_pct    5m: [0.05, 0.10, 0.20]        15m: [0.10, 0.20, 0.30]
    rsi_length      [7, 14]
    rsi_overbought  [68, 70, 72, 75]
    rsi_oversold    [25, 28, 30, 32]
    use_wick (off / 0.20 / 0.25 / 0.30)
    use_trend (off / on)
    cooldown_bars   [1, 2, 3]

70/30 IS/OOS split. Pick top IS candidates, re-score on OOS, report the
config with the best OOS WR. Reject any cell where the OOS winner doesn't
beat its current live config's OOS performance.
"""
import requests, time, json
import numpy as np
from typing import List, Dict

# ---------- DATA FETCH (reuse from BB-Extreme optimizer) ----------
from bb_extreme_optimize import fetch, BINANCE_MAP, fetch_binance, fetch_hyperliquid

# ---------- INDICATOR PRECOMPUTE ----------

def wilder_rsi(closes: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothed RSI matching the live calculate_rsi (last value)
    but vectorized across the whole series."""
    rsi = np.full(len(closes), 50.0)
    deltas = np.diff(closes, prepend=closes[0])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    if len(closes) <= n:
        return rsi
    ag = np.mean(gains[1:n+1])
    al = np.mean(losses[1:n+1])
    rsi[n] = (100 - 100/(1 + ag/al)) if al > 0 else 100.0
    for i in range(n+1, len(closes)):
        ag = (ag*(n-1) + gains[i]) / n
        al = (al*(n-1) + losses[i]) / n
        rsi[i] = (100 - 100/(1 + ag/al)) if al > 0 else 100.0
    return rsi

def ema_arr(closes: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(len(closes))
    k = 2.0/(n+1)
    out[0] = closes[0]
    for i in range(1, len(closes)):
        out[i] = closes[i]*k + out[i-1]*(1-k)
    return out

def precompute(candles: List[Dict]):
    opens  = np.array([c['o'] for c in candles])
    highs  = np.array([c['h'] for c in candles])
    lows   = np.array([c['l'] for c in candles])
    closes = np.array([c['c'] for c in candles])

    body_pcts = np.abs(closes - opens) / opens * 100  # body as % of open
    candle_range = highs - lows
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows
    upper_wr = np.where(candle_range > 0, upper_wick / candle_range, 0)
    lower_wr = np.where(candle_range > 0, lower_wick / candle_range, 0)
    is_green = closes > opens
    is_red   = closes < opens

    rsi7  = wilder_rsi(closes, 7)
    rsi14 = wilder_rsi(closes, 14)
    ema9  = ema_arr(closes, 9)
    ema21 = ema_arr(closes, 21)

    # Per-bar rolling percentile of body_pcts over last 500 bars (matches
    # the live FadeGigaAnalyzer.analyze "window = body_pcts[-500:]"
    # convention but applied to ALL bars).
    return dict(opens=opens, highs=highs, lows=lows, closes=closes,
                body_pcts=body_pcts, upper_wr=upper_wr, lower_wr=lower_wr,
                is_green=is_green, is_red=is_red,
                rsi7=rsi7, rsi14=rsi14, ema9=ema9, ema21=ema21)

def rolling_percentile(arr: np.ndarray, pv: float, window: int = 500) -> np.ndarray:
    """Rolling percentile (matches live code's `body_pcts[-500:]` semantics:
    at each index i, percentile over arr[max(0, i-window+1):i+1])."""
    n = len(arr)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = np.percentile(arr[lo:i+1], pv) if i >= 50 else np.inf
    return out

# ---------- BACKTEST (vectorized) ----------

def backtest(pre: Dict, p: Dict, start_i: int, end_i: int,
             thr_cache: Dict) -> Dict:
    """One config:
      p = dict(percentile, min_body_pct, min_move_pct, rsi_length,
               rsi_ob, rsi_os, use_wick, wick_ratio, use_trend, cooldown)
    """
    opens, closes = pre['opens'], pre['closes']
    body_pcts = pre['body_pcts']
    upper_wr, lower_wr = pre['upper_wr'], pre['lower_wr']
    is_green, is_red = pre['is_green'], pre['is_red']
    ema9, ema21 = pre['ema9'], pre['ema21']

    thr = thr_cache[p['percentile']]
    rsi = pre['rsi14'] if p['rsi_length'] == 14 else pre['rsi7']

    is_giga = (body_pcts >= thr) & (body_pcts >= p['min_body_pct']) & (body_pcts >= p['min_move_pct'])
    rsi_ext = (rsi >= p['rsi_ob']) | (rsi <= p['rsi_os'])

    if p['use_wick']:
        wr_bear = upper_wr >= p['wick_ratio']
        wr_bull = lower_wr >= p['wick_ratio']
    else:
        wr_bear = np.ones(len(closes), bool)
        wr_bull = np.ones(len(closes), bool)

    if p['use_trend']:
        # Fade against trend: bear signal wants bullish trend, bull wants bearish
        tc_bear = ema9 > ema21
        tc_bull = ema9 < ema21
    else:
        tc_bear = np.ones(len(closes), bool)
        tc_bull = np.ones(len(closes), bool)

    # Per-bar signal (before cooldown gate)
    fb_raw = is_giga & is_green & rsi_ext & wr_bear & tc_bear  # FADE BEAR (predict DOWN)
    fu_raw = is_giga & is_red   & rsi_ext & wr_bull & tc_bull  # FADE BULL (predict UP)

    cd = p['cooldown']
    wins = losses = 0
    n_bear = n_bull = 0
    bars_since = 999
    end_i = min(end_i, len(closes) - 1)
    for i in range(start_i, end_i):
        bars_since += 1
        if bars_since < cd:
            continue
        if fb_raw[i]:
            bars_since = 0
            n_bear += 1
            # Resolved by open[i+1] vs close[i+1]
            if closes[i+1] < opens[i+1]:
                wins += 1
            else:
                losses += 1
        elif fu_raw[i]:
            bars_since = 0
            n_bull += 1
            if closes[i+1] > opens[i+1]:
                wins += 1
            else:
                losses += 1
    n = wins + losses
    wr = (wins / n * 100) if n else 0.0
    ev = (wr/100 * (1.0 - 0.57)) - ((100-wr)/100 * 0.57)
    return {'n':n, 'wr':wr, 'ev_57':ev*100, 'n_bear':n_bear, 'n_bull':n_bull}

# ---------- DRIVER ----------

SYMBOLS = ['BTC','ETH','SOL','XRP','DOGE','BNB','HYPE']
TIMEFRAMES = ['15m', '5m']

N_BARS = {'15m': 8640, '5m': 25920}  # 90d each

GRID_PERCENTILE = [80, 85, 90, 93]
GRID_RSI_LEN    = [7, 14]
GRID_RSI_OB     = [68, 70, 72, 75]
GRID_RSI_OS     = [25, 28, 30, 32]
GRID_USE_WICK   = [(False, 0.0), (True, 0.20), (True, 0.25), (True, 0.30)]
GRID_USE_TREND  = [False, True]
GRID_COOLDOWN   = [1, 2, 3]

# TF-aware body/move grids
GRID_BODY_MOVE = {
    '15m': [(b, m) for b in (0.3, 0.4, 0.5) for m in (0.1, 0.2, 0.3)],
    '5m':  [(b, m) for b in (0.05, 0.1, 0.2, 0.3) for m in (0.05, 0.1, 0.2)],
}

MIN_IS  = 60
MIN_OOS = 25

# Current live FadeGiga config per (symbol, TF). For 5m where there's a
# dedicated *_5M config, use that; otherwise use the symbol's main config.
LIVE_CONFIGS = {
    ('BTC',  '15m'): {'percentile': 90, 'min_body_pct': 0.5, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':68, 'rsi_os':32, 'use_wick':True,  'wick_ratio':0.20, 'use_trend':False, 'cooldown':2},
    ('ETH',  '15m'): {'percentile': 93, 'min_body_pct': 0.3, 'min_move_pct': 0.2, 'rsi_length':14, 'rsi_ob':70, 'rsi_os':30, 'use_wick':True,  'wick_ratio':0.30, 'use_trend':True,  'cooldown':2},  # ETH_V36
    ('SOL',  '15m'): {'percentile': 90, 'min_body_pct': 0.6, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':72, 'rsi_os':28, 'use_wick':True,  'wick_ratio':0.28, 'use_trend':True,  'cooldown':2},
    ('XRP',  '15m'): {'percentile': 88, 'min_body_pct': 0.6, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':70, 'rsi_os':30, 'use_wick':True,  'wick_ratio':0.25, 'use_trend':True,  'cooldown':2},
    ('DOGE', '15m'): {'percentile': 93, 'min_body_pct': 0.3, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':72, 'rsi_os':30, 'use_wick':False, 'wick_ratio':0.0,  'use_trend':True,  'cooldown':2},
    ('BNB',  '15m'): {'percentile': 90, 'min_body_pct': 0.5, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':68, 'rsi_os':32, 'use_wick':True,  'wick_ratio':0.20, 'use_trend':False, 'cooldown':2},  # BNB uses BTC-like config on 15m
    ('HYPE', '15m'): {'percentile': 85, 'min_body_pct': 0.2, 'min_move_pct': 0.1, 'rsi_length':14, 'rsi_ob':70, 'rsi_os':30, 'use_wick':False, 'wick_ratio':0.0,  'use_trend':True,  'cooldown':2},
    # 5m — symbols without dedicated 5m config fall through to the v35 main config (BTC/ETH/SOL/XRP/DOGE)
    ('BTC',  '5m'):  {'percentile': 90, 'min_body_pct': 0.5, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':68, 'rsi_os':32, 'use_wick':True,  'wick_ratio':0.20, 'use_trend':False, 'cooldown':2},
    ('ETH',  '5m'):  {'percentile': 93, 'min_body_pct': 0.6, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':70, 'rsi_os':30, 'use_wick':True,  'wick_ratio':0.30, 'use_trend':True,  'cooldown':2},
    ('SOL',  '5m'):  {'percentile': 90, 'min_body_pct': 0.6, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':72, 'rsi_os':28, 'use_wick':True,  'wick_ratio':0.28, 'use_trend':True,  'cooldown':2},
    ('XRP',  '5m'):  {'percentile': 88, 'min_body_pct': 0.6, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':70, 'rsi_os':30, 'use_wick':True,  'wick_ratio':0.25, 'use_trend':True,  'cooldown':2},
    ('DOGE', '5m'):  {'percentile': 93, 'min_body_pct': 0.3, 'min_move_pct': 0.3, 'rsi_length':14, 'rsi_ob':72, 'rsi_os':30, 'use_wick':False, 'wick_ratio':0.0,  'use_trend':True,  'cooldown':2},
    ('BNB',  '5m'):  {'percentile': 80, 'min_body_pct': 0.3, 'min_move_pct': 0.05,'rsi_length':14, 'rsi_ob':75, 'rsi_os':30, 'use_wick':False, 'wick_ratio':0.0,  'use_trend':False, 'cooldown':1},  # BNB_5M
    ('HYPE', '5m'):  {'percentile': 80, 'min_body_pct': 0.3, 'min_move_pct': 0.05,'rsi_length':14, 'rsi_ob':68, 'rsi_os':25, 'use_wick':False, 'wick_ratio':0.0,  'use_trend':False, 'cooldown':2},  # HYPE_5M
}

def grid_iter(tf: str):
    for pv in GRID_PERCENTILE:
        for (mb, mm) in GRID_BODY_MOVE[tf]:
            for rl in GRID_RSI_LEN:
                for rob in GRID_RSI_OB:
                    for ros in GRID_RSI_OS:
                        for (uw, wm) in GRID_USE_WICK:
                            for ut in GRID_USE_TREND:
                                for cd in GRID_COOLDOWN:
                                    yield {
                                        'percentile': pv,
                                        'min_body_pct': mb,
                                        'min_move_pct': mm,
                                        'rsi_length': rl,
                                        'rsi_ob': rob,
                                        'rsi_os': ros,
                                        'use_wick': uw,
                                        'wick_ratio': wm,
                                        'use_trend': ut,
                                        'cooldown': cd,
                                    }

def cfg_summary(p: Dict) -> str:
    return (f"pct{p['percentile']} body{p['min_body_pct']} move{p['min_move_pct']} "
            f"RSI{p['rsi_length']}({p['rsi_ob']}/{p['rsi_os']}) "
            f"wick={('Y%.2f' % p['wick_ratio']) if p['use_wick'] else 'N'} "
            f"trend={'Y' if p['use_trend'] else 'N'} cd{p['cooldown']}")

def run_one(symbol: str, tf: str, results: Dict):
    print(f"\n=== {symbol} {tf} ===", flush=True)
    candles = fetch(symbol, tf, N_BARS[tf])
    if len(candles) < 1500:
        print(f"  Only {len(candles)} bars, skipping", flush=True)
        results[(symbol, tf)] = {'error': f'only {len(candles)} bars'}
        return
    days = len(candles) * (5 if tf == '5m' else 15) / 60 / 24
    print(f"  Fetched {len(candles)} bars ({days:.1f} days)", flush=True)

    pre = precompute(candles)
    # Cache rolling percentile per pv — include the live config's percentile
    # so the baseline can be scored even if it's outside the sweep grid.
    live_pct = LIVE_CONFIGS[(symbol, tf)]['percentile']
    pct_set = set(GRID_PERCENTILE) | {live_pct}
    thr_cache = {pv: rolling_percentile(pre['body_pcts'], pv, window=500) for pv in pct_set}
    print(f"  Precomputed indicators + {len(pct_set)} percentile arrays", flush=True)

    # IS/OOS split
    valid_start = 100  # need ~50+ bars for RSI/percentile to stabilize
    total = len(candles) - 1
    is_end = valid_start + int((total - valid_start) * 0.7)
    oos_end = total

    # Baseline = current live config for this cell
    live = LIVE_CONFIGS[(symbol, tf)]
    base_is  = backtest(pre, live, valid_start, is_end, thr_cache)
    base_oos = backtest(pre, live, is_end, oos_end, thr_cache)
    print(f"  LIVE: {cfg_summary(live)}", flush=True)
    print(f"        IS N={base_is['n']} WR={base_is['wr']:.1f}% | "
          f"OOS N={base_oos['n']} WR={base_oos['wr']:.1f}% EV={base_oos['ev_57']:+.2f}", flush=True)

    # Sweep IS
    t0 = time.time()
    is_results = []
    n_combo = 0
    for p in grid_iter(tf):
        n_combo += 1
        r = backtest(pre, p, valid_start, is_end, thr_cache)
        if r['n'] >= MIN_IS:
            r['cfg'] = dict(p)
            is_results.append(r)
    print(f"  Swept {n_combo} combos ({time.time()-t0:.1f}s), {len(is_results)} met MIN_IS={MIN_IS}", flush=True)
    if not is_results:
        results[(symbol, tf)] = {'error': 'no IS config met min sample'}
        return

    # Top 30 by IS WR, then evaluate OOS
    is_results.sort(key=lambda x: (-x['wr'], -x['n']))
    top = is_results[:30]
    candidates = []
    for r in top:
        oos = backtest(pre, r['cfg'], is_end, oos_end, thr_cache)
        if oos['n'] >= MIN_OOS:
            candidates.append({
                'cfg': r['cfg'],
                'is_n': r['n'], 'is_wr': r['wr'],
                'oos_n': oos['n'], 'oos_wr': oos['wr'], 'oos_ev': oos['ev_57'],
                'oos_bull': oos['n_bull'], 'oos_bear': oos['n_bear'],
            })
    if not candidates:
        results[(symbol, tf)] = {'baseline_oos': base_oos, 'winner': None,
                                  'reason': 'no OOS sample large enough among top IS'}
        print(f"  No OOS candidate met MIN_OOS={MIN_OOS}", flush=True)
        return

    candidates.sort(key=lambda x: (-x['oos_wr'], -x['oos_ev']))
    winner = candidates[0]
    lift_wr = winner['oos_wr'] - base_oos['wr']
    lift_ev = winner['oos_ev'] - base_oos['ev_57']

    print(f"  WINNER: {cfg_summary(winner['cfg'])}", flush=True)
    print(f"          IS  N={winner['is_n']} WR={winner['is_wr']:.1f}%", flush=True)
    print(f"          OOS N={winner['oos_n']} WR={winner['oos_wr']:.1f}% EV={winner['oos_ev']:+.2f} "
          f"(B/S {winner['oos_bull']}/{winner['oos_bear']})", flush=True)
    print(f"          LIFT vs live: WR {lift_wr:+.1f}pp  EV {lift_ev:+.2f}pp", flush=True)

    results[(symbol, tf)] = {
        'bars': len(candles),
        'live':         live,
        'baseline_is':  base_is,
        'baseline_oos': base_oos,
        'winner':       winner,
        'lift_wr': lift_wr,
        'lift_ev': lift_ev,
        'beats_live': lift_wr > 0 and winner['oos_ev'] > 0,
    }

def main():
    results = {}
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                run_one(sym, tf, results)
            except Exception as e:
                import traceback
                print(f"  ERROR {sym} {tf}: {e}", flush=True)
                traceback.print_exc()
                results[(sym, tf)] = {'error': str(e)}

    print("\n" + "="*120, flush=True)
    print("GIGA (FadeGiga) OPTIMIZATION — OUT-OF-SAMPLE WINNERS", flush=True)
    print("="*120, flush=True)
    print(f"{'Sym':>4} {'TF':>4} | {'Winner Cfg':<70} | {'OOS Winner':>22} | {'Live OOS':>17} | {'Lift WR':>8}", flush=True)
    print("-"*150, flush=True)
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            r = results.get((sym, tf), {})
            if 'error' in r:
                print(f"{sym:>4} {tf:>4} | ERROR: {r['error']}", flush=True)
                continue
            w = r.get('winner')
            b = r.get('baseline_oos', {})
            if not w:
                print(f"{sym:>4} {tf:>4} | {'no edge / no OOS sample':<70} | {'-':>22} | "
                      f"N={b.get('n','-'):>3} WR={b.get('wr',0):.1f}% | {'-':>8}", flush=True)
                continue
            sumr = cfg_summary(w['cfg'])
            print(f"{sym:>4} {tf:>4} | {sumr:<70} | "
                  f"N={w['oos_n']:>3} WR={w['oos_wr']:>5.1f}% EV={w['oos_ev']:+5.2f} | "
                  f"N={b['n']:>3} WR={b['wr']:>5.1f}% | {r['lift_wr']:+5.1f}pp", flush=True)

    def serialize(obj):
        import numpy as np
        if isinstance(obj, dict):  return {k: serialize(v) for k,v in obj.items()}
        if isinstance(obj, (list, tuple)): return [serialize(x) for x in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return obj
    json_out = {f"{s}_{t}": r for (s,t), r in results.items()}
    with open('giga_optimization_results.json', 'w') as f:
        json.dump(serialize(json_out), f, indent=2)
    print("\nResults saved to giga_optimization_results.json", flush=True)

if __name__ == '__main__':
    main()
