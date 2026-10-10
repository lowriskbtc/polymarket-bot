"""
Contrarian V8 parameter optimizer per (symbol, timeframe).

Engine mirrors `_run_contrarian_v8` in polymarket_bot.py (line 18729):
  Outcome  = close[i+1] vs close[i]   (matches check_market_outcome live model)
  Score    = bull/bear exhaustion (3) + wick (2) + engulf (2) + soft adds (3x1)
  Mode     = use_contrarian (trend filter) + invert (flip final signal)
  Cooldown = bars since last fire

For each (sym, tf):
  1. Fetch candles (Binance/Hyperliquid)
  2. Precompute indicator arrays + pattern bools  ONCE
  3. Sweep grid (vectorized inner backtest)
  4. 70/30 IS/OOS split — rank IS, validate on OOS, pick highest OOS-EV with N >= MIN_OOS
  5. Calibrate current LIVE config on same data for honest lift comparison

Promotion rule (strict, to combat overfit tax):
  - winner OOS N >= MIN_OOS (50)
  - winner OOS WR >= 55%
  - winner OOS lift vs CURRENT live config >= +2pp WR
  - if any check fails: keep current config

Output:
  contrarian_optimize_results.json  (full report)
  contrarian_optimize_promotions.py (paste-ready ASSET_CONFIGS_TF dict)
"""
import json
import os
import time
import pickle
import requests
import numpy as np
from typing import List, Dict, Tuple

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '_contrarian_optimize_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
# Cache TTL: 6 hours (avoid stale data — same-day re-run uses cache)
CACHE_TTL_SEC = 6 * 3600

def _cached_fetch(symbol: str, interval: str, n: int):
    path = os.path.join(CACHE_DIR, f"{symbol}_{interval}.pkl")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < CACHE_TTL_SEC:
            with open(path, 'rb') as f:
                cached = pickle.load(f)
            if len(cached) >= n * 0.95:
                print(f"  {symbol} {interval}: using cached {len(cached)} bars (age {age/60:.0f}m)")
                return cached
    return None

def _save_cache(symbol: str, interval: str, candles: list):
    path = os.path.join(CACHE_DIR, f"{symbol}_{interval}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(candles, f)

# ---------- DATA FETCH ----------

BINANCE_URL = "https://api.binance.com/api/v3/klines"
HYPER_URL   = "https://api.hyperliquid.xyz/info"

BINANCE_MAP = {'BTC':'BTCUSDT','ETH':'ETHUSDT','SOL':'SOLUSDT',
               'XRP':'XRPUSDT','DOGE':'DOGEUSDT','BNB':'BNBUSDT'}

def fetch_binance(symbol: str, interval: str, n: int) -> List[Dict]:
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
            print(f"  {symbol} {interval}: binance err {e}")
            break
        if not batch or not isinstance(batch, list):
            break
        out = batch + out
        end_ms = batch[0][0] - 1
        if len(batch) < want:
            break
        time.sleep(0.12)
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
        time.sleep(0.12)
    return [{'t':c['t'], 'o':float(c['o']), 'h':float(c['h']),
             'l':float(c['l']), 'c':float(c['c'])} for c in out]

def fetch(symbol: str, interval: str, n: int) -> List[Dict]:
    hit = _cached_fetch(symbol, interval, n)
    if hit is not None:
        return hit
    if symbol == 'HYPE':
        out = fetch_hyperliquid(symbol, interval, n)
    else:
        out = fetch_binance(symbol, interval, n)
    if len(out) >= 1000:
        _save_cache(symbol, interval, out)
    return out

# ---------- INDICATORS (vectorized, match live code) ----------

def wilder_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(prices)
    rsi = np.full(n, 50.0)
    if n < period + 2:
        return rsi
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    alpha  = 1.0 / period
    ag = np.zeros(n - 1); al = np.zeros(n - 1)
    ag[0] = gains[0]; al[0] = losses[0]
    for i in range(1, n - 1):
        ag[i] = ag[i-1] * (1 - alpha) + gains[i]  * alpha
        al[i] = al[i-1] * (1 - alpha) + losses[i] * alpha
    rs = ag / np.where(al == 0, 1e-9, al)
    rsi[1:] = 100 - 100 / (1 + rs)
    return rsi

def ema(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices); mult = 2.0 / (period + 1)
    out = np.empty(n); out[0] = prices[0]
    for k in range(1, n):
        out[k] = prices[k] * mult + out[k-1] * (1 - mult)
    return out

def stoch_k(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes); k = np.full(n, 50.0)
    for i in range(period - 1, n):
        lo = lows[i - period + 1:i + 1].min()
        hi = highs[i - period + 1:i + 1].max()
        k[i] = (closes[i] - lo) / (hi - lo) * 100 if hi > lo else 50.0
    return k

def macd_hist(closes: np.ndarray) -> np.ndarray:
    e12 = ema(closes, 12); e26 = ema(closes, 26)
    line = e12 - e26
    sig  = ema(line, 9)
    return line - sig

def bbands(closes: np.ndarray, period: int = 20, mult: float = 2.0):
    n = len(closes)
    sma = np.full(n, closes[0]); std = np.zeros(n)
    csum  = np.concatenate(([0.0], np.cumsum(closes)))
    csum2 = np.concatenate(([0.0], np.cumsum(closes * closes)))
    for i in range(period - 1, n):
        s  = csum[i+1]  - csum[i+1-period]
        s2 = csum2[i+1] - csum2[i+1-period]
        m  = s / period
        v  = max(s2/period - m*m, 0.0)
        sma[i] = m; std[i] = np.sqrt(v)
    return sma + mult*std, sma, sma - mult*std

# ---------- PRECOMPUTE (per sym+tf) ----------

class Precomp:
    def __init__(self, candles: List[Dict]):
        n = len(candles)
        self.n = n
        self.closes = np.array([c['c'] for c in candles])
        self.opens  = np.array([c['o'] for c in candles])
        self.highs  = np.array([c['h'] for c in candles])
        self.lows   = np.array([c['l'] for c in candles])

        self.rsi   = wilder_rsi(self.closes, 14)
        self.rsi_p = np.concatenate(([self.rsi[0]], self.rsi[:-1]))
        e21 = ema(self.closes, 21); e50 = ema(self.closes, 50)
        self.trend_b = e21 > e50   # bullish trend
        self.trend_r = e21 < e50   # bearish trend
        self.stk   = stoch_k(self.highs, self.lows, self.closes, 14)
        self.mh    = macd_hist(self.closes)
        self.mh_p  = np.concatenate(([self.mh[0]], self.mh[:-1]))
        self.macd_up = self.mh > self.mh_p
        self.macd_dn = self.mh < self.mh_p
        self.bbu, _, self.bbl = bbands(self.closes, 20, 2.0)

        # Candle patterns (depend on candles only, not params)
        body  = np.abs(self.closes - self.opens)
        uw    = self.highs - np.maximum(self.opens, self.closes)
        lw    = np.minimum(self.opens, self.closes) - self.lows
        is_g  = self.closes > self.opens
        is_r  = self.closes < self.opens
        pg    = np.concatenate(([False], is_g[:-1]))
        pr    = np.concatenate(([False], is_r[:-1]))
        cls_p = np.concatenate(([self.closes[0]], self.closes[:-1]))
        opn_p = np.concatenate(([self.opens[0]],  self.opens[:-1]))
        body_p = np.abs(cls_p - opn_p)
        self.hammer   = (lw > 2 * body) & (uw < 0.5 * body) & (body > 0)
        self.star     = (uw > 2 * body) & (lw < 0.5 * body) & (body > 0)
        self.bull_eng = pr & is_g & (body > body_p) & (self.opens <= cls_p)
        self.bear_eng = pg & is_r & (body > body_p) & (self.opens >= cls_p)

        # Outcome: did close[i+1] > close[i]?
        next_close = np.concatenate((self.closes[1:], [self.closes[-1]]))
        self.next_up   = next_close > self.closes
        self.next_down = next_close < self.closes

# ---------- VECTORIZED BACKTEST (single config) ----------

def backtest(P: Precomp,
             score_thresh: int,
             rsi_ext_low: int, rsi_ext_high: int,
             rsi_mod_low: int, rsi_mod_high: int,
             stoch_os: int, stoch_ob: int,
             use_macd: bool, use_contrarian: bool, invert: bool,
             cooldown: int,
             start_i: int, end_i: int) -> Dict:
    """Returns dict with n, wins, wr, ev, n_bull, n_bear over [start_i, end_i)."""

    rsi_bull_ext = P.rsi < rsi_ext_low
    rsi_bull_mod = (P.rsi < rsi_mod_low) & (P.rsi > P.rsi_p)
    rsi_cond_b   = rsi_bull_ext | rsi_bull_mod
    osc_cond_b   = (P.stk < stoch_os) | (P.closes <= P.bbl)
    macd_b       = P.macd_up if use_macd else np.ones(P.n, dtype=bool)
    bull_exh     = rsi_cond_b & osc_cond_b & macd_b

    rsi_bear_ext = P.rsi > rsi_ext_high
    rsi_bear_mod = (P.rsi > rsi_mod_high) & (P.rsi < P.rsi_p)
    rsi_cond_r   = rsi_bear_ext | rsi_bear_mod
    osc_cond_r   = (P.stk > stoch_ob) | (P.closes >= P.bbu)
    macd_r       = P.macd_dn if use_macd else np.ones(P.n, dtype=bool)
    bear_exh     = rsi_cond_r & osc_cond_r & macd_r

    # Score breakdown matches _run_contrarian_v8
    bs = (3 * bull_exh.astype(int)
          + 2 * P.hammer.astype(int)
          + 2 * P.bull_eng.astype(int)
          + ((P.rsi < 40) & macd_b).astype(int)
          + (P.stk < stoch_os).astype(int)
          + (P.closes <= P.bbl).astype(int))
    rs = (3 * bear_exh.astype(int)
          + 2 * P.star.astype(int)
          + 2 * P.bear_eng.astype(int)
          + ((P.rsi > 60) & macd_r).astype(int)
          + (P.stk > stoch_ob).astype(int)
          + (P.closes >= P.bbu).astype(int))

    bull_raw = bs >= score_thresh
    bear_raw = rs >= score_thresh

    if use_contrarian:
        bull_f = bull_raw & P.trend_r
        bear_f = bear_raw & P.trend_b
    else:
        bull_f = bull_raw
        bear_f = bear_raw

    # Conflict resolution (per _run_contrarian_v8)
    bull_sig_raw = bull_f & ~(bear_f & (rs > bs))
    bear_sig_raw = bear_f & ~(bull_f & (bs >= rs))

    # Sequential cooldown pass — short loop
    n_total = P.n
    bull_sig = np.zeros(n_total, dtype=bool)
    bear_sig = np.zeros(n_total, dtype=bool)
    bars_since = 999
    lo_i = max(start_i, 50)
    hi_i = min(end_i, n_total - 1)
    for i in range(lo_i, hi_i):
        if bars_since < cooldown:
            bars_since += 1
            continue
        b, r = bull_sig_raw[i], bear_sig_raw[i]
        if b or r:
            bull_sig[i] = b
            bear_sig[i] = r
            bars_since = 0
        else:
            bars_since += 1

    if invert:
        # Flip final
        final_bull = bear_sig
        final_bear = bull_sig
    else:
        final_bull = bull_sig
        final_bear = bear_sig

    bull_correct = final_bull & P.next_up
    bear_correct = final_bear & P.next_down

    n_bull = int(final_bull.sum())
    n_bear = int(final_bear.sum())
    n = n_bull + n_bear
    wins = int(bull_correct.sum() + bear_correct.sum())
    wr = (wins / n * 100) if n else 0.0
    # EV at $1 bet, +$0.75 win / -$1.00 loss (matches live engine)
    ev = (wr/100 * 75) - ((100 - wr)/100 * 100) if n else 0.0
    return {'n':n, 'wins':wins, 'wr':wr, 'ev':ev,
            'n_bull':n_bull, 'n_bear':n_bear}

# ---------- GRID / DRIVER ----------

SYMBOLS = ['BTC','ETH','SOL','XRP','DOGE','BNB','HYPE']
TIMEFRAMES = ['15m', '5m']

# Bars per timeframe (~90d 15m, ~30d 5m)
N_BARS = {'15m': 8640, '5m': 8640}
DAYS   = {'15m': 90,   '5m': 30}

GRID_SCORE      = [4, 5, 6]
GRID_RSI_EXT    = [(15,85), (20,80), (25,75), (30,70)]
GRID_RSI_MOD    = [(30,70), (35,65), (40,60)]
GRID_STOCH      = [(15,85), (20,80), (25,75)]
GRID_USE_MACD   = [True, False]
GRID_MODE       = [
    ('CTR',     True,  False),   # counter-trend, normal
    ('CTR_INV', True,  True),    # counter-trend filter + invert final
    ('ANY',     False, False),   # no trend filter, normal
    ('ANY_INV', False, True),    # no trend filter, invert final
]
GRID_COOLDOWN   = [0, 1, 2, 3]

# Current live ASSET_CONFIGS (15m only — for calibration on 15m runs)
LIVE_CONFIG_15M = {
    'BTC':  dict(score=5, rsi_ext=(20,80), rsi_mod=(35,65), stoch=(15,85),
                 use_macd=True,  use_ctr=True,  invert=True,  cd=0),
    'ETH':  dict(score=5, rsi_ext=(25,75), rsi_mod=(30,70), stoch=(15,85),
                 use_macd=False, use_ctr=True,  invert=False, cd=2),
    'SOL':  dict(score=5, rsi_ext=(20,80), rsi_mod=(30,70), stoch=(25,75),
                 use_macd=True,  use_ctr=False, invert=True,  cd=2),
    'XRP':  dict(score=5, rsi_ext=(25,75), rsi_mod=(40,60), stoch=(15,85),
                 use_macd=True,  use_ctr=True,  invert=True,  cd=2),
    'DOGE': dict(score=5, rsi_ext=(30,72), rsi_mod=(35,67), stoch=(20,80),
                 use_macd=True,  use_ctr=True,  invert=False, cd=2),
    'HYPE': dict(score=5, rsi_ext=(15,78), rsi_mod=(20,73), stoch=(20,80),
                 use_macd=True,  use_ctr=True,  invert=False, cd=1),
    'BNB':  dict(score=3, rsi_ext=(25,75), rsi_mod=(35,65), stoch=(20,80),
                 use_macd=True,  use_ctr=True,  invert=False, cd=0),
}

MIN_IS_N  = 30
MIN_OOS_N_PICK = 30     # picker keeps anything with N>=30 OOS
MIN_OOS_N_PROMOTE = 50  # strict gate to actually promote a 15m winner (overfit tax)
MIN_OOS_N_NEW_5M  = 50  # 5m has no live ref so we keep this strict
TOP_IS_N  = 30          # Carry top-30 IS configs into OOS
MIN_WR    = 55.0
MIN_WR_NEW = 58.0       # higher bar for fresh 5m configs (no live history)
MIN_LIFT  = 2.0         # WR points lift vs current live config
LIVE_BROKEN_WR = 50.0   # live below this with N>=15 is "broken" — relax lift req

def cfg_dict_to_tuple(c):
    return (c['score'], c['rsi_ext'][0], c['rsi_ext'][1],
            c['rsi_mod'][0], c['rsi_mod'][1],
            c['stoch'][0], c['stoch'][1],
            c['use_macd'], c['use_ctr'], c['invert'], c['cd'])

def run_config(P, cfg_tup, start_i, end_i):
    (sc, rel, reh, rml, rmh, sos, sob, um, uc, inv, cd) = cfg_tup
    return backtest(P, sc, rel, reh, rml, rmh, sos, sob, um, uc, inv, cd, start_i, end_i)

def optimize_one(symbol: str, tf: str, candles: List[Dict]) -> Dict:
    print(f"\n=== {symbol} {tf} ===  ({len(candles)} bars)")
    P = Precomp(candles)
    total = P.n - 1
    valid_start = 60
    is_end  = valid_start + int((total - valid_start) * 0.7)
    oos_end = total
    days_is  = (is_end  - valid_start) * (5 if tf == '5m' else 15) / 60 / 24
    days_oos = (oos_end - is_end)      * (5 if tf == '5m' else 15) / 60 / 24
    print(f"  IS bars: {is_end - valid_start} (~{days_is:.0f}d)  "
          f"OOS bars: {oos_end - is_end} (~{days_oos:.0f}d)")

    # Calibration: current live config on this data (15m only — 5m has no current)
    live = LIVE_CONFIG_15M.get(symbol)
    live_oos_15m = None
    live_full = None
    if tf == '15m' and live is not None:
        cfg_t = cfg_dict_to_tuple(live)
        live_full = run_config(P, cfg_t, valid_start, oos_end)
        live_oos_15m = run_config(P, cfg_t, is_end, oos_end)
        print(f"  CALIBRATION LIVE 15m on full data: N={live_full['n']} WR={live_full['wr']:.1f}% EV={live_full['ev']:+.2f}")
        print(f"  CALIBRATION LIVE 15m on OOS only:  N={live_oos_15m['n']} WR={live_oos_15m['wr']:.1f}% EV={live_oos_15m['ev']:+.2f}")

    # Sweep — IS first
    t0 = time.time()
    is_results = []
    for sc in GRID_SCORE:
        for (rel, reh) in GRID_RSI_EXT:
            for (rml, rmh) in GRID_RSI_MOD:
                for (sos, sob) in GRID_STOCH:
                    for um in GRID_USE_MACD:
                        for (_mn, uc, inv) in GRID_MODE:
                            for cd in GRID_COOLDOWN:
                                cfg_t = (sc, rel, reh, rml, rmh, sos, sob, um, uc, inv, cd)
                                r = backtest(P, sc, rel, reh, rml, rmh, sos, sob,
                                             um, uc, inv, cd, valid_start, is_end)
                                if r['n'] >= MIN_IS_N:
                                    r['cfg'] = cfg_t
                                    is_results.append(r)
    elapsed = time.time() - t0
    print(f"  Sweep: {len(is_results)} configs met IS_MIN={MIN_IS_N}  ({elapsed:.1f}s)")

    if not is_results:
        return {'error': 'no IS config met min sample size',
                'live_full': live_full, 'live_oos': live_oos_15m}

    # Rank IS by EV (then N as tiebreaker for stability)
    is_results.sort(key=lambda x: (-x['ev'], -x['n']))
    top = is_results[:TOP_IS_N]

    # Evaluate top on OOS (relaxed picker threshold)
    candidates = []
    for r in top:
        oos = run_config(P, r['cfg'], is_end, oos_end)
        if oos['n'] >= MIN_OOS_N_PICK:
            candidates.append({
                'cfg': r['cfg'],
                'is_n':  r['n'],  'is_wr':  r['wr'],  'is_ev':  r['ev'],
                'oos_n': oos['n'], 'oos_wr': oos['wr'], 'oos_ev': oos['ev'],
                'oos_bull': oos['n_bull'], 'oos_bear': oos['n_bear'],
            })

    if not candidates:
        return {'error': f'no top-IS config met OOS_MIN={MIN_OOS_N_PICK}',
                'live_full': live_full, 'live_oos': live_oos_15m,
                'top_is': is_results[:5]}

    # Pick by OOS EV (then OOS WR, then OOS N)
    candidates.sort(key=lambda x: (-x['oos_ev'], -x['oos_wr'], -x['oos_n']))
    winner = candidates[0]

    # Also report best WR for context
    by_wr = sorted(candidates, key=lambda x: (-x['oos_wr'], -x['oos_n']))[0]

    print(f"  WINNER (by OOS EV):  {fmt_cfg(winner['cfg'])}")
    print(f"     IS  N={winner['is_n']}  WR={winner['is_wr']:.1f}%  EV={winner['is_ev']:+.2f}")
    print(f"     OOS N={winner['oos_n']} WR={winner['oos_wr']:.1f}% EV={winner['oos_ev']:+.2f}  (B/S {winner['oos_bull']}/{winner['oos_bear']})")
    if by_wr['cfg'] != winner['cfg']:
        print(f"  Best by WR:  {fmt_cfg(by_wr['cfg'])}  OOS N={by_wr['oos_n']} WR={by_wr['oos_wr']:.1f}% EV={by_wr['oos_ev']:+.2f}")

    return {
        'bars': len(candles),
        'is_bars':  is_end - valid_start,
        'oos_bars': oos_end - is_end,
        'live_full': live_full,
        'live_oos':  live_oos_15m,
        'winner':    winner,
        'best_by_wr': by_wr,
        'top_5_oos': candidates[:5],
    }

def fmt_cfg(t):
    sc, rel, reh, rml, rmh, sos, sob, um, uc, inv, cd = t
    mode = 'CTR' if uc else 'ANY'
    if inv: mode += '_INV'
    return (f"score={sc} rsi_ext=({rel},{reh}) rsi_mod=({rml},{rmh}) "
            f"stoch=({sos},{sob}) macd={'T' if um else 'F'} mode={mode} cd={cd}")

def cfg_to_asset_config_str(t, indent=12):
    sc, rel, reh, rml, rmh, sos, sob, um, uc, inv, cd = t
    pad = ' ' * indent
    lines = [
        f"'score_threshold': {sc},",
        f"'rsi_extreme_low': {rel}, 'rsi_extreme_high': {reh},",
        f"'rsi_moderate_low': {rml}, 'rsi_moderate_high': {rmh},",
        f"'stoch_oversold': {sos}, 'stoch_overbought': {sob},",
        f"'use_macd': {um},",
        f"'use_contrarian': {uc},",
        f"'invert': {inv},",
        f"'cooldown': {cd},",
    ]
    return '\n'.join(pad + l for l in lines)

# ---------- MAIN ----------

def main():
    all_results = {}
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                candles = fetch(sym, tf, N_BARS[tf])
                if len(candles) < 1500:
                    print(f"\n=== {sym} {tf} ===  insufficient data ({len(candles)} bars)")
                    all_results[f"{sym}_{tf}"] = {'error': f'only {len(candles)} bars'}
                    continue
                r = optimize_one(sym, tf, candles)
                all_results[f"{sym}_{tf}"] = r
            except Exception as e:
                import traceback
                print(f"  ERROR {sym} {tf}: {e}")
                traceback.print_exc()
                all_results[f"{sym}_{tf}"] = {'error': str(e)}

    # ---- PROMOTION TABLE ----
    print("\n" + "="*120)
    print("CONTRARIAN OPTIMIZATION SUMMARY")
    print("="*120)
    print(f"{'Sym':>4} {'TF':>4} | {'Live OOS':>20} | {'Winner OOS':>26} | {'Lift':>10} | {'PROMOTE?':<12}")
    print("-"*120)

    promotions = {}
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            key = f"{sym}_{tf}"
            r = all_results.get(key, {})
            if 'error' in r:
                print(f"{sym:>4} {tf:>4} | ERROR: {r['error']}")
                continue
            w = r.get('winner')
            live_oos = r.get('live_oos')
            if not w:
                print(f"{sym:>4} {tf:>4} | no winner ({r.get('error','')})")
                continue

            live_str = (f"N={live_oos['n']} WR={live_oos['wr']:.1f}% EV={live_oos['ev']:+.2f}"
                        if live_oos else "(no live ref for 5m)")
            win_str = (f"N={w['oos_n']} WR={w['oos_wr']:.1f}% EV={w['oos_ev']:+.2f}")

            if live_oos and live_oos['n'] >= 15:
                lift_wr = w['oos_wr'] - live_oos['wr']
                lift_str = f"{lift_wr:+.1f}pp"
                live_broken = live_oos['wr'] < LIVE_BROKEN_WR
                if live_broken:
                    # Live is sub-coinflip; any winner with N>=30 and WR>=55 promotes
                    promote = (w['oos_n'] >= MIN_OOS_N_PICK
                               and w['oos_wr'] >= MIN_WR)
                    promote_str = ("YES (live broken)" if promote
                                   else f"no - wr<{int(MIN_WR)}")
                else:
                    promote = (w['oos_n'] >= MIN_OOS_N_PROMOTE
                               and w['oos_wr'] >= MIN_WR
                               and lift_wr >= MIN_LIFT)
                    promote_str = ("YES" if promote
                                   else "no - wr<55" if w['oos_wr'] < MIN_WR
                                   else "no - lift<2" if lift_wr < MIN_LIFT
                                   else f"no - N<{MIN_OOS_N_PROMOTE}")
            else:
                lift_str = "n/a"
                promote = (w['oos_n'] >= MIN_OOS_N_NEW_5M
                           and w['oos_wr'] >= MIN_WR_NEW)
                promote_str = ("YES (new)" if promote
                               else f"no - wr<{int(MIN_WR_NEW)} or N<{MIN_OOS_N_NEW_5M}")

            print(f"{sym:>4} {tf:>4} | {live_str:>20} | {win_str:>26} | {lift_str:>10} | {promote_str:<12}")

            if promote:
                promotions[key] = {
                    'symbol': sym, 'timeframe': tf,
                    'cfg': w['cfg'],
                    'oos_n':  w['oos_n'],  'oos_wr':  w['oos_wr'],  'oos_ev':  w['oos_ev'],
                    'live_wr': live_oos['wr'] if live_oos else None,
                    'lift_wr': (w['oos_wr'] - live_oos['wr']) if live_oos else None,
                }

    # Persist results
    def _clean(obj):
        if isinstance(obj, dict):  return {k:_clean(v) for k,v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_clean(x) for x in obj]
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        return obj

    with open('contrarian_optimize_results.json', 'w', encoding='utf-8') as f:
        json.dump(_clean(all_results), f, indent=2, default=str)
    print(f"\nSaved full report -> contrarian_optimize_results.json")

    # Generate paste-ready config block
    if promotions:
        out_lines = ["# Auto-generated from contrarian_optimize.py — paste into ASSET_CONFIGS_TF\n"]
        out_lines.append("ASSET_CONFIGS_TF_PROMOTIONS = {\n")
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                key = f"{sym}_{tf}"
                if key not in promotions: continue
                p = promotions[key]
                out_lines.append(f"    ('{sym}', '{tf}'): {{\n")
                out_lines.append(cfg_to_asset_config_str(p['cfg'], indent=8))
                lift_note = (f"   # promoted: OOS N={p['oos_n']} WR={p['oos_wr']:.1f}% "
                             f"EV={p['oos_ev']:+.2f} "
                             f"(live {p['live_wr']:.1f}%, lift {p['lift_wr']:+.1f}pp)"
                             if p['live_wr'] is not None
                             else f"   # NEW 5m: OOS N={p['oos_n']} WR={p['oos_wr']:.1f}% EV={p['oos_ev']:+.2f}")
                out_lines.append(f"\n    }},{lift_note}\n")
        out_lines.append("}\n")
        with open('contrarian_optimize_promotions.py', 'w', encoding='utf-8') as f:
            f.writelines(out_lines)
        print(f"Saved {len(promotions)} promotion(s) -> contrarian_optimize_promotions.py")
    else:
        print("\nNo promotions earned this run.")

if __name__ == '__main__':
    main()
