#!/usr/bin/env python3
"""
ml_calibration.py — per-sleeve probability calibration for the ML scanners.

WHY: _brier_report.py proves the model has real directional skill (BSS ~6-8%,
out-of-sample, fill-independent) but is OVERCONFIDENT at the extremes (the
0.8-0.9 bin predicts 82% UP when reality is 62%). Sizing by raw ml_prob therefore
bets biggest exactly where the model is most wrong. This module learns a monotone
map ml_prob -> p_cal that fixes the miscalibration WITHOUT adding any new signal.

It is fit OFFLINE on resolved, fill-independent outcomes (the candle resolves
whether or not we filled, so calibration is immune to the paper-vs-real fill
confound). Params -> calibration_params.json; the hot path only ever EVALUATES
calibrate() from cached params (O(1)).

Method per sleeve: isotonic (PAV) for BTC/ETH only (>5k resolved rows, safe for
the step function); Platt/temperature scaling (2 params) elsewhere — robust at
the ~1-2k rows the smaller sleeves have.

NOTHING here mutates trade logs, places orders, or affects the live bot.
Calibration is necessary-but-NOT-sufficient for profit (vig + adverse fills can
still lose real money) — see _calibration_report.py for the fill-honest backtest.
"""
import json, os, math, collections
from datetime import datetime, timezone

SYMS = ['btc', 'eth', 'sol', 'bnb', 'doge', 'xrp', 'hype']
# 2026-06-30: isotonic was MEASURED to hurt BTC/ETH out-of-sample (they're already
# well-calibrated, raw ECE ~0.01-0.02; PAV overfit the train steps). Platt-only.
ISOTONIC_SYMS = set()
PARAMS_FILE = 'calibration_params.json'
EPS = 1e-6


def actual_up(t):
    """Reconstruct the candle's realized up/down from bet direction + outcome.
    Identical to _brier_report — fill-independent (candle resolves regardless)."""
    o, d = t.get('outcome'), t.get('direction')
    if o not in ('WIN', 'LOSS'):
        return None
    if d == 'UP':
        return 1 if o == 'WIN' else 0
    if d == 'DOWN':
        return 0 if o == 'WIN' else 1
    return None


def load_resolved(sym, base_dir=None, cutoff=None, before=None):
    """(ml_prob, actual_up, logged_at) for resolved trades, deduped by id,
    pooled weekday + weekend (the weekend file dups some symbols — bug #2, so
    the id-dedup here is load-bearing). Optional [cutoff, before) time window."""
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    seen, rows = set(), []
    for fn in (f'{sym}_ml_trades.json', f'{sym}_ml_trades.weekend.json'):
        path = os.path.join(base_dir, fn)
        if not os.path.exists(path):
            continue
        try:
            data = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue
        for t in data:
            if not t.get('resolved'):
                continue
            tid = t.get('id')
            if tid in seen:
                continue
            seen.add(tid)
            a, p = actual_up(t), t.get('ml_prob')
            ts = t.get('logged_at')
            if a is None or p is None or not ts:
                continue
            if cutoff and ts < cutoff:
                continue
            if before and ts >= before:
                continue
            rows.append((float(p), int(a), ts))
    return rows


def _logit(p):
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ── Platt / temperature scaling: p_cal = sigmoid(a*logit(p) + b) ──────────────
def fit_platt(rows, iters=600, lr=0.1):
    """Fit (a, b) by log-loss gradient descent on standardized logit(p).
    a<1 shrinks overconfidence toward the base rate."""
    if len(rows) < 30:
        return None
    xs = [_logit(p) for p, _, _ in rows]
    ys = [y for _, y, _ in rows]
    n = len(xs)
    mx = sum(xs) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5 or 1.0
    xn = [(x - mx) / sx for x in xs]
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xn, ys):
            err = _sigmoid(a * x + b) - y
            ga += err * x
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
    # de-standardize back to raw-logit space
    A = a / sx
    B = b - a * mx / sx
    return {'type': 'platt', 'a': round(A, 6), 'b': round(B, 6)}


def apply_platt(p, params):
    return _sigmoid(params['a'] * _logit(p) + params['b'])


# ── Isotonic (Pool-Adjacent-Violators) ────────────────────────────────────────
def fit_isotonic(rows):
    """Non-decreasing step fit of actual_up on ml_prob. Returns (x_right, y) knots."""
    if len(rows) < 200:
        return None
    pts = sorted((p, y) for p, y, _ in rows)
    blocks = []                                  # [sum_y, count, x_right]
    for x, y in pts:
        blocks.append([float(y), 1, x])
        while len(blocks) >= 2 and \
                blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            sy, c, xr = blocks.pop()
            blocks[-1][0] += sy
            blocks[-1][1] += c
            blocks[-1][2] = xr
    return {'type': 'isotonic', 'knots': [(b[2], b[0] / b[1]) for b in blocks]}


def apply_isotonic(p, params):
    for x, y in params['knots']:
        if p <= x:
            return y
    return params['knots'][-1][1]


def calibrate(p, params):
    """Apply calibration params to a raw ml_prob. Identity if params is None."""
    if not params:
        return p
    if params.get('type') == 'platt':
        return apply_platt(p, params)
    if params.get('type') == 'isotonic':
        return apply_isotonic(p, params)
    return p


# ── metrics ───────────────────────────────────────────────────────────────────
def brier(rows, fn):
    return sum((fn(p) - y) ** 2 for p, y, _ in rows) / len(rows) if rows else None


def ece(rows, fn, bins=10):
    """Expected calibration error: bin-weighted mean |confidence - accuracy|."""
    if not rows:
        return None
    buckets = collections.defaultdict(list)
    for p, y, _ in rows:
        c = fn(p)
        buckets[min(bins - 1, int(c * bins))].append((c, y))
    n = len(rows)
    return sum((len(g) / n) * abs(sum(c for c, _ in g) / len(g) -
                                  sum(y for _, y in g) / len(g))
               for g in buckets.values())


def fit_sleeve(sym, base_dir=None, holdout_frac=0.3):
    """Fit on the train window, validate Brier/ECE on a time-held-out tail."""
    rows = load_resolved(sym, base_dir)
    if len(rows) < 200:
        return {'sym': sym, 'n': len(rows), 'params': None, 'note': 'insufficient data'}
    rows.sort(key=lambda r: r[2])
    cut = int(len(rows) * (1 - holdout_frac))
    train, test = rows[:cut], rows[cut:]
    method = 'isotonic' if sym in ISOTONIC_SYMS else 'platt'
    params = (fit_isotonic(train) if method == 'isotonic' else fit_platt(train)) \
        or fit_platt(train)
    ident = lambda p: p
    cal = lambda p: calibrate(p, params)
    return {
        'sym': sym, 'n': len(rows), 'n_train': len(train), 'n_test': len(test),
        'method': params.get('type') if params else None, 'params': params,
        'brier_raw': round(brier(test, ident), 5), 'brier_cal': round(brier(test, cal), 5),
        'ece_raw': round(ece(test, ident), 5), 'ece_cal': round(ece(test, cal), 5),
    }


def fit_all(symbols=None, base_dir=None, write=True):
    """Fit every sleeve; optionally persist params to calibration_params.json
    (fit on the FULL history, for the hot path to consume)."""
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    symbols = symbols or SYMS
    out = {s: fit_sleeve(s, base_dir) for s in symbols}
    if write:
        # persist ONLY validated calibration (improves holdout ECE without worsening
        # Brier). Sleeves where calibration doesn't help get no params -> hot path
        # uses identity (raw ml_prob). Never persist a calibration that hurts.
        payload = {'_fit_at': datetime.now(timezone.utc).isoformat()}
        for s in symbols:
            r = out[s]
            if not r.get('params'):
                continue
            improved = (r['ece_cal'] <= r['ece_raw'] and r['brier_cal'] <= r['brier_raw'] + 1e-4)
            if not improved:
                continue
            rows = load_resolved(s, base_dir)              # refit on FULL history
            p = (fit_isotonic(rows) if s in ISOTONIC_SYMS else fit_platt(rows)) or fit_platt(rows)
            if p:
                payload[s] = p
        try:
            with open(os.path.join(base_dir, PARAMS_FILE), 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f'[warn] could not write {PARAMS_FILE}: {e}')
    return out


def load_params(base_dir=None):
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base_dir, PARAMS_FILE), encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


if __name__ == '__main__':
    res = fit_all()
    print(f"\nCALIBRATION FIT (validated on time-held-out 30% tail)")
    print(f"{'sym':5}{'n':>7}{'method':>10}{'BrierRaw':>10}{'BrierCal':>10}"
          f"{'ECEraw':>9}{'ECEcal':>9}  verdict")
    for s, r in res.items():
        if not r.get('params'):
            print(f"{s.upper():5}{r['n']:>7}  {r.get('note', 'no params')}")
            continue
        improved = (r['ece_cal'] <= r['ece_raw'] and r['brier_cal'] <= r['brier_raw'] + 1e-4)
        print(f"{s.upper():5}{r['n']:>7}{r['method']:>10}{r['brier_raw']:>10.4f}"
              f"{r['brier_cal']:>10.4f}{r['ece_raw']:>9.4f}{r['ece_cal']:>9.4f}  "
              f"{'OK - calibration helps' if improved else 'no improvement'}")
    print(f"\n[written] {mc_path if (mc_path := os.path.join(os.path.dirname(os.path.abspath(__file__)), PARAMS_FILE)) else ''}")
