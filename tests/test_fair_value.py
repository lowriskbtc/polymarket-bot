import json
import math
import random

import pytest

from fair_value import (best_bid_ask, edge, fair_prob_up, market_slug, phi,
                        realized_vol_per_sec, window_start)
from fair_value_report import (brier, calibration, load, report, simulate,
                               split_halves, summarize)


# ── fair_value.py ────────────────────────────────────────────────────────────
def test_phi_known_values():
    assert phi(0) == pytest.approx(0.5)
    assert phi(1.96) == pytest.approx(0.975, abs=1e-3)
    assert phi(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_window_and_slug_match_polymarket_format():
    assert window_start(1_763_240_699.9, '5m') == 1_763_240_400
    assert window_start(1_763_240_699.9, '15m') == 1_763_240_400
    assert market_slug('BTC', '15m', 1_763_240_400) == 'btc-updown-15m-1763240400'


def test_fair_prob_at_the_line_is_half():
    assert fair_prob_up(100.0, 100.0, 300, 1e-4) == pytest.approx(0.5)


def test_fair_prob_moves_with_price_time_and_vol():
    s = 1e-4
    up_px, dn_px = 100.0 * math.exp(0.001), 100.0 * math.exp(-0.001)
    above = fair_prob_up(up_px, 100.0, 300, s)
    assert above > 0.5
    assert fair_prob_up(dn_px, 100.0, 300, s) == pytest.approx(1 - above)
    # Less time left -> more certain
    assert fair_prob_up(100.1, 100.0, 30, s) > above
    # More volatility -> less certain
    assert fair_prob_up(100.1, 100.0, 300, 3 * s) < above


def test_fair_prob_is_clamped_and_handles_expiry():
    assert fair_prob_up(110, 100, 1, 1e-5) == pytest.approx(0.995)
    assert fair_prob_up(90, 100, 1, 1e-5) == pytest.approx(0.005)
    assert fair_prob_up(100, 100, 0, 1e-4) == pytest.approx(0.995)   # tie resolves Up
    with pytest.raises(ValueError):
        fair_prob_up(100, 100, 60, 0)


def test_realized_vol_recovers_true_sigma():
    rng = random.Random(1)
    sigma_bar = 0.001
    closes = [100.0]
    for _ in range(5000):
        closes.append(closes[-1] * math.exp(rng.gauss(0, sigma_bar)))
    est = realized_vol_per_sec(closes, 60)
    assert est == pytest.approx(sigma_bar / math.sqrt(60), rel=0.05)


def test_realized_vol_needs_data():
    assert realized_vol_per_sec([100.0] * 5) is None
    assert realized_vol_per_sec([100.0] * 30) is None   # flat


def test_best_bid_ask_ignores_sort_order_and_junk():
    book = {'bids': [{'price': '0.40', 'size': '10'}, {'price': '0.45', 'size': '3'},
                     {'price': 'x', 'size': '1'}],
            'asks': [{'price': '0.60', 'size': '7'}, {'price': '0.52', 'size': '2'}]}
    assert best_bid_ask(book) == (0.45, 3.0, 0.52, 2.0)
    assert best_bid_ask({}) == (None, None, None, None)


def test_edge():
    assert edge(0.7, 0.6) == pytest.approx(0.1)
    assert edge(0.7, 0.6, fee_rate=0.02) == pytest.approx(0.7 - 0.612)
    assert edge(0.7, None) is None
    assert edge(0.7, 1.0) is None


# ── fair_value_report.py ─────────────────────────────────────────────────────
def _tick(slug, ts, secs_left, fair_up, up_ask, down_ask, start_ts=0, size=100.0):
    return {'type': 'tick', 'ts': ts, 'slug': slug, 'symbol': 'BTC', 'tf': '5m',
            'start_ts': start_ts, 'secs_left': secs_left, 'fair_up': fair_up,
            'up_bid': up_ask - 0.02, 'up_ask': up_ask, 'up_ask_size': size,
            'down_bid': down_ask - 0.02, 'down_ask': down_ask, 'down_ask_size': size}


def test_simulate_takes_first_qualifying_entry_only():
    ticks = {'m1': [_tick('m1', 1, 200, 0.55, 0.53, 0.49),     # edge 0.02: skip
                    _tick('m1', 2, 150, 0.70, 0.60, 0.42),     # Up edge 0.10: take
                    _tick('m1', 3, 100, 0.90, 0.60, 0.42)]}    # later: ignored
    trades = simulate(ticks, {'m1': 'Up'}, min_edge=0.05)
    assert len(trades) == 1
    t = trades[0]
    assert (t['side'], t['ask'], t['win']) == ('Up', 0.60, True)
    assert t['pnl'] == pytest.approx(0.40)


def test_simulate_down_side_size_filter_and_unresolved():
    ticks = {'m1': [_tick('m1', 1, 100, 0.20, 0.25, 0.70)],    # Down edge 0.10
             'm2': [_tick('m2', 1, 100, 0.20, 0.25, 0.70, size=1)],
             'm3': [_tick('m3', 1, 100, 0.20, 0.25, 0.70)]}     # no resolution
    trades = simulate(ticks, {'m1': 'Up', 'm2': 'Down'}, min_edge=0.05, min_size=5)
    assert [(t['slug'], t['side'], t['win']) for t in trades] == [('m1', 'Down', False)]
    assert trades[0]['pnl'] == pytest.approx(-0.70)


def test_simulate_time_bucket():
    ticks = {'m1': [_tick('m1', 1, 400, 0.70, 0.60, 0.42),
                    _tick('m1', 2, 60, 0.70, 0.55, 0.47)]}
    trades = simulate(ticks, {'m1': 'Up'}, 0.05, secs_lo=30, secs_hi=120)
    assert trades[0]['ask'] == 0.55


def test_summarize():
    s = summarize([{'pnl': 0.4, 'cost': 0.6, 'ask': 0.6, 'win': True},
                   {'pnl': -0.6, 'cost': 0.6, 'ask': 0.6, 'win': False}])
    assert s['n'] == 2 and s['wr'] == 0.5
    assert s['pnl'] == pytest.approx(-0.2)
    assert s['roi'] == pytest.approx(-0.2 / 1.2)
    assert summarize([]) == {'n': 0}


def test_calibration_brier_and_halves():
    ticks = {f'm{i}': [_tick(f'm{i}', 1, 100, 0.8, 0.79, 0.23, start_ts=i)] for i in range(10)}
    winners = {f'm{i}': ('Up' if i < 8 else 'Down') for i in range(10)}
    cal = calibration(ticks, winners)
    assert len(cal) == 1 and cal[0]['actual'] == pytest.approx(0.8)
    bm, bk, n = brier(ticks, winners)
    assert n == 10 and bm == pytest.approx(0.16)
    first, second = split_halves(ticks, winners)
    assert first == {f'm{i}' for i in range(5)} and second == {f'm{i}' for i in range(5, 10)}


# ── end to end on simulated markets ──────────────────────────────────────────
def _simulate_log(path, n_markets, lag_s, seed):
    """Random-walk markets. The Polymarket quote is the model probability from
    `lag_s` seconds ago plus a 1c half-spread. lag_s=0 -> efficient market."""
    rng = random.Random(seed)
    sigma = 2e-4                       # per sqrt(second)
    dt, length = 5, 300
    with open(path, 'w') as f:
        for m in range(n_markets):
            slug = f'btc-updown-5m-{m * length}'
            start = 100.0
            path_px = [start]
            for _ in range(length // dt):
                path_px.append(path_px[-1] * math.exp(rng.gauss(0, sigma * math.sqrt(dt))))
            lag = lag_s // dt
            for k in range(1, len(path_px) - 1):
                secs_left = length - k * dt
                fair = fair_prob_up(path_px[k], start, secs_left, sigma)
                k0 = max(0, k - lag)
                stale = fair_prob_up(path_px[k0], start, length - k0 * dt, sigma)
                f.write(json.dumps({**_tick(slug, m * length + k * dt, secs_left, fair,
                                            min(stale + 0.01, 0.99), min(1 - stale + 0.01, 0.99),
                                            start_ts=m * length)}) + '\n')
            win = 'Up' if path_px[-1] >= start else 'Down'
            f.write(json.dumps({'type': 'resolution', 'slug': slug, 'winner': win}) + '\n')


def test_end_to_end_finds_edge_only_when_market_lags(tmp_path):
    lagging = tmp_path / 'lag.jsonl'
    efficient = tmp_path / 'eff.jsonl'
    _simulate_log(lagging, 400, lag_s=30, seed=7)
    _simulate_log(efficient, 400, lag_s=0, seed=7)

    t_lag, w_lag = load(str(lagging))
    t_eff, w_eff = load(str(efficient))
    s_lag = summarize(simulate(t_lag, w_lag, 0.05))
    s_eff = summarize(simulate(t_eff, w_eff, 0.05))
    assert s_lag['n'] > 100 and s_lag['roi'] > 0.05 and s_lag['t'] > 2
    assert s_eff['n'] == 0     # efficient quotes never leave a 5c gap

    text = report(str(lagging), 0.0, 5.0)
    assert 'CALIBRATION' in text and 'Stability check' in text
