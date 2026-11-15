"""
Fair value for Polymarket crypto Up/Down markets (pure math, no network, stdlib only).

A "<sym>-updown-<tf>-<ts>" market resolves Up when the price at the end of the
window is >= the price at the start. Once the window is running, the chance of
Up is mostly a function of three things:

    move      = ln(spot / start_price)      how far price already is from the line
    time_left = seconds until the window closes
    sigma     = realized volatility per sqrt(second), from recent 1m candles

Under a driftless random walk:

    P(Up) = Phi( move / (sigma * sqrt(time_left)) )

Using the log ratio means a constant price difference between the exchange we
read (Binance) and the settlement oracle (Chainlink) mostly cancels out.

This model is a starting point to be checked, not trusted: fair_value_report.py
measures whether its probabilities are calibrated against real resolutions.
"""
import math
from typing import Iterable, Optional, Sequence, Tuple

TF_SECONDS = {'5m': 300, '15m': 900}

# Keep probabilities away from exactly 0/1: the model is not that sure, and a
# 0.999 "fair" vs a 0.97 ask would otherwise look like free money.
P_FLOOR = 0.005
P_CEIL = 0.995


def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def window_start(now: float, tf: str) -> int:
    """Unix timestamp of the window containing `now` (matches the market slug)."""
    interval = TF_SECONDS[tf]
    n = int(now)
    return n - (n % interval)


def market_slug(symbol: str, tf: str, start_ts: int) -> str:
    return f"{symbol.lower()}-updown-{tf}-{start_ts}"


def realized_vol_per_sec(closes: Sequence[float], bar_seconds: int = 60) -> Optional[float]:
    """Stdev of log returns between consecutive closes, scaled to one second.

    Returns None when there is not enough data (fewer than 10 returns) or the
    series is flat, so callers skip the row instead of logging a bogus number.
    """
    pts = [float(c) for c in closes if c and c > 0]
    if len(pts) < 11:
        return None
    rets = [math.log(b / a) for a, b in zip(pts, pts[1:])]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var <= 0:
        return None
    return math.sqrt(var) / math.sqrt(bar_seconds)


def fair_prob_up(spot: float, start_price: float, secs_left: float,
                 sigma_per_sec: float) -> float:
    """Probability the window closes at or above start_price."""
    if spot <= 0 or start_price <= 0 or sigma_per_sec is None or sigma_per_sec <= 0:
        raise ValueError("spot, start_price and sigma must be positive")
    move = math.log(spot / start_price)
    if secs_left <= 0:
        p = 1.0 if move >= 0 else 0.0
    else:
        p = phi(move / (sigma_per_sec * math.sqrt(secs_left)))
    return min(max(p, P_FLOOR), P_CEIL)


def best_bid_ask(book: dict) -> Tuple[Optional[float], Optional[float],
                                      Optional[float], Optional[float]]:
    """(best_bid, bid_size, best_ask, ask_size) from a CLOB /book response.

    Does not assume the API's sort order: takes max bid and min ask.
    """
    def _levels(side: Iterable) -> list:
        out = []
        for lvl in side or []:
            try:
                out.append((float(lvl['price']), float(lvl['size'])))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    bids = _levels(book.get('bids'))
    asks = _levels(book.get('asks'))
    bb = max(bids, key=lambda x: x[0]) if bids else (None, None)
    ba = min(asks, key=lambda x: x[0]) if asks else (None, None)
    return bb[0], bb[1], ba[0], ba[1]


def edge(fair_p: float, ask: Optional[float], fee_rate: float = 0.0) -> Optional[float]:
    """Expected profit per share of buying at `ask` if the true probability is fair_p.

    Cost convention matches the bot: ask * (1 + fee_rate).
    """
    if ask is None or not (0 < ask < 1):
        return None
    return fair_p - ask * (1.0 + fee_rate)
