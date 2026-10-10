"""
sizing_calibration.py — v2, 2026-06-01.

Tracks (strategy, conviction, asset) -> realized WR vs predicted p_hat.
Phase 1 (this module): COLLECTION ONLY. No automatic action on miscalibration.
Phase 2 additions (2026-06-01):
  - CalibrationTracker.suggested_multiplier(strategy, conviction, asset, ...)
    computes a stake multiplier in [0.5, 1.0] from the last 72h of records.
    Pure math; cached per-bucket for 5 minutes. Hot-path safe.
  - Hermes nightly (sizing_nightly.py) reads .summary() and writes proposed
    sizing_config.json deltas to sizing_proposals_journal.md for user review.
    The nightly never mutates sizing_config.json automatically.

JSONL format (one record per outcome, appended atomically):
    {"ts": "2026-06-01T12:34:56.789012", "strategy": "TIME-LOCK",
     "conviction": "HIGH", "asset": "BTC", "p_hat": 0.87, "won": true}
"""
from __future__ import annotations

import json
import os
import threading
import time as _time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple

DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sizing_calibration.jsonl"
)


class CalibrationTracker:
    """Append-only outcome log + windowed bucket summary.

    Append is a single open(a)+write+close per record, lock-guarded so
    concurrent strategies can record from different threads without
    interleaving. summary() reads the whole file (cheap until file grows
    past ~100MB, at which point Phase 2 can add rotation).
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._path = log_path
        self._lock = threading.Lock()
        # v2: per-bucket multiplier cache. key = "strategy|conviction|asset|min_n|thr"
        # value = (computed_at_unix, multiplier). 5-min TTL keeps the math cheap
        # under heavy .size() traffic without ever blocking the hot path.
        self._sm_cache: Dict[str, Tuple[float, float]] = {}
        self._sm_cache_lock = threading.Lock()
        self._sm_cache_ttl_s = 300.0  # 5 minutes

    # ------------------------------------------------------------------ #
    # write
    # ------------------------------------------------------------------ #
    def record(
        self,
        strategy: str,
        conviction: str,
        asset: str,
        p_hat: float,
        won: bool,
    ) -> None:
        """Append a single outcome record. Errors are swallowed (logging is
        opportunistic — must never break the trade-outcome handler)."""
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "strategy": str(strategy or ""),
            "conviction": str(conviction or ""),
            "asset": (asset or "").upper(),
            "p_hat": float(p_hat) if p_hat is not None else None,
            "won": bool(won),
        }
        try:
            line = json.dumps(rec, separators=(",", ":")) + "\n"
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            # Calibration tracking is best-effort; never raise into hot path.
            pass

    # ------------------------------------------------------------------ #
    # read
    # ------------------------------------------------------------------ #
    def summary(self, window_hours: int = 24, min_n: int = 1) -> Dict[str, Dict[str, Any]]:
        """Aggregate the last `window_hours` of records by strategy x conviction.

        v2: bucket key is "strategy|conviction|asset" (asset added) so the
        nightly LLM consumer can reason about per-asset miscalibration.
        Buckets with n < `min_n` are dropped.

        Returns: {"TIME-LOCK|HIGH|BTC": {"n": 12, "predicted_wr": 0.87,
                                          "realized_wr": 0.67, "error": 0.20}}
        `error` = predicted - realized (positive = model is overconfident).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"n": 0, "wins": 0, "sum_p": 0.0}
        )
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts_raw = r.get("ts")
                    if not ts_raw:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_raw)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                    if ts < cutoff:
                        continue
                    asset_key = (r.get("asset") or "").upper()
                    key = f"{r.get('strategy', '')}|{r.get('conviction', '')}|{asset_key}"
                    b = buckets[key]
                    b["n"] += 1
                    if r.get("won"):
                        b["wins"] += 1
                    p = r.get("p_hat")
                    if isinstance(p, (int, float)):
                        b["sum_p"] += float(p)
        except OSError:
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for key, b in buckets.items():
            n = int(b["n"]) or 0
            if n < int(min_n):
                continue
            predicted = b["sum_p"] / n
            realized = b["wins"] / n
            out[key] = {
                "n": n,
                "predicted_wr": round(predicted, 4),
                "realized_wr": round(realized, 4),
                "error": round(predicted - realized, 4),
            }
        return out

    # ------------------------------------------------------------------ #
    # v2: calibration-driven auto-halving (math only; never calls an LLM)
    # ------------------------------------------------------------------ #
    def suggested_multiplier(
        self,
        strategy: str,
        conviction: str,
        asset: str,
        min_n: int = 30,
        error_threshold: float = 0.10,
        window_hours: int = 72,
    ) -> float:
        """Suggest a stake multiplier in [0.5, 1.0] for the given bucket.

        Rules (matches phase 2 spec):
          - Filter the last `window_hours` of records to (strategy, conviction,
            asset). Bucket-exact match — no cross-bucket borrowing.
          - If n < min_n      -> return 1.0 (not enough signal).
          - error = predicted_wr - realized_wr  (positive = overconfident)
          - If error <= error_threshold -> return 1.0 (calibration OK).
          - Otherwise            -> return max(0.5, 1.0 - error).

        The 0.5 floor keeps the bot live even when miscalibration is severe;
        we shrink Kelly but never flatline it.

        Cached for 5 minutes per (strategy, conviction, asset, min_n, thr) so
        repeated .size() calls in a tight scan don't re-walk the JSONL.
        Errors degrade to 1.0 (legacy behavior).
        """
        key = f"{strategy}|{conviction}|{(asset or '').upper()}|{int(min_n)}|{float(error_threshold):.4f}"
        now = _time.time()
        with self._sm_cache_lock:
            hit = self._sm_cache.get(key)
            if hit is not None and (now - hit[0]) < self._sm_cache_ttl_s:
                return hit[1]

        mult = 1.0
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=int(window_hours))
            n = 0
            wins = 0
            sum_p = 0.0
            if os.path.exists(self._path):
                tgt_strategy = str(strategy or "")
                tgt_conv = str(conviction or "")
                tgt_asset = (asset or "").upper()
                with open(self._path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if r.get("strategy") != tgt_strategy:
                            continue
                        if r.get("conviction") != tgt_conv:
                            continue
                        if (r.get("asset") or "").upper() != tgt_asset:
                            continue
                        ts_raw = r.get("ts")
                        if not ts_raw:
                            continue
                        try:
                            ts = datetime.fromisoformat(ts_raw)
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
                        if ts < cutoff:
                            continue
                        n += 1
                        if r.get("won"):
                            wins += 1
                        p = r.get("p_hat")
                        if isinstance(p, (int, float)):
                            sum_p += float(p)
            if n >= int(min_n):
                predicted = sum_p / n
                realized = wins / n
                error = predicted - realized
                if error > float(error_threshold):
                    mult = max(0.5, 1.0 - error)
        except Exception:
            # Calibration math is best-effort; fail safe to 1.0.
            mult = 1.0

        with self._sm_cache_lock:
            self._sm_cache[key] = (now, mult)
        return mult


# ---------------------------------------------------------------------- #
# Module-level singleton.
# ---------------------------------------------------------------------- #
_default_tracker: Optional[CalibrationTracker] = None
_default_lock = threading.Lock()


def get_default_tracker() -> CalibrationTracker:
    global _default_tracker
    if _default_tracker is None:
        with _default_lock:
            if _default_tracker is None:
                _default_tracker = CalibrationTracker()
    return _default_tracker
