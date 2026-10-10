"""
sizing_agent.py — v2, 2026-06-01.

Per-strategy Kelly sizing with feature flag. Hybrid architecture:
  - Math runs INLINE in the bot hot path (zero added latency, mtime-checked cache).
  - LLM tuning happens OUT-OF-BAND via btc_hermes_nightly.py (Phase 2).

v2 additions (2026-06-01):
  - Calibration-driven auto-halving: when `auto_halving_enabled=true` for a
    strategy, .size() multiplies the final stake by
    CalibrationTracker.suggested_multiplier(strategy, conviction, asset).
    The multiplier is in [0.5, 1.0] and reflects 72h realized vs predicted WR.
    Pure math; no LLM in the hot path.

Public surface:
    from sizing_agent import SizingAgent
    agent = SizingAgent()                           # reads sizing_config.json
    stake = agent.size(strategy, conviction, p_hat, b, asset,
                       current_bankroll, daily_pnl)
    if stake is not None:
        # agent took control
        use stake
    else:
        # agent disabled for this strategy -> legacy path

Safety contract:
  - If `enabled` is False for the strategy (default), .size() returns None.
    The caller falls through to the existing legacy sizing path, unchanged.
  - If config file is missing/malformed/strategy unknown, .size() returns None
    (fail-safe: legacy path still runs).
  - If `auto_halving_enabled` is False (default), behavior matches v1 exactly.
  - Config reload is mtime-gated. The hot path performs ONE os.stat() per call
    (microseconds). No JSON parse unless mtime changed.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional, Dict, Any

# v2 (2026-06-01): calibration-driven auto-halving. Import is best-effort —
# if the calibration module is missing the agent runs exactly like v1.
try:
    from sizing_calibration import get_default_tracker as _get_calibration_tracker_v2
    _CALIB_V2_AVAILABLE = True
except Exception:
    _CALIB_V2_AVAILABLE = False
    def _get_calibration_tracker_v2():  # type: ignore
        return None

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sizing_config.json"
)


class SizingAgent:
    """Per-strategy Kelly sizer. Thread-safe config reload via mtime cache."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self._config_path = config_path
        self._cfg: Dict[str, Any] = {}
        self._mtime: float = -1.0
        self._lock = threading.Lock()
        # Eager initial load; failures are swallowed (agent stays disabled).
        self._maybe_reload()

    # ------------------------------------------------------------------ #
    # config plumbing
    # ------------------------------------------------------------------ #
    def _maybe_reload(self) -> None:
        """One os.stat call. Reparse JSON only when mtime moves."""
        try:
            mtime = os.path.getmtime(self._config_path)
        except OSError:
            # File gone — keep last good config, but mark stale.
            return
        if mtime == self._mtime:
            return
        with self._lock:
            # Re-check under lock (double-checked locking).
            if mtime == self._mtime:
                return
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Filter out the _README key from strategy lookup.
                self._cfg = {k: v for k, v in data.items() if not k.startswith("_")}
                self._mtime = mtime
            except (OSError, json.JSONDecodeError):
                # Malformed — keep last good config silently.
                pass

    def get_strategy_cfg(self, strategy: str) -> Optional[Dict[str, Any]]:
        """Public accessor — used by smoke tests and the nightly tuner."""
        self._maybe_reload()
        return self._cfg.get(strategy)

    def is_enabled(self, strategy: str) -> bool:
        cfg = self.get_strategy_cfg(strategy)
        return bool(cfg and cfg.get("enabled"))

    # ------------------------------------------------------------------ #
    # hot path
    # ------------------------------------------------------------------ #
    def size(
        self,
        strategy: str,
        conviction: str,
        p_hat: float,
        b: float,
        asset: str,
        current_bankroll: float,
        daily_pnl: float,
    ) -> Optional[float]:
        """Returns USD stake, or None if disabled / unknown strategy.

        Order of operations (v2, 2026-06-01):
          1. Kelly: f = (p*b - (1-p)) / b
          2. f *= kelly_fraction
          3. f *= conviction_multipliers[conviction]
          4. stake = f * bankroll
          5. cap at max_stake                              (ceiling only)
          6. cap at per_asset_caps[asset]                  (ceiling only)
          7. stake *= tracker.suggested_multiplier(...)    (NEW v2 — only if
             both `enabled` AND `auto_halving_enabled` are true)
          8. clamp UP to min_stake                         (floor only)
          (daily_pnl kill switch is checked first, at top of method.)
        """
        self._maybe_reload()

        cfg = self._cfg.get(strategy)
        if not cfg or not cfg.get("enabled"):
            return None  # legacy path runs

        # Kill switch first — even when enabled, an exhausted day returns $0
        # so the caller skips the trade entirely.
        try:
            kill = float(cfg.get("daily_loss_kill_usd", 0.0))
            if kill > 0 and abs(float(daily_pnl)) >= kill:
                return 0.0
        except (TypeError, ValueError):
            pass

        # Sanity: bad inputs -> fail safe.
        try:
            p = float(p_hat)
            bb = float(b)
            bankroll = float(current_bankroll)
        except (TypeError, ValueError):
            return None
        if bb <= 0.0 or bankroll <= 0.0 or not (0.0 < p < 1.0):
            return None

        # 1. Kelly
        f = (p * bb - (1.0 - p)) / bb
        if f <= 0.0:
            f = 0.0

        # 2. kelly_fraction
        try:
            kf = float(cfg.get("kelly_fraction", 1.0))
        except (TypeError, ValueError):
            kf = 1.0
        f *= kf

        # 3. conviction multiplier (unknown conviction -> 1.0, fail-soft)
        cm_map = cfg.get("conviction_multipliers", {}) or {}
        try:
            cm = float(cm_map.get(conviction, 1.0))
        except (TypeError, ValueError):
            cm = 1.0
        f *= cm

        # 4. dollar stake
        stake = f * bankroll

        # 5. max_stake ceiling (no floor yet — v2 reorders to apply min last)
        try:
            mn = float(cfg.get("min_stake", 0.0))
            mx = float(cfg.get("max_stake", stake))
        except (TypeError, ValueError):
            mn, mx = 0.0, stake
        if stake > mx:
            stake = mx

        # 6. per-asset cap (ceiling)
        caps = cfg.get("per_asset_caps", {}) or {}
        asset_key = (asset or "").upper()
        if asset_key in caps:
            try:
                ac = float(caps[asset_key])
                if stake > ac:
                    stake = ac
            except (TypeError, ValueError):
                pass

        # 7. v2 auto-halving — only when explicitly opted in per-strategy.
        # Reads the last 72h of calibration records for the exact (strategy,
        # conviction, asset) bucket. Returns 1.0 if n<min_n or error<=threshold.
        # Errors in the calibration math are swallowed (fail-safe -> 1.0).
        if _CALIB_V2_AVAILABLE and cfg.get("auto_halving_enabled"):
            try:
                _tracker = _get_calibration_tracker_v2()
                if _tracker is not None:
                    _mult = float(_tracker.suggested_multiplier(
                        strategy=strategy,
                        conviction=conviction,
                        asset=asset_key,
                    ))
                    if 0.0 < _mult <= 1.0:
                        stake = stake * _mult
            except Exception:
                pass

        # 8. min_stake floor (applied LAST so halving can pull stake down,
        # but never below the strategy's hard floor)
        if stake < mn:
            stake = mn

        # Round to cent.
        return round(stake, 2)


# ---------------------------------------------------------------------- #
# Module-level singleton for cheap import-and-use from the bot.
# ---------------------------------------------------------------------- #
_default_agent: Optional[SizingAgent] = None
_default_lock = threading.Lock()


def get_default_agent() -> SizingAgent:
    global _default_agent
    if _default_agent is None:
        with _default_lock:
            if _default_agent is None:
                _default_agent = SizingAgent()
    return _default_agent
