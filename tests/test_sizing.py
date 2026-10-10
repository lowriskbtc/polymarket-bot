import json
from datetime import datetime, timedelta, timezone

import pytest

import sizing_agent
import sizing_calibration
from sizing_agent import SizingAgent

STRAT = {
    "enabled": True, "kelly_fraction": 0.5, "min_stake": 1.0, "max_stake": 50.0,
    "conviction_multipliers": {"high": 1.0, "low": 0.5},
    "per_asset_caps": {"ETH": 5.0}, "daily_loss_kill_usd": 100.0,
}


@pytest.fixture()
def agent(tmp_path):
    cfg = tmp_path / "sizing_config.json"
    cfg.write_text(json.dumps({"_README": "ignored", "giga": STRAT, "off": {**STRAT, "enabled": False}}))
    return SizingAgent(str(cfg))


def size(agent, **kw):
    args = dict(strategy="giga", conviction="high", p_hat=0.6, b=1.0, asset="BTC",
                current_bankroll=1000.0, daily_pnl=0.0)
    args.update(kw)
    return agent.size(**args)


def test_half_kelly_stake(agent):
    # full Kelly at p=0.6, b=1 is 0.2; half Kelly 0.1 of 1000 = 100, capped at max_stake 50
    assert size(agent) == 50.0
    assert size(agent, current_bankroll=200.0) == 20.0


def test_conviction_multiplier(agent):
    assert size(agent, conviction="low", current_bankroll=200.0) == 10.0
    assert size(agent, conviction="unknown", current_bankroll=200.0) == 20.0   # fail-soft 1.0


def test_per_asset_cap_is_case_insensitive(agent):
    assert size(agent, asset="eth") == 5.0


@pytest.mark.parametrize("kw", [
    {"strategy": "off"}, {"strategy": "nope"}, {"p_hat": 1.0}, {"p_hat": 0.0},
    {"b": 0}, {"current_bankroll": 0}, {"p_hat": "x"},
])
def test_disabled_or_bad_inputs_fall_back_to_legacy(agent, kw):
    assert size(agent, **kw) is None


def test_kill_switch_returns_zero(agent):
    assert size(agent, daily_pnl=-100.0) == 0.0


def test_kill_switch_also_trips_on_a_big_winning_day(agent):
    """Characterizes current behavior: the check uses abs(daily_pnl), so +$100 also stops
    trading even though the setting is named daily_loss_kill_usd. Flagged for the owner."""
    assert size(agent, daily_pnl=+100.0) == 0.0


def test_negative_edge_still_floors_to_min_stake(agent):
    """Characterizes current behavior: Kelly <= 0 is floored up to min_stake. Flagged for the owner."""
    assert size(agent, p_hat=0.3) == 1.0


def test_missing_or_malformed_config_is_fail_safe(tmp_path):
    assert SizingAgent(str(tmp_path / "missing.json")).size("giga", "high", 0.6, 1, "BTC", 1000, 0) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert SizingAgent(str(bad)).size("giga", "high", 0.6, 1, "BTC", 1000, 0) is None


# ---------------- calibration-driven halving ----------------

def _record(path, n, p_hat, wins, hours_ago=1, **bucket):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with open(path, "a", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"ts": ts, "strategy": bucket.get("strategy", "giga"),
                                "conviction": bucket.get("conviction", "high"),
                                "asset": bucket.get("asset", "BTC"),
                                "p_hat": p_hat, "won": i < wins}) + "\n")


@pytest.fixture()
def tracker(tmp_path):
    return sizing_calibration.CalibrationTracker(str(tmp_path / "calibration.jsonl"))


def test_multiplier_is_one_without_enough_data(tracker):
    _record(tracker._path, 10, 0.8, 2)
    assert tracker.suggested_multiplier("giga", "high", "BTC") == 1.0


def test_multiplier_shrinks_when_overconfident(tracker):
    _record(tracker._path, 40, 0.8, 20)        # predicted 0.8, realized 0.5 -> error 0.3
    assert tracker.suggested_multiplier("giga", "high", "BTC") == pytest.approx(0.7)


def test_multiplier_floor_is_half(tracker):
    _record(tracker._path, 40, 0.95, 4)        # error 0.85 -> floored at 0.5
    assert tracker.suggested_multiplier("giga", "high", "btc") == 0.5


def test_multiplier_ignores_old_and_other_buckets(tracker):
    _record(tracker._path, 40, 0.9, 0, hours_ago=100)            # outside the 72h window
    _record(tracker._path, 40, 0.9, 0, asset="ETH")              # different bucket
    assert tracker.suggested_multiplier("giga", "high", "BTC") == 1.0


def test_agent_applies_halving_only_when_opted_in(tmp_path, monkeypatch, tracker):
    _record(tracker._path, 40, 0.8, 20)
    monkeypatch.setattr(sizing_agent, "_get_calibration_tracker_v2", lambda: tracker)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"giga": {**STRAT, "max_stake": 1000}}))
    plain = SizingAgent(str(cfg)).size("giga", "high", 0.6, 1.0, "BTC", 200.0, 0.0)
    cfg.write_text(json.dumps({"giga": {**STRAT, "max_stake": 1000, "auto_halving_enabled": True}}))
    import os
    os.utime(cfg, (1, 2))                       # force the mtime-gated reload
    halved = SizingAgent(str(cfg)).size("giga", "high", 0.6, 1.0, "BTC", 200.0, 0.0)
    assert plain == 20.0 and halved == pytest.approx(14.0)
