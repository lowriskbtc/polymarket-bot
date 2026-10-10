import os
import random

import pytest
from cryptography.fernet import Fernet

import decrypt_keys as dk
import ml_calibration as mc


# ---------------- encrypted env file ----------------

def _write_encrypted_env(tmp_path, password, values, plain=None):
    salt = os.urandom(16)
    fernet = Fernet(dk.generate_key_from_password(password, salt))
    env = tmp_path / ".env"
    lines = [f"{k}=ENC:{fernet.encrypt(v.encode()).decode()}" for k, v in values.items()]
    lines += [f"{k}={v}" for k, v in (plain or {}).items()]
    env.write_text("# comment\n" + "\n".join(lines) + "\n")
    (tmp_path / ".salt").write_bytes(salt)
    return env


def test_round_trip_decrypts_and_passes_plain_values(tmp_path):
    env = _write_encrypted_env(tmp_path, "correct horse", {"PRIVATE_KEY": "0xabc123"}, {"MODE": "paper"})
    out = dk.load_encrypted_env(str(env), "correct horse")
    assert out == {"PRIVATE_KEY": "0xabc123", "MODE": "paper"}


def test_wrong_password_raises_without_leaking_the_value(tmp_path):
    env = _write_encrypted_env(tmp_path, "right", {"PRIVATE_KEY": "0xsecret"})
    with pytest.raises(ValueError) as exc:
        dk.load_encrypted_env(str(env), "wrong")
    assert "0xsecret" not in str(exc.value) and "PRIVATE_KEY" in str(exc.value)


def test_missing_salt_is_an_error(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=ENC:xyz\n")
    with pytest.raises(FileNotFoundError):
        dk.load_encrypted_env(str(env), "pw")


def test_unencrypted_file_loads_without_password(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\nB = two \n# C=3\n")
    assert dk.load_env_with_decryption(str(env)) == {"A": "1", "B": "two"}


def test_missing_file_returns_empty(tmp_path):
    assert dk.load_env_with_decryption(str(tmp_path / "nope.env")) == {}


def test_encrypted_file_without_password_does_not_prompt_when_non_interactive(tmp_path, monkeypatch):
    env = _write_encrypted_env(tmp_path, "pw", {"K": "v"})
    monkeypatch.delenv("MASTER_PASSWORD", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert dk.load_env_with_decryption(str(env)) == {}


def test_master_password_env_is_used_and_trimmed(tmp_path, monkeypatch):
    env = _write_encrypted_env(tmp_path, "pw", {"K": "v"})
    monkeypatch.setenv("MASTER_PASSWORD", "pw\n")
    assert dk.load_env_with_decryption(str(env)) == {"K": "v"}


# ---------------- ML probability calibration ----------------

def _overconfident_rows(n=3000, seed=7):
    """The model says p, but outcomes happen with probability 0.5 + (p - 0.5) / 2."""
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        p = rnd.uniform(0.05, 0.95)
        true_p = 0.5 + (p - 0.5) / 2
        rows.append((p, 1 if rnd.random() < true_p else 0, None))
    return rows


def test_logit_sigmoid_round_trip():
    for p in (0.01, 0.3, 0.5, 0.9):
        assert mc._sigmoid(mc._logit(p)) == pytest.approx(p)
    assert mc._sigmoid(-800) == pytest.approx(0.0) and mc._sigmoid(800) == pytest.approx(1.0)


def test_platt_shrinks_overconfidence_and_improves_brier():
    rows = _overconfident_rows()
    params = mc.fit_platt(rows)
    assert params["type"] == "platt" and 0 < params["a"] < 1
    raw = mc.brier(rows, lambda p: p)
    cal = mc.brier(rows, lambda p: mc.calibrate(p, params))
    assert cal < raw
    assert mc.ece(rows, lambda p: mc.calibrate(p, params)) < mc.ece(rows, lambda p: p)


def test_platt_fixes_a_biased_model():
    """A model that always says ~0.75 when the truth is ~0.55 must be pulled down to ~0.55."""
    rnd = random.Random(11)
    rows = []
    for _ in range(3000):
        p = rnd.uniform(0.6, 0.9)
        rows.append((p, 1 if rnd.random() < p - 0.2 else 0, None))
    params = mc.fit_platt(rows)
    observed = sum(y for _, y, _ in rows) / len(rows)
    predicted = sum(mc.calibrate(p, params) for p, _, _ in rows) / len(rows)
    assert predicted == pytest.approx(observed, abs=0.02)


def test_isotonic_is_monotone_and_improves_brier():
    rows = _overconfident_rows()
    params = mc.fit_isotonic(rows)
    ys = [y for _, y in params["knots"]]
    assert ys == sorted(ys)
    assert mc.brier(rows, lambda p: mc.calibrate(p, params)) < mc.brier(rows, lambda p: p)


def test_fitters_refuse_tiny_samples():
    rows = _overconfident_rows(20)
    assert mc.fit_platt(rows) is None and mc.fit_isotonic(rows) is None


def test_calibrate_identity_without_params():
    assert mc.calibrate(0.73, None) == 0.73
    assert mc.calibrate(0.73, {"type": "unknown"}) == 0.73


def test_metrics_on_perfect_and_empty_predictions():
    perfect = [(1.0, 1, None), (0.0, 0, None)] * 10
    assert mc.brier(perfect, lambda p: p) == 0
    assert mc.ece(perfect, lambda p: p) == 0
    assert mc.brier([], lambda p: p) is None and mc.ece([], lambda p: p) is None
