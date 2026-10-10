"""
system_monitor.py - run every 2h via /loop. Surface anomalies for:
  HFT lab, BTC-ML, ETH-ML, Claude Picks
plus bot process health (log freshness + new ERROR/CRITICAL lines).

Stdout format:
  === SYSTEM MONITOR <iso> mode=<weekday|weekend> ===
  ALERT: ...                   <- one line per anomaly (may be zero)
  --- snapshot ---
  <per-system one-liners>
Persists last-seen state to system_monitor_state.json so subsequent runs
detect deltas (WR drift, block-ratio jumps, new error lines).
"""
import shutil
import sys
import json, re, sys, time, datetime as dt
from pathlib import Path

ROOT  = Path(__file__).parent
STATE = ROOT / 'system_monitor_state.json'

# (label, trade_file_base, [rl_state_file_bases], gate_skip_threshold, gate_min_visits)
# Suffix '.weekend.json' or '.json' is added at runtime based on weekend mode.
SYSTEMS = [
    ('HFT',          'hft_paper_trades',
        ['hft_rl_state'],                                                -0.30, 5),
    ('BTC-ML',       'btc_ml_trades',
        ['btc_ml_rl_state'],                                             -0.30, 5),
    ('ETH-ML',       'eth_ml_trades',
        ['eth_ml_rl_state'],                                             -0.30, 5),
    ('Claude-Picks', 'claude_paper_trades',
        ['claude_picks_mispriced_rl_state',
         'claude_picks_momentum_rl_state',
         'claude_picks_near_close_rl_state',
         'claude_picks_time_lock_rl_state',
         'claude_picks_vol_surge_rl_state'],                             -0.15, 5),
    # BTC-HERMES — pure Markov, no RL gate. Empty rl_bases means gate_metrics
    # returns (0,0,0.0) so block_pct stays 0% on the snapshot line. Trade-flow
    # and WR-drift checks still fire normally from the trade file.
    ('BTC-HERMES',   'btc_hermes_trades',
        [],                                                              -0.30, 5),
]

# BTC-HERMES nightly liveness — alert if the heartbeat file (written by every
# successful or partial nightly run) is older than this many hours. Set to ~28h
# (4h grace past the 24h schedule) to catch schtask outages without false-alarming
# normal jitter. Set to None to disable.
HERMES_NIGHTLY_STALE_HOURS = 28

WINDOW_SEC          = 2 * 3600   # 2h window for trade counts
TRADE_DROP_FRAC     = 0.60       # alert if cur < (1-frac)*prior
BLOCK_JUMP_PP       = 15.0       # alert if blocked% rose by this many pp vs prior snapshot
WR_DROP_PP          = 10.0       # alert if rolling-30 WR fell this many pp
LOG_STALE_MIN       = 15         # alert if bot_out.log not touched in this many minutes
WR_MIN_SAMPLE       = 20         # minimum resolved trades to trust a WR drift signal

def parse_iso_ts(s):
    """Returns POSIX seconds. HFT/BTC/ETH write tz-aware UTC ('...+00:00');
    Claude-Picks writes naive local time. Python's .timestamp() interprets
    naive datetimes as local time, which is what we want here."""
    if not s: return None
    s = str(s).replace('Z', '+00:00')
    try:
        return dt.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None

def load_json(p):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def is_weekend_active():
    wm = load_json(ROOT/'weekend_mode.json') or {}
    now = time.time()
    if wm.get('enabled') and wm.get('start_ts',0) <= now <= wm.get('end_ts',0):
        return True
    # Fallback heuristic: any .weekend.json trade file fresh in last 30 min
    for n in ('hft_paper_trades.weekend.json', 'btc_ml_trades.weekend.json',
              'eth_ml_trades.weekend.json', 'claude_paper_trades.weekend.json'):
        f = ROOT/n
        if f.exists() and now - f.stat().st_mtime < 1800:
            return True
    return False

def trade_metrics(trades, window_sec):
    """Returns (cur_count, prior_count, rolling_wr_pct_or_None, wr_sample_size).
    Volume counts include paper + real (heartbeat check). WR is computed on
    real_trade=True only, per feedback rule that circuit-breaker / drift
    signals must not fire on paper bleeds."""
    now = time.time()
    w_a = now - window_sec
    w_b = now - 2*window_sec
    cur_n = prior_n = 0
    resolved_real = []
    for t in trades:
        ts = parse_iso_ts(t.get('logged_at') or t.get('resolved_at'))
        if ts is None:
            continue
        if ts >= w_a:
            cur_n += 1
        elif ts >= w_b:
            prior_n += 1
        if (t.get('resolved') and t.get('outcome') in ('WIN', 'LOSS')
                and t.get('real_trade')):
            resolved_real.append((ts, t))
    resolved_real.sort(key=lambda x: x[0])
    last_n = [t for _, t in resolved_real[-30:]]
    if last_n:
        wins = sum(1 for t in last_n if t.get('outcome') == 'WIN')
        wr   = 100.0 * wins / len(last_n)
    else:
        wr = None
    return cur_n, prior_n, wr, len(last_n)

def gate_metrics(rl_state, skip_threshold, min_visits):
    """Returns (blocked_count, eligible_count, block_pct)."""
    if not rl_state:
        return 0, 0, 0.0
    eligible = [v for v in rl_state.values() if v.get('n', 0) >= min_visits]
    if not eligible:
        return 0, 0, 0.0
    blocked = sum(1 for v in eligible if v.get('q', 0) < skip_threshold)
    return blocked, len(eligible), 100.0 * blocked / len(eligible)

LOG_TS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
def parse_log_line_ts(line):
    m = LOG_TS_RE.match(line)
    if not m: return None
    try:
        return dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp()
    except Exception:
        return None

def collect_new_errors(log_path, since_ts):
    """Count ERROR/CRITICAL lines in log_path whose timestamp > since_ts.
    Returns (new_count, max_seen_ts, sample_messages)."""
    if not log_path.exists():
        return 0, since_ts, []
    try:
        raw = log_path.read_bytes().decode('utf-8', errors='replace')
    except Exception:
        return 0, since_ts, []
    new = 0
    max_ts = since_ts
    samples = []
    for line in raw.split('\n'):
        if 'ERROR' not in line and 'CRITICAL' not in line:
            continue
        ts = parse_log_line_ts(line)
        if ts is None:
            continue
        if since_ts is None or ts > since_ts:
            new += 1
            if len(samples) < 3:
                # strip the timestamp+level prefix for brevity
                msg = re.sub(r'^[^-]+- (ERROR|CRITICAL) - ', '', line).strip()
                samples.append(msg[:140])
            if max_ts is None or ts > max_ts:
                max_ts = ts
    return new, max_ts, samples

# Gate-event patterns: pause/unpause + circuit-breaker + auto-filter transitions.
# Bracketed tag is the discriminator — this AVOIDS matching the per-trade
# enforcement notices like `[ETH-ML-LIVE] AUTO-PAUSED -> PAPER (DOWN ...)`.
GATE_EVENT_RE = re.compile(
    r'\[(AUTO-PAUSE|AUTO-UNPAUSE|UNPAUSE|CIRCUIT-BREAKER|AUTO-FILTER)\]\s+(.{0,240})'
)

def collect_gate_events(log_path, since_ts):
    """Returns (deduped_summary_list, max_ts_seen).

    Dedupes by (event_type, first_strategy_token). CIRCUIT-BREAKER fires every
    5min for the same strategy - we want one alert per (evt, strategy) per
    cycle with count + latest message, not 24 individual alerts."""
    if not log_path.exists():
        return [], since_ts
    try:
        raw = log_path.read_bytes().decode('utf-8', errors='replace')
    except Exception:
        return [], since_ts

    groups = {}   # (evt, key) -> {count, msg, ts}
    max_ts = since_ts
    for line in raw.split('\n'):
        m = GATE_EVENT_RE.search(line)
        if not m:
            continue
        ts = parse_log_line_ts(line)
        if ts is None:
            continue
        if since_ts is not None and ts <= since_ts:
            continue
        evt = m.group(1)
        rest = m.group(2).strip()
        # Strategy key: first whitespace-delimited token after the tag.
        # Strip "play_type=" / "strategy=" prefixes; trim trailing punctuation.
        key_tok = (rest.split() or ['?'])[0]
        if '=' in key_tok:
            key_tok = key_tok.split('=', 1)[1]
        key_tok = key_tok.rstrip(':,.')
        k = (evt, key_tok)
        g = groups.setdefault(k, {'count': 0, 'msg': '', 'ts': 0})
        g['count'] += 1
        if ts > g['ts']:
            g['ts']  = ts
            g['msg'] = rest[:160]
        if max_ts is None or ts > max_ts:
            max_ts = ts

    summaries = []
    for (evt, key), g in sorted(groups.items(), key=lambda x: -x[1]['ts']):
        if g['count'] == 1:
            summaries.append(f'[{evt}] {key} :: {g["msg"]}')
        else:
            summaries.append(f'[{evt}] {key} x{g["count"]} :: latest: {g["msg"]}')
    return summaries, max_ts

WR_METHODOLOGY = 'real_only_v1'  # bump when WR/volume computation changes;
                                 # prior snapshots with a different tag skip delta-based alerts

def main():
    prior = load_json(STATE) or {}
    # Only compare deltas against a prior snapshot computed the same way
    prior_compatible = prior.get('wr_methodology') == WR_METHODOLOGY
    prior_sys = (prior.get('systems', {}) or {}) if prior_compatible else {}
    alerts = []
    snapshot_lines = []

    is_weekend = is_weekend_active()
    suffix     = '.weekend.json' if is_weekend else '.json'
    mode_tag   = 'weekend' if is_weekend else 'weekday'

    # 1. Bot log freshness
    bot_log = ROOT / 'bot_out.log'
    if not bot_log.exists():
        alerts.append('[BOT] bot_out.log missing - process likely never started')
        log_age_min = None
    else:
        log_age_min = (time.time() - bot_log.stat().st_mtime) / 60
        if log_age_min > LOG_STALE_MIN:
            alerts.append(f'[BOT] bot_out.log stale {log_age_min:.1f} min - process may be hung/dead')

    # 1b. Scanner auto-relaunch (2026-05-29). The main bot has crashed outright
    # (process gone) more than once. Recover it ONLY when truly dead — never when
    # a process still exists, so we don't double-launch a merely-hung instance
    # (the documented double-launch-race hazard). Trigger = log missing OR stale.
    SCANNER_SCRIPT = 'polymarket_bot.py'
    SCANNER_PY     = sys.executable
    bot_needs_relaunch = (log_age_min is None) or (log_age_min > LOG_STALE_MIN)
    if bot_needs_relaunch:
        try:
            import subprocess as _sp
            ps_cmd = (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -like 'python*' -and "
                f"$_.CommandLine -match '{SCANNER_SCRIPT}' }} | "
                "Select-Object -ExpandProperty ProcessId"
            )
            _r = _sp.run(['powershell', '-NoProfile', '-Command', ps_cmd],
                         capture_output=True, text=True, timeout=10)
            scanner_pids = [int(x) for x in (_r.stdout or '').strip().split() if x.isdigit()]
            if scanner_pids:
                # Alive but log stale => likely hung. Do NOT relaunch (double-launch
                # guard); surface for manual intervention instead.
                alerts.append(
                    f'[BOT] scanner log stale but process ALIVE (PID={scanner_pids}) - '
                    f'possible hang; NOT auto-relaunching (manual check needed)'
                )
            else:
                # Truly dead - safe to relaunch detached, capturing stdout+stderr.
                try:
                    launcher = [
                        'powershell', '-NoProfile', '-Command',
                        "$env:PYTHONIOENCODING='utf-8:replace'; "
                        f"Start-Process -FilePath '{SCANNER_PY}' "
                        f"-ArgumentList '{SCANNER_SCRIPT}' "
                        f"-WorkingDirectory '{ROOT}' "
                        f"-RedirectStandardOutput '{ROOT}\\bot_out.log' "
                        f"-RedirectStandardError '{ROOT}\\bot_err.log' "
                        "-WindowStyle Hidden -PassThru | Select-Object -ExpandProperty Id"
                    ]
                    _l = _sp.run(launcher, capture_output=True, text=True, timeout=15)
                    new_pid = (_l.stdout or '').strip()
                    alerts.append(f'[BOT] scanner was DOWN - relaunched (PID={new_pid or "?"})')
                except Exception as _le:
                    alerts.append(f'[BOT] scanner DOWN and relaunch failed: {_le}')
        except Exception as _se:
            alerts.append(f'[BOT] scanner relaunch check error: {_se}')

    # 2. New errors since last snapshot (skip alert on first run)
    new_errs, latest_err_ts, err_samples = collect_new_errors(bot_log, prior.get('last_err_ts'))
    if new_errs > 0 and prior.get('last_err_ts') is not None:
        sample_txt = ' | '.join(err_samples)
        alerts.append(f'[BOT] {new_errs} new ERROR/CRITICAL since last check: {sample_txt}')

    # 2b. New gate events (AUTO-PAUSE, CIRCUIT-BREAKER, AUTO-FILTER transitions)
    # at any log level. Skip alerts on first run (no baseline to compare).
    gate_summaries, latest_gate_ts = collect_gate_events(bot_log, prior.get('last_gate_event_ts'))
    if prior.get('last_gate_event_ts') is not None:
        for s in gate_summaries:
            alerts.append(s)

    # 2c. Telegram /cc responder health (2026-05-23).
    # The responder is what makes phone-message-from-Telegram → Claude Code work.
    # If it dies, the user gets no reply when they message @polyquantaibot.
    # Check for a python process running tg_cc_responder.py; if missing, try
    # to relaunch it detached. Alert on first run too — silent failure is the
    # whole bug we're guarding against.
    try:
        import subprocess as _sp
        ps_cmd = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'tg_cc_responder.py' } | "
            "Select-Object -ExpandProperty ProcessId"
        )
        _r = _sp.run(['powershell', '-NoProfile', '-Command', ps_cmd],
                     capture_output=True, text=True, timeout=10)
        responder_pids = [int(x) for x in (_r.stdout or '').strip().split() if x.isdigit()]
        if not responder_pids:
            # Try to relaunch detached. PYTHONIOENCODING avoids cp1252 issues
            # with em-dashes / smart quotes in phone messages.
            try:
                launcher = [
                    'powershell', '-NoProfile', '-Command',
                    "$env:PYTHONIOENCODING='utf-8:replace'; "
                    "Start-Process -FilePath 'python.exe' "
                    "-ArgumentList 'tg_cc_responder.py' "
                    f"-WorkingDirectory '{ROOT}' "
                    f"-RedirectStandardOutput '{ROOT}\\tg_cc_responder.log' "
                    f"-RedirectStandardError '{ROOT}\\tg_cc_responder.err.log' "
                    "-WindowStyle Hidden -PassThru | Select-Object -ExpandProperty Id"
                ]
                _l = _sp.run(launcher, capture_output=True, text=True, timeout=10)
                new_pid = (_l.stdout or '').strip()
                alerts.append(f'[BOT] tg_cc_responder.py was DOWN — relaunched (PID={new_pid or "?"})')
            except Exception as _le:
                alerts.append(f'[BOT] tg_cc_responder.py DOWN and relaunch failed: {_le}')
        # (silent OK path — no alert when responder is alive)
    except Exception as _re:
        alerts.append(f'[BOT] responder health check error: {_re}')

    # 2d. Cloudflared quick-tunnel health (2026-05-25).
    # cloudflared exposes the bot publicly via https://*.trycloudflare.com.
    # The slug ROTATES on each restart — when cloudflared dies, the previous
    # URL is dead. Auto-relaunch and TG the new URL.
    CLOUDFLARED_EXE = (shutil.which('cloudflared') or 'cloudflared')
    try:
        import subprocess as _sp
        ps_cmd = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'cloudflared.*tunnel.*--url' } | "
            "Select-Object -ExpandProperty ProcessId"
        )
        _r = _sp.run(['powershell', '-NoProfile', '-Command', ps_cmd],
                     capture_output=True, text=True, timeout=10)
        cf_pids = [int(x) for x in (_r.stdout or '').strip().split() if x.isdigit()]
        if not cf_pids:
            # Relaunch detached. stderr -> cloudflare_err.log (URL is printed there).
            try:
                launcher = [
                    'powershell', '-NoProfile', '-Command',
                    f"Start-Process -FilePath '{CLOUDFLARED_EXE}' "
                    # --protocol http2 (2026-05-29): QUIC/UDP is unstable through NordVPN
                    # (NordLynx), causing recurring tunnel drops + argotunnel DNS timeouts.
                    # HTTP/2 runs over TCP and survives the VPN far better.
                    "-ArgumentList @('tunnel','--url','http://localhost:5002','--no-autoupdate','--protocol','http2') "
                    f"-WorkingDirectory '{ROOT}' "
                    f"-RedirectStandardOutput '{ROOT}\\cloudflare.log' "
                    f"-RedirectStandardError '{ROOT}\\cloudflare_err.log' "
                    "-WindowStyle Hidden -PassThru | Select-Object -ExpandProperty Id"
                ]
                _l = _sp.run(launcher, capture_output=True, text=True, timeout=10)
                new_pid = (_l.stdout or '').strip()
                # Give cloudflared 8s to negotiate + print the URL
                time.sleep(8)
                # Grep the new URL from cloudflare_err.log
                err_path = ROOT / 'cloudflare_err.log'
                new_url = None
                if err_path.exists():
                    try:
                        raw = err_path.read_bytes().decode('utf-8', errors='replace')
                        urls = re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com', raw)
                        if urls:
                            new_url = urls[-1]
                    except Exception:
                        pass
                url_str = new_url or '(URL not yet visible — check cloudflare_err.log)'
                alerts.append(
                    f'[CLOUDFLARED] was DOWN — relaunched (PID={new_pid or "?"}). '
                    f'New URL: {url_str}'
                )
            except Exception as _le:
                alerts.append(f'[CLOUDFLARED] DOWN and relaunch failed: {_le}')
        # (silent OK path — no alert when tunnel is alive; URL hasn't changed)
    except Exception as _ce:
        alerts.append(f'[CLOUDFLARED] health check error: {_ce}')

    # 3. Per-system checks
    cur = {'mode': mode_tag, 'wr_methodology': WR_METHODOLOGY,
           'last_err_ts': latest_err_ts,
           'last_gate_event_ts': latest_gate_ts,
           'systems': {}}
    for name, trade_base, rl_bases, gate_thresh, gate_min in SYSTEMS:
        sys_state = {}
        tf = ROOT / (trade_base + suffix)
        if not tf.exists():
            # BTC-HERMES has no trade file until its Markov gate finally fires
            # (MIN_PROB=0.87 on a random-walk matrix means trades may be rare).
            # Missing file is the expected steady state — info, not alert.
            if name != 'BTC-HERMES':
                alerts.append(f'[{name}] trade file missing: {tf.name}')
            snapshot_lines.append(f'{name:<14} (no trade file)')
            cur['systems'][name] = sys_state
            continue
        trades = load_json(tf) or []
        cur_n, prior_n, wr, wr_n = trade_metrics(trades, WINDOW_SEC)
        sys_state.update({'trades_2h': cur_n, 'wr30': wr, 'wr_n': wr_n})

        if cur_n == 0 and prior_n >= 5:
            alerts.append(f'[{name}] NO trades in last 2h (prior 2h had {prior_n})')
        elif prior_n >= 10 and cur_n < (1 - TRADE_DROP_FRAC) * prior_n:
            alerts.append(f'[{name}] trade drop: prior 2h={prior_n} -> last 2h={cur_n}')

        prior_wr = (prior_sys.get(name) or {}).get('wr30')
        if wr is not None and prior_wr is not None and wr_n >= WR_MIN_SAMPLE:
            drop = prior_wr - wr
            if drop >= WR_DROP_PP:
                alerts.append(f'[{name}] rolling-30 WR drop {prior_wr:.1f}% -> {wr:.1f}% (n={wr_n})')

        # gate state (sum across per-type RL files for Claude-Picks)
        tot_b = tot_e = 0
        for rb in rl_bases:
            rl = load_json(ROOT / (rb + suffix)) or {}
            b, e, _ = gate_metrics(rl, gate_thresh, gate_min)
            tot_b += b
            tot_e += e
        cur_pct = (100.0 * tot_b / tot_e) if tot_e else 0.0
        sys_state.update({'block_pct': round(cur_pct, 1), 'blocked': tot_b, 'eligible': tot_e})

        prior_pct = (prior_sys.get(name) or {}).get('block_pct')
        if tot_e > 0 and prior_pct is not None and (cur_pct - prior_pct) >= BLOCK_JUMP_PP:
            alerts.append(
                f'[{name}] gate block ratio jumped {prior_pct:.1f}% -> {cur_pct:.1f}% '
                f'({tot_b}/{tot_e} eligible cells)'
            )
        if tot_e >= 5 and tot_b == tot_e:
            alerts.append(f'[{name}] 100% of {tot_e} gate cells BLOCKED - no signals can pass')

        wr_str = f'WR30={wr:.1f}%' if wr is not None else 'WR30=n/a'
        snapshot_lines.append(
            f'{name:<14} trades_2h={cur_n:>3} (prior {prior_n:>3})  '
            f'{wr_str:<13} block={tot_b}/{tot_e} ({cur_pct:.1f}%)'
        )
        cur['systems'][name] = sys_state

    # BTC-HERMES nightly liveness check. Reads the heartbeat file written by
    # btc_hermes_nightly.py at end of each run (Phase 3 of the BTC-HERMES build).
    if HERMES_NIGHTLY_STALE_HOURS is not None:
        lastrun = ROOT / 'btc_hermes_nightly_lastrun.json'
        trades  = ROOT / 'btc_hermes_trades.json'
        if lastrun.exists():
            age_h = (time.time() - lastrun.stat().st_mtime) / 3600
            if age_h > HERMES_NIGHTLY_STALE_HOURS:
                alerts.append(
                    f'[BTC-HERMES] nightly heartbeat stale {age_h:.1f}h '
                    f'(threshold {HERMES_NIGHTLY_STALE_HOURS}h) — schtask BTC_Hermes_Nightly may be down'
                )
        elif trades.exists():
            # Trades file exists but heartbeat never written = nightly has never completed.
            alerts.append('[BTC-HERMES] nightly never ran successfully (no btc_hermes_nightly_lastrun.json)')

    # Persist snapshot
    try:
        STATE.write_text(json.dumps(cur, indent=2))
    except Exception as e:
        alerts.append(f'[MON] failed to persist snapshot: {e}')

    # Output
    now_iso = dt.datetime.now().isoformat(timespec='seconds')
    print(f'=== SYSTEM MONITOR {now_iso} mode={mode_tag} ===')
    if log_age_min is not None:
        print(f'(bot_out.log age: {log_age_min:.1f} min)')
    for a in alerts:
        print(f'ALERT: {a}')
    print('--- snapshot ---')
    for line in snapshot_lines:
        print(line)
    # Final summary line for easy grep
    print(f'--- {len(alerts)} alert(s) ---')

if __name__ == '__main__':
    main()
