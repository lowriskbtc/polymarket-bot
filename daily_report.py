"""
PolyQuant AI — Daily Automated Report
Runs every day at 6am via Windows Task Scheduler.
Generates HTML report + sends Telegram summary to @polyquantaibot.
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
NOTES_FILE   = os.path.join(BASE_DIR, 'daily_notes.json')  # append research notes here
TG_TOKEN     = os.environ.get('TG_BOT_TOKEN', '')
TG_CHAT_ID   = os.environ.get('TG_CHAT_ID', '')
REPORT_DIR   = os.path.join(os.path.expanduser('~'), 'Desktop')
LOG_FILE     = os.path.join(BASE_DIR, 'crypto_predictions.log')

# ── Helpers ─────────────────────────────────────────────────────────────────
def load_json(path, default=None):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def tg_send(text):
    try:
        url  = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage'
        data = urllib.parse.urlencode({
            'chat_id': TG_CHAT_ID, 'text': text, 'parse_mode': 'HTML'
        }).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f'[TG] Send error: {e}')

def breakdown(trades):
    w   = sum(1 for t in trades if t.get('outcome') == 'WIN')
    l   = sum(1 for t in trades if t.get('outcome') == 'LOSS')
    pnl = sum(t.get('paper_pnl') or 0 for t in trades if t.get('resolved'))
    tot = w + l
    return {'wins': w, 'losses': l, 'total': tot,
            'wr': round(w / tot * 100, 1) if tot else 0,
            'pnl': round(pnl, 2)}

def pnl_str(v):
    return f'+${v:.2f}' if v >= 0 else f'-${abs(v):.2f}'

def wr_color(wr):
    if wr >= 65: return 'var(--green)'
    if wr >= 55: return 'var(--yellow)'
    return 'var(--red)'

# ── Load all data ───────────────────────────────────────────────────────────
def gather():
    now = datetime.now()

    # HFT paper trades
    hft_trades = load_json(os.path.join(BASE_DIR, 'hft_paper_trades.json'), [])
    hft_resolved = [t for t in hft_trades if t.get('resolved')]

    hft_all   = breakdown(hft_resolved)
    hft_by_strat = {}
    hft_by_sym   = {}
    for t in hft_resolved:
        s = t.get('strategy', 'UNKNOWN')
        sym = t.get('symbol', 'UNKNOWN')
        hft_by_strat.setdefault(s, []).append(t)
        hft_by_sym.setdefault(sym, []).append(t)
    hft_by_strat = {k: breakdown(v) for k, v in hft_by_strat.items()}
    hft_by_sym   = {k: breakdown(v) for k, v in hft_by_sym.items()}

    # Claude paper log — file was renamed; old name 'claude_paper_log.json'
    # didn't exist so reports always showed 0 Claude trades.
    claude_trades = load_json(os.path.join(BASE_DIR, 'claude_paper_trades.json'), [])
    claude_resolved = [t for t in claude_trades if t.get('resolved')]
    claude_all = breakdown(claude_resolved)

    claude_by_type = {}
    claude_by_conv = {}
    for t in claude_resolved:
        pt = t.get('play_type', 'UNKNOWN')
        cv = t.get('conviction', 'UNKNOWN')
        claude_by_type.setdefault(pt, []).append(t)
        claude_by_conv.setdefault(cv, []).append(t)
    claude_by_type = {k: breakdown(v) for k, v in claude_by_type.items()}
    claude_by_conv = {k: breakdown(v) for k, v in claude_by_conv.items()}
    claude_pending = sum(1 for t in claude_trades if not t.get('resolved'))
    claude_staked  = sum(t.get('paper_stake') or 0 for t in claude_trades)

    # RL states
    hft_rl   = load_json(os.path.join(BASE_DIR, 'hft_rl_state.json'))
    manip_rl = load_json(os.path.join(BASE_DIR, 'manip_rl_state.json'))

    hft_rl_states = [{'key': k, 'q': v['q'], 'n': v['n']}
                     for k, v in hft_rl.items() if isinstance(v, dict) and v.get('n', 0) >= 5]
    manip_rl_states = [{'key': k, 'q': v['q'], 'n': v['n']}
                       for k, v in manip_rl.items() if isinstance(v, dict) and v.get('n', 0) >= 5]

    hft_rl_states.sort(key=lambda x: x['q'], reverse=True)
    manip_rl_states.sort(key=lambda x: x['q'], reverse=True)

    # Backtest winners
    bt = load_json(os.path.join(BASE_DIR, 'max_backtest_results.json'))
    bt_winners  = bt.get('winners', [])
    bt_days     = bt.get('days', 0)
    bt_ts       = bt.get('timestamp', '')[:10]

    # HFT settings
    hft_settings = load_json(os.path.join(BASE_DIR, 'hft_settings.json'))
    live_mode  = hft_settings.get('live_mode', False)
    stake      = hft_settings.get('stake', 8.0)
    dd_floor   = hft_settings.get('drawdown_floor', 500)

    # Recent log lines — scan for anomalies in last 500 lines
    flags = []
    try:
        with open(LOG_FILE, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 80000))
            lines = f.read().decode('utf-8', errors='ignore').splitlines()
        recent = lines[-500:]
        loss_streak = 0
        max_loss_streak = 0
        for line in recent:
            if '[HFT] LOSS' in line:
                loss_streak += 1
                max_loss_streak = max(max_loss_streak, loss_streak)
            elif '[HFT] WIN' in line:
                loss_streak = 0
        if max_loss_streak >= 5:
            flags.append(f'⚠️ HFT loss streak of {max_loss_streak} detected in last 500 log lines')
        errors = [l for l in recent if 'ERROR' in l or 'Save error' in l or 'Traceback' in l]
        if errors:
            flags.append(f'⚠️ {len(errors)} error(s) in recent log')
        if not live_mode:
            flags.append('⚠️ HFT is in PAPER mode — not trading live')
    except Exception as e:
        flags.append(f'⚠️ Log read error: {e}')

    # Freshness check — if the trade files are stale, the bot is probably down
    # or this script is reading the wrong directory (which is exactly how reports
    # silently lied for weeks before this guard was added).
    import time as _t
    _now_ts = _t.time()
    _stale_files = []
    for _fn in ('hft_paper_trades.json', 'claude_paper_trades.json',
                'btc_ml_trades.json', 'eth_ml_trades.json'):
        _p = os.path.join(BASE_DIR, _fn)
        if not os.path.exists(_p):
            _stale_files.append(f'{_fn}: MISSING'); continue
        _age_min = (_now_ts - os.path.getmtime(_p)) / 60.0
        if _age_min > 60:
            _stale_files.append(f'{_fn}: {_age_min:.0f} min old')
    if _stale_files:
        flags.insert(0, '🛑 STALE DATA — bot may be down or report reading wrong dir: '
                        + '; '.join(_stale_files))

    # Research notes
    notes = load_json(NOTES_FILE, [])

    # Lifetime real-money PnL across every trade source. The HFT/Claude
    # paper-vs-real distinction was hidden in the legacy report, which is
    # how a -$1k live book got mistaken for a +$1k paper book.
    real_sources = [
        ('HFT',       'hft_paper_trades.json'),
        ('Claude',    'claude_paper_trades.json'),
        ('BTC-ML',    'btc_ml_trades.json'),
        ('ETH-ML',    'eth_ml_trades.json'),
    ]
    real_breakdown = []
    real_total_n = 0; real_total_pnl = 0.0; real_total_w = 0; real_total_l = 0
    for label, fn in real_sources:
        src = load_json(os.path.join(BASE_DIR, fn), [])
        real = [t for t in src if t.get('real_trade') is True and t.get('paper_pnl') is not None]
        n = len(real)
        if n == 0:
            real_breakdown.append({'src': label, 'n': 0, 'pnl': 0.0, 'wr': 0.0})
            continue
        pnl = sum(t.get('paper_pnl') or 0 for t in real)
        w = sum(1 for t in real if (t.get('paper_pnl') or 0) > 0)
        l = n - w
        real_breakdown.append({'src': label, 'n': n, 'pnl': round(pnl, 2),
                               'wr': round(100 * w / n, 1)})
        real_total_n += n; real_total_pnl += pnl; real_total_w += w; real_total_l += l
    real_money = {
        'total_n':   real_total_n,
        'total_pnl': round(real_total_pnl, 2),
        'wins':      real_total_w,
        'losses':    real_total_l,
        'wr':        round(100 * real_total_w / real_total_n, 1) if real_total_n else 0.0,
        'breakdown': real_breakdown,
    }

    return {
        'now': now,
        'hft_all': hft_all,
        'hft_by_strat': hft_by_strat,
        'hft_by_sym': hft_by_sym,
        'claude_all': claude_all,
        'claude_by_type': claude_by_type,
        'claude_by_conv': claude_by_conv,
        'claude_pending': claude_pending,
        'claude_staked': round(claude_staked, 2),
        'hft_rl_states': hft_rl_states,
        'manip_rl_states': manip_rl_states,
        'bt_winners': bt_winners,
        'bt_days': bt_days,
        'bt_ts': bt_ts,
        'live_mode': live_mode,
        'stake': stake,
        'dd_floor': dd_floor,
        'flags': flags,
        'notes': notes,
        'real_money': real_money,
        'claude_resolved': claude_resolved,
    }

# ── Telegram summary ─────────────────────────────────────────────────────────
def send_telegram(d):
    now_str   = d['now'].strftime('%b %d, %Y %H:%M')
    hft       = d['hft_all']
    claude    = d['claude_all']
    live_icon = '🟢' if d['live_mode'] else '🔴'
    flag_lines = '\n'.join(d['flags']) if d['flags'] else '✅ No issues detected'

    # Top HFT symbols by PnL
    sym_sorted = sorted(d['hft_by_sym'].items(), key=lambda x: x[1]['pnl'], reverse=True)
    sym_lines  = '\n'.join(
        f"  {'🔥' if i==0 else '·'} {sym}: {s['wr']}% WR  {pnl_str(s['pnl'])}"
        for i, (sym, s) in enumerate(sym_sorted[:4])
    )

    # Top strategy
    strat_lines = '\n'.join(
        f"  · {name}: {s['wr']}% WR  {pnl_str(s['pnl'])}"
        for name, s in d['hft_by_strat'].items()
    )

    # Top claude type
    tl = d['claude_by_type'].get('TIME-LOCK', {})
    mn = d['claude_by_type'].get('MANIP', {})

    msg = f"""
📊 <b>PolyQuant AI — Daily Report</b>
{now_str}

{live_icon} <b>HFT</b>  {hft['wr']}% WR · {pnl_str(hft['pnl'])} PnL
{hft['wins']}W / {hft['losses']}L · {hft['total']} total trades
Stake: ${d['stake']:.2f} · Floor: ${d['dd_floor']:.0f}

{strat_lines}

📍 <b>By Symbol (Top 4)</b>
{sym_lines}

🎯 <b>Claude Picks</b>  {claude['wr']}% WR · {pnl_str(claude['pnl'])}
{claude['wins']}W / {claude['losses']}L · {d['claude_pending']} pending
Staked: ${d['claude_staked']:.2f}

TIME-LOCK: {tl.get('wr',0)}% WR · {pnl_str(tl.get('pnl',0))} ({tl.get('total',0)} trades)
MANIP: {mn.get('wr',0)}% WR · {pnl_str(mn.get('pnl',0))} ({mn.get('total',0)} trades)

🔔 <b>Flags</b>
{flag_lines}

📁 Full report saved to Desktop
""".strip()

    tg_send(msg)
    print(f'[TG] Sent daily summary')

# ── Charts ───────────────────────────────────────────────────────────────────
def build_charts(d):
    """Render PNG charts and return them as base64 data-URL strings.
    Mirrors the styling of the bot's _tg_pnl_report so HTML and TG charts
    look like they came from the same place. Failure here must not break
    the report — return empty strings on any matplotlib problem.
    """
    out = {'cumulative': '', 'by_strategy': '', 'by_conviction': '', 'real_vs_paper': ''}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io as _io, base64 as _b64

        BG, FG, TXT, MUTED = '#0d1117', '#161b22', '#e6edf3', '#8b949e'
        SPINE, GRID, GREEN, RED = '#30363d', '#444', '#00c87a', '#ff3b3b'

        def _to_b64(fig):
            buf = _io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=BG)
            plt.close(fig)
            buf.seek(0)
            return 'data:image/png;base64,' + _b64.b64encode(buf.read()).decode()

        def _style_axes(ax):
            ax.set_facecolor(FG)
            ax.tick_params(colors=MUTED, labelsize=8)
            for s in ax.spines.values(): s.set_edgecolor(SPINE)

        resolved = d.get('claude_resolved', [])
        r_sorted = sorted(resolved, key=lambda t: t.get('logged_at',''))

        # ── 1. Cumulative PnL curve ─────────────────────────────────────────
        if r_sorted:
            cumulative, labels = [], []
            running = 0.0
            for t in r_sorted:
                running += t.get('paper_pnl') or 0
                cumulative.append(running)
                labels.append((t.get('logged_at') or '')[:10])
            fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=BG)
            _style_axes(ax)
            ax.fill_between(range(len(cumulative)), cumulative, 0,
                            where=[v >= 0 for v in cumulative], color=GREEN, alpha=0.15)
            ax.fill_between(range(len(cumulative)), cumulative, 0,
                            where=[v < 0 for v in cumulative], color=RED, alpha=0.15)
            line_color = GREEN if cumulative[-1] >= 0 else RED
            ax.plot(cumulative, color=line_color, linewidth=1.8)
            ax.axhline(0, color=GRID, linewidth=0.8, linestyle='--')
            step = max(1, len(labels) // 10)
            ax.set_xticks(range(0, len(labels), step))
            ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                               rotation=30, ha='right', fontsize=7, color=MUTED)
            ax.set_title(f'Cumulative Claude Paper PnL  ·  ${cumulative[-1]:+,.2f}',
                         color=TXT, fontsize=10, pad=8)
            ax.set_ylabel('$', color=MUTED, fontsize=8)
            out['cumulative'] = _to_b64(fig)

        # ── 2. Per-strategy PnL bars ────────────────────────────────────────
        strat_data = []
        for pt in ('TIME-LOCK','MISPRICED','MOMENTUM','NEAR-CLOSE','VOL-SURGE','MANIP'):
            ts = [t for t in resolved if t.get('play_type') == pt]
            if not ts: continue
            pnl = sum(t.get('paper_pnl') or 0 for t in ts)
            strat_data.append((pt, pnl, len(ts)))
        if strat_data:
            names = [s[0] for s in strat_data]
            pnls  = [s[1] for s in strat_data]
            ns    = [s[2] for s in strat_data]
            fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=BG)
            _style_axes(ax)
            colors = [GREEN if p >= 0 else RED for p in pnls]
            bars = ax.bar(names, pnls, color=colors, edgecolor=SPINE)
            ax.axhline(0, color=GRID, linewidth=0.8)
            for bar, val, n in zip(bars, pnls, ns):
                offset = max(abs(min(pnls)), max(pnls)) * 0.03
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (offset if val >= 0 else -offset),
                        f'${val:+.0f}\n(n={n})',
                        ha='center', va='bottom' if val >= 0 else 'top',
                        fontsize=7, color=TXT)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8, color=MUTED)
            ax.set_title('Claude Paper PnL by Strategy', color=TXT, fontsize=10, pad=8)
            ax.set_ylabel('$', color=MUTED, fontsize=8)
            out['by_strategy'] = _to_b64(fig)

        # ── 3. Win rate by conviction ───────────────────────────────────────
        conv_data = []
        for c in ('EXTREME','STRONG','HIGH','MEDIUM'):
            ts = [t for t in resolved if t.get('conviction') == c]
            if not ts: continue
            w = sum(1 for t in ts if (t.get('paper_pnl') or 0) > 0)
            conv_data.append((c, 100 * w / len(ts), len(ts)))
        if conv_data:
            names = [c[0] for c in conv_data]
            wrs   = [c[1] for c in conv_data]
            ns    = [c[2] for c in conv_data]
            fig, ax = plt.subplots(figsize=(10, 3.0), facecolor=BG)
            _style_axes(ax)
            ccolors = ['#ff3b3b','#00d4ff','#ff6600','#f5c518'][:len(names)]
            bars = ax.bar(names, wrs, color=ccolors, edgecolor=SPINE)
            ax.axhline(50, color=GRID, linewidth=0.8, linestyle='--')
            ax.set_ylim(0, 110)
            for bar, wr_v, n in zip(bars, wrs, ns):
                ax.text(bar.get_x() + bar.get_width()/2, wr_v + 2,
                        f'{wr_v:.0f}%\n(n={n})', ha='center', va='bottom',
                        fontsize=8, color=TXT)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=8, color=MUTED)
            ax.set_title('Win Rate by Conviction', color=TXT, fontsize=10, pad=8)
            ax.set_ylabel('%', color=MUTED, fontsize=8)
            out['by_conviction'] = _to_b64(fig)

        # ── 4. Real vs Paper by source (HTML-only, exposes the divergence) ──
        rm = d.get('real_money', {})
        bd = rm.get('breakdown', [])
        bd_active = [b for b in bd if b['n'] > 0]
        if bd_active:
            srcs = [b['src'] for b in bd_active]
            real_pnls = [b['pnl'] for b in bd_active]
            fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=BG)
            _style_axes(ax)
            colors = [GREEN if p >= 0 else RED for p in real_pnls]
            bars = ax.bar(srcs, real_pnls, color=colors, edgecolor=SPINE)
            ax.axhline(0, color=GRID, linewidth=0.8)
            rng = max(abs(min(real_pnls)), max(real_pnls)) or 1
            for bar, val, b in zip(bars, real_pnls, bd_active):
                offset = rng * 0.04
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (offset if val >= 0 else -offset),
                        f'${val:+.0f}\n{b["wr"]:.0f}% WR\n(n={b["n"]:,})',
                        ha='center', va='bottom' if val >= 0 else 'top',
                        fontsize=7, color=TXT)
            ax.set_xticks(range(len(srcs)))
            ax.set_xticklabels(srcs, fontsize=9, color=MUTED)
            total = rm.get('total_pnl', 0)
            ax.set_title(f'Real-Money PnL by Source  ·  Total ${total:+,.2f}',
                         color=TXT, fontsize=10, pad=8)
            ax.set_ylabel('$', color=MUTED, fontsize=8)
            out['real_vs_paper'] = _to_b64(fig)

    except Exception as e:
        print(f'[REPORT] Chart build failed: {e}')
    return out


# ── HTML Report ──────────────────────────────────────────────────────────────
def generate_html(d):
    now_str = d['now'].strftime('%B %d, %Y · %H:%M')
    date_str = d['now'].strftime('%Y-%m-%d')
    hft  = d['hft_all']
    cl   = d['claude_all']
    roi  = round(cl['pnl'] / max(d['claude_staked'], 1) * 100, 1)

    def wr_badge(wr):
        if wr >= 65: col = '#00c87a'
        elif wr >= 55: col = '#f5c518'
        else: col = '#ff3b3b'
        return f'<span style="color:{col};font-weight:700">{wr}%</span>'

    def pnl_badge(v):
        col = '#00c87a' if v >= 0 else '#ff3b3b'
        return f'<span style="color:{col}">{pnl_str(v)}</span>'

    # Strategy rows
    strat_rows = ''.join(
        f'<tr><td>{name}</td><td>{wr_badge(s["wr"])}</td>'
        f'<td>{pnl_badge(s["pnl"])}</td>'
        f'<td style="color:#5a5a7a">{s["total"]}</td></tr>'
        for name, s in d['hft_by_strat'].items()
    )

    # Symbol rows sorted by PnL
    sym_rows = ''.join(
        f'<tr><td>{sym}</td><td>{wr_badge(s["wr"])}</td>'
        f'<td>{pnl_badge(s["pnl"])}</td>'
        f'<td style="color:#5a5a7a">{s["total"]}</td></tr>'
        for sym, s in sorted(d['hft_by_sym'].items(), key=lambda x: x[1]['pnl'], reverse=True)
    )

    # Claude conviction rows
    conv_order = ['EXTREME', 'HIGH', 'STRONG', 'MEDIUM']
    conv_colors = {'EXTREME': '#ff3b3b', 'HIGH': '#ff8c00', 'STRONG': '#3b8ee0', 'MEDIUM': '#f5c518'}
    conv_rows = ''.join(
        f'<tr><td><span style="color:{conv_colors.get(c,"#fff")};font-weight:700">{c}</span></td>'
        f'<td>{wr_badge(d["claude_by_conv"].get(c,{}).get("wr",0))}</td>'
        f'<td>{pnl_badge(d["claude_by_conv"].get(c,{}).get("pnl",0))}</td>'
        f'<td style="color:#5a5a7a">{d["claude_by_conv"].get(c,{}).get("total",0)}</td></tr>'
        for c in conv_order if c in d['claude_by_conv']
    )

    # Claude type rows
    type_order = ['TIME-LOCK', 'NEAR-CLOSE', 'MISPRICED', 'MANIP', 'VOL-SURGE', 'MOMENTUM']
    type_colors = {'TIME-LOCK':'#ff3b3b','NEAR-CLOSE':'#ffa500','MISPRICED':'#00c87a',
                   'MANIP':'#b450ff','VOL-SURGE':'#b450ff','MOMENTUM':'#3b8ee0'}
    type_rows = ''.join(
        f'<tr><td><span style="color:{type_colors.get(pt,"#fff")};font-weight:700">{pt}</span></td>'
        f'<td>{wr_badge(d["claude_by_type"].get(pt,{}).get("wr",0))}</td>'
        f'<td>{pnl_badge(d["claude_by_type"].get(pt,{}).get("pnl",0))}</td>'
        f'<td style="color:#5a5a7a">{d["claude_by_type"].get(pt,{}).get("total",0)}</td></tr>'
        for pt in type_order if pt in d['claude_by_type']
    )

    # HFT RL top 5 / bottom 5
    rl_top = ''.join(
        f'<tr><td style="font-family:monospace;font-size:10px">{s["key"]}</td>'
        f'<td style="color:#00c87a">+{s["q"]:.3f}</td>'
        f'<td style="color:#5a5a7a">{s["n"]}</td></tr>'
        for s in d['hft_rl_states'][:5]
    )
    rl_bot = ''.join(
        f'<tr><td style="font-family:monospace;font-size:10px">{s["key"]}</td>'
        f'<td style="color:#ff3b3b">{s["q"]:.3f}</td>'
        f'<td style="color:#5a5a7a">{s["n"]}</td></tr>'
        for s in reversed(d['hft_rl_states'][-5:])
    )

    # Manip RL
    manip_top = ''.join(
        f'<tr><td style="font-family:monospace;font-size:10px">{s["key"]}</td>'
        f'<td style="color:#00c87a">+{s["q"]:.3f}</td>'
        f'<td style="color:#5a5a7a">{s["n"]}</td></tr>'
        for s in d['manip_rl_states'][:5]
    )
    manip_bot = ''.join(
        f'<tr><td style="font-family:monospace;font-size:10px">{s["key"]}</td>'
        f'<td style="color:#ff3b3b">{s["q"]:.3f}</td>'
        f'<td style="color:#5a5a7a">{s["n"]}</td></tr>'
        for s in reversed(d['manip_rl_states'][-5:])
    )

    # Backtest top 10
    bt_rows = ''.join(
        f'<tr><td style="color:#3b8ee0">{w["algo_name"]}</td><td>{w["symbol"]}</td>'
        f'<td style="color:#5a5a7a">{w["tf"]}</td>'
        f'<td>{wr_badge(w["win_rate"])}</td>'
        f'<td style="color:#00c87a">+{w["ev"]}</td>'
        f'<td style="color:#5a5a7a">{w["total"]}</td>'
        f'<td style="color:#5a5a7a">{w["signals_per_day"]}/d</td></tr>'
        for w in d['bt_winners'][:10]
    )

    # Research notes
    notes_html = ''
    if d['notes']:
        note_items = ''
        for n in sorted(d['notes'], key=lambda x: x.get('date',''), reverse=True):
            tag_color = {'TIME-LOCK':'#ff3b3b','MANIP':'#b450ff','HFT':'#3b8ee0',
                         'BACKTEST':'#f5c518','SYSTEM':'#9090b0'}.get(n.get('tag',''), '#9090b0')
            body_html = n.get('body','').replace('\n', '<br>')
            action_html = (
                f'<div style="background:rgba(0,200,122,.06);border:1px solid rgba(0,200,122,.2);'
                f'border-radius:4px;padding:8px;margin-top:8px;font-size:11px;color:#ccc">'
                f'<strong style="color:#00c87a">Action:</strong> {n["action"]}</div>'
            ) if n.get('action') else ''
            note_items += (
                f'<div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;'
                f'padding:14px;margin-bottom:10px">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
                f'<span style="background:{tag_color};color:#000;font-size:9px;font-weight:700;'
                f'padding:2px 7px;border-radius:10px;letter-spacing:.06em">{n.get("tag","NOTE")}</span>'
                f'<span style="font-weight:700;color:#fff">{n.get("title","")}</span>'
                f'<span style="color:var(--text3);font-size:10px;margin-left:auto">{n.get("date","")}</span>'
                f'</div>'
                f'<div style="color:var(--text2);font-size:12px;line-height:1.8">{body_html}</div>'
                f'{action_html}'
                f'</div>'
            )
        notes_html = (
            f'<div class="section">'
            f'<div class="sec-title">Research Notes</div>'
            f'{note_items}'
            f'</div>'
        )

    # Build charts (PNG → base64 data URLs). Returns empty strings on failure
    # so the report still renders even if matplotlib chokes.
    charts = build_charts(d)
    def _chart_img(key, alt):
        src = charts.get(key) or ''
        if not src:
            return ''
        return (f'<div style="background:var(--bg2);border:1px solid var(--border);'
                f'border-radius:8px;padding:10px;margin-bottom:12px">'
                f'<img src="{src}" alt="{alt}" style="width:100%;display:block;border-radius:4px"/>'
                f'</div>')

    # Real-money section — paper PnL has been misleading reports for weeks
    rm = d['real_money']
    rm_color = '#00c87a' if rm['total_pnl'] >= 0 else '#ff3b3b'
    rm_rows = ''.join(
        f'<tr><td>{b["src"]}</td>'
        f'<td style="color:#5a5a7a">{b["n"]:,}</td>'
        f'<td>{wr_badge(b["wr"]) if b["n"] else "<span style=color:#5a5a7a>—</span>"}</td>'
        f'<td>{pnl_badge(b["pnl"]) if b["n"] else "<span style=color:#5a5a7a>—</span>"}</td></tr>'
        for b in rm['breakdown']
    )
    real_money_html = (
        f'<div class="section">'
        f'<div class="sec-title">💰 Real-Money Performance — All Time</div>'
        f'<div class="stat-grid">'
        f'<div class="stat-card"><div class="lbl">REAL TRADES</div>'
        f'<div class="val">{rm["total_n"]:,}</div>'
        f'<div class="sub">{rm["wins"]}W / {rm["losses"]}L</div></div>'
        f'<div class="stat-card"><div class="lbl">REAL WIN RATE</div>'
        f'<div class="val" style="color:{"#00c87a" if rm["wr"]>=55 else "#ff3b3b"}">{rm["wr"]}%</div>'
        f'<div class="sub">live fills only</div></div>'
        f'<div class="stat-card"><div class="lbl">REAL PnL</div>'
        f'<div class="val" style="color:{rm_color}">{pnl_str(rm["total_pnl"])}</div>'
        f'<div class="sub">all sources combined</div></div>'
        f'</div>'
        f'{_chart_img("real_vs_paper", "Real-money PnL by source")}'
        f'<div class="tbl"><table>'
        f'<thead><tr><th>SOURCE</th><th>TRADES</th><th>WIN RATE</th><th>PnL</th></tr></thead>'
        f'<tbody>{rm_rows}</tbody>'
        f'</table></div>'
        f'</div>'
    )

    trends_html = (
        f'<div class="section">'
        f'<div class="sec-title">📈 Trends — Claude Picks</div>'
        f'{_chart_img("cumulative", "Cumulative PnL")}'
        f'{_chart_img("by_strategy", "PnL by strategy")}'
        f'{_chart_img("by_conviction", "Win rate by conviction")}'
        f'</div>'
    )

    # Flags
    live_badge = ('🟢 LIVE' if d['live_mode'] else '🔴 PAPER')
    def _flag_div(f):
        if f.startswith('🛑'):
            return (f'<div style="padding:10px 12px;background:rgba(255,59,59,.12);'
                    f'border-left:4px solid #ff3b3b;border-radius:0 6px 6px 0;'
                    f'margin-bottom:6px;font-size:13px;color:#fff;font-weight:600">{f}</div>')
        return (f'<div style="padding:6px 10px;background:rgba(245,197,24,.08);'
                f'border-left:3px solid #f5c518;border-radius:0 6px 6px 0;'
                f'margin-bottom:6px;font-size:11px;color:#ccc">{f}</div>')
    flags_html = ''.join(_flag_div(f) for f in d['flags']) \
                 or '<div style="color:#00c87a;font-size:11px">✅ No issues detected</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PolyQuant AI — Daily Report · {date_str}</title>
<style>
  :root {{
    --bg:#0a0a0f;--bg2:#111118;--bg3:#1a1a24;--border:rgba(255,255,255,.07);
    --text:#e2e2f0;--text2:#9090b0;--text3:#5a5a7a;
    --green:#00c87a;--red:#ff3b3b;--yellow:#f5c518;--cyan:#3b8ee0;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;line-height:1.6}}
  .page{{max-width:1100px;margin:0 auto;padding:32px 24px}}
  .header{{border-bottom:1px solid var(--border);padding-bottom:18px;margin-bottom:26px}}
  .header h1{{font-size:21px;font-weight:700;color:#fff}}
  .header .sub{{color:var(--text2);font-size:12px;margin-top:4px}}
  .section{{margin-bottom:32px}}
  .sec-title{{font-size:10px;font-weight:700;letter-spacing:.1em;color:var(--text3);text-transform:uppercase;
              margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:14px}}
  .stat-card{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:13px}}
  .stat-card .lbl{{font-size:9px;font-weight:700;letter-spacing:.08em;color:var(--text3);margin-bottom:3px}}
  .stat-card .val{{font-size:21px;font-weight:700;color:#fff}}
  .stat-card .sub{{font-size:10px;color:var(--text2);margin-top:2px}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
  .tbl{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:12px}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{font-size:9px;font-weight:700;letter-spacing:.07em;color:var(--text3);text-align:left;
      padding:6px 10px;border-bottom:1px solid var(--border)}}
  td{{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.03)}}
  tr:hover td{{background:rgba(255,255,255,.02)}}
  .green{{color:var(--green)}}
  .red{{color:var(--red)}}
  .footer{{border-top:1px solid var(--border);padding-top:14px;margin-top:28px;
           color:var(--text3);font-size:10px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <h1>PolyQuant AI — Daily Report</h1>
    <div class="sub">{now_str} &nbsp;·&nbsp; HFT: <strong style="color:{'#00c87a' if d['live_mode'] else '#ff3b3b'}">{live_badge}</strong> &nbsp;·&nbsp; Stake: ${d['stake']:.2f} &nbsp;·&nbsp; Drawdown floor: ${d['dd_floor']:.0f}</div>
  </div>

  <div class="section">
    <div class="sec-title">Flags &amp; Alerts</div>
    {flags_html}
  </div>

  {real_money_html}

  {trends_html}

  <div class="section">
    <div class="sec-title">HFT Performance — All Time</div>
    <div class="stat-grid">
      <div class="stat-card"><div class="lbl">TOTAL TRADES</div><div class="val">{hft['total']:,}</div><div class="sub">{hft['wins']}W / {hft['losses']}L</div></div>
      <div class="stat-card"><div class="lbl">WIN RATE</div><div class="val" style="color:#00c87a">{hft['wr']}%</div><div class="sub">threshold: 58%</div></div>
      <div class="stat-card"><div class="lbl">TOTAL PnL</div><div class="val" style="color:{'#00c87a' if hft['pnl']>=0 else '#ff3b3b'}">{pnl_str(hft['pnl'])}</div><div class="sub">${d['stake']:.2f}/trade</div></div>
      <div class="stat-card"><div class="lbl">HFT-RL STATES</div><div class="val style=" color:#3b8ee0">{len(d['hft_rl_states'])}</div><div class="sub">converged (n≥5)</div></div>
    </div>
    <div class="two-col">
      <div>
        <div style="font-size:10px;color:var(--text3);font-weight:700;margin-bottom:6px">BY STRATEGY</div>
        <div class="tbl"><table>
          <thead><tr><th>STRATEGY</th><th>WIN RATE</th><th>PnL</th><th>TRADES</th></tr></thead>
          <tbody>{strat_rows}</tbody>
        </table></div>
      </div>
      <div>
        <div style="font-size:10px;color:var(--text3);font-weight:700;margin-bottom:6px">BY SYMBOL</div>
        <div class="tbl"><table>
          <thead><tr><th>SYMBOL</th><th>WIN RATE</th><th>PnL</th><th>TRADES</th></tr></thead>
          <tbody>{sym_rows}</tbody>
        </table></div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">HFT Reinforcement Learning</div>
    <div class="two-col">
      <div>
        <div style="font-size:10px;color:#00c87a;font-weight:700;margin-bottom:6px">TOP APPROVED STATES</div>
        <div class="tbl"><table>
          <thead><tr><th>STATE</th><th>Q</th><th>N</th></tr></thead>
          <tbody>{rl_top}</tbody>
        </table></div>
      </div>
      <div>
        <div style="font-size:10px;color:#ff3b3b;font-weight:700;margin-bottom:6px">TOP BLOCKED STATES</div>
        <div class="tbl"><table>
          <thead><tr><th>STATE</th><th>Q</th><th>N</th></tr></thead>
          <tbody>{rl_bot}</tbody>
        </table></div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Manipulation RL</div>
    <div class="two-col">
      <div>
        <div style="font-size:10px;color:#00c87a;font-weight:700;margin-bottom:6px">STRONGEST APPROVED</div>
        <div class="tbl"><table>
          <thead><tr><th>STATE</th><th>Q</th><th>N</th></tr></thead>
          <tbody>{manip_top}</tbody>
        </table></div>
      </div>
      <div>
        <div style="font-size:10px;color:#ff3b3b;font-weight:700;margin-bottom:6px">HARD BLOCKED</div>
        <div class="tbl"><table>
          <thead><tr><th>STATE</th><th>Q</th><th>N</th></tr></thead>
          <tbody>{manip_bot}</tbody>
        </table></div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Claude Picks — Polymarket Strategies</div>
    <div class="stat-grid">
      <div class="stat-card"><div class="lbl">TOTAL TRADES</div><div class="val">{cl['total']:,}</div><div class="sub">{d['claude_pending']} pending</div></div>
      <div class="stat-card"><div class="lbl">WIN RATE</div><div class="val" style="color:{'#00c87a' if cl['wr']>=58 else '#f5c518'}">{cl['wr']}%</div><div class="sub">{cl['wins']}W / {cl['losses']}L</div></div>
      <div class="stat-card"><div class="lbl">TOTAL PnL</div><div class="val" style="color:{'#00c87a' if cl['pnl']>=0 else '#ff3b3b'}">{pnl_str(cl['pnl'])}</div><div class="sub">on ${d['claude_staked']:.2f} staked</div></div>
      <div class="stat-card"><div class="lbl">ROI</div><div class="val" style="color:{'#00c87a' if roi>=0 else '#ff3b3b'}">{roi:+.1f}%</div><div class="sub">return on staked</div></div>
    </div>
    <div class="two-col">
      <div>
        <div style="font-size:10px;color:var(--text3);font-weight:700;margin-bottom:6px">BY CONVICTION</div>
        <div class="tbl"><table>
          <thead><tr><th>LEVEL</th><th>WIN RATE</th><th>PnL</th><th>TRADES</th></tr></thead>
          <tbody>{conv_rows}</tbody>
        </table></div>
      </div>
      <div>
        <div style="font-size:10px;color:var(--text3);font-weight:700;margin-bottom:6px">BY STRATEGY TYPE</div>
        <div class="tbl"><table>
          <thead><tr><th>TYPE</th><th>WIN RATE</th><th>PnL</th><th>TRADES</th></tr></thead>
          <tbody>{type_rows}</tbody>
        </table></div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Signal Tab — Top Backtest Winners ({d['bt_days']}d run · {d['bt_ts']})</div>
    <div class="tbl"><table>
      <thead><tr><th>ALGO</th><th>SYMBOL</th><th>TF</th><th>WIN RATE</th><th>EV</th><th>SIGNALS</th><th>SIG/DAY</th></tr></thead>
      <tbody>{bt_rows}</tbody>
    </table></div>
  </div>

  {notes_html}

  <div class="footer">
    <span>PolyQuant AI · you@example.com · secc server · {date_str}</span>
    <span>Auto-generated at {d['now'].strftime('%H:%M')} by daily_report.py</span>
  </div>
</div>
</body>
</html>"""
    return html, date_str

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f'[REPORT] Starting daily report generation — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    d = gather()
    html, date_str = generate_html(d)

    # Save HTML
    out_path = os.path.join(REPORT_DIR, f'daily_report_{date_str}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'[REPORT] HTML saved -> {out_path}')

    # Send Telegram summary
    send_telegram(d)

    print('[REPORT] Done.')

if __name__ == '__main__':
    main()
