#!/usr/bin/env python
"""
HONEST SCORECARD (read-only) — run this first when you return.

One command -> the honest state of every system, computed from confirmed real_pnl
(never paper proxies), with trailing windows so you can't be fooled by a windfall.
Writes a dated report to scorecard_YYYY-MM-DD.txt and prints it.

Robust to weekend mode: for each logical trade log it auto-selects the LIVE file
(weekday X.json vs weekend X.weekend.json) by most-recent modification time, so it
works whichever mode is active when you run it.

Then it shells out to edge_hunt.py / honest_paper.py for the deep views.
"""
import json, os, glob, statistics, subprocess, sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))

LOGICAL = {
    'CP':     'claude_paper_trades',
    'HFT':    'hft_paper_trades',
    'BTC-ML': 'btc_ml_trades',
    'ETH-ML': 'eth_ml_trades',
    'HERMES': 'btc_hermes_trades',
}

def resolve_live(stem):
    """Pick the live file between <stem>.json and <stem>.weekend.json by mtime."""
    cands = [os.path.join(BASE, f'{stem}.json'), os.path.join(BASE, f'{stem}.weekend.json')]
    cands = [c for c in cands if os.path.exists(c)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)

def parse(ts):
    try:
        t = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def load(path):
    try:
        d = json.load(open(path, encoding='utf-8'))
        return d if isinstance(d, list) else []
    except Exception as e:
        return []

def tstat(vals):
    n = len(vals)
    if n < 2:
        return (n, sum(vals) if vals else 0.0, 0.0, 0.0)
    m = sum(vals)/n
    sd = statistics.pstdev(vals)
    return (n, sum(vals), m, (m/(sd/n**0.5) if sd else 0.0))

def fmt(label, vals):
    n, s, m, t = tstat(vals)
    if n == 0:
        return f'    {label:14} n=   0'
    flag = 'ROBUST' if abs(t) >= 2 else 'lean' if abs(t) >= 1 else 'noise'
    return f'    {label:14} n={n:4d}  sum={s:+9.2f}  avg={m:+.3f}  t={t:+.2f} [{flag}]'

def main():
    now = datetime.now(timezone.utc)
    def days_ago(r):
        t = parse(r.get('logged_at'))
        return 9e9 if t is None else (now - t).total_seconds()/86400

    out = []
    def p(s=''):
        out.append(s); print(s)

    p('='*72)
    p(f'HONEST SCORECARD  {now.isoformat()}')
    p('Confirmed real_pnl only (real_trade=True + real_pnl set = real FILLS).')
    p('='*72)

    for name, stem in LOGICAL.items():
        path = resolve_live(stem)
        if not path:
            p(f'\n{name}: no trade file found'); continue
        d = load(path)
        real = [r for r in d if r.get('real_trade') and r.get('real_pnl') is not None]
        mode = 'WEEKEND' if path.endswith('.weekend.json') else 'weekday'
        p(f'\n{name}  ({os.path.basename(path)}, {mode})  total={len(d)} real_w_pnl={len(real)}')
        if not real:
            p('    (no confirmed real fills — paper-only or pre-fill-logging)')
            continue
        for lab, lo, hi in [('real 0-7d', 0, 7), ('real 7-14d', 7, 14),
                            ('real 14-30d', 14, 30), ('real lifetime', 0, 9e9)]:
            p(fmt(lab, [r['real_pnl'] for r in real if lo <= days_ago(r) < hi]))
        # paper inflation flag
        att = [r for r in d if (r.get('attempted_live') or r.get('fill_confirmed') is not None)]
        if att:
            fr = sum(1 for r in att if r.get('fill_confirmed') is True)/len(att)
            ppa = sum(r.get('paper_pnl') or 0 for r in att)
            ppf = sum(r.get('paper_pnl') or 0 for r in att if r.get('fill_confirmed') is True)
            p(f'    fill_rate={fr:.0%}  paper(all)={ppa:+.0f} vs paper(filled)={ppf:+.0f} '
              f'-> {"TRUST filled-basis only" if fr<0.9 else "paper~=filled (usable)"}')

    p('\n' + '='*72)
    p('RULE: act only on cohorts robust in the TRAILING (0-7/0-14d) window, never')
    p('lifetime. Lifetime t-stats are windfall magnets. Use edge_hunt.py to confirm.')
    p('='*72)

    # write report
    stamp = now.strftime('%Y-%m-%d')
    rpt = os.path.join(BASE, f'scorecard_{stamp}.txt')
    try:
        with open(rpt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out) + '\n')
        print(f'\n[written] {rpt}')
    except Exception as e:
        print(f'[warn] could not write report: {e}')

    # deep views (best-effort)
    for tool in ('honest_paper.py', 'edge_hunt.py'):
        tp = os.path.join(BASE, tool)
        if os.path.exists(tp):
            print(f'\n----- {tool} -----')
            try:
                r = subprocess.run([sys.executable, tp], capture_output=True, text=True, timeout=120)
                print(r.stdout[-4000:] if r.stdout else '(no output)')
            except Exception as e:
                print(f'(could not run {tool}: {e})')

if __name__ == '__main__':
    main()
