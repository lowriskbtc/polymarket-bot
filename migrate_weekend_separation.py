"""
Migrate trade-log files so weekend-window data lives ONLY in *.weekend.json
and weekday-window data lives ONLY in base *.json. Preserves all trades —
weekday entries currently stuck in a .weekend.json get MOVED to the base
file (not deleted). Pending (unresolved) trades stay in their current file
to avoid race conditions with resolution loops.

Two modes:
  python migrate_weekend_separation.py           # DRY RUN — shows what would change
  python migrate_weekend_separation.py --apply   # actually writes (with .bak)

Idempotent: re-running after apply is a no-op.
"""
import json, os, sys, io, shutil
from datetime import datetime, timezone
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.abspath(__file__))
APPLY = '--apply' in sys.argv
WEEK = 7 * 86400

cfg = json.load(open(os.path.join(BASE, 'weekend_mode.json')))
A_START = cfg['start_ts']; A_END = cfg['end_ts']; DUR = A_END - A_START
RECURRING = bool(cfg.get('recurring'))
if not cfg.get('enabled') or DUR <= 0:
    print('weekend_mode.json must be enabled with valid window'); sys.exit(1)

def trade_ts(t):
    raw = t.get('logged_at') or t.get('entered_at')
    if not raw: return None
    try:
        if isinstance(raw, str):
            return datetime.fromisoformat(raw.replace('Z', '+00:00')).timestamp()
        return float(raw)
    except Exception:
        return None

def is_weekend(t):
    """True/False per current weekend config, or None if no timestamp."""
    ts = trade_ts(t)
    if ts is None: return None
    if not RECURRING: return A_START <= ts < A_END
    if ts < A_START: return False
    return ((ts - A_START) % WEEK) < DUR

def trade_key(t):
    """Best-effort dedup key. Prefers _key, falls back to order_id, then
    (logged_at, symbol, direction)."""
    if t.get('_key'): return ('K', t['_key'])
    if t.get('order_id') and t['order_id'] != 'N/A':
        return ('O', t['order_id'])
    return ('T', t.get('logged_at',''), t.get('symbol',''), t.get('direction',''))

PAIRS = [
    ('hft_paper_trades.json',       'hft_paper_trades.weekend.json',       'HFT'),
    ('btc_ml_trades.json',          'btc_ml_trades.weekend.json',          'BTC-ML'),
    ('eth_ml_trades.json',          'eth_ml_trades.weekend.json',          'ETH-ML'),
    ('claude_paper_trades.json',    'claude_paper_trades.weekend.json',    'Claude'),
]

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def load_list(p):
    if not os.path.exists(p): return []
    try:
        d = json.load(open(p, 'r'))
        return d if isinstance(d, list) else d.get('trades', [])
    except Exception as e:
        print(f'  ! parse error on {p}: {e}')
        return []

def save_list(p, lst):
    """Match existing bot save patterns: no indent, default=str, .tmp + os.replace."""
    import uuid
    tmp = p + f'.tmp_migrate_{uuid.uuid4().hex[:8]}'
    with open(tmp, 'w') as f:
        json.dump(lst, f, default=str)
    os.replace(tmp, p)

mode_label = 'APPLY' if APPLY else 'DRY-RUN'
print(f'=== migrate_weekend_separation [{mode_label}] ===')
print(f'window cfg: start={datetime.fromtimestamp(A_START,timezone.utc).isoformat()} '
      f'end={datetime.fromtimestamp(A_END,timezone.utc).isoformat()} '
      f'recurring={RECURRING}\n')

for base_fn, wkd_fn, label in PAIRS:
    base_p = os.path.join(BASE, base_fn)
    wkd_p  = os.path.join(BASE, wkd_fn)
    print(f'--- {label} ({base_fn} ↔ {wkd_fn}) ---')

    base_lst = load_list(base_p)
    wkd_lst  = load_list(wkd_p)
    print(f'  before: base={len(base_lst)}  weekend={len(wkd_lst)}')

    # Bucket every trade. Pending stays in its source file. Resolved goes
    # to whichever file its mode-window says.
    new_base = []
    new_wkd  = []
    moved_to_base = 0   # weekday trades moved out of weekend file
    moved_to_wkd  = 0   # weekend trades moved out of base file
    base_already = set()
    wkd_already  = set()

    # First pass: keep pending in their current file (no relocation)
    for t in base_lst:
        if not t.get('resolved'):
            new_base.append(t); base_already.add(trade_key(t))
    for t in wkd_lst:
        if not t.get('resolved'):
            new_wkd.append(t); wkd_already.add(trade_key(t))

    # Second pass: resolved trades, route by window
    # Merge both lists, dedup by key, then route
    merged_resolved = {}  # key -> trade
    src = {}              # key -> 'base'|'weekend' (where it came from)
    for t in base_lst:
        if t.get('resolved'):
            k = trade_key(t)
            if k not in merged_resolved:
                merged_resolved[k] = t; src[k] = 'base'
    for t in wkd_lst:
        if t.get('resolved'):
            k = trade_key(t)
            if k not in merged_resolved:
                merged_resolved[k] = t; src[k] = 'weekend'
            else:
                # If both have it, prefer the version with more fields (e.g. real_pnl backfill)
                existing = merged_resolved[k]
                if (t.get('actual_fill_price') and not existing.get('actual_fill_price')) \
                   or (t.get('real_pnl') is not None and existing.get('real_pnl') is None):
                    merged_resolved[k] = t; src[k] = 'weekend'

    no_ts_count = 0
    for k, t in merged_resolved.items():
        v = is_weekend(t)
        if v is None:
            # No timestamp — keep in its source file, don't move
            no_ts_count += 1
            if src[k] == 'weekend':
                if k not in wkd_already:
                    new_wkd.append(t); wkd_already.add(k)
            else:
                if k not in base_already:
                    new_base.append(t); base_already.add(k)
        elif v:
            # Weekend-window trade → weekend file
            if k not in wkd_already:
                new_wkd.append(t); wkd_already.add(k)
            if src[k] == 'base':
                moved_to_wkd += 1
        else:
            # Weekday-window trade → base file
            if k not in base_already:
                new_base.append(t); base_already.add(k)
            if src[k] == 'weekend':
                moved_to_base += 1

    # Sort by logged_at for sanity (bots also re-sort; this just makes diffs readable)
    new_base.sort(key=lambda t: t.get('logged_at','') or '')
    new_wkd.sort(key=lambda t: t.get('logged_at','') or '')

    print(f'  after:  base={len(new_base)}  weekend={len(new_wkd)}')
    print(f'  moved weekday→base: {moved_to_base}    moved weekend→base file: {moved_to_wkd}')
    if no_ts_count:
        print(f'  ⚠ {no_ts_count} resolved trades had no timestamp — left in source file')

    if APPLY:
        if base_lst:
            bak = f'{base_p}.bak_premig_{stamp}'
            shutil.copy2(base_p, bak); print(f'  backed up base   → {os.path.basename(bak)}')
        if wkd_lst:
            bak = f'{wkd_p}.bak_premig_{stamp}'
            shutil.copy2(wkd_p, bak); print(f'  backed up weekend→ {os.path.basename(bak)}')
        save_list(base_p, new_base); print(f'  wrote {base_fn}')
        save_list(wkd_p,  new_wkd);  print(f'  wrote {wkd_fn}')
    print()

print('Done.' if APPLY else 'Dry run — nothing written. Re-run with --apply to commit.')
