"""btc_hermes_nightly.py — daily 04:17 UTC self-learn loop for BTC-HERMES.

Three phases, each independent (failures don't block downstream):

  Phase 3a — Markov matrix retrain (deterministic, lightweight, no model).
             Paginated Binance fetch -> transition probabilities -> matrix file.
             Always runs. Bot picks up new matrix on next 75-scan reload tick.

  Phase 3b — Hermes parameter proposal (heavy, requires HuggingFace model).
             Loads NousResearch Hermes weights, reads recent trade journal,
             proposes new MIN_PROB / MIN_EDGE / Kelly fraction as JSON. Skipped
             cleanly if model fails to load (no params change).

  Phase 3c — Sanity clamp + write-back (mandatory safety).
             Hard-clamps proposed values to safe ranges, backs up the settings
             file, writes via _atomic_write_json, journals dual-write, TG alerts.

Env vars:
  HF_HOME             — HuggingFace cache root (e.g. D:\\hermes_models). Default:
                        whatever HuggingFace's default is (typically ~/.cache/huggingface).
  HERMES_MODEL_ID     — HuggingFace model id. Default: NousResearch/Hermes-3-Llama-3.2-3B
                        (smaller — fits comfortably on CPU). Set to a larger
                        variant if you have a GPU + the bandwidth.
  HERMES_SKIP_MODEL   — if "1", skip Phase 3b entirely (matrix only).
  HERMES_WINDOW_DAYS  — override matrix window. Default: read from settings file (7).

Standalone CLI:
  python btc_hermes_nightly.py                # run full pipeline
  python btc_hermes_nightly.py --matrix-only  # Phase 3a only
  python btc_hermes_nightly.py --dry-run      # everything except settings write
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

SETTINGS_FILE  = ROOT / 'btc_hermes_settings.json'
MATRIX_FILE    = ROOT / 'btc_hermes_matrix.json'
TRADES_FILE    = ROOT / 'btc_hermes_trades.json'
TRADES_WEEKEND = ROOT / 'btc_hermes_trades.weekend.json'
JOURNAL        = ROOT / 'btc_hermes_journal.json'
JOURNAL_FULL   = ROOT / 'btc_hermes_journal_full.json'
LAST_RUN       = ROOT / 'btc_hermes_nightly_lastrun.json'
BACKUPS        = ROOT / '_backups'

# ── weekend-mode awareness ────────────────────────────────────────────────────
# The live scanner reads *.weekend.json state in weekend mode (main bot's
# _resolve_state_path). So on weekends the nightly must tune the WEEKEND settings
# file, else weekend Hermes never self-learns. Mirrors weekend_mode.json's
# recurring-window logic. MATRIX stays the base file (global market structure;
# it is not weekday/weekend-split and is vestigial in straddle mode anyway).
_WEEK_SECONDS = 7 * 86400
def _is_weekend_now() -> bool:
    try:
        cfg = json.loads((ROOT / 'weekend_mode.json').read_text(encoding='utf-8')) or {}
    except Exception:
        return False
    if not cfg.get('enabled'):
        return False
    a = cfg.get('start_ts', 0) or 0
    b = cfg.get('end_ts', 0) or 0
    dur = b - a
    if dur <= 0 or a <= 0:
        return False
    now = time.time()
    if not cfg.get('recurring'):
        return a <= now < b
    if now < a:
        return False
    return ((now - a) % _WEEK_SECONDS) < dur

WEEKEND_ACTIVE = _is_weekend_now()
if WEEKEND_ACTIVE:
    SETTINGS_FILE = ROOT / 'btc_hermes_settings.weekend.json'  # tune the file the live scanner actually reads

# Safe ranges for Hermes's proposals — values outside these get hard-clamped.
# Straddle tunables (sell_threshold, price_balance_max_dev) actually drive the
# bot's behavior; min_prob/min_edge/kelly_fraction are vestigial (kept for back
# compat — straddle mode does not gate on them).
SAFE_RANGES = {
    'min_prob':                (0.00, 0.95),
    'min_edge':                (0.02, 0.20),
    'kelly_fraction':          (0.05, 0.50),
    'sell_threshold':          (0.02, 0.20),
    'price_balance_max_dev':   (0.05, 0.40),
    'cut_loss_threshold':      (0.10, 0.80),
    're_entry_threshold':      (0.02, 0.20),
    'binance_winner_threshold':(0.001, 0.010),   # 0.1% to 1.0% BTC move
    'binance_loser_threshold': (0.002, 0.020),   # 0.2% to 2.0% BTC move
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def log(msg):
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    print(f'[{ts}] {msg}', flush=True)

def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        return default

def write_lastrun(status, **extra):
    """Heartbeat for system_monitor.py's nightly liveness check."""
    payload = {
        'ts': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'status': status,
        **extra,
    }
    try:
        LAST_RUN.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass

def send_tg(text):
    """Best-effort Telegram via existing tg_reply.py."""
    try:
        subprocess.run([sys.executable, 'tg_reply.py', text],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=15)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3a — Markov matrix retrain
# ──────────────────────────────────────────────────────────────────────────────

def fetch_binance_5m_candles(window_days: int) -> list:
    """Paginated Binance kline fetch. Returns list of (open, close, openTime_ms).
    1000 candles per call (8.3h); need ~5 calls for 7d, ~22 calls for 30d."""
    needed = int(window_days * 24 * 12)  # 288 candles/day
    log(f'Phase 3a: need ~{needed:,} 5m candles for window_days={window_days}')
    out = []
    end_time = None  # None = current; subsequent calls set to oldest fetched - 1
    calls = 0
    while len(out) < needed and calls < 30:  # safety cap
        params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': '1000'}
        if end_time is not None:
            params['endTime'] = str(end_time)
        url = 'https://api.binance.com/api/v3/klines?' + '&'.join(f'{k}={v}' for k,v in params.items())
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                batch = json.loads(r.read())
        except Exception as e:
            log(f'  Binance fetch err on call {calls+1}: {e}')
            break
        if not batch:
            break
        # batch is oldest-first; each item: [openTime, open, high, low, close, vol, closeTime, ...]
        for k in batch:
            out.append((int(k[0]), float(k[1]), float(k[4])))  # (openTime_ms, open, close)
        oldest_ms = batch[0][0]
        end_time = oldest_ms - 1  # next call goes further back
        calls += 1
        log(f'  call {calls}: fetched {len(batch)} candles, total={len(out)}, oldest_ts={oldest_ms}')
        time.sleep(0.2)  # be polite
    # Dedupe by openTime, sort oldest -> newest, keep most recent `needed`
    seen = {}
    for k in out:
        seen[k[0]] = k
    sorted_candles = sorted(seen.values(), key=lambda x: x[0])
    return sorted_candles[-needed:]

def compute_markov_matrix(candles: list) -> dict:
    """Count transitions from candle direction sequence.
    Returns dict with p_uu, p_dd, n_samples, binance_first_ts, binance_last_ts.
    Laplace smoothing (+1) prevents p=1.0 extremes."""
    if len(candles) < 2:
        raise ValueError(f'Need >=2 candles, got {len(candles)}')
    # Direction of each candle: UP if close > open, else DOWN
    dirs = ['UP' if c[2] > c[1] else 'DOWN' for c in candles]
    n_uu = n_ud = n_du = n_dd = 0
    for i in range(1, len(dirs)):
        prev, curr = dirs[i-1], dirs[i]
        if   prev == 'UP'   and curr == 'UP':   n_uu += 1
        elif prev == 'UP'   and curr == 'DOWN': n_ud += 1
        elif prev == 'DOWN' and curr == 'UP':   n_du += 1
        elif prev == 'DOWN' and curr == 'DOWN': n_dd += 1
    # Laplace smoothing
    p_uu = (n_uu + 1) / (n_uu + n_ud + 2)
    p_dd = (n_dd + 1) / (n_du + n_dd + 2)
    return {
        'p_uu':             round(p_uu, 6),
        'p_dd':             round(p_dd, 6),
        'n_samples':        len(candles) - 1,
        'n_uu': n_uu, 'n_ud': n_ud, 'n_du': n_du, 'n_dd': n_dd,
        'binance_first_ts': candles[0][0]  // 1000,
        'binance_last_ts':  candles[-1][0] // 1000,
    }

def phase_3a_matrix_retrain(window_days: int):
    """Returns the matrix dict (also written to MATRIX_FILE)."""
    candles = fetch_binance_5m_candles(window_days)
    if not candles:
        raise RuntimeError('No candles fetched')
    matrix = compute_markov_matrix(candles)
    matrix['window_days']     = window_days
    matrix['last_refit_ts']   = dt.datetime.now(dt.timezone.utc).isoformat()
    # Backup existing, then atomic write
    if MATRIX_FILE.exists():
        BACKUPS.mkdir(exist_ok=True)
        shutil.copy(MATRIX_FILE, BACKUPS / f'btc_hermes_matrix.bak_{int(time.time())}.json')
    # Atomic write (simple replace — small file)
    tmp = MATRIX_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(matrix, indent=2))
    tmp.replace(MATRIX_FILE)
    log(f'Phase 3a OK: p_uu={matrix["p_uu"]} p_dd={matrix["p_dd"]} '
        f'n_samples={matrix["n_samples"]} '
        f'(n_uu={matrix["n_uu"]} n_ud={matrix["n_ud"]} '
        f'n_du={matrix["n_du"]} n_dd={matrix["n_dd"]})')
    return matrix

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3b — Hermes parameter proposal
# ──────────────────────────────────────────────────────────────────────────────

def compute_trade_stats(trades: list, since_ts: float) -> dict:
    """Performance stats over trades resolved since `since_ts` (POSIX seconds)."""
    def parse_iso(s):
        try: return dt.datetime.fromisoformat(str(s).replace('Z','+00:00')).timestamp()
        except: return None
    resolved = []
    for t in trades or []:
        if not t.get('resolved'): continue
        # Path B7: use exit_ts (populated on every paper close) instead of
        # resolved_at, which historically wasn't always written and left the
        # nightly LLM seeing n_resolved=0.
        rt = parse_iso(t.get('exit_ts'))
        if rt is None or rt < since_ts: continue
        resolved.append(t)
    if not resolved:
        return {'n_resolved': 0}
    wins  = [t for t in resolved if t.get('outcome') == 'WIN']
    losses = [t for t in resolved if t.get('outcome') == 'LOSS']
    pnl   = sum((t.get('paper_pnl') or 0) for t in resolved)
    stakes = sum((t.get('stake') or 0) for t in resolved)
    avg_p = sum((t.get('markov_p') or 0) for t in resolved) / len(resolved)
    avg_q = sum((t.get('market_q') or 0) for t in resolved) / len(resolved)
    avg_edge = sum((t.get('edge') or 0) for t in resolved) / len(resolved)
    # Continuation rate: did next 5m candle match the prediction?
    # (proxy — using outcome since paper resolves on actual market outcome)
    realized_continuation_rate = 100 * len(wins) / max(len(wins) + len(losses), 1)
    return {
        'n_resolved':                 len(resolved),
        'wins':                       len(wins),
        'losses':                     len(losses),
        'win_rate_pct':               round(realized_continuation_rate, 2),
        'total_paper_pnl':            round(pnl, 2),
        'total_stake':                round(stakes, 2),
        'roi_pct':                    round(100 * pnl / stakes, 2) if stakes > 0 else 0.0,
        'avg_predicted_p':            round(avg_p, 4),
        'avg_market_q':               round(avg_q, 4),
        'avg_predicted_edge':         round(avg_edge, 4),
        'calibration_gap_pp':         round(100 * (avg_p - realized_continuation_rate / 100), 2),
    }

def load_hermes_model(model_id: str):
    """Lazy-load Hermes weights. Tries 4-bit -> 8-bit -> fp16/fp32 in sequence.
    Returns (model, tokenizer) or raises on total failure."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log(f'Loading Hermes model: {model_id}')
    log(f'  HF_HOME={os.environ.get("HF_HOME", "<default>")}')
    tok = AutoTokenizer.from_pretrained(model_id)
    # Try 4-bit (needs bitsandbytes) -> 8-bit -> auto-dtype/CPU
    last_err = None
    try:
        import torch
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map='auto')
        log('  Loaded in 4-bit')
        return model, tok
    except Exception as e:
        last_err = e
        log(f'  4-bit load failed: {type(e).__name__}: {e}')
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map='auto')
        log('  Loaded in 8-bit')
        return model, tok
    except Exception as e:
        last_err = e
        log(f'  8-bit load failed: {type(e).__name__}: {e}')
    try:
        import torch
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map='auto')
        log('  Loaded in auto-dtype (likely fp16/fp32 on CPU or GPU)')
        return model, tok
    except Exception as e:
        last_err = e
        log(f'  auto-dtype load failed: {type(e).__name__}: {e}')
    raise RuntimeError(f'All Hermes load strategies failed; last err: {last_err}')

def build_messages(matrix: dict, current_params: dict, stats_24h: dict, stats_7d: dict) -> list:
    """Returns a chat-template-ready messages list. System message anchors the
    output format ('respond ONLY with JSON, no code blocks, no prose'). User
    message has all the trade context and what we're asking."""
    system_msg = (
        "You are a JSON-only parameter tuning assistant for a Polymarket trading bot. "
        "You MUST respond with exactly one JSON object and nothing else. "
        "Do not write Python code, do not use code blocks (no ```), do not write any "
        "explanation outside the JSON. Just the raw JSON object."
    )

    user_msg = f"""Strategy: BTC 5-minute Polymarket STRADDLE.
- At new 5m market open: buy equal stake on BOTH UP and DOWN tokens.
- Winning side: SELL when bid >= entry * (1 + SELL_THRESHOLD).
- Losing side: hold to market resolution (collects $0 if loses, $1/share if reverses).
- Profitable on whippy reversals; bleeds on consistent directional moves.

The FOUR knobs that ACTUALLY drive behavior:
  SELL_THRESHOLD: +X% trigger for winning-side sell. Lower (~0.03) = more frequent
    tiny sells. Higher (~0.10) = fewer larger sells, risk reversal before hit.
  PRICE_BALANCE_MAX_DEV: refuse straddle if up_q outside [0.5-dev, 0.5+dev].
    Tighter (0.10) = only pristine 50/50 markets. Wider (0.30) = enter more, often imbalanced.
  CUT_LOSS_THRESHOLD: sell losing side when bid drops to entry*(1-this). Lower
    (~0.20) = cut losses fast (smaller per-pair loss, more activity). Higher
    (~0.50) = let losers run, hope for reversal.
  RE_ENTRY_THRESHOLD: after a side sells, re-buy if ask drops to entry*(1-this).
    Enables continuous scalping. Lower (~0.03) = aggressive re-entry on small dips.

Vestigial knobs (not used by straddle; tune conservatively or leave alone):
  MIN_PROB, MIN_EDGE, KELLY_FRACTION — kept for backwards compatibility.

Current parameters:
  SELL_THRESHOLD         = {current_params.get('sell_threshold', 0.05):.4f}
  PRICE_BALANCE_MAX_DEV  = {current_params.get('price_balance_max_dev', 0.20):.4f}
  CUT_LOSS_THRESHOLD     = {current_params.get('cut_loss_threshold', 0.30):.4f}
  RE_ENTRY_THRESHOLD     = {current_params.get('re_entry_threshold', 0.05):.4f}
  MIN_PROB               = {current_params.get('min_prob', 0.0):.4f}
  MIN_EDGE               = {current_params.get('min_edge', 0.05):.4f}
  KELLY_FRACTION         = {current_params.get('kelly_fraction', 0.25):.4f}

Markov matrix (informational only; straddle doesn't use it):
  p(UP   -> UP)   = {matrix.get('p_uu', 0.5):.4f}
  p(DOWN -> DOWN) = {matrix.get('p_dd', 0.5):.4f}
  n_samples       = {matrix.get('n_samples', 0)}

Trade performance last 24h:
{json.dumps(stats_24h, indent=2)}

Trade performance last 7d:
{json.dumps(stats_7d, indent=2)}

Propose new values. Bounds (you MUST stay within):
  sell_threshold            in [0.02, 0.20]
  price_balance_max_dev     in [0.05, 0.40]
  cut_loss_threshold        in [0.10, 0.80]
  re_entry_threshold        in [0.02, 0.20]
  binance_winner_threshold  in [0.001, 0.010]   (% BTC move to trigger lag-arb winner sell)
  binance_loser_threshold   in [0.002, 0.020]   (% BTC move to trigger lag-arb loser cut)
  min_prob                  in [0.00, 0.95]
  min_edge                  in [0.02, 0.20]
  kelly_fraction            in [0.05, 0.50]

Guidance:
- If roi_pct < 0 AND n_resolved >= 10: tighten the active knobs. Either lower
  sell_threshold (sell sooner before losses pile up) OR tighten price_balance_max_dev
  (skip more imbalanced markets, fewer bad entries).
- If roi_pct > 0 AND n_resolved >= 10: small relaxation. Raise sell_threshold
  slightly to capture bigger wins, or widen price_balance_max_dev slightly.
- If n_resolved < 10: keep ALL parameters unchanged. Not enough signal.
- For min_prob/min_edge/kelly_fraction: only adjust if explicitly justified by
  the data. Default is to leave them at current values.
- Your rationale: one sentence, <= 200 chars, citing the specific stat that motivated the change.

Output exactly this JSON shape (no extra fields, no code blocks, no prose):
{{"sell_threshold": 0.05, "price_balance_max_dev": 0.20, "cut_loss_threshold": 0.30, "re_entry_threshold": 0.05, "min_prob": 0.0, "min_edge": 0.05, "kelly_fraction": 0.25, "rationale": "your one-sentence reason"}}"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

# Back-compat wrapper — old callers may have referenced build_prompt; keep them working.
def build_prompt(matrix, current_params, stats_24h, stats_7d):
    msgs = build_messages(matrix, current_params, stats_24h, stats_7d)
    return f"{msgs[0]['content']}\n\n{msgs[1]['content']}"

def phase_3b_hermes_propose(matrix: dict, current_params: dict) -> dict | None:
    """Returns proposed params dict, or None if Hermes step skipped/failed."""
    if os.environ.get('HERMES_SKIP_MODEL') == '1':
        log('Phase 3b: HERMES_SKIP_MODEL=1, skipping')
        return None
    model_id = os.environ.get('HERMES_MODEL_ID', 'NousResearch/Hermes-3-Llama-3.2-3B')
    try:
        model, tok = load_hermes_model(model_id)
    except Exception as e:
        log(f'Phase 3b SKIPPED: model load failed: {e}')
        send_tg(f'[BTC-HERMES] nightly: model load failed ({type(e).__name__}). Matrix updated; params unchanged.')
        return None

    # Build chat messages + apply tokenizer's chat template (Llama 3 style for Hermes-3)
    now_ts = time.time()
    trades = (load_json(TRADES_FILE, []) or []) + (load_json(TRADES_WEEKEND, []) or [])
    stats_24h = compute_trade_stats(trades, now_ts - 86400)
    stats_7d  = compute_trade_stats(trades, now_ts - 7*86400)
    messages = build_messages(matrix, current_params, stats_24h, stats_7d)

    try:
        import torch
        # apply_chat_template handles Hermes-3-Llama-3.2 prompt structure correctly.
        # add_generation_prompt=True appends the assistant header so model writes the response.
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        log(f'Phase 3b: chat-templated prompt {len(prompt_text)} chars, generating...')
        inputs = tok(prompt_text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        # Greedy decode with low max tokens — JSON proposal is short
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        # Decode ONLY the newly-generated tokens (skip the prompt)
        response = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        log(f'Phase 3b: raw response = {response[:400]}')
    except Exception as e:
        log(f'Phase 3b generate failed: {e}')
        return None
    finally:
        try:
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Lenient JSON extraction. Tries in order:
    #  (a) strip markdown fences if present (Hermes-3 sometimes wraps in ```json ... ```)
    #  (b) find balanced {...} block via regex
    #  (c) if first attempt fails, try greedy match (in case there's nested braces)
    import re as _re
    cleaned = response
    # Strip code fences (```json or ``` or ```python etc.)
    cleaned = _re.sub(r'^```(?:json|python)?\s*\n?', '', cleaned, flags=_re.MULTILINE)
    cleaned = _re.sub(r'\n?```\s*$', '', cleaned, flags=_re.MULTILINE)
    # Find JSON object. Prefer the LAST balanced object (model sometimes writes
    # "here is the JSON: {...}").
    candidates = []
    for m in _re.finditer(r'\{[^{}]*\}', cleaned, flags=_re.DOTALL):
        candidates.append(m.group(0))
    # Greedy fallback in case keys nested (rare for our schema)
    if not candidates:
        m = _re.search(r'\{.*\}', cleaned, flags=_re.DOTALL)
        if m:
            candidates = [m.group(0)]
    if not candidates:
        log(f'Phase 3b: no JSON object found in response (cleaned={cleaned[:200]!r})')
        return None
    # Try each candidate (last first) until one parses
    for cand in reversed(candidates):
        try:
            proposed = json.loads(cand)
            # Sanity: must have at least one expected key
            if any(k in proposed for k in ('min_prob', 'min_edge', 'kelly_fraction')):
                log(f'Phase 3b OK: proposed={proposed}')
                return proposed
        except Exception:
            continue
    log(f'Phase 3b: JSON candidates found but none parsed successfully (first={candidates[0][:200]!r})')
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3c — Sanity clamp + write-back + journal + TG
# ──────────────────────────────────────────────────────────────────────────────

def clamp_proposal(proposed: dict, current: dict) -> tuple[dict, list]:
    """Hard-clamp proposed values into SAFE_RANGES. Returns (final, clamped_keys)."""
    final = dict(current)
    clamped = []
    for key, (lo, hi) in SAFE_RANGES.items():
        if key not in proposed:
            continue
        try:
            v = float(proposed[key])
        except (TypeError, ValueError):
            continue
        v_clamped = max(lo, min(hi, v))
        if abs(v_clamped - v) > 1e-9:
            clamped.append(f'{key}: {v} -> {v_clamped} (out of [{lo}, {hi}])')
        final[key] = round(v_clamped, 4)
    return final, clamped

def write_settings(new_settings: dict):
    """Backup + atomic write of btc_hermes_settings.json."""
    BACKUPS.mkdir(exist_ok=True)
    if SETTINGS_FILE.exists():
        shutil.copy(SETTINGS_FILE, BACKUPS / f'btc_hermes_settings.bak_{int(time.time())}.json')
    tmp = SETTINGS_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(new_settings, indent=2))
    tmp.replace(SETTINGS_FILE)

def append_journal(entry: dict):
    """Dual-write compact + full archive (mirrors daily_notes pattern)."""
    for path in (JOURNAL, JOURNAL_FULL):
        existing = load_json(path, [])
        nxt_idx = max((e.get('archive_idx', 0) for e in existing), default=0) + 1
        e = dict(entry)
        if path == JOURNAL_FULL:
            e['archive_idx'] = nxt_idx
        existing.append(e)
        try:
            path.write_text(json.dumps(existing, indent=2))
        except Exception as ex:
            log(f'journal write err {path.name}: {ex}')

# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run(matrix_only: bool = False, dry_run: bool = False):
    log('=' * 60)
    log(f'BTC-HERMES nightly start (matrix_only={matrix_only} dry_run={dry_run})')
    log(f'settings target: {SETTINGS_FILE.name} (weekend_active={WEEKEND_ACTIVE})')
    log('=' * 60)

    # Load current settings to get window_days + current params
    settings = load_json(SETTINGS_FILE, {}) or {}
    window_days = int(os.environ.get('HERMES_WINDOW_DAYS', settings.get('matrix_window_days', 7)))
    window_days = max(3, min(30, window_days))
    current_params = {
        'sell_threshold':        float(settings.get('sell_threshold', 0.05)),
        'price_balance_max_dev': float(settings.get('price_balance_max_dev', 0.20)),
        'cut_loss_threshold':    float(settings.get('cut_loss_threshold', 0.30)),
        're_entry_threshold':    float(settings.get('re_entry_threshold', 0.05)),
        'binance_winner_threshold': float(settings.get('binance_winner_threshold', 0.002)),
        'binance_loser_threshold':  float(settings.get('binance_loser_threshold', 0.005)),
        'min_prob':              float(settings.get('min_prob', 0.0)),
        'min_edge':              float(settings.get('min_edge', 0.05)),
        'kelly_fraction':        float(settings.get('kelly_fraction', 0.25)),
    }

    # Phase 3a — always runs
    try:
        matrix = phase_3a_matrix_retrain(window_days)
    except Exception as e:
        log(f'Phase 3a FAILED: {e}')
        write_lastrun('error_3a', error=str(e))
        send_tg(f'[BTC-HERMES] nightly: Phase 3a (matrix) FAILED: {e}')
        sys.exit(2)

    if matrix_only:
        write_lastrun('ok_matrix_only', matrix={'p_uu': matrix['p_uu'], 'p_dd': matrix['p_dd'], 'n': matrix['n_samples']})
        log('matrix-only mode — done')
        send_tg(f'[BTC-HERMES] matrix refit: p_uu={matrix["p_uu"]:.3f} p_dd={matrix["p_dd"]:.3f} n={matrix["n_samples"]}')
        return

    # Phase 3b — best-effort
    proposed = phase_3b_hermes_propose(matrix, current_params)

    # Phase 3c — write-back even if Hermes proposed nothing (just journal)
    if proposed is None:
        append_journal({
            'ts': dt.datetime.now(dt.timezone.utc).isoformat(),
            'matrix_post_refit': matrix,
            'current_params':    current_params,
            'proposed_params':   None,
            'note':              'Hermes step skipped or failed; matrix updated; params unchanged.',
        })
        write_lastrun('ok_matrix_only_hermes_skipped',
                      matrix={'p_uu': matrix['p_uu'], 'p_dd': matrix['p_dd'], 'n': matrix['n_samples']})
        send_tg(f'[BTC-HERMES] matrix refit OK (p_uu={matrix["p_uu"]:.3f} p_dd={matrix["p_dd"]:.3f} n={matrix["n_samples"]}). Hermes step skipped; params unchanged.')
        return

    # Clamp + merge
    new_params, clamped = clamp_proposal(proposed, current_params)
    log(f'Final params: {new_params}')
    if clamped:
        log(f'CLAMPED: {clamped}')

    if dry_run:
        log('dry-run mode: not writing settings')
        write_lastrun('ok_dryrun', proposed=proposed, final=new_params)
        return

    # Merge into full settings (preserve live_mode, stake, tp_sl_enabled, etc.)
    new_settings = dict(settings)
    new_settings.update(new_params)
    new_settings['matrix_window_days'] = window_days
    new_settings['hermes_last_proposal_ts'] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_settings(new_settings)

    # Journal entry
    append_journal({
        'ts':                 dt.datetime.now(dt.timezone.utc).isoformat(),
        'matrix_post_refit':  matrix,
        'current_params':     current_params,
        'proposed_params':    proposed,
        'final_params':       new_params,
        'clamped':            clamped,
        'hermes_rationale':   proposed.get('rationale', ''),
    })
    write_lastrun('ok', proposed=proposed, final=new_params, clamped=clamped)

    # TG summary
    delta = []
    for k in ('min_prob', 'min_edge', 'kelly_fraction'):
        if abs(new_params[k] - current_params[k]) > 1e-4:
            delta.append(f'{k}: {current_params[k]:.3f}->{new_params[k]:.3f}')
    deltas_str = ' | '.join(delta) if delta else 'no changes'
    clamp_str  = f' [CLAMPED {len(clamped)}]' if clamped else ''
    send_tg(f'[BTC-HERMES] nightly OK. matrix p_uu={matrix["p_uu"]:.3f} p_dd={matrix["p_dd"]:.3f}; '
            f'{deltas_str}{clamp_str}')

    # ───── Sizing nightly sibling pass (v2, 2026-06-01) ─────
    # Propose-only — never mutates sizing_config.json. Writes deltas to
    # sizing_proposals_journal.md and posts a Telegram one-liner. Wrapped in
    # try/except so any failure here cannot break the Hermes nightly above.
    try:
        import sizing_nightly as _sn
        _sn.run(dry_run=False)
    except Exception as _sz_err:
        log(f'sizing_nightly sibling pass FAILED: {type(_sz_err).__name__}: {_sz_err}')

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix-only', action='store_true', help='Phase 3a only, skip Hermes')
    parser.add_argument('--dry-run',     action='store_true', help='compute everything, do not write settings')
    args = parser.parse_args()
    try:
        run(matrix_only=args.matrix_only, dry_run=args.dry_run)
    except KeyboardInterrupt:
        log('interrupted')
        sys.exit(130)
    except Exception as e:
        log(f'UNHANDLED: {type(e).__name__}: {e}')
        write_lastrun('uncaught', error=f'{type(e).__name__}: {e}')
        send_tg(f'[BTC-HERMES] nightly UNCAUGHT: {type(e).__name__}: {str(e)[:140]}')
        sys.exit(3)
