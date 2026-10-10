"""Rebuild btc_ml_model.json from last 7 days of trades only.
Mirrors BTCMLModel.train() exactly: 14 features, 200 epochs, LR=0.05, MIN_TRAIN_N=10.
Skips scalp_exited trades (noisy labels). Backs up old model + writes new one.
"""
import json, math, os, shutil, time
from datetime import datetime, timezone, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
TRADES = os.path.join(ROOT, 'btc_ml_trades.json')
MODEL  = os.path.join(ROOT, 'btc_ml_model.json')

N_FEATURES, EPOCHS, LR, MIN_TRAIN_N = 14, 200, 0.05, 10

with open(TRADES) as f:
    trades = json.load(f)

# Last 7 days from now (UTC)
cutoff = datetime.now(timezone.utc) - timedelta(days=7)

def parse_ts(t):
    s = t.get('logged_at')
    if not s: return None
    try:
        return datetime.fromisoformat(s.replace('Z','+00:00'))
    except Exception:
        return None

# Filter: resolved + non-scalp + valid features + within 7d
examples = []
total_recent = 0
for t in trades:
    ts = parse_ts(t)
    if not ts or ts < cutoff: continue
    total_recent += 1
    if not t.get('resolved') or t.get('outcome') not in ('WIN','LOSS'): continue
    if t.get('scalp_exited'): continue
    if t.get('premkt_probe'): continue   # 2026-06-29 pre-market probe cohort: paper-only, not training
    feats = t.get('ml_features')
    if not feats or len(feats) != N_FEATURES: continue
    direction = t.get('direction','UP').upper()
    outcome   = t.get('outcome','LOSS')
    label = (1 if outcome=='WIN' else 0) if direction=='UP' else (0 if outcome=='WIN' else 1)
    examples.append({'features': feats, 'label': label})

print(f'Recent (7d) total trades : {total_recent}')
print(f'Training examples         : {len(examples)}')
if len(examples) < MIN_TRAIN_N:
    print(f'ABORT: need {MIN_TRAIN_N}, have {len(examples)}')
    raise SystemExit(1)

# Class balance check
pos = sum(1 for e in examples if e['label']==1)
print(f'  label=1 (model right) : {pos}/{len(examples)} = {pos/len(examples)*100:.1f}%')
ups = sum(1 for e,t in zip(examples, [t for t in trades if parse_ts(t) and parse_ts(t) >= cutoff and t.get('resolved') and t.get('outcome') in ("WIN","LOSS") and not t.get('scalp_exited') and not t.get('premkt_probe') and t.get('ml_features') and len(t.get('ml_features'))==N_FEATURES]) if t.get('direction')=='UP')
print(f'  UP trades              : {ups}, DOWN: {len(examples)-ups}')

# Backup old model
old = json.load(open(MODEL))
backup = MODEL + f'.before_7d_retrain_{int(time.time())}.bak'
shutil.copy(MODEL, backup)
print(f'\nBackup saved: {os.path.basename(backup)}')
print(f'OLD: n_trained={old.get("n_trained")}, bias={old.get("bias",0):+.4f}, '
      f'accuracy={old.get("accuracy","?")}%')

# Train from ZERO (this is the wipe + retrain)
w = [0.0]*N_FEATURES
b = 0.0
n = len(examples)
for epoch in range(EPOCHS):
    dw = [0.0]*N_FEATURES
    db = 0.0
    total_loss = 0.0
    correct = 0
    for ex in examples:
        x, y = ex['features'], ex['label']
        z = sum(w[i]*x[i] for i in range(N_FEATURES)) + b
        if z >= 0:
            p = 1.0/(1.0+math.exp(-z))
        else:
            ez = math.exp(z); p = ez/(1.0+ez)
        err = p - y
        for i in range(N_FEATURES):
            dw[i] += err*x[i]
        db += err
        total_loss += -(y*math.log(max(p,1e-9)) + (1-y)*math.log(max(1-p,1e-9)))
        if (p>=0.5)==(y==1): correct += 1
    for i in range(N_FEATURES):
        w[i] -= LR*dw[i]/n
    b -= LR*db/n

accuracy = round(correct/n*100, 1)
new_weights = [round(wi, 6) for wi in w]
new_bias    = round(b, 6)

# Preserve other fields from old model (feature_names, etc.) but replace weights/bias/n
new = dict(old)
new['weights']         = new_weights
new['bias']            = new_bias
new['n_trained']       = n
new['accuracy']        = accuracy
new['last_retrain_ts'] = time.time()
new['retrain_note']    = f'7d-only wipe+retrain {datetime.now(timezone.utc).isoformat()[:19]}Z'

with open(MODEL, 'w') as f:
    json.dump(new, f, indent=2)

print(f'\nNEW: n_trained={n}, bias={new_bias:+.4f}, accuracy={accuracy}%')
print(f'NEW weights: {[round(wi,3) for wi in new_weights]}')
print(f'OLD weights: {[round(wi,3) for wi in old.get("weights",[])]}')
print(f'\nWrote {MODEL}')
print(f'Bot will pick up the new model on next restart.')
