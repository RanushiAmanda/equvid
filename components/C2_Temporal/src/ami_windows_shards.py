"""
AMI: 60 s windows -> pooled shards of 30 windows -> f1-f5 (same code as AVA and MELD).
f6 (presence_voice_gap) is NOT available for AMI: there is no screen-presence data.
Run from the project folder:  python3 src/ami_windows_shards.py
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from m2_temporal import build_turns, meeting_features, BACKCHANNEL_MAX_WORDS
from window_shards import shard_features, chunk_shards

WIN = 60.0
SHARD = int(os.environ.get('SHARD', 30))
SEED = 2026
OUT = ROOT + '/outputs/ami_windows'
os.makedirs(OUT, exist_ok=True)

ia = pd.read_csv(ROOT + '/outputs/shared/intervals_all.csv')
need = ['corpus', 'clip_id', 'person_id', 'series', 'gender', 'gender_source', 'start', 'end', 'dur', 'n_words']
miss = [c for c in need if c not in ia.columns]
if miss:
    raise SystemExit('intervals_all.csv is missing columns: %s' % miss)
a = ia[ia.corpus == 'AMI'].copy()
print('AMI rows:', len(a), '| meetings:', a.clip_id.nunique())
print('gender counts:', a.gender.value_counts(dropna=False).to_dict())
print('rows with missing n_words:', int(a.n_words.isna().sum()))
a = a[a.gender.isin(['F', 'M'])].copy()
a['meeting'] = a.clip_id
a['speaker'] = a.person_id
a['role'] = ''
a['backchannel'] = (a.n_words <= BACKCHANNEL_MAX_WORDS).fillna(False)
a['n_words'] = a.n_words.fillna(0)
print('treated as backchannel (<= %d words): %d of %d' % (BACKCHANNEL_MAX_WORDS, int(a.backchannel.sum()), len(a)))

# ---- check 1: does this input reproduce the whole-meeting features already used in the project?
fb = pd.read_csv(ROOT + '/outputs/features_all_baseline.csv')
fb = fb[fb.corpus == 'AMI'].drop_duplicates('clip_id').set_index('clip_id')
Tw = build_turns(a)
ok = tot = 0
for cid, d in Tw.groupby('meeting'):
    f = meeting_features(d)
    if f is None or cid not in fb.index:
        continue
    tot += 1
    r = fb.loc[cid]
    if abs(f['T_F'] - r['T_F']) < 0.01 and abs(f['T_M'] - r['T_M']) < 0.01 and f['N_F'] == r['N_F'] and f['N_M'] == r['N_M']:
        ok += 1
print('CHECK whole-meeting turns reproduce features_all_baseline.csv: %d of %d meetings' % (ok, tot))

# ---- 60 s windows (only meetings where both genders speak; last partial window dropped)
pieces, meta = [], {}
n_single = 0
for cid, d in a.groupby('clip_id'):
    nPF = d[d.gender == 'F'].person_id.nunique()
    nPM = d[d.gender == 'M'].person_id.nunique()
    if nPF == 0 or nPM == 0:
        n_single += 1
        continue
    st, en = d.start.to_numpy(), d.end.to_numpy()
    for k in range(int(d.end.max() // WIN)):
        lo, hi = k * WIN, (k + 1) * WIN
        m = (st < hi) & (en > lo)
        if not m.any():
            continue
        p = d[m].copy()
        p['start'] = np.maximum(p.start, lo)
        p['end'] = np.minimum(p.end, hi)
        p['dur'] = p.end - p.start
        wid = '%s#%03d' % (cid, k)
        p['meeting'] = wid
        p['ws'] = lo
        meta[wid] = (cid, nPF, nPM)
        pieces.append(p)
I = pd.concat(pieces, ignore_index=True)
print('meetings with one gender only (excluded):', n_single, '| meetings used:', len({v[0] for v in meta.values()}))
print('windows with speech:', len(meta))

T = build_turns(I)
T = T.merge(I[['meeting', 'ws']].drop_duplicates('meeting'), on='meeting', how='left')
Tg = {w: d for w, d in T.groupby('meeting')}
W = sorted(Tg)
print('windows with at least one turn:', len(W), '| turns:', len(T))

shards = chunk_shards(len(W), SHARD, SEED)
print('shards:', len(shards), '| windows left over:', len(W) - len(shards) * SHARD)
rows = []
for k, idx in enumerate(shards):
    wl = [W[i] for i in idx]
    nP = (sum(meta[w][1] for w in wl), sum(meta[w][2] for w in wl))
    f = shard_features([Tg[w] for w in wl], n_present=nP)
    row = {'corpus': 'AMI', 'shard_id': 'AMI_%03d' % k, 'n_windows': len(wl),
           'n_videos': len({meta[w][0] for w in wl})}
    if f is not None:
        row.update(f)
    row['presence_voice_gap'] = np.nan
    rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(OUT + '/ami_shard_table.csv', index=False)
T.to_csv(OUT + '/ami_window_turns.csv', index=False)
print('')
print('SHARD TABLE written:', OUT + '/ami_shard_table.csv', R.shape)
cols = ['shard_id', 'n_videos', 'log_parity_time', 'log_parity_turns', 'log_len_ratio', 'interruption_asym',
        'position_gap', 'len_sd_ratio', 'pos_sd_ratio', 'alt_log_parity_time']
print(R[cols].round(3).head(40).to_string(index=False))
print('(first 40 shards shown)')
print('column means:')
print(R[cols[2:]].mean().round(3).to_string())
print('missing values per column:')
print(R[cols[2:]].isna().sum().to_string())
print('interruptions per shard (cut_F + cut_M): min', int((R.cut_F + R.cut_M).min()), 'max', int((R.cut_F + R.cut_M).max()))
print('check f1 = f2 + f3 (max abs diff):', float((R.log_parity_time - R.log_parity_turns - R.log_len_ratio).abs().max()))
