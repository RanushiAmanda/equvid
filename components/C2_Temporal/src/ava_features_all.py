"""
AVA: features f1-f6 on 60 s windows, pooled into shards of 30 windows.
FIX: the frame step is estimated per video (some AVA videos are 30 fps, not 25 fps).
Reuses build_turns / meeting_features from m2_temporal (same code as AMI and MELD).
Run from the project folder:  python3 src/ava_features_all.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from m2_temporal import build_turns, meeting_features

DEFAULT_STEP = 0.04
EPS = 0.1
MIN_PRES = 2.0
SHARD = int(os.environ.get('SHARD', 30))
SEED = 2026
NBOOT = 2000
BC_MIN = 0.5
WIN = 60.0
OFFSET = 100.0
MIN_SPREAD = 3
OUT = ROOT + '/outputs/ava_features'
os.makedirs(OUT, exist_ok=True)


def gap_from_sums(Fs, Fp, Ms, Mp):
    return np.log((Fs + EPS) / Fp) - np.log((Ms + EPS) / Mp)


ent0 = pd.read_csv(ROOT + '/outputs/ava_feature6/entity_table.csv')
old_W = pd.read_csv(ROOT + '/outputs/ava_feature6/window_table.csv')
pres_rows, recs, steps = [], [], {}
for v, e in ent0.groupby('video'):
    files = sorted(glob.glob('%s/data/AVA/**/%s-activespeaker.csv' % (ROOT, v), recursive=True))
    if not files:
        print('MISSING annotation file for video', v)
        continue
    a = pd.read_csv(files[0], header=None, usecols=[0, 1, 6, 7], names=['video', 'ts', 'label', 'entity'])
    a = a[a.entity.isin(set(e.entity))].drop_duplicates(['entity', 'ts']).sort_values(['entity', 'ts'])
    d = a.groupby('entity').ts.diff().dropna()
    d = d[(d > 0.02) & (d < 0.06)]
    step = float(d.mean()) if len(d) else DEFAULT_STEP
    if abs(step - 1 / 30) < 0.003:
        step = 1 / 30
    elif abs(step - 0.04) < 0.003:
        step = 0.04
    steps[v] = step
    pres = a.groupby('entity').size() * step
    sp_ = a[a.label == 'SPEAKING_AUDIBLE']
    spk = sp_.groupby('entity').size().reindex(pres.index).fillna(0) * step
    pres_rows.append(pd.DataFrame({'entity': pres.index, 'presence_s': pres.values, 'speaking_s': spk.values}))
    for eid, g in sp_.groupby('entity'):
        ts = g.ts.to_numpy()
        br = np.where(np.diff(ts) > step * 1.5)[0]
        b = np.r_[0, br + 1, len(ts)]
        for s_, c_ in zip(ts[b[:-1]], np.diff(b)):
            recs.append((eid, s_, s_ + c_ * step))

print('frame step per video (seconds):')
print(pd.Series(steps).round(4).value_counts().to_string())
ent = ent0[['video', 'entity', 'window', 'gender']].merge(pd.concat(pres_rows), on='entity')
print('total on-screen seconds: old (0.04 assumed) =', round(float(ent0.presence_s.sum())),
      '| corrected =', round(float(ent.presence_s.sum())))

k = ent[ent.gender.isin(['F', 'M'])].copy()
k['isF'] = k.gender == 'F'
out = []
for (v, w), d in k.groupby(['video', 'window']):
    Fp = d.loc[d.isF, 'presence_s'].sum()
    Mp = d.loc[~d.isF, 'presence_s'].sum()
    if Fp >= MIN_PRES and Mp >= MIN_PRES:
        out.append((v, w, d.loc[d.isF, 'speaking_s'].sum(), Fp, d.loc[~d.isF, 'speaking_s'].sum(), Mp))
U = pd.DataFrame(out, columns=['video', 'window', 'F_s', 'F_p', 'M_s', 'M_p'])
U['gap'] = gap_from_sums(U.F_s, U.F_p, U.M_s, U.M_p)
U = U.sort_values(['video', 'window']).reset_index(drop=True)
U.to_csv(OUT + '/window_table_fixed.csv', index=False)
print('windows total:', k.window.nunique(), '| usable: old =', len(old_W), ', corrected =', len(U))


def pooled(df):
    S = df[['F_s', 'F_p', 'M_s', 'M_p']].sum()
    return gap_from_sums(S.F_s, S.F_p, S.M_s, S.M_p)


V = U.groupby('video')[['F_s', 'F_p', 'M_s', 'M_p']].sum().values
idx = np.random.default_rng(SEED).integers(0, len(V), size=(NBOOT, len(V)))
s = V[idx].sum(axis=1)
bs = gap_from_sums(s[:, 0], s[:, 1], s[:, 2], s[:, 3])
lo, hi = np.percentile(bs, [2.5, 97.5])
print('POOLED presence_voice_gap: old = %.3f | corrected = %.3f, 95%% CI (bootstrap over videos) [%.3f, %.3f]'
      % (pooled(old_W), pooled(U), lo, hi))

use = set(U.window)
I = pd.DataFrame(recs, columns=['speaker', 'start', 'end'])
info = ent[['entity', 'window', 'video', 'gender']].rename(columns={'entity': 'speaker'})
I = I.merge(info, on='speaker', how='inner')
I = I[I.window.isin(use)].copy()
sp = I.window.str.rsplit('_', n=2, expand=True)
I['ws'] = sp[1].astype(float)
I['we'] = sp[2].astype(float)
I['start'] = np.maximum(I.start, I.ws)
I['end'] = np.minimum(I.end, I.we)
I = I[I.end > I.start + 1e-6].copy()
I['dur'] = I.end - I.start

chk = I.groupby(['window', 'gender']).dur.sum().unstack(fill_value=0).reindex(U.window).fillna(0)
dF = (chk.get('F', 0) - U.set_index('window').F_s).abs().max()
dM = (chk.get('M', 0) - U.set_index('window').M_s).abs().max()
print('CHECK speaking seconds vs corrected window table: max abs diff F =', round(float(dF), 3),
      ', M =', round(float(dM), 3), '(should be about 0)')

I['corpus'] = 'AVA'
I['meeting'] = I.window
I['series'] = I.video
I['role'] = ''
I['gender_source'] = 'predicted'
I['n_words'] = 0
I['backchannel'] = I.dur < BC_MIN
print('speaking runs:', len(I), '| treated as backchannel (shorter than', BC_MIN, 's):', int(I.backchannel.sum()))

T = build_turns(I)
T = T.merge(I[['speaker', 'ws']].drop_duplicates('speaker'), on='speaker', how='left')
T['u'] = ((T.start + T.end) / 2 - T.ws) / WIN
Tg = {w: d for w, d in T.groupby('meeting')}
print('turns:', len(T), 'in', len(Tg), 'of', len(U), 'windows')

n = len(U)
asg_path = ROOT + '/outputs/ava_feature6/shard_assignment.csv'
if os.path.exists(asg_path):
    asg = pd.read_csv(asg_path)
    key = {w: i for i, w in enumerate(U.window)}
    shards = [[key[w] for w in g_.window if w in key] for _, g_ in asg.groupby('shard_id', sort=True)]
    print('shards read from ava_feature6.py (shard_assignment.csv):', len(shards))
else:
    perm = np.random.default_rng(SEED).permutation(n)
    shards = [list(perm[i * SHARD:(i + 1) * SHARD]) for i in range(n // SHARD)]
print('shards:', len(shards), '| windows left over (not in a full shard):', n - len(shards) * SHARD)


def wmean_sd(u, d):
    m = (u * d).sum() / d.sum()
    return m, np.sqrt((((u - m) ** 2) * d).sum() / d.sum())


rows = []
for kk, ix in enumerate(shards):
    r = U.iloc[ix]
    parts = []
    for j, w in enumerate(r.window):
        if w not in Tg:
            continue
        d = Tg[w].copy()
        d['start'] = d.start - d.ws + j * OFFSET
        d['end'] = d.end - d.ws + j * OFFSET
        parts.append(d)
    row = {'corpus': 'AVA', 'shard_id': 'AVA_%03d' % kk, 'n_windows': len(r), 'n_videos': r.video.nunique()}
    if not parts:
        rows.append(row)
        continue
    Ts = pd.concat(parts, ignore_index=True)
    f = meeting_features(Ts)
    if f is None:
        rows.append(row)
        continue
    isF, isM = (Ts.gender == 'F').to_numpy(), (Ts.gender == 'M').to_numpy()
    mF, sF = wmean_sd(Ts.u.to_numpy()[isF], Ts.dur.to_numpy()[isF])
    mM, sM = wmean_sd(Ts.u.to_numpy()[isM], Ts.dur.to_numpy()[isM])
    row.update({k_: f[k_] for k_ in ['n_F', 'n_M', 'N_F', 'N_M', 'T_F', 'T_M', 'cut_F', 'cut_M',
                                     'log_parity_time', 'log_parity_turns', 'log_len_ratio',
                                     'interruption_asym']})
    row['struct_asym'] = (f['log_parity_turns'] - f['log_len_ratio']) / 2
    row['position_gap'] = mF - mM
    ok = isF.sum() >= MIN_SPREAD and isM.sum() >= MIN_SPREAD
    row['pos_sd_ratio'] = float(np.log(sF / sM)) if ok else np.nan
    row['presence_voice_gap'] = gap_from_sums(r.F_s.sum(), r.F_p.sum(), r.M_s.sum(), r.M_p.sum())
    rows.append(row)

R = pd.DataFrame(rows)
R.to_csv(OUT + '/ava_shard_table.csv', index=False)
T.to_csv(OUT + '/ava_window_turns.csv', index=False)
print('')
print('SHARD TABLE written:', OUT + '/ava_shard_table.csv', R.shape)
cols = ['shard_id', 'n_windows', 'n_videos', 'log_parity_time', 'log_parity_turns', 'log_len_ratio',
        'interruption_asym', 'position_gap', 'pos_sd_ratio', 'presence_voice_gap']
print(R[cols].round(3).to_string(index=False))
print('')
print('column means:')
print(R[cols[3:]].mean().round(3).to_string())
print('check f1 = f2 + f3 (max abs diff):',
      float((R.log_parity_time - R.log_parity_turns - R.log_len_ratio).abs().max()))
print('shards with missing pos_sd_ratio:', int(R.pos_sd_ratio.isna().sum()), 'of', len(R))
print('f6 per shard (must equal the shard gaps in results.txt):', [round(x, 3) for x in R.presence_voice_gap.tolist()])
print('interruptions per shard (cut_F + cut_M): min', int((R.cut_F + R.cut_M).min()),
      'max', int((R.cut_F + R.cut_M).max()), '(very small = f4 is not informative on AVA)')
