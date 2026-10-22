import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
OUT = ROOT + '/outputs/ava_feature6'
os.makedirs(OUT, exist_ok=True)

STEP = 0.04      # seconds per annotated frame (25 fps)
EPS = 0.1        # seconds added to speaking time so that ln(0) never happens
MIN_PRES = 2.0   # each gender needs at least this many on-screen seconds in a window
SHARD = 30       # windows per shard
SEED = 2026
NBOOT = 2000
rng = np.random.default_rng(SEED)

log_lines = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log_lines.append(s)

# ---------- 1. read the AVA annotations of the videos that have predicted gender ----------
g = pd.read_csv(ROOT + '/outputs/ava_gender/entity_gender.csv')
vids = sorted(g.video.unique())
say('videos with predicted gender:', len(vids))

parts = []
steps = {}
for v in vids:
    path = None
    for d in ['ava_activespeaker_train_v1.0', 'ava_activespeaker_test_v1.0']:
        p = '%s/data/AVA/%s/%s-activespeaker.csv' % (ROOT, d, v)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        say('no annotation file for', v)
        continue
    a = pd.read_csv(path, header=None, usecols=[0, 1, 6, 7], names=['video', 'ts', 'label', 'entity'])
    a['ts'] = pd.to_numeric(a['ts'], errors='coerce')
    a = a.dropna(subset=['ts'])
    a = a.drop_duplicates(['entity', 'ts']).sort_values(['entity', 'ts'])
    dd = a.groupby('entity').ts.diff().dropna()
    dd = dd[(dd > 0.02) & (dd < 0.06)]
    step = float(dd.mean()) if len(dd) else STEP
    if abs(step - 1 / 30) < 0.003:
        step = 1 / 30
    elif abs(step - 0.04) < 0.003:
        step = 0.04
    steps[v] = step
    pres = a.groupby('entity').size() * step
    spk = a[a.label == 'SPEAKING_AUDIBLE'].groupby('entity').size() * step
    e = pd.DataFrame({'presence_s': pres})
    e['speaking_s'] = spk.reindex(e.index).fillna(0.0)
    e['video'] = v
    e['entity'] = e.index
    parts.append(e.reset_index(drop=True))

ent = pd.concat(parts, ignore_index=True)
say('frame step per video (s):', pd.Series(steps).round(4).value_counts().to_dict())
ent['window'] = ent.entity.str.split(':').str[0]
ent = ent.merge(g[['entity', 'gender']], on='entity', how='left')
ent['gender'] = ent['gender'].fillna('unknown')
ent.to_csv(OUT + '/entity_table.csv', index=False)

say('entities annotated:', len(ent))
say('gender of entities:', ent.gender.value_counts().to_dict())
tot = ent.presence_s.sum()
say('share of on-screen time with unknown gender: %.1f%%' % (100 * ent.loc[ent.gender == 'unknown', 'presence_s'].sum() / tot))

# ---------- 2. window table (known genders only) ----------
k = ent[ent.gender.isin(['F', 'M'])].copy()
k['isF'] = (k.gender == 'F')

def gap_from_sums(Fs, Fp, Ms, Mp):
    return np.log((Fs + EPS) / Fp) - np.log((Ms + EPS) / Mp)

def window_sums(df):
    out = []
    for (v, w), d in df.groupby(['video', 'window']):
        Fp = d.loc[d.isF, 'presence_s'].sum()
        Mp = d.loc[~d.isF, 'presence_s'].sum()
        if Fp >= MIN_PRES and Mp >= MIN_PRES:
            out.append((v, w, d.loc[d.isF, 'speaking_s'].sum(), Fp, d.loc[~d.isF, 'speaking_s'].sum(), Mp))
    return pd.DataFrame(out, columns=['video', 'window', 'F_s', 'F_p', 'M_s', 'M_p'])

W = window_sums(k)
W['gap'] = gap_from_sums(W.F_s, W.F_p, W.M_s, W.M_p)
W.to_csv(OUT + '/window_table.csv', index=False)
say('windows total:', k.window.nunique(), ' usable (both genders >= %.0f s on screen): %d' % (MIN_PRES, len(W)))
say('videos with usable windows:', W.video.nunique())
if len(W) < 10:
    say('too few usable windows, stop')
    open(OUT + '/results.txt', 'w').write('\n'.join(log_lines))
    sys.exit()

# ---------- 3. pooled gap and video-cluster bootstrap ----------
S = W[['F_s', 'F_p', 'M_s', 'M_p']].sum()
pooled = gap_from_sums(S.F_s, S.F_p, S.M_s, S.M_p)
V = W.groupby('video')[['F_s', 'F_p', 'M_s', 'M_p']].sum().values

def boot(Vm, n=NBOOT, f_scale=1.0):
    idx = rng.integers(0, len(Vm), size=(n, len(Vm)))
    s = Vm[idx].sum(axis=1)
    return gap_from_sums(s[:, 0] * f_scale, s[:, 1], s[:, 2], s[:, 3])

b = boot(V)
lo, hi = np.percentile(b, [2.5, 97.5])
say('')
say('POOLED presence_voice_gap: %.3f  95%% CI (bootstrap over videos) [%.3f, %.3f]' % (pooled, lo, hi))
say('rate F speaking/on-screen: %.3f   rate M: %.3f' % (S.F_s / S.F_p, S.M_s / S.M_p))
say('window-level gap: median %.2f, SD %.2f (noisy, shown for completeness)' % (W.gap.median(), W.gap.std()))
if lo > 0:
    say('reading: women speak MORE per on-screen second than men')
elif hi < 0:
    say('reading: women are seen but not heard (speak LESS per on-screen second)')
else:
    say('reading: CI contains 0, no evidence of bias')

# ---------- 4. shards ----------
order = rng.permutation(len(W))
n_sh = len(W) // SHARD
say('')
say('shards of %d windows: %d full shards' % (SHARD, n_sh))
sh = []
assign = []
for i in range(n_sh):
    d = W.iloc[order[i * SHARD:(i + 1) * SHARD]]
    assign.append(d[['video', 'window']].assign(shard_id='AVA_%03d' % i))
    s = d[['F_s', 'F_p', 'M_s', 'M_p']].sum()
    sh.append(gap_from_sums(s.F_s, s.F_p, s.M_s, s.M_p))
if assign:
    pd.concat(assign).to_csv(OUT + '/shard_assignment.csv', index=False)
if sh:
    say('shard gaps:', [round(x, 3) for x in sh])
    say('mean %.3f, SD %.3f (few shards, descriptive only)' % (np.mean(sh), np.std(sh, ddof=1) if len(sh) > 1 else float('nan')))

# ---------- 5. null: shuffle gender inside each window (headcount kept) ----------
groups = [(d.presence_s.values, d.speaking_s.values, d.isF.values) for _, d in k.groupby(['video', 'window'])]
def pooled_perm():
    Fs = Fp = Ms = Mp = 0.0
    for p, s, f in groups:
        f = rng.permutation(f)
        fp, mp = p[f].sum(), p[~f].sum()
        if fp >= MIN_PRES and mp >= MIN_PRES:
            Fs += s[f].sum(); Fp += fp; Ms += s[~f].sum(); Mp += mp
    return gap_from_sums(Fs, Fp, Ms, Mp) if Fp > 0 and Mp > 0 else np.nan
null = np.array([pooled_perm() for _ in range(200)])
null = null[~np.isnan(null)]
nlo, nhi = np.percentile(null, [2.5, 97.5])
say('')
say('NULL (gender shuffled inside windows, 200 draws): mean %.3f, band [%.3f, %.3f]' % (null.mean(), nlo, nhi))
say('observed pooled gap %s the null band' % ('is outside' if (pooled < nlo or pooled > nhi) else 'is inside'))

# ---------- 6. injection test: silence women ----------
say('')
say('INJECTION: women speaking time reduced by x  (expected gap shift = ln(1-x))')
for x in [0.2, 0.5, 0.8]:
    bb = boot(V, f_scale=1 - x)
    l2, h2 = np.percentile(bb, [2.5, 97.5])
    shift = gap_from_sums(S.F_s * (1 - x), S.F_p, S.M_s, S.M_p) - pooled
    say('  x=%.1f  gap %.3f  CI [%.3f, %.3f]  shift %.3f (expected %.3f)  detected: %s' % (
        x, pooled + shift, l2, h2, shift, np.log(1 - x), 'yes' if h2 < 0 else 'no'))

# ---------- 7. label noise: random gender flips ----------
say('')
say('LABEL NOISE: random flips of F/M labels (50 draws)')
for p_flip in [0.05, 0.20]:
    vals = []
    for _ in range(50):
        kk = k.copy()
        flip = rng.random(len(kk)) < p_flip
        kk['isF'] = np.where(flip, ~kk.isF, kk.isF)
        Wn = window_sums(kk)
        if len(Wn) > 0:
            s = Wn[['F_s', 'F_p', 'M_s', 'M_p']].sum()
            vals.append(gap_from_sums(s.F_s, s.F_p, s.M_s, s.M_p))
    say('  flip %.0f%%: pooled gap mean %.3f (SD %.3f)  vs observed %.3f' % (100 * p_flip, np.mean(vals), np.std(vals), pooled))

say('')
say('NOTE: gender is PREDICTED from faces (InsightFace). Verify on 100 faces and report per-gender error before using this result.')
open(OUT + '/results.txt', 'w').write('\n'.join(log_lines))
say('saved to', OUT)
