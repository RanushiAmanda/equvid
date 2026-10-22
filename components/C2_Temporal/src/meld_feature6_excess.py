import os
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
EPS = 0.1
MIN_PRES = 2.0
NBOOT = 2000
SEED = 2026
rng = np.random.default_rng(SEED)
lines = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    lines.append(s)

def gap(Fs, Fp, Ms, Mp):
    return np.log((Fs + EPS) / Fp) - np.log((Ms + EPS) / Mp)

def add_null(df):
    # expected speaking time if both genders speak at the same rate inside each unit
    rate = (df.Fs + df.Ms) / (df.Fp + df.Mp)
    df = df.copy()
    df['EFs'] = df.Fp * rate
    df['EMs'] = df.Mp * rate
    return df

def excess_stats(df, cluster=None, name=''):
    df = add_null(df)
    if cluster is not None:
        df = df.groupby(cluster)[['Fs', 'Fp', 'Ms', 'Mp', 'EFs', 'EMs']].sum()
    A = df[['Fs', 'Fp', 'Ms', 'Mp', 'EFs', 'EMs']].values
    t = A.sum(axis=0)
    obs = gap(t[0], t[1], t[2], t[3])
    nul = gap(t[4], t[1], t[5], t[3])
    idx = rng.integers(0, len(A), size=(NBOOT, len(A)))
    s = A[idx].sum(axis=1)
    bo = gap(s[:, 0], s[:, 1], s[:, 2], s[:, 3])
    bn = gap(s[:, 4], s[:, 1], s[:, 5], s[:, 3])
    ex = bo - bn
    lo, hi = np.percentile(ex, [2.5, 97.5])
    say('%-34s units %3d | gap %.3f | null %.3f | EXCESS %.3f  95%% CI [%.3f, %.3f]' % (name, len(A), obs, nul, obs - nul, lo, hi))

# ---------------- MELD ----------------
m = pd.read_csv(ROOT + '/outputs/meld_face/presence_by_dialogue.csv')
def mk(d, addU=None):
    pf = d.pres_F + (d.pres_U if addU == 'F' else 0)
    pm = d.pres_M + (d.pres_U if addU == 'M' else 0)
    out = pd.DataFrame({'Fs': d.speak_F, 'Fp': pf, 'Ms': d.speak_M, 'Mp': pm})
    return out[(out.Fp >= MIN_PRES) & (out.Mp >= MIN_PRES)]
base = mk(m)
flag = m[m.usable == True]
base_file = pd.DataFrame({'Fs': flag.speak_F, 'Fp': flag.pres_F, 'Ms': flag.speak_M, 'Mp': flag.pres_M})
chk = gap(base_file.Fs, base_file.Fp, base_file.Ms, base_file.Mp)
say('MELD check: dialogues usable by the file flag %d; my gap vs file gap, max difference %.6f' % (
    len(base_file), float(np.abs(chk.values - flag.gap.values).max())))
extra = m[m.index.isin(base.index) & ~m.index.isin(base_file.index)]
say('usable by my 2 s rule but NOT by the file flag: %d dialogues' % len(extra))
print(extra[['dia', 'speak_F', 'speak_M', 'pres_F', 'pres_M', 'pres_U']].round(1).to_string())
say('MELD unknown-face share of face time: %.1f%%' % (100 * m.pres_U.sum() / (m.pres_F.sum() + m.pres_M.sum() + m.pres_U.sum())))
say('')
say('MELD (dialogue = unit, bootstrap over dialogues)')
excess_stats(base_file, name='observed (file usable rule)')
excess_stats(base, name='observed (2 s rule)')
excess_stats(mk(m, 'F'), name='unknown faces counted as F')
excess_stats(mk(m, 'M'), name='unknown faces counted as M')

# ---------------- AVA check of the analytic null ----------------
p = ROOT + '/outputs/ava_feature6/window_table.csv'
if os.path.exists(p):
    w = pd.read_csv(p)
    a = pd.DataFrame({'Fs': w.F_s, 'Fp': w.F_p, 'Ms': w.M_s, 'Mp': w.M_p, 'video': w.video})
    say('')
    say('AVA (video = unit): same formula, as a check against the label-shuffle null')
    excess_stats(a, cluster='video', name='AVA observed')
    say('compare the AVA null above with "null mean" on the observed line of results_sensitivity.txt')
    say('(they should be close; if so, the formula is a fair replacement for shuffling)')
say('')
say('EXCESS = observed pooled gap minus the gap expected if both genders spoke at the same rate in every dialogue.')
os.makedirs(ROOT + '/outputs/meld_face', exist_ok=True)
open(ROOT + '/outputs/meld_face/excess_results.txt', 'w').write('\n'.join(lines))
