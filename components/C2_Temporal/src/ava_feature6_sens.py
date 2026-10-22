import os
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
OUT = ROOT + '/outputs/ava_feature6'
EPS = 0.1
MIN_PRES = 2.0
SEED = 2026
NBOOT = 2000
NNULL = 40
# error rates from the 100-face check: share of faces predicted F that are really M, and the reverse
E_F, E_F_LO, E_F_HI = 0.100, 0.043, 0.214
E_M, E_M_LO, E_M_HI = 0.082, 0.032, 0.192
rng = np.random.default_rng(SEED)

lines = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    lines.append(s)

ent = pd.read_csv(OUT + '/entity_table.csv').sort_values(['video', 'window']).reset_index(drop=True)
win_codes, win_idx = np.unique(ent.window.values, return_inverse=True)
vid_of_win = pd.Series(ent.video.values).groupby(win_idx).first().values
vid_codes, vid_idx = np.unique(vid_of_win, return_inverse=True)
NW, NV = len(win_codes), len(vid_codes)
pres = ent.presence_s.values
spk = ent.speaking_s.values
gen = ent.gender.values

def sums(isF, known):
    kf = known & isF
    km = known & ~isF
    Fp = np.bincount(win_idx, weights=pres * kf, minlength=NW)
    Mp = np.bincount(win_idx, weights=pres * km, minlength=NW)
    Fs = np.bincount(win_idx, weights=spk * kf, minlength=NW)
    Ms = np.bincount(win_idx, weights=spk * km, minlength=NW)
    ok = (Fp >= MIN_PRES) & (Mp >= MIN_PRES)
    return Fs, Fp, Ms, Mp, ok

def gap(Fs, Fp, Ms, Mp):
    return np.log((Fs + EPS) / Fp) - np.log((Ms + EPS) / Mp)

def pooled(isF, known):
    Fs, Fp, Ms, Mp, ok = sums(isF, known)
    if ok.sum() == 0:
        return np.nan
    return gap(Fs[ok].sum(), Fp[ok].sum(), Ms[ok].sum(), Mp[ok].sum())

def null_mean(isF, known, n=NNULL):
    idx = np.where(known)[0]
    idx = idx[np.argsort(win_idx[idx], kind='stable')]
    w = win_idx[idx]
    L = isF[idx]
    vals = []
    for _ in range(n):
        perm = np.lexsort((rng.random(len(idx)), w))
        f2 = np.zeros(len(isF), bool)
        f2[idx] = L[perm]
        vals.append(pooled(f2, known))
    return float(np.nanmean(vals))

def boot_raw(isF, known):
    Fs, Fp, Ms, Mp, ok = sums(isF, known)
    A = np.stack([np.bincount(vid_idx[ok], weights=x[ok], minlength=NV) for x in (Fs, Fp, Ms, Mp)], axis=1)
    A = A[A[:, 1] > 0]
    idx = rng.integers(0, len(A), size=(NBOOT, len(A)))
    s = A[idx].sum(axis=1)
    return gap(s[:, 0], s[:, 1], s[:, 2], s[:, 3])

def report(name, isF, known):
    g = pooled(isF, known)
    nm = null_mean(isF, known)
    lo, hi = np.percentile(boot_raw(isF, known), [2.5, 97.5])
    say('%-36s gap %.3f | null mean %.3f | EXCESS %.3f  95%% CI [%.3f, %.3f]' % (name, g, nm, g - nm, lo - nm, hi - nm))

known0 = np.isin(gen, ['F', 'M'])
isF0 = gen == 'F'
allk = np.ones(len(gen), bool)
say('EXCESS = pooled gap minus the mean gap you get when gender is shuffled inside each window.')
say('It removes the composition effect, so 0 means no gender difference.')
say('')
say('1. OBSERVED (predicted gender, unknown faces left out)')
report('observed', isF0, known0)

say('')
say('2. UNKNOWN FACES (%.1f%% of on-screen time): where could they go?' % (100 * pres[~known0].sum() / pres.sum()))
report('all unknown counted as F', np.where(known0, isF0, True), allk)
report('all unknown counted as M', np.where(known0, isF0, False), allk)
pF = isF0[known0].mean()
report('unknown split at random (F %.2f)' % pF, np.where(known0, isF0, rng.random(len(gen)) < pF), allk)

say('')
say('3. CLASSIFIER ERRORS from the 100-face check (50 draws each, mean shown)')
def with_errors(eF, eM, draws=50):
    gs, ex = [], []
    for _ in range(draws):
        u = rng.random(len(gen))
        f2 = np.where(known0 & isF0 & (u < eF), False, isF0)
        f2 = np.where(known0 & ~isF0 & (u < eM), True, f2)
        g = pooled(f2, known0)
        gs.append(g)
        ex.append(g - null_mean(f2, known0, n=8))
    return np.nanmean(gs), np.nanstd(gs), np.nanmean(ex), np.nanstd(ex)
for name, eF, eM in [('central (%.1f%% / %.1f%%)' % (100 * E_F, 100 * E_M), E_F, E_M),
                     ('women hidden in M group (%.1f%% / %.1f%%)' % (100 * E_F_LO, 100 * E_M_HI), E_F_LO, E_M_HI),
                     ('men hidden in F group (%.1f%% / %.1f%%)' % (100 * E_F_HI, 100 * E_M_LO), E_F_HI, E_M_LO)]:
    a, b, c, d = with_errors(eF, eM)
    say('%-44s gap %.3f (SD %.3f) | excess %.3f (SD %.3f)' % (name, a, b, c, d))
say('(pair = share of predicted-F that are really M / share of predicted-M that are really F)')
say('')
say('Reading guide: if EXCESS keeps the same sign and stays small in every line, the conclusion does not depend on unknown faces or classifier errors.')
open(OUT + '/results_sensitivity.txt', 'w').write('\n'.join(lines))
say('saved to', OUT + '/results_sensitivity.txt')
