"""M2 v0.4 build: fairness metric from REAL data of AVA, AMI, MELD only. No synthetic labels, no model training.
 scale: +-10 = 4:1 for log ratios; interruptions scaled by the 95th percentile of real shards.
 evidence z: genders of the speakers are shuffled inside each real shard (permutation null) -> noise level when gender is unrelated.
 check: injected bias (operators) must move the score the expected way.  Run: python3 src/m2v4_build.py  (PERMS=100 WORKERS=7)"""
import os
import sys
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from window_shards import shard_features, gap_from_sums
from m2v2_data import load_windows, assign_folds, make_shards, SEED
from m2v2_ops import OPS, presence_op
from m2v4_core import components

OUT = ROOT + '/outputs/m2v2'
PERMS = int(os.environ.get('PERMS', 100))
KEYS = ['log_parity_time', 'log_parity_turns', 'log_len_ratio', 'interruption_asym', 'position_gap', 'pos_sd_ratio']
W = turns = Wi = None


def init():
    global W, turns, Wi
    W, turns = load_windows()
    Wi = W.set_index('window')


def feats(tabs, pres):
    f = shard_features(tabs)
    if f is None:
        return None
    r = {k: f[k] for k in KEYS}
    r['f6'] = gap_from_sums(*pres) if pres is not None else np.nan
    return r


def permute(tabs, rng):
    """shuffle the genders of the speakers of this shard (counts of F and M speakers stay the same)."""
    sp = sorted({(j, s) for j, t in enumerate(tabs) for s in t.speaker})
    g = {(j, s): t.gender.iloc[(t.speaker == s).to_numpy().argmax()] for j, t in enumerate(tabs) for s in t.speaker.unique()}
    lab = rng.permutation([g[k] for k in sp])
    new = dict(zip(sp, lab))
    return [t.assign(gender=[new[(j, s)] for s in t.speaker]) for j, t in enumerate(tabs)]


def work(job):
    idx, s, pres = job
    rng = np.random.default_rng([SEED, idx])
    tabs = [turns[w] for w in s['windows']]
    base = feats(tabs, pres)
    if base is None:
        return None
    out = {'shard_id': s['shard_id'], 'corpus': s['corpus'], 'base': base, 'perm': [], 'inj': []}
    for _ in range(PERMS):
        pt = permute(tabs, rng)
        f = feats(pt, None)
        if f is None:
            continue
        if pres is not None:
            tf = sum(t[t.gender == 'F'].dur.sum() for t in pt)
            tm = sum(t[t.gender == 'M'].dur.sum() for t in pt)
            tot = pres[0] + pres[2]
            share = tf / (tf + tm) if tf + tm > 0 else 0.5
            f['f6'] = gap_from_sums(tot * share, pres[1], tot * (1 - share), pres[3])
        out['perm'].append(f)
    for op, fn in OPS.items():
        for tgt, sg in [('F', 1), ('M', -1)]:
            for lv in (0.5, 1.0):
                f = feats([fn(t, tgt, lv, rng) for t in tabs], pres)
                if f:
                    out['inj'].append({'op': op, 'level': sg * lv, **f})
    if pres is not None:
        for sg in (1, -1):
            for lv in (0.5, 1.0):
                out['inj'].append({'op': 'O4_presence', 'level': sg * lv, **{k: base[k] for k in KEYS}, 'f6': gap_from_sums(*presence_op(pres, sg * lv))})
    return out


if __name__ == '__main__':
    t0 = time.time()
    init()
    W['fold'] = assign_folds(W)
    shards = make_shards(W)
    jobs = []
    for i, s in enumerate(shards):
        r = Wi.loc[s['windows']]
        pres = (r.F_s.sum(), r.F_p.sum(), r.M_s.sum(), r.M_p.sum()) if s['fgroup'] == 'A' else None
        jobs.append((i, s, pres))
    print('real shards: %d (with screen data %d) | permutations per shard: %d' % (len(jobs), sum(j[2] is not None for j in jobs), PERMS))
    n = int(os.environ.get('WORKERS', max(1, (os.cpu_count() or 2) - 1)))
    with ProcessPoolExecutor(n, mp_context=mp.get_context('spawn'), initializer=init) as ex:
        res = [r for r in ex.map(work, jobs, chunksize=2) if r]
    print('computed in %.0f s' % (time.time() - t0))
    D = lambda lst: pd.DataFrame(lst)
    base = D([{**r['base'], 'shard_id': r['shard_id'], 'corpus': r['corpus']} for r in res])
    spec = {'scale_f4': float(np.percentile(base.interruption_asym.abs(), 95)), 'scale_gap': float(np.percentile(base.position_gap.abs(), 95)),
            'scale_sd': float(np.nanpercentile(base.pos_sd_ratio.abs(), 95))}
    print('real-data scales (95th percentile of |value| over real shards):', {k: round(v, 3) for k, v in spec.items()})

    def comp(f, use6=True):
        return components(f, f['f6'] if use6 else np.nan, spec)
    rows = []
    for r in res:
        c = comp(r['base']); c.update(shard_id=r['shard_id'], corpus=r['corpus'], has6=r['base']['f6'] == r['base']['f6'])
        rows.append(c)
    B0 = D(rows)
    pr = []
    for r in res:
        for f in r['perm']:
            pr.append({'corpus': r['corpus'], 'shard_id': r['shard_id'], 'P': comp(f)['fair'] if r['base']['f6'] == r['base']['f6'] else np.nan, 'N': comp(f, False)['fair']})
    Pm = D(pr)
    nulls = {'P': {'mean': float(Pm.P.mean()), 'sd': float(Pm.P.std()), 'n': int(Pm.P.notna().sum())},
             'N': {'mean': float(Pm.N.mean()), 'sd': float(Pm.N.std()), 'n': int(len(Pm))}}
    print('')
    print('PERMUTATION NULL (genders shuffled inside real shards):', {k: {a: round(b, 3) for a, b in v.items()} for k, v in nulls.items()})
    # calibration: perms split in two halves, null from half 1, test on half 2
    h = Pm.sample(frac=0.5, random_state=1)
    t = Pm.drop(h.index)
    for m in 'NP':
        a, b = h[m].dropna(), t[m].dropna()
        print('  calibration %s: shuffled shards flagged |z|>1.96 = %.1f%% (about 5%% expected)' % (m, 100 * float((((b - a.mean()) / a.std()).abs() > 1.96).mean())))
    zf = lambda row: (row.fair - nulls['P' if row.has6 else 'N']['mean']) / nulls['P' if row.has6 else 'N']['sd']
    B0['z'] = B0.apply(zf, axis=1)
    print('')
    print('REAL SHARDS (what the data actually shows): mean score, share deviating from "gender is unrelated" (|z|>1.96)')
    g = B0.groupby('corpus')
    print(pd.DataFrame({'shards': g.size(), 'mean_score': g.fair.mean().round(2), 'mean_z': g.z.mean().round(2),
                        'dataset_Z': (g.z.mean() * np.sqrt(g.size())).round(2), 'flagged_%': (100 * g.z.apply(lambda s: (s.abs() > 1.96).mean())).round(1),
                        'speaking': g.A.mean().round(2), 'turns_len': g.B.mean().round(2), 'interrupt': g.C.mean().round(2), 'timing_size': g['T'].mean().round(2)}).to_string())
    print('')
    print('INJECTION CHECK (bias added with operators to real shards). score change = injected - unmodified. + target F means the bias is put on women, so score should go DOWN.')
    I = []
    for r in res:
        b0 = comp(r['base'])['fair']
        for f in r['inj']:
            c = comp(f)
            c6 = r['base']['f6'] == r['base']['f6']
            I.append({'op': f['op'], 'level': f['level'], 'd_score': c['fair'] - b0, 'timing': c['T'],
                      'z': (c['fair'] - nulls['P' if c6 else 'N']['mean']) / nulls['P' if c6 else 'N']['sd']})
    I = D(I)
    print(I.groupby(['op', 'level']).agg(mean_change=('d_score', 'mean'), flagged_pct=('z', lambda s: 100 * (s.abs() > 1.96).mean()), timing_size=('timing', 'mean')).round(2).to_string())
    json.dump({'feature_version': 'm2_v0.4', 'anchor_ratio': 4.0, 'scales': spec, 'nulls': nulls, 'tiers_abs_z': [1.96, 3, 5],
               'sign': '+ = women favoured, - = men favoured', 'score': 'mean of speaking-time, turns/turn-length, interruptions lines (equal weights)'},
              open(OUT + '/m2_model_v0.4.json', 'w'), indent=1)
    B0.to_csv(OUT + '/real_shards_v04.csv', index=False)
    print('saved', OUT + '/m2_model_v0.4.json', 'and real_shards_v04.csv')
