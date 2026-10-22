"""M2 v0.2 step A: (1) check the loader reproduces the existing shard tables,
(2) build the training rows: real shards (label 0) + operator-modified copies (O1-O4)."""
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from window_shards import shard_features, gap_from_sums, chunk_shards
from m2v2_data import load_windows, assign_folds, make_shards, SHARD, SEED
from m2v2_ops import OPS, presence_op

OUT = ROOT + '/outputs/m2v2'
F5 = ['log_parity_time', 'log_parity_turns', 'log_len_ratio', 'interruption_asym',
      'position_gap', 'pos_sd_ratio', 'struct_asym']
LEVELS = np.round(np.linspace(0.1, 1.0, 10), 1)
W = turns = Wi = None


def init():
    global W, turns, Wi
    W, turns = load_windows()
    Wi = W.set_index('window')


def pres_of(wl):
    r = Wi.loc[wl]
    return (r.F_s.sum(), r.F_p.sum(), r.M_s.sum(), r.M_p.sum())


def feats_row(tabs, pres):
    f = shard_features(tabs)
    if f is None:
        return None
    r = {k: f[k] for k in F5}
    r['f6'] = gap_from_sums(*pres) if pres is not None else np.nan
    return r


def work(job):
    """All rows of ONE shard: the real shard + its operator-modified copies."""
    idx, s, pres = job
    rng = np.random.default_rng([SEED, idx])
    tabs = [turns[w] for w in s['windows']]
    base = feats_row(tabs, pres)
    if base is None:
        return []
    com = {k: s[k] for k in ['shard_id', 'corpus', 'fold', 'fgroup']}
    rows = [{**com, 'operator': 'baseline', 'signed_level': 0.0, **base}]
    for op, fn in OPS.items():
        for tgt, sign in [('F', 1), ('M', -1)]:
            for lv in LEVELS:
                f = feats_row([fn(t, tgt, lv, rng) for t in tabs], pres)
                if f:
                    rows.append({**com, 'operator': op, 'signed_level': sign * lv, **f})
    if pres is not None:
        for sign in (1, -1):
            for lv in LEVELS:
                p = presence_op(pres, sign * lv)
                rows.append({**com, 'operator': 'O4_presence', 'signed_level': sign * lv,
                             **{k: base[k] for k in F5}, 'f6': gap_from_sums(*p)})
    return rows


def check_existing():
    U = pd.read_csv(ROOT + '/outputs/ava_features/window_table_fixed.csv').set_index('window')
    asg = pd.read_csv(ROOT + '/outputs/ava_feature6/shard_assignment.csv')
    jobs = [('AVA', sid, list(g.window), True) for sid, g in asg.groupby('shard_id', sort=True)]
    ami = sorted(W.window[W.corpus == 'AMI'])
    for k, ix in enumerate(chunk_shards(len(ami), SHARD, SEED)):
        jobs.append(('AMI', 'AMI_%03d' % k, [ami[i] for i in ix], False))
    for name, fg in [('MELDp', 'A'), ('MELD', 'B')]:
        wa = sorted(W.window[(W.corpus == 'MELD') & (W.fgroup == fg)])
        for k, ix in enumerate(chunk_shards(len(wa), SHARD, SEED)):
            jobs.append(('MELD', '%s_%03d' % (name, k), [wa[i] for i in ix], fg == 'A'))
    rows = []
    for c, sid, wl, has6 in jobs:
        if c == 'AVA':
            r = U.loc[wl]
            pres = (r.F_s.sum(), r.F_p.sum(), r.M_s.sum(), r.M_p.sum())
        else:
            pres = pres_of(wl) if has6 else None
        f = feats_row([turns[w] for w in wl if w in turns], pres)
        if f:
            rows.append({'shard_id': sid, **f})
    mine = pd.DataFrame(rows)
    paths = ['/outputs/ava_features/ava_shard_table_v3.csv', '/outputs/ami_windows/ami_shard_table.csv',
             '/outputs/meld_windows/meld_shard_table.csv']
    ex = pd.concat([pd.read_csv(ROOT + p) for p in paths]).rename(columns={'presence_voice_gap': 'f6'})
    m = mine.merge(ex[['shard_id'] + F5 + ['f6']], on='shard_id', suffixes=('', '_ex'))
    print('CHECK existing shards: rebuilt %d | in your tables %d | matched %d' % (len(mine), len(ex), len(m)))
    for c in F5 + ['f6']:
        d = (m[c] - m[c + '_ex']).abs()
        print('  max abs diff %-18s %.2e | NaN mismatch: %d' % (c, d.max(), int((m[c].isna() != m[c + '_ex'].isna()).sum())))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    init()
    check_existing()
    W['fold'] = assign_folds(W)
    shards = make_shards(W)
    print('')
    print('windows with turns:', len(W), '| shards (fold-first):', len(shards))
    print(pd.DataFrame(shards).groupby(['corpus', 'fgroup']).size().rename('shards').to_string())
    used = {w for s in shards for w in s['windows']}
    print('windows not used (left over after filling shards of %d):' % SHARD, W[~W.window.isin(used)].groupby('corpus').size().to_dict())
    print('shards per corpus and fold:')
    print(pd.crosstab(pd.DataFrame(shards).corpus, pd.DataFrame(shards).fold).to_string())
    if os.environ.get('SUBSET'):
        shards = shards[::int(os.environ['SUBSET'])]
    jobs = [(i, s, pres_of(s['windows']) if s['fgroup'] == 'A' else None) for i, s in enumerate(shards)]
    nw = int(os.environ.get('WORKERS', max(1, min(8, (os.cpu_count() or 2) - 1))))
    print('workers:', nw)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context('spawn'), initializer=init) as ex:
        for n, r in enumerate(ex.map(work, jobs, chunksize=1)):
            rows += r
            if (n + 1) % 25 == 0:
                print('  shards done %d of %d | %.0f s' % (n + 1, len(jobs), time.time() - t0))
    R = pd.DataFrame(rows)
    R['label'] = 10 * R.signed_level
    R.to_csv(OUT + '/rows.csv', index=False)
    print('')
    print('rows written:', OUT + '/rows.csv', R.shape, '| %.0f s' % (time.time() - t0))
    print(R.groupby(['corpus', 'operator']).size().unstack(fill_value=0).to_string())
    b = R[R.operator == 'baseline'].set_index('shard_id')
    o = R[R.operator.isin(list(OPS))]
    d1 = o.log_parity_time.values - b.loc[o.shard_id, 'log_parity_time'].values
    a = R[R.operator == 'O4_presence']
    d4 = a.interruption_asym.values - b.loc[a.shard_id, 'interruption_asym'].values
    print('')
    print('CHECK f1 unchanged by O1-O3 (max abs change): %.2e' % np.nanmax(np.abs(d1)))
    print('CHECK O4 changes only f6 (max change of f4): %.2e' % np.nanmax(np.abs(d4)))
    target = {'O1_fragment': 'struct_asym', 'O2_interrupt': 'interruption_asym',
              'O3_cluster': 'position_gap', 'O4_presence': 'f6'}
    print('Spearman(signed level, its target feature):')
    for c, d in R.groupby('corpus'):
        print('  %-5s' % c, {op: round(float(spearmanr(x.signed_level, x[target[op]], nan_policy='omit')[0]), 2)
                             for op, x in d[d.operator != 'baseline'].groupby('operator')})
