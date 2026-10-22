"""M2 v0.2 step C: score the three real corpora with the frozen model + sanity tests (for the viva)."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from m2_score import M2Scorer
from m2v2_ops import OPS, presence_op

pd.set_option('display.width', 200)
M = M2Scorer()
rd = lambda p: pd.read_csv(ROOT + p)
T_ava, T_ami, T_meld = (rd('/outputs/ava_features/ava_window_turns.csv'), rd('/outputs/ami_windows/ami_window_turns.csv'),
                        rd('/outputs/meld_windows/meld_window_turns.csv'))
P_ava = rd('/outputs/ava_features/window_table_fixed.csv')
P_meld = rd('/outputs/meld_windows/meld_window_table.csv').rename(columns={'meeting': 'window'})[['window', 'F_s', 'F_p', 'M_s', 'M_p']]

print('1) REAL DATA scored by the frozen model (in-sample: these windows were also used for training)')
res = {}
for name, T, P in [('AVA', T_ava, P_ava), ('AMI', T_ami, None), ('MELD', T_meld, P_meld)]:
    S, s = M.score_dataset(T, P, no_presence='not_applicable')
    res[name] = S
    print('')
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    print(S.head(5)[['shard_id', 'model', 'f6_status', 'z', 'tier', 'allocation']].round(2).to_string(index=False))
allS = pd.concat([v.assign(corpus=k) for k, v in res.items()])
allS.to_csv(ROOT + '/outputs/m2v2/real_shards_scored.csv', index=False)

print('')
print('2) SYMMETRY CHECK: swap F and M everywhere. Inside one model, z + z_swapped must be a constant')
print('   (the null is centred on typical data, not on parity, so the constant is not 0 and differs between model A and B)')
sw = lambda T: T.assign(gender=T.gender.map({'F': 'M', 'M': 'F'}))
for name, T, P in [('AVA', T_ava, P_ava), ('MELD', T_meld, P_meld)]:
    P2 = P.rename(columns={'F_s': 'M_s', 'F_p': 'M_p', 'M_s': 'F_s', 'M_p': 'F_p'})
    S1, _ = M.score_dataset(T, P)
    S2, _ = M.score_dataset(sw(T), P2)
    for mod, x in S1.groupby('model'):
        zs = x.z + S2.loc[x.index, 'z']
        print('  %-5s model %s | shards %3d | corr(z, z_swapped) %.3f | z + z_swapped: mean %.3f, SD %.1e (SD must be ~0)' % (
            name, mod, len(x), np.corrcoef(x.z, S2.loc[x.index, 'z'])[0, 1], zs.mean(), zs.std()))

print('')
print('3) INJECTION CHECK: one real shard-set, bias added with the operators -> z must move the expected way')
wl = sorted(T_meld.meeting.unique())
ok_w = set(P_meld[(P_meld.F_p >= 2) & (P_meld.M_p >= 2)].window)
use = [w for w in wl if w in ok_w][:30]
tabs = [T_meld[T_meld.meeting == w][['speaker', 'gender', 'start', 'end', 'dur', 'ws']].reset_index(drop=True) for w in use]
Pi = P_meld.set_index('window').loc[use]
pres = tuple(Pi[c].sum() for c in ['F_s', 'F_p', 'M_s', 'M_p'])
rng = np.random.default_rng(5)
rows = [('unmodified', M.score_shard(tabs, pres)['z'])]
for op, fn in OPS.items():
    for tgt in ('F', 'M'):
        rows.append(('%s -> %s, level 1.0' % (op, tgt), M.score_shard([fn(t, tgt, 1.0, rng) for t in tabs], pres)['z']))
for sg in (1, -1):
    rows.append(('O4_presence, level %+d' % sg, M.score_shard(tabs, presence_op(pres, sg * 1.0))['z']))
print(pd.DataFrame(rows, columns=['case', 'z']).round(2).to_string(index=False))
print('(F-targeted operators should push z up, M-targeted down; one shard only, random operators -> indicative)')
