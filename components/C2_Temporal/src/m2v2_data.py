"""M2 v0.2 data layer: 60 s windows of AVA / AMI / MELD, recording groups, folds, shards."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from window_shards import chunk_shards

SHARD, SEED, NFOLD = 30, 2026, 5
COLS = ['speaker', 'gender', 'start', 'end', 'dur', 'ws']


def load_windows():
    """Return W (one row per window) and turns (dict window -> DataFrame of turns)."""
    rows, turns = [], {}
    # ---- AVA (all windows here have both genders on screen >= 2 s -> f6 group A)
    T = pd.read_csv(ROOT + '/outputs/ava_features/ava_window_turns.csv')
    U = pd.read_csv(ROOT + '/outputs/ava_features/window_table_fixed.csv').set_index('window')
    for w, d in T.groupby('meeting'):
        if w in U.index:
            turns[w] = d[COLS].reset_index(drop=True)
            r = U.loc[w]
            rows.append(('AVA', w, r.video, r.video, 'A', r.F_s, r.F_p, r.M_s, r.M_p))
    # ---- AMI (no presence data -> group B, status not_applicable)
    T = pd.read_csv(ROOT + '/outputs/ami_windows/ami_window_turns.csv')
    for w, d in T.groupby('meeting'):
        turns[w] = d[COLS].reset_index(drop=True)
        rec = w.split('#')[0]
        rows.append(('AMI', w, rec, rec[:-1], 'B', np.nan, np.nan, np.nan, np.nan))
    # ---- MELD (usable6 windows -> group A, the rest -> group B)
    T = pd.read_csv(ROOT + '/outputs/meld_windows/meld_window_turns.csv')
    Wt = pd.read_csv(ROOT + '/outputs/meld_windows/meld_window_table.csv').set_index('meeting')
    for w, d in T.groupby('meeting'):
        if w not in Wt.index:
            continue
        turns[w] = d[COLS].reset_index(drop=True)
        r = Wt.loc[w]
        a = bool(r.usable6)
        rows.append(('MELD', w, r.epid, r.epid[:3], 'A' if a else 'B',
                     r.F_s, r.F_p if a else np.nan, r.M_s, r.M_p if a else np.nan))
    W = pd.DataFrame(rows, columns=['corpus', 'window', 'rec', 'group', 'fgroup', 'F_s', 'F_p', 'M_s', 'M_p'])
    return W, turns


def assign_folds(W, nfold=NFOLD):
    """Whole groups (AVA video / AMI meeting series / MELD season) go to ONE fold: no identity leakage."""
    fold = {}
    for c, d in W.groupby('corpus'):
        load = np.zeros(nfold)
        for g, n in d.groupby('group').size().sort_values(ascending=False).items():
            k = int(load.argmin())
            fold[(c, g)] = k
            load[k] += n
    return [fold[(c, g)] for c, g in zip(W.corpus, W.group)]


def make_shards(W):
    """Shards of 30 windows inside (corpus, fold, f6 group). Returns list of dicts."""
    out = []
    for (c, f, fg), d in W.groupby(['corpus', 'fold', 'fgroup']):
        wl = sorted(d.window)
        for k, idx in enumerate(chunk_shards(len(wl), SHARD, SEED)):
            out.append({'shard_id': '%s_%s%d_%03d' % (c, fg, f, k), 'corpus': c, 'fold': f,
                        'fgroup': fg, 'windows': [wl[i] for i in idx]})
    return out
