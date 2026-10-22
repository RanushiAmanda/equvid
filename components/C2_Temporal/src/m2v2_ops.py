"""M2 v0.2 operators, applied INSIDE each 60 s window (turn tables), plus the presence operator O4.
O1-O3 are the AMI notebook operators; selection is Bernoulli(level) per eligible item (unbiased
even when a 60 s window has only a few turns). Total speaking time of each gender never changes."""
import numpy as np
import pandas as pd

ANCHOR = 4.0   # same ratio as the allocation score: level 1.0 = a 4:1 change


def _sorted(t):
    return t.sort_values(['start', 'end']).reset_index(drop=True)


def fragment(t, tgt, lv, rng, k=3, gap=1.2, minp=0.5):
    t = _sorted(t)
    room = (t.groupby('speaker')['start'].shift(-1) - t['end']).fillna(np.inf)
    elig = ((t.gender == tgt) & (t.dur >= k * minp) & (room >= (k - 1) * gap)).to_numpy()
    pick = elig & (rng.random(len(t)) < lv)
    if not pick.any():
        return t
    rows = []
    for i, r in enumerate(t.to_dict('records')):
        if not pick[i]:
            rows.append(r)
            continue
        piece = r['dur'] / k
        for j in range(k):
            p = dict(r)
            p['start'] = r['start'] + j * (piece + gap)
            p['end'] = p['start'] + piece
            p['dur'] = piece
            rows.append(p)
    return pd.DataFrame(rows)


def interrupt(t, tgt, lv, rng, ov=0.3, maxgap=3.0):
    t = _sorted(t)
    st, en = t.start.to_numpy().copy(), t.end.to_numpy().copy()
    dur, gen = t.dur.to_numpy(), t.gender.to_numpy()
    prev_end = t.groupby('speaker')['end'].shift().fillna(-np.inf).to_numpy()
    used, pairs = set(), []
    for i in np.flatnonzero((gen == tgt) & (dur >= 1.0)):
        ns = en[i] - ov * dur[i]
        for j in np.flatnonzero((st > en[i]) & (st <= en[i] + maxgap) & (gen != tgt)):
            if j not in used and ns >= prev_end[j] + 0.05:
                pairs.append((j, ns))
                used.add(j)
                break
    for j, ns in pairs:
        if rng.random() < lv:
            st[j], en[j] = ns, ns + dur[j]
    out = t.copy()
    out['start'], out['end'] = st, en
    return out


def cluster(t, tgt, lv, rng=None):
    t = _sorted(t)
    st, en = t.start.to_numpy().copy(), t.end.to_numpy().copy()
    for s in t.loc[t.gender == tgt, 'speaker'].unique():
        idx = np.flatnonzero((t.speaker == s).to_numpy())
        d = en[idx] - st[idx]
        gaps = np.maximum(st[idx[1:]] - en[idx[:-1]], 0.0)
        new = np.empty(len(idx))
        new[0] = st[idx[0]]
        for k in range(1, len(idx)):
            new[k] = new[k - 1] + d[k - 1] + (1 - lv) * gaps[k - 1]
        st[idx], en[idx] = new, new + d
    out = t.copy()
    out['start'], out['end'] = st, en
    return out


OPS = {'O1_fragment': fragment, 'O2_interrupt': interrupt, 'O3_cluster': cluster}


def presence_op(pres, signed_level, ratio=ANCHOR):
    """O4: change the on-screen seconds only. + : F seen more (seen, not heard); - : M seen more."""
    Fs, Fp, Ms, Mp = pres
    f = ratio ** abs(signed_level)
    return (Fs, Fp * f, Ms, Mp) if signed_level > 0 else (Fs, Fp, Ms, Mp * f)
