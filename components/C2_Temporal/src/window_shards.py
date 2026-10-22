"""
Shared code: turns of 60 s windows -> pooled shard features f1-f5 (project functions) .
Used by AVA, AMI and MELD so that all three tables are computed in the same way.
"""
import numpy as np
import pandas as pd
from m2_temporal import meeting_features, spread_features

PAD = 1000.0        # windows are placed 1000 s apart so turns of different windows never touch
EPS6 = 0.1          # for f6 (same as ava_feature6.py)


def gap_from_sums(Fs, Fp, Ms, Mp):
    return np.log((Fs + EPS6) / Fp) - np.log((Ms + EPS6) / Mp)


def shard_features(turn_tables, n_present=None):
    """turn_tables: list of DataFrames (one per window) with columns
    speaker, gender, start, end, dur, ws   (ws = start time of the window, same clock as start/end).
    n_present: optional (sum of F persons, sum of M persons) over the windows, for the alternative f1."""
    seq, pos = [], []
    for j, d in enumerate(turn_tables):
        spk = 'w%d|' % j + d['speaker'].astype(str)
        rs, re_ = d['start'] - d['ws'], d['end'] - d['ws']
        a = d[['gender', 'dur']].copy()
        a['speaker'] = spk.values
        b = a.copy()
        a['start'], a['end'] = (rs + j * PAD).values, (re_ + j * PAD).values
        b['start'], b['end'] = rs.values, re_.values
        seq.append(a)
        pos.append(b)
    Ts, P = pd.concat(seq, ignore_index=True), pd.concat(pos, ignore_index=True)
    f = meeting_features(Ts)
    if f is None:
        return None
    out = {k: f[k] for k in ['n_F', 'n_M', 'N_F', 'N_M', 'T_F', 'T_M', 'cut_F', 'cut_M',
                             'log_parity_time', 'log_parity_turns', 'log_len_ratio', 'interruption_asym']}
    out['struct_asym'] = (f['log_parity_turns'] - f['log_len_ratio']) / 2
    # position inside the window (all windows share one 0..60 s axis)
    st, en, dur = P['start'].to_numpy(), P['end'].to_numpy(), P['dur'].to_numpy()
    isF, isM = (P['gender'] == 'F').to_numpy(), (P['gender'] == 'M').to_numpy()
    mid = ((st + en) / 2 - st.min()) / max(en.max() - st.min(), 1e-9)
    out['position_gap'] = ((mid[isF] * dur[isF]).sum() / dur[isF].sum()
                           - (mid[isM] * dur[isM]).sum() / dur[isM].sum())
    s = spread_features(P) or {'len_sd_ratio': np.nan, 'pos_sd_ratio': np.nan}
    out['len_sd_ratio'] = s['len_sd_ratio']
    out['pos_sd_ratio'] = s['pos_sd_ratio']
    if n_present is not None and n_present[0] > 0 and n_present[1] > 0:
        out['alt_log_parity_time'] = float(np.log((out['T_F'] / n_present[0]) / (out['T_M'] / n_present[1])))
    else:
        out['alt_log_parity_time'] = np.nan
    return out


def chunk_shards(n, shard, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [list(perm[i * shard:(i + 1) * shard]) for i in range(n // shard)]
