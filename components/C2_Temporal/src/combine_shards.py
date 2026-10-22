"""
Combine the three shard tables (AVA, AMI, MELD) into ONE table for the frozen model (method 2).
f6 (presence_voice_gap) stays EMPTY (NaN) where it is not measured. It is never filled with 0, a mean or f1.
Run:  python3 src/combine_shards.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
OUT = ROOT + '/outputs/shared'
os.makedirs(OUT, exist_ok=True)
src = {'AVA': ROOT + '/outputs/ava_features/ava_shard_table_v3.csv',
       'AMI': ROOT + '/outputs/ami_windows/ami_shard_table.csv',
       'MELD': ROOT + '/outputs/meld_windows/meld_shard_table.csv'}
parts = []
for c, p in src.items():
    d = pd.read_csv(p)
    print('%-5s %-55s %4d shards' % (c, p.replace(ROOT, '~'), len(d)))
    parts.append(d)
T = pd.concat(parts, ignore_index=True)

# f6 status: measured / not applicable (AMI) / not measured yet (MELD without presence)
T['f6_status'] = np.where(T.presence_voice_gap.notna(), 'measured', 'insufficient_presence')
ami = T.corpus == 'AMI'
T.loc[ami, 'presence_voice_gap'] = np.nan
T.loc[ami, 'f6_status'] = 'not_applicable'
T['f6_available'] = T.presence_voice_gap.notna()

# the proposal names of the six features
T['f1'] = T.log_parity_time
T['f2'] = T.log_parity_turns
T['f3'] = T.log_len_ratio
T['f4'] = T.interruption_asym
T['f5'] = T.pos_sd_ratio
T['f6'] = T.presence_voice_gap

lead = ['corpus', 'shard_id', 'n_windows', 'n_videos', 'f6_status', 'f6_available', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6']
rest = [c for c in T.columns if c not in lead]
T = T[lead + rest]
T.to_csv(OUT + '/shard_table_windows.csv', index=False)

print('')
print('COMBINED TABLE:', OUT + '/shard_table_windows.csv', T.shape)
print('shards per corpus and f6 status:')
print(pd.crosstab(T.corpus, T.f6_status).to_string())
print('')
print('missing values per feature (rows = corpus):')
print(T.groupby('corpus')[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].apply(lambda g: g.isna().sum()).to_string())
print('')
print('mean of each feature per corpus:')
print(T.groupby('corpus')[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].mean().round(3).to_string())
print('')
print('CHECK f1 = f2 + f3 (max abs diff):', float((T.f1 - T.f2 - T.f3).abs().max()))
print('CHECK AMI f6 values (must be 0 non-empty):', int(T.loc[T.corpus == 'AMI', 'f6'].notna().sum()))
print('shards with all six features:', int(T[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].notna().all(axis=1).sum()),
      '| with f1-f5 only:', int((T[['f1', 'f2', 'f3', 'f4', 'f5']].notna().all(axis=1) & T.f6.isna()).sum()),
      '| with some f1-f5 missing:', int(T[['f1', 'f2', 'f3', 'f4', 'f5']].isna().any(axis=1).sum()))
