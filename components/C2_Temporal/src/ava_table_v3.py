"""AVA shard table v3: same shard features code as AMI and MELD (window_shards.py)."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from window_shards import shard_features, gap_from_sums

OUT = ROOT + '/outputs/ava_features'
T = pd.read_csv(OUT + '/ava_window_turns.csv')
asg = pd.read_csv(ROOT + '/outputs/ava_feature6/shard_assignment.csv')
U = pd.read_csv(OUT + '/window_table_fixed.csv').set_index('window')
Tg = {w: d for w, d in T.groupby('meeting')}

rows = []
for sid, g in asg.groupby('shard_id', sort=True):
    r = U.loc[g.window]
    tabs = [Tg[w] for w in g.window if w in Tg]
    f = shard_features(tabs)
    row = {'corpus': 'AVA', 'shard_id': sid, 'n_windows': len(g), 'n_videos': g.video.nunique()}
    if f is not None:
        row.update(f)
    row['presence_voice_gap'] = gap_from_sums(r.F_s.sum(), r.F_p.sum(), r.M_s.sum(), r.M_p.sum())
    rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(OUT + '/ava_shard_table_v3.csv', index=False)
print('written', OUT + '/ava_shard_table_v3.csv', R.shape)
old = pd.read_csv(OUT + '/ava_shard_table.csv')
for c in ['log_parity_time', 'log_parity_turns', 'log_len_ratio', 'interruption_asym', 'presence_voice_gap']:
    print('max abs diff vs previous table:', c, round(float((R[c].values - old[c].values).__abs__().max()), 6))
cols = ['shard_id', 'log_parity_time', 'log_parity_turns', 'log_len_ratio', 'interruption_asym',
        'position_gap', 'len_sd_ratio', 'pos_sd_ratio', 'presence_voice_gap']
print(R[cols].round(3).to_string(index=False))
print('column means:')
print(R[cols[1:]].mean().round(3).to_string())
print('missing values per column:')
print(R[cols[1:]].isna().sum().to_string())
