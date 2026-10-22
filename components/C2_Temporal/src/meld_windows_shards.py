"""
MELD: 60 s windows -> pooled shards of 30 windows -> f1-f5, and f6 where face-presence data exist.
Windows are real 60 s slices of the episode time line (same as AMI). An utterance belongs to the
window in which it STARTS (presence is measured per utterance clip, so it cannot be split).
Run from the project folder:  python3 src/meld_windows_shards.py
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from m2_temporal import build_turns, BACKCHANNEL_MAX_WORDS
from window_shards import shard_features, chunk_shards, gap_from_sums

WIN = 60.0
SHARD = int(os.environ.get('SHARD', 30))
SEED = 2026
MIN_PRES = 2.0
OUT = ROOT + '/outputs/meld_windows'
os.makedirs(OUT, exist_ok=True)

m = pd.read_csv(ROOT + '/outputs/meld_intervals_words.csv')
need = ['split', 'season', 'episode', 'dialogue_id', 'utterance_id', 'speaker', 'gender', 'start', 'end', 'dur', 'n_words']
miss = [c for c in need if c not in m.columns]
if miss:
    raise SystemExit('meld_intervals_words.csv is missing columns: %s' % miss)
print('MELD utterances:', len(m), '| episodes:', m.groupby(['season', 'episode']).ngroups)
print('utterances per split:', m.split.value_counts().to_dict())
m = m[m.gender.isin(['F', 'M'])].copy()
m['epid'] = m.season.astype(str).str.zfill(2).radd('S') + m.episode.astype(str).str.zfill(2).radd('E')
m['k'] = (m.start // WIN).astype(int)
m['meeting'] = m.epid + '#' + m.k.astype(str).str.zfill(3)
m['ws'] = m.k * WIN
m['series'] = m.epid
m['role'] = ''
m['corpus'] = 'MELD'
m['gender_source'] = 'manual'
m['backchannel'] = m.n_words <= BACKCHANNEL_MAX_WORDS
print('treated as backchannel (<= %d words): %d of %d' % (BACKCHANNEL_MAX_WORDS, int(m.backchannel.sum()), len(m)))

# sanity: same speaking seconds as the table used for the clip-level features?
try:
    ia = pd.read_csv(ROOT + '/outputs/shared/intervals_all.csv')
    ia = ia[ia.corpus == 'MELD']
    print('CHECK speaking seconds F / M: this table = %.1f / %.1f | intervals_all.csv = %.1f / %.1f' % (
        m[m.gender == 'F'].dur.sum(), m[m.gender == 'M'].dur.sum(),
        ia[ia.gender == 'F'].dur.sum(), ia[ia.gender == 'M'].dur.sum()))
except Exception as e:
    print('intervals_all.csv check skipped:', e)

# ---- face presence per utterance (pres_F / pres_M = seconds of F / M faces on screen inside the clip)
pres_path = ROOT + '/outputs/meld_face/presence_by_clip_all.csv'
m['F_s'] = np.where(m.gender == 'F', m.dur, 0.0)
m['M_s'] = np.where(m.gender == 'M', m.dur, 0.0)
m['F_p'] = np.nan
m['M_p'] = np.nan
if os.path.exists(pres_path):
    pr = pd.read_csv(pres_path).rename(columns={'dia': 'dialogue_id', 'utt': 'utterance_id'})
    pr = pr.drop_duplicates(['split', 'dialogue_id', 'utterance_id'])
    j = m[['split', 'dialogue_id', 'utterance_id', 'dur']].merge(
        pr, on=['split', 'dialogue_id', 'utterance_id'], how='left')
    m['F_p'] = j.pres_F.to_numpy()
    m['M_p'] = j.pres_M.to_numpy()
    ok = j.samples.notna().to_numpy()
    agree = ((j.samples - j.dur.round()).abs() <= 1).to_numpy()
    for s_ in sorted(m.split.unique()):
        sel = (m.split == s_).to_numpy()
        print('  split %-6s utterances %5d | with presence %5d | clip length agrees in %.0f%%' % (
            s_, sel.sum(), (sel & ok).sum(), 100 * (sel & ok & agree).sum() / max(1, (sel & ok).sum())))
else:
    print('no presence file -> f6 left empty')

# ---- window table
g = m.groupby('meeting')
Wt = pd.DataFrame({'epid': g.epid.first(), 'split': g.split.first(), 'ws': g.ws.first(), 'n_utt': g.size(),
                   'F_s': g.F_s.sum(), 'M_s': g.M_s.sum(),
                   'covered': g.F_p.apply(lambda x: x.notna().all()),
                   'F_p': g.F_p.sum(min_count=1), 'M_p': g.M_p.sum(min_count=1)})
Wt['usable6'] = Wt.covered & (Wt.F_p >= MIN_PRES) & (Wt.M_p >= MIN_PRES)
print('windows with speech:', len(Wt), '| covered by presence:', int(Wt.covered.sum()), '| usable for f6:', int(Wt.usable6.sum()))
Wt.to_csv(OUT + '/meld_window_table.csv')

T = build_turns(m)
T = T.merge(m[['meeting', 'ws']].drop_duplicates('meeting'), on='meeting', how='left')
Tg = {w: d for w, d in T.groupby('meeting')}
has_turn = set(Tg)
print('windows with at least one turn:', len(has_turn), '| turns:', len(T))

groupA = sorted(w for w in Wt.index[Wt.usable6] if w in has_turn)       # f1-f6
groupB = sorted(w for w in Wt.index[~Wt.usable6] if w in has_turn)      # f1-f5 only
rows = []
for name, wl_all, with6 in [('MELDp', groupA, True), ('MELD', groupB, False)]:
    shards = chunk_shards(len(wl_all), SHARD, SEED)
    print('%s: windows %d -> shards %d (left over %d)' % (name, len(wl_all), len(shards), len(wl_all) - len(shards) * SHARD))
    for k, idx in enumerate(shards):
        wl = [wl_all[i] for i in idx]
        f = shard_features([Tg[w] for w in wl])
        row = {'corpus': 'MELD', 'shard_id': '%s_%03d' % (name, k), 'n_windows': len(wl),
               'n_videos': Wt.loc[wl, 'epid'].nunique(), 'f6_available': with6}
        if f is not None:
            row.update(f)
        if with6:
            r = Wt.loc[wl]
            row['presence_voice_gap'] = gap_from_sums(r.F_s.sum(), r.F_p.sum(), r.M_s.sum(), r.M_p.sum())
        else:
            row['presence_voice_gap'] = np.nan
        rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(OUT + '/meld_shard_table.csv', index=False)
T.to_csv(OUT + '/meld_window_turns.csv', index=False)
print('')
print('SHARD TABLE written:', OUT + '/meld_shard_table.csv', R.shape)
if len(R):
    cols = ['shard_id', 'n_videos', 'log_parity_time', 'log_parity_turns', 'log_len_ratio', 'interruption_asym',
            'position_gap', 'len_sd_ratio', 'pos_sd_ratio', 'presence_voice_gap']
    print(R[cols].round(3).head(40).to_string(index=False))
    print('(first 40 shards shown)')
    print('column means:')
    print(R[cols[2:]].mean().round(3).to_string())
    print('missing values per column:')
    print(R[cols[2:]].isna().sum().to_string())
    print('interruptions per shard (cut_F + cut_M): min', int((R.cut_F + R.cut_M).min()), 'max', int((R.cut_F + R.cut_M).max()))
