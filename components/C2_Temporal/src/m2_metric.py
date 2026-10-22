"""M2 v0.2 END-TO-END metric. The user gives ONE raw table; the tool does everything and prints the bias score.
   python3 src/m2_metric.py segments.csv [screen.csv] [--detail]
 segments.csv (one row per spoken segment): recording, speaker, gender (F/M), start, end   [+ n_words, optional]
 screen.csv   (optional, one row per 60 s window): recording, window_start, F_p, M_p   (seconds F / M faces were on screen)
 Inside: gender filter -> 60 s windows -> backchannels dropped -> turns -> shards of 30 windows -> f1-f6 -> z, tier, allocation, f6 status."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from m2_temporal import build_turns, BACKCHANNEL_MAX_WORDS
from m2_score import M2Scorer

WIN = 60.0


def run(seg, scr=None, verbose=True):
    say = print if verbose else (lambda *a: None)
    s = seg[seg.gender.isin(['F', 'M'])].copy()
    s['dur'] = s.end - s.start
    if 'n_words' in s.columns and s.n_words.notna().any():
        s['backchannel'] = (s.n_words <= BACKCHANNEL_MAX_WORDS).fillna(False)
        say('backchannel rule: <= %d words' % BACKCHANNEL_MAX_WORDS)
    else:
        s['n_words'] = 0
        s['backchannel'] = s.dur < 0.5
        say('no n_words column -> backchannel rule: segment shorter than 0.5 s')
    pieces = []
    for rec, d in s.groupby('recording'):
        if d.gender.nunique() < 2:
            say('skipped (one gender only):', rec)
            continue
        for k in range(int(d.end.max() // WIN)):
            lo, hi = k * WIN, (k + 1) * WIN
            p = d[(d.start < hi) & (d.end > lo)].copy()
            if p.empty:
                continue
            p['start'], p['end'] = p.start.clip(lower=lo), p.end.clip(upper=hi)
            p['dur'] = p.end - p.start
            p['meeting'], p['ws'], p['window_start'] = '%s#%04d' % (rec, k), lo, lo
            pieces.append(p)
    if not pieces:
        raise SystemExit('no usable 60 s windows (need both genders and at least 60 s)')
    I = pd.concat(pieces, ignore_index=True)
    for c, v in [('corpus', 'USER'), ('series', ''), ('role', ''), ('gender_source', 'user')]:
        I[c] = v
    T = build_turns(I).merge(I[['meeting', 'ws']].drop_duplicates('meeting'), on='meeting')
    say('windows:', I.meeting.nunique(), '| turns:', len(T))
    P = None
    if scr is not None:
        sp = I.assign(F=np.where(I.gender == 'F', I.dur, 0.0), M=np.where(I.gender == 'M', I.dur, 0.0))
        sp = sp.groupby(['recording', 'window_start', 'meeting'])[['F', 'M']].sum().reset_index()
        P = sp.merge(scr, on=['recording', 'window_start'], how='left').rename(columns={'F': 'F_s', 'M': 'M_s', 'meeting': 'window'})
        P = P[['window', 'F_s', 'F_p', 'M_s', 'M_p']].fillna(0.0)
    return M2Scorer().score_dataset(T, P, no_presence='not_provided')


def report(S, summ):
    """ONE answer for the whole dataset: participation fairness, -10 (men favoured) .. +10 (women favoured)."""
    n = summ['shards_scored']
    if not n:
        raise SystemExit('no shard could be scored (need at least 30 windows = 30 minutes of speech with both genders)')
    if 'line_time' not in S.columns:
        raise SystemExit('outputs/m2v2/m2_model_v0.4.json is missing: run python3 src/m2v4_build.py first')
    Z, F = summ['dataset_Zfair'], summ['fairness']
    tier = summ['dataset_tier_fair']
    code = tier.split()[0]
    verdict = 'fair (%s)' % code if code == 'T0' else 'NOT fair: %s favoured (%s)' % ('women' if Z > 0 else 'men', code)
    pm = S.F_p.notna()
    if pm.any():
        r = S.F_p[pm].sum() / S.M_p[pm].sum()
        opp = '%.1f : 1  (equal-opportunity check%s)' % (r, '' if 0.5 <= r <= 2 else ', UNEQUAL: read with care')
    else:
        opp = 'unknown (no screen data, equal opportunity cannot be checked)'
    print('')
    print('PARTICIPATION FAIRNESS : %+.1f   (-10 men favoured ... +10 women favoured)' % F)
    print('Verdict                : %s   [evidence z = %+.2f]' % (verdict, Z))
    print(' - speaking time per screen-second : %+.1f' % S.line_time.mean())
    print(' - turn length (more turns = shorter): %+.1f' % S.line_len.mean())
    print(' - interruptions                   : %+.1f' % S.line_int.mean())
    print(' - timing pattern (size only, not in score) : %.1f of 10' % S.line_timing.mean())
    print('Screen presence F:M    : %s' % opp)
    print('Based on %d shards x 30 windows (60 s) = %d min | shards flagged %.1f%% | |score| below about 2 is noise level' % (
        n, n * 30, 100 * float((S.z_fair.abs() > 1.96).mean())))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        raise SystemExit(__doc__)
    S, summ = run(pd.read_csv(args[0]), pd.read_csv(args[1]) if len(args) > 1 else None, verbose='--detail' in sys.argv)
    report(S, summ)
    if '--detail' in sys.argv:
        pd.set_option('display.width', 200)
        print(S[['shard_id', 'n_windows', 'model', 'f6_status', 'fairness', 'z_fair', 'tier_fair', 'allocation']].round(2).to_string(index=False))
