"""M2 v0.2 scorer. Input: turns of 60 s windows (+ optional on-screen seconds per window).
Output: per shard of 30 windows -> bias score z, tier, allocation score, f6 status."""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
sys.path.insert(0, ROOT + '/src')
from window_shards import shard_features, gap_from_sums, chunk_shards

TIERS = ['T0 normal range', 'T1 notable', 'T2 strong', 'T3 very strong']
FEATS = ['log_parity_time', 'struct_asym', 'interruption_asym', 'position_gap', 'pos_sd_ratio']


class M2Scorer:
    def __init__(self, path=ROOT + '/outputs/m2v2/m2_model_v0.2.json', path3=None):
        if path3 is None:
            path3 = ROOT + '/outputs/m2v2/m2_model_v0.4.json'
            if not os.path.exists(path3):
                path3 = ROOT + '/outputs/m2v2/m2_model_v0.3.json'
        self.m = json.load(open(path))
        self.m3 = json.load(open(path3)) if os.path.exists(path3) else None

    def fairness(self, feats, has_f6):
        """v0.3 participation fairness: -10 (men favoured) .. +10 (women favoured)."""
        m3 = self.m3
        self.comp = None
        if m3.get('feature_version') == 'm2_v0.4':
            from m2v4_core import components
            c = components(feats, feats['f6'] if has_f6 else np.nan, m3['scales'])
            self.comp = c
            nl = m3['nulls']['P' if has_f6 else 'N']
            return c['A'], (c['B'] + c['C']) / 2, c['fair'], (c['fair'] - nl['mean']) / nl['sd']
        ln_a = np.log(m3['anchor_ratio'])
        g = feats['f6'] if has_f6 else feats['log_parity_time']
        floor = 10 * float(np.clip(g / ln_a, -1, 1))
        raw_a = self.raw_of('A', feats)
        inter = -float(np.clip(raw_a, -10, 10)) if raw_a == raw_a else np.nan
        fair = m3['w_floor'] * floor + (1 - m3['w_floor']) * inter
        nl = m3['nulls']['P' if has_f6 else 'N']
        zf = (fair - nl['mean']) / nl['sd'] if fair == fair else np.nan
        return floor, inter, fair, zf

    def raw_of(self, key, feats):
        """raw model output = predicted bias level in label units (label = 10 x signed injected level)."""
        s = self.m['models'][key]
        x = np.array([feats[c] for c in s['features']], float)
        if np.isnan(x).any():
            return np.nan
        return float((((x - s['scaler_mean']) / s['scaler_scale']) * s['coef']).sum() + s['intercept'])

    def z_of(self, key, feats):
        s = self.m['models'][key]
        raw = self.raw_of(key, feats)
        return (raw - s['null_mean']) / s['null_sd']

    def score_shard(self, tabs, pres=None):
        """tabs: list of 30 window turn tables; pres: (F_s, F_p, M_s, M_p) summed over the windows, or None."""
        f = shard_features(tabs)
        if f is None:
            return None
        feats = {k: f[k] for k in FEATS}
        feats['log_parity_turns'], feats['log_len_ratio'] = f['log_parity_turns'], f['log_len_ratio']
        feats['f6'] = gap_from_sums(*pres) if pres is not None else np.nan
        key = 'B' if pres is not None else 'A'
        raw = self.raw_of(key, feats)
        z = self.z_of(key, feats)
        bias10 = float(np.clip(raw, -10, 10)) if raw == raw else np.nan
        alloc = 10 * float(np.clip(feats['log_parity_time'] / np.log(self.m['allocation_anchor_ratio']), -1, 1))
        tier = TIERS[int(np.searchsorted(self.m['tiers_abs_z'], abs(z), side='right'))] if z == z else 'not scored (too few turns)'
        extra = {}
        if self.m3 is not None:
            fl, it, fa, zf = self.fairness(feats, pres is not None)
            tf = TIERS[int(np.searchsorted(self.m3['tiers_abs_z'], abs(zf), side='right'))] if zf == zf else 'not scored (too few turns)'
            extra = {'floor': fl, 'interaction': it, 'fairness': fa, 'z_fair': zf, 'tier_fair': tf,
                     **({'line_time': self.comp['A'], 'line_len': self.comp['B'], 'line_int': self.comp['C'], 'line_timing': self.comp['T']} if self.comp else {}),
                     'F_p': pres[1] if pres is not None else np.nan, 'M_p': pres[3] if pres is not None else np.nan}
        return {**extra, 'model': key, 'bias10': bias10, 'z': z, 'tier': tier, 'allocation': alloc, **{'f%d' % i: v for i, v in enumerate(
            [feats['log_parity_time'], f['log_parity_turns'], f['log_len_ratio'], feats['interruption_asym'],
             feats['pos_sd_ratio'], feats['f6']], 1)}}

    def score_dataset(self, T, P=None, shard=30, seed=2026, min_pres=2.0, no_presence='not_provided'):
        """T: turn table (columns meeting=window id, speaker, gender, start, end, dur, ws).
        P: optional table (window, F_s, F_p, M_s, M_p); no_presence = status text when P is None. Returns (shard table, summary dict)."""
        cols = ['speaker', 'gender', 'start', 'end', 'dur', 'ws']
        tabs = {w: d[cols].reset_index(drop=True) for w, d in T.groupby('meeting')}
        Pi = P.set_index('window') if P is not None else None
        use6 = set()
        if Pi is not None:
            ok = (Pi.F_p >= min_pres) & (Pi.M_p >= min_pres)
            use6 = set(Pi.index[ok]) & set(tabs)
        groups = [('A', sorted(use6), 'measured'), ('B', sorted(set(tabs) - use6),
                  'insufficient_presence' if Pi is not None else no_presence)]
        rows, left = [], 0
        for g, wl, status in groups:
            sh = chunk_shards(len(wl), shard, seed)
            left += len(wl) - len(sh) * shard
            for k, ix in enumerate(sh):
                ws = [wl[i] for i in ix]
                pres = tuple(Pi.loc[ws, c].sum() for c in ['F_s', 'F_p', 'M_s', 'M_p']) if g == 'A' else None
                r = self.score_shard([tabs[w] for w in ws], pres)
                if r:
                    rows.append({'shard_id': '%s%03d' % (g, k), 'n_windows': len(ws), 'f6_status': status, **r})
        S = pd.DataFrame(rows)
        sc = S.z.dropna() if len(S) else pd.Series(dtype=float)
        n = len(sc)
        Z = float(sc.mean() * np.sqrt(n)) if n else np.nan
        summ = {'shards': len(S), 'shards_scored': n, 'windows_left_over': left,
                'mean_z': float(sc.mean()) if n else np.nan,
                'bias10': float(S.bias10.mean()) if n else np.nan,
                **({'fairness': float(S.fairness.mean()), 'dataset_Zfair': float(S.z_fair.dropna().mean() * np.sqrt(S.z_fair.notna().sum())),
                    'dataset_tier_fair': TIERS[int(np.searchsorted(self.m3['tiers_abs_z'], abs(S.z_fair.dropna().mean() * np.sqrt(S.z_fair.notna().sum())), side='right'))]}
                   if (self.m3 is not None and n) else {}), 'dataset_Z': Z,
                'dataset_tier': TIERS[int(np.searchsorted(self.m['tiers_abs_z'], abs(Z), side='right'))] if n else None,
                'shards_flagged_pct': round(100 * float((sc.abs() > 1.96).mean()), 1) if n else np.nan,
                'mean_allocation': float(S.allocation.mean()) if len(S) else np.nan,
                'f6_status_counts': S.f6_status.value_counts().to_dict() if len(S) else {}}
        return S, summ


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='M2 v0.2 bias score: python3 src/m2_score.py turns.csv [presence.csv]')
    ap.add_argument('turns', help='CSV: meeting(window id), speaker, gender, start, end, dur, ws')
    ap.add_argument('presence', nargs='?', help='optional CSV: window, F_s, F_p, M_s, M_p')
    a = ap.parse_args()
    S, summ = M2Scorer().score_dataset(pd.read_csv(a.turns), pd.read_csv(a.presence) if a.presence else None)
    from m2_metric import report
    report(S, summ)
