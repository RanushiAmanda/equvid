"""M2 v0.2 step B: fit Model A (f1-f5 structure) and Model B (A + f6), null, z, validations, freeze."""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import r2_score

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
OUT = ROOT + '/outputs/m2v2'
A = ['struct_asym', 'interruption_asym', 'position_gap', 'pos_sd_ratio']
B = A + ['f6']
GRID = {'ridge__alpha': [1, 10, 100, 1000]}
R = pd.read_csv(OUT + '/rows.csv')
# an operator row is "effective" only if it really changed the shard; otherwise its label (10 x level)
# would contradict its unchanged features (label noise), so such rows are not used for training
_base = R[R.operator == 'baseline'].set_index('shard_id')[A + ['f6']]
_a, _b = R[A + ['f6']].to_numpy(float), _base.loc[R.shard_id].to_numpy(float)
_d = np.abs(_a - _b)
_d[np.isnan(_a) != np.isnan(_b)] = np.inf
R['changed'] = np.nan_to_num(_d, nan=0.0).max(axis=1) > 1e-9
R['eff'] = (R.operator == 'baseline') | R.changed


def pipe():
    return make_pipeline(StandardScaler(), Ridge())


def oof(d, cols, train_ok):
    """Nested CV. Outer folds hold out whole recording groups; inner CV groups by base shard."""
    X, y, g = d[cols].to_numpy(), d.label.to_numpy(), d.shard_id.to_numpy()
    pred = np.full(len(d), np.nan)
    for k in sorted(d.fold.unique()):
        te = (d.fold == k).to_numpy()
        tr = ~te & train_ok
        gs = GridSearchCV(pipe(), GRID, cv=GroupKFold(3), scoring='r2').fit(X[tr], y[tr], groups=g[tr])
        pred[te] = gs.predict(X[te])
    return pred


def spear(d, col='pred'):
    return {op: round(float(spearmanr(x.signed_level, x[col])[0]), 2)
            for op, x in d[d.operator != 'baseline'].groupby('operator')}


def report(name, d):
    """d has columns pred, z, label, operator, corpus, signed_level, shard_id."""
    e = d[d.eff]
    m = e[e.operator != 'O4_presence'] if name == 'A' else e
    print('')
    print('=== Model %s | rows %d | base shards %d ===' % (name, len(d), d.shard_id.nunique()))
    print('nested-CV R2: pooled %.3f | %s' % (r2_score(m.label, m.pred), {c: round(r2_score(x.label, x.pred), 3) for c, x in m.groupby('corpus')}))
    print('Spearman(prediction, injected level) per operator (effective rows):', spear(e))
    b = d[d.operator == 'baseline']
    print('NULL (unmodified shards, out-of-fold): mean %.3f | SD %.3f' % (b.pred.mean(), b.pred.std()))
    print(b.groupby('corpus').pred.agg(['count', 'mean', 'std']).round(3).to_string())
    print('unmodified shards flagged |z|>1.96: %.1f%% (about 5%% expected) | by corpus: %s' % (
        100 * (b.z.abs() > 1.96).mean(), {c: round(100 * float((x.z.abs() > 1.96).mean()), 1) for c, x in b.groupby('corpus')}))
    rows = []
    for op, x in e[e.operator != 'baseline'].groupby('operator'):
        for lv in (0.2, 0.5, 1.0):
            s = x[x.signed_level.abs().round(2) == lv]
            rows.append({'operator': op, '|level|': lv, 'flagged %': round(100 * float((s.z.abs() > 1.96).mean()), 1)})
    print(pd.DataFrame(rows).pivot(index='operator', columns='|level|', values='flagged %').to_string())
    rng = np.random.default_rng(1)
    zs = b.z.to_numpy()
    for n in (5, 10, 20):
        if len(zs) > n:
            Z = np.array([zs[rng.choice(len(zs), n, replace=False)].mean() * np.sqrt(n) for _ in range(2000)])
            print('dataset-level Z = mean(z)*sqrt(n), n=%d unmodified shards: SD %.2f | 95%% of |Z| below %.2f (ideal 1.00 / 1.96)' % (n, Z.std(), np.percentile(np.abs(Z), 95)))


def loco(d, cols, train_ok):
    X, y, g = d[cols].to_numpy(), d.label.to_numpy(), d.shard_id.to_numpy()
    for c in sorted(d.corpus.unique()):
        te = (d.corpus == c).to_numpy()
        tr = ~te & train_ok
        if len(np.unique(g[tr])) < 6 or (te & train_ok).sum() < 50:
            print('  (skipped %s: too few shards to train on the others)' % c)
            continue
        gs = GridSearchCV(pipe(), GRID, cv=GroupKFold(3), scoring='r2').fit(X[tr], y[tr], groups=g[tr])
        t = d[te & train_ok].assign(p=gs.predict(X[te & train_ok]))
        print('  train on the other corpora -> test %-5s R2 %.3f | Spearman %s' % (c, r2_score(t.label, t.p), spear(t, 'p')))


def run(name, cols, d, train_ok):
    d = d.reset_index(drop=True)
    train_ok = train_ok & d.eff.to_numpy()
    d['pred'] = oof(d, cols, train_ok)
    b = d[d.operator == 'baseline']
    mu, sd = float(b.pred.mean()), float(b.pred.std())
    d['z'] = (d.pred - mu) / sd
    report(name, d)
    print('Leave-one-corpus-out (honest transfer check):')
    loco(d, cols, train_ok)
    X, y, g = d[cols].to_numpy()[train_ok], d.label.to_numpy()[train_ok], d.shard_id.to_numpy()[train_ok]
    gs = GridSearchCV(pipe(), GRID, cv=GroupKFold(5), scoring='r2').fit(X, y, groups=g)
    sc, rg = gs.best_estimator_.named_steps['standardscaler'], gs.best_estimator_.named_steps['ridge']
    alpha = float(gs.best_params_['ridge__alpha'])
    print('FINAL FIT: alpha %g | grouped-CV R2 %.3f' % (alpha, gs.best_score_))
    idx_by = {s: np.flatnonzero(g == s) for s in np.unique(g)}
    ids, rng, cb = np.array(sorted(idx_by)), np.random.default_rng(7), []
    for _ in range(300):
        tr = np.concatenate([idx_by[s] for s in rng.choice(ids, len(ids), replace=True)])
        cb.append(make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr]).named_steps['ridge'].coef_)
    cb = np.array(cb)
    print('standardised coefficients [2.5%, median, 97.5%] (bootstrap over shards):')
    for j, c in enumerate(cols):
        print('  %-18s %7.2f  [%.2f, %.2f, %.2f]' % (c, rg.coef_[j], *np.percentile(cb[:, j], [2.5, 50, 97.5])))
    spec = {'features': cols, 'scaler_mean': sc.mean_.tolist(), 'scaler_scale': sc.scale_.tolist(),
            'coef': rg.coef_.tolist(), 'intercept': float(rg.intercept_), 'alpha': alpha,
            'null_mean': mu, 'null_sd': sd, 'n_base_shards': int(b.shard_id.nunique()), 'n_rows': int(len(d)),
            'nested_cv_r2': round(float(r2_score(d.label[train_ok], d.pred[train_ok])), 3), 'n_rows_used': int(train_ok.sum())}
    chk = ((d[cols].to_numpy() - sc.mean_) / sc.scale_) @ rg.coef_ + rg.intercept_ - gs.predict(d[cols].to_numpy())
    print('JSON scoring reproduces the fitted model: max abs difference %.2e' % np.abs(chk).max())
    return d, spec


if __name__ == '__main__':
    ap = R[R.operator.isin(['O1_fragment', 'O2_interrupt', 'O3_cluster', 'O4_presence'])]
    print('Share of operator rows that really changed the shard (%), by corpus - rows that did not change are NOT used for training:')
    print((100 * ap.groupby(['operator', 'corpus']).changed.mean()).round(0).unstack().to_string())
    dA = R[R.operator.notna()].dropna(subset=A)
    resA, specA = run('A', A, dA, (dA.operator != 'O4_presence').to_numpy())
    dB = R[R.fgroup == 'A'].dropna(subset=B)
    resB, specB = run('B', B, dB, np.ones(len(dB), bool))
    print('')
    print('=== VALUE OF f6: shards that have f6, Spearman(prediction, injected level) ===')
    print('  Model A (no f6):', spear(resA[resA.fgroup == 'A']))
    print('  Model B (with f6):', spear(resB))
    model = {'feature_version': 'm2_v0.2', 'window_s': 60, 'shard_windows': 30,
             'allocation_anchor_ratio': 4.0, 'o4_anchor_ratio': 4.0, 'tiers_abs_z': [1.96, 3, 5],
             'f6_rule': {'eps': 0.1, 'min_presence_s_per_gender_per_window': 2.0},
             'models': {'A': specA, 'B': specB}}
    json.dump(model, open(OUT + '/m2_model_v0.2.json', 'w'), indent=2)
    pd.concat([resA.assign(model='A'), resB.assign(model='B')]).to_csv(OUT + '/rows_scored.csv', index=False)
    print('saved', OUT + '/m2_model_v0.2.json', 'and rows_scored.csv')
