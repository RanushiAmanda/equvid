import os
import re
import sys
import numpy as np
import pandas as pd
import cv2

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
D = ROOT + '/outputs/ava_gender'
V = D + '/verify'
os.makedirs(V, exist_ok=True)
PER_GENDER = 50
PER_PAGE = 20
CAP_PER_VIDEO = 4
SEED = 2026
mode = sys.argv[1] if len(sys.argv) > 1 else 'make'

def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

if mode == 'make' and os.path.exists(V + '/verify_key.csv') and 'force' not in sys.argv:
    print('verify_key.csv already exists. Not overwriting your work. Use: make force')
    sys.exit()

if mode == 'make':
    rng = np.random.default_rng(SEED)
    pat = re.compile(r'^(.+?)_\1_(\d{4})_(\d{4})_(\d+)_([0-9.]+)\.jpg$')
    rows = []
    for f in os.listdir(D + '/crops'):
        m = pat.match(f)
        if m:
            rows.append((f, m.group(1) + '_' + m.group(2) + '_' + m.group(3) + ':' + m.group(4)))
    c = pd.DataFrame(rows, columns=['file', 'entity'])
    g = pd.read_csv(D + '/entity_gender.csv')
    c = c.merge(g[['video', 'entity', 'gender']], on='entity', how='inner')
    print('crops matched to entities:', len(c), ' entities with a crop:', c.entity.nunique())
    print(c.drop_duplicates('entity').gender.value_counts().to_dict())
    pick = []
    for gen in ['F', 'M']:
        e = c[c.gender == gen].groupby('entity').first().reset_index()
        e = e.sample(frac=1.0, random_state=SEED)
        cnt = {}
        chosen = []
        for _, r in e.iterrows():
            if cnt.get(r.video, 0) < CAP_PER_VIDEO:
                chosen.append(r)
                cnt[r.video] = cnt.get(r.video, 0) + 1
            if len(chosen) == PER_GENDER:
                break
        if len(chosen) < PER_GENDER:
            have = set(x.entity for x in chosen)
            for _, r in e.iterrows():
                if r.entity not in have:
                    chosen.append(r)
                if len(chosen) == PER_GENDER:
                    break
        print(gen, 'chosen:', len(chosen))
        pick += chosen
    P = pd.DataFrame(pick).sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)
    P['id'] = np.arange(1, len(P) + 1)
    P[['id', 'file', 'entity', 'video', 'gender']].rename(columns={'gender': 'predicted'}).to_csv(V + '/verify_key.csv', index=False)
    pd.DataFrame({'id': P['id'], 'true_gender': ''}).to_csv(V + '/verify_to_fill.csv', index=False)
    CELL = 220
    cols = 5
    for pg in range(int(np.ceil(len(P) / PER_PAGE))):
        sub = P.iloc[pg * PER_PAGE:(pg + 1) * PER_PAGE]
        rws = int(np.ceil(PER_PAGE / cols))
        sheet = np.full((rws * CELL, cols * CELL, 3), 255, np.uint8)
        for i, (_, r) in enumerate(sub.iterrows()):
            im = cv2.imread(D + '/crops/' + r.file)
            if im is None:
                continue
            h, w = im.shape[:2]
            s = (CELL - 30) / max(h, w)
            im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
            y0 = (i // cols) * CELL + 25
            x0 = (i % cols) * CELL + 5
            sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.putText(sheet, str(r['id']), (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        cv2.imwrite(V + '/sheet_%d.jpg' % (pg + 1), sheet)
    print('made', int(np.ceil(len(P) / PER_PAGE)), 'sheets in', V)
    print('fill', V + '/verify_to_fill.csv', '(true_gender = F, M or U). Do NOT open verify_key.csv first.')

if mode == 'eval':
    key = pd.read_csv(V + '/verify_key.csv')
    fill = pd.read_csv(V + '/verify_to_fill.csv')
    d = key.merge(fill, on='id')
    d['true_gender'] = d.true_gender.astype(str).str.strip().str.upper()
    print('rows:', len(d), ' labelled F/M:', d.true_gender.isin(['F', 'M']).sum(), ' U or blank (excluded):', (~d.true_gender.isin(['F', 'M'])).sum())
    d = d[d.true_gender.isin(['F', 'M'])]
    print('confusion (rows = predicted, columns = true):')
    print(pd.crosstab(d.predicted, d.true_gender))
    acc = (d.predicted == d.true_gender).mean()
    lo, hi = wilson(int((d.predicted == d.true_gender).sum()), len(d))
    print('overall accuracy %.1f%%  (95%% CI %.1f to %.1f)' % (100 * acc, 100 * lo, 100 * hi))
    for gen, other in [('F', 'M'), ('M', 'F')]:
        pr = d[d.predicted == gen]
        wrong = int((pr.true_gender == other).sum())
        lo, hi = wilson(wrong, len(pr))
        print('predicted %s: %d faces, wrong %d  -> error rate %.1f%% (95%% CI %.1f to %.1f)' % (gen, len(pr), wrong, 100 * wrong / max(len(pr), 1), 100 * lo, 100 * hi))
    for gen, other in [('F', 'M'), ('M', 'F')]:
        tr = d[d.true_gender == gen]
        miss = int((tr.predicted == other).sum())
        print('true %s: %d faces, recognised as %s: %d (%.1f%%)' % (gen, len(tr), other, miss, 100 * miss / max(len(tr), 1)))
    print('If the two error rates differ clearly, errors are asymmetric and the gap can be biased.')
