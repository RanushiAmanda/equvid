"""
MELD face presence for ALL splits (dev + train + test). Resumable: run it again and it continues.
Same method as notebook 05 (1 frame per second, face-gender from the dev gallery, faces >= 40 px wide).
Output: outputs/meld_face/presence_by_clip_all.csv  (split, dia, utt, samples, dt, pres_F, pres_M, pres_U)
Run:  python3 src/meld_presence_all.py            (test first, then train)
      python3 src/meld_presence_all.py test       (only test)
"""
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import cv2
warnings.filterwarnings('ignore')

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
OUT = ROOT + '/outputs/meld_face'
RES = OUT + '/presence_by_clip_all.csv'
RAW = ROOT + '/data_videos/meld_raw'
CH = ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross']
GENDER = {'Chandler': 'M', 'Joey': 'M', 'Ross': 'M', 'Monica': 'F', 'Phoebe': 'F', 'Rachel': 'F'}
K, MIN_SIM, MIN_W = 5, 0.3, 40
ORDER = sys.argv[1:] or ['test', 'train']

m = pd.read_csv(ROOT + '/outputs/meld_intervals_words.csv')
m = m[m.gender.isin(['F', 'M'])]
need = set(zip(m.split, m.dialogue_id.astype(int), m.utterance_id.astype(int)))
print('clips needed (main-character utterances):', len(need), m.split.value_counts().to_dict())

paths = {}
for p in glob.glob(RAW + '/**/dia*_utt*.mp4', recursive=True):
    rel = os.path.relpath(p, RAW).lower()
    sp = 'train' if 'train' in rel else ('test' if 'test' in rel else ('dev' if 'dev' in rel else None))
    if sp is None:
        continue
    b = os.path.basename(p)[:-4].split('_')
    try:
        key = (sp, int(b[0][3:]), int(b[1][3:]))
    except ValueError:
        continue
    paths.setdefault(key, p)
cnt = pd.Series([k[0] for k in paths]).value_counts().to_dict()
print('mp4 files found per split:', cnt)
print('needed clips that have a video:', {s: sum(1 for k in need if k[0] == s and k in paths) for s in sorted(set(k[0] for k in need))})

# ---- previous results (dev from notebook 05 are re-used, not recomputed)
cols = ['split', 'dia', 'utt', 'samples', 'dt', 'pres_F', 'pres_M', 'pres_U']
if os.path.exists(RES):
    done_df = pd.read_csv(RES)
else:
    done_df = pd.DataFrame(columns=cols)
    old = OUT + '/presence_by_clip.csv'
    if os.path.exists(old):
        o = pd.read_csv(old)
        o.insert(0, 'split', 'dev')
        done_df = o[cols]
        print('dev rows copied from presence_by_clip.csv:', len(done_df))
done = set(zip(done_df.split, done_df.dia.astype(int), done_df.utt.astype(int)))
todo = [k for s in ORDER for k in sorted(need) if k[0] == s and k in paths and k not in done]
print('already done:', len(done), '| still to do now:', len(todo))
if not todo:
    print('nothing to do')
    sys.exit(0)

# ---- face-gender from the dev gallery (same as notebook 05)
from insightface.app import FaceAnalysis
E = np.load(OUT + '/gallery_v2_embs.npy')
gm = pd.read_csv(OUT + '/gallery_v2_meta.csv')
idx_c = {c: np.where(gm['character'].to_numpy() == c)[0] for c in CH}
F_CH = [c for c in CH if GENDER[c] == 'F']
M_CH = [c for c in CH if GENDER[c] == 'M']


def face_gender(emb):
    sc = {c: np.sort(E[idx_c[c]] @ emb)[-K:].mean() for c in CH}
    f = max(sc[c] for c in F_CH)
    mm = max(sc[c] for c in M_CH)
    if max(f, mm) < MIN_SIM:
        return 'U'
    return 'F' if f > mm else 'M'


app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

rec = []


def save():
    pd.concat([done_df, pd.DataFrame(rec, columns=cols)], ignore_index=True).to_csv(RES, index=False)


try:
    for i, key in enumerate(todo):
        sp, d, u = key
        cap = cv2.VideoCapture(paths[key])
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        step = max(int(round(fps)), 1)
        dt = step / fps
        c = {'F': 0, 'M': 0, 'U': 0}
        ns, j = 0, 0
        while True:
            if not cap.grab():
                break
            if j % step == 0:
                ok, fr = cap.retrieve()
                if ok:
                    ns += 1
                    for f in app.get(fr):
                        if (f.bbox[2] - f.bbox[0]) >= MIN_W:
                            c[face_gender(f.normed_embedding)] += 1
            j += 1
        cap.release()
        rec.append((sp, d, u, ns, dt, c['F'] * dt, c['M'] * dt, c['U'] * dt))
        if (i + 1) % 50 == 0:
            save()
            print('processed %d of %d' % (i + 1, len(todo)), flush=True)
finally:
    save()
    print('saved', RES, '| rows now:', len(done_df) + len(rec))
