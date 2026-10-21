import os, sys, time, warnings
import numpy as np
import pandas as pd
import cv2
from insightface.app import FaceAnalysis
warnings.filterwarnings("ignore")

MAX_VIDEOS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
D = os.path.expanduser("~/Downloads/euqvid_temporal/data")
OUT = os.path.expanduser("~/Downloads/euqvid_temporal/outputs/ava_gender")
os.makedirs(OUT + "/crops", exist_ok=True)
CLIPS = D + "/AVA_videos/clips"
ANN_DIRS = [D + "/AVA/ava_activespeaker_train_v1.0", D + "/AVA/ava_activespeaker_test_v1.0"]
COLS = ["video", "ts", "x1", "y1", "x2", "y2", "label", "entity"]
N_FR, MIN_PX, IOU_MIN, SAVE_P = 3, 36, 0.3, 0.05
CACHE = OUT + "/entity_frames.csv"

rng = np.random.default_rng(0)
done = set(pd.read_csv(CACHE)["video"]) if os.path.exists(CACHE) else set()
vids = []
for fn in sorted(os.listdir(CLIPS)):
    if ".part" in fn or fn.endswith(".ytdl"):
        continue
    if os.path.splitext(fn)[1] not in [".mp4", ".webm", ".mkv"]:
        continue
    vids.append((os.path.splitext(fn)[0], fn))
todo = [v for v in vids if v[0] not in done][:MAX_VIDEOS]
if len(sys.argv) > 2:
    todo = [v for v in vids if v[0] == sys.argv[2] and v[0] not in done]
print("finished videos:", len(vids), "already processed:", len(done), "to process now:", len(todo))

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

for vid, fn in todo:
    t0 = time.time()
    path = None
    for d in ANN_DIRS:
        p = "%s/%s-activespeaker.csv" % (d, vid)
        if os.path.exists(p):
            path = p
    if path is None:
        print("no annotation for", vid)
        continue
    cap = cv2.VideoCapture(CLIPS + "/" + fn)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
    OFF = 0.0 if dur > 1000 else 900.0
    print('video', vid, 'duration %.0f s, time offset %.0f' % (dur, OFF))
    ann = pd.read_csv(path, header=None, names=COLS)
    ann = ann[(ann["x2"] - ann["x1"]) * W >= MIN_PX]
    picks = []
    for ent, g in ann.groupby("entity"):
        g = g.sort_values("ts")
        idx = np.unique(np.linspace(0, len(g) - 1, N_FR).round().astype(int))
        picks.append(g.iloc[idx])
    picks = pd.concat(picks)
    rec = []
    for ts, g in picks.groupby("ts"):
        cap.set(cv2.CAP_PROP_POS_MSEC, (ts - OFF) * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        faces = app.get(fr)
        for _, r in g.iterrows():
            box = [r["x1"] * W, r["y1"] * H, r["x2"] * W, r["y2"] * H]
            best, bf = 0.0, None
            for f in faces:
                v = iou(box, [float(z) for z in f.bbox])
                if v > best:
                    best, bf = v, f
            if bf is None or best < IOU_MIN:
                continue
            crop_path = ""
            if rng.random() < SAVE_P:
                x1, y1, x2, y2 = [int(z) for z in box]
                c = fr[max(y1, 0):y2, max(x1, 0):x2]
                if c.size > 0:
                    crop_path = "crops/%s_%s_%.2f.jpg" % (vid, str(r["entity"]).replace("/", "_").replace(":", "_"), ts)
                    cv2.imwrite(OUT + "/" + crop_path, cv2.resize(c, (112, 112)))
            rec.append({"video": vid, "entity": r["entity"], "ts": ts, "sex": bf.sex,
                        "det_score": float(bf.det_score), "iou": round(best, 2), "crop": crop_path})
    cap.release()
    new = pd.DataFrame(rec)
    new.to_csv(CACHE, mode="a", header=not os.path.exists(CACHE), index=False)
    n_ent = picks["entity"].nunique()
    print("%s: entities %d, frames matched %d, %.0f s (%.2f s per entity)" % (vid, n_ent, len(new), time.time() - t0, (time.time() - t0) / max(n_ent, 1)))

if os.path.exists(CACHE):
    fr_all = pd.read_csv(CACHE)
    agg = fr_all.groupby(["video", "entity"]).agg(n_votes=("sex", "size"), frac_F=("sex", lambda s: float((s == "F").mean()))).reset_index()
    agg["gender"] = np.where((agg["n_votes"] >= 2) & (agg["frac_F"] >= 0.8), "F",
                     np.where((agg["n_votes"] >= 2) & (agg["frac_F"] <= 0.2), "M", "unknown"))
    agg["gender_source"] = "predicted_insightface_buffalo_l"
    agg.to_csv(OUT + "/entity_gender.csv", index=False)
    print("entities with a label:", len(agg))
    print(agg["gender"].value_counts().to_string())
