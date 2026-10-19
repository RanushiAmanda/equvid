"""
C1 — MELD extraction. Runs on the machine that holds MELD.Raw.

WHY NO FACE RECOGNITION IS NEEDED
---------------------------------
MELD's CSVs give the SPEAKER of each utterance, not who is on screen, so
at first sight C1 needs to recognise which of the six main characters each
detected face belongs to.  It does not, because of how the corpus is cut:
one clip is one utterance by one speaker, averaging 3.6 s, and in a
multi-camera sitcom the shot almost always shows the person talking.

So this script keeps only the clips where exactly ONE face is on screen
throughout.  In those, the visible face is the speaker, and the speaker's
gender is already known from a hand-written map of six characters.  No
face recognition model, no gender classifier, no new dependency -- and,
unlike AVA, no predicted labels anywhere in the chain.

WHAT THAT COSTS, STATED PLAINLY
-------------------------------
Single-face clips are not a random sample of MELD: they are the clips shot
in close coverage.  So these numbers describe "how the two genders are
framed when shown alone", not all of MELD.  Headcount and solo-framing
cannot be measured this way at all.  Face scale and centrality can, and
those are two of the three dimensions AVA left standing.

The multi-face clips are still counted and reported, so the share excluded
is visible rather than hidden.

RESUMABLE
---------
Results are appended per clip and already-processed clips are skipped, so
the run can be stopped and restarted, or left overnight.

Usage:
  python c1_meld_extract.py <MELD.Raw dir> <out.csv> [max_clips]

<MELD.Raw dir> is the folder holding dev_sent_emo.csv and dev/.
"""

import csv
import math
import os
import sys
import time

import cv2

# Hand-written, not inferred.  The six leads only; every other speaker is
# excluded and the excluded share is reported.
GENDER = {
    "Monica": "F", "Phoebe": "F", "Rachel": "F",
    "Chandler": "M", "Joey": "M", "Ross": "M",
}

FRAME_STEP = 8        # sample every 8th frame; clips average 3.6 s
DETECT_WIDTH = 640    # downscale before detection. Every feature here is a
                      # RATIO (area as a fraction of the frame, centre
                      # distance as a fraction of the half-diagonal), so
                      # resizing changes nothing in the output while making
                      # MTCNN several times faster on CPU.
CONF = 0.90           # detector confidence floor, as in the proposal
CLOSEUP_AREA = 0.08   # same threshold as the AVA run, so the two compare
SINGLE_FACE_MIN = 0.80  # a clip counts as single-face when at least this
                        # share of its sampled frames hold exactly one face


def load_detector():
    """MTCNN if present (it is what the pilot used); YuNet as a fallback.

    Returns (name, callable) where the callable takes a BGR frame and
    returns a list of (x, y, w, h) boxes in pixels.
    """
    try:
        from mtcnn import MTCNN
        det = MTCNN()

        def run(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return [f["box"] for f in det.detect_faces(rgb)
                    if f["confidence"] >= CONF]
        return "mtcnn", run
    except Exception:
        pass

    model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(model):
        raise SystemExit(
            "No detector available.\n"
            "Either:  pip install mtcnn\n"
            "or download YuNet next to this script:\n"
            "  https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx")
    det = cv2.FaceDetectorYN.create(model, "", (320, 320),
                                    score_threshold=CONF)

    def run(frame):
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]
    return "yunet", run


def clip_path(root, split, dia, utt):
    folder = {"dev": "dev_splits_complete",
              "train": "train_splits",
              "test": "output_repeated_splits_test"}[split]
    for base in (os.path.join(root, split, folder),
                 os.path.join(root, split)):
        p = os.path.join(base, "dia%s_utt%s.mp4" % (dia, utt))
        if os.path.exists(p):
            return p
    return None


def process(path, run):
    """Per-clip geometry. Returns None when the clip cannot be read."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not W or not H:
        cap.release()
        return None
    half_diag = math.hypot(0.5, 0.5)

    idx = sampled = single = with_face = 0
    areas, dists, closeups = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % FRAME_STEP == 0:
            sampled += 1
            if W > DETECT_WIDTH:
                sc = DETECT_WIDTH / W
                small = cv2.resize(frame, (int(W * sc), int(H * sc)),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = frame
            fh_, fw_ = small.shape[:2]
            faces = run(small)
            if faces:
                with_face += 1
            if len(faces) == 1:
                single += 1
                x, y, w, h = faces[0]
                a = (w * h) / float(fw_ * fh_)
                cx, cy = (x + w / 2.0) / fw_, (y + h / 2.0) / fh_
                areas.append(a)
                dists.append(math.hypot(cx - 0.5, cy - 0.5) / half_diag)
                closeups.append(1 if a >= CLOSEUP_AREA else 0)
        idx += 1
    cap.release()
    if sampled == 0:
        return None
    return {
        "frames_sampled": sampled,
        "frames_with_face": with_face,
        "frames_single_face": single,
        "single_share": single / sampled,
        "mean_rel_area": (sum(areas) / len(areas)) if areas else "",
        "mean_centre_dist": (sum(dists) / len(dists)) if dists else "",
        "closeup_rate": (sum(closeups) / len(closeups)) if closeups else "",
    }


def main(root, out_path, max_clips=0):
    max_clips = int(max_clips)
    name, run = load_detector()
    print("detector: %s" % name)

    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add(r["clip_id"])
        print("already done: %d clips (will be skipped)" % len(done))

    cols = ["clip_id", "split", "season", "episode", "dialogue", "utterance",
            "speaker", "gender", "frames_sampled", "frames_with_face",
            "frames_single_face", "single_share", "mean_rel_area",
            "mean_centre_dist", "closeup_rate", "status"]
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="", encoding="utf-8")
    wtr = csv.DictWriter(fh, fieldnames=cols)
    if new:
        wtr.writeheader()

    n = skipped_speaker = missing = 0
    t0 = time.time()
    # dev first: it is the smallest split, so problems surface in minutes
    # rather than after an hour of train.
    for split in ("dev", "train", "test"):
        # The release is inconsistent: dev/test CSVs sit at the root while
        # train's sits inside train/. Check both rather than assume.
        csv_path = None
        for cand in (os.path.join(root, "%s_sent_emo.csv" % split),
                     os.path.join(root, split, "%s_sent_emo.csv" % split)):
            if os.path.exists(cand):
                csv_path = cand
                break
        if csv_path is None:
            print("%s: no CSV found, skipping" % split)
            continue
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
        print("%s: %d utterances" % (split, len(rows)))
        for r in rows:
            sp = r["Speaker"].strip()
            if sp not in GENDER:
                skipped_speaker += 1       # guest character -> excluded
                continue
            cid = "MELD:%s:d%s_u%s" % (split, r["Dialogue_ID"],
                                       r["Utterance_ID"])
            if cid in done:
                continue
            p = clip_path(root, split, r["Dialogue_ID"], r["Utterance_ID"])
            if not p:
                missing += 1
                continue
            res = process(p, run)
            row = {"clip_id": cid, "split": split, "season": r["Season"],
                   "episode": r["Episode"], "dialogue": r["Dialogue_ID"],
                   "utterance": r["Utterance_ID"], "speaker": sp,
                   "gender": GENDER[sp]}
            if res is None:
                row["status"] = "unreadable"
            else:
                row.update(res)
                row["status"] = ("single_face"
                                 if res["single_share"] >= SINGLE_FACE_MIN
                                 else "multi_or_no_face")
            wtr.writerow(row)
            n += 1
            if n % 10 == 0:
                fh.flush()
                el = time.time() - t0
                print("  %d clips | %.1f s/clip | %.0f s elapsed"
                      % (n, el / n, el), flush=True)
            if max_clips and n >= max_clips:
                print("reached max_clips=%d" % max_clips)
                fh.close()
                return
    fh.close()
    print()
    print("processed this run     : %d" % n)
    print("skipped, guest speaker : %d" % skipped_speaker)
    print("clip file not found    : %d" % missing)
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
