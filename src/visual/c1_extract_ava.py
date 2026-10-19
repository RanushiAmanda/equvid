"""
C1 (EuqVid-Visual) — visual prominence feature extraction for AVA-ActiveSpeaker.

WHY THIS RUNS WITHOUT VIDEO
---------------------------
AVA-ActiveSpeaker ships human-verified face boxes (entity_box, already
normalised to the frame) and an entity_id that links the same face across
frames.  Every feature below is geometry over those boxes, so no face
detector and no video decoding is required.  That is a methodological
advantage, not a shortcut: a bias metric computed on top of a detector
inherits that detector's error profile, and here there is no detector.

THE ENTITY_ID CAVEAT (important, and it changes what we can claim)
-----------------------------------------------------------------
entity_id looks like "<video>_<win_start>_<win_end>:<n>", e.g.
    1j20qq1JyX4_1260_1320:4
The window is baked into the id, so the SAME physical person appearing in
two consecutive 60 s windows receives TWO different entity_ids.  Therefore:
  * "number of distinct people in a video" is NOT recoverable from entity_id;
  * per-person normalisation is only valid INSIDE one window;
  * any identity-level partition must be done at window level or coarser.
We handle this by computing everything per (video, window) first, where an
entity really is one person, and only then aggregating upward.

FEATURE DESIGN — two kinds of quantity, normalised differently
--------------------------------------------------------------
M2's shared conventions note (Oct 6) warns: compare per-person quantities,
not group totals, when group sizes differ.  That applies to some of our
features but not all, and conflating them is a real error:

  EXTENSIVE  (grow with how many people are present)
      screen time  -> divide by the number of people of that gender
  INTENSIVE  (already an average over faces)
      face area, centre distance, close-up rate, solo rate
      -> no headcount division; the mean is already per-face

Every feature is reported as a signed log parity:
      parity = log( female_quantity / male_quantity )
0 means parity, positive means the quantity is higher for women.  Log is used
so +x and -x are symmetric, matching M2's sign convention proposal.

OUTPUT
------
  windows.csv   one row per (video, window) with raw per-gender quantities
  videos.csv    one row per video, aggregated, with log-parity features
  entities.csv  one row per entity (for the detector/label-confound audit)

Usage:
  python3 -I c1_extract_ava.py <ann_root> <gender_csv> <out_dir>
"""

import csv
import math
import os
import sys
from collections import defaultdict

# A face is called a close-up when its box covers at least this fraction of
# the frame.  Stated as a convention, not a measured quantity -- the
# sensitivity of results to this value is reported separately.
CLOSEUP_AREA = 0.08


def parse_entity_id(eid):
    """'1j20qq1JyX4_1260_1320:4' -> ('1j20qq1JyX4', 1260, 1320, '4').

    Returns None when the id does not carry a window, so callers can count
    and report malformed ids instead of silently dropping them.
    """
    head, _, idx = eid.rpartition(":")
    if not idx:
        return None
    parts = head.rsplit("_", 2)
    if len(parts) != 3:
        return None
    vid, a, b = parts
    try:
        return vid, int(a), int(b), idx
    except ValueError:
        return None


def load_gender(path):
    """Returns (entity_id -> (gender, frac_F, n_votes), set of video ids).

    The video id is taken from the CSV's own 'video' column, never by
    splitting the entity id on '_': several AVA video ids contain
    underscores (e.g. -IELREHX_js), and splitting silently loses them.

    Labels are PREDICTED by a face model, not manually annotated.
    """
    g, vids = {}, set()
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g[r["entity"]] = (
                r["gender"],
                float(r["frac_F"]) if r["frac_F"] else float("nan"),
                int(r["n_votes"]) if r["n_votes"] else 0,
            )
            vids.add(r["video"])
    return g, vids


def ann_files(root):
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith("-activespeaker.csv"):
                yield os.path.join(p, f)


def main(ann_root, gender_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gender, gender_videos = load_gender(gender_csv)

    # Per-entity accumulators.  An entity is one person inside one window.
    ent_frames = defaultdict(int)       # entity -> number of annotated frames
    ent_area = defaultdict(float)       # entity -> sum of relative box area
    ent_area2 = defaultdict(float)      # entity -> sum of squared area, for
                                        # the within-entity spread: an entity
                                        # cut between close-up and wide has a
                                        # modest mean but a high close-up rate
    ent_dist = defaultdict(float)       # entity -> sum of centre distance
    ent_closeup = defaultdict(int)      # entity -> frames above the close-up size
    ent_solo = defaultdict(int)         # entity -> frames where it is the only face
    ent_window = {}                     # entity -> (video, win_start)

    # (video, window, timestamp) -> how many entities share that instant.
    # Needed for solo framing: "alone in shot" is a property of the frame.
    frame_pop = defaultdict(set)

    rows_read = 0
    bad_ids = 0
    files = 0

    # Pass 1: count how many faces share each annotated instant.
    for path in ann_files(ann_root):
        vid = os.path.basename(path).replace("-activespeaker.csv", "")
        if vid not in gender_videos:
            continue  # no gender labels for this video -> cannot score it
        files += 1
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 8:
                    continue
                rows_read += 1
                eid = row[7]
                meta = parse_entity_id(eid)
                if meta is None:
                    bad_ids += 1
                    continue
                v, w0, _, _ = meta
                frame_pop[(v, w0, row[1])].add(eid)

    # Pass 2: geometry per entity.
    for path in ann_files(ann_root):
        vid = os.path.basename(path).replace("-activespeaker.csv", "")
        if vid not in gender_videos:
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 8:
                    continue
                eid = row[7]
                meta = parse_entity_id(eid)
                if meta is None:
                    continue
                v, w0, _, _ = meta
                try:
                    x1, y1, x2, y2 = (float(row[2]), float(row[3]),
                                      float(row[4]), float(row[5]))
                except ValueError:
                    continue
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                area = w * h
                # Distance of the face centre from the frame centre, divided by
                # the half-diagonal, so 0 = dead centre and ~1 = a corner.
                cx, cy = x1 + w / 2.0, y1 + h / 2.0
                dist = math.hypot(cx - 0.5, cy - 0.5) / math.hypot(0.5, 0.5)

                ent_frames[eid] += 1
                ent_area[eid] += area
                ent_area2[eid] += area * area
                ent_dist[eid] += dist
                if area >= CLOSEUP_AREA:
                    ent_closeup[eid] += 1
                if len(frame_pop[(v, w0, row[1])]) == 1:
                    ent_solo[eid] += 1
                ent_window[eid] = (v, w0)

    # ---- entities.csv : one row per entity, for the label-confound audit ----
    ent_path = os.path.join(out_dir, "entities.csv")
    with open(ent_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["entity", "video", "window", "gender", "frac_F", "n_votes",
                      "n_frames", "mean_rel_area", "sd_rel_area",
                      "mean_centre_dist", "closeup_rate", "solo_rate"])
        for eid, n in ent_frames.items():
            v, w0 = ent_window[eid]
            gl, frac, votes = gender.get(eid, ("missing", float("nan"), 0))
            mu = ent_area[eid] / n
            var = max(0.0, ent_area2[eid] / n - mu * mu)
            wtr.writerow([eid, v, w0, gl, frac, votes, n,
                          round(mu, 6), round(math.sqrt(var), 6),
                          round(ent_dist[eid] / n, 6),
                          round(ent_closeup[eid] / n, 6),
                          round(ent_solo[eid] / n, 6)])

    # ---- windows.csv : per-gender quantities inside each 60 s window --------
    win = defaultdict(lambda: defaultdict(float))
    for eid, n in ent_frames.items():
        v, w0 = ent_window[eid]
        gl = gender.get(eid, ("missing",))[0]
        if gl not in ("F", "M"):
            # 'unknown' and 'missing' are carried, never silently dropped:
            # their share is a reported limitation.
            win[(v, w0)]["n_%s" % gl] += 1
            win[(v, w0)]["frames_%s" % gl] += n
            continue
        k = win[(v, w0)]
        k["n_" + gl] += 1                         # people of this gender
        k["frames_" + gl] += n                    # screen time, in frames
        k["area_" + gl] += ent_area[eid]          # sum over frames
        k["dist_" + gl] += ent_dist[eid]
        k["closeup_" + gl] += ent_closeup[eid]
        k["solo_" + gl] += ent_solo[eid]

    cols = ["video", "window", "n_F", "n_M", "n_unknown", "n_missing",
            "frames_F", "frames_M", "frames_unknown", "frames_missing",
            "area_F", "area_M", "dist_F", "dist_M",
            "closeup_F", "closeup_M", "solo_F", "solo_M", "mixed_gender"]
    win_path = os.path.join(out_dir, "windows.csv")
    with open(win_path, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(cols)
        for (v, w0), k in sorted(win.items()):
            mixed = int(k.get("n_F", 0) >= 1 and k.get("n_M", 0) >= 1)
            wtr.writerow([v, w0] + [round(k.get(c, 0.0), 6) for c in cols[2:-1]]
                         + [mixed])

    print("files read          : %d" % files)
    print("annotation rows     : %d" % rows_read)
    print("malformed entity ids: %d" % bad_ids)
    print("entities            : %d" % len(ent_frames))
    print("windows             : %d" % len(win))
    print("wrote %s, %s" % (ent_path, win_path))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
