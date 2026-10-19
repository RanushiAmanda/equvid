"""
C1 step 4 — build a BLIND manual validation sample.

WHY
---
Every gender label in use is predicted by a face model.  Stratifying on
face size ruled out the size pathway, but not a pose pathway: central
faces are more often frontal, and a labeller that answers "female" more
readily on frontal faces would reproduce the centrality result with no
difference in how anyone is filmed.

The only way to settle it is labels that did not come from a face model.
This script builds that set.

DESIGN DECISIONS THAT MATTER
----------------------------
1. BLIND.  The manifest the annotator sees carries no predicted label and
   is shuffled.  If the annotator sees the prediction first they will
   agree with it, and the exercise measures nothing.
2. STRATIFIED BY SIZE.  Equal numbers from each size band, so the
   agreement rate can be reported per band rather than dominated by the
   large easy faces.
3. 'unknown' AND 'missing' ARE INCLUDED.  They are 29% of the data and the
   whole question is what the model failed on.  A sample of only the
   confident predictions would measure nothing useful.
4. A REPRESENTATIVE FRAME.  The frame nearest the middle of the entity's
   appearance is used, not the first or the largest, so the crop is
   typical of how that person was filmed.

OUTPUT
------
  sample_manifest.csv  what the annotator works from: id, video,
                       timestamp, box. No label, shuffled.
  sample_key.csv       id -> entity + predicted label. Not opened until
                       the manual labels are finished.

Usage:
  python3 -I c1_make_validation_sample.py <ann_root> <entities.csv> \
      <videos_dir> <out_dir> [n_per_group]
"""

import csv
import os
import random
import sys
from collections import defaultdict

SEED = 0                      # fixed so the sample is reproducible
BANDS = [(0.00, 0.02), (0.02, 0.08), (0.08, 0.20), (0.20, 1.01)]
GROUPS = ["F", "M", "unknown", "missing"]


def parse_entity_id(eid):
    head, _, idx = eid.rpartition(":")
    if not idx:
        return None
    parts = head.rsplit("_", 2)
    if len(parts) != 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2]), idx
    except ValueError:
        return None


def main(ann_root, ent_path, videos_dir, out_dir, n_per_group=50):
    n_per_group = int(n_per_group)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED)

    # Only videos actually present locally can be cropped.
    local = set()
    for f in os.listdir(videos_dir):
        base, ext = os.path.splitext(f)
        if ext.lower() in (".mp4", ".mkv", ".webm"):
            local.add(base)
    print("videos available locally: %d" % len(local))

    ents = [r for r in csv.DictReader(open(ent_path, newline="",
                                           encoding="utf-8"))
            if r["video"] in local]
    for r in ents:
        r["mean_rel_area"] = float(r["mean_rel_area"])
    print("entities in those videos : %d" % len(ents))

    # Stratified draw: equal share from each size band within each group.
    chosen = []
    per_band = max(1, n_per_group // len(BANDS))
    for g in GROUPS:
        for lo, hi in BANDS:
            pool = [r for r in ents if r["gender"] == g
                    and lo <= r["mean_rel_area"] < hi]
            rng.shuffle(pool)
            take = pool[:per_band]
            chosen.extend(take)
            print("  %-8s area %.2f-%.2f : pool %5d, took %d"
                  % (g, lo, hi, len(pool), len(take)))
    print("sampled entities         : %d" % len(chosen))

    want = {r["entity"] for r in chosen}

    # Find a representative frame for each sampled entity: scan the
    # annotations once and keep every row belonging to a sampled entity.
    frames = defaultdict(list)
    for d in sorted(os.listdir(ann_root)):
        p = os.path.join(ann_root, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if not f.endswith("-activespeaker.csv"):
                continue
            vid = f.replace("-activespeaker.csv", "")
            if vid not in local:
                continue
            for row in csv.reader(open(os.path.join(p, f), newline="",
                                       encoding="utf-8")):
                if len(row) >= 8 and row[7] in want:
                    frames[row[7]].append(row)

    manifest, key = [], []
    for r in chosen:
        rows = frames.get(r["entity"])
        if not rows:
            continue
        rows.sort(key=lambda x: float(x[1]))
        mid = rows[len(rows) // 2]          # middle of the appearance
        sid = "s%04d" % (len(manifest) + 1)
        manifest.append({
            "sample_id": sid,
            "video": mid[0],
            "timestamp": mid[1],
            "x1": mid[2], "y1": mid[3], "x2": mid[4], "y2": mid[5],
        })
        key.append({
            "sample_id": sid,
            "entity": r["entity"],
            "predicted_gender": r["gender"],
            "frac_F": r["frac_F"],
            "mean_rel_area": "%.6f" % r["mean_rel_area"],
            "mean_centre_dist": r["mean_centre_dist"],
        })

    # Shuffle so neither size nor predicted label is guessable from order.
    order = list(range(len(manifest)))
    rng.shuffle(order)
    manifest = [manifest[i] for i in order]
    # manual_gender is left blank on purpose: that is the annotator's column.
    for m in manifest:
        m["manual_gender"] = ""
        m["notes"] = ""

    mp = os.path.join(out_dir, "sample_manifest.csv")
    with open(mp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "video", "timestamp",
                                           "x1", "y1", "x2", "y2",
                                           "manual_gender", "notes"])
        w.writeheader()
        w.writerows(manifest)

    kp = os.path.join(out_dir, "sample_key.csv")
    with open(kp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "entity",
                                           "predicted_gender", "frac_F",
                                           "mean_rel_area",
                                           "mean_centre_dist"])
        w.writeheader()
        w.writerows(key)

    print()
    print("wrote %s  (%d rows, no labels, shuffled)" % (mp, len(manifest)))
    print("wrote %s  (do not open until labelling is done)" % kp)


if __name__ == "__main__":
    main(*sys.argv[1:])
