"""
C1 step 7 — the strongest available test: recompute the centrality gap
from HUMAN gender labels instead of the face model's.

WHY THIS TEST MATTERS
---------------------
Everything so far rests on predicted labels.  Stratifying on face size
ruled out the size pathway.  It could not rule out the pathway where the
model mistakes central men for women, because that error lives inside the
labels themselves.

The validation sample now carries a human label for 167 entities (160
usable), and the centre distance of each one is already measured.  So the
gap can be computed directly from human labels and compared with the gap
computed from model labels.  If they agree, the finding survives.  If the
human version shrinks to nothing, the finding was the model's error all
along.

WHAT IT DOES NOT SETTLE
-----------------------
The gender VALUES compared are human, but the model is not entirely out of
the chain: the population weights below use the model's predicted group
(and face size) as strata, and the face boxes are the dataset's own.  The
160 labels come from 20 of the 88 videos and one annotator, and the
interval resamples entities within cells, not videos, so it is narrower
than a video-level interval would be.  Read the result as evidence on the
DIRECTION of the label bias, not as a population estimate.

WEIGHTING
---------
The sample is deliberately NOT representative: it was drawn equally across
four predicted groups and four size bands so that rare cells could be
measured.  Comparing it to the population without correction would be
wrong.  Each sampled entity is therefore weighted by

        population count of its (predicted group, size band) cell
        -------------------------------------------------------
        sample count of that same cell

which restores the population composition.  Unweighted figures are printed
alongside so the effect of the weighting is visible rather than hidden.

Usage: python3 -I c1_decide.py <entities.csv> <sample_key.csv> \
         <manual_labels.csv>
"""

import csv
import math
import random
import statistics as st
import sys
from collections import defaultdict

BANDS = [(0.00, 0.02), (0.02, 0.08), (0.08, 0.20), (0.20, 1.01)]
SEED = 0
BOOT = 2000


def band_of(a):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= a < hi:
            return i
    return len(BANDS) - 1


def wmean(vals, wts):
    s = sum(wts)
    return sum(v * w for v, w in zip(vals, wts)) / s if s else float("nan")


def main(ent_path, key_path, man_path):
    # Population cell counts, from the full extraction.
    pop = defaultdict(int)
    for r in csv.DictReader(open(ent_path)):
        pop[(r["gender"], band_of(float(r["mean_rel_area"])))] += 1

    key = {r["sample_id"]: r for r in csv.DictReader(open(key_path))}
    man = {r["sample_id"]: r["manual_gender"].strip().upper()
           for r in csv.DictReader(open(man_path))
           if r["manual_gender"].strip()}

    rows = []
    for sid, m in man.items():
        k = key.get(sid)
        if not k or m not in ("F", "M"):
            continue
        cell = (k["predicted_gender"], band_of(float(k["mean_rel_area"])))
        rows.append({"man": m, "cell": cell,
                     "dist": float(k["mean_centre_dist"]),
                     "area": float(k["mean_rel_area"])})

    smp = defaultdict(int)
    for r in rows:
        smp[r["cell"]] += 1
    for r in rows:
        r["w"] = pop[r["cell"]] / smp[r["cell"]]

    F = [r for r in rows if r["man"] == "F"]
    M = [r for r in rows if r["man"] == "M"]
    print("manually confirmed: %d women, %d men" % (len(F), len(M)))

    print()
    print("=" * 70)
    print("CENTRE DISTANCE BY LABEL SOURCE   (lower = closer to frame centre)")
    print("=" * 70)

    # Model labels, whole population -- the original finding.
    pF, pM = [], []
    for r in csv.DictReader(open(ent_path)):
        if r["gender"] == "F":
            pF.append(float(r["mean_centre_dist"]))
        elif r["gender"] == "M":
            pM.append(float(r["mean_centre_dist"]))
    print("  MODEL labels (all %d entities)" % (len(pF) + len(pM)))
    print("     F %.4f   M %.4f   gap %+.4f"
          % (st.fmean(pF), st.fmean(pM), st.fmean(pF) - st.fmean(pM)))

    uF, uM = st.fmean([r["dist"] for r in F]), st.fmean([r["dist"] for r in M])
    print("  MANUAL labels, unweighted (n=%d)" % len(rows))
    print("     F %.4f   M %.4f   gap %+.4f" % (uF, uM, uF - uM))

    wF = wmean([r["dist"] for r in F], [r["w"] for r in F])
    wM = wmean([r["dist"] for r in M], [r["w"] for r in M])
    gap = wF - wM
    print("  MANUAL labels, population-weighted")
    print("     F %.4f   M %.4f   gap %+.4f" % (wF, wM, gap))

    # Bootstrap the weighted gap: resample entities within cells, which is
    # how the sample was drawn, so the interval reflects the real design.
    rng = random.Random(SEED)
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    gaps = []
    for _ in range(BOOT):
        res = []
        for cell, items in by_cell.items():
            res.extend(rng.choice(items) for _ in items)
        f = [r for r in res if r["man"] == "F"]
        m = [r for r in res if r["man"] == "M"]
        if not f or not m:
            continue
        gaps.append(wmean([r["dist"] for r in f], [r["w"] for r in f])
                    - wmean([r["dist"] for r in m], [r["w"] for r in m]))
    gaps.sort()
    lo = gaps[int(0.025 * len(gaps))]
    hi = gaps[int(0.975 * len(gaps))]
    print("     95%% CI on the gap  [%+.4f, %+.4f]   (%d bootstrap draws)"
          % (lo, hi, len(gaps)))

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    model_gap = st.fmean(pF) - st.fmean(pM)
    if hi < 0:
        print("  The gap is still negative with human labels and the interval")
        print("  excludes zero: women ARE framed closer to centre, and the")
        print("  model's errors are not what produced it.")
    elif lo > 0:
        print("  The gap REVERSES with human labels. The original result was")
        print("  an artefact of the face model.")
    else:
        print("  The interval includes zero. With human labels this sample")
        print("  cannot establish a centrality gap in either direction.")
        print("  The model-label finding is NOT confirmed.")
    print()
    print("  model-label gap  %+.4f" % model_gap)
    print("  human-label gap  %+.4f   (%.0f%% of the model-label gap)"
          % (gap, 100.0 * gap / model_gap if model_gap else float("nan")))
    print()
    print("  n is %d people. This settles the direction of the bias in the" % len(rows))
    print("  labels, not the population value of the gap.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
