"""
C1 step 3 — is the centrality result a size artefact? Stratified test.

THE PROBLEM
-----------
The label audit showed the labeller fails on small faces, and small faces
sit further from frame centre.  So "women are more central" could be
produced entirely by differential dropout: if more small (peripheral)
female faces fall into `unknown`, the surviving female sample is pulled
toward the centre without any difference in how the two genders are filmed.

THE TEST
--------
Compare F and M *within narrow bands of face size*.  Inside a band, both
genders have faces of essentially the same size, so a size-driven dropout
cannot produce a gap.  If the centrality gap survives in band after band,
size-dependent labelling cannot be the explanation.

This is stratification on the confounder — the standard way to answer this
question, and much cheaper than re-labelling.

A second check uses only mixed-gender windows, so the comparison is always
between people filmed in the same scene, not across different films.

Usage: python3 -I c1_stratify.py <entities.csv>
"""

import csv
import math
import statistics as st
import sys
from collections import defaultdict

BANDS = [(0.00, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08),
         (0.08, 0.16), (0.16, 0.32), (0.32, 1.01)]


def welch(a, b):
    """Welch's t statistic — unequal variances, unequal n. Returns (t, df)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), 0
    ma, mb = st.fmean(a), st.fmean(b)
    va, vb = st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), 0
    t = (ma - mb) / se
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    return t, (num / den if den else 0)


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    s = math.sqrt(((na - 1) * st.variance(a) + (nb - 1) * st.variance(b))
                  / (na + nb - 2))
    return (st.fmean(a) - st.fmean(b)) / s if s else float("nan")


def main(path):
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["gender"] not in ("F", "M"):
            continue
        r["mean_rel_area"] = float(r["mean_rel_area"])
        r["mean_centre_dist"] = float(r["mean_centre_dist"])
        r["n_frames"] = int(r["n_frames"])
        rows.append(r)

    print("=" * 72)
    print("STRATIFIED BY FACE SIZE — centre distance (lower = more central)")
    print("=" * 72)
    print("%-14s %7s %7s %9s %9s %8s %8s" %
          ("area band", "n_F", "n_M", "F dist", "M dist", "diff", "Cohen d"))
    print("-" * 72)

    agree = 0
    bands_used = 0
    for lo, hi in BANDS:
        F = [r["mean_centre_dist"] for r in rows
             if r["gender"] == "F" and lo <= r["mean_rel_area"] < hi]
        M = [r["mean_centre_dist"] for r in rows
             if r["gender"] == "M" and lo <= r["mean_rel_area"] < hi]
        if len(F) < 20 or len(M) < 20:
            print("%-14s %7d %7d   (too few to compare)"
                  % ("%.2f-%.2f" % (lo, hi), len(F), len(M)))
            continue
        bands_used += 1
        mF, mM = st.fmean(F), st.fmean(M)
        d = cohen_d(F, M)
        if mF < mM:          # women closer to centre
            agree += 1
        print("%-14s %7d %7d %9.4f %9.4f %+8.4f %8.3f"
              % ("%.2f-%.2f" % (lo, hi), len(F), len(M), mF, mM, mF - mM, d))

    print("-" * 72)
    print("bands where women are more central: %d of %d" % (agree, bands_used))
    print()
    print("Reading: if the sign of 'diff' stays negative across bands, the")
    print("gap is not produced by size-dependent label dropout, because")
    print("within a band both genders have the same face size.")

    # ---- within-window paired test ------------------------------------
    # Comparing people filmed in the same scene removes film-level
    # differences (genre, era, director) as an explanation.
    print()
    print("=" * 72)
    print("WITHIN-WINDOW TEST — same scene, same moment")
    print("=" * 72)
    byw = defaultdict(lambda: {"F": [], "M": []})
    for r in rows:
        byw[(r["video"], r["window"])][r["gender"]].append(r)

    diffs, wdiffs = [], []
    for _, g in byw.items():
        if not g["F"] or not g["M"]:
            continue
        # Frame-weighted mean inside the window, so a person on screen
        # longer counts proportionally.
        def wmean(es):
            n = sum(e["n_frames"] for e in es)
            return sum(e["mean_centre_dist"] * e["n_frames"] for e in es) / n
        dF, dM = wmean(g["F"]), wmean(g["M"])
        diffs.append(dF - dM)
        wdiffs.append((dF - dM, sum(e["n_frames"] for e in g["F"] + g["M"])))

    if diffs:
        t, df = welch(diffs, [0.0] * len(diffs))
        neg = sum(1 for x in diffs if x < 0)
        print("  mixed-gender windows      : %d" % len(diffs))
        print("  mean (F dist - M dist)    : %+.4f" % st.fmean(diffs))
        print("  median                    : %+.4f" % st.median(diffs))
        print("  women more central in     : %d (%.0f%%)"
              % (neg, 100.0 * neg / len(diffs)))
        sd = st.pstdev(diffs)
        se = sd / math.sqrt(len(diffs))
        m = st.fmean(diffs)
        print("  95%% CI on the mean        : [%+.4f, %+.4f]"
              % (m - 1.96 * se, m + 1.96 * se))
        print()
        print("  A CI that excludes 0 means the within-scene gap is larger")
        print("  than sampling noise. It does NOT prove the labels are sound.")


if __name__ == "__main__":
    main(sys.argv[1])
