"""
C1 audit step 1 — can the gender labels be trusted for THIS measurement?

The labels in entity_gender.csv are produced by a face model
(insightface buffalo_l), not by hand.  C1 measures how large and how
central each gender's faces are.  So there is a circularity risk that has
to be measured before any score is reported:

    if the labeller fails more often on small, off-centre faces,
    and one gender is filmed smaller and further from centre,
    then that gender's faces are dropped more often,
    and the surviving sample is biased toward the opposite conclusion.

This script measures that directly.  It is a negative-control style check:
we WANT label availability to be unrelated to face geometry, and if it is
not, the size of the effect is a declared limitation.

Usage: python3 -I c1_audit.py <entities.csv>
"""

import csv
import math
import statistics as st
import sys


def pct(a, b):
    return 0.0 if b == 0 else 100.0 * a / b


def describe(name, vals):
    if not vals:
        print("  %-22s n=0" % name)
        return
    vals = sorted(vals)
    print("  %-22s n=%-6d mean=%.4f  median=%.4f" %
          (name, len(vals), st.fmean(vals), vals[len(vals) // 2]))


def main(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    for r in rows:
        for k in ("n_frames",):
            r[k] = int(r[k])
        for k in ("mean_rel_area", "mean_centre_dist", "closeup_rate",
                  "solo_rate"):
            r[k] = float(r[k])

    n = len(rows)
    by = {}
    for r in rows:
        by.setdefault(r["gender"], []).append(r)

    print("=" * 68)
    print("1. LABEL COVERAGE  (entities = one person inside one 60 s window)")
    print("=" * 68)
    for g in ("F", "M", "unknown", "missing"):
        k = len(by.get(g, []))
        print("  %-9s %6d   %5.1f%%" % (g, k, pct(k, n)))
    labelled = len(by.get("F", [])) + len(by.get("M", []))
    unlab = n - labelled
    print("  %-9s %6d   %5.1f%%   <- excluded share, must be reported"
          % ("UNUSABLE", unlab, pct(unlab, n)))
    if by.get("M") and by.get("F"):
        print("  M:F entity ratio = %.2f : 1"
              % (len(by["M"]) / len(by["F"])))

    print()
    print("=" * 68)
    print("2. CONFOUND TEST — is label availability related to face geometry?")
    print("=" * 68)
    print("  If unlabelled faces are systematically smaller or more")
    print("  peripheral, the labeller is not missing at random and the")
    print("  surviving sample is biased.")
    print()
    print("  mean relative face area")
    for g in ("F", "M", "unknown", "missing"):
        describe(g, [r["mean_rel_area"] for r in by.get(g, [])])
    print()
    print("  mean centre distance (0 = dead centre, 1 = corner)")
    for g in ("F", "M", "unknown", "missing"):
        describe(g, [r["mean_centre_dist"] for r in by.get(g, [])])
    print()
    print("  frames annotated per entity (how long they are on screen)")
    for g in ("F", "M", "unknown", "missing"):
        describe(g, [float(r["n_frames"]) for r in by.get(g, [])])

    lab = [r for r in rows if r["gender"] in ("F", "M")]
    un = [r for r in rows if r["gender"] not in ("F", "M")]
    if lab and un:
        la = st.fmean([r["mean_rel_area"] for r in lab])
        ua = st.fmean([r["mean_rel_area"] for r in un])
        print()
        print("  labelled mean area   = %.4f" % la)
        print("  unlabelled mean area = %.4f" % ua)
        print("  ratio labelled/unlabelled = %.2fx" % (la / ua if ua else float("nan")))

    print()
    print("=" * 68)
    print("3. VOTE CONFIDENCE vs GEOMETRY")
    print("=" * 68)
    print("  frac_F near 0.5 means the labeller's face crops disagreed.")
    dec, amb = [], []
    for r in rows:
        if r["gender"] not in ("F", "M"):
            continue
        try:
            f = float(r["frac_F"])
        except ValueError:
            continue
        (amb if 0.25 < f < 0.75 else dec).append(r)
    describe("unanimous area", [r["mean_rel_area"] for r in dec])
    describe("split-vote area", [r["mean_rel_area"] for r in amb])


if __name__ == "__main__":
    main(sys.argv[1])
