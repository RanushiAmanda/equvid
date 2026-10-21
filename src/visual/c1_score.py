"""
C1 step 2 — per-gender visual prominence parities, with a robustness check.

Every feature is a signed log parity:

    parity = log( female quantity / male quantity )

0 = parity, positive = the quantity is higher for women.  Logs are used so
that +x and -x are symmetric, matching M2's proposed sign convention.

EXTENSIVE vs INTENSIVE (this distinction is the whole point)
------------------------------------------------------------
  screen time is EXTENSIVE: four men accumulate more face-frames than two
      women even under perfectly equal treatment, so it is divided by the
      number of people of that gender before the ratio is taken.
  face area / centrality / close-up rate / solo rate are INTENSIVE: they
      are already averages over frames, so dividing by headcount a second
      time would be wrong.

Scoring is restricted to MIXED-GENDER windows.  EuqVid measures
within-content disparity: "given people of both genders in the same
material, is one afforded more prominence?"  A window containing only men
has no comparison to make and is returned as NA with a reason, never
silently dropped (M2's output contract).

NOTE (added after RESULTS.md Addendum 4): close-up and solo are LOG RATIOS
here, as they were when Findings 2-3 were written.  That form drops every
window in which one gender has a zero rate (292 of 819 windows, and
one-sidedly), so the close-up and solo figures this script produces are the
defective ones.  The sub-metric uses differences of proportions instead
(c1_features6.py).  This script is left unchanged so that Findings 2-3
still reproduce.

ROBUSTNESS CHECK
----------------
The label audit showed the labeller fails on small, peripheral faces.
So every parity is recomputed on entities whose faces are large enough
for labelling to be reliable.  If a result only exists below that
threshold, it is a labelling artefact rather than a framing finding.

Usage: python3 -I c1_score.py <entities.csv> <out_dir>
"""

import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

# Feature set.  Each entry: name, how to get the per-entity numerator,
# and whether higher values mean "more prominent".
INTENSIVE = [
    ("area", "mean_rel_area", True),        # bigger face  = more prominent
    ("centrality", "mean_centre_dist", False),  # smaller distance = more prominent
    ("closeup", "closeup_rate", True),
    ("solo", "solo_rate", True),
]


def log_ratio(f, m):
    """log(F/M), or None when either side is zero (reported as NA)."""
    if f <= 0 or m <= 0:
        return None
    return math.log(f / m)


def load(path, min_area=0.0):
    """entities.csv -> {(video, window): {gender: [entity dicts]}}"""
    win = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        r["n_frames"] = int(r["n_frames"])
        for k in ("mean_rel_area", "mean_centre_dist", "closeup_rate",
                  "solo_rate"):
            r[k] = float(r[k])
        if r["mean_rel_area"] < min_area:
            continue
        win[(r["video"], r["window"])][r["gender"]].append(r)
    return win


def score_windows(win):
    """One row per window: parities, or NA with a reason."""
    out = []
    for (v, w), g in sorted(win.items()):
        F, M = g.get("F", []), g.get("M", [])
        unusable = len(g.get("unknown", [])) + len(g.get("missing", []))
        row = {"video": v, "window": w, "n_F": len(F), "n_M": len(M),
               "n_unusable": unusable}
        if not F or not M:
            row["status"] = "NA"
            row["reason"] = "single-gender window" if (F or M) else "no labels"
            out.append(row)
            continue
        row["status"] = "valid"
        row["reason"] = ""

        # EXTENSIVE: screen time per person.
        tF = sum(e["n_frames"] for e in F) / len(F)
        tM = sum(e["n_frames"] for e in M) / len(M)
        row["screen_time_parity"] = log_ratio(tF, tM)

        # INTENSIVE: frame-weighted means, so a person on screen longer
        # counts proportionally -- the quantity describes the footage,
        # not the headcount.
        for name, col, higher_is_more in INTENSIVE:
            fF = sum(e[col] * e["n_frames"] for e in F)
            nF = sum(e["n_frames"] for e in F)
            fM = sum(e[col] * e["n_frames"] for e in M)
            nM = sum(e["n_frames"] for e in M)
            a, b = (fF / nF if nF else 0), (fM / nM if nM else 0)
            lr = log_ratio(a, b)
            # Flip the sign where a SMALLER raw value means more prominence,
            # so that positive always reads as "women more prominent".
            if lr is not None and not higher_is_more:
                lr = -lr
            row[name + "_parity"] = lr
        out.append(row)
    return out


def summarise(rows, label):
    valid = [r for r in rows if r["status"] == "valid"]
    print()
    print("-" * 66)
    print(label)
    print("-" * 66)
    print("  windows: %d total, %d valid (mixed-gender), %d NA"
          % (len(rows), len(valid), len(rows) - len(valid)))
    if not valid:
        return {}
    res = {}
    for name in ["screen_time"] + [n for n, _, _ in INTENSIVE]:
        key = name + "_parity"
        vals = [r[key] for r in valid if r.get(key) is not None]
        if not vals:
            print("  %-20s no usable windows" % name)
            continue
        m = st.fmean(vals)
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        # Direction counts are more robust than the mean to a few extreme
        # windows, so both are reported.
        pos = sum(1 for x in vals if x > 0)
        res[name] = (m, sd, len(vals))
        print("  %-20s mean log parity %+.4f  (sd %.3f, n=%d)  "
              "women higher in %.0f%% of windows"
              % (name, m, sd, len(vals), 100.0 * pos / len(vals)))
    return res


def main(ent_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    rows = score_windows(load(ent_path))
    base = summarise(rows, "ALL LABELLED ENTITIES  (no size filter)")

    # Write the full window table, NA rows included.
    cols = ["video", "window", "n_F", "n_M", "n_unusable", "status", "reason",
            "screen_time_parity"] + [n + "_parity" for n, _, _ in INTENSIVE]
    p = os.path.join(out_dir, "window_scores.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: ("" if r.get(k) is None else r.get(k, ""))
                          for k in cols})

    # Robustness: rerun above the size where labelling becomes reliable.
    print()
    print("=" * 66)
    print("ROBUSTNESS — does the result survive where labels are reliable?")
    print("=" * 66)
    for thr in (0.02, 0.05, 0.10):
        r2 = score_windows(load(ent_path, min_area=thr))
        summarise(r2, "ENTITIES WITH mean_rel_area >= %.2f" % thr)

    print()
    print("wrote %s" % p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
