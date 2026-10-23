"""
Does the aggregation unit change the C1 score?

The group's shared-conventions note (M2) makes the row a whole episode or
meeting (clip_id = MELD:S08E21, AMI:ES2002a).  C1's atoms are 60-second
windows on AVA.  Aggregation only goes fine -> coarse, so the question is
not "can the two meet" but "does the answer depend on HOW we sum up".

Three rules, all computed from the SAME per-window counts in windows.csv:

  A  window-mean    mean over windows of the per-window parity   (what the
                    submetric reports today)
  B  pooled         add every count first, take ONE parity       (ratio of
                    sums; what C3 already does)
  C  video-mean     pool inside each video, take its parity, then average
                    videos -- i.e. treat the video as M2's "clip"

All three share one set of bootstrap resamples (resampling VIDEOS), so a
difference between rules is a paired difference, not bootstrap noise.

Usage: python3 -I c1_agg_check.py <windows.csv> <submetric.csv>
"""

import csv
import math
import random
import statistics as st
import sys
from collections import defaultdict

SEED, BOOT = 0, 2000
DIMS = [("presence", "log", 4.0), ("centrality", "log", 1.5),
        ("closeup", "diff", 0.25), ("solo", "diff", 0.25)]
KEYS = ("n_F", "n_M", "frames_F", "frames_M", "dist_F", "dist_M",
        "closeup_F", "closeup_M", "solo_F", "solo_M")


def parity(S, d):
    """Signed parity from a dict of SUMMED counts. + = women higher."""
    if d == "presence":
        return math.log(S["n_F"] / S["n_M"])
    if d == "centrality":
        # smaller distance = more central, so flip the sign
        return -math.log((S["dist_F"] / S["frames_F"])
                         / (S["dist_M"] / S["frames_M"]))
    if d == "closeup":
        return S["closeup_F"] / S["frames_F"] - S["closeup_M"] / S["frames_M"]
    return S["solo_F"] / S["frames_F"] - S["solo_M"] / S["frames_M"]


def score(p, kind, anchor):
    div = math.log(anchor) if kind == "log" else anchor
    return 10.0 * max(-1.0, min(1.0, p / div))


def tier(s):
    a = abs(s)
    return "Neutral" if a < 2 else ("Moderate" if a <= 6 else "Strong")


def load(path):
    wins = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["mixed_gender"] != "1":
            continue
        S = {k: float(r[k]) for k in KEYS}
        # same inclusion rule as c1_features6.build()
        if min(S["n_F"], S["n_M"], S["frames_F"], S["frames_M"]) <= 0:
            continue
        S["video"] = r["video"]
        wins.append(S)
    return wins


def total(rows):
    return {k: sum(r[k] for r in rows) for k in KEYS}


def main(win_path, sub_path):
    wins = load(win_path)
    by = defaultdict(list)
    for w in wins:
        by[w["video"]].append(w)
    vids = sorted(by)
    print("windows %d   videos %d   (mixed-gender windows per video: "
          "median %d)" % (len(wins), len(vids),
                          sorted(len(v) for v in by.values())[len(vids) // 2]))

    # per-video precomputation so each bootstrap draw is cheap
    V = {v: total(by[v]) for v in vids}
    sumA = {d: {v: sum(parity(w, d) for w in by[v]) for v in vids}
            for d, _, _ in DIMS}
    cnt = {v: len(by[v]) for v in vids}
    vpar = {d: {v: parity(V[v], d) for v in vids} for d, _, _ in DIMS}

    def stat_A(d, pick):
        return sum(sumA[d][v] for v in pick) / sum(cnt[v] for v in pick)

    def stat_B(d, pick):
        S = {k: sum(V[v][k] for v in pick) for k in KEYS}
        return parity(S, d)

    def stat_C(d, pick):
        return st.fmean(vpar[d][v] for v in pick)

    STATS = {"A window-mean": stat_A, "B pooled": stat_B, "C video-mean": stat_C}

    # check A reproduces what the sub-metric already reports
    ref = {r["dimension"]: float(r["parity"])
           for r in csv.DictReader(open(sub_path))}
    for d, _, _ in DIMS:
        a = stat_A(d, vids)
        assert abs(a - ref[d]) < 1e-5, (d, a, ref[d])
    print("rule A reproduces submetric.csv for all four dimensions: OK")

    rng = random.Random(SEED)
    draws = {(d, s): [] for d, _, _ in DIMS for s in STATS}
    for _ in range(BOOT):
        pick = [rng.choice(vids) for _ in vids]       # shared by all rules
        for d, _, _ in DIMS:
            for s, f in STATS.items():
                draws[(d, s)].append(f(d, pick))

    def ci(x):
        x = sorted(x)
        return x[int(.025 * len(x))], x[int(.975 * len(x))]

    for d, kind, anchor in DIMS:
        div = math.log(anchor) if kind == "log" else anchor
        print()
        print("=" * 78)
        print("%s   (score = 10 x parity / %.4f)" % (d.upper(), div))
        print("=" * 78)
        print("%-15s %9s %20s %8s %9s" % ("rule", "parity", "95% CI by video",
                                          "score", "tier"))
        pts = {}
        for s, f in STATS.items():
            p = f(d, vids)
            lo, hi = ci(draws[(d, s)])
            sc = score(p, kind, anchor)
            pts[s] = p
            print("%-15s %+9.4f   [%+.4f,%+.4f] %+8.2f %9s"
                  % (s, p, lo, hi, sc, tier(sc)))
        A = "A window-mean"
        for s in ("B pooled", "C video-mean"):
            diff = [(x - y) for x, y in zip(draws[(d, s)], draws[(d, A)])]
            lo, hi = ci(diff)
            k = 10.0 / div
            sig = "" if lo < 0 < hi else "   <-- paired CI excludes 0"
            print("   %-12s - A  = %+6.2f score pts   paired 95%% CI "
                  "[%+.2f, %+.2f]%s"
                  % (s[:1], (pts[s] - pts[A]) * k, lo * k, hi * k, sig))

    # how noisy is ONE video, i.e. one M2-style clip, read on its own?
    print()
    print("=" * 78)
    print("ONE VIDEO READ ALONE  (the row M2's contract would carry)")
    print("=" * 78)
    for d, kind, anchor in DIMS:
        div = math.log(anchor) if kind == "log" else anchor
        sc = [vpar[d][v] * 10.0 / div for v in vids]
        sc3 = [vpar[d][v] * 10.0 / div for v in vids if cnt[v] >= 3]
        print("  %-11s SD across videos %5.2f score pts   "
              "(videos with >=3 mixed windows: %d of %d, SD %5.2f)"
              % (d, st.pstdev(sc), len(sc3), len(vids), st.pstdev(sc3)))
    print("  -> read at dataset/shard level, never per clip: same lesson M2 "
          "reports (per-meeting SD 2.68 vs 0.61 for 10-meeting sets).")


if __name__ == "__main__":
    main(*sys.argv[1:3])
