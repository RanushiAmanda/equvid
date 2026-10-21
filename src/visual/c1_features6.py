"""
C1 step 5 — the full six-feature table, with collinearity diagnostics.

THE SIX FEATURES (as specified for this component)
--------------------------------------------------
  1 screen_time   presence            (baseline comparator)
  2 area          face size
  3 centrality    closeness to centre
  4 closeup       shot scale
  5 solo          visual isolation
  6 identity      headcount           (comparator)

Each is reported as a signed parity, positive = women higher: log(F/M) for
the ratio-scale quantities, and a difference of proportions (rate_F -
rate_M) for close-up and solo, which are bounded rates (see build() below
and RESULTS.md, Addendum 4).

TWO DEPENDENCIES THAT ARE NOT OPTIONAL
--------------------------------------
(a) Exact, by algebra:

        log(frames_F / frames_M)                       <- raw presence
          = log( (frames_F/n_F) / (frames_M/n_M) )     <- presence per person
          + log(n_F / n_M)                             <- identity

    So raw presence, per-person presence and identity are a linearly
    dependent trio: any two fix the third.  Only two can enter a model.
    This script verifies the identity numerically rather than asserting it.

(b) By construction: a close-up is defined by an area threshold, so
    closeup and area cannot be independent.

Which form of presence to use is therefore a decision, not a default.
This script reports BOTH screen_time variants so the choice is made on
evidence and recorded in the decision log.

Usage: python3 -I c1_features6.py <windows.csv> <entities.csv> <out_dir>
"""

import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

FEATURES = ["screen_time_pp", "area", "centrality", "closeup", "solo",
            "identity"]


def lg(a, b):
    return math.log(a / b) if (a > 0 and b > 0) else None


def lg_rate(c_f, n_f, c_m, n_m):
    """Log ratio of two proportions, with a +0.5/+1 continuity correction.

    Without it, any window where one gender has no close-up at all gives
    log(0) and is dropped.  Those windows are not missing at random: in
    this corpus 145 of them are windows where WOMEN get no close-up
    against 82 where men get none, so dropping them removes the most
    female-disadvantaged cases and biases the feature upward.  The
    correction keeps every window and costs only a slight shrink toward
    zero for small counts.

    NOT CALLED anywhere in this repository: build() uses the plain
    difference of the two rates for close-up and solo, which needs no
    correction.  Left in place unchanged; see RESULTS.md, Addendum 4.
    """
    if n_f <= 0 or n_m <= 0:
        return None
    rf = (c_f + 0.5) / (n_f + 1.0)
    rm = (c_m + 0.5) / (n_m + 1.0)
    return math.log(rf / rm)


def build(win_path, ent_path):
    """One row per mixed-gender window, six parities (log ratios, except
    close-up and solo, which are differences of rates)."""
    # Intensive quantities are rebuilt from entities so they can be
    # frame-weighted: a person on screen longer should count more.
    ents = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(ent_path, newline="", encoding="utf-8")):
        if r["gender"] not in ("F", "M"):
            continue
        r["n_frames"] = int(r["n_frames"])
        for k in ("mean_rel_area", "mean_centre_dist", "closeup_rate",
                  "solo_rate"):
            r[k] = float(r[k])
        ents[(r["video"], r["window"])][r["gender"]].append(r)

    rows = []
    for r in csv.DictReader(open(win_path, newline="", encoding="utf-8")):
        if r["mixed_gender"] != "1":
            continue
        nF, nM = float(r["n_F"]), float(r["n_M"])
        fF, fM = float(r["frames_F"]), float(r["frames_M"])
        if min(nF, nM, fF, fM) <= 0:
            continue
        g = ents[(r["video"], r["window"])]
        if not g["F"] or not g["M"]:
            continue

        def wm(es, col):
            n = sum(e["n_frames"] for e in es)
            return sum(e[col] * e["n_frames"] for e in es) / n if n else 0.0

        out = {"video": r["video"], "window": r["window"],
               "n_F": int(nF), "n_M": int(nM),
               "n_unusable": int(float(r["n_unknown"]) + float(r["n_missing"]))}

        out["identity"] = lg(nF, nM)
        out["screen_time_pp"] = lg(fF / nF, fM / nM)     # per person
        out["screen_time_raw"] = lg(fF, fM)              # group total
        out["area"] = lg(wm(g["F"], "mean_rel_area"), wm(g["M"], "mean_rel_area"))
        # Smaller distance means more central, so the sign is flipped to
        # keep "positive = women more prominent" true for every feature.
        c = lg(wm(g["F"], "mean_centre_dist"), wm(g["M"], "mean_centre_dist"))
        out["centrality"] = None if c is None else -c
        # Close-up and solo are PROPORTIONS of frames, bounded in [0,1].
        # A log ratio is the wrong form for them: as either rate nears
        # zero the log diverges, which both drops windows (log 0) and
        # produces a heavy tail that swamps the mean.  Measured on this
        # corpus, the log form gives closeup +0.081 with a CI spanning
        # zero, while the difference form gives +0.095 [+0.065, +0.126].
        # Log parity stays for the ratio-scale quantities -- headcount,
        # screen time, area, centre distance -- where it is correct.
        out["closeup"] = float(r["closeup_F"]) / fF - float(r["closeup_M"]) / fM
        out["solo"] = float(r["solo_F"]) / fF - float(r["solo_M"]) / fM
        rows.append(out)
    return rows


def pearson(x, y):
    n = len(x)
    mx, my = st.fmean(x), st.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else float("nan")


def ols_r2(y, X):
    """R^2 of y on X (with intercept), via normal equations + Gaussian
    elimination.  Used for VIF; no external dependency."""
    n, k = len(y), len(X[0])
    Z = [[1.0] + list(row) for row in X]
    p = k + 1
    A = [[sum(Z[i][a] * Z[i][b] for i in range(n)) for b in range(p)]
         + [sum(Z[i][a] * y[i] for i in range(n))] for a in range(p)]
    for c in range(p):                      # Gaussian elimination
        piv = max(range(c, p), key=lambda r: abs(A[r][c]))
        if abs(A[piv][c]) < 1e-12:
            return float("nan")             # singular -> exact collinearity
        A[c], A[piv] = A[piv], A[c]
        d = A[c][c]
        A[c] = [v / d for v in A[c]]
        for r2 in range(p):
            if r2 != c and A[r2][c]:
                f = A[r2][c]
                A[r2] = [v - f * w for v, w in zip(A[r2], A[c])]
    beta = [A[i][p] for i in range(p)]
    yh = [sum(b * z for b, z in zip(beta, Z[i])) for i in range(n)]
    my = st.fmean(y)
    ss_res = sum((y[i] - yh[i]) ** 2 for i in range(n))
    ss_tot = sum((v - my) ** 2 for v in y)
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def main(win_path, ent_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = build(win_path, ent_path)
    print("mixed-gender windows with all six features: %d" % len(rows))

    # --- verify the algebraic identity rather than assert it ------------
    res = [r["screen_time_raw"] - (r["screen_time_pp"] + r["identity"])
           for r in rows
           if None not in (r["screen_time_raw"], r["screen_time_pp"],
                           r["identity"])]
    print("identity check  raw - (per_person + identity): max |residual| = %.2e"
          % max(abs(v) for v in res))
    print("  -> exact dependency confirmed; only two of the three are free")

    print()
    print("=" * 70)
    print("FEATURE MEANS  (+ = women higher; closeup, solo = rate differences)")
    print("=" * 70)
    for f in FEATURES:
        v = [r[f] for r in rows if r[f] is not None]
        if not v:
            continue
        pos = sum(1 for x in v if x > 0)
        print("  %-16s mean %+.4f   median %+.4f   sd %.3f   n=%-4d  "
              "women higher %3.0f%%"
              % (f, st.fmean(v), st.median(v), st.pstdev(v), len(v),
                 100.0 * pos / len(v)))

    # --- correlation matrix on complete cases ---------------------------
    keep = [r for r in rows if all(r[f] is not None for f in FEATURES)]
    print()
    print("=" * 70)
    print("CORRELATION MATRIX  (complete cases, n=%d)" % len(keep))
    print("=" * 70)
    cols = {f: [r[f] for r in keep] for f in FEATURES}
    hdr = "%-16s" % ""
    for f in FEATURES:
        hdr += "%10s" % f[:9]
    print(hdr)
    for a in FEATURES:
        line = "%-16s" % a
        for b in FEATURES:
            line += "%10.3f" % pearson(cols[a], cols[b])
        print(line)

    # --- VIF -------------------------------------------------------------
    print()
    print("=" * 70)
    print("VIF  (drop rule from the proposal: remove above 10)")
    print("=" * 70)
    for f in FEATURES:
        others = [o for o in FEATURES if o != f]
        X = [[r[o] for o in others] for r in keep]
        r2 = ols_r2(cols[f], X)
        vif = float("inf") if (r2 != r2 or r2 >= 1) else 1.0 / (1.0 - r2)
        flag = "  <-- above 10" if vif > 10 else ""
        print("  %-16s R2=%.4f   VIF=%8.2f%s" % (f, r2, vif, flag))

    p = os.path.join(out_dir, "features6.csv")
    cols_out = (["video", "window", "n_F", "n_M", "n_unusable"]
                + FEATURES + ["screen_time_raw"])
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k))
                        for k in cols_out})
    print()
    print("wrote %s" % p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
