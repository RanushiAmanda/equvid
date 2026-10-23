"""
Visual-EuqVid v1 — the C1 sub-metric.

WHAT IT SCORES
--------------
A dataset, not a clip.  The question EuqVid asks is "given this video
corpus, how unequally are the genders treated?", so the unit of the score
is the corpus.  Per-window values are the raw material and are far too
noisy to read individually.

FOUR DIMENSIONS, NOT SIX
------------------------
The six nominal features reduced to four measured dimensions once each
was tested (see RESULTS.md):

  presence    headcount        log(n_F / n_M)
  centrality  framing position -log(dist_F / dist_M)
  closeup     shot scale       rate_F - rate_M
  solo        visual isolation rate_F - rate_M

The last two are differences rather than log ratios because they are
bounded proportions: a log ratio diverges as either rate approaches zero,
which drops windows and produces a tail that swallows the signal.

Dropped, with reasons: mean face area is a weaker estimator of the same
quantity as close-up share, and screen time per person carries no signal.
Solo framing was set aside at one point (RESULTS.md, Addendum 3) and
brought back once it was measured as a difference of proportions
(Addendum 4); its tier is the least secure of the four (Addendum 7).

Every parity is signed so that POSITIVE means women are higher, matching
M2's proposed convention.  EuqAud's convention is the opposite (positive =
male-dominant bias).  The group has not yet decided which one to use.

WHY NO REGRESSION HERE
----------------------
The proposal specified a degree-2 polynomial Ridge equation over six
features.  Two of those six were dropped on evidence (above).  A degree-2
model has 28 coefficients over six features and 15 over the four reported
dimensions, and it would be fitted against labels produced by our own
augmentation operators.  That is the failure M2 documented: their learned
composite scored Spearman -0.28 against DOME while their simple parity
measure scored +0.465 (50 slices from 11 meetings, wide intervals; see
docs/shared_conventions_M2_2026-10-06.md).  So v1 is the simple measure,
and the regression stays in the plan as the comparison.

CONFIDENCE INTERVALS ARE BOOTSTRAPPED OVER VIDEOS
-------------------------------------------------
Windows inside one film are not independent: same cast, same director,
same lighting.  Resampling windows would give intervals several times too
narrow.  The bootstrap resamples VIDEOS, which is the level at which the
observations really are exchangeable.  This is the same point M2 makes
about bootstrapping over series rather than shards.

SCALE AND TIERS
---------------
Each dimension maps onto -10..+10 through an anchor:

    score = 10 * clip( parity / ln(ANCHOR), -1, +1 )

ANCHOR is the ratio that saturates the scale.  Tiers follow the project
definition: |score| < 2 Neutral, 2-6 Moderate, > 6 Strong.

WHERE THE PRESENCE ANCHOR COMES FROM  (it is derived, not chosen)
-----------------------------------------------------------------
EuqAud (Section III-E) scales its raw output to -10..+10 by min-max
scaling, with bounds built from extreme feature settings; the paper does
not say whether those bounds are recomputed for each dataset.  The tier
cut-points 2 and 6 are EuqAud's and are used here unchanged.  What is
derived is the PRESENCE ANCHOR: the headcount ratio at which this scale
reaches those cut-points, fixed against two published sources.  Nothing in
the equation is estimated from the corpus being scored, so a score cannot
move when another corpus is added.

Set the Moderate/Strong boundary at the measured industry norm:

    USC Annenberg, "Inequality in 800 Popular Films" (Smith, Choueiti &
    Pieper), 800 films 2007-2015, 35,205 characters: male-to-female
    speaking characters = 2.30 : 1.

    anchor = exp(10 * ln(2.30) / 6) = 4.0076  ->  4.0

Read back, anchor 4.0 places the tier boundaries at:

    |score| = 6  ->  2.297 : 1   (selection ratio 0.435)  the industry norm
    |score| = 2  ->  1.320 : 1   (selection ratio 0.758)

and the second sits close to the four-fifths rule, 29 C.F.R. 1607.4(D):
a selection rate below 80% of the highest group's rate (1.25 : 1) "will
generally be regarded by the Federal enforcement agencies as evidence of
adverse impact".  0.758 is slightly MORE LENIENT than that 0.80: a
selection rate of 0.78 (1.28 : 1) is flagged by the four-fifths rule and
is Neutral here.

So "Strong" begins where a corpus is more unequal than the measured
industry average, and "Neutral" ends a little beyond the legal
adverse-impact line.  Both boundaries sit beside a published figure.

LIMITS OF THAT DERIVATION, STATED
---------------------------------
1. The four-fifths rule is United States employment law.  Applying it to
   screen representation is an analogy; the ML fairness literature uses
   the same analogy routinely, but it is still an analogy and is declared
   as one.
2. Annenberg counts SPEAKING CHARACTERS, which matches the presence
   dimension and nothing else.  The centrality, close-up and solo anchors
   below have no published norm behind them and remain conventions.
   Grounding them needs the human-perception study or the downstream
   disparity work, both of which sit in C4.

Usage: python3 -I c1_submetric.py <features6.csv> [out_dir]
"""

import csv
import math
import os
import random
import statistics as st
import sys
from collections import defaultdict

SEED = 0
BOOT = 2000

# dimension -> (column, kind, anchor, note)
#   kind "log"  : the column is a log parity; the anchor is the RATIO that
#                 saturates the scale, so the divisor is ln(anchor).
#   kind "diff" : the column is a difference of two proportions; the anchor
#                 is the DIFFERENCE that saturates the scale.
# The two kinds exist because the quantities differ: headcount and distance
# are ratio-scale, close-up and solo are bounded proportions.
DIMS = [
    ("presence",   "identity",   "log",  4.0,
     "DERIVED: exp(10*ln(2.30)/6) = 4.0076, frozen as 4.00, setting "
     "Moderate/Strong at the Annenberg 2.30:1 industry norm; Neutral ends "
     "at 0.758, slightly more lenient than the four-fifths rule"),
    ("centrality", "centrality", "log",  1.5,
     "convention: 1.5:1 centre-distance saturates the scale (no published "
     "norm exists for framing position)"),
    ("closeup",    "closeup",    "diff", 0.25,
     "convention: a 25-point gap in close-up share saturates the scale"),
    ("solo",       "solo",       "diff", 0.25,
     "convention: a 25-point gap in solo-framing share saturates the scale"),
]


def tier(s):
    a = abs(s)
    if a < 2:
        return "Neutral"
    if a <= 6:
        return "Moderate"
    return "Strong"


def load(path):
    rows = []
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        o = {"video": r["video"]}
        for _, col, _, _, _ in DIMS:
            o[col] = float(r[col]) if r[col] not in ("", None) else None
        rows.append(o)
    return rows


def dataset_mean(rows, col):
    """Mean parity over windows that have the feature."""
    v = [r[col] for r in rows if r[col] is not None]
    return (st.fmean(v), len(v)) if v else (float("nan"), 0)


def boot_ci(rows, col, rng, n=BOOT):
    """Resample VIDEOS, keeping each video's windows together."""
    by = defaultdict(list)
    for r in rows:
        if r[col] is not None:
            by[r["video"]].append(r[col])
    vids = list(by)
    if len(vids) < 3:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(n):
        pool = []
        for _ in vids:
            pool.extend(by[rng.choice(vids)])
        if pool:
            out.append(st.fmean(pool))
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main(path, out_dir="."):
    rows = load(path)
    rng = random.Random(SEED)
    vids = {r["video"] for r in rows}
    print("windows: %d   videos: %d" % (len(rows), len(vids)))
    print()
    print("=" * 74)
    print("VISUAL-EuqVid v1   (positive = women higher)")
    print("=" * 74)
    print("%-12s %9s %7s %19s %8s %10s"
          % ("dimension", "parity", "n", "95% CI (by video)", "score", "tier"))
    print("-" * 74)

    results = []
    for name, col, kind, anchor, note in DIMS:
        m, n = dataset_mean(rows, col)
        lo, hi = boot_ci(rows, col, rng)
        div = math.log(anchor) if kind == "log" else anchor
        s = 10.0 * max(-1.0, min(1.0, m / div))
        slo = 10.0 * max(-1.0, min(1.0, lo / div))
        shi = 10.0 * max(-1.0, min(1.0, hi / div))
        sig = "" if (lo < 0 < hi) else " *"
        print("%-12s %+9.4f %7d  [%+.4f,%+.4f] %+7.2f %10s%s"
              % (name, m, n, lo, hi, s, tier(s), sig))
        results.append({"dimension": name, "parity": round(m, 6),
                        "n_windows": n, "ci_lo": round(lo, 6),
                        "ci_hi": round(hi, 6), "score": round(s, 3),
                        "score_lo": round(slo, 3), "score_hi": round(shi, 3),
                        "tier": tier(s), "kind": kind, "anchor": anchor,
                        "excludes_zero": int(not (lo < 0 < hi))})

    print("-" * 74)
    print("  * = the 95% interval excludes zero")
    print()
    print("  anchors:")
    for name, _, _, anchor, note in DIMS:
        print("    %-11s %s" % (name, note))
    print()
    print("  presence is anchored on published figures; the three framing")
    print("  anchors are conventions and are declared as such.")
    print("  Sources: USC Annenberg, Inequality in 800 Popular Films;")
    print("           29 C.F.R. 1607.4(D), four-fifths rule.")

    print()
    print("=" * 74)
    print("READING THE SCORE")
    print("=" * 74)
    pres = next(r for r in results if r["dimension"] == "presence")
    fram = [r for r in results if r["dimension"] != "presence"]
    print("  presence  %+.2f (%s)" % (pres["score"], pres["tier"]))
    for r in fram:
        print("  %-9s %+.2f (%s)" % (r["dimension"], r["score"], r["tier"]))
    print()
    print("  The dimensions are reported separately and NOT averaged into")
    print("  one number. They disagree in sign here, and averaging would")
    print("  cancel a large presence gap against a smaller framing gap and")
    print("  report a corpus as near-neutral when it is not.")

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "submetric.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print()
    print("wrote %s" % p)


if __name__ == "__main__":
    main(*sys.argv[1:])
