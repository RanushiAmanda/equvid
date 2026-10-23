"""
C1 step 6 — what the blind manual labels say about the predicted ones.

Three questions, in order of importance to the component:

 Q1  How often is the face model right, and does that depend on face size?
 Q2  Do its errors depend on CENTRALITY?  This is the open question: if the
     model is more willing to answer "female" on a central face, the whole
     centrality finding collapses, and stratifying on size could not have
     detected it.
 Q3  What are the 29% it refused to label?  If the `unknown` and `missing`
     pools are disproportionately female, then n_F is undercounted and the
     identity (headcount) parity is overstated.

Usage: python3 -I c1_validate.py <sample_key.csv> <manual_labels.csv> \
         <entities.csv>
"""

import csv
import math
import statistics as st
import sys
from collections import defaultdict

# Population counts from the full extraction, used to project the sample
# back onto the whole corpus in Q3.
POP = {"F": 3991, "M": 11554, "unknown": 4118, "missing": 2267}

BANDS = [(0.00, 0.02), (0.02, 0.08), (0.08, 0.20), (0.20, 1.01)]


def wilson(k, n, z=1.96):
    """Wilson score interval — behaves sensibly at small n and near 0/1,
    unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main(key_path, manual_path, ent_path):
    key = {r["sample_id"]: r for r in csv.DictReader(open(key_path))}
    man = {r["sample_id"]: r["manual_gender"].strip().upper()
           for r in csv.DictReader(open(manual_path))
           if r["manual_gender"].strip()}

    rows = []
    for sid, m in man.items():
        k = key.get(sid)
        if not k:
            continue
        rows.append({
            "sid": sid, "pred": k["predicted_gender"], "man": m,
            "area": float(k["mean_rel_area"]),
            "dist": float(k["mean_centre_dist"]),
        })
    print("manual labels received : %d" % len(man))
    print("matched to the key     : %d" % len(rows))

    # ---------------------------------------------------------------- Q1
    print()
    print("=" * 70)
    print("Q1  CONFUSION  (rows = model, columns = you)")
    print("=" * 70)
    cm = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cm[r["pred"]][r["man"]] += 1
    print("%-10s %6s %6s %6s %7s" % ("model", "F", "M", "U", "total"))
    for p in ("F", "M", "unknown", "missing"):
        t = sum(cm[p].values())
        print("%-10s %6d %6d %6d %7d"
              % (p, cm[p]["F"], cm[p]["M"], cm[p]["U"], t))

    dec = [r for r in rows if r["pred"] in ("F", "M") and r["man"] in ("F", "M")]
    ok = sum(1 for r in dec if r["pred"] == r["man"])
    lo, hi = wilson(ok, len(dec))
    print()
    print("  accuracy where both decided: %d/%d = %.1f%%  (95%% CI %.1f-%.1f%%)"
          % (ok, len(dec), 100.0 * ok / len(dec), 100 * lo, 100 * hi))

    print()
    print("  accuracy by face size")
    for a, b in BANDS:
        s = [r for r in dec if a <= r["area"] < b]
        if not s:
            continue
        k = sum(1 for r in s if r["pred"] == r["man"])
        l, h = wilson(k, len(s))
        print("    area %.2f-%.2f : %2d/%2d = %5.1f%%  (CI %.0f-%.0f%%)"
              % (a, b, k, len(s), 100.0 * k / len(s), 100 * l, 100 * h))

    # ---------------------------------------------------------------- Q2
    print()
    print("=" * 70)
    print("Q2  DO ERRORS DEPEND ON CENTRALITY?")
    print("=" * 70)
    right = [r["dist"] for r in dec if r["pred"] == r["man"]]
    wrong = [r["dist"] for r in dec if r["pred"] != r["man"]]
    print("  correct predictions  : n=%-3d mean centre distance %.4f"
          % (len(right), st.fmean(right) if right else float("nan")))
    print("  incorrect predictions: n=%-3d mean centre distance %.4f"
          % (len(wrong), st.fmean(wrong) if wrong else float("nan")))
    if len(wrong) >= 3 and len(right) >= 3:
        diff = st.fmean(wrong) - st.fmean(right)
        print("  difference           : %+.4f" % diff)
        print()
        print("  If errors sat nearer the centre (negative difference), the")
        print("  centrality result would be suspect. A positive difference")
        print("  means errors happen further out, which is the benign case.")
    else:
        print("  too few errors to test this directly -- which is itself")
        print("  the answer: the model is not making many mistakes on the")
        print("  faces it is willing to label.")

    # also: is the model's willingness to label related to centrality?
    lab = [r["dist"] for r in rows if r["pred"] in ("F", "M")]
    ref = [r["dist"] for r in rows if r["pred"] not in ("F", "M")]
    if lab and ref:
        print()
        print("  model labelled    : n=%-3d mean centre distance %.4f"
              % (len(lab), st.fmean(lab)))
        print("  model refused     : n=%-3d mean centre distance %.4f"
              % (len(ref), st.fmean(ref)))

    # ---------------------------------------------------------------- Q3
    print()
    print("=" * 70)
    print("Q3  WHAT ARE THE 29%% THE MODEL REFUSED?")
    print("=" * 70)
    share = {}
    for p in ("F", "M", "unknown", "missing"):
        f, m, u = cm[p]["F"], cm[p]["M"], cm[p]["U"]
        d = f + m
        if d == 0:
            print("  %-9s nothing decidable in the sample" % p)
            share[p] = None
            continue
        lo2, hi2 = wilson(f, d)
        share[p] = (f / d, lo2, hi2, d, u)
        print("  %-9s truly female in %2d/%2d decidable = %5.1f%%  "
              "(CI %.0f-%.0f%%), %d unjudgeable"
              % (p, f, d, 100.0 * f / d, 100 * lo2, 100 * hi2, u))

    # Project onto the population and recompute the headcount parity.
    if all(share.get(p) for p in POP):
        nF = sum(POP[p] * share[p][0] for p in POP)
        nM = sum(POP[p] * (1 - share[p][0]) for p in POP)
        raw = math.log(POP["F"] / POP["M"])
        cor = math.log(nF / nM)
        print()
        print("  projected onto the full corpus:")
        print("    using model labels only : n_F=%5d  n_M=%5d  log parity %+.4f"
              % (POP["F"], POP["M"], raw))
        print("    corrected for refusals  : n_F=%5.0f  n_M=%5.0f  log parity %+.4f"
              % (nF, nM, cor))
        print("    men per woman: %.2f  ->  %.2f"
              % (POP["M"] / POP["F"], nM / nF))
        print()
        print("  This is an entity-level projection from %d labelled crops."
              % len(rows))
        print("  It is an estimate with real uncertainty, not a measurement.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
