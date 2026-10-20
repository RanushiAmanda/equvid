"""
C4 - MELD: the duration confound, and how many clips the real run needs.

WHY THIS EXISTS
---------------
On AMI, `dynamic_range` as max-minus-min turned out to be measuring turn
length rather than delivery, because the range of a sample grows with the
number of samples. That was caught by c4_check_duration.py and fixed by
switching to the IQR (decision D1).

MELD can repeat the same mistake in a new disguise. `p90 - p10` of the
loudness contour looks like a speech-to-background estimate, but p10 is a low
percentile, and a low percentile of a LONGER contour reaches further down. So
the statistic can grow with clip duration while measuring nothing about the
mix. Clip durations here span 0.8 s to 13.7 s - a factor of seventeen - so the
risk is larger than it was on AMI, not smaller.

This script tests every candidate statistic against duration before any of
them is allowed into the equation.

IT ALSO ANSWERS A DESIGN QUESTION
---------------------------------
`mean_voice_loudness` is only comparable across clips if clip loudness is not
dominated by which episode the clip came from. If episodes differ from each
other more than clips differ within an episode, then pooling all female clips
against all male clips is unsafe: if the women's clips happen to come from
quieter episodes, that shows up as a gender gap.

The fix is the same one used for AMI series - compare WITHIN a unit, then
aggregate the differences. This script computes both the pooled comparison and
the within-episode comparison, so the difference between them is visible
rather than assumed.

Finally it counts the corpus: how many utterances, how many episodes, and how
many episodes can supply enough clips per gender for a stable episode-level
mean. That is what decides how many clips the real extraction run needs.

USAGE
-----
    python c4_check_meld_duration.py
    python c4_check_meld_duration.py --csv ../output/c4_pilot_meld_train.csv
"""

import argparse
import csv
import math
import os
import sys

# Reuse the pilot's locators and gender map rather than copying them, so the
# two scripts cannot drift apart.
try:
    from c4_pilot_meld import (GENDER_OF, find_corpus_dir, find_meld_root,
                               find_split_csv, read_utterances, speaker_key)
except ImportError:
    sys.exit("Run this from the same folder as c4_pilot_meld.py.")

# Statistics to test. The question for each is the same: does it move with
# clip duration? If it does, it is not safe to put in the equation.
STATS = [
    ("integrated_lufs", "mean_voice_loudness"),
    ("rms_dbfs", "unweighted RMS (control)"),
    ("p90_lufs", "speech level (p90)"),
    ("p10_lufs", "floor (p10)"),
    ("speech_to_background", "p90 - p10"),
    ("block_iqr", "block loudness IQR"),
    ("block_sd", "block loudness SD"),
    ("quiet_frac_15lu", "non-speech fraction"),
]

# A statistic whose |r| with duration exceeds this is treated as contaminated.
# 0.30 is deliberately strict: at n of a few hundred, r = 0.3 is roughly 9% of
# variance explained by an irrelevant variable, which is enough to manufacture
# or hide a gender difference.
R_LIMIT = 0.30


def need(mod, hint):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"Missing package '{mod}'.  pip install {hint}")


def pearson(x, y, np):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y, np):
    """Pearson on ranks. Robust to the skew in clip durations, which are
    heavily bunched at the short end."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry, np)


def binom_two_sided(k, n):
    """Exact two-sided sign test - same function as c4_analyse.py."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def ols(y, X, np):
    """Least squares with coefficient t-statistics, by hand.

    Written out rather than pulled from statsmodels because it is short, it
    adds no dependency, and every line of it can be defended in a viva:

        b     = (X'X)^-1 X'y          the fitted coefficients
        RSS   = sum of squared residuals
        s2    = RSS / (n - k)         residual variance
        var(b)= s2 * (X'X)^-1         coefficient covariance
        t_j   = b_j / sqrt(var(b)_jj)
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    if n <= k:
        return None
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = n - k
    s2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.maximum(s2 * np.diag(xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, b / se, np.nan)
    return b, se, t, dof


def t_pvalue(t, dof):
    """Two-sided p from a t statistic. scipy if present, else the normal
    approximation, which is close enough above about 30 degrees of freedom."""
    if not (t == t):            # nan
        return float("nan")
    try:
        from scipy import stats
        return float(2.0 * stats.t.sf(abs(t), dof))
    except ImportError:
        return float(math.erfc(abs(t) / math.sqrt(2.0)))


# ==========================================================================
#  Part 1 - corpus census
# ==========================================================================

def census(rows, np):
    """How big is MELD, and how many episodes can support an episode-level
    gender comparison?"""
    eps = {}
    main = 0
    for r in rows:
        g = GENDER_OF.get(speaker_key(r["speaker_raw"]))
        if not g:
            continue
        main += 1
        key = (r["season"], r["episode"])
        d = eps.setdefault(key, {"F": 0, "M": 0})
        d[g] += 1

    per_ep = np.array([v["F"] + v["M"] for v in eps.values()])
    print("\n  PART 1  CORPUS CENSUS (this split)")
    print(f"      utterances in the split          {len(rows)}")
    print(f"      with a gender-labelled speaker   {main}  "
          f"({main / len(rows):.1%})")
    print(f"      distinct episodes                {len(eps)}")
    print(f"      main-cast clips per episode      median {np.median(per_ep):.0f}   "
          f"range {per_ep.min()}-{per_ep.max()}")

    print("\n      Episodes able to supply N clips of EACH gender:")
    print(f"      {'N':>4}{'episodes':>11}{'share':>9}{'clips used':>13}")
    for n in (5, 10, 15, 20, 30):
        qual = [v for v in eps.values() if v["F"] >= n and v["M"] >= n]
        used = sum(min(v["F"], n) + min(v["M"], n) for v in qual)
        print(f"      {n:>4}{len(qual):>11}{len(qual) / len(eps):>8.0%}{used:>13}")

    print("\n      Reading this table. The left column is how many clips per")
    print("      gender you demand from an episode; the right is how many clips")
    print("      the whole run would then process. Demanding more per episode")
    print("      gives steadier episode means but throws away episodes that")
    print("      cannot meet the quota, which shrinks the effective n.")
    return eps


# ==========================================================================
#  Part 2 - the duration confound
# ==========================================================================

def duration_check(rows, np):
    dur = np.array([r["duration_s"] for r in rows])
    logd = np.log(dur)
    short = dur < np.percentile(dur, 33)
    long_ = dur > np.percentile(dur, 67)

    print("\n  PART 2  DOES THE STATISTIC TRACK CLIP DURATION?")
    print(f"      n = {len(rows)}   duration median {np.median(dur):.2f}s   "
          f"range {dur.min():.2f}-{dur.max():.2f}s")
    print(f"      shortest third < {np.percentile(dur, 33):.2f}s, "
          f"longest third > {np.percentile(dur, 67):.2f}s\n")
    hdr = (f"      {'statistic':<24}{'r':>8}{'rho':>8}{'r log':>8}"
           f"{'short':>9}{'long':>9}{'':>4}")
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))

    verdicts = {}
    for key, label in STATS:
        v = np.array([r[key] for r in rows], float)
        m = np.isfinite(v)
        if m.sum() < 10:
            continue
        r = pearson(dur[m], v[m], np)
        rho = spearman(dur[m], v[m], np)
        rl = pearson(logd[m], v[m], np)
        ms = np.nanmean(v[m & short]) if (m & short).any() else float("nan")
        ml = np.nanmean(v[m & long_]) if (m & long_).any() else float("nan")
        worst = max(abs(x) for x in (r, rho, rl) if x == x)
        bad = worst > R_LIMIT
        verdicts[key] = (bad, worst)
        print(f"      {label:<24}{r:>+8.2f}{rho:>+8.2f}{rl:>+8.2f}"
              f"{ms:>9.2f}{ml:>9.2f}{'  <-- BAD' if bad else '':>4}")

    print(f"\n      r  Pearson with duration, rho Spearman, r log with "
          f"log(duration)")
    print(f"      Flagged when the largest of the three exceeds {R_LIMIT:.2f}.")
    return verdicts


# ==========================================================================
#  Part 3 - pooled versus within-episode gender comparison
# ==========================================================================

def gender_check(rows, np, verdicts):
    print("\n  PART 3  GENDER COMPARISON, THREE WAYS")
    print("      (a) pooled: all male clips against all female clips")
    print("      (b) pooled, adjusted: same, with log(duration) in the model")
    print("      (c) within episode: difference inside each episode, then")
    print("          aggregated - the design C4 should actually use\n")

    gen = np.array([r["gender"] for r in rows])
    dur = np.array([r["duration_s"] for r in rows])
    logd = np.log(dur)
    isM = (gen == "M").astype(float)

    # group clips by episode for comparison (c)
    by_ep = {}
    for i, r in enumerate(rows):
        by_ep.setdefault(r["clip_id"], {"F": [], "M": []})[r["gender"]].append(i)
    usable_eps = [k for k, v in by_ep.items() if v["F"] and v["M"]]

    hdr = (f"      {'statistic':<24}{'(a) M-F':>10}{'p':>8}"
           f"{'(b) adj':>10}{'p':>8}{'(c) within':>12}{'eps':>6}{'p':>8}")
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))

    for key, label in STATS:
        v = np.array([r[key] for r in rows], float)
        m = np.isfinite(v)
        if m.sum() < 20:
            continue

        # (a) pooled difference of means
        a = float(np.mean(v[m & (gen == "M")]) - np.mean(v[m & (gen == "F")]))
        fit = ols(v[m], np.column_stack([np.ones(m.sum()), isM[m]]), np)
        pa = t_pvalue(fit[2][1], fit[3]) if fit else float("nan")

        # (b) same, controlling for log duration
        fitb = ols(v[m], np.column_stack(
            [np.ones(m.sum()), isM[m], logd[m]]), np)
        b = float(fitb[0][1]) if fitb else float("nan")
        pb = t_pvalue(fitb[2][1], fitb[3]) if fitb else float("nan")

        # (c) within-episode differences, then sign test across episodes
        diffs = []
        for k in usable_eps:
            fi = [i for i in by_ep[k]["F"] if m[i]]
            mi = [i for i in by_ep[k]["M"] if m[i]]
            if fi and mi:
                diffs.append(float(np.mean(v[mi]) - np.mean(v[fi])))
        if diffs:
            arr = np.array(diffs)
            c = float(arr.mean())
            kpos = int((arr > 0).sum())
            pc = binom_two_sided(kpos, len(arr))
        else:
            c, pc, arr = float("nan"), float("nan"), np.array([])

        flag = "  <-- BAD" if verdicts.get(key, (False,))[0] else ""
        print(f"      {label:<24}{a:>+10.2f}{pa:>8.3f}{b:>+10.2f}{pb:>8.3f}"
              f"{c:>+12.2f}{len(arr):>6}{pc:>8.3f}{flag}")

    print(f"\n      Episodes with clips of both genders: {len(usable_eps)} of "
          f"{len(by_ep)}")
    print("      A statistic whose (a) and (b) columns differ substantially is")
    print("      one where duration was doing the work. Column (c) is the only")
    print("      one immune to episode-level mix offsets, and it is the number")
    print("      to report - but it needs enough episodes to be worth reading.")


def variance_split(rows, np):
    """How much of the loudness spread is between episodes rather than within?

    If between-episode dominates, pooling clips across episodes is unsafe and
    the gender comparison must be made inside each episode.
    """
    by_ep = {}
    for r in rows:
        if r["integrated_lufs"] == r["integrated_lufs"]:
            by_ep.setdefault(r["clip_id"], []).append(r["integrated_lufs"])
    multi = {k: v for k, v in by_ep.items() if len(v) >= 3}
    if len(multi) < 5:
        print("\n  PART 4  VARIANCE SPLIT")
        print(f"      Only {len(multi)} episodes have 3+ clips. Re-run the pilot")
        print("      with more clips before trusting this.")
        return

    allv = np.array([x for v in by_ep.values() for x in v])
    # pooled within-episode variance, weighted by degrees of freedom
    num = sum(float(np.var(v, ddof=1)) * (len(v) - 1) for v in multi.values())
    den = sum(len(v) - 1 for v in multi.values())
    within = math.sqrt(num / den)
    total = float(np.std(allv, ddof=1))
    between2 = max(total ** 2 - within ** 2, 0.0)

    print("\n  PART 4  WHERE DOES LOUDNESS VARIATION LIVE?")
    print(f"      episodes with 3+ clips      {len(multi)}")
    print(f"      total SD                    {total:.2f} LU")
    print(f"      within-episode SD (pooled)  {within:.2f} LU")
    print(f"      between-episode SD          {math.sqrt(between2):.2f} LU")
    share = between2 / (total ** 2) if total else float("nan")
    print(f"      between-episode share of variance  {share:.0%}")
    print()
    if share > 0.4:
        print("      Most of the spread is BETWEEN episodes, so absolute clip")
        print("      loudness partly encodes which episode it came from.")
        print("      Pooling all female clips against all male clips is unsafe:")
        print("      an unlucky draw of episodes would read as a gender gap.")
        print("      Compare within episode and aggregate - the same correction")
        print("      applied to AMI series in decision D3.")
    else:
        print("      Within-episode spread dominates, so clip loudness is driven")
        print("      by scene and delivery rather than a per-episode offset.")
        print("      Pooling is defensible, though within-episode is still safer.")


# ==========================================================================

def main():
    np = need("numpy", "numpy")
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="MELD duration confound and sample-size planning.")
    ap.add_argument("--csv", default=os.path.normpath(
        os.path.join(here, "..", "output", "c4_pilot_meld_train.csv")))
    ap.add_argument("--meld-dir", default=find_corpus_dir(here, "MELD"))
    ap.add_argument("--split", default="train")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        sys.exit(f"Not found: {a.csv}\n  Run c4_pilot_meld.py first.")

    # --- pilot rows, valid only ------------------------------------------
    rows = []
    with open(a.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "valid":
                continue
            rec = {"clip_id": r["clip_id"], "gender": r["gender"],
                   "person_id": r.get("person_id", "")}
            ok = True
            for k, _ in STATS:
                try:
                    rec[k] = float(r[k]) if r.get(k) not in ("", None) else float("nan")
                except ValueError:
                    rec[k] = float("nan")
            try:
                rec["duration_s"] = float(r["duration_s"])
            except (KeyError, ValueError):
                ok = False
            if ok and rec["duration_s"] > 0:
                rows.append(rec)

    if len(rows) < 20:
        sys.exit(f"Only {len(rows)} valid rows. Re-run the pilot with "
                 f"--n-clips 60 or more.")

    print(f"\n  pilot CSV  {a.csv}  ({len(rows)} valid clips)")

    # --- census from the annotation CSV ----------------------------------
    root = find_meld_root(a.meld_dir)
    if root:
        cpath = find_split_csv(root, a.split)
        if cpath:
            census(read_utterances(cpath), np)
    else:
        print("\n  (MELD folder not found, skipping the corpus census.)")

    verdicts = duration_check(rows, np)
    gender_check(rows, np, verdicts)
    variance_split(rows, np)

    print("\n  Caveat: this is the train split, explored after seeing the data.")
    print("  Any statistic chosen on the strength of these numbers must be")
    print("  confirmed on dev or test, not re-reported from here.")


if __name__ == "__main__":
    main()
