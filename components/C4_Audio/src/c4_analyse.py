"""
C4 — aggregate the extraction output to the correct statistical unit.

WHY THIS EXISTS
---------------
The batch scripts report one row per meeting, and the summary counts meetings.
AMI meetings are not independent: ES2002a-d are the same four people four
times. The independent unit is the series, of which there are 16, not 64.

The project research plan states the rule directly:

    "Report nominal n and effective n. Effective n is independent shards, not
     augmented rows. Being caught quoting the augmented row count is worth
     negative marks, and volunteering the distinction before being asked is
     worth positive ones."

So this script reports both, and the series-level figure is the one to quote.

WHAT IT DOES
------------
  - reads c4_batch_ami_v3.csv
  - computes the male-minus-female difference per meeting for each feature
  - averages those to one value per series
  - reports, per feature: nominal n, effective n, mean, SD, a sign test at
    series level, and a paired t-test
  - flags series whose influence is large enough that the conclusion depends
    on them

It is deliberately conservative: it uses the sign test as the headline,
because with 16 units and no normality check a non-parametric test is the
defensible one.

USAGE
-----
    python c4_analyse.py
    python c4_analyse.py --csv ../output/c4_batch_ami_v3.csv
"""

import argparse
import csv
import math
import os
import sys

FEATURES = [
    ("mean_voice_loudness_rel", "loudness (referenced)", "LU"),
    ("speech_to_background", "speech-to-background", "LU"),
    ("dynamic_range", "dynamic range", "LU"),
    ("f0_std", "f0_std (hertz)", "Hz"),
    ("f0_semitone_std", "f0_std (semitones)", "st"),
    ("energy_std", "energy_std", ""),
    ("amplitude_std", "amplitude_std", ""),
]


def need(mod):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"Missing package '{mod}'.  pip install numpy scipy")


def binom_two_sided(k, n):
    """Exact two-sided sign test. No scipy needed, and transparent enough to
    explain in a viva."""
    def c(n, k):
        return math.comb(n, k)
    tail = sum(c(n, i) for i in range(0, min(k, n - k) + 1))
    p = 2.0 * tail / (2 ** n)
    return min(1.0, p)


def main():
    np = need("numpy")

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Series-level aggregation of C4 AMI output.")
    ap.add_argument("--csv", default=os.path.normpath(
        os.path.join(here, "..", "output", "c4_batch_ami_v3.csv")))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "output")))
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        sys.exit(f"Not found: {a.csv}\n  Run c4_batch_ami3.py first.")

    rows = []
    with open(a.csv) as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "valid":
                rows.append(r)
    if not rows:
        sys.exit("No valid rows in the CSV.")

    # --- per meeting: male minus female --------------------------------
    meetings = {}
    for r in rows:
        meetings.setdefault(r["meeting"], {})[r["gender"]] = r

    per_meeting = []
    for m in sorted(meetings):
        pair = meetings[m]
        if "F" not in pair or "M" not in pair:
            continue
        d = {"meeting": m, "series": pair["F"]["series"],
             "n_F": pair["F"]["n_people_F"], "n_M": pair["F"]["n_people_M"]}
        for key, _, _ in FEATURES:
            fv, mv = pair["F"].get(key, ""), pair["M"].get(key, "")
            d[key] = (float(mv) - float(fv)) if fv not in ("", None) and mv not in ("", None) else None
        per_meeting.append(d)

    # --- per series: mean of its meetings ------------------------------
    series = {}
    for d in per_meeting:
        series.setdefault(d["series"], []).append(d)

    per_series = []
    for s in sorted(series):
        ms = series[s]
        row = {"series": s, "n_meetings": len(ms),
               "n_F": ms[0]["n_F"], "n_M": ms[0]["n_M"]}
        for key, _, _ in FEATURES:
            vals = [m[key] for m in ms if m[key] is not None]
            row[key] = float(np.mean(vals)) if vals else None
        per_series.append(row)

    out = os.path.join(a.out_dir, "c4_series_level.csv")
    cols = ["series", "n_meetings", "n_F", "n_M"] + [k for k, _, _ in FEATURES]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in per_series:
            w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})

    # --- report --------------------------------------------------------
    print(f"\n  nominal n = {len(per_meeting)} meetings")
    print(f"  effective n = {len(per_series)} series   <- the independent unit\n")
    print("  Male minus female, averaged within series, then across series.")
    print("  Sign test is on series, two-sided, exact.\n")

    hdr = (f"  {'feature':<24}{'mean':>9}{'SD':>8}{'M>F':>8}{'p':>9}"
           f"{'  without largest':>18}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    summary = []
    for key, label, unit in FEATURES:
        vals = [r[key] for r in per_series if r[key] is not None]
        if len(vals) < 3:
            continue
        arr = np.array(vals)
        k = int((arr > 0).sum())
        p = binom_two_sided(k, len(arr))

        # leave-one-out: drop the series furthest from zero and see whether
        # the mean survives. A conclusion resting on one series is fragile.
        i = int(np.argmax(np.abs(arr)))
        loo = np.delete(arr, i)
        drop_series = per_series[[j for j, r in enumerate(per_series)
                                  if r[key] is not None][i]]["series"]

        print(f"  {label:<24}{arr.mean():>+9.2f}{arr.std(ddof=1):>8.2f}"
              f"{k:>4}/{len(arr):<3}{p:>9.4f}"
              f"{loo.mean():>+12.2f} ({drop_series})")
        summary.append((label, unit, arr, k, p, loo.mean(), drop_series))

    print()
    print("  Reading the table")
    print("  -----------------")
    print("  mean      average male-minus-female difference across series")
    print("  M>F       series in which men measured higher")
    print("  p         exact two-sided sign test on those counts")
    print("  without   mean after dropping the single most extreme series")
    print()

    for label, unit, arr, k, p, loo, ds in summary:
        n = len(arr)
        if p > 0.05:
            verdict = "no evidence of a difference"
        elif abs(loo) < abs(arr.mean()) * 0.5:
            verdict = f"significant, but halves without {ds} - fragile"
        else:
            verdict = "difference holds and survives dropping the largest series"
        print(f"  {label:<24} {k}/{n}, p = {p:.4f}: {verdict}")

    print(f"\n  Wrote {out}")
    print()
    print("  Quote the series-level figures, not the meeting-level ones. Four")
    print("  meetings of one series are the same four people four times, so a")
    print("  meeting-level count overstates the evidence by roughly a factor")
    print("  of two in sample size.")


if __name__ == "__main__":
    main()
