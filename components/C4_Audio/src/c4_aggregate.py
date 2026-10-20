"""
C4 - aggregate per-clip features to the shared output contract.

WHY THIS EXISTS
---------------
Two separate problems, one script.

1. THE OUTPUT CONTRACT. M2's shared conventions require every component to
   write ONE ROW PER clip_id, carrying its score, status, reason and
   feature_version. C4 currently writes one row per utterance on MELD and two
   rows per meeting on AMI. Per-utterance rows are intermediate data; this
   produces the deliverable.

2. THE SIGN. M2 decision 3: a positive score means WOMEN above men. Every C4
   script so far computed male minus female, so every number produced to date
   carries the wrong sign. This script is the single place the convention is
   applied, so it cannot drift again.

HOW PARITY IS COMPUTED, AND WHY IT DIFFERS BY FEATURE
-----------------------------------------------------
M2: "scores are built from log ratios of per-person quantities (F over M), so
+x and -x are symmetric."

Some C4 features are ALREADY in a log domain, and for those a plain difference
IS the log ratio:

  - loudness and dynamic range are in LU. A loudness unit is 10*log10 of a
    power ratio, so F minus M in LU is already a log ratio. Subtracting is
    correct; dividing would be meaningless.
  - f0_std in semitones is log2-based for the same reason.

Others are linear and need an explicit log ratio:

  - energy_std and amplitude_std are linear standard deviations, so parity is
    log(F / M).

Mixing these up would make some features asymmetric in sign and break the
+x / -x symmetry the convention depends on.

PER-PERSON NORMALISATION
------------------------
M2 divides by the number of persons of each gender, because speaking time is a
SHARED resource: three men split the floor differently from one woman, and
comparing group totals would confuse headcount with behaviour.

C4's quantities are not shared resources. Each utterance has its own loudness
regardless of how many people are in the room, and this script already averages
across each gender's clips, so the per-person average is what it computes. No
further division applies.

What does carry over is the warning: on AMI, ten of sixteen series are 1F:3M,
so the male mean is averaged over three voices and the female mean over one.
The male figure is better estimated. n_F and n_M are written to every row so
that asymmetry stays visible.

USAGE
-----
    python c4_aggregate.py
    python c4_aggregate.py --csv ../output/c4_meld_v2_train_w2.csv
    python c4_aggregate.py --corpus AMI --csv ../output/c4_batch_ami_v4.csv
"""

import argparse
import csv
import math
import os
import sys

FEATURE_VERSION = "c4-aggregate-0.1"

# Minimum clips per gender before an episode's mean is trusted. Below this the
# row is written as NA with a reason rather than silently dropped, which the
# output contract requires.
MIN_CLIPS_PER_GENDER = 3

# (column, label, domain, required). The domain decides how parity is computed:
#   "log"    already in a log domain -> parity is F minus M
#   "linear" linear quantity         -> parity is log(F / M)
#
# `required` marks the four features that actually enter the equation. A clip
# missing one of those cannot be scored and becomes NA. The other two are a
# control (rms_dbfs, which only the MELD extractor computes) and a placeholder
# (s2b_proxy, pending a separation method) - a corpus not supplying them is a
# gap in the comparison, not a failure to score the clip, so they must not
# force the row to NA.
FEATURES = [
    ("mean_voice_loudness", "loudness", "log", True),
    ("dynamic_range", "dynamic_range", "log", True),
    ("f0_semitone_std", "f0_std_semitones", "log", True),
    ("energy_std", "energy_std", "linear", True),
    ("rms_dbfs", "rms_control", "log", False),
    ("s2b_proxy", "s2b_proxy", "log", False),
]

# Dropped by decision D6: r = +0.85 with mean_voice_loudness, which breaches
# DR4. As implemented it is the std of the raw samples, which for a zero-mean
# signal is the RMS - a level measure, not a spread measure. Kept in the output
# for the record, excluded from the feature set.
RETIRED = [("amplitude_std", "amplitude_std_RETIRED", "linear", False),
           ("f0_std", "f0_std_hz_RETIRED", "linear", False)]

# The AMI and MELD extractors grew separately and do not use the same column
# names. In one case they do not even hold the same statistic, so mapping them
# here is not cosmetic.
COLUMN_MAP = {
    # MELD's extractor already writes the names used above.
    "MELD": {},
    "AMI": {
        # AMI loudness is referenced within each meeting, because Edinburgh and
        # Idiap recorded at different gain (roughly 20 LU apart) and absolute
        # levels are not comparable across sites. MELD's is absolute. The raw
        # F and M means are therefore NOT comparable across the two corpora -
        # but parity is, because subtracting M from F cancels the reference.
        "mean_voice_loudness": "mean_voice_loudness_rel",
        # Decision D1: on AMI the IQR is the defensible estimator, because
        # max-minus-min grows with turn length and AMI turn durations are not
        # balanced by gender. The AMI column called plain "dynamic_range" IS
        # max-minus-min - the one D1 rejected - so aggregating it would quietly
        # contradict the decision log.
        "dynamic_range": "dynamic_range_iqr",
        "s2b_proxy": "speech_to_background",
    },
}


def need(mod, hint):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"Missing package '{mod}'.  pip install {hint}")


def parity(f_val, m_val, domain):
    """Women minus men, in the right domain. None if not computable."""
    if f_val is None or m_val is None:
        return None
    if domain == "log":
        return float(f_val) - float(m_val)
    # linear: a log ratio needs both sides strictly positive
    if f_val <= 0 or m_val <= 0:
        return None
    return float(math.log(f_val / m_val))


def mean_or_none(vals, np):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def read_meld(path, np, colmap):
    """Per-utterance rows from c4_pilot_meld2.py, grouped by clip_id."""
    groups = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "valid":
                continue
            cid = r["clip_id"]
            g = r["gender"]
            if g not in ("F", "M"):
                continue
            d = groups.setdefault(cid, {
                "clip_id": cid, "corpus": "MELD",
                # MELD:S08E21 -> MELD_S08. Series groups episodes of one season,
                # matching M2's MELD_S08 example.
                "series": "MELD_S" + cid.split(":")[1][1:3],
                # All of MELD is one person-connected component: the same six
                # characters appear in every episode, so it can only ever be
                # held out whole (leave-one-corpus-out).
                "component": "MELD_ALL",
                "gender_source": "manual_map",
                "F": {}, "M": {}, "persons": {"F": set(), "M": set()},
            })
            d["persons"][g].add(r.get("person_id", ""))
            for key, _, _, _ in FEATURES + RETIRED:
                src = colmap.get(key, key)
                try:
                    v = float(r[src]) if r.get(src) not in ("", None) else None
                except (KeyError, ValueError):
                    v = None
                d[g].setdefault(key, []).append(v)
    return groups


def read_ami(path, np, colmap):
    """AMI rows are already one per meeting per gender, so this only reshapes
    them and attaches the shared ID columns."""
    groups = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") not in (None, "", "valid"):
                continue
            meeting = r.get("meeting") or r.get("clip_id", "")
            if not meeting:
                continue
            meeting = meeting.split(":")[-1]
            g = r.get("gender")
            if g not in ("F", "M"):
                continue
            cid = f"AMI:{meeting}"
            d = groups.setdefault(cid, {
                "clip_id": cid, "corpus": "AMI",
                "series": r.get("series") or meeting[:6],
                "component": r.get("component", ""),
                "gender_source": "participants_xml",
                "F": {}, "M": {}, "persons": {"F": set(), "M": set()},
            })
            # AMI rows carry a headcount rather than person IDs
            try:
                n = int(float(r.get(f"n_people_{g}", 0) or 0))
            except ValueError:
                n = 0
            d["persons"][g] = set(range(n)) if n else d["persons"][g]
            for key, _, _, _ in FEATURES + RETIRED:
                src = colmap.get(key, key)
                try:
                    v = float(r[src]) if r.get(src) not in ("", None) else None
                except (KeyError, ValueError):
                    v = None
                d[g].setdefault(key, []).append(v)
    return groups


def main():
    np = need("numpy", "numpy")
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Aggregate C4 per-clip features to one row per clip_id.")
    ap.add_argument("--csv", default=os.path.normpath(os.path.join(
        here, "..", "output", "c4_meld_v2_train_w2.csv")))
    ap.add_argument("--corpus", default="MELD", choices=["MELD", "AMI"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-clips", type=int, default=None,
                    help="minimum clips per gender; default 3 for MELD, "
                         "1 for AMI whose rows are already aggregates")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        sys.exit(f"Not found: {a.csv}")

    # An AMI row is one meeting-gender aggregate over up to 12 turns, so a
    # single row IS the complete observation for that gender. A MELD row is one
    # utterance, so several are needed before an episode mean is trustworthy.
    if a.min_clips is None:
        a.min_clips = 1 if a.corpus == "AMI" else MIN_CLIPS_PER_GENDER
    colmap = COLUMN_MAP[a.corpus]
    groups = (read_meld if a.corpus == "MELD" else read_ami)(a.csv, np, colmap)
    if not groups:
        sys.exit("No valid rows found. Check --corpus matches the file.")

    out_rows = []
    for cid in sorted(groups):
        d = groups[cid]
        n_clips_F = len([v for v in d["F"].get("mean_voice_loudness", [])
                         if v is not None])
        n_clips_M = len([v for v in d["M"].get("mean_voice_loudness", [])
                         if v is not None])
        row = {
            "clip_id": cid, "corpus": d["corpus"], "series": d["series"],
            "component": d["component"], "gender_source": d["gender_source"],
            "n_F": len([p for p in d["persons"]["F"] if p != ""]) or None,
            "n_M": len([p for p in d["persons"]["M"] if p != ""]) or None,
            "n_clips_F": n_clips_F, "n_clips_M": n_clips_M,
            "feature_version": FEATURE_VERSION,
            "status": "valid", "reason": "",
        }

        # Contract: a clip that cannot be scored gets NA with a reason, never
        # a silent drop.
        if n_clips_F < a.min_clips or n_clips_M < a.min_clips:
            row["status"] = "NA"
            row["reason"] = (f"fewer than {a.min_clips} clips for one gender "
                             f"(F={n_clips_F}, M={n_clips_M})")

        for key, label, domain, req in FEATURES + RETIRED:
            fm = mean_or_none(d["F"].get(key, []), np)
            mm = mean_or_none(d["M"].get(key, []), np)
            row[f"F_{label}"] = fm
            row[f"M_{label}"] = mm
            # Positive = women above men (M2 decision 3). This is the one place
            # the sign convention is applied.
            row[f"parity_{label}"] = (parity(fm, mm, domain)
                                      if row["status"] == "valid" else None)
            # Only a REQUIRED feature can send the row to NA.
            if (req and row["status"] == "valid"
                    and row[f"parity_{label}"] is None):
                row["status"] = "NA"
                row["reason"] = row["reason"] or f"{label} not computable"

        out_rows.append(row)

    out = a.out or os.path.join(os.path.dirname(a.csv),
                                f"c4_clip_level_{a.corpus.lower()}.csv")
    cols = list(out_rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (round(r[k], 6) if isinstance(r.get(k), float)
                            else ("" if r.get(k) is None else r[k]))
                        for k in cols})

    # ---- report -------------------------------------------------------
    ok = [r for r in out_rows if r["status"] == "valid"]
    print(f"\n  {a.corpus}: {len(out_rows)} clip_ids, {len(ok)} valid, "
          f"{len(out_rows) - len(ok)} NA")
    if not ok:
        print("  Nothing scorable. Lower --min-clips or check the input.")
        return

    print(f"  series: {len(set(r['series'] for r in ok))}     "
          f"sign convention: POSITIVE = women above men\n")
    def binom(k, n):
        if n == 0:
            return float("nan")
        tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
        return min(1.0, 2.0 * tail / (2 ** n))

    print(f"      {'feature':<22}{'F mean':>10}{'M mean':>10}"
          f"{'parity':>10}{'SD':>8}{'F>M':>8}{'p':>9}")
    print("      " + "-" * 69)

    ties_seen = 0
    for key, label, _, _ in FEATURES:
        vals = [r[f"parity_{label}"] for r in ok
                if r[f"parity_{label}"] is not None]
        if len(vals) < 4:
            continue
        arr = np.array(vals)
        # A sign test is only defined on non-zero differences: an exact tie
        # favours neither gender and must be excluded, not counted against
        # women. With continuous features ties are rare, but a feature with no
        # variance at all would otherwise report every episode as "not F>M"
        # and produce a spuriously small p.
        nz = arr[arr != 0]
        ties = len(arr) - len(nz)
        ties_seen += ties
        fm = mean_or_none([r[f"F_{label}"] for r in ok], np)
        mm = mean_or_none([r[f"M_{label}"] for r in ok], np)
        if len(nz) < 4:
            print(f"      {label:<22}{fm:>10.3f}{mm:>10.3f}{arr.mean():>+10.3f}"
                  f"{arr.std(ddof=1):>8.3f}{'-':>8}{'(all tied)':>9}")
            continue
        kpos = int((nz > 0).sum())
        print(f"      {label:<22}{fm:>10.3f}{mm:>10.3f}{arr.mean():>+10.3f}"
              f"{arr.std(ddof=1):>8.3f}{kpos:>4}/{len(nz):<3}"
              f"{binom(kpos, len(nz)):>9.4f}"
              f"{f'   {ties} tied' if ties else ''}")

    print("\n      parity is women minus men (log domain) or log(F/M) (linear).")
    print("      A positive value means women measured higher.")

    # ---- series level, which for AMI is the only defensible unit -------
    # Decision D3: AMI meetings are not independent. ES2002a-d are the same
    # four people meeting four times, so a sign test over 64 meetings
    # overstates the evidence by roughly a factor of two in n. The series is
    # the independent unit, and there are 16 of them.
    #
    # MELD episodes are closer to independent - different scenes, different
    # dialogue - so the episode level above is defensible there. But M2's
    # conventions require the bootstrap to be over series in both corpora,
    # so the series figure is reported for MELD as well.
    by_series = {}
    for r in ok:
        by_series.setdefault(r["series"], []).append(r)

    print(f"\n  SERIES LEVEL   nominal n = {len(ok)} {('meetings' if ok[0]['corpus'] == 'AMI' else 'episodes')}"
          f"   effective n = {len(by_series)} series")
    if ok[0]["corpus"] == "AMI":
        print("  For AMI this is THE number to quote, not the one above (D3).")
    print()
    print(f"      {'feature':<22}{'parity':>10}{'SD':>8}{'F>M':>8}{'p':>9}"
          f"{'   clip-level p':>16}")
    print("      " + "-" * 73)
    for key, label, _, _ in FEATURES:
        per_series = []
        for sname in sorted(by_series):
            vals = [r[f"parity_{label}"] for r in by_series[sname]
                    if r[f"parity_{label}"] is not None]
            if vals:
                per_series.append(float(np.mean(vals)))
        clip_vals = [r[f"parity_{label}"] for r in ok
                     if r[f"parity_{label}"] is not None]
        if len(per_series) < 4:
            continue
        arr = np.array(per_series)
        nz = arr[arr != 0]
        if len(nz) < 4:
            print(f"      {label:<22}{arr.mean():>+10.3f}"
                  f"{arr.std(ddof=1):>8.3f}{'-':>8}{'(all tied)':>9}")
            continue
        kpos = int((nz > 0).sum())
        cv = np.array(clip_vals)
        cnz = cv[cv != 0]
        clip_p = (binom(int((cnz > 0).sum()), len(cnz)) if len(cnz) >= 4
                  else float("nan"))
        print(f"      {label:<22}{arr.mean():>+10.3f}{arr.std(ddof=1):>8.3f}"
              f"{kpos:>4}/{len(nz):<3}{binom(kpos, len(nz)):>9.4f}"
              f"{clip_p:>16.4f}")

    print("\n      A feature significant in the last column but not the one")
    print("      before it was significant only because repeated recordings of")
    print("      the same people were counted as independent observations.")
    print(f"\n  Wrote {out}")
    print("\n  This file is the C4 deliverable: one row per clip_id, with")
    print("  status, reason and feature_version, in M2's sign convention.")
    print("  It is also the input to the augmentation and the equation.")


if __name__ == "__main__":
    main()
