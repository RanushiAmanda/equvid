"""
C4 — diagnostic: is dynamic_range measuring delivery, or turn length?

THE SUSPICION
-------------
The series-level analysis found dynamic_range significant on AMI (13/16,
p = 0.021, men wider by about 3 LU). AMI has no sound mixer, so a genuine
"this character was allowed a wider range" effect cannot be what is being
measured there. Something else is.

The likely cause is the statistic itself. dynamic_range is computed as

    max(short-term loudness blocks) - min(short-term loudness blocks)

and the range of a sample grows with sample size: more blocks means more
chances to catch an extreme. Blocks come from turn duration, so a longer turn
yields a wider range even if the speaker's delivery is identical.

The extraction picks the 12 LONGEST turns per gender. Ten of the sixteen AMI
series are 1F:3M - three men against one woman. If men's longest-12 are
systematically longer than women's longest-12, dynamic_range is partly
measuring duration, which belongs to the temporal component, not this one.

WHAT THIS CHECKS
----------------
It replays the exact selection rule used by c4_batch_ami3.py - longest turns
first, at least MIN_SEG_S, up to PER_GENDER per gender - and reports the
duration of the selected turns by gender, per series. No audio is read, so it
finishes in seconds.

If durations are balanced, the dynamic_range result survives this challenge.
If men's turns are longer, the feature needs replacing with a length-robust
spread measure (interquartile range or the standard deviation of the block
loudnesses) before the finding means anything.

USAGE
-----
    python c4_check_duration.py
"""

import argparse
import csv
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

MIN_SEG_S = 1.0        # must match c4_batch_ami3.py
PER_GENDER = 12        # must match c4_batch_ami3.py
PITCH_CAP_S = 12.0     # for reference: pitch is capped, loudness blocks are not
NS = "{http://nite.sourceforge.net/}"
ROLE_SUFFIX = re.compile(r"(PM|ME|UID|UI|ID)$")


def find_ami_dir(start):
    d = os.path.abspath(start)
    for _ in range(8):
        cand = os.path.join(d, "AMI")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.normpath(os.path.join(start, "..", "..", "..", "..", "AMI"))


def binom_two_sided(k, n):
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def seat_maps(ann_zip):
    with zipfile.ZipFile(ann_zip) as z:
        meets = ET.fromstring(z.read("corpusResources/meetings.xml"))
    out = {}
    for m in meets.iter("meeting"):
        obs = m.get("observation")
        if not obs:
            continue
        seats = {}
        for s in m.iter("speaker"):
            seat = s.get("nxt_agent") or s.get("agent") or s.get("channel")
            g = s.get("global_name")
            if seat and g:
                seats[seat] = ROLE_SUFFIX.sub("", g)
        out[obs] = seats
    return out


def segments(ann_zip, obs):
    segs = []
    with zipfile.ZipFile(ann_zip) as z:
        for name in z.namelist():
            if not re.search(rf"segments/{re.escape(obs)}\.[A-Z]\.segments\.xml$", name):
                continue
            letter = re.search(rf"{re.escape(obs)}\.([A-Z])\.segments\.xml$", name).group(1)
            root = ET.fromstring(z.read(name))
            for seg in root.iter("segment"):
                st, en = seg.get("transcriber_start"), seg.get("transcriber_end")
                if st is not None and en is not None:
                    segs.append((float(st), float(en), letter))
    return segs


def main():
    try:
        import numpy as np
    except ImportError:
        sys.exit("Missing numpy.")

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Does dynamic_range track turn duration?")
    ap.add_argument("--ami-dir", default=find_ami_dir(here))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "output")))
    a = ap.parse_args()

    ann_zip = os.path.join(a.ami_dir, "ami_public_manual_1.6.2.zip")
    corpus = os.path.join(a.ami_dir, "amicorpus")
    gm_path = os.path.join(a.out_dir, "gender_map_ami.csv")
    for p in (ann_zip, gm_path):
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}")

    gender_of = {}
    with open(gm_path) as fh:
        for row in csv.DictReader(fh):
            gender_of[row["person_id"].split(":", 1)[1]] = row["gender"]

    smaps = seat_maps(ann_zip)

    meetings = []
    if os.path.isdir(corpus):
        for obs in sorted(os.listdir(corpus)):
            if os.path.exists(os.path.join(corpus, obs, "audio", f"{obs}.Mix-Headset.wav")):
                meetings.append(obs)
    if not meetings:
        sys.exit(f"No audio found under {corpus}")

    rows = []
    for obs in meetings:
        seats = smaps.get(obs, {})
        segs = segments(ann_zip, obs)
        usable = []
        for st, en, seat in segs:
            if en - st < MIN_SEG_S:
                continue
            person = seats.get(seat)
            g = gender_of.get(person, "")
            if g in ("F", "M"):
                usable.append((en - st, g))
        usable.sort(reverse=True)

        picked = {"F": [], "M": []}
        for dur, g in usable:
            if len(picked[g]) < PER_GENDER:
                picked[g].append(dur)
        if not picked["F"] or not picked["M"]:
            continue

        rows.append({
            "meeting": obs, "series": obs[:6],
            "mean_F": float(np.mean(picked["F"])), "mean_M": float(np.mean(picked["M"])),
            "median_F": float(np.median(picked["F"])), "median_M": float(np.median(picked["M"])),
            "n_F": len(picked["F"]), "n_M": len(picked["M"]),
            # total pool available to choose from, which is the other half of
            # the story: with three men in the room they simply have more turns
            "pool_F": sum(1 for _, g in usable if g == "F"),
            "pool_M": sum(1 for _, g in usable if g == "M"),
        })

    if not rows:
        sys.exit("No meetings produced both genders.")

    # --- aggregate to series ------------------------------------------
    by_series = {}
    for r in rows:
        by_series.setdefault(r["series"], []).append(r)

    print(f"\n  nominal n = {len(rows)} meetings, effective n = {len(by_series)} series\n")
    hdr = (f"  {'series':<9}{'mean dur F':>12}{'mean dur M':>12}{'  M-F':>9}"
           f"{'   pool F':>10}{'pool M':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    diffs, pool_ratio = [], []
    for s in sorted(by_series):
        ms = by_series[s]
        mf = float(np.mean([m["mean_F"] for m in ms]))
        mm = float(np.mean([m["mean_M"] for m in ms]))
        pf = float(np.mean([m["pool_F"] for m in ms]))
        pm = float(np.mean([m["pool_M"] for m in ms]))
        diffs.append(mm - mf)
        pool_ratio.append(pm / pf if pf else float("nan"))
        print(f"  {s:<9}{mf:>12.1f}{mm:>12.1f}{mm - mf:>+9.1f}{pf:>10.0f}{pm:>9.0f}")

    arr = np.array(diffs)
    k = int((arr > 0).sum())
    p = binom_two_sided(k, len(arr))

    print(f"\n  selected-turn duration, male minus female")
    print(f"    mean {arr.mean():+.2f} s   SD {arr.std(ddof=1):.2f}   "
          f"M-longer in {k}/{len(arr)}   p = {p:.4f}")
    pr = np.array([x for x in pool_ratio if np.isfinite(x)])
    print(f"    available-turn pool, M:F ratio   mean {pr.mean():.2f}")

    out = os.path.join(a.out_dir, "c4_turn_durations.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    if p < 0.05:
        print("  VERDICT: selected-turn durations differ significantly by gender.")
        print("  max-minus-min grows with the number of blocks measured, so")
        print("  dynamic_range is partly measuring duration. The 13/16 result")
        print("  cannot be read as a delivery difference. Replace the statistic")
        print("  with a length-robust one - interquartile range, or the standard")
        print("  deviation of the block loudnesses - and re-run before reporting.")
    else:
        print("  VERDICT: selected-turn durations are balanced across genders")
        print(f"  (p = {p:.3f}). The duration confound does not explain the")
        print("  dynamic_range result, which survives this challenge. Note that")
        print("  a range statistic is still less stable than a spread statistic;")
        print("  reporting both would be stronger.")
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
