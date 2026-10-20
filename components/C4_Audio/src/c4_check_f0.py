"""
C4 — diagnostic: is f0_std measuring voices, or pitch-tracker errors?

WHY THIS EXISTS
---------------
The v2 batch run found that women's f0_std is higher than men's in 61 of 64
meetings, in semitones as well as hertz. That was the opposite of what the
scaling argument predicted, so the result needs a competing explanation ruled
out before it is believed.

THE COMPETING EXPLANATION: OCTAVE ERRORS
----------------------------------------
librosa.yin sometimes reports half or double the true pitch. A single octave
jump in an otherwise steady track roughly doubles the standard deviation.
Female voices sit nearer the top of the search range (fmax = 400 Hz), so
halving errors are more likely for them - which would produce exactly the
observed pattern without any real difference in how people speak.

Two things make this suspicious in the batch output:
  - the gender gap is huge at Edinburgh (about -40 Hz) and near zero at Idiap
    (about -2 Hz), though the code is identical;
  - Idiap recordings are roughly 20 LU louder, so the signal the tracker sees
    is quite different.

WHAT THIS SCRIPT DOES
---------------------
For a sample of turns from each gender it compares two estimators on the same
audio:

  yin   - what the batch used. Fast, no voicing decision, octave-error prone.
  pyin  - probabilistic YIN. Returns a voicing flag and is markedly more
          robust to octave errors, at the cost of speed.

and reports, per gender:

  median f0              sanity check (female ~165-255 Hz, male ~85-180 Hz)
  f0_std (yin, pyin)     the feature itself, both ways
  octave-jump rate       fraction of frame-to-frame steps near a factor of 2;
                         a clean track is well under 1 per cent
  voiced fraction        how much of the turn pyin considers voiced at all

If pyin's gender gap is much smaller than yin's, the batch result was a
measurement artefact and f0_std must be recomputed with pyin. If both agree,
the finding stands and the explanation lies elsewhere.

USAGE
-----
    python c4_check_f0.py
    python c4_check_f0.py --meetings ES2008a,IS1000b --per-gender 6
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

MIN_SEG_S = 1.5
NS = "{http://nite.sourceforge.net/}"
ROLE_SUFFIX = re.compile(r"(PM|ME|UID|UI|ID)$")

# One Edinburgh series and one Idiap series: the two sites whose f0 gap
# differed by about 40 Hz in the batch run.
DEFAULT_MEETINGS = "ES2008a,ES2008b,IS1000a,IS1000b"


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


def need(mod):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"Missing package '{mod}'.  pip install librosa numpy")


def seat_map(ann_zip, obs):
    with zipfile.ZipFile(ann_zip) as z:
        meets = ET.fromstring(z.read("corpusResources/meetings.xml"))
    for m in meets.iter("meeting"):
        if m.get("observation") != obs:
            continue
        out = {}
        for s in m.iter("speaker"):
            seat = s.get("nxt_agent") or s.get("agent") or s.get("channel")
            g = s.get("global_name")
            if seat and g:
                out[seat] = ROLE_SUFFIX.sub("", g)
        return out
    return {}


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
    segs.sort()
    return segs


def octave_jump_rate(f0, np):
    """Fraction of consecutive voiced frames whose ratio is near 2 or 0.5.

    A genuine pitch contour moves smoothly; a doubling or halving between
    adjacent frames is the tracker slipping an octave, not a speaker.
    """
    v = f0[np.isfinite(f0) & (f0 > 0)]
    if len(v) < 3:
        return float("nan")
    ratio = v[1:] / v[:-1]
    jumps = ((ratio > 1.8) & (ratio < 2.2)) | ((ratio > 0.45) & (ratio < 0.55))
    return float(jumps.mean())


def main():
    np = need("numpy")
    librosa = need("librosa")

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Compare yin and pyin on the same AMI turns.")
    ap.add_argument("--ami-dir", default=find_ami_dir(here))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "output")))
    ap.add_argument("--meetings", default=DEFAULT_MEETINGS)
    ap.add_argument("--per-gender", type=int, default=6)
    a = ap.parse_args()

    ann_zip = os.path.join(a.ami_dir, "ami_public_manual_1.6.2.zip")
    corpus = os.path.join(a.ami_dir, "amicorpus")
    gm_path = os.path.join(a.out_dir, "gender_map_ami.csv")
    os.makedirs(a.out_dir, exist_ok=True)
    for p in (ann_zip, gm_path):
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}")

    gender_of = {}
    with open(gm_path) as fh:
        for row in csv.DictReader(fh):
            gender_of[row["person_id"].split(":", 1)[1]] = row["gender"]

    rows = []
    for obs in [m.strip() for m in a.meetings.split(",") if m.strip()]:
        wav = os.path.join(corpus, obs, "audio", f"{obs}.Mix-Headset.wav")
        if not os.path.exists(wav):
            print(f"  {obs}: no audio on disk, skipping")
            continue
        seats = seat_map(ann_zip, obs)
        segs = segments(ann_zip, obs)
        y, sr = librosa.load(wav, sr=None, mono=True)

        usable = [(st, en, seats.get(seat), gender_of.get(seats.get(seat), ""))
                  for st, en, seat in segs if en - st >= MIN_SEG_S]
        usable = [u for u in usable if u[3] in ("F", "M")]
        usable.sort(key=lambda r: r[1] - r[0], reverse=True)

        counts = {"F": 0, "M": 0}
        print(f"\n  {obs}  ({len(usable)} usable turns)")
        for st, en, person, g in usable:
            if counts[g] >= a.per_gender:
                continue
            counts[g] += 1
            seg = y[int(st * sr):int(en * sr)].astype(float)

            f_yin = librosa.yin(seg, fmin=60, fmax=400, sr=sr, frame_length=1024)
            v_yin = f_yin[np.isfinite(f_yin) & (f_yin > 60) & (f_yin < 400)]

            f_pyin, voiced, _ = librosa.pyin(seg, fmin=60, fmax=400, sr=sr,
                                             frame_length=1024)
            v_pyin = f_pyin[np.isfinite(f_pyin)]

            if len(v_yin) < 6 or len(v_pyin) < 6:
                continue

            rows.append({
                "meeting": obs, "person": person, "gender": g,
                "dur_s": round(en - st, 1),
                "median_yin": float(np.median(v_yin)),
                "std_yin": float(np.std(v_yin)),
                "semi_yin": float(np.std(12 * np.log2(v_yin))),
                "median_pyin": float(np.median(v_pyin)),
                "std_pyin": float(np.std(v_pyin)),
                "semi_pyin": float(np.std(12 * np.log2(v_pyin))),
                "octave_yin": octave_jump_rate(f_yin, np),
                "octave_pyin": octave_jump_rate(f_pyin, np),
                "voiced_frac": float(np.mean(voiced)) if voiced is not None else float("nan"),
            })
            print(f"    {g} {person:<8} {en-st:5.1f}s   "
                  f"yin med {np.median(v_yin):6.1f} std {np.std(v_yin):6.1f}   "
                  f"pyin med {np.median(v_pyin):6.1f} std {np.std(v_pyin):6.1f}   "
                  f"oct {octave_jump_rate(f_yin, np):5.1%}/{octave_jump_rate(f_pyin, np):5.1%}")

    if not rows:
        sys.exit("\nNo measurable turns. Check the meeting names and the gender map.")

    out = os.path.join(a.out_dir, "c4_f0_check.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- the comparison that answers the question --------------------
    print("\n" + "=" * 72)
    print("  Per gender, averaged over the sampled turns\n")
    print(f"  {'':8}{'n':>4}{'med f0':>9}{'std yin':>10}{'std pyin':>11}"
          f"{'semi yin':>11}{'semi pyin':>12}{'oct yin':>10}{'oct pyin':>10}")
    agg = {}
    for g in ("F", "M"):
        sel = [r for r in rows if r["gender"] == g]
        if not sel:
            continue

        def m(k):
            v = [r[k] for r in sel if np.isfinite(r[k])]
            return float(np.mean(v)) if v else float("nan")

        agg[g] = {k: m(k) for k in ("median_yin", "std_yin", "std_pyin",
                                    "semi_yin", "semi_pyin",
                                    "octave_yin", "octave_pyin")}
        a2 = agg[g]
        print(f"  {g:<8}{len(sel):>4}{a2['median_yin']:>9.1f}{a2['std_yin']:>10.1f}"
              f"{a2['std_pyin']:>11.1f}{a2['semi_yin']:>11.2f}{a2['semi_pyin']:>12.2f}"
              f"{a2['octave_yin']:>9.1%}{a2['octave_pyin']:>10.1%}")

    if "F" in agg and "M" in agg:
        gap_yin = agg["M"]["semi_yin"] - agg["F"]["semi_yin"]
        gap_pyin = agg["M"]["semi_pyin"] - agg["F"]["semi_pyin"]
        print(f"\n  semitone gap (M - F):   yin {gap_yin:+.2f} st    pyin {gap_pyin:+.2f} st")

        oct_f = agg["F"]["octave_yin"]
        oct_m = agg["M"]["octave_yin"]
        print(f"  octave-jump rate (yin): F {oct_f:.1%}   M {oct_m:.1%}")
        print()
        if oct_f > 0.02 or oct_m > 0.02:
            print("  Octave jumps above 2% - yin is slipping octaves on this audio.")
            print("  f0_std from the batch run is contaminated. Recompute with pyin.")
        if abs(gap_yin) > 0.3 and abs(gap_pyin) < abs(gap_yin) * 0.5:
            print("  pyin more than halves the gender gap: the batch finding was")
            print("  largely a tracker artefact, not a property of the speakers.")
        elif abs(gap_pyin) >= abs(gap_yin) * 0.5:
            print("  pyin reproduces most of the gap: the finding survives this check.")
            print("  Octave errors are not the explanation; report it as it stands.")

    print(f"\n  Wrote {out}")
    print("\n  Sanity reference: typical median f0 is roughly 165-255 Hz for adult")
    print("  female speech and 85-180 Hz for adult male. A median far outside that")
    print("  range for a speaker is itself evidence of tracking failure.")


if __name__ == "__main__":
    main()
