"""
C4 — AMI extraction, version 3: pyin pitch tracking.

WHAT CHANGED FROM VERSION 2
---------------------------
v2 used librosa.yin for pitch. The diagnostic in c4_check_f0.py showed that
yin systematically halves female pitch on this audio: for ES2008 it reported
female speakers at 95-111 Hz, which is male range, while pyin reported the
same turns at 172-201 Hz. Male speakers agreed between the two trackers to
within a hertz.

That is a gender-differential measurement failure, and it manufactured most of
v2's apparent finding: the semitone gap fell from -2.08 st under yin to
-0.63 st under pyin. Octave-jump rates were 5.3% for women against 3.0% for
men.

v3 therefore:
  - uses librosa.pyin, which carries a voicing decision and is far more
    resistant to octave slips;
  - caps the audio analysed for pitch at PITCH_CAP_S seconds per turn, because
    pyin is much slower and a long turn adds little to a variability estimate;
  - records the octave-jump rate per gender, so contamination is visible in
    the output instead of having to be hunted for.

Nothing else changes. Loudness, background and the dynamic-range features are
identical to v2, so the two runs are comparable on everything except pitch.

Original v2 notes follow.

---

C4 — AMI extraction, version 2: per-speaker background and referenced loudness.

WHAT CHANGED FROM c4_batch_ami.py, AND WHY
------------------------------------------
Version 1 estimated ONE noise floor per meeting and subtracted it from both
genders. That makes speech_to_background_ratio useless for a gender comparison,
because the shared constant cancels exactly:

    s2b_F - s2b_M = (loud_F - nf) - (loud_M - nf) = loud_F - loud_M

Every row of the v1 output confirmed it: the s2b difference equalled the
loudness difference to the decimal. The feature was measuring nothing that
mean_voice_loudness did not already measure.

Version 2 fixes two things.

1. PER-SPEAKER BACKGROUND.  For each speaking turn, the background is measured
   from the non-speech audio immediately before and after THAT turn, not from
   the whole meeting. On AMI this will stay close to null, because there is one
   room and one noise source - that is the expected control behaviour. The
   reason to fix it here is that on film the background genuinely differs per
   line: music swells under one character and ducks under another. The feature
   must be correctly defined before it meets data where it can move.

2. REFERENCED LOUDNESS.  v1 reported raw LUFS. Across AMI that ranges from
   about -14 (Idiap) to about -54 (Edinburgh), a 20 LU gap caused by gain
   staging, not voices. v2 also reports loudness relative to each meeting's own
   dialogue level, which removes it.

The script reports the correlation between the two gender differences, so you
can see on your own data whether the features have actually decoupled.

A NOTE ON METHOD
----------------
Background windows near a turn are concatenated before measurement. BS.1770
gates on short blocks, so joining non-adjacent silence is acceptable for an
integrated figure, but it is an approximation and belongs in the limitations.

OUTPUT
------
    output/c4_batch_ami_v2.csv      one row per meeting per gender

v1's output is left alone so the two can be compared.

USAGE
-----
    python c4_batch_ami2.py
    python c4_batch_ami2.py --no-pitch --limit 8
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

FEATURE_VERSION = "c4-ami-0.3"
MIN_SEG_S = 1.0          # BS.1770 gating needs >= 400 ms; 1 s is a safe floor
MIN_BG_S = 0.5           # minimum usable silence to call it background
BG_WINDOW_S = 4.0        # look this far either side of a turn for silence
BG_WINDOW_MAX_S = 12.0   # widen to this before giving up
PITCH_CAP_S = 12.0       # pyin is slow; 12 s is ample for a variability estimate
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


def need(mod):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"Missing package '{mod}'.  pip install librosa pyloudnorm soundfile numpy")


def load_seat_maps(ann_zip):
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
            gname = s.get("global_name")
            if seat and gname:
                seats[seat] = ROLE_SUFFIX.sub("", gname)
        out[obs] = seats
    return out


def load_segments(ann_zip, obs):
    segs = []
    with zipfile.ZipFile(ann_zip) as z:
        names = [n for n in z.namelist()
                 if re.search(rf"segments/{re.escape(obs)}\.[A-Z]\.segments\.xml$", n)]
        for name in names:
            letter = re.search(rf"{re.escape(obs)}\.([A-Z])\.segments\.xml$", name).group(1)
            root = ET.fromstring(z.read(name))
            for seg in root.iter("segment"):
                st, en = seg.get("transcriber_start"), seg.get("transcriber_end")
                if st is not None and en is not None:
                    segs.append((float(st), float(en), letter))
    segs.sort()
    return segs


def merge_speech(segs):
    """Union of all speech intervals, from every speaker. Anything outside is
    silence as far as the annotation is concerned."""
    merged = []
    for st, en, _ in sorted(segs):
        if merged and st <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged


def free_gaps(merged, lo, hi):
    """Sub-intervals of [lo, hi] where nobody is speaking."""
    out, cursor = [], lo
    for st, en in merged:
        if en <= lo:
            continue
        if st >= hi:
            break
        if st > cursor:
            out.append((cursor, min(st, hi)))
        cursor = max(cursor, en)
        if cursor >= hi:
            break
    if cursor < hi:
        out.append((cursor, hi))
    return [(a, b) for a, b in out if b - a > 0.05]


def local_background(y, sr, merged, st, en, dur, meter, np):
    """Loudness of the silence immediately around one turn.

    Widens the search window until enough silence is found, because a busy
    stretch of meeting can have none nearby. Returns None if even the widest
    window has less than MIN_BG_S of silence - that turn then gets no s2b
    value rather than a fabricated one.
    """
    w = BG_WINDOW_S
    while w <= BG_WINDOW_MAX_S:
        gaps = (free_gaps(merged, max(0.0, st - w), st)
                + free_gaps(merged, en, min(dur, en + w)))
        chunks, total = [], 0.0
        for a, b in gaps:
            chunks.append(y[int(a * sr):int(b * sr)])
            total += b - a
        if total >= MIN_BG_S and chunks:
            blk = np.concatenate(chunks).astype(float)
            if len(blk) >= int(0.4 * sr):
                try:
                    v = meter.integrated_loudness(blk)
                    if np.isfinite(v):
                        return float(v)
                except Exception:
                    pass
        w *= 2
    return None


def octave_jump_rate(f0, np):
    """Fraction of adjacent voiced frames whose ratio is near 2 or 0.5.

    A real pitch contour moves smoothly. A doubling or halving between
    neighbouring frames is the tracker slipping an octave. Reported so that a
    contaminated run is visible in the output rather than silently trusted.
    """
    v = f0[np.isfinite(f0) & (f0 > 0)]
    if len(v) < 3:
        return None
    ratio = v[1:] / v[:-1]
    jumps = ((ratio > 1.8) & (ratio < 2.2)) | ((ratio > 0.45) & (ratio < 0.55))
    return float(jumps.mean())


def measure_meeting(obs, wav, ann_zip, gender_of, seat_maps, per_gender,
                    do_pitch, np, librosa, pyln):
    seats = seat_maps.get(obs, {})
    if not seats:
        return None, "no seat map"
    segs = load_segments(ann_zip, obs)
    if not segs:
        return None, "no segments XML"

    y, sr = librosa.load(wav, sr=None, mono=True)
    dur = len(y) / sr
    meter = pyln.Meter(sr)
    merged = merge_speech(segs)

    usable = []
    for st, en, seat in segs:
        if en - st < MIN_SEG_S:
            continue
        person = seats.get(seat)
        g = gender_of.get(person, "")
        if g in ("F", "M"):
            usable.append((st, en, person, g))
    usable.sort(key=lambda r: r[1] - r[0], reverse=True)

    chosen, counts = [], {"F": 0, "M": 0}
    for row in usable:
        if counts[row[3]] < per_gender:
            chosen.append(row)
            counts[row[3]] += 1
    if counts["F"] == 0 or counts["M"] == 0:
        return None, f"one gender only (F={counts['F']}, M={counts['M']})"

    rows, no_bg = [], 0
    for st, en, person, g in chosen:
        seg = y[int(st * sr):int(en * sr)].astype(float)
        try:
            loud = meter.integrated_loudness(seg)
            if not np.isfinite(loud):
                continue
        except Exception:
            continue

        bg = local_background(y, sr, merged, st, en, dur, meter, np)
        if bg is None:
            no_bg += 1
        s2b = (float(loud) - bg) if bg is not None else None

        rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256)[0]

        f0_std = f0_semi = f0_med = octv = None
        if do_pitch:
            # pyin, not yin: see the header. Capped because pyin is slow and a
            # 50 s turn tells you no more about pitch variability than 12 s.
            pseg = seg[:int(PITCH_CAP_S * sr)]
            f0, voiced, _ = librosa.pyin(pseg, fmin=60, fmax=400, sr=sr,
                                         frame_length=1024)
            v = f0[np.isfinite(f0)]
            if len(v) > 5:
                f0_std = float(np.std(v))
                f0_med = float(np.median(v))
                # Semitones are a ratio scale: the same proportional wobble
                # gives the same number whatever the speaker's base pitch.
                f0_semi = float(np.std(12 * np.log2(v)))
                octv = octave_jump_rate(f0, np)

        win = int(0.4 * sr)
        st_loud = []
        for i in range(0, max(1, len(seg) - win), win // 2):
            blk = seg[i:i + win]
            if len(blk) < win:
                break
            try:
                v2 = meter.integrated_loudness(blk)
                if np.isfinite(v2):
                    st_loud.append(v2)
            except Exception:
                pass
        dyn = (max(st_loud) - min(st_loud)) if len(st_loud) >= 2 else None

        rows.append({"gender": g, "person": person,
                     "loud": float(loud), "bg": bg, "s2b": s2b, "dyn": dyn,
                     "f0_std": f0_std, "f0_semi": f0_semi,
                     "f0_med": f0_med, "octv": octv,
                     "energy_std": float(np.std(rms)),
                     "amplitude_std": float(np.std(np.abs(seg)))})

    if not rows:
        return None, "no measurable segments"

    # The meeting's own dialogue level: what every level feature is referenced
    # to, so a quiet recording and a loud one are comparable.
    ref = float(np.mean([r["loud"] for r in rows]))
    return rows, {"dur_min": dur / 60, "counts": counts, "ref": ref,
                  "no_bg": no_bg, "sr": sr}


def main():
    np = need("numpy")
    librosa = need("librosa")
    pyln = need("pyloudnorm")

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="C4 AMI extraction v3 (pyin pitch).")
    ap.add_argument("--ami-dir", default=find_ami_dir(here))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "output")))
    ap.add_argument("--per-gender", type=int, default=12)
    ap.add_argument("--no-pitch", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ann_zip = os.path.join(a.ami_dir, "ami_public_manual_1.6.2.zip")
    corpus = os.path.join(a.ami_dir, "amicorpus")
    gm_path = os.path.join(a.out_dir, "gender_map_ami.csv")
    clips_path = os.path.join(a.out_dir, "clips_ami.csv")
    os.makedirs(a.out_dir, exist_ok=True)

    for p, how in [(ann_zip, "python download_ami.py --list"),
                   (gm_path, "python build_group_tables.py")]:
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}\n  Produce it with:  {how}")

    gender_of = {}
    with open(gm_path) as fh:
        for row in csv.DictReader(fh):
            gender_of[row["person_id"].split(":", 1)[1]] = row["gender"]

    comp = {}
    if os.path.exists(clips_path):
        with open(clips_path) as fh:
            for row in csv.DictReader(fh):
                comp[row["meeting"]] = (row["n_F"], row["n_M"])

    seat_maps = load_seat_maps(ann_zip)

    meetings = []
    if os.path.isdir(corpus):
        for obs in sorted(os.listdir(corpus)):
            wav = os.path.join(corpus, obs, "audio", f"{obs}.Mix-Headset.wav")
            if os.path.exists(wav):
                meetings.append((obs, wav))
    if not meetings:
        sys.exit(f"No Mix-Headset audio found under {corpus}")
    if a.limit:
        meetings = meetings[:a.limit]

    print(f"{len(meetings)} meetings with audio on disk")
    print(f"up to {a.per_gender} segments per gender, per-speaker background"
          f"{'  (pitch skipped)' if a.no_pitch else ''}\n")

    out_rows, skipped, total_no_bg = [], [], 0
    for i, (obs, wav) in enumerate(meetings, 1):
        print(f"  [{i}/{len(meetings)}] {obs} ", end="", flush=True)
        try:
            rows, info = measure_meeting(obs, wav, ann_zip, gender_of, seat_maps,
                                         a.per_gender, not a.no_pitch, np, librosa, pyln)
        except Exception as e:
            print(f"ERROR: {e}")
            skipped.append((obs, str(e)))
            continue
        if rows is None:
            print(f"skipped: {info}")
            skipped.append((obs, info))
            continue

        total_no_bg += info["no_bg"]
        nF, nM = comp.get(obs, ("", ""))
        for g in ("F", "M"):
            sel = [r for r in rows if r["gender"] == g]
            if not sel:
                continue

            def mean(key):
                vals = [r[key] for r in sel if r[key] is not None]
                return round(float(np.mean(vals)), 3) if vals else ""

            loud_mean = mean("loud")
            n_s2b = sum(1 for r in sel if r["s2b"] is not None)
            out_rows.append({
                "clip_id": f"AMI:{obs}", "corpus": "AMI", "meeting": obs,
                "series": obs[:6], "gender": g,
                "n_people_F": nF, "n_people_M": nM,
                "n_segments": len(sel), "n_with_background": n_s2b,
                "s2b_coverage": round(n_s2b / len(sel), 3) if sel else "",
                "duration_min": round(info["dur_min"], 1),
                "dialogue_ref_lufs": round(info["ref"], 2),
                # contribution arm
                "mean_voice_loudness_rel": round(loud_mean - info["ref"], 3) if loud_mean != "" else "",
                "speech_to_background": mean("s2b"),
                "dynamic_range": mean("dyn"),
                # diagnostics
                "mean_voice_loudness_raw": loud_mean,
                "local_background_lufs": mean("bg"),
                # baseline arm
                "f0_std": mean("f0_std"),
                "f0_semitone_std": mean("f0_semi"),
                "f0_median": mean("f0_med"),
                "octave_jump_rate": mean("octv"),
                "energy_std": mean("energy_std"),
                "amplitude_std": mean("amplitude_std"),
                "status": "valid", "reason": "", "feature_version": FEATURE_VERSION,
            })
        print(f"ok  ({info['counts']['F']}F {info['counts']['M']}M, "
              f"{info['dur_min']:.0f} min"
              f"{', ' + str(info['no_bg']) + ' no-bg' if info['no_bg'] else ''})")

    for obs, reason in skipped:
        out_rows.append({"clip_id": f"AMI:{obs}", "corpus": "AMI", "meeting": obs,
                         "series": obs[:6], "gender": "", "status": "NA",
                         "reason": str(reason)[:120], "feature_version": FEATURE_VERSION})

    cols = ["clip_id", "corpus", "meeting", "series", "gender",
            "n_people_F", "n_people_M", "n_segments", "n_with_background",
            "s2b_coverage", "duration_min",
            "dialogue_ref_lufs", "mean_voice_loudness_rel", "speech_to_background",
            "dynamic_range", "mean_voice_loudness_raw", "local_background_lufs",
            "f0_std", "f0_semitone_std", "f0_median", "octave_jump_rate",
            "energy_std", "amplitude_std",
            "status", "reason", "feature_version"]
    out_path = os.path.join(a.out_dir, "c4_batch_ami_v3.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---- summary ----------------------------------------------------
    valid = [r for r in out_rows if r["status"] == "valid"]
    meets = sorted(set(r["meeting"] for r in valid))
    print(f"\n  scored {len(meets)} meetings, {len(skipped)} NA, "
          f"{total_no_bg} turns with no usable background\n")

    hdr = (f"  {'meeting':<10}{'F:M':>6}{'  loud diff':>12}{'  s2b diff':>11}"
           f"{'  f0 diff':>10}{'  semi diff':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    loud_d, s2b_d, f0_d, semi_d = [], [], [], []
    for m in meets:
        f = next((r for r in valid if r["meeting"] == m and r["gender"] == "F"), None)
        mm = next((r for r in valid if r["meeting"] == m and r["gender"] == "M"), None)
        if not f or not mm:
            continue

        def d(key):
            if f[key] == "" or mm[key] == "":
                return None
            return float(mm[key]) - float(f[key])

        dl, ds = d("mean_voice_loudness_rel"), d("speech_to_background")
        df0, dsemi = d("f0_std"), d("f0_semitone_std")
        if dl is not None:
            loud_d.append(dl)
        if ds is not None:
            s2b_d.append(ds)
        if df0 is not None:
            f0_d.append(df0)
        if dsemi is not None:
            semi_d.append(dsemi)
        def cell(v, width, prec):
            return f"{v:>+{width}.{prec}f}" if v is not None else " " * width

        print(f"  {m:<10}{f['n_people_F']}:{mm['n_people_M']:<4}"
              + cell(dl, 12, 2) + cell(ds, 11, 2)
              + cell(df0, 10, 1) + cell(dsemi, 12, 3))

    def report(name, arr, unit):
        if not arr:
            return
        a2 = np.array(arr)
        print(f"  {name:<28} mean {a2.mean():+7.2f} {unit}   SD {a2.std(ddof=1):6.2f}   "
              f"M-higher in {int((a2 > 0).sum())}/{len(a2)}")

    # Is the missing-background problem gendered? If one gender loses more
    # turns than the other, s2b is measured on different samples per gender
    # and the comparison is not like-for-like.
    cov = {}
    for g in ("F", "M"):
        rows_g = [r for r in valid if r["gender"] == g]
        got = sum(int(r["n_with_background"]) for r in rows_g)
        tot = sum(int(r["n_segments"]) for r in rows_g)
        cov[g] = (got, tot, got / tot if tot else 0.0)
    print()
    print(f"  background coverage   F {cov['F'][0]}/{cov['F'][1]} "
          f"({cov['F'][2]:.1%})   M {cov['M'][0]}/{cov['M'][1]} ({cov['M'][2]:.1%})")
    gap = abs(cov['F'][2] - cov['M'][2])
    if gap > 0.05:
        print(f"  WARNING: coverage differs by {gap:.1%} between genders. s2b is being")
        print("  measured on systematically different samples - report this.")
    else:
        print("  Coverage is comparable between genders.")

    # Median pitch and octave rate per gender: the two numbers that say
    # whether the tracker behaved. Female speech is roughly 165-255 Hz and
    # male 85-180 Hz; a female median near 110 means octave halving.
    print()
    for g in ("F", "M"):
        rows_g = [r for r in valid if r["gender"] == g and r["f0_median"] != ""]
        if not rows_g:
            continue
        med = float(np.mean([float(r["f0_median"]) for r in rows_g]))
        oc = [float(r["octave_jump_rate"]) for r in rows_g if r["octave_jump_rate"] != ""]
        ocm = float(np.mean(oc)) if oc else float("nan")
        print(f"  {g}: median f0 {med:6.1f} Hz   octave-jump rate {ocm:.2%}")
    print("  (expected: F 165-255 Hz, M 85-180 Hz; octave rate well under 1%)")

    print()
    report("loudness (referenced)", loud_d, "LU")
    report("speech-to-background", s2b_d, "LU")
    report("f0_std (hertz)", f0_d, "Hz")
    report("f0_std (semitones)", semi_d, "st")

    # The number that says whether the v2 fix changed anything
    if len(loud_d) == len(s2b_d) and len(loud_d) > 2:
        r = float(np.corrcoef(loud_d, s2b_d)[0, 1])
        print(f"\n  correlation(loudness diff, s2b diff) = {r:.4f}")
        if r > 0.995:
            print("  Still essentially identical. On AMI that is expected: one room,")
            print("  one noise source. The per-speaker estimate has something to")
            print("  separate only where the background itself differs per speaker.")
        else:
            print("  The two features have decoupled - they are no longer measuring")
            print("  the same quantity.")

    print(f"\n  Wrote {out_path}")
    print()
    print("  AMI is unmixed. Treat these as room-acoustic and gain readings,")
    print("  not mixing decisions.")


if __name__ == "__main__":
    main()
