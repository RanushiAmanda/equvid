"""
C4 — run the coverage pilot across every AMI meeting whose audio is on disk.

WHY A BATCH RUN
---------------
c4_pilot_ami.py measures one meeting. Ten segments from one meeting proves the
pipeline runs, but it cannot show whether anything it measures is stable. This
script runs the same measurement over every meeting available and writes one
table, so a pattern (or the absence of one) is visible.

It is still a pilot. The per-meeting sample is small, AMI is unmixed, and
nothing here is a finding about gender bias. What it produces is evidence that
the extraction works at scale and a first look at the spread.

OUTPUT
------
    output/c4_batch_ami.csv      one row per meeting per gender
    printed summary              composition alongside measurement

USAGE
-----
    python c4_batch_ami.py
    python c4_batch_ami.py --per-gender 15
    python c4_batch_ami.py --no-pitch        # much faster; skips f0_std
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

FEATURE_VERSION = "c4-batch-0.1"
MIN_SEG_S = 1.0
MIN_GAP_S = 0.5
NS = "{http://nite.sourceforge.net/}"
ROLE_SUFFIX = re.compile(r"(PM|ME|UID|UI|ID)$")


def find_ami_dir(start):
    """Walk up from the script looking for the AMI corpus folder."""
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
    """{meeting: {seat_letter: person_code}} for every meeting."""
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
    """All speaker turns for one meeting: (start, end, seat), time-ordered."""
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


def measure_meeting(obs, wav, ann_zip, gender_of, seat_maps, per_gender, do_pitch, np, librosa, pyln):
    """Returns (rows, info) or (None, reason)."""
    seats = seat_maps.get(obs, {})
    if not seats:
        return None, "no seat map"
    segs = load_segments(ann_zip, obs)
    if not segs:
        return None, "no segments XML"

    y, sr = librosa.load(wav, sr=None, mono=True)
    dur = len(y) / sr
    meter = pyln.Meter(sr)

    # --- noise floor from inter-turn silence -------------------------
    floors, cursor = [], 0.0
    for st, en, _ in segs:
        if st - cursor >= MIN_GAP_S:
            blk = y[int(cursor * sr):int(st * sr)].astype(float)
            if len(blk) >= int(0.4 * sr):
                try:
                    v = meter.integrated_loudness(blk)
                    if np.isfinite(v):
                        floors.append(v)
                except Exception:
                    pass
        cursor = max(cursor, en)
    if not floors:
        return None, "no measurable silence"
    noise_floor = float(np.median(floors))

    # --- choose segments, balanced by gender -------------------------
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

    # --- measure -----------------------------------------------------
    rows = []
    for st, en, person, g in chosen:
        seg = y[int(st * sr):int(en * sr)].astype(float)
        try:
            loud = meter.integrated_loudness(seg)
            if not np.isfinite(loud):
                continue
        except Exception:
            continue

        rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256)[0]

        f0_std = ""
        f0_semitone_std = ""
        if do_pitch:
            f0 = librosa.yin(seg, fmin=60, fmax=400, sr=sr, frame_length=1024)
            v = f0[np.isfinite(f0) & (f0 > 60) & (f0 < 400)]
            if len(v) > 5:
                f0_std = round(float(np.std(v)), 3)
                # Scale-free alternative: variation in semitones does not grow
                # with mean pitch the way a Hz standard deviation does.
                f0_semitone_std = round(float(np.std(12 * np.log2(v))), 3)

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
        dyn = (max(st_loud) - min(st_loud)) if len(st_loud) >= 2 else float("nan")

        rows.append({"gender": g, "person": person,
                     "loud": float(loud),
                     "s2b": float(loud - noise_floor),
                     "dyn": float(dyn) if np.isfinite(dyn) else None,
                     "f0_std": f0_std, "f0_semitone_std": f0_semitone_std,
                     "energy_std": float(np.std(rms)),
                     "amplitude_std": float(np.std(np.abs(seg)))})

    return rows, {"noise_floor": noise_floor, "dur_min": dur / 60,
                  "n_seg": len(segs), "counts": counts, "sr": sr}


def main():
    np = need("numpy")
    librosa = need("librosa")
    pyln = need("pyloudnorm")

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Batch C4 pilot across AMI meetings on disk.")
    ap.add_argument("--ami-dir", default=find_ami_dir(here))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "output")))
    ap.add_argument("--per-gender", type=int, default=12)
    ap.add_argument("--no-pitch", action="store_true", help="skip f0 (much faster)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N meetings (for a quick test)")
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

    # Every meeting whose Mix-Headset audio is actually on disk
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
    print(f"up to {a.per_gender} segments per gender each"
          f"{'  (pitch skipped)' if a.no_pitch else ''}\n")

    out_rows, skipped = [], []
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

        nF, nM = comp.get(obs, ("", ""))
        for g in ("F", "M"):
            sel = [r for r in rows if r["gender"] == g]
            if not sel:
                continue

            def mean(key):
                vals = [r[key] for r in sel if r[key] not in ("", None)]
                return round(float(np.mean([float(v) for v in vals])), 3) if vals else ""

            out_rows.append({
                "clip_id": f"AMI:{obs}", "corpus": "AMI", "meeting": obs,
                "series": obs[:6], "gender": g,
                "n_people_F": nF, "n_people_M": nM,
                "n_segments": len(sel),
                "duration_min": round(info["dur_min"], 1),
                "noise_floor_lufs": round(info["noise_floor"], 2),
                "mean_voice_loudness": mean("loud"),
                "speech_to_background": mean("s2b"),
                "dynamic_range": mean("dyn"),
                "f0_std": mean("f0_std"),
                "f0_semitone_std": mean("f0_semitone_std"),
                "energy_std": mean("energy_std"),
                "amplitude_std": mean("amplitude_std"),
                "status": "valid", "reason": "", "feature_version": FEATURE_VERSION,
            })
        print(f"ok  ({info['counts']['F']}F {info['counts']['M']}M, {info['dur_min']:.0f} min)")

    # NA rows for meetings that could not be scored - M2's output contract
    for obs, reason in skipped:
        out_rows.append({"clip_id": f"AMI:{obs}", "corpus": "AMI", "meeting": obs,
                         "series": obs[:6], "gender": "", "status": "NA",
                         "reason": str(reason)[:120], "feature_version": FEATURE_VERSION})

    cols = ["clip_id", "corpus", "meeting", "series", "gender",
            "n_people_F", "n_people_M", "n_segments", "duration_min",
            "noise_floor_lufs", "mean_voice_loudness", "speech_to_background",
            "dynamic_range", "f0_std", "f0_semitone_std", "energy_std",
            "amplitude_std", "status", "reason", "feature_version"]
    out_path = os.path.join(a.out_dir, "c4_batch_ami.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---- summary ----------------------------------------------------
    valid = [r for r in out_rows if r["status"] == "valid"]
    print(f"\n  scored {len(set(r['meeting'] for r in valid))} meetings, "
          f"{len(skipped)} NA\n")

    hdr = (f"  {'meeting':<10}{'F:M':>6}  {'loud F':>9}{'loud M':>9}{'  diff':>8}"
           f"{'   s2b F':>9}{'s2b M':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    diffs = []
    for m in sorted(set(r["meeting"] for r in valid)):
        f = next((r for r in valid if r["meeting"] == m and r["gender"] == "F"), None)
        mm = next((r for r in valid if r["meeting"] == m and r["gender"] == "M"), None)
        if not f or not mm:
            continue
        d = mm["mean_voice_loudness"] - f["mean_voice_loudness"]
        diffs.append(d)
        print(f"  {m:<10}{f['n_people_F']}:{mm['n_people_M']:<4}"
              f"{f['mean_voice_loudness']:>9.1f}{mm['mean_voice_loudness']:>9.1f}"
              f"{d:>+8.1f}{f['speech_to_background']:>9.1f}{mm['speech_to_background']:>9.1f}")

    if diffs:
        arr = np.array(diffs)
        print(f"\n  male-minus-female loudness: mean {arr.mean():+.2f} LU, "
              f"SD {arr.std(ddof=1):.2f}, range {arr.min():+.1f} to {arr.max():+.1f}")
        print(f"  meetings where men measured louder: {int((arr > 0).sum())} of {len(arr)}")

    print(f"\n  Wrote {out_path}")
    print()
    print("  AMI is unmixed. These are room-acoustic and microphone-gain readings,")
    print("  not mixing decisions. Mix-Headset sums four close-talk mics, so a")
    print("  per-speaker loudness difference may be channel gain rather than voice.")
    print("  Treat every number here as pipeline evidence, not as a finding.")


if __name__ == "__main__":
    main()
