"""
C4 — coverage pilot on AMI (Gate 0 item G0.7).

WHAT THIS PROVES
----------------
That all six C4 features can be computed from real corpus audio, end to end,
on a corpus I can obtain without waiting for anyone. It is deliberately small:
one meeting, a handful of segments per gender. The point is that the pipeline
runs before the full extraction starts, and that the failure modes are known.

    Contribution arm   mean_voice_loudness_rel, speech_to_background_ratio,
                       dynamic_range
    Baseline arm       f0_std, energy_std, amplitude_std   (EuqAud's, verbatim)

WHAT speech_to_background_ratio MEANS ON AMI — READ THIS
--------------------------------------------------------
AMI is an UNMIXED corpus. Nobody chose these levels; there is no composer, no
effects track and no re-recording mixer. So on AMI this feature is NOT measuring
a mixing decision. It measures the gap between a speaker's level and the room's
noise floor between turns — room acoustics and microphone placement.

That makes AMI a structural control, not a demonstration: if the feature is
measuring authored mixing, it should behave differently here than on mixed
content. Reporting the number without this caveat would be the single easiest
thing for an examiner to attack.

INPUTS (all obtainable alone, no teammates needed)
--------------------------------------------------
    AMI/ami_public_manual_1.6.2.zip            <- download_ami.py fetches it
    AMI/amicorpus/<obs>/audio/<obs>.Mix-Headset.wav
    output/gender_map_ami.csv                  <- build_group_tables.py

MEASUREMENT NOTE TO DECLARE
---------------------------
Mix-Headset is 16 kHz mono. ITU-R BS.1770 is specified for broadcast sample
rates; pyloudnorm computes K-weighting coefficients for whatever rate it is
given, so the numbers are internally consistent and comparable across AMI
meetings, but they are not directly comparable to a 48 kHz film measurement.
Say so rather than letting someone else notice.

USAGE
-----
    python c4_pilot_ami.py --meeting ES2002a
    python c4_pilot_ami.py --meeting ES2002a --per-gender 10
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

FEATURE_VERSION = "c4-pilot-0.1"
MIN_SEG_S = 1.0        # BS.1770 gating needs >= 400 ms; 1 s is a safe floor
MIN_GAP_S = 0.5        # inter-turn silence long enough to read a noise floor
NS = "{http://nite.sourceforge.net/}"
ROLE_SUFFIX = re.compile(r"(PM|ME|UID|UI|ID)$")


def find_ami_dir(start):
    """Walk up from the script looking for the AMI corpus folder.

    The repo layout puts corpora BESIDE the repo, not inside it:

        Research Project/
        |-- equvid/                        <- the git repo
        |   |-- data/download_ami.py
        |   `-- components/C4_Audio/src/   <- this script
        |-- AMI/                           <- what we are looking for
        |-- MELD/
        `-- AVA/

    Searching upward means the script works wherever you put it, instead of
    depending on a hard-coded number of '..' steps.
    """
    d = os.path.abspath(start)
    for _ in range(8):
        cand = os.path.join(d, "AMI")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # Not found: return the conventional location so the error message names a
    # real path rather than something relative and confusing.
    return os.path.normpath(os.path.join(start, "..", "..", "..", "..", "AMI"))


def need(mod):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"Missing package '{mod}'.\n"
                 f"    pip install librosa pyloudnorm soundfile numpy pandas")


def speaker_letter_to_person(ann_zip, obs):
    """Seat letter (A-D) -> global participant code, for THIS meeting only.

    The letters are seats, not people: the same letter is a different person in
    a different meeting. This is why person_id uses global_name everywhere else.
    """
    with zipfile.ZipFile(ann_zip) as z:
        meets = ET.fromstring(z.read("corpusResources/meetings.xml"))
    for m in meets.iter("meeting"):
        if m.get("observation") != obs:
            continue
        out = {}
        for s in m.iter("speaker"):
            seat = s.get("nxt_agent") or s.get("agent") or s.get("channel")
            gname = s.get("global_name")
            if seat and gname:
                out[seat] = ROLE_SUFFIX.sub("", gname)
        return out
    sys.exit(f"Meeting {obs} not found in meetings.xml")


def load_segments(ann_zip, obs):
    """All speaker turns for a meeting: (start, end, seat_letter), time-ordered."""
    segs = []
    with zipfile.ZipFile(ann_zip) as z:
        names = [n for n in z.namelist()
                 if re.search(rf"segments/{re.escape(obs)}\.[A-Z]\.segments\.xml$", n)]
        if not names:
            sys.exit(f"No segments XML for {obs} inside the annotation zip.\n"
                     f"Checked for: segments/{obs}.<LETTER>.segments.xml")
        for name in names:
            letter = re.search(rf"{re.escape(obs)}\.([A-Z])\.segments\.xml$", name).group(1)
            root = ET.fromstring(z.read(name))
            for seg in root.iter("segment"):
                st, en = seg.get("transcriber_start"), seg.get("transcriber_end")
                if st is None or en is None:
                    continue
                segs.append((float(st), float(en), letter))
    segs.sort()
    return segs


def gaps_between(segs, audio_dur):
    """Silence windows where nobody is annotated as speaking."""
    out, cursor = [], 0.0
    for st, en, _ in segs:
        if st - cursor >= MIN_GAP_S:
            out.append((cursor, st))
        cursor = max(cursor, en)
    if audio_dur - cursor >= MIN_GAP_S:
        out.append((cursor, audio_dur))
    return out


def main():
    np = need("numpy")
    librosa = need("librosa")
    pyln = need("pyloudnorm")

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="C4 coverage pilot on one AMI meeting.")
    ap.add_argument("--meeting", default="ES2002a")
    ap.add_argument("--ami-dir", default=find_ami_dir(here))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "output")))
    ap.add_argument("--per-gender", type=int, default=5,
                    help="segments to measure per gender (default 5)")
    a = ap.parse_args()

    obs = a.meeting
    ann_zip = os.path.join(a.ami_dir, "ami_public_manual_1.6.2.zip")
    wav = os.path.join(a.ami_dir, "amicorpus", obs, "audio", f"{obs}.Mix-Headset.wav")
    gm_path = os.path.join(a.out_dir, "gender_map_ami.csv")
    os.makedirs(a.out_dir, exist_ok=True)

    for p, how in [(ann_zip, "python download_ami.py --list"),
                   (wav, f"python download_ami.py --audio-only --series {obs[:6]}"),
                   (gm_path, "python build_group_tables.py")]:
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}\n  Produce it with:  {how}")

    gender_of = {}
    with open(gm_path) as fh:
        for row in csv.DictReader(fh):
            gender_of[row["person_id"].split(":", 1)[1]] = row["gender"]

    seat_to_person = speaker_letter_to_person(ann_zip, obs)
    segs = load_segments(ann_zip, obs)

    print(f"Meeting {obs}")
    print(f"  seats        {seat_to_person}")
    print(f"  segments     {len(segs)} annotated turns")

    y, sr = librosa.load(wav, sr=None, mono=True)
    dur = len(y) / sr
    print(f"  audio        {dur/60:.1f} min at {sr} Hz")
    if sr < 44100:
        print(f"  NOTE         {sr} Hz is below broadcast rate; BS.1770 numbers are")
        print( "               internally consistent but not comparable to 48 kHz film")

    meter = pyln.Meter(sr)

    # ---- noise floor from inter-turn silence ---------------------------
    floors = []
    for st, en in gaps_between(segs, dur):
        seg = y[int(st * sr):int(en * sr)]
        if len(seg) < int(0.4 * sr):
            continue
        try:
            lo = meter.integrated_loudness(seg.astype(float))
            if np.isfinite(lo):
                floors.append(lo)
        except Exception:
            pass
    noise_floor = float(np.median(floors)) if floors else float("nan")
    print(f"  noise floor  {noise_floor:.1f} LUFS  (median of {len(floors)} silent windows)")

    # ---- pick segments, balanced by gender -----------------------------
    usable = []
    for st, en, seat in segs:
        if en - st < MIN_SEG_S:
            continue
        person = seat_to_person.get(seat)
        g = gender_of.get(person, "")
        if g in ("F", "M"):
            usable.append((st, en, seat, person, g))
    usable.sort(key=lambda r: r[1] - r[0], reverse=True)     # longest first

    chosen, counts = [], {"F": 0, "M": 0}
    for row in usable:
        g = row[4]
        if counts[g] < a.per_gender:
            chosen.append(row)
            counts[g] += 1
    chosen.sort()

    if not chosen:
        sys.exit("No usable segments. Check the gender map covers this meeting's speakers.")

    # ---- measure -------------------------------------------------------
    rows = []
    skipped = []
    for st, en, seat, person, g in chosen:
        seg = y[int(st * sr):int(en * sr)].astype(float)
        try:
            loud = meter.integrated_loudness(seg)
            if not np.isfinite(loud):
                raise ValueError("non-finite loudness")
        except Exception as e:
            skipped.append((f"{st:.1f}-{en:.1f}", str(e)))
            continue

        rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256)[0]
        f0 = librosa.yin(seg, fmin=60, fmax=400, sr=sr, frame_length=1024)
        f0v = f0[np.isfinite(f0) & (f0 > 60) & (f0 < 400)]

        # Short-term loudness spread within the turn -> delivered dynamic range
        win = int(0.4 * sr)
        st_loud = []
        for i in range(0, max(1, len(seg) - win), win // 2):
            blk = seg[i:i + win]
            if len(blk) < win:
                break
            try:
                v = meter.integrated_loudness(blk)
                if np.isfinite(v):
                    st_loud.append(v)
            except Exception:
                pass
        dyn = (max(st_loud) - min(st_loud)) if len(st_loud) >= 2 else float("nan")

        rows.append({
            "clip_id": f"AMI:{obs}",
            "person_id": f"AMI:{person}",
            "gender": g,
            "start_s": round(st, 2),
            "end_s": round(en, 2),
            "dur_s": round(en - st, 2),
            # contribution arm
            "loudness_lufs": round(float(loud), 2),
            "speech_to_background_lu": round(float(loud - noise_floor), 2),
            "dynamic_range_lu": round(float(dyn), 2) if np.isfinite(dyn) else "",
            # baseline arm (EuqAud's three, transferred verbatim)
            "f0_std": round(float(np.std(f0v)), 2) if len(f0v) > 5 else "",
            "energy_std": round(float(np.std(rms)), 5),
            "amplitude_std": round(float(np.std(np.abs(seg))), 5),
        })

    seg_path = os.path.join(a.out_dir, f"c4_pilot_segments_{obs}.csv")
    with open(seg_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate to the shared output contract -----------------------
    # M2's contract: one row per clip_id, with status / reason / feature_version.
    # A clip that cannot be scored gets NA and a reason, never a silent drop.
    def agg(g, key):
        vals = [r[key] for r in rows if r["gender"] == g and r[key] != ""]
        return round(float(np.mean(vals)), 3) if vals else ""

    ref = float(np.mean([r["loudness_lufs"] for r in rows]))   # shard dialogue reference
    clip = {"clip_id": f"AMI:{obs}", "corpus": "AMI",
            "status": "valid", "reason": "", "feature_version": FEATURE_VERSION,
            "n_seg_F": counts["F"], "n_seg_M": counts["M"],
            "dialogue_ref_lufs": round(ref, 2)}
    for g in ("F", "M"):
        clip[f"mean_voice_loudness_rel_{g}"] = round(agg(g, "loudness_lufs") - ref, 3) \
            if agg(g, "loudness_lufs") != "" else ""
        clip[f"speech_to_background_{g}"] = agg(g, "speech_to_background_lu")
        clip[f"dynamic_range_{g}"] = agg(g, "dynamic_range_lu")
        clip[f"f0_std_{g}"] = agg(g, "f0_std")
        clip[f"energy_std_{g}"] = agg(g, "energy_std")
        clip[f"amplitude_std_{g}"] = agg(g, "amplitude_std")

    clip_path = os.path.join(a.out_dir, f"c4_pilot_clip_{obs}.csv")
    with open(clip_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(clip.keys()))
        w.writeheader()
        w.writerow(clip)

    # ---- report --------------------------------------------------------
    print(f"\n  measured     {len(rows)} segments ({counts['F']} F, {counts['M']} M)")
    if skipped:
        print(f"  skipped      {len(skipped)}: {skipped[:3]}")
    print()
    hdr = f"  {'gender':<7}{'n':>3}  {'loud':>8}{'s2b':>8}{'dyn':>8}{'f0_std':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for g in ("F", "M"):
        n = counts[g]
        print(f"  {g:<7}{n:>3}  {agg(g,'loudness_lufs'):>8}{agg(g,'speech_to_background_lu'):>8}"
              f"{agg(g,'dynamic_range_lu'):>8}{agg(g,'f0_std'):>9}")
    print()
    print(f"  Wrote {seg_path}")
    print(f"  Wrote {clip_path}")
    print()
    print("  REMINDER: AMI is unmixed. speech_to_background here is room acoustics,")
    print("  not a mixing decision. Do not present it as evidence of authored bias.")
    print("  n is tiny. These numbers demonstrate the pipeline, not a finding.")


if __name__ == "__main__":
    main()
