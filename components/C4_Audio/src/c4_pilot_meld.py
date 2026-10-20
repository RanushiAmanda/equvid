"""
C4 - MELD pilot: can the mix features be measured on 3.6-second clips?

WHY A PILOT, AND NOT STRAIGHT TO A BATCH RUN
--------------------------------------------
AMI worked because meetings are 30 minutes long. Background level was
estimated from the gaps where nobody was speaking, and there were thousands of
them. MELD is the opposite: each .mp4 is ONE utterance, averaging 3.59 s. Three
things could therefore break, and all three must be checked before extracting
anything:

  1. DECODING. MELD ships .mp4, not .wav. soundfile cannot read it, so ffmpeg
     has to decode first. Some MELD clips are also known to be truncated.

  2. GATED LOUDNESS ON SHORT AUDIO. ITU-R BS.1770 measures loudness in 400 ms
     blocks and then GATES - it discards blocks below an absolute floor and
     below a relative threshold. On a 3 s clip that can discard almost
     everything and return -inf. If it does, mean_voice_loudness is not
     computable per clip and must be aggregated differently.

  3. NO SILENCE TO MEASURE BACKGROUND IN. This is the real risk. On AMI,
     background = room noise, present during gaps and roughly constant. In a
     sitcom mix the laugh track and music sit UNDERNEATH the dialogue and are
     LOUDEST in the gaps - right after a punchline. So "the quiet part of the
     clip" may be laughter, not background, and sampling it would measure
     comic timing rather than mix placement.

This script does not extract features. It measures how bad each of those three
problems is, so the choice of method is made on evidence.

A FOURTH QUESTION, WHICH IS GOOD NEWS IF IT HOLDS
-------------------------------------------------
On AMI, absolute loudness was meaningless across meetings - Edinburgh and
Idiap used different recording gain, which is why loudness had to be
referenced within each meeting. MELD clips are cut from broadcast masters,
which ARE loudness-normalised at programme level. If absolute LUFS is stable
within an episode, mean_voice_loudness becomes directly comparable across
clips, which is strictly better than what AMI allowed. The summary reports
within-episode and between-episode spread so this can be checked rather than
assumed.

A MEASUREMENT CONFOUND TO BE HONEST ABOUT
-----------------------------------------
BS.1770 loudness is K-weighted: it high-passes below about 100 Hz and lifts
everything above 1.5 kHz. Male and female voices have different spectra, so
two voices at identical physical level can report different LUFS. That is a
property of the measure, not of the mix, and an examiner will ask about it.

The script therefore computes an unweighted RMS alongside the K-weighted
loudness for every clip. If a gender gap appears in LUFS but not in RMS, the
gap is spectral, not editorial. Having both columns from the start makes that
robustness check free later.

USAGE
-----
    python c4_pilot_meld.py --list-speakers      <- run this FIRST, no audio
    python c4_pilot_meld.py
    python c4_pilot_meld.py --split dev --n-clips 15
"""

import argparse
import csv
import io
import os
import random
import shutil
import subprocess
import sys

# --- constants that must match the AMI extraction -------------------------
# If c4_batch_ami4.py uses different values, change these to match or the two
# corpora are not measured on the same scale and cannot be compared.
BLOCK_S = 0.400        # BS.1770 block length
HOP_S = 0.100          # step between blocks (75% overlap)
TARGET_SR = 48000      # MELD is broadcast audio; do not downsample to 16 k
FEATURE_VERSION = "c4-meld-pilot-0.1"

# MELD main cast. These six are the only speakers with a gender label we can
# assign without manual annotation, and they are unambiguous.
GENDER_OF = {
    "monica": "F", "phoebe": "F", "rachel": "F",
    "chandler": "M", "joey": "M", "ross": "M",
}

# Per split: the folder holding the .mp4 files, relative to the MELD root.
SPLIT_VIDEO_DIR = {
    "train": os.path.join("train", "train_splits"),
    "dev": os.path.join("dev", "dev_splits_complete"),
    "test": os.path.join("test", "output_repeated_splits_test"),
}


# ==========================================================================
#  Locating things
# ==========================================================================

def find_corpus_dir(start, name):
    """Walk upwards looking for a sibling folder called `name`.

    The corpora live BESIDE the repo, not inside it - licences forbid
    redistribution and .gitignore blocks these folder names. So from
    equvid/components/C4_Audio/src/ we need to climb four levels to reach
    'Research Project/' and find 'MELD/' there. Searching upwards rather than
    counting '..' levels is what fixed the AMI path bug.
    """
    d = os.path.abspath(start)
    for _ in range(8):
        cand = os.path.join(d, name)
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.normpath(os.path.join(start, "..", "..", "..", "..", name))


def find_meld_root(meld_dir):
    """Find the directory that directly contains train/, dev/ and test/.

    The tarballs extract to MELD.Raw/MELD.Raw/ - the nesting is an artefact of
    how the archive was built, and depends on how it was unpacked. Rather than
    hardcode two levels, look for the folder that actually has the splits in
    it.
    """
    for root, dirs, _files in os.walk(meld_dir):
        if os.path.isdir(os.path.join(root, "train", "train_splits")):
            return root
        # stop descending after a few levels - no need to walk 10k video files
        depth = root.count(os.sep) - meld_dir.count(os.sep)
        if depth >= 3:
            dirs[:] = []
    return None


def find_split_csv(root, split):
    """The annotation CSVs are not all in the same place.

    train_sent_emo.csv sits inside train/, but dev_sent_emo.csv and
    test_sent_emo.csv sit at the MELD root. Try both.
    """
    name = f"{split}_sent_emo.csv"
    for cand in (os.path.join(root, name),
                 os.path.join(root, split, name)):
        if os.path.exists(cand):
            return cand
    return None


def find_ffmpeg():
    """ffmpeg is needed to decode .mp4 audio; soundfile cannot.

    Preference order: the binary pip already installed for C3 (imageio-ffmpeg
    bundles one, so no system install is needed), then anything on PATH.
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    return None


# ==========================================================================
#  Audio
# ==========================================================================

def decode_mono(ffmpeg, path, sr=TARGET_SR):
    """Decode an .mp4's audio track to a mono float array via ffmpeg.

    Returns (samples, sr) or raises RuntimeError with ffmpeg's own message.

    Mono by averaging, not summing: dialogue in a TV mix is centre-panned, so
    both channels carry nearly the same signal. Averaging leaves the level
    unchanged; summing would add 6 dB and make the numbers incomparable with
    AMI's single-channel audio.

    Piped to stdout rather than a temp file - 10k clips of temp .wav would be
    tens of gigabytes of needless disk writes.
    """
    import numpy as np
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", path,
           "-map", "a:0",          # first audio stream only
           "-ac", "1",             # downmix to mono (ffmpeg averages)
           "-ar", str(sr),
           "-f", "f32le", "-"]     # raw float32, no WAV header to parse
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        msg = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(msg[-1] if msg else "ffmpeg produced no audio")
    y = np.frombuffer(p.stdout, dtype="<f4").astype(float)
    return y, sr


def block_loudness(meter, y, sr, np):
    """Short-term loudness of every BLOCK_S window, stepping by HOP_S.

    pyloudnorm's integrated_loudness applied to a single 400 ms window gives
    that window's K-weighted level. Sliding it gives the loudness contour,
    from which both the speech level and the floor are read.

    Windows overlap by 75%, so these values are not independent samples. That
    is fine for percentiles and for a spread statistic, but it means the count
    of blocks is not a sample size and must never be reported as one.
    """
    n = int(round(BLOCK_S * sr))
    hop = int(round(HOP_S * sr))
    out = []
    if len(y) < n:
        return out
    for i in range(0, len(y) - n + 1, hop):
        w = y[i:i + n]
        if not np.any(w):
            continue                      # digital silence -> -inf, skip
        try:
            L = meter.integrated_loudness(w)
        except Exception:
            continue
        if np.isfinite(L):
            out.append(float(L))
    return out


def rms_dbfs(y, np):
    """Unweighted RMS in dBFS - the robustness check against K-weighting.

    No frequency weighting at all, so a gender gap that appears here as well
    as in LUFS cannot be blamed on the filter shape.
    """
    if len(y) == 0:
        return None
    r = float(np.sqrt(np.mean(np.square(y))))
    if r <= 0:
        return None
    return float(20.0 * np.log10(r))


# ==========================================================================
#  Annotation
# ==========================================================================

def read_utterances(csv_path):
    """Read a MELD *_sent_emo.csv into dicts.

    utf-8-sig because the files carry a BOM, which otherwise corrupts the
    first column name and makes the lookup silently fail.
    """
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "speaker_raw": (r.get("Speaker") or "").strip(),
                    "dialogue_id": int(r["Dialogue_ID"]),
                    "utterance_id": int(r["Utterance_ID"]),
                    "season": int(r["Season"]),
                    "episode": int(r["Episode"]),
                    "emotion": (r.get("Emotion") or "").strip(),
                    "start": (r.get("StartTime") or "").strip(),
                    "end": (r.get("EndTime") or "").strip(),
                })
            except (KeyError, ValueError, TypeError):
                continue          # malformed row; MELD has a handful
    return rows


def speaker_key(raw):
    """Normalise the Speaker field to a lowercase single token.

    MELD's Speaker column is free text written by annotators, so the same
    person appears as 'Chandler', 'chandler', and occasionally with trailing
    punctuation. Lowercasing and taking the first word catches the main cast
    without a hand-built alias table.
    """
    s = raw.strip().strip('"').lower()
    if not s:
        return ""
    return s.split()[0].strip(".,:;!?")


# ==========================================================================
#  Modes
# ==========================================================================

def list_speakers(rows):
    """CSV-only report: how much of the split do the six main cast cover?

    This matters because restricting to six speakers is a selection decision
    that has to be declared. If the six cover 70% of utterances the
    restriction is cheap; if they cover 25% it is a serious limitation and
    more speakers need gender labels.
    """
    counts = {}
    for r in rows:
        counts[speaker_key(r["speaker_raw"])] = counts.get(speaker_key(r["speaker_raw"]), 0) + 1
    total = sum(counts.values())

    print(f"\n  {len(counts)} distinct speaker strings, {total} utterances\n")
    print(f"  {'speaker':<22}{'utterances':>12}{'share':>9}{'gender':>9}")
    print("  " + "-" * 50)
    for name, c in sorted(counts.items(), key=lambda kv: -kv[1])[:20]:
        g = GENDER_OF.get(name, "-")
        print(f"  {name:<22}{c:>12}{c / total:>8.1%}{g:>9}")

    main = {g: 0 for g in ("F", "M")}
    for name, c in counts.items():
        g = GENDER_OF.get(name)
        if g:
            main[g] += c
    covered = main["F"] + main["M"]
    print("  " + "-" * 50)
    print(f"  main cast covered   F {main['F']}   M {main['M']}   "
          f"total {covered}  ({covered / total:.1%} of the split)")
    print(f"  utterance count ratio M:F = {main['M'] / main['F']:.3f}"
          if main["F"] else "")
    print()
    print("  Note: this is the UTTERANCE COUNT ratio, not speaking time.")
    print("  The 1.094 figure in the project docs is speaking time from")
    print("  StartTime/EndTime. They are different quantities and should")
    print("  not be quoted interchangeably.")
    print()
    if covered / total < 0.5:
        print("  WARNING: the six main cast cover less than half the split.")
        print("  Restricting to them is a substantial selection effect and")
        print("  must be stated as a limitation, not left implicit.")


def run_pilot(args, root, rows, ffmpeg):
    import numpy as np
    try:
        import pyloudnorm as pyln
    except ImportError:
        sys.exit("Missing pyloudnorm.  pip install pyloudnorm")

    vid_dir = os.path.join(root, SPLIT_VIDEO_DIR[args.split])
    if not os.path.isdir(vid_dir):
        sys.exit(f"Video folder not found: {vid_dir}")

    # --- choose clips -----------------------------------------------------
    # Sampled at random rather than taking the first N, because dia0, dia1...
    # are consecutive scenes from one episode and would share a single mix.
    pool = {"F": [], "M": []}
    for r in rows:
        g = GENDER_OF.get(speaker_key(r["speaker_raw"]))
        if g:
            pool[g].append(r)
    if not pool["F"] or not pool["M"]:
        sys.exit("No main-cast utterances found. Check the Speaker column.")

    rng = random.Random(args.seed)
    picked = []
    for g in ("F", "M"):
        rng.shuffle(pool[g])
        picked += [(g, r) for r in pool[g][:args.n_clips]]

    meter = pyln.Meter(TARGET_SR, block_size=BLOCK_S)

    print(f"\n  split {args.split}   {args.n_clips} clips per gender   "
          f"seed {args.seed}")
    print(f"  videos  {vid_dir}")
    print(f"  ffmpeg  {ffmpeg}\n")
    hdr = (f"  {'file':<20}{'g':>2}{'spk':>10}{'dur':>7}{'blk':>5}"
           f"{'LUFS':>9}{'RMS':>8}{'p90':>8}{'p10':>8}{'s2b':>7}{'quiet':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    out_rows = []
    for g, r in picked:
        stem = f"dia{r['dialogue_id']}_utt{r['utterance_id']}"
        path = os.path.join(vid_dir, stem + ".mp4")
        rec = {
            "clip_id": f"MELD:S{r['season']:02d}E{r['episode']:02d}",
            "utt_id": f"MELD:{stem}",
            "person_id": f"MELD:{speaker_key(r['speaker_raw'])}",
            "gender": g,
            "split": args.split,
            "emotion": r["emotion"],
            "feature_version": FEATURE_VERSION,
            "status": "NA", "reason": "",
            "duration_s": None, "n_blocks": None,
            "integrated_lufs": None, "rms_dbfs": None,
            "p90_lufs": None, "p50_lufs": None, "p10_lufs": None,
            "speech_to_background": None,
            "block_iqr": None, "block_sd": None,
            "quiet_frac_15lu": None, "quiet_frac_10lu": None,
        }

        if not os.path.exists(path):
            rec["reason"] = "video file missing"
            out_rows.append(rec)
            print(f"  {stem:<20}{g:>2}{'':>10}  file missing")
            continue

        try:
            y, sr = decode_mono(ffmpeg, path)
        except RuntimeError as e:
            rec["reason"] = f"decode failed: {e}"
            out_rows.append(rec)
            print(f"  {stem:<20}{g:>2}{'':>10}  decode failed")
            continue

        dur = len(y) / sr
        rec["duration_s"] = round(dur, 3)
        if dur < BLOCK_S * 2:
            rec["reason"] = f"too short ({dur:.2f}s)"
            out_rows.append(rec)
            print(f"  {stem:<20}{g:>2}{'':>10}{dur:>7.2f}  too short")
            continue

        blocks = block_loudness(meter, y, sr, np)
        rec["n_blocks"] = len(blocks)
        rec["rms_dbfs"] = rms_dbfs(y, np)

        try:
            L = meter.integrated_loudness(y)
            rec["integrated_lufs"] = float(L) if np.isfinite(L) else None
        except Exception:
            rec["integrated_lufs"] = None

        if len(blocks) < 4:
            rec["reason"] = f"only {len(blocks)} usable blocks"
            out_rows.append(rec)
            print(f"  {stem:<20}{g:>2}{'':>10}{dur:>7.2f}{len(blocks):>5}  "
                  f"too few blocks")
            continue

        b = np.array(blocks)
        p90, p50, p10 = (float(np.percentile(b, q)) for q in (90, 50, 10))
        rec.update({
            "p90_lufs": p90, "p50_lufs": p50, "p10_lufs": p10,
            # Speech level taken as p90 and floor as p10 - a percentile split
            # rather than a silence detector, because a 3.6 s clip may contain
            # no silence at all. This is an ESTIMATE of mix separation, and
            # the pilot exists partly to judge whether it is a usable one.
            "speech_to_background": p90 - p10,
            "block_iqr": float(np.percentile(b, 75) - np.percentile(b, 25)),
            "block_sd": float(np.std(b, ddof=1)),
            # How much of the clip is far below its own speech level. If this
            # is near zero there is nothing quiet to call background.
            "quiet_frac_15lu": float(np.mean(b < p90 - 15.0)),
            "quiet_frac_10lu": float(np.mean(b < p90 - 10.0)),
            "status": "valid",
        })
        out_rows.append(rec)

        lufs = rec["integrated_lufs"]
        print(f"  {stem:<20}{g:>2}{speaker_key(r['speaker_raw']):>10}"
              f"{dur:>7.2f}{len(blocks):>5}"
              f"{(f'{lufs:.1f}' if lufs is not None else 'gated'):>9}"
              f"{rec['rms_dbfs']:>8.1f}{p90:>8.1f}{p10:>8.1f}"
              f"{p90 - p10:>7.1f}{rec['quiet_frac_15lu']:>7.0%}")

    # --- write ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"c4_pilot_meld_{args.split}.csv")
    cols = list(out_rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r.get(c), float)
                            else r.get(c, "")) for c in cols})

    report(out_rows, np)
    print(f"\n  Wrote {out}")


def report(rows, np):
    """The three feasibility questions, answered from the numbers."""
    ok = [r for r in rows if r["status"] == "valid"]
    print("\n" + "=" * 74)
    print(f"  {len(ok)} of {len(rows)} clips measured\n")

    if not ok:
        print("  Nothing measured. Fix decoding before going further.")
        return

    # --- Q1 decoding ------------------------------------------------------
    fails = [r for r in rows if r["status"] != "valid"]
    print("  Q1  DECODING")
    if not fails:
        print("      all clips decoded and measured.")
    else:
        print(f"      {len(fails)} failed:")
        seen = {}
        for r in fails:
            key = r["reason"].split(":")[0]
            seen[key] = seen.get(key, 0) + 1
        for k, c in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"        {c:>3}  {k}")

    # --- Q2 gated loudness ------------------------------------------------
    gated = [r for r in ok if r["integrated_lufs"] is None]
    nb = np.array([r["n_blocks"] for r in ok])
    dur = np.array([r["duration_s"] for r in ok])
    print("\n  Q2  GATED LOUDNESS ON SHORT CLIPS")
    print(f"      duration   median {np.median(dur):.2f}s   "
          f"range {dur.min():.2f}-{dur.max():.2f}s")
    print(f"      blocks     median {np.median(nb):.0f}   "
          f"min {nb.min()}  (overlapping windows, not a sample size)")
    if gated:
        print(f"      {len(gated)} of {len(ok)} clips returned no integrated "
              f"loudness - BS.1770 gating discarded everything.")
        print("      mean_voice_loudness cannot be computed per clip for those.")
    else:
        print("      every clip returned a finite integrated loudness. The")
        print("      gate is not a problem at this clip length.")

    # --- Q3 is there any background to measure ---------------------------
    q15 = np.array([r["quiet_frac_15lu"] for r in ok])
    q10 = np.array([r["quiet_frac_10lu"] for r in ok])
    s2b = np.array([r["speech_to_background"] for r in ok])
    print("\n  Q3  IS THERE ANY NON-SPEECH TO CALL BACKGROUND?")
    print(f"      fraction of each clip >15 LU below its own speech level:")
    print(f"        median {np.median(q15):.1%}   "
          f"clips with none at all: {int((q15 == 0).sum())}/{len(ok)}")
    print(f"      same at >10 LU:  median {np.median(q10):.1%}   "
          f"none at all: {int((q10 == 0).sum())}/{len(ok)}")
    print(f"      p90-p10 separation   median {np.median(s2b):.1f} LU   "
          f"range {s2b.min():.1f}-{s2b.max():.1f}")

    if np.median(q15) < 0.05:
        print("\n      VERDICT: clips are essentially wall-to-wall speech.")
        print("      The AMI method - estimate background from gaps - does not")
        print("      transfer. Two options remain, and the choice must be")
        print("      pre-registered before looking at any gender difference:")
        print("        (a) treat p90-p10 as a mix-separation proxy and declare")
        print("            it a different operationalisation from AMI's, or")
        print("        (b) separate the dialogue stem from the rest of the mix")
        print("            and measure each directly. Heavier, but it measures")
        print("            the thing the feature is supposed to measure.")
    else:
        print("\n      VERDICT: there is measurable non-speech within clips.")
        print("      A gap-based background estimate is worth trying, but")
        print("      check first whether those quiet frames are room tone or")
        print("      laugh-track decay - the laugh is loudest just after a")
        print("      line ends, which would invert the measurement.")

    # --- Q4 is absolute loudness comparable -------------------------------
    by_ep = {}
    for r in ok:
        if r["integrated_lufs"] is not None:
            by_ep.setdefault(r["clip_id"], []).append(r["integrated_lufs"])
    within = [float(np.std(v, ddof=1)) for v in by_ep.values() if len(v) > 1]
    allv = [r["integrated_lufs"] for r in ok if r["integrated_lufs"] is not None]
    print("\n  Q4  IS ABSOLUTE LOUDNESS COMPARABLE ACROSS CLIPS?")
    if allv:
        print(f"      overall SD of integrated loudness  {np.std(allv, ddof=1):.1f} LU")
    if within:
        print(f"      within-episode SD (median of {len(within)} episodes)  "
              f"{np.median(within):.1f} LU")
        if np.median(within) > 4.0:
            print("      Within-episode spread is large, so clip loudness is")
            print("      driven by scene and delivery, not a per-episode gain")
            print("      offset. Absolute LUFS is usable without referencing,")
            print("      but it is also noisy - many clips per speaker needed.")
    else:
        print("      too few clips per episode in this sample to separate")
        print("      within-episode from between-episode spread. Re-run with")
        print("      --n-clips 40 if this matters.")

    # --- per gender, flagged as not a result -----------------------------
    print("\n  Per gender (DIAGNOSTIC ONLY - n is tiny and clips are not")
    print("  matched on scene, emotion or speaker. Do not read this as a")
    print("  finding or put it in a report.)")
    print(f"      {'':4}{'n':>4}{'LUFS':>9}{'RMS':>9}{'p90-p10':>10}{'IQR':>8}")
    for g in ("F", "M"):
        sel = [r for r in ok if r["gender"] == g]
        if not sel:
            continue

        def m(k):
            v = [r[k] for r in sel if r[k] is not None]
            return float(np.mean(v)) if v else float("nan")

        print(f"      {g:<4}{len(sel):>4}{m('integrated_lufs'):>9.1f}"
              f"{m('rms_dbfs'):>9.1f}{m('speech_to_background'):>10.1f}"
              f"{m('block_iqr'):>8.1f}")
    print("\n      If a gap appears in LUFS but not in RMS it is spectral -")
    print("      K-weighting treating the two voice ranges differently - not")
    print("      an editorial decision. Both columns exist so that check can")
    print("      be made on the full run.")


# ==========================================================================

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="C4 MELD feasibility pilot (does not extract features).")
    ap.add_argument("--meld-dir", default=find_corpus_dir(here, "MELD"))
    ap.add_argument("--out-dir", default=os.path.normpath(
        os.path.join(here, "..", "output")))
    ap.add_argument("--split", default="train", choices=sorted(SPLIT_VIDEO_DIR))
    ap.add_argument("--n-clips", type=int, default=10,
                    help="clips per gender (default 10, so 20 total)")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--list-speakers", action="store_true",
                    help="report speaker coverage from the CSV only, no audio")
    a = ap.parse_args()

    if not os.path.isdir(a.meld_dir):
        sys.exit(f"MELD folder not found: {a.meld_dir}\n"
                 f"  Expected it beside the repo, as a sibling of equvid/.\n"
                 f"  Pass --meld-dir if it is elsewhere.")

    root = find_meld_root(a.meld_dir)
    if not root:
        sys.exit(f"Could not find train/train_splits under {a.meld_dir}\n"
                 f"  Has MELD.Raw.tar.gz been extracted?")

    csv_path = find_split_csv(root, a.split)
    if not csv_path:
        sys.exit(f"{a.split}_sent_emo.csv not found under {root}")

    rows = read_utterances(csv_path)
    if not rows:
        sys.exit(f"No usable rows in {csv_path}")

    print(f"\n  MELD root  {root}")
    print(f"  annotations {csv_path}  ({len(rows)} utterances)")

    if a.list_speakers:
        list_speakers(rows)
        return

    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("Missing numpy.  pip install -r requirements-c4.txt")

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        sys.exit("ffmpeg not found, and .mp4 audio cannot be decoded without "
                 "it.\n  Easiest fix, no system install needed:\n"
                 "      pip install imageio-ffmpeg")

    run_pilot(a, root, rows, ffmpeg)


if __name__ == "__main__":
    main()
