"""
C4 - MELD v0.2: fixed-window extraction of all six features.

WHAT CHANGED FROM c4_pilot_meld.py, AND WHY
-------------------------------------------
v0.1 measured each clip over its whole length. c4_check_meld_duration.py then
showed that every spread statistic tracked clip duration:

    floor (p10)           r = -0.32
    p90 - p10             r = +0.48
    block loudness IQR    r = +0.34
    block loudness SD     r = +0.50

The IQR result is the awkward one, because decision D1 adopted the IQR on AMI
specifically for being length-robust. It is not length-robust in general. It
only looked that way on AMI because the selection rule there took the twelve
LONGEST turns per gender, so durations clustered between 19 and 26 seconds.
MELD clips run from 0.8 s to 13.7 s - a factor of seventeen - and two separate
mechanisms then bite:

  1. A 0.8 s clip yields 5 overlapping 400 ms blocks. The spread of a
     distribution cannot be estimated from 5 points, and the error is not
     symmetric: too few samples systematically UNDERSTATES spread.
  2. A two-second utterance genuinely has less room for prosodic variation
     than a ten-second one. That is a fact about sentence length, not about
     how the voice was mixed.

Neither has anything to do with the feature's declared meaning, and a metric
whose features correlate with clip length will report bias wherever one gender
happens to get longer lines.

THE FIX
-------
Every spread statistic is computed over a FIXED-LENGTH window, so each clip
contributes exactly the same number of blocks and frames. The artefact is
removed by construction rather than by hoping it is small. This is also the
easier claim to defend: spread statistics are comparable only when computed
over equal-length windows.

The window is taken from the CENTRE of the clip. Centre rather than start,
because the first and last moments of a cut utterance carry onset and offset
artefacts from the edit. Centre rather than "the loudest window", because
choosing the window by loudness would be selecting on the very quantity being
measured.

The cost is that clips shorter than the window are unusable. The summary
reports how many that is, so the trade is visible rather than buried.

ACCEPTANCE TEST
---------------
The run prints, for every statistic, its correlation with FULL clip duration.
If the fix worked, all of them fall below 0.30. Any statistic that does not is
not fit to enter the equation, whatever it appears to show about gender.

SAMPLING
--------
v0.1 drew clips at random across the whole split, which scattered 114 clips
over 79 episodes and left only 12 episodes holding both genders - so the
gender comparison was uninformative. This version samples EPISODES first, then
a quota of clips per gender inside each one. That is also the design the real
extraction run needs, because the episode is the statistical unit: Part 4 of
the duration check found 52% of loudness variance sitting between episodes,
so clips must be compared within an episode before being aggregated.

A NOTE ON THE BASELINE ARM
--------------------------
f0_std, energy_std and amplitude_std are transferred from EuqAud. Their
definitions here must match whatever c4_batch_ami4.py does, or the two corpora
are not measured on the same scale. They are written out explicitly below so
they can be diffed against the AMI script:

    f0_std          std of librosa.pyin voiced frames, in Hz and in semitones
    energy_std      std of frame-wise RMS (frame 1024, hop 256), linear
    amplitude_std   std of the raw samples

Check these against the AMI version before comparing any numbers across
corpora.

USAGE
-----
    python c4_pilot_meld2.py
    python c4_pilot_meld2.py --window 1.5
    python c4_pilot_meld2.py --window 2.0 --n-episodes 40 --per-episode 10
"""

import argparse
import csv
import math
import os
import random
import sys

try:
    from c4_pilot_meld import (BLOCK_S, GENDER_OF, HOP_S, SPLIT_VIDEO_DIR,
                               TARGET_SR, block_loudness, decode_mono,
                               find_corpus_dir, find_ffmpeg, find_meld_root,
                               find_split_csv, read_utterances, rms_dbfs,
                               speaker_key)
except ImportError:
    sys.exit("Run this from the same folder as c4_pilot_meld.py.")

FEATURE_VERSION = "c4-meld-0.2-fixedwindow"

# Baseline arm is computed at 16 kHz to match AMI's Mix-Headset audio, so the
# transferred EuqAud features are measured on the same footing across corpora.
# Loudness stays at 48 kHz: BS.1770 is defined per sample rate, and C4 only
# ever compares within a corpus, so a constant offset between corpora is
# harmless.
BASELINE_SR = 16000
PITCH_FMIN, PITCH_FMAX = 60, 400
FRAME, HOP = 1024, 256

# Statistics checked against duration. Anything above this is unusable.
R_LIMIT = 0.30


def need(mod, hint):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"Missing package '{mod}'.  pip install {hint}")


def blocks_in(window_s):
    """How many 400 ms blocks fit in the window at a 100 ms hop."""
    return int((window_s - BLOCK_S) / HOP_S) + 1


def centre_window(y, sr, window_s):
    """The middle `window_s` seconds of the clip, or None if too short."""
    n = int(round(window_s * sr))
    if len(y) < n:
        return None
    start = (len(y) - n) // 2
    return y[start:start + n]


def features(seg48, sr48, meter, np, librosa):
    """All six features plus controls, from one fixed-length window.

    Everything downstream depends on this being computed over an identical
    number of samples for every clip, so no statistic here may depend on the
    clip's own length.
    """
    out = {}

    # --- contribution arm, from the 48 kHz window ----------------------
    try:
        L = meter.integrated_loudness(seg48)
        out["mean_voice_loudness"] = float(L) if np.isfinite(L) else None
    except Exception:
        out["mean_voice_loudness"] = None
    out["rms_dbfs"] = rms_dbfs(seg48, np)

    b = block_loudness(meter, seg48, sr48, np)
    out["n_blocks"] = len(b)
    if len(b) >= 4:
        arr = np.array(b)
        p90, p10 = float(np.percentile(arr, 90)), float(np.percentile(arr, 10))
        out["p90_lufs"], out["p10_lufs"] = p90, p10
        # Still a percentile proxy, NOT a separated measurement. It stays in
        # the output as a comparison for the separation pilot, and must not be
        # reported as speech-to-background until separation confirms it.
        out["s2b_proxy"] = p90 - p10
        out["dynamic_range"] = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        out["block_sd"] = float(np.std(arr, ddof=1))
    else:
        for k in ("p90_lufs", "p10_lufs", "s2b_proxy", "dynamic_range", "block_sd"):
            out[k] = None

    # --- baseline arm, from the same window at 16 kHz ------------------
    seg16 = librosa.resample(seg48, orig_sr=sr48, target_sr=BASELINE_SR)

    try:
        f0, voiced, _ = librosa.pyin(seg16, fmin=PITCH_FMIN, fmax=PITCH_FMAX,
                                     sr=BASELINE_SR, frame_length=FRAME)
        v = f0[np.isfinite(f0)]
        if len(v) >= 6:
            out["f0_std"] = float(np.std(v, ddof=1))
            out["f0_semitone_std"] = float(np.std(12 * np.log2(v), ddof=1))
            out["f0_median"] = float(np.median(v))
            out["voiced_frac"] = float(np.mean(voiced)) if voiced is not None else None
        else:
            for k in ("f0_std", "f0_semitone_std", "f0_median", "voiced_frac"):
                out[k] = None
    except Exception:
        for k in ("f0_std", "f0_semitone_std", "f0_median", "voiced_frac"):
            out[k] = None

    rms = librosa.feature.rms(y=seg16, frame_length=FRAME, hop_length=HOP)[0]
    out["energy_std"] = float(np.std(rms, ddof=1)) if len(rms) > 2 else None
    out["amplitude_std"] = float(np.std(seg16, ddof=1)) if len(seg16) > 2 else None

    return out


# Columns reported in the duration table and the correlation matrix.
CHECK = [
    ("mean_voice_loudness", "mean_voice_loudness"),
    ("rms_dbfs", "unweighted RMS (control)"),
    ("dynamic_range", "dynamic_range (IQR)"),
    ("s2b_proxy", "s2b proxy (p90-p10)"),
    ("f0_std", "f0_std (Hz)"),
    ("f0_semitone_std", "f0_std (semitones)"),
    ("energy_std", "energy_std"),
    ("amplitude_std", "amplitude_std"),
]


def pearson(x, y, np):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y, np):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    return pearson(np.argsort(np.argsort(x)).astype(float),
                   np.argsort(np.argsort(y)).astype(float), np)


def binom_two_sided(k, n):
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def sample_episodes(rows, per_ep, n_eps, rng):
    """Pick episodes that can meet the quota, then clips inside them.

    Episode-first, because the episode is the statistical unit and the gender
    comparison has to happen inside one. Sampling clips at random across the
    split - what v0.1 did - leaves almost no episode holding both genders.
    """
    eps = {}
    for r in rows:
        g = GENDER_OF.get(speaker_key(r["speaker_raw"]))
        if not g:
            continue
        eps.setdefault((r["season"], r["episode"]), {"F": [], "M": []})[g].append(r)

    qualified = [k for k, v in eps.items()
                 if len(v["F"]) >= per_ep and len(v["M"]) >= per_ep]
    rng.shuffle(qualified)
    chosen = qualified[:n_eps] if n_eps > 0 else qualified

    picked = []
    for k in chosen:
        for g in ("F", "M"):
            pool = list(eps[k][g])
            rng.shuffle(pool)
            picked += [(g, r) for r in pool[:per_ep]]
    return picked, len(qualified), len(eps)


def main():
    np = need("numpy", "numpy")
    librosa = need("librosa", "librosa")
    try:
        import pyloudnorm as pyln
    except ImportError:
        sys.exit("Missing pyloudnorm.  pip install pyloudnorm")

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="C4 MELD v0.2 - fixed-window extraction of all six features.")
    ap.add_argument("--meld-dir", default=find_corpus_dir(here, "MELD"))
    ap.add_argument("--out-dir", default=os.path.normpath(
        os.path.join(here, "..", "output")))
    ap.add_argument("--split", default="train", choices=sorted(SPLIT_VIDEO_DIR))
    ap.add_argument("--window", type=float, default=2.0,
                    help="fixed analysis window in seconds (default 2.0)")
    ap.add_argument("--per-episode", type=int, default=10,
                    help="clips per gender per episode (default 10)")
    ap.add_argument("--n-episodes", type=int, default=30,
                    help="episodes to sample; 0 means all qualified")
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()

    root = find_meld_root(a.meld_dir)
    if not root:
        sys.exit(f"Could not find train/train_splits under {a.meld_dir}")
    cpath = find_split_csv(root, a.split)
    if not cpath:
        sys.exit(f"{a.split}_sent_emo.csv not found under {root}")
    vid_dir = os.path.join(root, SPLIT_VIDEO_DIR[a.split])
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        sys.exit("ffmpeg not found.  pip install imageio-ffmpeg")

    rows = read_utterances(cpath)
    rng = random.Random(a.seed)
    picked, n_qual, n_total = sample_episodes(rows, a.per_episode,
                                              a.n_episodes, rng)
    if not picked:
        sys.exit(f"No episode has {a.per_episode} clips of each gender. "
                 f"Lower --per-episode.")

    nb = blocks_in(a.window)
    meter = pyln.Meter(TARGET_SR, block_size=BLOCK_S)

    print(f"\n  MELD root   {root}")
    print(f"  split {a.split}   window {a.window:.2f}s ({nb} blocks)   "
          f"{a.per_episode} clips/gender/episode   seed {a.seed}")
    print(f"  episodes    {n_qual} of {n_total} meet the quota; "
          f"sampling {len(set((r['season'], r['episode']) for _, r in picked))}")
    print(f"  clips       {len(picked)}\n")

    hdr = (f"  {'file':<18}{'g':>2}{'spk':>9}{'dur':>6}{'LUFS':>8}{'RMS':>7}"
           f"{'IQR':>6}{'f0':>7}{'f0sd':>7}{'semi':>6}{'enSD':>7}{'ampSD':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    out_rows, n_short, n_fail = [], 0, 0
    for g, r in picked:
        stem = f"dia{r['dialogue_id']}_utt{r['utterance_id']}"
        path = os.path.join(vid_dir, stem + ".mp4")
        rec = {
            "clip_id": f"MELD:S{r['season']:02d}E{r['episode']:02d}",
            "utt_id": f"MELD:{stem}",
            "person_id": f"MELD:{speaker_key(r['speaker_raw'])}",
            "gender": g, "split": a.split, "emotion": r["emotion"],
            "window_s": a.window, "feature_version": FEATURE_VERSION,
            "status": "NA", "reason": "", "duration_s": None,
        }
        for k, _ in CHECK:
            rec[k] = None
        for k in ("n_blocks", "p90_lufs", "p10_lufs", "block_sd",
                  "f0_median", "voiced_frac"):
            rec[k] = None

        if not os.path.exists(path):
            rec["reason"] = "video file missing"
            out_rows.append(rec)
            n_fail += 1
            continue
        try:
            y, sr = decode_mono(ffmpeg, path)
        except RuntimeError as e:
            rec["reason"] = f"decode failed: {e}"
            out_rows.append(rec)
            n_fail += 1
            continue

        rec["duration_s"] = round(len(y) / sr, 3)
        seg = centre_window(y, sr, a.window)
        if seg is None:
            rec["reason"] = f"shorter than the {a.window:.2f}s window"
            out_rows.append(rec)
            n_short += 1
            continue

        rec.update(features(seg, sr, meter, np, librosa))
        rec["status"] = "valid"
        out_rows.append(rec)

        def c(v, w, p=1):
            return f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'-':>{w}}"

        print(f"  {stem:<18}{g:>2}{speaker_key(r['speaker_raw']):>9}"
              f"{rec['duration_s']:>6.1f}{c(rec['mean_voice_loudness'], 8)}"
              f"{c(rec['rms_dbfs'], 7)}{c(rec['dynamic_range'], 6)}"
              f"{c(rec['f0_median'], 7, 0)}{c(rec['f0_std'], 7)}"
              f"{c(rec['f0_semitone_std'], 6)}"
              f"{c(rec['energy_std'], 7, 3)}{c(rec['amplitude_std'], 7, 3)}")

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"c4_meld_v2_{a.split}_w{a.window:g}.csv")
    cols = list(out_rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for rr in out_rows:
            w.writerow({k: (round(rr[k], 6) if isinstance(rr.get(k), float)
                            else rr.get(k, "")) for k in cols})

    report(out_rows, a, np, n_short, n_fail)
    print(f"\n  Wrote {out}")


def report(out_rows, a, np, n_short, n_fail):
    ok = [r for r in out_rows if r["status"] == "valid"]
    print("\n" + "=" * 78)
    print(f"  {len(ok)} of {len(out_rows)} clips measured     "
          f"{n_short} shorter than the window, {n_fail} failed to decode")
    if not ok:
        print("  Nothing measured.")
        return
    print(f"  cost of the fixed window: {n_short / len(out_rows):.0%} of clips "
          f"dropped at {a.window:.2f}s")

    dur = np.array([r["duration_s"] for r in ok], float)

    # ---- the acceptance test -----------------------------------------
    print("\n  ACCEPTANCE TEST - does any statistic still track clip duration?")
    print(f"      duration median {np.median(dur):.2f}s   "
          f"range {dur.min():.2f}-{dur.max():.2f}s\n")
    print(f"      {'statistic':<26}{'r':>8}{'rho':>8}{'verdict':>12}")
    print("      " + "-" * 54)
    failed = []
    for key, label in CHECK:
        v = np.array([r[key] if r[key] is not None else np.nan for r in ok], float)
        m = np.isfinite(v)
        if m.sum() < 10:
            continue
        r = pearson(dur[m], v[m], np)
        rho = spearman(dur[m], v[m], np)
        worst = max(abs(x) for x in (r, rho) if x == x)
        bad = worst > R_LIMIT
        if bad:
            failed.append(label)
        print(f"      {label:<26}{r:>+8.2f}{rho:>+8.2f}"
              f"{('FAIL' if bad else 'pass'):>12}")
    print()
    if failed:
        print(f"      STILL CONTAMINATED: {', '.join(failed)}")
        print(f"      A longer window would help - try --window "
              f"{a.window + 1.0:g} - at the cost of dropping more clips.")
        print("      Any statistic that cannot be made length-independent is")
        print("      not usable on MELD and must be declared corpus-unavailable.")
    else:
        print("      All statistics are now independent of clip duration. The")
        print("      fixed window did its job; these features are fit to enter")
        print("      the equation.")

    # ---- collinearity, for DR2 and DR4 -------------------------------
    keys = [k for k, _ in CHECK]
    mat = np.array([[r[k] if r[k] is not None else np.nan for k in keys]
                    for r in ok], float)
    print("\n  FEATURE CORRELATION MATRIX (drop rules DR2, DR4)")
    print("      " + "".join(f"{i + 1:>7}" for i in range(len(keys))))
    for i, (k, label) in enumerate(CHECK):
        cells = ""
        for j in range(len(keys)):
            m = np.isfinite(mat[:, i]) & np.isfinite(mat[:, j])
            rr = pearson(mat[m, i], mat[m, j], np) if m.sum() > 3 else float("nan")
            cells += f"{rr:>+7.2f}" if rr == rr else f"{'-':>7}"
        print(f"  {i + 1:>3} {label:<26}"[:30] + cells)
    print("\n      Pairs above |0.80| breach DR4 and one of the two must go.")
    print("      energy_std and amplitude_std are the pair to watch: both are")
    print("      amplitude-domain, and on AMI both returned +0.00 with SD 0.01.")

    # ---- within-episode gender comparison ----------------------------
    by_ep = {}
    for i, r in enumerate(ok):
        by_ep.setdefault(r["clip_id"], {"F": [], "M": []})[r["gender"]].append(i)
    usable = [k for k, v in by_ep.items() if v["F"] and v["M"]]

    print(f"\n  WITHIN-EPISODE GENDER COMPARISON   "
          f"{len(usable)} of {len(by_ep)} episodes usable")
    print("  (exploratory: train split, chosen after seeing the data)\n")
    print(f"      {'statistic':<26}{'M-F':>9}{'SD':>8}{'M>F':>8}{'p':>9}")
    print("      " + "-" * 60)
    for key, label in CHECK:
        diffs = []
        for k in usable:
            fv = [ok[i][key] for i in by_ep[k]["F"] if ok[i][key] is not None]
            mv = [ok[i][key] for i in by_ep[k]["M"] if ok[i][key] is not None]
            if fv and mv:
                diffs.append(float(np.mean(mv) - np.mean(fv)))
        if len(diffs) < 4:
            continue
        arr = np.array(diffs)
        kpos = int((arr > 0).sum())
        p = binom_two_sided(kpos, len(arr))
        print(f"      {label:<26}{arr.mean():>+9.3f}{arr.std(ddof=1):>8.3f}"
              f"{kpos:>4}/{len(arr):<3}{p:>9.4f}")
    print("\n      A statistic that FAILED the acceptance test above must not")
    print("      be read here, whatever its p value - the difference could be")
    print("      clip length rather than gender.")


if __name__ == "__main__":
    main()
