"""
C4 - MELD v0.3: put the measurement window where the voice is.

WHY v0.2 FAILED ITS CONFIRMATORY RUN
------------------------------------
v0.2 fixed the window length, which removed the duration dependence on the
train split. On the dev split the three LEVEL features failed:

                          train          dev
    mean_voice_loudness   -0.15 pass    -0.40 FAIL
    energy_std            -0.13 pass    -0.32 FAIL
    amplitude_std         -0.09 pass    -0.34 FAIL
    dynamic_range (IQR)   +0.10 pass    -0.02 pass
    f0_std (semitones)    -0.05 pass    -0.01 pass

Three things identify the cause:

  1. Pearson was roughly double Spearman (-0.40 against -0.17), which is the
     signature of a few extreme points rather than a monotone trend.
  2. The long clips were the quiet ones. dia74_utt19 is 28.5 s at -31.3 LUFS;
     dia77_utt13 is 9.0 s at -45.1 LUFS with an f0 median of 70 Hz, which is
     silence being tracked as a voice.
  3. dynamic_range passed on the IDENTICAL window. If long clips genuinely had
     different content, the spread features would have moved too. Only level
     moved.

A long MELD "utterance" is frequently not continuous speech - it is a scene
span assigned to one speaker, containing pauses, music and reaction shots. The
centre two seconds of a 28-second clip is usually not the speaker talking. So
v0.2 was measuring silence and calling it the speaker's loudness, and it did so
more often in long clips.

A LONGER WINDOW IS THE WRONG FIX
--------------------------------
It drops more clips and still does not guarantee the window contains speech.
The window has to be placed where the voice is.

WHAT v0.3 DOES
--------------
1. WINDOW PLACEMENT. The block-loudness contour is computed once over the whole
   clip, then the window whose MEDIAN block loudness is highest is selected.

   The legitimacy of this rests on one property: the choice is made from each
   clip's own contour, so it is invariant to the clip's overall level. Multiply
   a clip by any constant gain and every candidate window shifts by the same
   amount, so the same window is still selected - and the loudness then
   measured differs by exactly that gain. Between-clip level differences, which
   are the entire signal C4 is after, therefore survive the selection
   untouched. (test_window_invariance() below checks this, and it is the thing
   to be able to prove in a viva.)

   What the selection does bias is within-clip: it systematically measures the
   speaker's loudest sustained passage rather than an average including their
   pauses. That bias is the same for every clip and for both genders, so
   comparisons hold. It is also closer to what the feature claims to measure -
   the level at which the mix placed this voice, not the level of the silence
   around it.

2. A VALIDITY GATE. A window is rejected, with a reason, when:
     - its integrated loudness is below -40 LUFS, i.e. there is no speech in it
     - its f0 median sits at the 60 Hz or 400 Hz search boundary, which means
       pyin failed rather than found a voice
   Rejected clips become status NA with a reason, as the output contract
   requires. Never a silent drop.

3. REPORTS THE NA RATE BY DURATION TERCILE. If the gate itself removes long
   clips preferentially, that biases the sample instead of the measurement, and
   it has to be visible rather than discovered later.

METHODOLOGICAL STATUS
---------------------
Dev has now been used to diagnose a defect, so dev is spent as a confirmatory
set. M2's conventions are explicit: fixing a design after seeing the test data
makes it exploratory. The TEST split is untouched and is the only clean
confirmation left. Run v0.3 on train and dev first; go to test once, and report
whatever it says.

USAGE
-----
    python c4_pilot_meld3.py --self-test        <- prove the invariance first
    python c4_pilot_meld3.py
    python c4_pilot_meld3.py --split dev --n-episodes 0
    python c4_pilot_meld3.py --window-mode centre     <- reproduce v0.2
"""

import argparse
import csv
import os
import random
import sys

try:
    from c4_pilot_meld import (BLOCK_S, GENDER_OF, HOP_S, SPLIT_VIDEO_DIR,
                               TARGET_SR, decode_mono, find_corpus_dir,
                               find_ffmpeg, find_meld_root, find_split_csv,
                               read_utterances, speaker_key)
    from c4_pilot_meld2 import (CHECK, PITCH_FMAX, PITCH_FMIN, R_LIMIT,
                                binom_two_sided, blocks_in, features, pearson,
                                sample_episodes, spearman)
except ImportError as e:
    sys.exit(f"Run this from the same folder as c4_pilot_meld.py and "
             f"c4_pilot_meld2.py  ({e})")

FEATURE_VERSION = "c4-meld-0.3-voicedwindow"

# A 2 s window of real dialogue in a broadcast mix does not sit below this.
# Anything quieter is a pause, a reaction shot or music, not the speaker.
LOUDNESS_FLOOR = -40.0

# pyin reports values inside [PITCH_FMIN, PITCH_FMAX]. A median sitting on
# either boundary means the tracker ran out of range rather than finding a
# voice, so the pitch features for that clip are not measurements.
PITCH_EDGE_TOL = 2.0


def level_contour(y, sr, np):
    """Ungated block RMS in dB, aligned to position, one pass.

    WHY NOT BS.1770 HERE
    --------------------
    The obvious choice for this contour is the same gated loudness used for the
    feature itself. It is the wrong choice, and the self-test catches why.

    BS.1770 GATES: it discards any block below -70 LUFS absolute, plus blocks
    more than 10 LU under the ungated level. Which blocks get discarded
    therefore depends on the clip's absolute level. Scale a clip down by 18 dB
    and its quiet blocks cross the absolute floor, drop out of the contour, and
    the median of every window containing them changes - so a DIFFERENT window
    is selected for what is acoustically the same clip. Selection would then
    depend on overall level, which is precisely the quantity being measured.

    Ungated RMS has no threshold, so scaling y by g shifts every block by
    exactly 20*log10(g). Every candidate window's median shifts by the same
    constant, the ranking is untouched, and the same window is always chosen.
    Invariance is exact rather than approximate.

    Digital silence stays NaN at any gain, since 0 * g = 0, so it cannot break
    invariance either.

    The feature is still measured with BS.1770 on the chosen window. Selection
    and measurement are separate jobs and do not have to use the same measure.
    """
    n = int(round(BLOCK_S * sr))
    hop = int(round(HOP_S * sr))
    if len(y) < n:
        return np.array([]), hop
    out = []
    for i in range(0, len(y) - n + 1, hop):
        w = y[i:i + n]
        r = float(np.sqrt(np.mean(np.square(w))))
        out.append(20.0 * np.log10(r) if r > 0 else np.nan)
    return np.array(out, dtype=float), hop


def select_window(y, sr, window_s, mode, np):
    """Choose the measurement window. Returns (segment, start_s, n_candidates).

    mode "centre"  - the middle of the clip, as v0.2 did. Kept so the two can
                     be compared on the same data.
    mode "loudest" - the window whose median block loudness is highest.
    """
    n = int(round(window_s * sr))
    if len(y) < n:
        return None, None, 0

    if mode == "centre":
        s = (len(y) - n) // 2
        return y[s:s + n], s / sr, 1

    contour, hop = level_contour(y, sr, np)
    k = blocks_in(window_s)
    if len(contour) < k:
        s = (len(y) - n) // 2
        return y[s:s + n], s / sr, 1

    best_i, best_score = 0, -np.inf
    for i in range(len(contour) - k + 1):
        seg = contour[i:i + k]
        v = seg[np.isfinite(seg)]
        # Require most of the window to be measurable, or a window that is
        # mostly digital silence could win on one loud block.
        if len(v) < k // 2:
            continue
        score = float(np.median(v))
        if score > best_score:
            best_score, best_i = score, i

    s = min(best_i * hop, len(y) - n)
    return y[s:s + n], s / sr, len(contour) - k + 1


def test_window_invariance():
    """Prove the property the method depends on.

    If selection were not invariant to overall gain, a quiet clip and a loud
    copy of the same clip could have different windows chosen, and the measured
    difference between them would then be an artefact of the selection rather
    than a fact about the mix. This checks that:

      - the same window is chosen for a clip and for a scaled copy
      - the loudness measured differs by exactly the applied gain

    Run with --self-test. It uses synthetic audio, so it needs no corpus.
    """
    import numpy as np
    try:
        import pyloudnorm as pyln
    except ImportError:
        sys.exit("Missing pyloudnorm.")

    sr = TARGET_SR
    meter = pyln.Meter(sr, block_size=BLOCK_S)
    rng = np.random.default_rng(0)

    # 8 s: quiet - loud speech-like burst - quiet - medium burst
    t = np.arange(int(8 * sr)) / sr
    y = 0.001 * rng.standard_normal(len(t))
    def burst(a, b, amp):
        i, j = int(a * sr), int(b * sr)
        env = np.hanning(j - i)
        y[i:j] += amp * env * np.sin(2 * np.pi * 180 * t[i:j])
    burst(2.0, 4.2, 0.30)      # the loudest passage
    burst(5.5, 7.0, 0.08)      # a quieter one

    print("\n  Window-selection invariance test")
    print("  " + "-" * 62)
    print("    Selection uses ungated RMS; measurement uses BS.1770. A gated")
    print("    measure cannot be used for selection - see level_contour().")
    print()
    rows = []
    # -40 dB is included deliberately: it pushes the background well under the
    # -70 LUFS absolute gate, which is what broke the gated-contour version.
    for gain_db in (0.0, -6.0, -18.0, -40.0, +6.0):
        g = 10.0 ** (gain_db / 20.0)
        seg, start, ncand = select_window(y * g, sr, 2.0, "loudest", np)
        L = meter.integrated_loudness(seg)
        rows.append((gain_db, start, L, ncand))
        print(f"    gain {gain_db:>+6.1f} dB   window starts {start:>6.2f}s   "
              f"loudness {L:>8.2f} LUFS   ({ncand} candidates)")

    base_start, base_L = rows[0][1], rows[0][2]
    same = all(abs(r[1] - base_start) < 1e-9 for r in rows)
    # The measurement is only expected to track gain while it stays clear of
    # BS.1770's own gates, so -40 dB is excluded from this second check.
    exact = all(abs((r[2] - base_L) - r[0]) < 0.05
                for r in rows if r[0] > -30.0)
    print()
    print(f"    same window chosen at every gain:        "
          f"{'YES' if same else 'NO'}")
    print(f"    loudness tracks the gain exactly:        "
          f"{'YES' if exact else 'NO'}")
    print()
    if same and exact:
        print("    Selection is invariant to overall level, so between-clip")
        print("    level differences survive it. This is what makes choosing")
        print("    the loudest window legitimate rather than circular.")
    else:
        print("    FAILED. Do not use --window-mode loudest until this passes.")
        sys.exit(1)
    overlap = max(0.0, min(base_start + 2.0, 4.2) - max(base_start, 2.0))
    print(f"    The loud passage runs 2.0-4.2s; the selected window is")
    print(f"    {base_start:.2f}-{base_start + 2.0:.2f}s, overlapping it by "
          f"{overlap:.2f}s of 2.00s ({overlap / 2.0:.0%}).")


def main():
    if "--self-test" in sys.argv:
        test_window_invariance()
        return

    import numpy as np
    try:
        import librosa
    except ImportError:
        sys.exit("Missing librosa.  pip install librosa")
    try:
        import pyloudnorm as pyln
    except ImportError:
        sys.exit("Missing pyloudnorm.  pip install pyloudnorm")

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="C4 MELD v0.3 - window placed by loudness, with a gate.")
    ap.add_argument("--meld-dir", default=find_corpus_dir(here, "MELD"))
    ap.add_argument("--out-dir", default=os.path.normpath(
        os.path.join(here, "..", "output")))
    ap.add_argument("--split", default="train", choices=sorted(SPLIT_VIDEO_DIR))
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--window-mode", default="loudest",
                    choices=["loudest", "centre"])
    ap.add_argument("--per-episode", type=int, default=10)
    ap.add_argument("--n-episodes", type=int, default=0,
                    help="0 means every episode meeting the quota")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--self-test", action="store_true",
                    help="prove window-selection invariance and exit")
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
        sys.exit(f"No episode has {a.per_episode} clips of each gender.")

    meter = pyln.Meter(TARGET_SR, block_size=BLOCK_S)
    print(f"\n  MELD root   {root}")
    print(f"  split {a.split}   window {a.window:.2f}s "
          f"({blocks_in(a.window)} blocks, mode={a.window_mode})")
    print(f"  episodes    {n_qual} of {n_total} meet the quota; "
          f"using {len(set((r['season'], r['episode']) for _, r in picked))}")
    print(f"  clips       {len(picked)}")
    print(f"  gate        loudness >= {LOUDNESS_FLOOR:g} LUFS, "
          f"f0 median off the {PITCH_FMIN}/{PITCH_FMAX} Hz boundary\n")

    out_rows = []
    for idx, (g, r) in enumerate(picked):
        stem = f"dia{r['dialogue_id']}_utt{r['utterance_id']}"
        path = os.path.join(vid_dir, stem + ".mp4")
        rec = {
            "clip_id": f"MELD:S{r['season']:02d}E{r['episode']:02d}",
            "utt_id": f"MELD:{stem}",
            # Title-case, per M2's MELD:Chandler example.
            "person_id": f"MELD:{speaker_key(r['speaker_raw']).title()}",
            "corpus": "MELD",
            "series": f"MELD_S{r['season']:02d}",
            "component": "MELD_ALL",
            "gender_source": "manual_map",
            "gender": g, "split": a.split, "emotion": r["emotion"],
            "window_s": a.window, "window_mode": a.window_mode,
            "window_start_s": None,
            "feature_version": FEATURE_VERSION,
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
            continue
        try:
            y, sr = decode_mono(ffmpeg, path)
        except RuntimeError as e:
            rec["reason"] = f"decode failed: {e}"
            out_rows.append(rec)
            continue

        rec["duration_s"] = round(len(y) / sr, 3)
        seg, start_s, _ = select_window(y, sr, a.window, a.window_mode, np)
        if seg is None:
            rec["reason"] = f"shorter than the {a.window:.2f}s window"
            out_rows.append(rec)
            continue
        rec["window_start_s"] = round(start_s, 3)
        rec.update(features(seg, sr, meter, np, librosa))

        # ---- the gate -------------------------------------------------
        L = rec.get("mean_voice_loudness")
        f0m = rec.get("f0_median")
        if L is None:
            rec["reason"] = "no integrated loudness (fully gated)"
        elif L < LOUDNESS_FLOOR:
            rec["reason"] = (f"window at {L:.1f} LUFS, below the "
                             f"{LOUDNESS_FLOOR:g} floor - no speech in it")
        elif f0m is not None and (f0m <= PITCH_FMIN + PITCH_EDGE_TOL
                                  or f0m >= PITCH_FMAX - PITCH_EDGE_TOL):
            rec["reason"] = (f"f0 median {f0m:.0f} Hz is pinned at the search "
                             f"boundary - tracking failure, not a voice")
        else:
            rec["status"] = "valid"
        out_rows.append(rec)

        if (idx + 1) % 100 == 0:
            n_ok = sum(1 for x in out_rows if x["status"] == "valid")
            print(f"    {idx + 1}/{len(picked)} clips, {n_ok} valid")

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir,
                       f"c4_meld_v3_{a.split}_w{a.window:g}_{a.window_mode}.csv")
    cols = list(out_rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for rr in out_rows:
            w.writerow({k: (round(rr[k], 6) if isinstance(rr.get(k), float)
                            else ("" if rr.get(k) is None else rr[k]))
                        for k in cols})

    report(out_rows, a, np)
    print(f"\n  Wrote {out}")


def report(out_rows, a, np):
    ok = [r for r in out_rows if r["status"] == "valid"]
    print("\n" + "=" * 78)
    print(f"  {len(ok)} of {len(out_rows)} clips valid")

    # ---- why the rest were rejected ----------------------------------
    reasons = {}
    for r in out_rows:
        if r["status"] != "valid":
            key = r["reason"].split(" -")[0].split(",")[0][:48]
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("\n  REJECTED, BY REASON")
        for k, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"      {c:>5}  {k}")

    if not ok:
        print("\n  Nothing valid.")
        return

    dur = np.array([r["duration_s"] for r in ok], float)

    # ---- is the gate itself duration-biased? -------------------------
    alld = [r["duration_s"] for r in out_rows if r["duration_s"] is not None]
    if len(alld) > 20:
        q1, q2 = np.percentile(alld, [33, 67])
        print("\n  IS THE GATE BIASED BY CLIP LENGTH?")
        print(f"      {'tercile':<22}{'clips':>8}{'valid':>8}{'rate':>8}")
        for label, lo, hi in (("short", -1e9, q1), ("middle", q1, q2),
                              ("long", q2, 1e9)):
            sel = [r for r in out_rows
                   if r["duration_s"] is not None and lo < r["duration_s"] <= hi]
            v = sum(1 for r in sel if r["status"] == "valid")
            if sel:
                print(f"      {label + f' (<={hi:.1f}s)':<22}{len(sel):>8}"
                      f"{v:>8}{v / len(sel):>8.0%}")
        print("      Rates that differ a lot across terciles mean the gate is")
        print("      removing long clips preferentially, which biases the")
        print("      SAMPLE rather than the measurement. Declare it either way.")

    # ---- the acceptance test -----------------------------------------
    print("\n  ACCEPTANCE TEST - does any statistic still track clip duration?")
    print(f"      duration median {np.median(dur):.2f}s   "
          f"range {dur.min():.2f}-{dur.max():.2f}s\n")
    print(f"      {'statistic':<26}{'r':>8}{'rho':>8}{'verdict':>12}")
    print("      " + "-" * 54)
    failed = []
    for key, label in CHECK:
        v = np.array([r[key] if r[key] is not None else np.nan
                      for r in ok], float)
        m = np.isfinite(v)
        if m.sum() < 10:
            continue
        rr = pearson(dur[m], v[m], np)
        rho = spearman(dur[m], v[m], np)
        worst = max(abs(x) for x in (rr, rho) if x == x)
        bad = worst > R_LIMIT
        if bad:
            failed.append(label)
        print(f"      {label:<26}{rr:>+8.2f}{rho:>+8.2f}"
              f"{('FAIL' if bad else 'pass'):>12}")
    print()
    if failed:
        print(f"      STILL CONTAMINATED: {', '.join(failed)}")
        print("      Check whether Pearson far exceeds Spearman again. If so it")
        print("      is still a handful of clips, not a trend, and the gate")
        print("      needs tightening rather than the window lengthening.")
    else:
        print("      Every statistic is independent of clip duration.")
        print("      If this is the dev or test split, that is the confirmatory")
        print("      result - report it as such.")

    # ---- collinearity -------------------------------------------------
    keys = [k for k, _ in CHECK]
    mat = np.array([[r[k] if r[k] is not None else np.nan for k in keys]
                    for r in ok], float)
    print("\n  FEATURE CORRELATION MATRIX (DR2, DR4)")
    print("      " + "".join(f"{i + 1:>7}" for i in range(len(keys))))
    for i, (k, label) in enumerate(CHECK):
        cells = ""
        for j in range(len(keys)):
            m = np.isfinite(mat[:, i]) & np.isfinite(mat[:, j])
            c = pearson(mat[m, i], mat[m, j], np) if m.sum() > 3 else float("nan")
            cells += f"{c:>+7.2f}" if c == c else f"{'-':>7}"
        print(f"  {i + 1:>3} {label:<26}"[:30] + cells)
    print("\n      Pairs above |0.80| breach DR4. Watch energy_std against")
    print("      mean_voice_loudness: 0.76 on train, 0.80 on dev under v0.2.")

    # ---- within-episode gender comparison, in M2's sign --------------
    by_ep = {}
    for i, r in enumerate(ok):
        by_ep.setdefault(r["clip_id"], {"F": [], "M": []})[r["gender"]].append(i)
    usable = [k for k, v in by_ep.items() if v["F"] and v["M"]]

    print(f"\n  WITHIN-EPISODE GENDER COMPARISON   {len(usable)} of "
          f"{len(by_ep)} episodes usable")
    print("  POSITIVE = WOMEN ABOVE MEN (M2 decision 3)\n")
    print(f"      {'statistic':<26}{'F-M':>9}{'SD':>8}{'F>M':>8}{'p':>9}")
    print("      " + "-" * 60)
    for key, label in CHECK:
        diffs = []
        for k in usable:
            fv = [ok[i][key] for i in by_ep[k]["F"] if ok[i][key] is not None]
            mv = [ok[i][key] for i in by_ep[k]["M"] if ok[i][key] is not None]
            if fv and mv:
                diffs.append(float(np.mean(fv) - np.mean(mv)))
        if len(diffs) < 4:
            continue
        arr = np.array(diffs)
        nz = arr[arr != 0]
        if len(nz) < 4:
            continue
        kpos = int((nz > 0).sum())
        print(f"      {label:<26}{arr.mean():>+9.3f}{arr.std(ddof=1):>8.3f}"
              f"{kpos:>4}/{len(nz):<3}{binom_two_sided(kpos, len(nz)):>9.4f}")
    print("\n      A statistic that FAILED above must not be read here.")


if __name__ == "__main__":
    main()
