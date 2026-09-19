"""
MELD speaking-time baseline -- first empirical result for EuqVid.

WHY THIS EXISTS
---------------
Before extracting a single video frame, we can measure one of the twelve EuqVid
features directly from MELD's annotation CSVs. The `Speaker` column names the
character, and `StartTime`/`EndTime` bound the utterance. Speaking duration per
gender therefore needs no face detection, no VAD, and no lip-activity model.

That makes this the cheapest possible sanity check on the corpus, and it
establishes the count-ratio and speaking-ratio baselines that M4 must later
compare EuqVid against.

WHAT IT ANSWERS
---------------
Friends has three female and three male leads -- perfectly balanced by headcount.
So a face-count ratio baseline will call MELD unbiased by construction. The
question is whether the *temporal* dimension agrees.

Run:
    python src/meld_speaking_baseline.py
"""

import os
import pandas as pd

# CSVs live outside the repo with the rest of the corpus -- see data/README.md
MELD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "MELD")

# Main cast only. MELD has 304 unique speakers, but the six leads account for
# 83.3% of all utterances. The long tail is mostly one-line guest characters
# whose gender would have to be assigned by hand for very little signal.
#
# Documented policy: restrict to the six leads. This is deliberately
# conservative -- it is the subset where gender is unambiguous and where the
# 3F/3M count balance makes the comparison clean.
MAIN_CAST = {
    "Joey": "M",
    "Ross": "M",
    "Chandler": "M",
    "Rachel": "F",
    "Monica": "F",
    "Phoebe": "F",
}


def load_meld(meld_dir=MELD_DIR):
    """Concatenate the three MELD splits into one frame."""
    splits = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]
    return pd.concat(
        [pd.read_csv(os.path.join(meld_dir, s)) for s in splits],
        ignore_index=True,
    )


def to_seconds(series):
    """
    MELD timestamps are 'hh:mm:ss,ms' -- comma as the decimal separator, which
    pandas will not parse. Swap it for a period first.
    """
    return pd.to_timedelta(
        series.str.replace(",", ".", regex=False), errors="coerce"
    )


def main():
    df = load_meld()
    print(f"utterances: {len(df)}   unique speakers: {df.Speaker.nunique()}")

    counts = df.Speaker.value_counts()
    print(f"top 6 speakers cover {counts.head(6).sum() / len(df):.1%} of utterances\n")

    # Restrict to the main cast and compute per-utterance duration
    df["gender"] = df.Speaker.map(MAIN_CAST)
    cast = df.dropna(subset=["gender"]).copy()
    cast["dur"] = (to_seconds(cast.EndTime) - to_seconds(cast.StartTime)).dt.total_seconds()

    # Two rows in the full corpus have malformed or cross-scene timestamps.
    # Bounding to (0, 60) seconds drops them without touching anything real --
    # the longest legitimate utterance in MELD is well under a minute.
    before = len(cast)
    cast = cast[(cast.dur > 0) & (cast.dur < 60)]
    print(f"dropped {before - len(cast)} rows with malformed timestamps\n")

    # --- aggregate by gender ------------------------------------------------
    g = cast.groupby("gender")["dur"].agg(
        utterances="count", total_sec="sum", mean_sec="mean"
    )
    g["utt_share"] = g.utterances / g.utterances.sum()
    g["time_share"] = g.total_sec / g.total_sec.sum()
    print(g.round(3).to_string(), "\n")

    f, m = g.loc["F"], g.loc["M"]
    print(f"utterance count ratio   M:F = {m.utterances / f.utterances:.3f}")
    print(f"speaking time ratio     M:F = {m.total_sec / f.total_sec:.3f}")
    print(
        f"mean utterance length   M = {m.mean_sec:.2f}s   F = {f.mean_sec:.2f}s "
        f"({(m.mean_sec / f.mean_sec - 1) * 100:+.1f}% male)\n"
    )

    # --- per character ------------------------------------------------------
    per = cast.groupby("Speaker")["dur"].agg(utt="count", total_sec="sum", mean_sec="mean")
    per["gender"] = per.index.map(MAIN_CAST)
    print(per.sort_values("total_sec", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
