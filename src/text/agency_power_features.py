#!/usr/bin/env python3
"""C3: agency_score_mean and power_score_mean from the parsed frames.

    python agency_power_features.py
    python agency_power_features.py --root "D:/SLIIT/4th Year/Y4 S1/Research Project"

Reads   derived/c3/c3_verb_frames.csv.gz       frames from parse_frames.py
        derived/c3/c3_lexical_features.csv      which clips are scorable (status)
        Lexicons/sap2017_agency_power/agency_power.csv
Writes  derived/c3/c3_agency_power_stats.csv    sums and counts per clip x gender
        derived/c3/c3_agency_power_clip.csv     one row per clip (output contract)
        equvid/components/C3_Text/results/c3_agency_power_summary.csv   per corpus, 95% CI
        equvid/components/C3_Text/results/c3_agency_power_coverage.csv  lexicon coverage
        equvid/components/C3_Text/results/c3_agency_power_verbs.csv     most frequent verbs

Needs no spaCy: the parse is done once, by parse_frames.py.

Definitions, fixed before the first run (decision D5 in C3_decision_log.md):

  agency  (equation)  mean agency of the verbs whose subject is the speaker
                      ("I quit"), in lines spoken by each gender. Feature =
                      women minus men; positive = women's own lines give them
                      more agency. Each verb scores +1, 0 or -1, so the
                      difference lies between -2 and +2.
  power   (equation)  mean agent-role power over the same verbs.
  Negated verbs ("I didn't quit") are left out: the event is not asserted.

Pre-declared checks, reported but never in the equation:
  *_with_negation     the same, keeping negated verbs
  *_no_markers        the same without think, mean, guess, know, suppose, see
                      and bet, which after "I" are mostly discourse markers
                      ("I mean", "I guess"), not actions
  ref_agency, ref_power   verbs whose subject is "she" or "he": how each
                      gender is described by others (exploratory)
  theme_power         power when the character is the object ("he fired me",
                      "they fired her"), with theme-role values
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lexicon import (AGENCY_VALUE, POWER_AS_AGENT, POWER_AS_THEME, checksum,
                     load_lexicon, lookup, LEXICON_MD5)

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]
FEATURE_VERSION = "c3-ap-0.1"
MIN_FRAMES = 5          # frames per gender for a clip-level value
GENDERS = ("F", "M")
MARKERS = {"think", "mean", "guess", "know", "suppose", "see", "bet"}

# name, roles used, keep negated?, keep markers?, score, role in C3
VARIANTS = [
    ("agency", ["self_agent"], False, True, "agency", "equation"),
    ("power", ["self_agent"], False, True, "power", "equation"),
    ("agency_with_negation", ["self_agent"], True, True, "agency", "sensitivity"),
    ("power_with_negation", ["self_agent"], True, True, "power", "sensitivity"),
    ("agency_no_markers", ["self_agent"], False, False, "agency", "sensitivity"),
    ("power_no_markers", ["self_agent"], False, False, "power", "sensitivity"),
    ("ref_agency", ["ref_agent"], False, True, "agency", "exploratory"),
    ("ref_power", ["ref_agent"], False, True, "power", "exploratory"),
    ("theme_power", ["self_theme", "ref_theme"], False, True, "power", "exploratory"),
]
ROLE = {name: role for name, *_, role in VARIANTS}


# ----------------------------------------------------------------------------
# 1. scores per frame
# ----------------------------------------------------------------------------
def score_frames(frames, lex):
    """Add the lexicon key, labels and the agency/power score of each frame.

    Agency is defined for the agent (subject) only. Power has a value for
    both roles, with the sign flipped for the theme: if a verb gives power to
    its agent ("X commands Y"), the theme is the one without power."""
    f = frames.copy()
    f["particles"] = f.particles.fillna("")
    f["key"] = [lookup(lex, l, p) for l, p in zip(f.lemma, f.particles)]
    f["agency_label"] = [lex[k]["agency"] if k else None for k in f.key]
    f["power_label"] = [lex[k]["power"] if k else None for k in f.key]
    agent = f.role.str.endswith("agent")
    f["agency"] = f.agency_label.map(AGENCY_VALUE).where(agent)
    f["power"] = np.where(agent, f.power_label.map(POWER_AS_AGENT),
                          f.power_label.map(POWER_AS_THEME))
    return f


# ----------------------------------------------------------------------------
# 2. sums and counts per clip x gender, one block per variant
# ----------------------------------------------------------------------------
def variant_stats(f):
    rows = []
    for name, roles, keep_neg, keep_markers, score, _ in VARIANTS:
        m = f.role.isin(roles) & f[score].notna()
        if not keep_neg:
            m &= f.negated == 0
        if not keep_markers:
            m &= ~f.key.isin(MARKERS)
        g = (f[m].groupby(["clip_id", "character_gender"])[score]
             .agg(["sum", "count"]).reset_index())
        g.insert(0, "variant", name)
        rows.append(g.rename(columns={"character_gender": "gender", "count": "n"}))
    return pd.concat(rows, ignore_index=True)


def diff_of_means(sF, nF, sM, nM):
    return sF / nF - sM / nM if nF > 0 and nM > 0 else np.nan


# ----------------------------------------------------------------------------
# 3. per clip (output contract) and per corpus (pooled, bootstrap over series)
# ----------------------------------------------------------------------------
def wide(stats):
    """variant x gender sums and counts as columns, one row per clip."""
    w = stats.pivot_table(index="clip_id", columns=["variant", "gender"],
                          values=["sum", "n"], aggfunc="sum", fill_value=0)
    return w


def clip_table(stats, clips):
    w = wide(stats)
    out = clips[["clip_id", "corpus", "series", "status", "na_reason"]].copy()
    notes = {c: [] for c in out.clip_id}
    for name, *_ in VARIANTS:
        vals = []
        for c in out.clip_id:
            def get(kind, g):
                key = (kind, name, g)
                return w.loc[c, key] if c in w.index and key in w.columns else 0
            nF, nM = get("n", "F"), get("n", "M")
            if min(nF, nM) >= MIN_FRAMES:
                vals.append(diff_of_means(get("sum", "F"), nF, get("sum", "M"), nM))
            else:
                vals.append(np.nan)
                if ROLE[name] == "equation":
                    notes[c].append(f"{name}: fewer than {MIN_FRAMES} frames for one gender")
        out[name] = vals
    out["feature_na"] = ["; ".join(notes[c]) for c in out.clip_id]
    out["feature_version"] = FEATURE_VERSION
    return out


def corpus_summary(stats, clips, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    w = wide(stats)
    rows = []
    for corpus, cl in clips[clips.status == "valid"].groupby("corpus"):
        series = sorted(cl.series.unique())
        s_index = {s: i for i, s in enumerate(series)}
        for name, *_, role in VARIANTS:
            # per-series totals: sum_F, n_F, sum_M, n_M
            tot = np.zeros((len(series), 4))
            for c, s in zip(cl.clip_id, cl.series):
                if c not in w.index:
                    continue
                for j, (kind, g) in enumerate([("sum", "F"), ("n", "F"), ("sum", "M"), ("n", "M")]):
                    key = (kind, name, g)
                    if key in w.columns:
                        tot[s_index[s], j] += w.loc[c, key]
            sF, nF, sM, nM = tot.sum(axis=0)
            est = diff_of_means(sF, nF, sM, nM)
            # resample whole series, as for the lexical features
            picks = rng.integers(0, len(series), (n_boot, len(series)))
            b = tot[picks].sum(axis=1)                          # n_boot x 4
            with np.errstate(divide="ignore", invalid="ignore"):
                boots = b[:, 0] / b[:, 1] - b[:, 2] / b[:, 3]
            lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (np.nan, np.nan)
            rows.append({"corpus": corpus, "feature": name, "role": role,
                         "estimate": est, "ci_low": lo, "ci_high": hi,
                         "mean_F": sF / nF if nF else np.nan,
                         "mean_M": sM / nM if nM else np.nan,
                         "frames_F": int(nF), "frames_M": int(nM),
                         "n_clips": len(cl), "n_series": len(series), "n_boot": n_boot,
                         "feature_version": FEATURE_VERSION})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. diagnostics: coverage and the verbs behind the numbers
# ----------------------------------------------------------------------------
def coverage(f):
    rows = []
    for (corpus, role, g), d in f.groupby(["corpus", "role", "character_gender"]):
        if role not in ("self_agent", "ref_agent"):
            continue
        kept = d[d.negated == 0]
        lab = kept.agency_label.dropna()
        rows.append({
            "corpus": corpus, "role": role, "gender": g, "verbs": len(d),
            "negated_share": d.negated.mean(),
            "agency_coverage": d.agency_label.notna().mean(),
            "power_coverage": d.power_label.notna().mean(),
            "agency_pos_share": (lab == "agency_pos").mean() if len(lab) else np.nan,
            "agency_neg_share": (lab == "agency_neg").mean() if len(lab) else np.nan,
            "agency_equal_share": (lab == "agency_equal").mean() if len(lab) else np.nan,
            "marker_share": kept.key.isin(MARKERS).mean() if len(kept) else np.nan,
        })
    return pd.DataFrame(rows)


def top_verbs(f, n=15):
    d = f[(f.role == "self_agent") & (f.negated == 0)].copy()
    d["verb"] = d.key.where(d.key != "", d.lemma + " (not in lexicon)")
    rows = []
    for (corpus, g), x in d.groupby(["corpus", "character_gender"]):
        counts = x.verb.value_counts()
        for verb, k in counts.head(n).items():
            first = x[x.verb == verb].iloc[0]
            rows.append({"corpus": corpus, "gender": g, "verb": verb, "frames": int(k),
                         "share": k / len(x), "agency_label": first.agency_label,
                         "power_label": first.power_label})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    d = args.root / "derived" / "c3"
    res = args.root / "equvid" / "components" / "C3_Text" / "results"
    lex_path = args.root / "Lexicons" / "sap2017_agency_power" / "agency_power.csv"

    if checksum(lex_path) != LEXICON_MD5:
        print("WARNING: the lexicon file differs from the copy documented in SOURCE.md")
    lex = load_lexicon(lex_path)
    frames = pd.read_csv(d / "c3_verb_frames.csv.gz", keep_default_na=False,
                         na_values={"utt_idx": [""]})
    # Read as text: the status column holds the string "NA", which pandas
    # would otherwise turn into a missing value.
    clips = pd.read_csv(d / "c3_lexical_features.csv", dtype=str, keep_default_na=False,
                        usecols=["clip_id", "corpus", "series", "status", "na_reason"])

    # Same population as every other C3 feature: lines by speakers with a
    # verified gender, in clips that passed the lexical step's checks.
    valid = set(clips.clip_id[clips.status == "valid"])
    f = score_frames(frames, lex)
    f = f[f.speaker_gender.isin(GENDERS) & f.character_gender.isin(GENDERS)
          & f.clip_id.isin(valid)]

    stats = variant_stats(f)
    stats.insert(len(stats.columns), "feature_version", FEATURE_VERSION)
    stats.to_csv(d / "c3_agency_power_stats.csv", index=False)
    clip_table(stats, clips).to_csv(d / "c3_agency_power_clip.csv", index=False)

    summary = corpus_summary(stats, clips, args.boot)
    summary.to_csv(res / "c3_agency_power_summary.csv", index=False)
    cov = coverage(f)
    cov.to_csv(res / "c3_agency_power_coverage.csv", index=False)
    top_verbs(f).to_csv(res / "c3_agency_power_verbs.csv", index=False)

    pd.set_option("display.width", 160)
    print(summary[["corpus", "feature", "role", "estimate", "ci_low", "ci_high",
                   "mean_F", "mean_M", "frames_F", "frames_M"]].round(3).to_string(index=False))
    print()
    print(cov.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
