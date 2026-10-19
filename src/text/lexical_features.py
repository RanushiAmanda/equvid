#!/usr/bin/env python3
"""C3 (EuqVid-Text) steps 5-7 for the four features that need no parser.

    python lexical_features.py
    python lexical_features.py --root "D:/SLIIT/4th Year/Y4 S1/Research Project"

Reads   derived/c3/utterances_meld.csv, utterances_ami.csv, clips_provisional.csv
Writes  derived/c3/c3_clip_gender_stats.csv    counts per clip x gender (re-aggregable)
        derived/c3/c3_lexical_features.csv      one row per clip (output contract)
        equvid/components/C3_Text/results/c3_lexical_corpus_summary.csv
        equvid/components/C3_Text/results/c3_lexical_correlations.csv

Why counts first, features second: a ratio of sums is not the mean of ratios.
Storing the raw counts per clip and gender means any grouping the group agrees
later (shard, series, corpus) is computed exactly by summing counts, with no
re-extraction. That keeps C3 unblocked while the shard design is still open.

Sign convention (shared-conventions proposal): positive = higher for women.

  volume_log_parity        F1  log((words_F / persons_F) / (words_M / persons_M))
                               CHECK ONLY since 19 Oct 2026: it repeats C2's
                               speaking-time parity (r = 0.96-0.97), so it left
                               the C3 equation (decision D1 in C3_decision_log.md)
  volume_share_F           F1 exactly as the spec words it: words_F / all words.
                               Reported, not modelled: it moves with headcount.
  length_log_ratio         F3  log(mean words per utterance, F / M), all lines
  length_log_ratio_ge3     F3 on lines of 3+ words only (C2's backchannel rule)
  same_gender_excess_diff  same_gender_dialogue_share, corrected for headcount
  ref_log_parity           who gets talked about: he/she references to each
                           gender, per person (proposed gendered_reference_ratio)
  gref_share_diff          gendered_reference_ratio exactly as documented
  cross_ref_share_diff     variant kept to show it reduces to the referent mix
  talked_about_log         variant kept to show it equals ref - volume parity
  time_log_parity          NOT a C3 feature. C2's quantity recomputed from the
                           same rows, only to test whether F1 duplicates C2.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wordlists import FEATURE_VERSION, pronoun_counts, tokens

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]

MIN_UTT = 5          # utterances per gender for a clip to be scored at all
MIN_TRANS = 5        # turn changes into each gender, for the adjacency feature
MIN_PRON = 10        # person pronouns per gender, for gref_share
MIN_GREF = 5         # he/she pronouns per gender, for the two exploratory features
MELD_MAX_DUR = 30.0  # a MELD utterance is one subtitle line; longer = bad time stamp

STAT_COLS = ["n_persons", "n_utt", "n_words", "n_utt_ge3", "n_words_ge3", "time",
             "trans", "same", "exp_same", "fem3", "masc3", "neut3", "first", "second"]


# ----------------------------------------------------------------------------
# 1. per-utterance counts
# ----------------------------------------------------------------------------
def add_utterance_counts(u):
    """Add pronoun counts and a usable-duration column to every utterance."""
    pc = pd.DataFrame([pronoun_counts(tokens(t)) for t in u.text], index=u.index)
    u = pd.concat([u, pc], axis=1)
    dur = u.end - u.start
    ok = dur > 0
    ok &= ~((u.corpus == "MELD") & (dur > MELD_MAX_DUR))
    u["dur_ok"] = np.where(ok, dur, 0.0)
    return u


# ----------------------------------------------------------------------------
# 2. turn adjacency (who speaks right after whom)
# ----------------------------------------------------------------------------
def add_previous_speaker(u):
    """For each turn, the gender of the speaker of the turn just before it.

    Built on lines of 3+ words, the shared turn rule: a "mm-hmm" during
    someone's long turn is feedback, not a reply, and would otherwise split
    that turn in two and create two fake transitions.
    A turn = consecutive lines by the same person within one unit (a MELD
    scene or an AMI meeting). A transition is counted only when both turns
    have a known gender; a guest (U) in between breaks the chain rather than
    being skipped over."""
    t = u[u.n_words >= 3].sort_values(["unit_id", "utt_idx"], kind="stable")
    new_turn = (t.person_id != t.person_id.shift()) | (t.unit_id != t.unit_id.shift())
    turns = t[new_turn].copy()                          # first line of each turn

    # What "same gender by chance" means. A first version used headcount,
    # (n_g - 1) / (N - 1), and the result tracked floor share almost exactly
    # (r = 0.95 with volume parity on AMI): if the women talk most, a woman is
    # usually followed by a woman whatever the script does. So chance is
    # weighted by activity instead: the previous speaker is drawn from the
    # other people in proportion to how many turns each takes in the clip.
    #   expected = (turns by gender g - turns by me) / (all turns - turns by me)
    known = turns[turns.gender.isin(["F", "M"])]
    t_person = known.groupby(["clip_id", "person_id"]).size()
    t_gender = known.groupby(["clip_id", "gender"]).size()
    t_clip = known.groupby("clip_id").size()

    same_unit = turns.unit_id == turns.unit_id.shift()
    turns["prev_gender"] = turns.gender.shift().where(same_unit)
    turns = turns[turns.gender.isin(["F", "M"]) & turns.prev_gender.isin(["F", "M"])].copy()
    mine = t_person.reindex(pd.MultiIndex.from_arrays([turns.clip_id, turns.person_id])).to_numpy()
    my_gender = t_gender.reindex(pd.MultiIndex.from_arrays([turns.clip_id, turns.gender])).to_numpy()
    everyone = t_clip.reindex(turns.clip_id).to_numpy()
    turns["expected"] = (my_gender - mine) / (everyone - mine)
    turns["is_same"] = (turns.gender == turns.prev_gender).astype(int)
    return turns.groupby(["clip_id", "gender"]).agg(
        trans=("is_same", "size"), same=("is_same", "sum"),
        exp_same=("expected", "sum")).reset_index()


# ----------------------------------------------------------------------------
# 3. counts per clip x gender
# ----------------------------------------------------------------------------
def clip_gender_stats(u):
    k = u[u.gender.isin(["F", "M"])].copy()
    k["ge3"] = k.n_words >= 3
    k["w_ge3"] = k.n_words.where(k.ge3, 0)
    s = k.groupby(["clip_id", "gender"]).agg(
        n_persons=("person_id", "nunique"), n_utt=("n_words", "size"),
        n_words=("n_words", "sum"), n_utt_ge3=("ge3", "sum"),
        n_words_ge3=("w_ge3", "sum"), time=("dur_ok", "sum"),
        fem3=("fem3", "sum"), masc3=("masc3", "sum"), neut3=("neut3", "sum"),
        first=("first", "sum"), second=("second", "sum")).reset_index()

    # Chance is computed per clip, not per scene, on purpose: which characters
    # a writer puts into the same scene is part of what we want to measure.
    s = s.merge(add_previous_speaker(u), on=["clip_id", "gender"], how="left")
    s[["trans", "same"]] = s[["trans", "same"]].fillna(0).astype(int)
    s["exp_same"] = s.exp_same.fillna(0.0)
    return s


# ----------------------------------------------------------------------------
# 4. features from counts (works for a clip, a series, a corpus or a shard)
# ----------------------------------------------------------------------------
def features(F, M):
    """F and M are dicts (or Series) of summed counts for one group of clips."""
    def lg(a, b):
        return float(np.log(a / b)) if a > 0 and b > 0 else np.nan

    out = {
        "volume_log_parity": lg(F["n_words"] / F["n_persons"], M["n_words"] / M["n_persons"]),
        "volume_share_F": F["n_words"] / (F["n_words"] + M["n_words"]),
        "length_log_ratio": lg(F["n_words"] / F["n_utt"], M["n_words"] / M["n_utt"]),
        "length_log_ratio_ge3": lg(F["n_words_ge3"] / max(F["n_utt_ge3"], 1),
                                   M["n_words_ge3"] / max(M["n_utt_ge3"], 1)),
        "time_log_parity": lg(F["time"] / F["n_persons"], M["time"] / M["n_persons"]),
    }
    # same_gender_dialogue_share: observed minus expected, women minus men.
    if min(F["trans"], M["trans"]) > 0:
        exc_F = (F["same"] - F["exp_same"]) / F["trans"]
        exc_M = (M["same"] - M["exp_same"]) / M["trans"]
        out.update(same_gender_obs_F=F["same"] / F["trans"],
                   same_gender_obs_M=M["same"] / M["trans"],
                   same_gender_excess_diff=exc_F - exc_M)
    else:
        out.update(same_gender_obs_F=np.nan, same_gender_obs_M=np.nan,
                   same_gender_excess_diff=np.nan)

    # gendered_reference_ratio as documented: he/she-type pronouns as a share
    # of all pronouns that refer to people, in each gender's own speech.
    def pron(x):
        return x["fem3"] + x["masc3"] + x["neut3"] + x["first"] + x["second"]

    def gref(x):
        return x["fem3"] + x["masc3"]

    out["gref_share_diff"] = (gref(F) / pron(F) - gref(M) / pron(M)
                              if min(pron(F), pron(M)) > 0 else np.nan)
    # Who gets talked about. R_g = he/she-type pronouns pointing at gender g,
    # whoever says them. Per person, like every other parity feature:
    #   ref_log_parity = log((R_F / persons_F) / (R_M / persons_M))
    # Positive = women are referred to more per woman present. Proposed as the
    # replacement for gendered_reference_ratio (see the night-1 results note).
    R_F, R_M = F["fem3"] + M["fem3"], F["masc3"] + M["masc3"]
    out["ref_log_parity"] = lg(R_F / F["n_persons"], R_M / M["n_persons"])
    # Two variants kept only to show why they are not used:
    # (a) of the he/she references each gender makes, the share pointing at the
    #     other gender, women minus men. Algebraically this equals
    #     P(male ref | F speaker) + P(male ref | M speaker) - 1, i.e. it mostly
    #     measures how male the pool of people being talked about is.
    out["cross_ref_share_diff"] = (F["masc3"] / gref(F) - M["fem3"] / gref(M)
                                   if min(gref(F), gref(M)) > 0 else np.nan)
    # (b) "talked about rather than talking" (the old mention_ratio, without a
    #     cast list): references per word spoken. It is exactly
    #     ref_log_parity - volume_log_parity, so it re-imports F1 (and so C2).
    out["talked_about_log"] = lg(R_F / F["n_words"], R_M / M["n_words"])
    return out


def score_clips(stats, clips):
    """One row per clip in clips_provisional.csv, scored or NA with a reason.
    Every clip gets a row: the output contract forbids silent drops."""
    by_clip = dict(tuple(stats.groupby("clip_id")))
    rows = []
    for clip_id, info in clips.iterrows():
        row = {"clip_id": clip_id, "corpus": info.corpus, "series": info.series,
               "component": info.component, "label_status": info.label_status,
               "n_F": int(info.n_F), "n_M": int(info.n_M)}
        by = by_clip.get(clip_id, stats.iloc[:0]).set_index("gender")
        F = by.loc["F"] if "F" in by.index else None
        M = by.loc["M"] if "M" in by.index else None

        # clip-level validity, checked in order; the first failure is the reason
        reason = ""
        if info.label_status not in ("verified", "main_cast"):
            reason = f"gender labels not verified ({info.label_status})"
        elif F is None and M is None:
            reason = "no speaker with a known gender"
        elif F is None or M is None:
            reason = "single-gender clip"
        elif min(F.n_utt, M.n_utt) < MIN_UTT:
            reason = f"fewer than {MIN_UTT} utterances for one gender"
        row["status"] = "NA" if reason else "valid"
        row["na_reason"] = reason
        if F is not None and M is not None:
            row.update(features(F, M))
            row.update({f"{c}_{x}": int(v[c]) for x, v in (("F", F), ("M", M))
                        for c in ("n_utt", "n_words", "trans", "fem3", "masc3")})

        # feature-level validity on top of the clip-level one
        notes = []
        if not reason:
            if min(info.n_F, info.n_M) < 2:
                notes.append("same_gender_excess_diff: one gender has a single speaker")
            elif min(F.trans, M.trans) < MIN_TRANS:
                notes.append(f"same_gender_excess_diff: fewer than {MIN_TRANS} turn changes")
            pron_F = F[["fem3", "masc3", "neut3", "first", "second"]].sum()
            pron_M = M[["fem3", "masc3", "neut3", "first", "second"]].sum()
            if min(pron_F, pron_M) < MIN_PRON:
                notes.append(f"gref_share_diff: fewer than {MIN_PRON} person pronouns")
            if min(F.fem3 + F.masc3, M.fem3 + M.masc3) < MIN_GREF:
                notes.append(f"cross_ref_share_diff: fewer than {MIN_GREF} he/she pronouns")
            if min(F.fem3 + M.fem3, F.masc3 + M.masc3) < MIN_GREF:
                notes.append(f"ref_log_parity: fewer than {MIN_GREF} references to one gender")
                notes.append(f"talked_about_log: fewer than {MIN_GREF} references to one gender")
        for n in notes:
            row[n.split(":")[0]] = np.nan
        row["feature_na"] = "; ".join(notes)
        row["feature_version"] = FEATURE_VERSION
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. corpus-level pooled values with a bootstrap over series
# ----------------------------------------------------------------------------
FEATURES = ["volume_log_parity", "volume_share_F", "length_log_ratio",
            "length_log_ratio_ge3", "same_gender_excess_diff", "ref_log_parity",
            "gref_share_diff", "cross_ref_share_diff", "talked_about_log",
            "time_log_parity"]

# What each number is for. Only "equation" features go into the C3 model;
# agency, power and sentiment are added by their own scripts. The reasons are
# in components/C3_Text/C3_decision_log.md (D1 = F1 dropped, D2 = ref_log_parity).
ROLE = {
    "length_log_ratio": "equation",          # F3 mean_utterance_length
    "same_gender_excess_diff": "equation",   # same_gender_dialogue_share
    "ref_log_parity": "equation",            # replaces gendered_reference_ratio (D2)
    "length_log_ratio_ge3": "sensitivity",   # F3 without lines of 1-2 words
    "volume_log_parity": "check",            # F1, dropped from the equation (D1)
    "volume_share_F": "check",               # F1 as the spec words it
    "time_log_parity": "check",              # C2's quantity, only for the F1 check
    "gref_share_diff": "superseded",         # gendered_reference_ratio as documented
    "cross_ref_share_diff": "superseded",    # reduces to the referent mix
    "talked_about_log": "superseded",        # = ref_log_parity - F1
}
MODEL_FEATURES = [f for f, role in ROLE.items() if role == "equation"]


def pooled(stats, clip_ids):
    """Sum counts over a list of clips (repeats allowed) and compute features."""
    sub = stats[stats.clip_id.isin(set(clip_ids))]
    reps = pd.Series(clip_ids).value_counts()            # bootstrap repeats
    w = sub.clip_id.map(reps).to_numpy()[:, None]
    summed = pd.DataFrame(sub[STAT_COLS].to_numpy() * w, columns=STAT_COLS)
    summed["gender"] = sub.gender.to_numpy()
    tot = summed.groupby("gender")[STAT_COLS].sum()
    return features(tot.loc["F"], tot.loc["M"])


def corpus_summary(stats, scored, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for corpus, sc in scored[scored.status == "valid"].groupby("corpus"):
        base = sc.clip_id.tolist()
        # same-gender adjacency is only defined where both genders have 2+ people
        sg = sc[(sc.n_F >= 2) & (sc.n_M >= 2)].clip_id.tolist()
        series = sc.groupby("series").clip_id.apply(list)
        est = pooled(stats, base)
        est["same_gender_excess_diff"] = pooled(stats, sg)["same_gender_excess_diff"]

        # Resample whole series (MELD: seasons; AMI: the 4 meetings of one team),
        # never single clips: clips in a series share people and are not independent.
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(series), len(series))
            clips_b = [c for i in pick for c in series.iloc[i]]
            b = pooled(stats, clips_b)
            sg_b = [c for c in clips_b if c in set(sg)]
            b["same_gender_excess_diff"] = (pooled(stats, sg_b)["same_gender_excess_diff"]
                                            if sg_b else np.nan)
            boots.append(b)
        boots = pd.DataFrame(boots)
        for f in FEATURES:
            lo, hi = np.nanpercentile(boots[f], [2.5, 97.5])
            rows.append({"corpus": corpus, "feature": f, "role": ROLE[f],
                         "estimate": est[f],
                         "ci_low": lo, "ci_high": hi,
                         "n_clips": len(sg) if f == "same_gender_excess_diff" else len(base),
                         "n_series": len(series), "n_boot": n_boot,
                         "feature_version": FEATURE_VERSION})
    return pd.DataFrame(rows)


def n_components(clip_ids, u):
    """Person-connected groups among the given clips only."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = u[u.clip_id.isin(clip_ids)][["clip_id", "person_id"]].drop_duplicates()
    for c, p in pairs.itertuples(index=False):
        parent[find(("c", c))] = find(("p", p))
    return len({find(("c", c)) for c in clip_ids})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    d = args.root / "derived" / "c3"
    res = args.root / "equvid" / "components" / "C3_Text" / "results"
    res.mkdir(parents=True, exist_ok=True)

    u = pd.concat([pd.read_csv(d / f"utterances_{c}.csv", keep_default_na=False,
                               na_values={"start": [""], "end": [""]})
                   for c in ("meld", "ami")], ignore_index=True)
    clips = pd.read_csv(d / "clips_provisional.csv").set_index("clip_id")
    u = add_utterance_counts(u)

    stats = clip_gender_stats(u)
    stats.insert(2, "feature_version", FEATURE_VERSION)
    stats.to_csv(d / "c3_clip_gender_stats.csv", index=False)

    scored = score_clips(stats, clips)
    scored.to_csv(d / "c3_lexical_features.csv", index=False)

    summary = corpus_summary(stats, scored, args.boot)
    summary.to_csv(res / "c3_lexical_corpus_summary.csv", index=False)

    valid = scored[scored.status == "valid"]
    corr = []
    for corpus, sc in valid.groupby("corpus"):
        for a, b in [("volume_log_parity", "time_log_parity"),
                     ("volume_log_parity", "length_log_ratio"),
                     ("length_log_ratio", "length_log_ratio_ge3"),
                     ("volume_log_parity", "same_gender_excess_diff"),
                     ("volume_log_parity", "ref_log_parity"),
                     ("ref_log_parity", "cross_ref_share_diff"),
                     ("volume_log_parity", "talked_about_log"),
                     ("gref_share_diff", "ref_log_parity")]:
            x = sc[[a, b]].dropna()
            corr.append({"corpus": corpus, "a": a, "b": b, "n_clips": len(x),
                         "pearson": x[a].corr(x[b]),
                         "spearman": x[a].corr(x[b], method="spearman")})
    pd.DataFrame(corr).to_csv(res / "c3_lexical_correlations.csv", index=False)

    # ---- console report -------------------------------------------------
    print("clips by status:")
    print(scored.groupby(["corpus", "status", "na_reason"]).size().to_string())
    for corpus, sc in valid.groupby("corpus"):
        print(f"{corpus}: {len(sc)} valid clips, {sc.series.nunique()} series, "
              f"{n_components(sc.clip_id.tolist(), u)} person-connected groups")
        na = sc.feature_na.str.split("; ").explode()
        print("   feature-level NA:", na[na != ""].str.split(":").str[0].value_counts().to_dict())
    pd.set_option("display.width", 160)
    print(summary.pivot(index="feature", columns="corpus",
                        values=["estimate", "ci_low", "ci_high"]).round(3).to_string())
    print(pd.DataFrame(corr).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
