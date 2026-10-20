#!/usr/bin/env python3
"""C3 decision D6: is an off-the-shelf sentiment model good enough to keep F8?

    python sentiment_check.py          (run_c3_sentiment.bat runs it for you)

Needs   transformers + torch (setup_sentiment.bat installs them); internet once,
        to download the model (about 500 MB, cached afterwards)
Reads   MELD/{train,dev,test}_sent_emo.csv        MELD's human sentiment labels
        derived/c3/utterances_meld.csv, utterances_ami.csv
        derived/c3/c3_lexical_features.csv         which clips are scorable
Writes  derived/c3/c3_sentiment_scores.csv.gz      P(neg), P(neu), P(pos) per line
        derived/c3/c3_sentiment_stats.csv          sums per clip x gender (for shards)
        equvid/components/C3_Text/results/c3_sentiment_check.csv     the checks, PASS/FAIL
        equvid/components/C3_Text/results/c3_sentiment_summary.csv   per corpus, 95% CI

The model is cardiffnlp/twitter-roberta-base-sentiment-latest, used as it is
(no fine-tuning). Valence of a line = P(positive) - P(negative), from -1 to +1.
Feature: mean valence over lines of 3+ words, women minus men.

Criteria fixed before the run (D6 in C3_decision_log.md). F8 stays only if all
three pass; otherwise it is dropped and this evidence is what the log cites.
  1  accuracy  on MELD's human labels: macro-F1 >= 0.50 and accuracy >= 0.52
               (always answering "neutral" scores 0.470; VADER scored 0.430)
  2  swap      gender words swapped in MELD lines (he/she, man/woman, the six
               main names ...): the label changes on <= 3% of lines and the
               mean valence shift (female version minus male version) is
               within +-0.02. A bias metric must not inherit a biased tool.
  3  agree     on MELD, the model's women-minus-men difference lies inside the
               95% CI of the same difference computed from the human labels.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FEATURE_VERSION = "c3-sent-0.1"
LABELS = ("negative", "neutral", "positive")
HUMAN_VALUE = {"negative": -1, "neutral": 0, "positive": 1}
MIN_WORDS = 3            # the shared turn rule: 1-2 word lines are mostly feedback
CRITERIA = {"macro_f1": 0.50, "accuracy": 0.52, "flip_rate": 0.03, "swap_shift": 0.02}

# Gender-word pairs for the swap test. Only one-to-one pairs: "her" is left
# out because it maps to both "him" and "his". The main cast's names are
# swapped in fixed pairs, since a name also carries gender.
PAIRS = [("he", "she"), ("himself", "herself"), ("man", "woman"), ("men", "women"),
         ("boy", "girl"), ("boys", "girls"), ("husband", "wife"), ("husbands", "wives"),
         ("boyfriend", "girlfriend"), ("boyfriends", "girlfriends"), ("father", "mother"),
         ("dad", "mom"), ("son", "daughter"), ("sons", "daughters"), ("brother", "sister"),
         ("brothers", "sisters"), ("uncle", "aunt"), ("nephew", "niece"), ("king", "queen"),
         ("mr", "mrs"), ("gentleman", "lady"), ("gentlemen", "ladies"),
         ("grandfather", "grandmother"), ("grandpa", "grandma"),
         ("ross", "rachel"), ("chandler", "monica"), ("joey", "phoebe")]
SWAP = {m: f for m, f in PAIRS} | {f: m for m, f in PAIRS}
MALE_WORDS = {m for m, _ in PAIRS}
SWAP_RE = re.compile(r"\b(?:" + "|".join(sorted(SWAP, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def swap_gender(text):
    """Replace every listed gender word by its partner, keeping capitals."""
    def repl(m):
        w = m.group(0)
        out = SWAP[w.lower()]
        return out.upper() if w.isupper() and len(w) > 1 else out.capitalize() if w[0].isupper() else out
    return SWAP_RE.sub(repl, text)


def gender_word_balance(text):
    """(male words, female words) among the swappable words of a line."""
    words = [w.lower() for w in SWAP_RE.findall(text)]
    male = sum(w in MALE_WORDS for w in words)
    return male, len(words) - male


# ----------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------
def load_scorer(name=MODEL):
    """Returns a function: list of texts -> array (n, 3) of P(neg, neu, pos)."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        sys.exit(f"transformers/torch missing ({e}). Run "
                 "equvid/components/C3_Text/setup/setup_sentiment.bat once.")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    # read the label order from the model itself instead of assuming it
    order = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    cols = [order.index(lab) for lab in LABELS]

    def score(texts, batch=64):
        out = np.zeros((len(texts), 3))
        by_len = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding
        t0 = time.time()
        with torch.inference_mode():
            for s in range(0, len(by_len), batch):
                idx = by_len[s:s + batch]
                enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
                probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
                out[idx] = probs[:, cols]
                if (s // batch) % 100 == 0:
                    print(f"  {s + len(idx):,} of {len(texts):,} lines ({time.time() - t0:.0f} s)",
                          flush=True)
        return out
    return score


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def classification_report(gold, pred):
    acc = float(np.mean(gold == pred))
    f1s, rows = [], []
    for lab in LABELS:
        tp = np.sum((gold == lab) & (pred == lab))
        prec = tp / max(1, np.sum(pred == lab))
        rec = tp / max(1, np.sum(gold == lab))
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
        rows.append((lab, prec, rec, f1))
    return acc, float(np.mean(f1s)), rows


def series_diff(df, value, n_boot, rng):
    """Women-minus-men difference of the mean of `value`, pooled over lines,
    with a 95% CI from resampling whole series (MELD seasons, AMI teams)."""
    g = df.groupby(["series", "gender"])[value].agg(["sum", "count"]).unstack(fill_value=0)
    tot = np.column_stack([g[("sum", "F")], g[("count", "F")], g[("sum", "M")], g[("count", "M")]])
    s = tot.sum(axis=0)
    est = s[0] / s[1] - s[2] / s[3]
    b = tot[rng.integers(0, len(tot), (n_boot, len(tot)))].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = b[:, 0] / b[:, 1] - b[:, 2] / b[:, 3]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return est, lo, hi, s[0] / s[1], s[2] / s[3], int(s[1]), int(s[3])


def meld_labels(meld_dir):
    parts = []
    for split in ("train", "dev", "test"):
        d = pd.read_csv(meld_dir / f"{split}_sent_emo.csv", encoding="utf-8")
        parts.append(pd.DataFrame({
            "unit_id": [f"MELD:{split}:{x}" for x in d.Dialogue_ID],
            "utt_idx": d.Utterance_ID.astype(int),
            "human": d.Sentiment.str.strip().str.lower()}))
    return pd.concat(parts, ignore_index=True)


def main(score_fn=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    d = args.root / "derived" / "c3"
    res = args.root / "equvid" / "components" / "C3_Text" / "results"
    rng = np.random.default_rng(0)

    u = pd.concat([pd.read_csv(d / f"utterances_{c}.csv", keep_default_na=False,
                               na_values={"start": [""], "end": [""]})
                   for c in ("meld", "ami")], ignore_index=True)
    clips = pd.read_csv(d / "c3_lexical_features.csv", dtype=str, keep_default_na=False,
                        usecols=["clip_id", "series", "status"])
    valid = set(clips.clip_id[clips.status == "valid"])
    u = u.merge(meld_labels(args.root / "MELD"), on=["unit_id", "utt_idx"], how="left")

    # Lines to score: every MELD line (check 1 needs them all) and the AMI
    # lines that can enter the feature (known gender, scorable clip, 3+ words).
    feature_line = u.gender.isin(["F", "M"]) & u.clip_id.isin(valid) & (u.n_words >= MIN_WORDS)
    todo = u[(u.corpus == "MELD") | feature_line].copy()
    swap_src = todo[(todo.corpus == "MELD") & todo.text.str.contains(SWAP_RE)].copy()
    swap_src["swapped"] = swap_src.text.map(swap_gender)

    score = score_fn or load_scorer()
    print(f"scoring {len(todo):,} lines and {len(swap_src):,} gender-swapped MELD lines "
          f"with {MODEL} ...", flush=True)
    p = score(todo.text.tolist())
    todo[["p_neg", "p_neu", "p_pos"]] = p
    todo["valence"] = todo.p_pos - todo.p_neg
    todo["pred"] = np.array(LABELS)[p.argmax(axis=1)]
    ps = score(swap_src.swapped.tolist())
    swap_src["valence_swapped"] = ps[:, 2] - ps[:, 0]
    swap_src["pred_swapped"] = np.array(LABELS)[ps.argmax(axis=1)]
    swap_src = swap_src.join(todo[["valence", "pred"]])

    keep = ["corpus", "clip_id", "series", "unit_id", "utt_idx", "person_id", "gender",
            "n_words", "human", "p_neg", "p_neu", "p_pos", "valence", "pred"]
    todo[keep].to_csv(d / "c3_sentiment_scores.csv.gz", index=False, compression="gzip")

    checks = []
    # --- check 1: accuracy against MELD's human labels (all MELD lines)
    m = todo[(todo.corpus == "MELD") & todo.human.isin(LABELS)]
    acc, mf1, per_class = classification_report(m.human.to_numpy(), m.pred.to_numpy())
    majority = m.human.value_counts(normalize=True).max()
    checks += [("1 accuracy", "macro_f1", mf1, f">= {CRITERIA['macro_f1']}", mf1 >= CRITERIA["macro_f1"]),
               ("1 accuracy", "accuracy", acc, f">= {CRITERIA['accuracy']}", acc >= CRITERIA["accuracy"]),
               ("1 accuracy", "majority_class_accuracy", majority, "reference", None)]
    for lab, prec, rec, f1 in per_class:
        checks.append(("1 accuracy", f"{lab}_precision/recall/f1",
                       f"{prec:.3f}/{rec:.3f}/{f1:.3f}", "reference", None))
    try:   # VADER side by side, if installed (setup_c3.bat installed it)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        c = np.array([sia.polarity_scores(t)["compound"] for t in m.text])
        vpred = np.where(c >= 0.05, "positive", np.where(c <= -0.05, "negative", "neutral"))
        vacc, vf1, _ = classification_report(m.human.to_numpy(), vpred)
        checks += [("1 accuracy", "vader_accuracy", vacc, "reference", None),
                   ("1 accuracy", "vader_macro_f1", vf1, "reference", None)]
    except ImportError:
        pass

    # --- check 2: gender-word swap
    flip = float(np.mean(swap_src.pred != swap_src.pred_swapped))
    bal = swap_src.text.map(gender_word_balance)
    male_dom = bal.map(lambda b: b[0] > b[1])
    fem_dom = bal.map(lambda b: b[1] > b[0])
    # female version minus male version, whichever of the two is the original
    shift = pd.concat([(swap_src.valence_swapped - swap_src.valence)[male_dom],
                       (swap_src.valence - swap_src.valence_swapped)[fem_dom]])
    mean_shift = float(shift.mean()) if len(shift) else np.nan
    checks += [("2 swap", "label_flip_rate", flip, f"<= {CRITERIA['flip_rate']}",
                flip <= CRITERIA["flip_rate"]),
               ("2 swap", "mean_shift_female_minus_male", mean_shift,
                f"within +-{CRITERIA['swap_shift']}", abs(mean_shift) <= CRITERIA["swap_shift"]),
               ("2 swap", "lines_swapped", len(swap_src), "reference", None)]

    # --- check 3: the model's gender difference vs the human labels' (MELD)
    mf = todo[(todo.corpus == "MELD") & feature_line.reindex(todo.index) & todo.human.isin(LABELS)].copy()
    mf["human_value"] = mf.human.map(HUMAN_VALUE)
    h = series_diff(mf, "human_value", args.boot, rng)
    t = series_diff(mf, "valence", args.boot, rng)
    inside = h[1] <= t[0] <= h[2]
    checks += [("3 agree", "human_labels_F_minus_M", f"{h[0]:+.3f} [{h[1]:+.3f}, {h[2]:+.3f}]",
                "reference", None),
               ("3 agree", "model_F_minus_M", f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]",
                "inside the human 95% CI", inside)]

    verdict = all(c[4] for c in checks if c[4] is not None)
    checks.append(("decision", "keep_F8", "yes" if verdict else "no", "all checks pass", verdict))
    out = pd.DataFrame(checks, columns=["check", "measure", "value", "criterion", "passed"])
    out.to_csv(res / "c3_sentiment_check.csv", index=False)

    # --- the feature itself, per corpus (reported whatever the verdict)
    f = todo[feature_line.reindex(todo.index)]
    rows = []
    for corpus, x in f.groupby("corpus"):
        e = series_diff(x, "valence", args.boot, rng)
        rows.append({"corpus": corpus, "feature": "sentiment_valence_diff",
                     "role": "equation" if verdict else "dropped",
                     "estimate": e[0], "ci_low": e[1], "ci_high": e[2], "mean_F": e[3],
                     "mean_M": e[4], "lines_F": e[5], "lines_M": e[6],
                     "n_series": x.series.nunique(), "n_boot": args.boot,
                     "feature_version": FEATURE_VERSION})
    pd.DataFrame(rows).to_csv(res / "c3_sentiment_summary.csv", index=False)
    (f.groupby(["clip_id", "gender"]).valence.agg(["sum", "count"]).reset_index()
      .rename(columns={"count": "n"}).assign(feature_version=FEATURE_VERSION)
      .to_csv(d / "c3_sentiment_stats.csv", index=False))

    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nDECISION: keep F8" if verdict else "\nDECISION: drop F8 (see the failed checks above)")


if __name__ == "__main__":
    main()
