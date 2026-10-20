#!/usr/bin/env python3
"""C3 (EuqVid-Text): fit, freeze, normalise and tier the C3 equation.

    python c3_equation.py              the full run, about a minute
    python c3_equation.py --dry-run    runs every step on the real tables but prints
                                       and writes no result: an integrity check only
    python c3_equation.py --root "D:/SLIIT/4th Year/Y4 S1/Research Project"

The design is decision D7 in components/C3_Text/C3_decision_log.md. It was
pushed to GitHub before the first real run. Do not change anything here without
a new decision-log entry: a design changed after seeing results is exploratory.

Reads   derived/c3/c3_clip_gender_stats.csv    lexical counts per clip x gender
        derived/c3/c3_agency_power_stats.csv   agency/power sums per clip x gender
        derived/c3/c3_lexical_features.csv     scorable clips, series, n_F, n_M and
                                               the clip-level lexical features
        derived/c3/c3_agency_power_clip.csv    clip-level agency and power
Writes  derived/c3/c3_clip_to_shard_provisional.csv   shard_id, clip_id
        derived/c3/c3_shard_features.csv              the unmodified shards, scored
        derived/c3/c3_augmented_rows.csv.gz           every augmented variant
        derived/c3/c3_clip_scores.csv                 one row per clip (output contract)
        results/c3_equation.json                      THE FROZEN EQUATION
        results/c3_equation_cv.csv                    degree x alpha, leave-one-corpus-out
        results/c3_equation_calibration.csv           held-out score per operator x level
        results/c3_equation_checks.csv                fit, monotonicity, detection limits,
                                                      assumption checks
        results/c3_equation_bootstrap.csv             1,000 refits: how stable each number is
        results/c3_equation_scores.csv                MELD and AMI: score, 95% CIs, tier,
                                                      and what each feature contributes

The method in five steps
  1. Rows are shards: 30 episodes or meetings pooled into a small "dataset".
  2. Each shard is biased on purpose by a known amount: a fraction l of one
     gender's material is replaced by its least favourable form, in EuqAud's 5%
     steps.
  3. The label is that known amount, on EuqAud's scale: 10 x the fraction.
  4. Ridge regression learns label ~ the change this made to the five features.
     There is no intercept, so a dataset at parity (all five features 0) scores
     exactly 0 -- parity, not the corpus's natural state, is the zero point.
  5. The fitted polynomial is written out in raw feature units and frozen.

Augmentation is calibration, not validation: it fixes the weights and the
scale. A high fit on augmented rows only shows the equation inverts the
operators (M2's lesson 2). Validity needs evidence with its own ground truth.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from agency_power_features import FEATURE_VERSION as AP_VERSION, MIN_FRAMES
from lexical_features import MIN_GREF, MIN_TRANS, MIN_UTT
from wordlists import FEATURE_VERSION as LEX_VERSION

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]
EQ_VERSION = "c3-eq-0.1"

# ----------------------------------------------------------------------------
# Design constants, all fixed in D7 before the first real run
# ----------------------------------------------------------------------------
FEATURES = [  # (name in the specification, column in the clip tables, symbol)
    ("mean_utterance_length", "length_log_ratio", "L"),
    ("same_gender_dialogue_share", "same_gender_excess_diff", "S"),
    ("ref_log_parity", "ref_log_parity", "R"),
    ("agency_score_mean", "agency", "A"),
    ("power_score_mean", "power", "P"),
]
NAMES = [f[0] for f in FEATURES]
COLUMNS = [f[1] for f in FEATURES]
SYMBOLS = [f[2] for f in FEATURES]

SHARDS_PER_CORPUS = 100      # 2 corpora x 100 = M2's 200 shards
CLIPS_PER_SHARD = 30         # M2's proposal: 30 episodes or meetings
STEP = 0.05                  # EuqAud's 5% increments ...
LEVELS = np.round(STEP * np.arange(1, 11), 2)          # ... from 0.05 to 0.50
SIGNED_LEVELS = np.concatenate([-LEVELS[::-1], LEVELS])  # + modifies women (M2)
SINGLE_OPS = ["O_len", "O_same", "O_ref", "O_agency", "O_power"]  # same order as FEATURES
OPERATORS = ["O_all"] + SINGLE_OPS
LABEL_SCALE = 10.0           # EuqAud: score = 10 x the fraction reduced
ANCHOR_LEVEL = 0.50          # the largest level trained on ...
ANCHOR_SCORE = LABEL_SCALE * ANCHOR_LEVEL                # ... lands at a score of 5
STRONG_FROM = ANCHOR_SCORE   # Strong tier from |score| = 5 (D7)
DEGREES = [1, 2]
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]                  # EuqAud's alpha grid
TIE = 0.02                   # held-out RMSE within 2% of the best counts as a tie
SHARD_SEED, BOOT_SEED = 0, 1
N_BOOT = 1000

# Counts each gender contributes to a clip. Pooling clips into a shard, a
# series or a corpus is then just adding rows, so every level of pooling uses
# one code path -- and a ratio of sums, never a mean of ratios.
PER_GENDER = ["persons", "utt", "words", "fem3", "masc3", "trans", "same", "exp",
              "ag_sum", "ag_n", "pw_sum", "pw_n"]
COLS = [f"{g}_{k}" for g in ("F", "M") for k in PER_GENDER]
IX = {c: i for i, c in enumerate(COLS)}


# ----------------------------------------------------------------------------
# 1. counts per clip
# ----------------------------------------------------------------------------
def load_tables(root):
    d = root / "derived" / "c3"
    keep = dict(keep_default_na=False, na_values=[""])   # status holds the text "NA"
    return (pd.read_csv(d / "c3_lexical_features.csv", **keep),
            pd.read_csv(d / "c3_agency_power_clip.csv", **keep),
            pd.read_csv(d / "c3_clip_gender_stats.csv"),
            pd.read_csv(d / "c3_agency_power_stats.csv"))


def clip_matrix(lex, st, aps):
    """One row per scorable clip, one column per count in COLS."""
    clips = (lex[lex.status == "valid"][["clip_id", "corpus", "series", "n_F", "n_M"]]
             .sort_values("clip_id").reset_index(drop=True))
    s = st.set_index(["clip_id", "gender"])
    a = (aps[aps.variant.isin(["agency", "power"])]
         .pivot_table(index=["clip_id", "gender"], columns="variant",
                      values=["sum", "n"], aggfunc="sum", fill_value=0))
    # Same-gender adjacency counts only where both genders have 2+ people, as in
    # the night-1 corpus summary: a lone woman has nobody of her own gender to
    # follow, so her turns could only dilute the feature towards zero.
    eligible = ((clips.n_F >= 2) & (clips.n_M >= 2)).to_numpy()
    C = np.zeros((len(clips), len(COLS)))
    for i, cid in enumerate(clips.clip_id):
        for g in ("F", "M"):
            r = s.loc[(cid, g)]
            C[i, IX[f"{g}_persons"]] = r.n_persons
            C[i, IX[f"{g}_utt"]] = r.n_utt
            C[i, IX[f"{g}_words"]] = r.n_words
            C[i, IX[f"{g}_fem3"]] = r.fem3
            C[i, IX[f"{g}_masc3"]] = r.masc3
            if eligible[i]:
                C[i, IX[f"{g}_trans"]] = r.trans
                C[i, IX[f"{g}_same"]] = r.same
                C[i, IX[f"{g}_exp"]] = r.exp_same
            if (cid, g) in a.index:
                C[i, IX[f"{g}_ag_sum"]] = a.loc[(cid, g), ("sum", "agency")]
                C[i, IX[f"{g}_ag_n"]] = a.loc[(cid, g), ("n", "agency")]
                C[i, IX[f"{g}_pw_sum"]] = a.loc[(cid, g), ("sum", "power")]
                C[i, IX[f"{g}_pw_n"]] = a.loc[(cid, g), ("n", "power")]
    clips["same_gender_eligible"] = eligible
    return clips, C


# ----------------------------------------------------------------------------
# 2. the five features from summed counts (shard, clip or corpus)
# ----------------------------------------------------------------------------
def features(S, check=True):
    """The five equation features, one row per row of S (women minus men, or
    log women over men; positive = women higher). With check=True a feature is
    NaN when one gender has too little material for it, using the same minimum
    counts as the clip-level scripts."""
    def v(k):
        return S[:, IX[k]]
    with np.errstate(divide="ignore", invalid="ignore"):
        length = np.log((v("F_words") / v("F_utt")) / (v("M_words") / v("M_utt")))
        same = ((v("F_same") - v("F_exp")) / v("F_trans")
                - (v("M_same") - v("M_exp")) / v("M_trans"))
        R_F, R_M = v("F_fem3") + v("M_fem3"), v("F_masc3") + v("M_masc3")
        ref = np.log((R_F / v("F_persons")) / (R_M / v("M_persons")))
        agency = v("F_ag_sum") / v("F_ag_n") - v("M_ag_sum") / v("M_ag_n")
        power = v("F_pw_sum") / v("F_pw_n") - v("M_pw_sum") / v("M_pw_n")
    X = np.column_stack([length, same, ref, agency, power])
    if not check:
        return X
    enough = np.column_stack([
        np.minimum(v("F_utt"), v("M_utt")) >= MIN_UTT,
        np.minimum(v("F_trans"), v("M_trans")) >= MIN_TRANS,
        np.minimum(R_F, R_M) >= MIN_GREF,
        np.minimum(v("F_ag_n"), v("M_ag_n")) >= MIN_FRAMES,
        np.minimum(v("F_pw_n"), v("M_pw_n")) >= MIN_FRAMES,
    ])
    return np.where(enough, X, np.nan)


# ----------------------------------------------------------------------------
# 3. augmentation: bias a shard on purpose, by a known amount
# ----------------------------------------------------------------------------
def apply_operator(S, op, v):
    """Copy of the summed counts S in which a fraction l = |v| of group g's
    material is replaced by the least favourable form it can take. g = women
    when v > 0, men when v < 0 (M2's convention: positive levels modify women).

    One rule for all five operators: a group mean m becomes (1 - l)*m + l*min.
    When the minimum is 0 this is EuqAud's own rule, "multiply by 0.95, 0.90 ...".
      O_len     a line becomes a one-word reaction            (min 1 word per line)
      O_same    a reply from her own gender comes from the
                other gender instead                          (min share 0)
      O_ref     a reference to her (she / her) is removed     (min 0)
      O_agency  her verb is replaced by one scoring -1        (min -1)
      O_power   the same, for power                           (min -1)
    O_all applies all five together, as EuqAud reduced all its features together.
    Applied in expectation to the counts, so there is no random draw here."""
    S = S.copy()
    lv, g = abs(v), ("F" if v > 0 else "M")
    for o in (SINGLE_OPS if op == "O_all" else [op]):
        if o == "O_len":
            S[:, IX[g + "_words"]] = (1 - lv) * S[:, IX[g + "_words"]] + lv * S[:, IX[g + "_utt"]]
        elif o == "O_same":
            S[:, IX[g + "_same"]] *= 1 - lv
        elif o == "O_ref":                     # references TO g, whoever makes them
            k = "fem3" if g == "F" else "masc3"
            S[:, IX["F_" + k]] *= 1 - lv
            S[:, IX["M_" + k]] *= 1 - lv
        else:
            k = "ag" if o == "O_agency" else "pw"
            S[:, IX[f"{g}_{k}_sum"]] = ((1 - lv) * S[:, IX[f"{g}_{k}_sum"]]
                                        - lv * S[:, IX[f"{g}_{k}_n"]])
    return S


def label(op, v):
    """Ground truth on EuqAud's scale: 10 x the fraction modified, negative when
    women's material was modified (positive score = women higher). A
    single-feature operator moves one feature of five, so it carries a fifth:
    replacing a fraction l of any ONE kind of material counts the same for every
    feature. This exchange rate is a stated convention (D7), not a finding."""
    op, v = np.asarray(op), np.asarray(v, dtype=float)
    share = np.where(op == "O_all", 1.0, 1.0 / len(SINGLE_OPS))
    return np.where(op == "none", 0.0, -LABEL_SCALE * v * share)


def augment(S_base):
    """Every shard under every operator and signed level; the unmodified shard first."""
    n = len(S_base)
    meta = [("none", 0.0)] + [(op, v) for op in OPERATORS for v in SIGNED_LEVELS]
    S_aug = np.vstack([S_base] + [apply_operator(S_base, op, v) for op, v in meta[1:]])
    op = np.repeat([m[0] for m in meta], n)
    lev = np.round(np.repeat([m[1] for m in meta], n), 2)
    shard = np.tile(np.arange(n), len(meta))
    return S_aug, op, lev, shard, label(op, lev)


def build(C, idx, corp):
    """Shards from clip rows, their features, and all augmented variants.
    The model is trained on the CHANGE in the features (dX), not on the features:
    each shard keeps its own natural disparity, and only what the operator added
    is labelled. So the corpora's own disparities never enter the fit."""
    S = C[idx].sum(axis=1)
    X = features(S)
    ok = np.isfinite(X).all(axis=1)
    S, X, corp, idx = S[ok], X[ok], corp[ok], idx[ok]
    S_aug, op, lev, shard, y = augment(S)
    dX = features(S_aug, check=False) - X[shard]
    return dict(S=S, X=X, corp=corp, idx=idx, n_dropped=int((~ok).sum()), op=op,
                lev=lev, shard=shard, y=y, dX=dX, train=op != "none",
                row_corpus=corp[shard])


# ----------------------------------------------------------------------------
# 4. polynomial Ridge without intercept, written out in raw units
# ----------------------------------------------------------------------------
def terms_for(degree, p=len(FEATURES)):
    t = [(i,) for i in range(p)]
    if degree == 2:
        t += list(itertools.combinations_with_replacement(range(p), 2))
    return t


def design(X, terms):
    return np.column_stack([X[:, list(t)].prod(axis=1) for t in terms])


def fit(dX, y, degree, alpha):
    """Ridge: minimise mean((y - D b)^2) + alpha * |b|^2, closed form
        b = (D'D / n + alpha I)^-1 D'y / n.
    The inputs are divided by their root mean square first, so the penalty
    treats the five features alike; they are NOT centred and there is no
    intercept, so f(0) = 0 survives. (scikit-learn's Ridge minimises the sum,
    not the mean, of squares: its alpha = this alpha x n_rows.)
    Returns the coefficient of every term in raw feature units, i.e. the
    closed-form equation."""
    scale = np.sqrt(np.mean(dX ** 2, axis=0))
    terms = terms_for(degree, dX.shape[1])
    D = design(dX / scale, terms)
    n, k = D.shape
    beta = np.linalg.solve(D.T @ D / n + alpha * np.eye(k), D.T @ y / n)
    raw = beta / np.array([scale[list(t)].prod() for t in terms])
    return {"terms": terms, "coef": raw, "degree": degree, "alpha": alpha,
            "scale": scale, "beta": beta}


def raw_score(eq, X):
    """The closed-form equation f(x) = sum of coefficient x term, in raw units."""
    return design(X, eq["terms"]) @ eq["coef"]


def anchors(eq, B):
    """EuqAud's sign-specific min-max bounds. Lower bound: parity, where f = 0
    by construction. Upper bounds: the median f of the shards with every
    feature of one gender reduced by the largest trained level (50%)."""
    f = raw_score(eq, B["dX"])
    top = B["op"] == "O_all"
    u_neg = float(np.median(f[top & np.isclose(B["lev"], ANCHOR_LEVEL)]))   # women modified
    u_pos = float(np.median(f[top & np.isclose(B["lev"], -ANCHOR_LEVEL)]))  # men modified
    if not u_pos > 0 > u_neg:
        raise ValueError(f"anchors have the wrong sign: {u_pos}, {u_neg}")
    return u_pos, u_neg


def normalise(f, u_pos, u_neg):
    """Map f onto -10 ... +10: parity -> 0, the anchors -> +5 / -5 (= 10 x 50%).
    Beyond +/-5 the score extrapolates past the trained range; it is cut at 10."""
    f = np.asarray(f, dtype=float)
    s = np.where(f >= 0, ANCHOR_SCORE * f / u_pos, ANCHOR_SCORE * f / abs(u_neg))
    return np.clip(s, -10.0, 10.0)


# ----------------------------------------------------------------------------
# 5. tiers
# ----------------------------------------------------------------------------
def noise_floor(scores, corp, pool_sizes):
    """How far two 30-clip datasets from the same source differ by chance.

    Deviation of each unmodified shard's score from its corpus median, divided
    by the finite-population correction sqrt((N - 30) / (N - 1)): shards are
    drawn without replacement from N clips, which makes them look more alike
    than independent datasets would be. Returns the 95th percentile and that
    value rounded UP to the next 0.5, which is the Neutral boundary."""
    devs = []
    for c in np.unique(corp):
        s = scores[corp == c]
        N = pool_sizes[c]
        fpc = np.sqrt((N - CLIPS_PER_SHARD) / (N - 1))
        devs.append(np.abs(s - np.median(s)) / fpc)
    q95 = float(np.percentile(np.concatenate(devs), 95))
    return q95, max(0.5, float(np.ceil(q95 / 0.5) * 0.5))


def tier(score, t1):
    """Neutral below the noise floor t1; Strong from 5 (a disparity as large as
    half of one gender's material in every C3 feature); Moderate between."""
    a = np.abs(np.asarray(score, dtype=float))
    t1 = np.asarray(t1, dtype=float)           # one value, or one per bootstrap draw
    return np.where(a < t1, "Neutral",
                    np.where(a >= np.maximum(t1, STRONG_FROM), "Strong", "Moderate"))


def direction(score):
    s = np.asarray(score, dtype=float)
    return np.where(s > 0, "women higher", np.where(s < 0, "men higher", "parity"))


# ----------------------------------------------------------------------------
# 6. assumption checks, written out so each line can be explained
# ----------------------------------------------------------------------------
def r_squared(target, X):
    """R^2 of an OLS regression of target on X plus an intercept."""
    A = np.column_stack([np.ones(len(X)), X])
    resid = target - A @ np.linalg.lstsq(A, target, rcond=None)[0]
    return 1 - resid @ resid / np.sum((target - target.mean()) ** 2)


def vif(X):
    """Variance inflation factor of each column: 1 / (1 - R^2 of it on the others)."""
    return np.array([1 / (1 - r_squared(X[:, j], np.delete(X, j, axis=1)))
                     for j in range(X.shape[1])])


def durbin_watson(e):
    """sum of squared successive differences / sum of squares. About 2 when
    neighbouring residuals are unrelated; below 2 when they move together."""
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


def breusch_pagan(e, X):
    """Koenker's studentised Breusch-Pagan test: n x R^2 of e^2 on X, against a
    chi-square with one degree of freedom per column of X. A small p means the
    residual spread changes with X (heteroscedasticity)."""
    lm = len(e) * r_squared(e ** 2, X)
    return float(lm), X.shape[1], float(sps.chi2.sf(lm, X.shape[1]))


# ----------------------------------------------------------------------------
# 7. shards and the bootstrap
# ----------------------------------------------------------------------------
def draw_shards(pools, rng):
    """CLIPS_PER_SHARD clips per shard, no repeats inside a shard, overlap
    between shards (M2's proposal), drawn within each corpus."""
    idx, corp = [], []
    for c in sorted(pools):
        pool = pools[c]
        for _ in range(SHARDS_PER_CORPUS):
            idx.append(pool[rng.choice(len(pool), CLIPS_PER_SHARD, replace=False)])
            corp.append(c)
    return np.array(idx), np.array(corp)


def corpus_rows(clips):
    return {c: np.flatnonzero(clips.corpus.to_numpy() == c) for c in sorted(clips.corpus.unique())}


def bootstrap(C, clips, frozen, frozen_anchors, n_boot, rng):
    """Resample SERIES (MELD seasons, AMI meeting series) with replacement,
    rebuild the shards, re-augment, refit the frozen degree and alpha. Never
    rows and never shards: neither is independent of its neighbours."""
    series = {c: [np.flatnonzero(((clips.corpus == c) & (clips.series == s)).to_numpy())
                  for s in sorted(clips.series[clips.corpus == c].unique())]
              for c in sorted(clips.corpus.unique())}
    out = []
    for b in range(n_boot):
        pools = {}
        for c, groups in series.items():
            pick = rng.integers(0, len(groups), len(groups))
            pools[c] = np.concatenate([groups[i] for i in pick])
        idx, corp = draw_shards(pools, rng)
        B = build(C, idx, corp)
        eq = fit(B["dX"][B["train"]], B["y"][B["train"]], frozen["degree"], frozen["alpha"])
        u_pos, u_neg = anchors(eq, B)
        base = normalise(raw_score(eq, B["X"]), u_pos, u_neg)
        q95, t1 = noise_floor(base, B["corp"], {c: len(p) for c, p in pools.items()})
        rec = {"iteration": b, "u_pos": u_pos, "u_neg": u_neg, "noise_q95": q95,
               "neutral_below": t1, "shards_dropped": B["n_dropped"]}
        rec.update({term_name(t): c for t, c in zip(eq["terms"], eq["coef"])})
        for c, pool in pools.items():
            x = features(C[pool].sum(axis=0, keepdims=True))
            rec[f"{c}_score_frozen"] = float(normalise(raw_score(frozen, x), *frozen_anchors)[0])
            rec[f"{c}_score_refit"] = float(normalise(raw_score(eq, x), u_pos, u_neg)[0])
        out.append(rec)
    return pd.DataFrame(out)


def term_name(t):
    if len(t) == 1:
        return SYMBOLS[t[0]]
    return SYMBOLS[t[0]] + "^2" if t[0] == t[1] else SYMBOLS[t[0]] + "*" + SYMBOLS[t[1]]


def equation_text(eq):
    return "f = " + " ".join(f"{c:+.4f}*{term_name(t)}" for t, c in zip(eq["terms"], eq["coef"]))


# ----------------------------------------------------------------------------
# 8. model choice and the held-out checks
# ----------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def loco_cv(B):
    """Leave one corpus out: fit on MELD's shards, predict AMI's, and the reverse.
    With two corpora this is the strictest split available, and no person is in
    both halves (M2's identity firewall: MELD is one person-connected group)."""
    corpora = sorted(set(B["corp"]))
    rows = []
    for d in DEGREES:
        for a in ALPHAS:
            r = {"degree": d, "alpha": a, "n_terms": len(terms_for(d))}
            for test in corpora:
                tr = B["train"] & (B["row_corpus"] != test)
                te = B["train"] & (B["row_corpus"] == test)
                eq = fit(B["dX"][tr], B["y"][tr], d, a)
                r[f"rmse_heldout_{test}"] = rmse(raw_score(eq, B["dX"][te]), B["y"][te])
            r["rmse_heldout_mean"] = float(np.mean([r[f"rmse_heldout_{c}"] for c in corpora]))
            rows.append(r)
    cv = pd.DataFrame(rows)
    # Pre-declared rule: lowest mean held-out RMSE; anything within 2% of it is a
    # tie, and ties go to the simpler model -- degree 1 first, then more shrinkage.
    cv["within_2pct_of_best"] = cv.rmse_heldout_mean <= (1 + TIE) * cv.rmse_heldout_mean.min()
    pick = (cv[cv.within_2pct_of_best].sort_values(["degree", "alpha"], ascending=[True, False])
            .iloc[0])
    cv["chosen"] = (cv.degree == pick.degree) & (cv.alpha == pick.alpha)
    return cv, int(pick.degree), float(pick.alpha)


def heldout_scores(B, degree, alpha):
    """Score of every training row under the model fitted on the OTHER corpus,
    normalised with that model's own anchors."""
    out = np.full(len(B["y"]), np.nan)
    for test in sorted(set(B["corp"])):
        tr = B["train"] & (B["row_corpus"] != test)
        te = B["row_corpus"] == test
        eq = fit(B["dX"][tr], B["y"][tr], degree, alpha)
        Btr = {k: B[k][tr] for k in ("dX", "op", "lev")}
        out[te] = normalise(raw_score(eq, B["dX"][te]), *anchors(eq, Btr))
    return out


def fit_metrics(y, s):
    return {"n": len(y), "rmse": rmse(s, y), "mae": float(np.mean(np.abs(s - y))),
            "r2": float(1 - np.sum((y - s) ** 2) / np.sum((y - y.mean()) ** 2)),
            "pearson": float(np.corrcoef(y, s)[0, 1]),
            "spearman": float(sps.spearmanr(y, s)[0])}


def calibration_table(B, held):
    """Median held-out score (and the 5th-95th percentile over shards) at every
    operator x level, per corpus. This is what fig5 draws."""
    d = pd.DataFrame({"corpus": B["row_corpus"], "operator": B["op"], "level": B["lev"],
                      "label": B["y"], "score": held})
    d = d[d.operator != "none"]
    g = d.groupby(["corpus", "operator", "level"])
    t = g.agg(label=("label", "first"), n_shards=("score", "size"),
              score_median=("score", "median"),
              score_p05=("score", lambda s: np.percentile(s, 5)),
              score_p95=("score", lambda s: np.percentile(s, 95))).reset_index()
    return t


def checks_table(B, held, final, t1, pool_sizes):
    rows = []

    def add(check, scope, item, value, note=""):
        rows.append({"check": check, "scope": scope, "item": item, "value": value, "note": note})

    tr = B["train"]
    # (a) EuqAud's Table 1 statistics, but on the HELD-OUT corpus
    for c in sorted(set(B["corp"])):
        for op in ["all operators"] + OPERATORS:
            m = tr & (B["row_corpus"] == c) & ((B["op"] == op) if op != "all operators" else True)
            for k, v in fit_metrics(B["y"][m], held[m]).items():
                add("heldout_fit", f"{c} held out", f"{op}: {k}", v,
                    "score vs injected label; sensitivity to the operators, not validity")
    # (b) monotonicity of the held-out median score in the level
    cal = calibration_table(B, held)
    for (c, op), d in cal.groupby(["corpus", "operator"]):
        d = d.sort_values("level")
        rho = sps.spearmanr(d.level, d.score_median)[0]
        add("monotonicity", f"{c} held out", f"{op}: spearman(level, median score)", float(rho),
            "-1 = the median score falls at every step as women's material is reduced")
        add("monotonicity", f"{c} held out", f"{op}: strictly decreasing",
            bool(np.all(np.diff(d.score_median.to_numpy()) < 0)))
        m = tr & (B["row_corpus"] == c) & (B["op"] == op)
        per_shard = pd.DataFrame({"shard": B["shard"][m], "lev": B["lev"][m], "s": held[m]})
        mono = per_shard.sort_values("lev").groupby("shard").s.apply(
            lambda s: bool(np.all(np.diff(s.to_numpy()) <= 0)))
        add("monotonicity", f"{c} held out", f"{op}: share of shards never going up",
            float(mono.mean()))
    # (c) detection limit of each feature in a 30-clip dataset
    for c in sorted(set(B["corp"])):
        N = pool_sizes[c]
        fpc = np.sqrt((N - CLIPS_PER_SHARD) / (N - 1))
        Xc = B["X"][B["corp"] == c]
        for j, (name, _, sym) in enumerate(FEATURES):
            sd = float(np.std(Xc[:, j], ddof=1) / fpc)
            op = SINGLE_OPS[j]
            add("detection", c, f"{name}: natural SD across shards", sd,
                "unmodified 30-clip shards, finite-population corrected")
            min_level = np.nan
            for lv in LEVELS:
                m = tr & (B["row_corpus"] == c) & (B["op"] == op) & np.isclose(np.abs(B["lev"]), lv)
                shift = float(np.median(np.abs(B["dX"][m, j])))
                if lv == ANCHOR_LEVEL:
                    add("detection", c, f"{name}: shift at 50%", shift)
                    add("detection", c, f"{name}: shift at 50% / natural SD", shift / sd)
                if np.isnan(min_level) and shift >= 1.96 * sd:
                    min_level = float(lv)
            add("detection", c, f"{name}: smallest level whose shift exceeds 1.96 SD",
                min_level, "NaN = not even at 50%")
    # (d) assumption checks on the final fit, as EuqAud reported them
    dX, y = B["dX"][tr], B["y"][tr]
    resid = y - raw_score(final, dX)
    D = design(dX / final["scale"], final["terms"])
    for name, v in zip(NAMES, vif(dX)):
        add("assumption", "training design", f"VIF {name}", float(v), "linear terms of the change dX")
    for name, v in zip(NAMES, vif(B["X"])):
        add("assumption", "unmodified shards", f"VIF {name}", float(v),
            "the features as they occur naturally")
    if final["degree"] == 2:
        add("assumption", "training design", "largest VIF over all polynomial terms",
            float(vif(D).max()), "squares and products track their own linear terms")
    order = np.lexsort((B["lev"][tr], B["op"][tr], B["shard"][tr], B["row_corpus"][tr]))
    add("assumption", "final fit", "Durbin-Watson", durbin_watson(resid[order]),
        "rows ordered corpus, shard, operator, level: below 2 = misfit that runs along the levels")
    lm, df, p = breusch_pagan(resid, D)
    add("assumption", "final fit", "Breusch-Pagan LM", lm, f"df = {df}")
    add("assumption", "final fit", "Breusch-Pagan p", p, "small = residual spread changes with x")
    add("assumption", "final fit", "residual skewness", float(sps.skew(resid)))
    add("assumption", "final fit", "residual excess kurtosis", float(sps.kurtosis(resid)))
    add("assumption", "final fit", "D'Agostino normality p", float(sps.normaltest(resid)[1]))
    add("assumption", "final fit", "in-sample RMSE", rmse(raw_score(final, dX), y))
    # (e) M3 task 3: how each feature's change relates to the label
    for j, name in enumerate(NAMES):
        add("pearson", "training rows", f"{name} change vs label",
            float(np.corrcoef(dX[:, j], y)[0, 1]),
            "a single-feature operator moves one feature, O_all moves all five")
    add("tiers", "noise floor", "Neutral below", t1)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 9. the real corpora and the per-clip output contract
# ----------------------------------------------------------------------------
def contributions(eq, x, u_pos, u_neg):
    """What each feature adds to the (unclipped) score. Squares and products,
    if any, are reported together as one extra item."""
    f = float(raw_score(eq, x[None, :])[0])
    k = ANCHOR_SCORE / (u_pos if f >= 0 else abs(u_neg))
    out = {name: 0.0 for name in NAMES}
    out["squares_and_products"] = 0.0
    for t, c in zip(eq["terms"], eq["coef"]):
        part = k * c * np.prod(x[list(t)])
        out[NAMES[t[0]] if len(t) == 1 else "squares_and_products"] += float(part)
    return out


def clip_scores(lex, ap_clip, eq, u_pos, u_neg):
    t = lex[["clip_id", "corpus", "series", "status", "na_reason"] + COLUMNS[:3]].merge(
        ap_clip[["clip_id", "agency", "power"]], on="clip_id", how="left")
    X = t[COLUMNS].to_numpy(dtype=float)
    ok = (t.status == "valid").to_numpy() & np.isfinite(X).all(axis=1)
    s = normalise(raw_score(eq, np.nan_to_num(X)), u_pos, u_neg)
    reason = []
    for i, r in t.iterrows():
        if r.status != "valid":
            reason.append(r.na_reason)
        elif not ok[i]:
            miss = [n for n, col in zip(NAMES, COLUMNS) if not np.isfinite(r[col])]
            reason.append("too little material in one clip for: " + ", ".join(miss))
        else:
            reason.append("")
    out = t[["clip_id", "corpus", "series"]].copy()
    out["status"] = np.where(ok, "valid", "NA")
    out["na_reason"] = reason
    out["score"] = np.where(ok, s, np.nan)
    out["note"] = np.where(ok, "one clip is noisy: read scores at shard or corpus level", "")
    out["feature_version"] = EQ_VERSION
    return out


def fingerprint(C):
    """Platform-independent fingerprint of the input counts (CSV line endings
    differ between Windows and Linux; the numbers do not)."""
    return hashlib.sha256(np.round(C, 6).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# 10. main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every step but print and write no result")
    args = ap.parse_args()
    t0 = time.time()
    d = args.root / "derived" / "c3"
    res = args.root / "equvid" / "components" / "C3_Text" / "results"
    n_boot = 3 if args.dry_run else args.boot

    lex, ap_clip, st, aps = load_tables(args.root)
    clips, C = clip_matrix(lex, st, aps)
    pools = corpus_rows(clips)
    pool_sizes = {c: len(p) for c, p in pools.items()}

    # ---- shards, augmentation ------------------------------------------------
    idx, corp = draw_shards(pools, np.random.default_rng(SHARD_SEED))
    shard_ids = np.array([f"{c}:shard_{i % SHARDS_PER_CORPUS:03d}" for i, c in enumerate(corp)])
    B = build(C, idx, corp)
    kept_ids = shard_ids[np.isfinite(features(C[idx].sum(axis=1))).all(axis=1)]

    # ---- choose degree and alpha, then fit the final equation on both corpora --
    cv, degree, alpha = loco_cv(B)
    final = fit(B["dX"][B["train"]], B["y"][B["train"]], degree, alpha)
    u_pos, u_neg = anchors(final, B)

    # Check the frozen closed form against the fitted model it was written from.
    D = design(B["dX"] / final["scale"], final["terms"])
    reproduction = float(np.max(np.abs(D @ final["beta"] - raw_score(final, B["dX"]))))
    assert reproduction < 1e-9, reproduction
    assert raw_score(final, np.zeros((1, len(FEATURES))))[0] == 0.0   # parity scores 0

    base_scores = normalise(raw_score(final, B["X"]), u_pos, u_neg)
    q95, t1 = noise_floor(base_scores, B["corp"], pool_sizes)
    held = heldout_scores(B, degree, alpha)
    checks = checks_table(B, held, final, t1, pool_sizes)

    # ---- the real corpora ----------------------------------------------------
    corpus_x = {c: features(C[p].sum(axis=0, keepdims=True))[0] for c, p in pools.items()}
    boot = bootstrap(C, clips, final, (u_pos, u_neg), n_boot,
                     np.random.default_rng(BOOT_SEED))

    # ---- scores, CIs and tiers for MELD and AMI -----------------------------------
    score_rows = []
    for c, x in corpus_x.items():
        f = float(raw_score(final, x[None, :])[0])
        s = float(normalise(f, u_pos, u_neg))
        frozen_b, refit_b = boot[f"{c}_score_frozen"], boot[f"{c}_score_refit"]
        sh = base_scores[B["corp"] == c]
        row = {"corpus": c, "n_clips": pool_sizes[c],
               "n_series": int(clips.series[clips.corpus == c].nunique()),
               "raw_f": f, "score": s, "tier": str(tier(s, t1)), "direction": str(direction(s)),
               "beyond_calibrated_range": abs(s) > ANCHOR_SCORE,
               "ci_low_data": float(np.percentile(frozen_b, 2.5)),
               "ci_high_data": float(np.percentile(frozen_b, 97.5)),
               "ci_low_total": float(np.percentile(refit_b, 2.5)),
               "ci_high_total": float(np.percentile(refit_b, 97.5)),
               "same_tier_share_data": float(np.mean(tier(frozen_b, t1) == tier(s, t1))),
               "same_tier_share_total": float(np.mean(tier(refit_b, boot.neutral_below)
                                                      == tier(s, t1))),
               "shard_score_median": float(np.median(sh)),
               "shard_score_p05": float(np.percentile(sh, 5)),
               "shard_score_p95": float(np.percentile(sh, 95)),
               "n_boot": n_boot}
        row.update({f"x_{n}": float(v) for n, v in zip(NAMES, x)})
        row.update({f"contrib_{k}": v for k, v in contributions(final, x, u_pos, u_neg).items()})
        row["equation_version"] = EQ_VERSION
        score_rows.append(row)
    scores = pd.DataFrame(score_rows)

    # ---- acceptance checks for the machinery, fixed in D7 before the first run -----
    acc = []
    for c in sorted(pools):
        r2 = checks.value[(checks.scope == f"{c} held out")
                          & (checks.item == "O_all: r2")].iloc[0]
        acc.append(("1 held-out R^2 of O_all >= 0.90", c, float(r2), bool(r2 >= 0.90)))
        mono = checks[(checks.scope == f"{c} held out")
                      & checks.item.str.endswith("strictly decreasing")].value
        acc.append(("2 median score falls at every level, all six operators", c,
                    int(sum(bool(v) for v in mono)), bool(all(bool(v) for v in mono))))
    for t, coef in zip(final["terms"], final["coef"]):
        if len(t) == 1:
            share = float(np.mean(boot[term_name(t)] > 0))
            acc.append(("3 linear coefficient positive in >= 95% of refits",
                        NAMES[t[0]], share, bool(coef > 0 and share >= 0.95)))
    acc = pd.DataFrame(acc, columns=["item", "scope", "value", "passed"])
    checks = pd.concat([checks, pd.DataFrame({
        "check": "acceptance", "scope": acc.scope, "item": acc.item, "value": acc.value,
        "note": np.where(acc.passed, "passed", "FAILED")})], ignore_index=True)
    for c in sorted(pools):     # the noise floor per corpus, for the record
        s = base_scores[B["corp"] == c]
        fpc = np.sqrt((pool_sizes[c] - CLIPS_PER_SHARD) / (pool_sizes[c] - 1))
        checks.loc[len(checks)] = ["tiers", c, "noise q95 (this corpus alone)",
                                   float(np.percentile(np.abs(s - np.median(s)) / fpc, 95)), ""]

    # ---- coefficient stability over the bootstrap ------------------------------------
    stab = []
    frozen_vals = {term_name(t): c for t, c in zip(final["terms"], final["coef"])}
    frozen_vals.update({"u_pos": u_pos, "u_neg": u_neg, "noise_q95": q95, "neutral_below": t1})
    for k, v in frozen_vals.items():
        b = boot[k].to_numpy()
        stab.append({"item": k, "frozen": v, "boot_mean": float(b.mean()),
                     "boot_sd": float(b.std(ddof=1)),
                     "ci_low": float(np.percentile(b, 2.5)),
                     "ci_high": float(np.percentile(b, 97.5)),
                     "same_sign_share": float(np.mean(np.sign(b) == np.sign(v))),
                     "n_boot": n_boot})
    stab = pd.DataFrame(stab)

    # ---- the output tables ---------------------------------------------------------------
    c2s = pd.DataFrame({"shard_id": np.repeat(shard_ids, CLIPS_PER_SHARD),
                        "clip_id": clips.clip_id.to_numpy()[idx.ravel()]})
    sf = pd.DataFrame(B["X"], columns=NAMES)
    sf.insert(0, "corpus", B["corp"])
    sf.insert(0, "shard_id", kept_ids)
    sf["raw_f"] = raw_score(final, B["X"])
    sf["score"] = base_scores
    sf["tier"] = tier(base_scores, t1)
    sf["feature_version"] = EQ_VERSION
    aug = pd.DataFrame({
        "variant_id": [f"{kept_ids[s]}|{o}|{v:+.2f}" for s, o, v in zip(B["shard"], B["op"], B["lev"])],
        "corpus": B["row_corpus"], "operator": B["op"], "level": B["lev"], "label": B["y"]})
    for j, n in enumerate(NAMES):
        aug[f"d_{n}"] = B["dX"][:, j]
    aug["raw_f"] = raw_score(final, B["dX"])
    aug["score_final"] = normalise(aug.raw_f, u_pos, u_neg)
    aug["score_heldout"] = held
    cs = clip_scores(lex, ap_clip, final, u_pos, u_neg)
    cal = calibration_table(B, held)
    frozen = {
        "equation_version": EQ_VERSION,
        "design": "decision D7, components/C3_Text/C3_decision_log.md",
        "features": [{"symbol": s, "name": n, "clip_table_column": c}
                     for n, c, s in FEATURES],
        "degree": degree, "alpha": alpha,
        "alpha_convention": "per-row mean squared error + alpha*|b|^2 on RMS-scaled inputs; "
                            "scikit-learn's alpha = this alpha x number of rows",
        "terms": [{"term": term_name(t), "coefficient": float(c)}
                  for t, c in zip(final["terms"], final["coef"])],
        "equation_raw": equation_text(final),
        "normalisation": {
            "rule": "score = clip(5*f/u_pos if f >= 0 else 5*f/|u_neg|, -10, 10)",
            "u_pos": u_pos, "u_neg": u_neg, "anchor_level": ANCHOR_LEVEL,
            "anchor_score": ANCHOR_SCORE,
            "meaning": "0 = parity; +/-5 = every C3 feature of one gender reduced by 50%; "
                       "beyond +/-5 the score extrapolates past the trained range"},
        "tiers": {"neutral_below": t1, "noise_q95": q95, "strong_from": STRONG_FROM,
                  "rule": "Neutral |s| < neutral_below; Strong |s| >= strong_from; "
                          "Moderate between; the sign gives the direction"},
        "training": {"corpora": sorted(pools), "clips": pool_sizes,
                     "shards_per_corpus": SHARDS_PER_CORPUS,
                     "clips_per_shard": CLIPS_PER_SHARD, "shards_used": len(B["S"]),
                     "operators": OPERATORS, "levels": SIGNED_LEVELS.tolist(),
                     "training_rows": int(B["train"].sum()),
                     "shard_seed": SHARD_SEED, "bootstrap_seed": BOOT_SEED,
                     "bootstrap_iterations": n_boot,
                     "bootstrap_shards_dropped": int(boot.shards_dropped.sum()),
                     "selection": "leave-one-corpus-out RMSE; ties within 2% go to "
                                  "degree 1, then the larger alpha"},
        "acceptance": [{"check": i, "scope": s, "value": v, "passed": p}
                       for i, s, v, p in acc.itertuples(index=False)],
        "inputs": {"lexical_feature_version": LEX_VERSION,
                   "agency_power_feature_version": AP_VERSION,
                   "count_fingerprint": fingerprint(C)},
    }
    text = json.dumps(frozen, indent=2)          # fails here if anything is not plain data

    if args.dry_run:   # every step has run; show only that the machinery holds together
        X_aug = features(augment(B["S"])[0], check=False)
        tables = {"cv": cv, "calibration": cal, "checks": checks, "bootstrap": stab,
                  "scores": scores, "shards": sf, "augmented": aug, "clip scores": cs}
        print("DRY RUN: integrity only, no result is printed or written")
        print(f"  clips per corpus: {pool_sizes}")
        print(f"  shards kept: {len(B['S'])} of {len(idx)} (dropped {B['n_dropped']})")
        print(f"  augmented rows: {len(B['y'])}, training rows: {int(B['train'].sum())}")
        print(f"  non-finite values: base {int((~np.isfinite(B['X'])).sum())}, "
              f"augmented {int((~np.isfinite(X_aug)).sum())}, "
              f"held-out scores {int((~np.isfinite(held)).sum())}")
        print(f"  closed form reproduces the fitted model to {reproduction:.1e}")
        print(f"  corpus features finite: "
              f"{ {c: bool(np.isfinite(x).all()) for c, x in corpus_x.items()} }")
        print(f"  bootstrap iterations run: {len(boot)}, all finite: "
              f"{bool(np.isfinite(boot.select_dtypes('number').to_numpy()).all())}")
        print("  table rows:", {k: len(v) for k, v in tables.items()})
        print(f"  clip scores: {int((cs.status == 'valid').sum())} valid, "
              f"{int((cs.status == 'NA').sum())} NA; JSON {len(text)} characters")
        print(f"  input fingerprint {fingerprint(C)}; time {time.time() - t0:.1f} s")
        return

    # ---- write everything ----------------------------------------------------------------
    res.mkdir(parents=True, exist_ok=True)
    c2s.to_csv(d / "c3_clip_to_shard_provisional.csv", index=False)
    sf.to_csv(d / "c3_shard_features.csv", index=False)
    aug.to_csv(d / "c3_augmented_rows.csv.gz", index=False)
    cs.to_csv(d / "c3_clip_scores.csv", index=False)
    cv.to_csv(res / "c3_equation_cv.csv", index=False)
    cal.to_csv(res / "c3_equation_calibration.csv", index=False)
    checks.to_csv(res / "c3_equation_checks.csv", index=False)
    stab.to_csv(res / "c3_equation_bootstrap.csv", index=False)
    scores.to_csv(res / "c3_equation_scores.csv", index=False)
    (res / "c3_equation.json").write_text(text, encoding="utf-8")

    # ---- console report ---------------------------------------------------------------
    pd.set_option("display.width", 170)
    print("Leave-one-corpus-out grid (held-out RMSE in score units):")
    print(cv.round(4).to_string(index=False))
    print(f"\nChosen: degree {degree}, alpha {alpha}")
    print("Frozen equation (raw units):", equation_text(final))
    print(f"Anchors: u_pos {u_pos:.4f}, u_neg {u_neg:.4f}; noise q95 {q95:.3f} -> "
          f"Neutral below {t1}, Strong from {STRONG_FROM}")
    print("\nScores:")
    print(scores[["corpus", "score", "tier", "direction", "ci_low_data", "ci_high_data",
                  "ci_low_total", "ci_high_total", "same_tier_share_total"]]
          .round(3).to_string(index=False))
    print("\nContributions to the score:")
    print(scores[["corpus"] + [c for c in scores.columns if c.startswith("contrib_")]]
          .round(3).to_string(index=False))
    print("\nCoefficient stability:")
    print(stab.round(4).to_string(index=False))
    print("\nAcceptance checks (fixed in D7):")
    print(acc.round(4).to_string(index=False))
    show = checks[checks.check.isin(["assumption", "pearson", "tiers"])
                  | checks.item.str.contains("all operators: r2|strictly|smallest level")]
    print("\nChecks:")
    print(show.round(4).to_string(index=False))
    print(f"\nwrote results to {res} and tables to {d} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
