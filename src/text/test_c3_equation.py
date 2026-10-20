#!/usr/bin/env python3
"""Hand-checkable tests for c3_equation.py. Run:  python test_c3_equation.py

Every expected number is worked out in the comment above its assert, so each
test can be explained line by line in a viva. The last test runs the whole
pipeline on made-up clips: it checks the machinery without looking at MELD or
AMI.
"""
import math

import numpy as np

import c3_equation as eq
from c3_equation import IX, COLS


def one_shard(**counts):
    S = np.zeros((1, len(COLS)))
    for k, v in counts.items():
        S[0, IX[k]] = v
    return S


# Women and men with identical lines, verbs and references, except that the
# women reply to each other a little less often than chance:
#   words per line 100/10 = 10 for both              -> length 0
#   same-gender: (6 - 4)/20 - (8 - 4)/20 = -0.1     -> same -0.1
#   references to women 6 + 14 = 20, to men 4 + 16 = 20, 2 people each -> ref 0
#   agency 4/10 - 4/10 = 0, power 2/10 - 2/10 = 0
BASE = dict(F_persons=2, F_utt=10, F_words=100, F_fem3=6, F_masc3=4, F_trans=20,
            F_same=6, F_exp=4, F_ag_sum=4, F_ag_n=10, F_pw_sum=2, F_pw_n=10,
            M_persons=2, M_utt=10, M_words=100, M_fem3=14, M_masc3=16, M_trans=20,
            M_same=8, M_exp=4, M_ag_sum=4, M_ag_n=10, M_pw_sum=2, M_pw_n=10)


def test_features_by_hand():
    x = eq.features(one_shard(**BASE))[0]
    assert np.allclose(x, [0.0, -0.1, 0.0, 0.0, 0.0])


def test_operators_by_hand():
    S = one_shard(**BASE)
    # O_all at +0.50: half of the WOMEN's material in its least favourable form.
    #   lines:   words 0.5*100 + 0.5*10 = 55 over 10 lines -> 5.5 vs 10 -> log(0.55)
    #   replies: same 6 -> 3: (3 - 4)/20 - 0.2 = -0.25, a change of -0.15
    #            = -l * observed share 6/20 = -0.5 * 0.3
    #   refs:    references to women 20 -> 10 -> log((10/2) / (20/2)) = log(0.5)
    #   agency:  sum 0.5*4 - 0.5*10 = -3 -> -0.3 - 0.4 = -0.7 = -l * (mean + 1)
    #   power:   sum 0.5*2 - 0.5*10 = -4 -> -0.4 - 0.2 = -0.6
    x = eq.features(eq.apply_operator(S, "O_all", 0.5), check=False)[0]
    assert np.allclose(x, [math.log(0.55), -0.25, math.log(0.5), -0.7, -0.6])
    # O_len at -0.20 modifies the MEN only: 0.8*100 + 0.2*10 = 82 words over 10
    # lines -> log(10 / 8.2) = +0.198, and nothing else moves.
    x = eq.features(eq.apply_operator(S, "O_len", -0.2), check=False)[0]
    assert np.allclose(x, [math.log(10 / 8.2), -0.1, 0.0, 0.0, 0.0])
    # O_ref at -0.50: references to men 20 -> 10 -> log((20/2) / (10/2)) = log 2
    x = eq.features(eq.apply_operator(S, "O_ref", -0.5), check=False)[0]
    assert np.allclose(x, [0.0, -0.1, math.log(2), 0.0, 0.0])
    # the input is never changed in place
    assert np.array_equal(S, one_shard(**BASE))


def test_labels():
    # EuqAud's scale: 10 x the fraction; a single feature carries a fifth;
    # modifying women (+) gives a negative label, because + means women higher.
    assert float(eq.label("O_all", 0.25)) == -2.5
    assert float(eq.label("O_all", -0.5)) == 5.0
    assert math.isclose(float(eq.label("O_len", -0.5)), 1.0)
    assert float(eq.label("none", 0.3)) == 0.0


def test_ridge_by_hand():
    # One feature with y = 3x exactly. After dividing x by its RMS, the design
    # column has mean square 1, so the closed form gives b = (1 + alpha)^-1 x the
    # least-squares slope: alpha = 1 halves the coefficient, to 1.5 in raw units.
    x = np.array([[1.0], [-2.0], [3.0], [0.5]])
    y = 3 * x[:, 0]
    assert math.isclose(eq.fit(x, y, 1, 1.0)["coef"][0], 1.5)
    assert math.isclose(eq.fit(x, y, 1, 1e-12)["coef"][0], 3.0)
    try:                                        # optional: same as scikit-learn
        from sklearn.linear_model import Ridge
    except ImportError:
        return
    z = x / np.sqrt(np.mean(x ** 2))
    skl = Ridge(alpha=1.0 * len(x), fit_intercept=False).fit(z, y).coef_[0]
    assert math.isclose(skl, eq.fit(x, y, 1, 1.0)["beta"][0])


def test_fit_recovers_a_planted_equation():
    # y = 2*L - 3*R + 0.5*L^2 + 1.5*S*A, no noise: a degree-2 fit with almost no
    # penalty must give these numbers back, every other term ~0, and f(0) = 0.
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 5))
    y = 2 * X[:, 0] - 3 * X[:, 2] + 0.5 * X[:, 0] ** 2 + 1.5 * X[:, 1] * X[:, 3]
    m = eq.fit(X, y, 2, 1e-12)
    coef = dict(zip(map(eq.term_name, m["terms"]), m["coef"]))
    want = {"L": 2, "R": -3, "L^2": 0.5, "S*A": 1.5}
    for k, v in coef.items():
        assert math.isclose(v, want.get(k, 0.0), abs_tol=1e-6), (k, v)
    # the closed form in raw units equals the model fitted on scaled inputs
    D = eq.design(X / m["scale"], m["terms"])
    assert np.allclose(D @ m["beta"], eq.raw_score(m, X))
    assert eq.raw_score(m, np.zeros((1, 5)))[0] == 0.0


def test_normalise_and_tiers():
    # anchors: +4 (men's material halved) and -5 (women's halved) both become 5
    s = eq.normalise([2.0, -5.0, -2.0, 30.0, 0.0], u_pos=4.0, u_neg=-5.0)
    assert np.allclose(s, [2.5, -5.0, -2.0, 10.0, 0.0])        # 30 is cut at 10
    t = eq.tier([1.0, -3.0, 5.0, 4.99, -7.0], 1.5)
    assert list(t) == ["Neutral", "Moderate", "Strong", "Moderate", "Strong"]
    # a noise floor above 5 leaves no Moderate tier at all
    assert list(eq.tier([5.5, 7.0], 6.0)) == ["Neutral", "Strong"]
    assert list(eq.direction([0.3, -0.3, 0.0])) == ["women higher", "men higher", "parity"]


def test_noise_floor():
    # 21 shard scores -1.0 ... +1.0, median 0; |deviations| sorted are
    # 0, 0.1, 0.1, ..., 1.0, 1.0 and the 95th percentile (position 19 of 20) is 1.0.
    # N = 59 clips: the correction is sqrt((59 - 30) / 58) = sqrt(0.5) -> 1.414,
    # rounded up to the next 0.5 -> 1.5.
    scores = np.round(np.arange(-10, 11) * 0.1, 10)
    q95, t1 = eq.noise_floor(scores, np.array(["X"] * 21), {"X": 59})
    assert math.isclose(q95, 1 / math.sqrt(0.5)) and t1 == 1.5


def test_assumption_statistics():
    # Durbin-Watson: residuals that flip sign every row -> 3 x 2^2 / 4 = 3
    assert eq.durbin_watson(np.array([1.0, -1.0, 1.0, -1.0])) == 3.0
    # VIF: two features with r = 0.8 -> 1 / (1 - 0.64) = 2.78 each
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]], dtype=float)
    assert np.allclose(eq.vif(X), 1 / (1 - 0.64))
    # Breusch-Pagan: e^2 is exactly x, so R^2 = 1 and LM = n = 4 on 1 df
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    lm, df, p = eq.breusch_pagan(np.sqrt(x[:, 0]), x)
    assert math.isclose(lm, 4.0) and df == 1


def test_contributions_add_up():
    # A degree-1 equation f = 2L - 1R, anchors +4 / -4: at L = 0.5, R = 0.1,
    # f = 0.9 and the score is 5 * 0.9 / 4 = 1.125 = 1.25 (from L) - 0.125 (from R).
    m = {"terms": [(0,), (1,), (2,), (3,), (4,)], "coef": np.array([2.0, 0, -1.0, 0, 0])}
    x = np.array([0.5, 0.0, 0.1, 0.0, 0.0])
    c = eq.contributions(m, x, 4.0, -4.0)
    assert math.isclose(c["mean_utterance_length"], 1.25)
    assert math.isclose(c["ref_log_parity"], -0.125)
    assert math.isclose(sum(c.values()), float(eq.normalise(0.9, 4.0, -4.0)))


def test_pipeline_on_made_up_clips():
    # 2 made-up corpora x 40 clips with random but plausible counts. Checks the
    # machinery end to end: shards, operators, the grid, the anchors, the sign.
    rng = np.random.default_rng(3)
    C = np.zeros((80, len(COLS)))
    for g in ("F", "M"):
        C[:, IX[g + "_persons"]] = rng.integers(1, 4, 80)
        C[:, IX[g + "_utt"]] = rng.integers(40, 120, 80)
        C[:, IX[g + "_words"]] = C[:, IX[g + "_utt"]] * rng.uniform(6, 12, 80)
        C[:, IX[g + "_fem3"]] = rng.integers(2, 15, 80)
        C[:, IX[g + "_masc3"]] = rng.integers(2, 15, 80)
        C[:, IX[g + "_trans"]] = rng.integers(20, 60, 80)
        C[:, IX[g + "_same"]] = (C[:, IX[g + "_trans"]] * rng.uniform(0.2, 0.6, 80)).round()
        C[:, IX[g + "_exp"]] = C[:, IX[g + "_trans"]] * 0.4
        for k in ("ag", "pw"):
            C[:, IX[f"{g}_{k}_n"]] = rng.integers(10, 30, 80)
            C[:, IX[f"{g}_{k}_sum"]] = C[:, IX[f"{g}_{k}_n"]] * rng.uniform(-0.2, 0.4, 80)
    pools = {"X": np.arange(40), "Y": np.arange(40, 80)}
    idx, corp = eq.draw_shards(pools, np.random.default_rng(0))
    assert idx.shape == (2 * eq.SHARDS_PER_CORPUS, eq.CLIPS_PER_SHARD)
    assert all(len(set(r)) == eq.CLIPS_PER_SHARD for r in idx)   # no repeats in a shard
    B = eq.build(C, idx, corp)
    assert B["n_dropped"] == 0 and np.isfinite(B["dX"]).all()
    # 1 unmodified + 6 operators x 20 levels per shard
    assert len(B["y"]) == len(B["S"]) * (1 + 6 * 20)
    cv, d, a = eq.loco_cv(B)
    assert cv.chosen.sum() == 1 and len(cv) == len(eq.DEGREES) * len(eq.ALPHAS)
    m = eq.fit(B["dX"][B["train"]], B["y"][B["train"]], d, a)
    u_pos, u_neg = eq.anchors(m, B)
    s = eq.normalise(eq.raw_score(m, B["dX"]), u_pos, u_neg)
    top = (B["op"] == "O_all") & np.isclose(B["lev"], 0.5)
    assert math.isclose(np.median(s[top]), -5.0)       # women's material halved -> -5
    assert np.all(s[B["op"] == "none"] == 0.0)          # nothing injected -> 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
