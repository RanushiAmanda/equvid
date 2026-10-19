#!/usr/bin/env python3
"""Hand-checkable tests for the C3 feature code. Run:  python test_text_features.py

Each test uses a toy conversation small enough to work out on paper, so the
expected numbers in the asserts can be explained line by line in a viva.
"""
import math

import pandas as pd

from lexical_features import FEATURES, MODEL_FEATURES, ROLE, add_previous_speaker, features
from wordlists import pronoun_counts, tokens


def test_adjacency():
    # One scene, two women (A, B) and two men (C, D), eight turns:
    #   A B A C D C A B
    # Turns into a woman that follow someone: B<-A same, A<-B same, A<-C cross,
    # B<-A same  -> 4 transitions, 3 same-gender.
    # Turns into a man: C<-A cross, D<-C same, C<-D same -> 3 transitions, 2 same.
    # Turns per person: A 3, B 2, C 2, D 1; women 5, men 3, all 8.
    # Chance for a turn by person i of gender g = (turns_g - turns_i) / (8 - turns_i)
    #   A: (5-3)/(8-3) = 0.4   B: (5-2)/(8-2) = 0.5
    #   C: (3-2)/(8-2) = 1/6   D: (3-1)/(8-1) = 2/7
    # Expected same, women: B + A + A + B = 0.5 + 0.4 + 0.4 + 0.5 = 1.8
    # Expected same, men:   C + D + C = 1/6 + 2/7 + 1/6
    seq = "ABACDCAB"
    g = {"A": "F", "B": "F", "C": "M", "D": "M"}
    u = pd.DataFrame({"clip_id": "T:1", "unit_id": "T:1", "utt_idx": range(8),
                      "person_id": list(seq), "gender": [g[p] for p in seq],
                      "n_words": 5})
    r = add_previous_speaker(u).set_index("gender")
    assert (r.loc["F", "trans"], r.loc["F", "same"]) == (4, 3)
    assert (r.loc["M", "trans"], r.loc["M", "same"]) == (3, 2)
    assert math.isclose(r.loc["F", "exp_same"], 1.8)
    assert math.isclose(r.loc["M", "exp_same"], 1 / 6 + 2 / 7 + 1 / 6)

    # A short reply ("Yeah." = 1 word) is feedback, not a turn: removing it must
    # merge A's two lines into one turn, leaving A C = one transition.
    u2 = pd.DataFrame({"clip_id": "T:2", "unit_id": "T:2", "utt_idx": range(4),
                       "person_id": list("ABAC"), "gender": list("FFFM"),
                       "n_words": [8, 1, 8, 8]})
    r2 = add_previous_speaker(u2).set_index("gender")
    assert "F" not in r2.index and r2.loc["M", "trans"] == 1


def test_parity_features():
    # Two women say 300 words in 30 lines; one man says 200 words in 10 lines.
    # Per person: women 150 words each, the man 200 -> log(150/200) = log(0.75).
    # The plain share would say women 60% -- the per-person view says the
    # opposite, because there are two of them. This is why parity is per person.
    F = dict(n_words=300, n_persons=2, n_utt=30, n_words_ge3=280, n_utt_ge3=20,
             time=100.0, trans=10, same=4, exp_same=3.0,
             fem3=4, masc3=6, neut3=2, first=20, second=8)
    M = dict(n_words=200, n_persons=1, n_utt=10, n_words_ge3=200, n_utt_ge3=10,
             time=90.0, trans=10, same=2, exp_same=2.0,
             fem3=10, masc3=2, neut3=1, first=10, second=7)
    f = features(F, M)
    assert math.isclose(f["volume_log_parity"], math.log(0.75))
    assert math.isclose(f["volume_share_F"], 0.6)
    assert math.isclose(f["length_log_ratio"], math.log((300 / 30) / (200 / 10)))
    assert math.isclose(f["same_gender_excess_diff"], (4 - 3) / 10 - (2 - 2) / 10)
    # references to women: 4 + 10 = 14 over 2 women; to men: 6 + 2 = 8 over 1 man
    assert math.isclose(f["ref_log_parity"], math.log((14 / 2) / (8 / 1)))
    # the identity used in the results note: talked_about = ref - volume
    assert math.isclose(f["talked_about_log"],
                        f["ref_log_parity"] - f["volume_log_parity"])
    # cross_ref_share_diff = P(male ref | F) + P(male ref | M) - 1
    assert math.isclose(f["cross_ref_share_diff"], 6 / 10 + 2 / 12 - 1)


def test_pronouns():
    t = tokens("She told him that HER brother and his wife can't come; they'll call you.")
    assert pronoun_counts(t) == {"fem3": 2, "masc3": 2, "neut3": 1, "first": 0, "second": 1}


def test_roles():
    # Every computed number has a stated role, and F1 is not in the equation (D1).
    assert set(ROLE) == set(FEATURES)
    assert "volume_log_parity" not in MODEL_FEATURES
    assert MODEL_FEATURES == ["length_log_ratio", "same_gender_excess_diff", "ref_log_parity"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
