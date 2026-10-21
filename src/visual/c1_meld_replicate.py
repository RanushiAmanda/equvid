"""
C1 -- MELD framing gaps: reproduce the dev numbers, replicate on train.

Standard library only (like the other scripts in src/visual/).

    python src\\c1_meld_replicate.py out\\meld_clips.csv

WHY THIS SCRIPT EXISTS
----------------------
Addendum 6 of RESULTS.md reports framing gaps between women and men on the
MELD dev split.  That analysis was done ad hoc and is not in the repo.  This
script re-derives it from the extractor's output (out\\meld_clips.csv) and
runs the train split through the identical procedure, so the dev numbers
can be checked and the replication can be re-run by anyone.

THE PROCEDURE (fixed before train was looked at)
------------------------------------------------
1. Unit = one single-face clip (status == "single_face").  The visible
   face is TAKEN TO BE the speaker, so the speaker's gender (a hand-written
   map of six characters, never a model) labels the face.  That assumption
   has not been spot-checked: a reaction shot would break it.
2. gap = mean(women) - mean(men).  Negative = women smaller / fewer close-ups.
3. close-up = the clip's mean_rel_area >= CUT.  CUT is the 75th percentile
   of DEV single-face mean_rel_area, then FROZEN.  The absolute 0.08 cut
   tuned on AVA gives zero close-ups on MELD (faces are a quarter the size).
4. 95% interval: bootstrap that resamples whole EPISODES (season, episode)
   with replacement, 2,000 draws.  Clips from one episode share a set, a
   director and a camera plan, so clips are not independent.
5. presence score = 10 * clip(ln(nF/nM) / ln 4.00, -1, +1): the frozen
   closed form of Visual-EuqVid v1, applied to the single-face counts.

THREE EXTRA CHECKS (run after the plan above, so they are labelled as such)
---------------------------------------------------------------------------
A. Train restricted to episodes that do NOT appear in dev.  MELD's dev and
   train splits are not episode-disjoint (the same episode can appear in
   both); this removes the overlap.
B. Within-episode gap (post hoc): women and men are compared inside the same
   episode, then averaged with weight nF*nM/(nF+nM).  It removes
   between-episode differences (season, director) from the comparison.
C. Character permutation: MELD has 3 women and 3 men.  Relabel which three of
   the six characters are "F" in all C(6,3)=20 ways and see how unusual the
   real labelling is.  With 20 labellings the smallest possible two-sided p is
   2/20 = 0.10, so a 3-v-3 corpus cannot show a gender effect that is
   separable from "these six characters are framed differently".
"""
import csv
import itertools
import math
import random
import sys

B = 2000                      # bootstrap draws
SEED = 20261020
DIV_PRESENCE = math.log(4.00)  # divisor of the presence dimension (Addendum 5)


# ---------------------------------------------------------------- loading
def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "single_face":
                continue
            rows.append({
                "split": r["split"],
                "ep": "%s-%s" % (int(r["season"]), int(r["episode"])),
                "gender": r["gender"],
                "speaker": r["speaker"],
                "area": float(r["mean_rel_area"]),
                "dist": float(r["mean_centre_dist"]),
            })
    return rows


def all_episodes(path):
    """Every episode that has ANY dev clip (not only single-face ones)."""
    eps = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "dev":
                eps.add("%s-%s" % (int(r["season"]), int(r["episode"])))
    return eps


def percentile(values, p):
    """Linear-interpolation percentile (same rule as numpy's default)."""
    x = sorted(values)
    k = (len(x) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return x[lo] + (x[hi] - x[lo]) * (k - lo)


# ---------------------------------------------------------------- statistics
# Per-episode aggregate, so a bootstrap draw is a sum over chosen episodes.
# order: nF, areaF, closeF, distF, nM, areaM, closeM, distM
def episode_table(rows):
    agg = {}
    for r in rows:
        a = agg.setdefault(r["ep"], [0.0] * 8)
        o = 0 if r["gender"] == "F" else 4
        a[o] += 1
        a[o + 1] += r["area"]
        a[o + 2] += r["closeup"]
        a[o + 3] += r["dist"]
    return list(agg.values())


def stats(t):
    nF, aF, cF, dF, nM, aM, cM, dM = t
    if nF == 0 or nM == 0:
        return None
    parity = math.log(nF / nM)
    return {
        "area_gap": aF / nF - aM / nM,
        "closeup_gap": cF / nF - cM / nM,
        "dist_gap": dF / nF - dM / nM,
        "presence_parity": parity,
        "presence_score": 10 * max(-1.0, min(1.0, parity / DIV_PRESENCE)),
    }


def interval(draws, key):
    v = sorted(d[key] for d in draws if d is not None)
    return percentile(v, 2.5), percentile(v, 97.5)


def episode_bootstrap(table, rng):
    E = len(table)
    out = []
    for _ in range(B):
        tot = [0.0] * 8
        for i in (rng.randrange(E) for _ in range(E)):
            row = table[i]
            for j in range(8):
                tot[j] += row[j]
        out.append(stats(tot))
    return out


def analyse(rows, label, results):
    table = episode_table(rows)
    point = stats([sum(col) for col in zip(*table)])
    draws = episode_bootstrap(table, random.Random(SEED))
    nF = sum(1 for r in rows if r["gender"] == "F")
    nM = len(rows) - nF
    cF = sum(r["closeup"] for r in rows if r["gender"] == "F") / nF
    cM = sum(r["closeup"] for r in rows if r["gender"] == "M") / nM
    print("== %s: %d F, %d M clips, %d episodes | close-up share F %.3f, M %.3f"
          % (label, nF, nM, len(table), cF, cM))
    for k in ("area_gap", "closeup_gap", "dist_gap", "presence_score"):
        lo, hi = interval(draws, k)
        print("   %-15s %+.4f   95%% CI [%+.4f, %+.4f]" % (k, point[k], lo, hi))
        results.append((label, k, point[k], lo, hi))
    print()


def within_episode(rows, key, rng):
    per = {}
    for r in rows:
        per.setdefault(r["ep"], {"F": [], "M": []})[r["gender"]].append(r[key])
    parts = []                                   # (w * diff, w) per usable episode
    for d in per.values():
        if d["F"] and d["M"]:
            w = len(d["F"]) * len(d["M"]) / (len(d["F"]) + len(d["M"]))
            parts.append((w * (sum(d["F"]) / len(d["F"]) - sum(d["M"]) / len(d["M"])), w))
    est = sum(p[0] for p in parts) / sum(p[1] for p in parts)
    E, boots = len(parts), []
    for _ in range(B):
        idx = [rng.randrange(E) for _ in range(E)]
        boots.append(sum(parts[i][0] for i in idx) / sum(parts[i][1] for i in idx))
    boots.sort()
    return est, percentile(boots, 2.5), percentile(boots, 97.5), E


def character_permutation(rows, key):
    chars = sorted({r["speaker"] for r in rows})
    gender = {r["speaker"]: r["gender"] for r in rows}

    def gap(fset):
        f = [r[key] for r in rows if r["speaker"] in fset]
        m = [r[key] for r in rows if r["speaker"] not in fset]
        return sum(f) / len(f) - sum(m) / len(m)
    obs = gap({c for c in chars if gender[c] == "F"})
    allg = [gap(set(c)) for c in itertools.combinations(chars, 3)]
    p = sum(1 for g in allg if abs(g) >= abs(obs) - 1e-12) / len(allg)
    rank = sum(1 for g in allg if g <= obs + 1e-12)
    return obs, p, rank, len(allg)


# ---------------------------------------------------------------- main
def main(path, out_path=None):
    rows = load(path)
    dev_eps = all_episodes(path)
    dev = [r for r in rows if r["split"] == "dev"]
    cut = percentile([r["area"] for r in dev], 75)
    for r in rows:
        r["closeup"] = 1 if r["area"] >= cut else 0
    train = [r for r in rows if r["split"] == "train"]
    print("close-up cut (75th percentile of dev single-face mean_rel_area): %.6f" % cut)
    print("share of single-face clips at or above the cut: dev %.3f | train %.3f\n" % (
        sum(r["closeup"] for r in dev) / len(dev), sum(r["closeup"] for r in train) / len(train)))

    results = []
    subsets = [("dev", dev), ("train (all)", train),
               ("train, episodes NOT in dev", [r for r in train if r["ep"] not in dev_eps]),
               ("dev + train", dev + train)]
    for label, sub in subsets:
        analyse(sub, label, results)     # "train, episodes NOT in dev" is check A

    print("B. within-episode gap (post hoc), precision-weighted, episodes with both genders")
    for label, sub in (("dev", dev), ("train (all)", train), ("dev + train", dev + train)):
        for name, key in (("face area", "area"), ("close-up", "closeup"), ("centre dist", "dist")):
            est, lo, hi, E = within_episode(sub, key, random.Random(SEED))
            print("   %-12s %-12s %+.4f  95%% CI [%+.4f, %+.4f]  (%d episodes)" % (label, name, est, lo, hi, E))
            results.append((label + " [within-episode]", name, est, lo, hi))
    print()

    print("C. character permutation (20 labellings; smallest possible two-sided p = 0.10)")
    for label, sub in (("dev", dev), ("train (all)", train)):
        for name, key in (("face area", "area"), ("close-up", "closeup"), ("centre dist", "dist")):
            obs, p, rank, n = character_permutation(sub, key)
            print("   %-12s %-12s observed %+.4f | p = %.2f | rank %d of %d (1 = most negative)"
                  % (label, name, obs, p, rank, n))
    print()

    print("per character (single-face clips): n, mean face area, close-up share")
    for split, sub in (("dev", dev), ("train", train)):
        for c in ("Monica", "Phoebe", "Rachel", "Chandler", "Joey", "Ross"):
            s = [r for r in sub if r["speaker"] == c]
            print("   %-5s %-9s n=%4d  area %.4f  close-up %.3f" % (
                split, c, len(s), sum(r["area"] for r in s) / len(s),
                sum(r["closeup"] for r in s) / len(s)))

    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["subset", "measure", "estimate", "ci_lo", "ci_hi"])
            for r in results:
                w.writerow([r[0], r[1]] + ["%.6f" % v for v in r[2:]])
        print("\nwrote", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(*sys.argv[1:3])
