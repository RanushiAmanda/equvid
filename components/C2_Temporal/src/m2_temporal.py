"""
EuqVid M2 (Temporal): floor-allocation measures from speaker intervals.

Pipeline: interval table -> turns -> clip features -> allocation score and structure z-score.
Input table columns: corpus, meeting, series, speaker, role, gender (F/M), gender_source,
start, end (seconds), n_words.

Usage:
    scorer = M2Scorer.load("m2_model_v0.1.json")
    turns = build_turns(mark_backchannels(intervals))
    row = score_clip(turns[turns["meeting"] == "ES2002a"], scorer)

Conventions (proposed, see the shared note):
    * Positive score = women above men per person.
    * All ratios are per person (log of F per person over M per person).
    * Operators keep each gender's total speaking time constant.
    * Positive signed level modifies women's turns, negative modifies men's.
"""
import json
import numpy as np
import pandas as pd

FEATURE_VERSION = "m2_v0.1"
GAP = 1.0                     # max silence (s) when merging a speaker's utterances into one turn
EPS = 0.2                     # tolerance (s) when deciding that a turn starts inside another
BACKCHANNEL_MAX_WORDS = 2     # utterances with this many words or fewer are not turns
MIN_TURNS = 20                # a clip needs at least this many turns to be scored
MIN_TURNS_EACH_GENDER_FOR_SPREAD = 3
ANCHOR_RATIO = 4.0            # allocation score +/-10 means a 4:1 per-person ratio
TIERS_ABS_Z = (1.96, 3.0, 5.0)
LEVELS = np.round(np.linspace(0.1, 1.0, 10), 1)
KEEP = ["corpus", "meeting", "series", "speaker", "role", "gender", "gender_source"]
STRUCT_FEATURES = ["struct_asym", "interruption_asym", "position_gap", "pos_sd_ratio"]


# ----------------------------------------------------------------------------
# turns
# ----------------------------------------------------------------------------
def mark_backchannels(df, max_words=BACKCHANNEL_MAX_WORDS):
    """Add a boolean column 'backchannel' (utterances of max_words words or fewer)."""
    out = df.copy()
    out["backchannel"] = out["n_words"] <= max_words
    return out


def build_turns(df, gap=GAP, keep=KEEP):
    """Merge each speaker's consecutive utterances into turns.
    Two utterances merge when the silence between them is at most `gap` seconds and
    nobody else starts speaking in between. Backchannels are dropped first."""
    out = []
    for meeting, d in df[~df["backchannel"]].groupby("meeting"):
        for spk, s in d.groupby("speaker"):
            others = np.sort(d.loc[d["speaker"] != spk, "start"].to_numpy())
            cur = None
            for r in s.sort_values("start").itertuples():
                if cur is not None:
                    i0 = np.searchsorted(others, cur["end"], side="right")
                    i1 = np.searchsorted(others, r.start, side="left")
                    if (r.start - cur["end"]) <= gap and i1 <= i0:
                        cur["end"] = max(cur["end"], r.end)
                        cur["n_words"] += r.n_words
                        continue
                    out.append(cur)
                cur = {**{k: getattr(r, k) for k in keep},
                       "start": r.start, "end": r.end, "n_words": r.n_words}
            if cur is not None:
                out.append(cur)
    t = pd.DataFrame(out)
    t["dur"] = t["end"] - t["start"]
    return t


# ----------------------------------------------------------------------------
# features (one clip = the turns of one meeting or episode)
# ----------------------------------------------------------------------------
def meeting_features(t, eps=EPS):
    """Per-person log ratios (F over M) and interruption / position asymmetries."""
    t = t.sort_values("start")
    st, en = t["start"].to_numpy(), t["end"].to_numpy()
    spk, gen = t["speaker"].to_numpy(), t["gender"].to_numpy()
    dur = t["dur"].to_numpy()
    isF, isM = gen == "F", gen == "M"
    nF, nM = len(set(spk[isF])), len(set(spk[isM]))
    if nF == 0 or nM == 0:
        return None
    TF, TM = dur[isF].sum(), dur[isM].sum()
    NF, NM = int(isF.sum()), int(isM.sum())

    # turn i is cut into when a turn by ANOTHER speaker starts inside it
    inside = ((st[None, :] > st[:, None] + eps) & (st[None, :] < en[:, None] - eps)
              & (spk[None, :] != spk[:, None]))
    cutF = int((inside & isF[:, None] & isM[None, :]).sum())   # F turn cut into by an M turn
    cutM = int((inside & isM[:, None] & isF[None, :]).sum())   # M turn cut into by an F turn

    t0, t1 = st.min(), en.max()
    mid = ((st + en) / 2 - t0) / max(t1 - t0, 1e-9)
    cen_F = (mid[isF] * dur[isF]).sum() / TF
    cen_M = (mid[isM] * dur[isM]).sum() / TM

    return {"n_F": nF, "n_M": nM, "N_F": NF, "N_M": NM, "T_F": TF, "T_M": TM,
            "log_parity_time": np.log((TF / nF) / (TM / nM)),
            "log_parity_turns": np.log((NF / nF) / (NM / nM)),
            "log_len_ratio": np.log((TF / NF) / (TM / NM)),
            "interruption_asym": (cutF - cutM) / (NF + NM),
            "position_gap": cen_F - cen_M,
            "cut_F": cutF, "cut_M": cutM}


def spread_features(t, min_turns=MIN_TURNS_EACH_GENDER_FOR_SPREAD):
    """Log ratio of how widely each gender's speaking is spread over the clip."""
    isF = (t["gender"] == "F").to_numpy()
    isM = (t["gender"] == "M").to_numpy()
    if isF.sum() < min_turns or isM.sum() < min_turns:
        return None
    dur = t["dur"].to_numpy()
    st, en = t["start"].to_numpy(), t["end"].to_numpy()
    t0, t1 = st.min(), en.max()
    mid = ((st + en) / 2 - t0) / max(t1 - t0, 1e-9)

    def wsd(mask):
        w, m = dur[mask], mid[mask]
        c = (m * w).sum() / w.sum()
        return np.sqrt((w * (m - c) ** 2).sum() / w.sum())

    eps = 1e-9
    return {"len_sd_ratio": np.log((dur[isF].std(ddof=1) + eps) / (dur[isM].std(ddof=1) + eps)),
            "pos_sd_ratio": np.log((wsd(isF) + eps) / (wsd(isM) + eps))}


def clip_features(t):
    """All features of one clip, or None if one gender is absent."""
    f = meeting_features(t)
    if f is None:
        return None
    s = spread_features(t) or {"len_sd_ratio": np.nan, "pos_sd_ratio": np.nan}
    out = {**f, **s}
    out["struct_asym"] = (out["log_parity_turns"] - out["log_len_ratio"]) / 2
    return out


# ----------------------------------------------------------------------------
# structure-preserving operators (each keeps every gender's total speaking time)
# ----------------------------------------------------------------------------
def fragment_ami(t, target, level, k=3, gap=1.2, min_piece=0.5, seed=0):
    """Split a fraction `level` of the ELIGIBLE target turns into k pieces (silence `gap` between)."""
    rng = np.random.default_rng(seed)
    t = t.sort_values(["start", "end"]).reset_index(drop=True)
    nxt_own = t.groupby("speaker")["start"].shift(-1)
    room = (nxt_own - t["end"]).fillna(np.inf)
    elig = ((t["gender"] == target) & (t["dur"] >= k * min_piece) & (room >= (k - 1) * gap)).to_numpy()
    chosen = np.zeros(len(t), bool)
    n_pick = int(round(level * elig.sum()))
    if n_pick > 0:
        chosen[rng.choice(np.flatnonzero(elig), size=n_pick, replace=False)] = True
    rows = []
    for i, r in enumerate(t.to_dict("records")):
        if not chosen[i]:
            rows.append(r)
            continue
        piece = r["dur"] / k
        for j in range(k):
            p = dict(r)
            p["start"] = r["start"] + j * (piece + gap)
            p["end"] = p["start"] + piece
            p["dur"] = piece
            rows.append(p)
    return pd.DataFrame(rows)


def interrupt_ami(t, target, level, overlap_frac=0.3, max_gap=3.0, seed=0):
    """For a fraction `level` of eligible target turns, move the next other-gender turn
    earlier so that it starts inside the target turn. Durations and counts never change."""
    rng = np.random.default_rng(seed)
    t = t.sort_values(["start", "end"]).reset_index(drop=True)
    st, en = t["start"].to_numpy().copy(), t["end"].to_numpy().copy()
    dur, gen = t["dur"].to_numpy(), t["gender"].to_numpy()
    prev_end = t.groupby("speaker")["end"].shift().fillna(-np.inf).to_numpy()
    used, pairs = set(), []
    for i in np.flatnonzero((gen == target) & (dur >= 1.0)):
        ns = en[i] - overlap_frac * dur[i]
        for j in np.flatnonzero((st > en[i]) & (st <= en[i] + max_gap) & (gen != target)):
            if j in used:
                continue
            if ns >= prev_end[j] + 0.05:
                pairs.append((j, ns))
                used.add(j)
                break
    n_pick = int(round(level * len(pairs)))
    if n_pick > 0:
        for p in rng.choice(len(pairs), size=n_pick, replace=False):
            j, ns = pairs[p]
            st[j], en[j] = ns, ns + dur[j]
    out = t.copy()
    out["start"], out["end"] = st, en
    return out


def cluster_ami(t, target, level, seed=0):
    """Shrink the silences between each target speaker's own turns by (1 - level)."""
    t = t.sort_values(["start", "end"]).reset_index(drop=True)
    st, en = t["start"].to_numpy().copy(), t["end"].to_numpy().copy()
    for s in t.loc[t["gender"] == target, "speaker"].unique():
        idx = np.flatnonzero((t["speaker"] == s).to_numpy())
        d = en[idx] - st[idx]
        gaps = np.maximum(st[idx[1:]] - en[idx[:-1]], 0.0)
        new = np.empty(len(idx))
        new[0] = st[idx[0]]
        for k in range(1, len(idx)):
            new[k] = new[k - 1] + d[k - 1] + (1 - level) * gaps[k - 1]
        st[idx], en[idx] = new, new + d
    out = t.copy()
    out["start"], out["end"] = st, en
    return out


OPERATORS = {"O1_fragment": fragment_ami, "O2_interrupt": interrupt_ami, "O3_cluster": cluster_ami}


def make_variant(t, operator, signed_level, seed=0):
    """Apply an operator at a signed level. Positive modifies women's turns, negative men's."""
    target = "F" if signed_level > 0 else "M"
    return OPERATORS[operator](t, target, abs(signed_level), seed=seed)


def variant_id(clip_id, operator, signed_level):
    return "%s|%s|%+.2f" % (clip_id, operator, signed_level)


# ----------------------------------------------------------------------------
# scores
# ----------------------------------------------------------------------------
def allocation_score(log_parity_time, anchor_ratio=ANCHOR_RATIO):
    """+10 / -10 = one gender speaks anchor_ratio times as much per person. Positive = women more."""
    return 10.0 * np.clip(np.asarray(log_parity_time) / np.log(anchor_ratio), -1.0, 1.0)


def tier_from_z(z, tiers=TIERS_ABS_Z):
    a = abs(z)
    if np.isnan(a):
        return "NA"
    if a < tiers[0]:
        return "within noise"
    if a < tiers[1]:
        return "mild"
    if a < tiers[2]:
        return "moderate"
    return "strong"


class M2Scorer:
    """Frozen structure model: standardise, linear combination, z against unmodified clips."""

    def __init__(self, spec):
        self.spec = spec

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def structure_raw(self, features):
        m = self.spec
        x = np.array([features[c] for c in m["features"]], dtype=float)
        z = (x - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])
        return float(z @ np.array(m["coef"]) + m["intercept"])

    def structure_z(self, raw):
        return (raw - self.spec["null_mean"]) / self.spec["null_sd"]


def score_clip(t, scorer, clip_id=None):
    """One output row for one clip, following the shared output contract."""
    row = {"clip_id": clip_id, "feature_version": FEATURE_VERSION, "allocation_score": np.nan,
           "structure_raw": np.nan, "structure_z": np.nan, "structure_tier": "NA",
           "status_allocation": "NA", "status_structure": "NA", "reason": ""}
    f = clip_features(t) if len(t) else None
    if f is None:
        row["reason"] = "no speaker of one gender"
        return row
    if f["N_F"] + f["N_M"] < MIN_TURNS:
        row["reason"] = "fewer than %d turns" % MIN_TURNS
        return row
    row["allocation_score"] = float(allocation_score(f["log_parity_time"]))
    row["status_allocation"] = "valid"
    if any(np.isnan(f[c]) for c in scorer.spec["features"]):
        row["reason"] = "fewer than %d turns of one gender: no spread features" % MIN_TURNS_EACH_GENDER_FOR_SPREAD
        return row
    raw = scorer.structure_raw(f)
    z = scorer.structure_z(raw)
    row.update({"structure_raw": raw, "structure_z": float(z),
                "structure_tier": tier_from_z(z), "status_structure": "valid"})
    return row