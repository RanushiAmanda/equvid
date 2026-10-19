#!/usr/bin/env python3
"""C3 (EuqVid-Text) steps 1-2: one utterance table per corpus, plus a
provisional gender map.

    python build_utterances.py
    python build_utterances.py --root "D:/SLIIT/4th Year/Y4 S1/Research Project"

Reads (never modifies):
    MELD/train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
    AMI/ami_public_manual_1.6.2.zip        read in place, no unzipping needed
Writes:
    derived/c3/utterances_meld.csv
    derived/c3/utterances_ami.csv
    derived/c3/gender_map_provisional.csv  person_id, gender, gender_source, component
    derived/c3/clips_provisional.csv       one row per clip: composition and label status

One row per utterance, the same columns for both corpora:
    corpus, clip_id, series, unit_id, utt_idx, person_id, speaker_raw, role,
    gender, gender_source, start, end, text, n_words, split

IDs follow the shared-conventions proposal (C2, 6 Oct 2026):
    clip_id   = corpus:unit       MELD:S08E21, AMI:ES2002a
    person_id = corpus:speaker    MELD:Chandler, AMI:FEE005
    series    = MELD_S08, ES2002
unit_id is the stretch of talk in which turn order means something: one MELD
dialogue (a scene) or one AMI meeting. Adjacency is never counted across units.

The gender map here is PROVISIONAL. C2 already generated gender_map.csv; the
group must agree on one file (shared decision 7), and that file replaces this.
"""
import argparse
import re
import zipfile
from pathlib import Path

import xml.etree.ElementTree as etree   # standard library: nothing extra to install

import pandas as pd

from wordlists import normalise, tokens

HERE = Path(__file__).resolve().parent          # .../equvid/src/text
DEFAULT_ROOT = HERE.parents[2]                  # .../Research Project
NITE = "{http://nite.sourceforge.net/}"         # namespace of nite:id attributes

COLUMNS = ["corpus", "clip_id", "series", "unit_id", "utt_idx", "person_id",
           "speaker_raw", "role", "gender", "gender_source", "start", "end",
           "text", "n_words", "split"]

# MELD ships speaker names, not genders. The six leads are mapped by hand, the
# same map C2 uses. Everyone else (guests, "All", "Ross and Rachel", ...) stays
# U and is excluded from features, with the excluded share reported.
MELD_GENDER = {"Monica": "F", "Phoebe": "F", "Rachel": "F",
               "Chandler": "M", "Joey": "M", "Ross": "M"}


# ----------------------------------------------------------------------------
# MELD
# ----------------------------------------------------------------------------
def hms_to_seconds(x):
    """MELD time stamp "00:16:16,059" -> 976.059 seconds."""
    hms, ms = x.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def read_meld(meld_dir):
    parts = []
    for split in ("train", "dev", "test"):
        d = pd.read_csv(meld_dir / f"{split}_sent_emo.csv", encoding="utf-8")
        d["split"] = split
        parts.append(d)
    m = pd.concat(parts, ignore_index=True)

    out = pd.DataFrame({
        "corpus": "MELD",
        # One clip = one episode, pooled over MELD's train/dev/test splits
        # (156 of the 190 episodes have dialogues in more than one split).
        "clip_id": [f"MELD:S{s:02d}E{e:02d}" for s, e in zip(m.Season, m.Episode)],
        "series": [f"MELD_S{s:02d}" for s in m.Season],
        # Dialogue_ID restarts at 0 in each split, so the split is part of the key.
        "unit_id": [f"MELD:{sp}:{d}" for sp, d in zip(m.split, m.Dialogue_ID)],
        "utt_idx": m.Utterance_ID.astype(int),
        "person_id": "MELD:" + m.Speaker.str.strip(),
        "speaker_raw": m.Speaker,
        "role": "",
        "gender": m.Speaker.str.strip().map(MELD_GENDER).fillna("U"),
        "start": m.StartTime.map(hms_to_seconds),
        "end": m.EndTime.map(hms_to_seconds),
        "text": m.Utterance.map(normalise).str.strip(),
        "split": m.split,
    })
    out["gender_source"] = out.gender.map(lambda g: "manual_map" if g in "FM" else "unmapped")
    return out


# ----------------------------------------------------------------------------
# AMI
# ----------------------------------------------------------------------------
HREF_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def parse_meetings(z):
    """meetings.xml -> {meeting: {"type": ..., "agents": {"A": (global_name, role)}}}"""
    root = etree.parse(z.open("corpusResources/meetings.xml")).getroot()
    meetings = {}
    for mt in root.iter("meeting"):
        agents = {sp.get("nxt_agent"): (sp.get("global_name"), sp.get("role", ""))
                  for sp in mt.iter("speaker")}
        meetings[mt.get("observation")] = {"type": mt.get("type"), "agents": agents}
    return meetings


def parse_participants(z):
    """participants.xml -> {participant id: "F" or "M"}"""
    root = etree.parse(z.open("corpusResources/participants.xml")).getroot()
    return {p.get(NITE + "id"): p.get("sex") for p in root.iter("participant")}


def parse_words(z, name):
    """Words file -> (list of elements in document order, {nite:id: position}).

    Segments point at words by id ranges ("id(w0)..id(w12)"), and a range means
    document order, which includes the non-word elements in between
    (<vocalsound>, <disfmarker>, <gap>). So every element with an id is kept."""
    root = etree.parse(z.open(name)).getroot()
    elems = [el for el in root if el.get(NITE + "id") is not None]
    return elems, {el.get(NITE + "id"): i for i, el in enumerate(elems)}


def segment_text(elems):
    """Join the words of one segment into readable text.

    * punctuation (punc="true") is attached to the previous word;
    * truncated words (trunc="true", e.g. "Ye" for an abandoned "Yes") are
      dropped: they are false starts, not words, and they confuse a parser;
    * laughs, coughs and disfluency markers are not words and are skipped."""
    out, n_trunc = [], 0
    for el in elems:
        if el.tag != "w" or not el.text:
            continue
        if el.get("trunc") == "true":
            n_trunc += 1
        elif el.get("punc") == "true" and out:
            out[-1] += el.text
        else:
            out.append(el.text)
    return " ".join(out), n_trunc


def read_ami(zip_path):
    rows, n_trunc_total = [], 0
    with zipfile.ZipFile(zip_path) as z:
        meetings = parse_meetings(z)
        sex = parse_participants(z)
        word_files = sorted(n for n in z.namelist()
                            if n.startswith("words/") and n.endswith(".words.xml"))
        for name in word_files:
            meeting, agent = Path(name).name.split(".")[:2]
            seg_name = f"segments/{meeting}.{agent}.segments.xml"
            if agent not in meetings[meeting]["agents"]:
                # Five 3-person meetings (EN2002c, EN2003a, EN2009b, EN2009c,
                # IN1001) ship an empty file for a fourth seat. Skip it, but
                # stop if such a file ever contains speech.
                assert b"<segment " not in z.read(seg_name), f"unlisted speaker with speech: {name}"
                continue
            global_name, role = meetings[meeting]["agents"][agent]
            # Gender: participants.xml when the person is listed there. The 40
            # TNO (TS) participants are not listed; their codes start with F or M,
            # which agrees with the recorded sex for 189/189 listed people, so the
            # first letter is used but labelled id_prefix (unverified) and kept
            # out of the main analysis, as C2 does.
            if global_name in sex:
                gender, source = sex[global_name], "participants_xml"
            elif global_name[:1] in ("F", "M"):
                gender, source = global_name[0], "id_prefix"
            else:
                gender, source = "U", "unmapped"

            elems, pos = parse_words(z, name)
            seg_root = etree.parse(z.open(seg_name)).getroot()
            for seg in seg_root.iter("segment"):
                child = seg.find(NITE + "child")
                first, last = HREF_RE.search(child.get("href")).groups()
                i, j = pos[first], pos[last or first]
                text, n_trunc = segment_text(elems[i:j + 1])
                n_trunc_total += n_trunc
                rows.append({
                    "corpus": "AMI",
                    "clip_id": f"AMI:{meeting}",
                    "series": re.sub(r"[a-z]$", "", meeting),   # ES2002a -> ES2002
                    "unit_id": f"AMI:{meeting}",
                    "start": float(seg.get("transcriber_start")),
                    "end": float(seg.get("transcriber_end")),
                    "person_id": f"AMI:{global_name}",
                    "speaker_raw": f"{meeting}.{agent}",
                    "role": role,
                    "gender": gender,
                    "gender_source": source,
                    "text": text,
                    "split": meetings[meeting]["type"],          # scenario / nonscenario
                })
    out = pd.DataFrame(rows)
    # Meetings have no utterance numbers, so order by start time (then end time,
    # then person) and number the utterances within each meeting.
    out = out.sort_values(["clip_id", "start", "end", "person_id"], kind="stable")
    out["utt_idx"] = out.groupby("clip_id").cumcount()
    print(f"AMI: dropped {n_trunc_total:,} truncated words")
    return out


# ----------------------------------------------------------------------------
# shared steps
# ----------------------------------------------------------------------------
def components(df):
    """Person-connected groups: clips that share any person are linked.

    Needed for the identity firewall: folds and bootstraps must keep a person's
    clips together. All of MELD is one group (the same six leads everywhere)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for clip, person in df[["clip_id", "person_id"]].drop_duplicates().itertuples(index=False):
        parent[find(("c", clip))] = find(("p", person))

    groups = {}
    for clip in sorted(df.clip_id.unique()):
        groups.setdefault(find(("c", clip)), []).append(clip)
    label = {}
    for k, clips in enumerate(sorted(groups.values()), start=1):
        for clip in clips:
            label[clip] = "MELD_ALL" if clip.startswith("MELD:") else f"AMI_{k:03d}"
    return df.clip_id.map(label)


def clip_table(df):
    """One row per clip: who speaks, and whether every label is verified."""
    people = df.drop_duplicates(["clip_id", "person_id"])
    t = people.groupby("clip_id").agg(
        corpus=("corpus", "first"), series=("series", "first"),
        component=("component", "first"), meeting_type=("split", "first"),
        n_F=("gender", lambda g: (g == "F").sum()),
        n_M=("gender", lambda g: (g == "M").sum()),
        n_U=("gender", lambda g: (g == "U").sum()),
        n_id_prefix=("gender_source", lambda s: (s == "id_prefix").sum()),
    ).reset_index()
    # MELD episodes always contain unmapped guests by design, so they are
    # labelled main_cast; an AMI meeting is verified only if all its people are.
    t["label_status"] = "verified"
    t.loc[t.corpus == "MELD", "label_status"] = "main_cast"
    t.loc[(t.corpus == "AMI") & (t.n_id_prefix > 0), "label_status"] = "id_prefix"
    t.loc[(t.corpus == "AMI") & (t.n_U > 0), "label_status"] = "unknown"
    t.loc[t.corpus == "MELD", "meeting_type"] = ""
    t["mixed_gender"] = (t.n_F > 0) & (t.n_M > 0)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="the 'Research Project' folder (default: three levels up)")
    args = ap.parse_args()
    out_dir = args.root / "derived" / "c3"
    out_dir.mkdir(parents=True, exist_ok=True)

    meld = read_meld(args.root / "MELD")
    ami = read_ami(args.root / "AMI" / "ami_public_manual_1.6.2.zip")
    allu = pd.concat([meld, ami], ignore_index=True)

    allu["n_words"] = allu.text.map(lambda s: len(tokens(s)))
    empty = allu.n_words == 0          # e.g. a segment that was only a laugh
    print("utterances with no words, dropped:",
          allu[empty].groupby("corpus").size().to_dict())
    allu = allu[~empty].copy()
    allu["component"] = components(allu)

    for corpus, part in allu.groupby("corpus"):
        part = part.sort_values(["clip_id", "unit_id", "utt_idx"], kind="stable")
        part[COLUMNS].to_csv(out_dir / f"utterances_{corpus.lower()}.csv",
                             index=False, encoding="utf-8")
        print(f"{corpus}: {len(part):,} utterances, {part.clip_id.nunique()} clips, "
              f"{part.person_id.nunique()} speakers")

    gmap = (allu.groupby("person_id")
            .agg(gender=("gender", "first"), gender_source=("gender_source", "first"),
                 component=("component", lambda c: "|".join(sorted(set(c)))))
            .reset_index())
    gmap.to_csv(out_dir / "gender_map_provisional.csv", index=False)
    clip_table(allu).to_csv(out_dir / "clips_provisional.csv", index=False)
    print("gender sources:", gmap.groupby(["gender_source", "gender"]).size().to_dict())
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
