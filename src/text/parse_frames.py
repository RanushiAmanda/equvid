#!/usr/bin/env python3
"""C3: parse every line with spaCy and keep the agency/power frames.

    python parse_frames.py            (run_c3_text.bat runs it for you)
    python parse_frames.py --force    parse again even if the output is up to date

Reads   derived/c3/utterances_meld.csv, utterances_ami.csv
Writes  derived/c3/c3_verb_frames.csv.gz   one row per frame (rules: frames.py)
        derived/c3/c3_parse_report.txt     versions, counts, timing

This is the only C3 step that needs spaCy, so it is kept separate. Everything
after it (lexicon scores, features, bootstrap) is in agency_power_features.py
and runs anywhere. Parsing takes a few minutes, so it is skipped when the
utterance tables and the parsing code are byte-for-byte what the last parse
used (their fingerprint is kept in c3_verb_frames.inputs.md5).
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

from frames import frames_from_doc

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]
MODEL = "en_core_web_sm"

# A line can only produce a frame if it contains one of these pronouns, so
# lines without them are not parsed at all. This roughly halves the time and
# cannot change a result.
PRONOUN_PATTERN = r"\b(?:i|me|she|he|her|him)\b"


def fingerprint(paths):
    """One md5 over the bytes of every input. File dates are not used, because
    build_utterances.py rewrites identical tables on every run."""
    h = hashlib.md5()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--force", action="store_true", help="parse even if up to date")
    args = ap.parse_args()
    d = args.root / "derived" / "c3"
    tables = [d / "utterances_meld.csv", d / "utterances_ami.csv"]
    out = d / "c3_verb_frames.csv.gz"
    stamp = d / "c3_verb_frames.inputs.md5"
    inputs = fingerprint(tables + [HERE / "frames.py", Path(__file__).resolve()])
    if (not args.force and out.exists() and stamp.exists()
            and stamp.read_text().strip() == inputs):
        print("frames are up to date; parse skipped (add --force to parse again)")
        return

    try:
        import spacy
        nlp = spacy.load(MODEL, disable=["ner"])   # names are not needed, so skip NER
    except (ImportError, OSError) as e:
        sys.exit(f"spaCy or its English model is missing ({e}).\n"
                 "Run equvid/components/C3_Text/setup/setup_c3.bat once, then try again.")

    u = pd.concat([pd.read_csv(t, keep_default_na=False, na_values={"start": [""], "end": [""]})
                   for t in tables], ignore_index=True)
    todo = u[(u.n_words >= 2) & u.text.str.contains(PRONOUN_PATTERN, case=False, regex=True)]
    print(f"parsing {len(todo):,} of {len(u):,} lines with spaCy {spacy.__version__} "
          f"({MODEL} {nlp.meta.get('version')}) ...", flush=True)

    rows, t0 = [], time.time()
    pairs = zip(todo.text.tolist(), todo.index.tolist())
    for n, (doc, row_id) in enumerate(nlp.pipe(pairs, as_tuples=True, batch_size=512), start=1):
        for f in frames_from_doc(doc):
            f["row_id"] = row_id
            rows.append(f)
        if n % 10000 == 0:
            print(f"  {n:,} lines parsed ({time.time() - t0:.0f} s)", flush=True)
    seconds = time.time() - t0

    if not rows:
        sys.exit("no frames found - check that the utterance tables are not empty")
    frames = pd.DataFrame(rows)
    meta = u.loc[frames.row_id, ["corpus", "clip_id", "unit_id", "utt_idx", "person_id", "gender"]]
    frames = pd.concat([meta.reset_index(drop=True), frames.drop(columns="row_id")], axis=1)
    frames = frames.rename(columns={"gender": "speaker_gender"})
    # The character the frame is about: the speaker for "I"/"me", otherwise
    # the gender of the pronoun. Speakers with unknown gender (MELD guests)
    # keep "U" and are dropped later, like everywhere else in C3.
    frames.insert(frames.columns.get_loc("speaker_gender") + 1, "character_gender",
                  frames.speaker_gender.where(frames.who == "speaker", frames.who))
    frames.to_csv(out, index=False, compression="gzip")
    stamp.write_text(inputs + "\n")

    counts = frames.groupby(["corpus", "role"]).size().unstack(fill_value=0)
    report = [
        f"C3 parse report - {time.strftime('%Y-%m-%d %H:%M')}",
        f"Python {sys.version.split()[0]} | spaCy {spacy.__version__} | {MODEL} {nlp.meta.get('version')}",
        f"lines: {len(u):,} in total, {len(todo):,} parsed (contain I/me/she/he/her/him, 2+ words)",
        f"time: {seconds:.0f} s",
        f"frames: {len(frames):,} written to {out.name}",
        "",
        counts.to_string(),
    ]
    (d / "c3_parse_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
