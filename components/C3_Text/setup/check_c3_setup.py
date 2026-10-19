"""
Check that everything C3 (EuqVid-Text) needs on this computer is installed and
working, then run two quick smoke tests on MELD.

    python check_c3_setup.py          (setup_c3.bat runs this for you)

Results are printed and also written to c3_setup_report.txt next to this file.

PART 1 - INSTALL CHECKS (each must say OK)
  spaCy + English model, VADER, ffmpeg, the agency/power lexicon.

PART 2 - SMOKE TESTS ON MELD (information, not pass/fail)
  A. Lexicon coverage: what share of the verbs in MELD dialogue the Sap et al.
     lexicon has a label for. This is risk R4 in the C3 spec ("lexicon domain
     mismatch"), and the OOV rate is deliverable D5.
  B. VADER vs MELD's own sentiment labels: how often a text-only sentiment tool
     agrees with MELD's human annotators. This is evidence for the F8 decision,
     not a verdict: MELD's annotators also saw the video and heard the audio,
     so a text-only tool cannot reach 100% even if it were perfect.

Nothing here changes any file except c3_setup_report.txt.
"""

import csv
import hashlib
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # setup -> C3_Text -> components -> equvid -> "Research Project"
LEXICON = ROOT / "Lexicons" / "sap2017_agency_power" / "agency_power.csv"
LEXICON_MD5 = "8f14a8baf6f1c6dd5a9d279ba4e0e452"  # copy taken from Riveter commit eef4f12, 11 Oct 2026
MELD_DIR = ROOT / "MELD"
REPORT = HERE / "c3_setup_report.txt"

# Three lexicon entries were stored in 3rd-person form instead of as lemmas
# (see Lexicons/sap2017_agency_power/SOURCE.md). Map them so lemma lookups find them.
UNLEMMATISED = {"assists": "assist", "bars": "bar", "pops": "pop"}

lines = []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


def result(name, ok, detail):
    say(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")
    return ok


# ---------------------------------------------------------------- loaders


def load_lexicon(path):
    """verb lemma -> {"agency": label or None, "power": label or None}.
    Rows for the same verb are merged (echo and go each appear twice, one label per row)."""
    lex = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            verb = row["verb"].strip().lower()
            verb = UNLEMMATISED.get(verb, verb)
            entry = lex.setdefault(verb, {"agency": None, "power": None})
            for dim in ("agency", "power"):
                if row[dim].strip():
                    entry[dim] = row[dim].strip()
    return lex


def load_meld():
    """All MELD utterances with their human sentiment label, from the GitHub CSVs."""
    rows = []
    for split in ("train", "dev", "test"):
        with open(MELD_DIR / f"{split}_sent_emo.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                # MELD mixes curly and straight apostrophes; unify them before any NLP.
                text = r["Utterance"].replace("’", "'").replace("‘", "'")
                rows.append((split, text, r["Sentiment"].strip().lower()))
    return rows


# ---------------------------------------------------------------- part 1


def check_install():
    say("PART 1 - INSTALL CHECKS")
    ok = {}
    tools = {}

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        doc = nlp("She grabbed the keys and he waited.")
        pairs = [(t.text, t.head.lemma_) for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
        ok["spacy"] = result("spaCy + en_core_web_sm", True,
                             f"spaCy {spacy.__version__}, model {nlp.meta.get('version')}; "
                             f"subject-verb pairs in a test sentence: {pairs}")
        tools["nlp"] = nlp
    except Exception as e:
        ok["spacy"] = result("spaCy + en_core_web_sm", False, f"{type(e).__name__}: {e}")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        pos = sia.polarity_scores("I love this, it's wonderful!")["compound"]
        neg = sia.polarity_scores("This is awful and I hate it.")["compound"]
        ok["vader"] = result("VADER", pos > 0 > neg, f"compound +{pos:.2f} for a positive and {neg:.2f} for a negative test line")
        tools["sia"] = sia
    except Exception as e:
        ok["vader"] = result("VADER", False, f"{type(e).__name__}: {e}")

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        first = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
        ok["ffmpeg"] = result("ffmpeg (imageio-ffmpeg)", first.startswith("ffmpeg"), f"{first[:60]} | {exe}")
    except Exception as e:
        ok["ffmpeg"] = result("ffmpeg (imageio-ffmpeg)", False, f"{type(e).__name__}: {e}")

    try:
        if not LEXICON.exists():
            raise FileNotFoundError(f"not found at {LEXICON}")
        digest = hashlib.md5(LEXICON.read_bytes()).hexdigest()
        lex = load_lexicon(LEXICON)
        n_ag = sum(1 for v in lex.values() if v["agency"])
        n_pw = sum(1 for v in lex.values() if v["power"])
        ok["lexicon"] = result("Agency/power lexicon", digest == LEXICON_MD5 and len(lex) == 2148,
                               f"{len(lex)} verbs ({n_ag} with agency, {n_pw} with power); "
                               f"checksum {'matches' if digest == LEXICON_MD5 else 'DIFFERS - file changed?'}")
        tools["lex"] = lex
    except Exception as e:
        ok["lexicon"] = result("Agency/power lexicon", False, f"{type(e).__name__}: {e}")

    return ok, tools


# ---------------------------------------------------------------- part 2


def coverage_test(nlp, lex, utterances):
    """Share of MELD verb tokens the lexicon labels: all verbs, and verbs whose subject is 'I'
    (the speaker acting - the case C3's speaker-centred F4/F5 rely on most)."""
    say("\n  A. Lexicon coverage on MELD (train) - spec risk R4 / deliverable D5")
    t0 = time.time()
    counts = {"all": Counter(), "I": Counter()}
    oov = Counter()
    for doc in nlp.pipe(utterances, batch_size=256):
        for tok in doc:
            if tok.pos_ != "VERB":  # auxiliaries (be, have, do, modals) are tagged AUX and excluded
                continue
            lemma = tok.lemma_.lower()
            entry = lex.get(lemma)
            groups = ["all"]
            if any(c.dep_ in ("nsubj", "nsubjpass") and c.lower_ == "i" for c in tok.children):
                groups.append("I")
            for g in groups:
                counts[g]["verbs"] += 1
                counts[g]["agency"] += bool(entry and entry["agency"])
                counts[g]["power"] += bool(entry and entry["power"])
            if not entry:
                oov[lemma] += 1
    for g, label in (("all", "all verb tokens"), ("I", "verbs with 'I' as subject")):
        c = counts[g]
        if c["verbs"]:
            say(f"     {label:26} {c['verbs']:6,d}   agency label {c['agency'] / c['verbs']:5.1%}   "
                f"power label {c['power'] / c['verbs']:5.1%}")
    say(f"     most frequent verbs with no entry: {', '.join(f'{v} ({n})' for v, n in oov.most_common(12))}")
    say(f"     ({len(utterances):,} utterances parsed in {time.time() - t0:.0f} s)")


def vader_test(sia, rows):
    """Agreement between VADER (standard +/-0.05 thresholds) and MELD's human sentiment labels."""
    say("\n  B. VADER vs MELD's human sentiment labels (train + dev + test) - evidence for F8")
    labels = ("negative", "neutral", "positive")
    conf = Counter()
    for _, text, gold in rows:
        c = sia.polarity_scores(text)["compound"]
        pred = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
        conf[(gold, pred)] += 1
    n = sum(conf.values())
    acc = sum(conf[(l, l)] for l in labels) / n
    f1s = []
    for l in labels:
        tp = conf[(l, l)]
        prec = tp / max(1, sum(conf[(g, l)] for g in labels))
        rec = tp / max(1, sum(conf[(l, p)] for p in labels))
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        say(f"     {l:9} precision {prec:5.1%}   recall {rec:5.1%}")
    gold_counts = Counter(g for g, _ in conf.elements())
    majority = max(gold_counts.values()) / n
    say(f"     accuracy {acc:.1%} (always guessing the most common label would score {majority:.1%}); "
        f"macro-F1 {sum(f1s) / 3:.2f}; n = {n:,}")
    say("     Note: MELD's annotators also saw the video and heard the audio, so a text-only tool has a ceiling below 100%.")


def main():
    say(f"C3 setup check - {time.strftime('%Y-%m-%d %H:%M')}")
    say(f"Python {sys.version.split()[0]} at {sys.executable}")
    say(f"Research Project folder: {ROOT}\n")
    ok, tools = check_install()

    say("\nPART 2 - SMOKE TESTS ON MELD")
    meld_ok = all((MELD_DIR / f"{s}_sent_emo.csv").exists() for s in ("train", "dev", "test"))
    if not meld_ok:
        say(f"  skipped: MELD CSVs not found in {MELD_DIR}")
    else:
        rows = load_meld()
        if ok.get("spacy") and ok.get("lexicon"):
            try:
                coverage_test(tools["nlp"], tools["lex"], [t for s, t, _ in rows if s == "train"])
            except Exception as e:
                say(f"  A. failed: {type(e).__name__}: {e}")
        else:
            say("  A. skipped: needs spaCy and the lexicon")
        if ok.get("vader"):
            try:
                vader_test(tools["sia"], rows)
            except Exception as e:
                say(f"  B. failed: {type(e).__name__}: {e}")
        else:
            say("  B. skipped: needs VADER")

    passed = all(ok.values())
    say("\nRESULT: " + ("everything C3 needs on this computer is installed and working."
                        if passed else "something failed - see the FAIL lines above."))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
