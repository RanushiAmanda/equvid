"""Tokenizer and pronoun lists shared by every C3 (EuqVid-Text) lexical feature.

Everything that decides "what counts as a word" or "what counts as a
reference to a person" lives in this one file, so that:
  * MELD and AMI are counted with exactly the same rules, and
  * the lists can be printed as-is in the thesis appendix.

If anything here changes, bump FEATURE_VERSION: it is written into every
output row, so old and new numbers can never be mixed by accident.
"""
import re

FEATURE_VERSION = "c3-lex-0.1"

# --- text normalisation -----------------------------------------------------
# MELD uses typographic characters (2,598 lines contain a curly apostrophe).
# Without this step "don't" and "don’t" would be two different words, and a
# pronoun like "he’s" would be missed.
_REPLACE = {
    "’": "'", "‘": "'",   # curly apostrophes
    "“": '"', "”": '"',   # curly double quotes
    "…": " ... ",              # ellipsis
    "—": " - ", "–": " - ",  # em / en dash: padded, so "I—I" stays two words
    " ": " ",                  # non-breaking space
}
_REPLACE_RE = re.compile("|".join(map(re.escape, _REPLACE)))


def normalise(text):
    """Replace typographic characters with plain ASCII equivalents."""
    return _REPLACE_RE.sub(lambda m: _REPLACE[m.group(0)], str(text))


# A word = letters/digits, optionally joined by an apostrophe or hyphen.
# So "I'm", "mm-hmm" and "twenty-five" are one word each; punctuation is not
# a word. [^\W_] means "a letter or digit in any alphabet" (keeps "café").
WORD_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")


def tokens(text):
    """Lower-cased word tokens of one utterance."""
    return WORD_RE.findall(normalise(text).lower())


# --- references to people -----------------------------------------------------
# Pronouns only, on purpose:
#   * a pronoun is unambiguously a reference to a person (except "they/them",
#     see the note below) and the third-person ones carry the referent's gender;
#   * gendered nouns are not safe: "man" is often an interjection ("oh man"),
#     "guys" is usually gender-neutral ("you guys"), "miss" is often a verb.
# Contractions are matched on the part before the apostrophe, so "she's",
# "he'd", "they're", "I'm", "you'll" are all counted.
FEMALE_3P = {"she", "her", "hers", "herself"}
MALE_3P = {"he", "him", "his", "himself"}
NEUTRAL_3P = {"they", "them", "their", "theirs", "themselves", "themself"}
FIRST_P = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_P = {"you", "your", "yours", "yourself", "yourselves"}
# Known noise, stated rather than hidden: "they/them" also refers to things
# ("put them on the table"), and "you" is sometimes generic ("you know").
# The noise is the same for both speaker genders unless their topics differ,
# which is why every pronoun feature is reported as an F-vs-M comparison.

PRONOUN_FAMILIES = {
    "fem3": FEMALE_3P,
    "masc3": MALE_3P,
    "neut3": NEUTRAL_3P,
    "first": FIRST_P,
    "second": SECOND_P,
}


def pronoun_counts(toks):
    """Count each pronoun family in a list of tokens from tokens()."""
    counts = dict.fromkeys(PRONOUN_FAMILIES, 0)
    for t in toks:
        base = t.split("'", 1)[0]          # "she's" -> "she"
        for fam, words in PRONOUN_FAMILIES.items():
            if base in words:
                counts[fam] += 1
                break
    return counts


if __name__ == "__main__":
    # Quick self-check you can run in front of an examiner.
    s = "She’s sure he’ll call—but I—I don't think they'll tell you, mm-hmm."
    t = tokens(s)
    print(t)
    print(pronoun_counts(t))
    assert t[:3] == ["she's", "sure", "he'll"] and "i" in t and "mm-hmm" in t
    assert pronoun_counts(t) == {"fem3": 1, "masc3": 1, "neut3": 1, "first": 2, "second": 1}
    print("ok")
