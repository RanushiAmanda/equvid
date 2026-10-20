"""The Sap et al. (2017) connotation-frame lexicon of agency and power, for C3.

Reads   Research Project/Lexicons/sap2017_agency_power/agency_power.csv
        (provenance, licence and quirks: SOURCE.md in that folder)

The lexicon gives each verb two labels, rated by crowd workers on sentences
like "X alters Y", where X is the agent (subject) and Y the theme (object):

    agency  agency_pos / agency_neg / agency_equal
            Does the verb give its SUBJECT high or low agency?
    power   power_agent / power_theme / power_equal
            Does the verb put the subject (agent) or the object (theme) in
            the position of power?

Labels are turned into numbers exactly as Riveter does (Antoniak et al., 2023),
so C3's numbers stay comparable with published work:

                        agent (subject)   theme (object)
    agency_pos/neg/eq     +1 / -1 / 0       not defined
    power_agent/theme/eq  +1 / -1 / 0       -1 / +1 / 0
"""
import csv
import hashlib
from pathlib import Path

LEXICON_MD5 = "8f14a8baf6f1c6dd5a9d279ba4e0e452"   # copy from Riveter commit eef4f12

# Quirks of this copy (SOURCE.md): three entries are not lemmas.
UNLEMMATISED = {"assists": "assist", "bars": "bar", "pops": "pop"}

# spaCy's small English model sometimes leaves informal forms unlemmatised
# (the setup check found "wanna" 95 times and "got" 9 times in MELD train).
# Declared before any results were seen; every mapping is a plain inflection
# or contraction of the target verb.
INFORMAL = {"wanna": "want", "wan": "want", "gonna": "go", "gon": "go",
            "got": "get", "gotten": "get"}

AGENCY_VALUE = {"agency_pos": 1, "agency_neg": -1, "agency_equal": 0}
POWER_AS_AGENT = {"power_agent": 1, "power_theme": -1, "power_equal": 0}
POWER_AS_THEME = {"power_agent": -1, "power_theme": 1, "power_equal": 0}


def load_lexicon(path):
    """verb -> {"agency": label or None, "power": label or None}.

    Rows for the same verb are merged: "echo" and "go" each appear twice in
    this copy, with one label per row, so keeping only the last row would
    lose a label."""
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


def checksum(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def lookup(lex, lemma, particles=""):
    """Lexicon key for a verb, or "" if the verb is not in the lexicon.

    65 entries are verb + particle/preposition ("freak out", "apologize to").
    Those are tried first, using the particles the parser attached to the
    verb; otherwise the plain lemma is used."""
    lemma = INFORMAL.get(lemma, lemma)
    for p in particles.split():
        if f"{lemma} {p}" in lex:
            return f"{lemma} {p}"
    return lemma if lemma in lex else ""
