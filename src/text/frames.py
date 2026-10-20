"""Rules that turn one parsed line into agency/power "frames", for C3.

No spaCy import here on purpose: the rules only read token attributes
(pos_, dep_, lower_, lemma_, text, head, children), so they can be tested with
hand-built tokens (test_text_features.py) on a computer without spaCy.

A frame is one verb plus one character in a known role:

    self_agent   the speaker is the subject     "I quit", "I'll call him"
    self_theme   the speaker is the object      "she fired me", "I was told"
    ref_agent    a woman or man is the subject  "she quit", "he called"
    ref_theme    a woman or man is the object   "they fired her", "he was told"

Characters are found through pronouns only. A pronoun carries the gender and
needs no cast list, so the rule works on MELD, AMI and AVA alike (the same
roster-free principle as ref_log_parity):

    I (subject), me (object)        -> the speaker of the line
    she / he (subject), her / him (object) -> a woman / a man

"we", "you" and "they" are skipped: their gender is unknown. Possessive "her"
("her car") is not an object: the parser labels it poss, so the object rule
never matches it.
"""

SUBJECT = "nsubj"            # active subject: the agent ("I quit")
PASSIVE_SUBJECT = "nsubjpass"  # grammatical subject of a passive: the theme ("I was told")
OBJECT = "dobj"              # direct object: the theme ("they fired her")
NEGATION = "neg"             # not, n't, never
PARTICLES = {"prt", "prep"}  # "freak out", "apologize to": used for multi-word entries

PRONOUNS = {                 # word -> (who, grammatical position)
    "i": ("speaker", "subject"),
    "me": ("speaker", "object"),
    "she": ("F", "subject"),
    "he": ("M", "subject"),
    "her": ("F", "object"),
    "him": ("M", "object"),
}


def subjects_of(verb):
    """The subject tokens of a verb.

    In "I came and saw", the parser attaches "I" only to "came"; the second
    verb "saw" (dep conj) has no subject of its own, so it borrows the active
    subject of the verb it is joined to. Only active subjects are borrowed,
    and only across "and"-type joins (conj), never into "I want to leave":
    there "leave" is wanted, not done, so only "want" gets the frame."""
    own = [c for c in verb.children if c.dep_ in (SUBJECT, PASSIVE_SUBJECT)]
    if own or verb.dep_ != "conj" or verb.head.pos_ != "VERB":
        return own
    return [c for c in verb.head.children if c.dep_ == SUBJECT]


def frames_from_doc(doc):
    """All frames in one parsed line, as a list of dicts."""
    frames = []
    for verb in doc:
        # Main verbs only. spaCy tags auxiliaries and copulas (be, have, do,
        # will, can) as AUX, so "I am happy" and "I can" give no frame.
        if verb.pos_ != "VERB":
            continue
        children = list(verb.children)
        negated = int(any(c.dep_ == NEGATION for c in children))
        particles = " ".join(c.lower_ for c in children if c.dep_ in PARTICLES)
        candidates = subjects_of(verb) + [c for c in children if c.dep_ == OBJECT]
        for c in candidates:
            if c.lower_ not in PRONOUNS:
                continue
            who, position = PRONOUNS[c.lower_]
            if c.dep_ == SUBJECT and position == "subject":
                role = "agent"
            elif (c.dep_ == PASSIVE_SUBJECT and position == "subject") or \
                 (c.dep_ == OBJECT and position == "object"):
                role = "theme"
            else:                       # e.g. "her" as a subject is a parse error
                continue
            frames.append({
                "role": ("self_" if who == "speaker" else "ref_") + role,
                "who": who,
                "lemma": verb.lemma_.lower(),
                "particles": particles,
                "negated": negated,
                "verb_text": verb.text,
            })
    return frames
