# C3 results 02 — agency and power (spaCy + Sap et al. lexicon)

Owner: Fernando R. S. R. (IT23449282) · Run: 20 Oct 2026, on the C3 PC
(Python 3.13.3, spaCy 3.8.16, en_core_web_sm 3.8.0) · Feature version: `c3-ap-0.1`

**The definitions were fixed before the run.** They are decision D5 in
`../C3_decision_log.md`. The code and D5 reached GitHub in commit `3e95eb8` at 20:00 +0530,
two minutes before the parse report was written at 20:02.

## What was run

| Step | Script | Result |
|---|---|---|
| Parse | `src/text/parse_frames.py` | 22,918 of 96,784 lines contain I / me / she / he / her / him and have 2+ words. They were parsed in 40 s, giving 30,734 frames |
| Score | `src/text/agency_power_features.py` | Lexicon scores, pooled per corpus, 2,000 bootstrap resamples of series (MELD: 9 seasons, AMI: 28 teams) |

Frames by role:

| Corpus | Speaker is subject ("I quit") | Speaker is object ("he fired me") | "she"/"he" is subject | "her"/"him" is object |
|---|---|---|---|---|
| MELD | 4,802 | 463 | 724 | 362 |
| AMI | 22,993 | 419 | 816 | 155 |

## Coverage (specification deliverable D5)

These figures cover the speaker's own verbs: "I" as the active subject, lines by speakers
with a verified gender, in scorable clips.

| | MELD women | MELD men | AMI women | AMI men |
|---|---|---|---|---|
| Verbs | 1,892 | 2,117 | 4,586 | 6,876 |
| Agency label in lexicon | 92.6% | 91.5% | 92.8% | 92.0% |
| Power label in lexicon | 89.6% | 88.4% | 91.2% | 90.3% |
| Negated (left out) | 18.7% | 18.9% | 15.3% | 13.5% |
| Discourse-marker verbs among the rest | 30.0% | 25.6% | 55.4% | 52.2% |

Coverage is almost the same for women and men, so missing verbs do not tilt the comparison.
The most frequent verb missing from the lexicon is *have*.

## Results (women minus men; each verb scores −1, 0 or +1; 95% CI)

| Feature | Role | MELD | AMI |
|---|---|---|---|
| **Agency of the speaker's own verbs** | equation | −0.057 [−0.137, +0.027] | −0.027 [−0.134, +0.090] |
| **Power of the speaker's own verbs** | equation | +0.006 [−0.029, +0.051] | −0.002 [−0.055, +0.046] |
| Agency, negated verbs kept | check | −0.028 [−0.100, +0.039] | −0.026 [−0.134, +0.087] |
| Power, negated verbs kept | check | −0.012 [−0.041, +0.015] | +0.015 [−0.033, +0.059] |
| Agency without marker verbs | check | −0.037 [−0.099, +0.026] | −0.049 [−0.139, +0.035] |
| Power without marker verbs | check | −0.031 [−0.097, +0.048] | +0.009 [−0.082, +0.100] |
| Agency of "she" vs "he" as subject | exploratory | **−0.132 [−0.253, −0.013]** | +0.108 [−0.109, +0.279] |
| Power of "she" vs "he" as subject | exploratory | +0.010 [−0.162, +0.163] | −0.091 [−0.309, +0.104] |
| Power as object ("me", "her", "him") | exploratory | **+0.096 [+0.048, +0.153]** | −0.018 [−0.287, +0.216] |

Group means for scale: MELD agency is +0.158 for women and +0.215 for men.
Full table: `c3_agency_power_summary.csv`. The verbs behind the numbers are in
`c3_agency_power_verbs.csv`.

## Findings

**1. The speakers' own lines give women and men the same agency and power.** Both
equation features are close to zero, and every interval includes zero, in both corpora.
For MELD this is what a negative control should show.

**2. The checks fixed in advance agree.** Keeping negated verbs, or dropping the seven
discourse-marker verbs, leaves the conclusion unchanged.

The marker check was worth declaring. Markers make up more than half of AMI's "I + verb"
frames: "I think" alone appears 1,220 times for women and 1,734 times for men. In MELD
they are a quarter to a third.

**3. One exploratory pattern, not a finding yet.** When other people describe a character
in MELD, verbs with "she" as the subject carry less agency than verbs with "he":
−0.13 [−0.25, −0.01]. That is the same direction as Sap et al. (2017) report for film
scripts. Power as object also differs in MELD (+0.10).

Two of the 14 exploratory and check intervals exclude zero, both in MELD. With 14
intervals, about one would do so by chance, and the "she"/"he" analysis rests on about
500 verbs. Treat it as a hypothesis to test on AVA, where it was never seen.

**4. Shards must pool counts here too.** With at least 5 verbs per gender required, a
single clip can be scored for 93% of AMI meetings but only 51–53% of MELD episodes.

## Parser check: 100 random frames

`c3_frame_check_sample.csv` holds 50 random "I + verb" frames per corpus (seed 20261020),
each with its full line. Claude read every line and judged whether "I" really is the
subject of the extracted verb and whether the negation flag is right.

**Result: 95 of 100 are correct** (MELD 48/50, AMI 47/50). Every negation flag was right.

The 5 errors:

- **3 elliptical replies** ("I don't.", "neither do I", "No, I didn't."), where the parser
  took auxiliary *do* for a main verb;
- **1 verb joined by "and"** that took "I" when its real subject was someone else;
- **1 lemma error** ("overheard"), which only makes the verb miss the lexicon.

Five more frames are grammatically right but semantically weak: "I'm gonna / going to"
counts as the verb *go*, which scores 0 for agency.

**This needs a second rater before it is quoted.** The `author_check` column is left
empty for the C3 lead to fill in, which turns it into a two-rater check.

## What this means for the C3 equation

Agency and power enter the equation as defined in D5:

- they can be computed on every corpus;
- lexicon coverage is high;
- coverage is equal for women and men.

A near-zero value on MELD is not a reason to drop them. Their job is to detect disparity
where it exists.

What is **not yet shown** is that they *can* detect it. That needs two tests:

- the controlled augmentation step (Night 3): change the lines of one gender by a known
  amount and check that the feature moves;
- AVA (film).

Expect an examiner to ask exactly this.

## Limitations to state

- **Parser quality.** en_core_web_sm is a small model trained on written text, and AMI is
  disfluent speech. About 5% of frames are wrong (see the parser check above). The most
  common error, elliptical *do*, could be filtered out, but that would be a rule changed
  after seeing data, so it is not done yet.
- **Subjects.** Only pronoun subjects are used. Named subjects ("Rachel decided") are left
  out, so the rule stays roster-free and works on AVA.
- **Reported speech.** In "he said 'I quit'", the "I" is attributed to the speaker.
- **Lexicon.** It was crowd-rated on film scripts and is applied here to TV dialogue and
  meetings (specification risk R4). Coverage is high, but labels such as *know* =
  low agency are the lexicon's, not ours.
- **AMI roles.** The role confound is not adjusted: the project manager speaks more by
  design.
- **Confidence intervals.** The MELD CIs rest on only 9 seasons.

## Files

- `Research Project/derived/c3/`: `c3_verb_frames.csv.gz` (every frame),
  `c3_parse_report.txt`, `c3_agency_power_stats.csv` (sums per clip × gender, for shards)
  and `c3_agency_power_clip.csv`.
- This folder: `c3_agency_power_summary.csv`, `c3_agency_power_coverage.csv`,
  `c3_agency_power_verbs.csv`, `c3_frame_check_sample.csv` and `fig4_agency_power.png`.
