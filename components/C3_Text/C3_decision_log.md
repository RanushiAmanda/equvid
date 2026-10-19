# C3 (EuqVid-Text) — decision log

Owner: Fernando R. S. R. (IT23449282). Newest first. Each entry has the decision, the
evidence behind it, and who still has to agree. Features are named, not numbered: the
F-numbers in the specification and the AVA decision note do not agree.

**Feature set for the C3 equation after these decisions (6):** `mean_utterance_length`,
`agency_score_mean`, `power_score_mean`, `same_gender_dialogue_share`, `ref_log_parity`,
`sentiment_valence_mean` (tool not yet validated). At degree 2 that is 28 coefficients,
the per-component cap.

---

## D1 · 19 Oct 2026 · `dialogue_volume_share` (F1) leaves the C3 equation

**Decision.** F1 is still computed and reported, as `volume_log_parity`, but only as a
pipeline check. It does not go into the C3 equation.

**Why.**

1. **It measures what C2 already measures.** Across clips, words per person and C2's
   speaking time per person correlate at r = 0.96 on MELD (173 episodes) and r = 0.97 on
   AMI (82 meetings). Women and men speak at almost the same rate: the speech-rate
   difference (words parity minus time parity) is +0.017 on MELD and −0.001 on AMI.
2. **The fusion would count speaking share twice.** This is the same rule that removed
   `utterance_count_share`, which duplicated C2's `turn_count_share`.
3. **There is no room for it.** The equation is capped at 6 features (28 coefficients at
   degree 2), and 7 features would need 36.

**Evidence.** `results/fig2_words_vs_time.png` and `results/c3_lexical_correlations.csv`.

**Status.** Decided by the C3 lead. C2 (Harischandra) to be told, because after this
speaking allocation reaches the fusion only through C2.

## D2 · 19 Oct 2026 · `ref_log_parity` replaces `gendered_reference_ratio`

**Decision.** The reference feature is
`ref_log_parity = log((R_F / n_F) / (R_M / n_M))`.

- R_g counts the he/she-type pronouns that point at gender g, whoever says them.
- n_g is the number of people of gender g.

It measures who gets talked about, per person.

**Why.** The documented version (he/she as a share of all person pronouns in each
gender's speech) shows no difference between women and men:

| Corpus | F − M | 95% CI |
|---|---|---|
| MELD | +0.008 | −0.004 to +0.023 |
| AMI | +0.004 | −0.007 to +0.014 |

The two obvious alternatives reduce to other quantities:

- *Whom each gender talks about* (cross-gender share, F − M) equals
  P(male ref | F) + P(male ref | M) − 1. That is the referent mix again: r = −0.92 with
  `ref_log_parity`.
- *Talked about rather than talking* (the old `mention_ratio` idea) equals
  `ref_log_parity` − F1 exactly, so it would bring C2's signal back in.

**Result.**

| Corpus | `ref_log_parity` | 95% CI | Meaning |
|---|---|---|---|
| MELD | −0.244 | −0.393 to −0.105 | women referred to 22% less per person |
| AMI | −0.254 | −0.808 to +0.312 | too few he/she pronouns to say anything (about 2 per 1,000 words) |

**Caveat.** The definition was chosen after seeing MELD and AMI, so results on these two
corpora are exploratory. The confirmatory test must use AVA or held-out shards.

**Status.** Adopted by the C3 lead. The group still has to confirm, because the earlier
F6 rename was declared a group-level change.

## D3 · 19 Oct 2026 · Chance level for `same_gender_dialogue_share` is weighted by turns

**Decision.** "Same gender by chance" assumes that the previous speaker is drawn from the
other people in proportion to their turns, not their headcount.

**Why.** With headcount as chance, the feature tracked floor share (r = 0.95 with F1 on
AMI). With turn-weighted chance, the correlation drops to −0.20 on MELD but is still 0.78
on AMI.

**Status.** C3 method choice. Still open: keep or drop the feature. It can be computed on
only 34% of AMI meetings, because a 1:3 meeting has no second person of one gender.

## D4 · 19 Oct 2026 · Which lines count

**Decision.**

- Turn adjacency uses lines of 3 or more words, the shared turn rule: a "mm-hmm" is
  feedback, not a reply.
- `mean_utterance_length` keeps every line, because full lines versus short reactions is
  exactly what it measures. The 3+-word version is reported as a sensitivity check
  (r = 0.85 MELD, 0.81 AMI with the main version).

**Status.** C3 method choice.

---

## Earlier decisions

| When | Decision | Record |
|---|---|---|
| Oct 2026 | LSMDC removed: only one member can use it, and every corpus must be usable by all four | `data/README.md` |
| Sep 2026 | AVA becomes the third C3 corpus. `mention_ratio` becomes `gendered_reference_ratio`, because AVA has no cast list (now superseded by D2) | `docs/C3_AVA_decision_and_F6_change.md` (Claude project) |
| before Sep 2026 | `utterance_count_share` deleted: it duplicates C2's `turn_count_share` | `components/C3_Text/README.md` |
