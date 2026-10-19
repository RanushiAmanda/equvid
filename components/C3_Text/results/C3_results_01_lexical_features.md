# C3 results 01 — text tables and the four parser-free features

Owner: Fernando R. S. R. (IT23449282) · Run: 19 Oct 2026 · Feature version: `c3-lex-0.1`
Status: **first results, provisional.** The gender map and the clip unit follow C2's
shared-conventions proposal (6 Oct), which the group has not yet confirmed.

## What was run

| Step | Script | Output (in `Research Project/derived/c3/`) |
|---|---|---|
| Utterance tables, gender map, clip list | `src/text/build_utterances.py` | `utterances_meld.csv`, `utterances_ami.csv`, `gender_map_provisional.csv`, `clips_provisional.csv` |
| Counts per clip and gender, features, bootstrap | `src/text/lexical_features.py` | `c3_clip_gender_stats.csv`, `c3_lexical_features.csv` |
| Charts | `src/text/plot_lexical.py` | `fig1`–`fig3` in this folder |
| Hand-worked tests | `src/text/test_text_features.py` | 3 tests pass |

- **MELD:** 13,707 utterances (1 empty line dropped), 190 episodes. Clip = one episode,
  pooled over train/dev/test. The six leads are mapped by hand; guests are excluded:
  **16.7% of utterances and of words**.
- **AMI:** 83,077 utterances from the manual transcripts, 171 meetings. 10,560 segments
  containing only laughs, coughs or abandoned words were dropped, and 19,238 truncated words.
  Gender comes from `participants.xml`. The 40 TNO participants are labelled `id_prefix`
  and kept out of the main analysis, as C2 does.
- **Scored:** 173 MELD episodes and 82 AMI meetings. Every other clip has a row with an NA
  reason (single gender, unverified labels, fewer than 5 utterances per gender).

## Pipeline checks against numbers the group already has

| Check | Ours | Group |
|---|---|---|
| AMI mixed-gender meetings with verified labels | 82, plus 4 with `id_prefix` labels (TS3005a–d) | C2: 82, or 86 with the unverified 4 |
| MELD main-cast speaking time, M:F | 1.092 | 1.094 |
| AMI person-connected groups among the 82 | 16 | earlier count: 16 |
| Words recounted from the raw files | MELD S08E21: exact. AMI ES2002a: within 0.6% (AMI writes acronyms as `T_V_`, counted per letter) | — |

## Results (pooled per corpus, 95% CI, 2,000 bootstrap resamples of series)

Positive means higher for women. Log ratios: −0.08 ≈ 8% lower for women.

| Feature | MELD | AMI |
|---|---|---|
| F1 words per person, log(F/M) | −0.080 [−0.180, +0.018] | −0.043 [−0.333, +0.222] |
| F3 words per utterance, log(F/M) | −0.011 [−0.046, +0.023] | −0.110 [−0.262, +0.041] |
| Same-gender replies above chance, F − M | +0.041 [+0.010, +0.080] (163 clips) | −0.003 [−0.064, +0.065] (28 clips) |
| Proposed: references as he/she per person, log(F/M) | **−0.244 [−0.393, −0.105]** | −0.254 [−0.808, +0.312] |
| `gendered_reference_ratio` as documented, F − M | +0.008 [−0.004, +0.023] | +0.004 [−0.007, +0.014] |

The MELD series are its 9 seasons. The AMI series are its 28 four-meeting teams.
Full table: `c3_lexical_corpus_summary.csv`. Correlations: `c3_lexical_correlations.csv`.

## Findings — each one needs a decision

**1. F1 duplicates C2 and should leave the C3 equation.** Words per person track speaking
time per person at r = 0.96 (MELD, 173 clips) and r = 0.97 (AMI, 82 clips) (fig2). This is
the reason `utterance_count_share` was dropped earlier: in the fusion it would count C2's
signal twice. Recommendation: keep F1 only as a pipeline cross-check.

**2. The same-gender reply feature needed a better definition of chance.** The first
version used headcount as chance, (n_g − 1)/(N − 1). It tracked floor share almost
exactly (r = 0.95 on AMI): when women hold the floor, a woman tends to follow a woman
whatever the script does. Chance is now weighted by how many turns each person takes. The
link with floor share falls to r = −0.20 on MELD, but stays at **0.78 on AMI** (only 28
meetings). It is defined on 34% of AMI meetings, because 1:3 meetings have no second
person of one gender. On MELD, women follow women 4.1 points more often than men follow
men, beyond chance.

**3. `gendered_reference_ratio` as written carries no signal, and two of the obvious
fixes reduce to something else.**
- *As documented* (he/she as a share of all person pronouns in each gender's speech):
  the F − M difference is about 0 on both corpora, with a very narrow spread.
- *"Whom does each gender talk about"* (cross-gender share, F − M) equals
  P(male referent | woman speaking) + P(male referent | man speaking) − 1. Algebraically
  it measures how male the pool of people being talked about is (r = −0.92 with the
  referent feature).
- *"Talked about rather than talking"* (the old `mention_ratio`) equals
  referent parity − F1, exactly. So it brings C2's signal back in.
- **Proposal:** `ref_log_parity` = log((refs to women ÷ women) / (refs to men ÷ men)),
  counting he/she-type pronouns by anyone. It is roster-free, per person and directional.
  It is unrelated to volume in MELD (r = 0.01, but only 31 episodes have enough
  pronouns). On MELD, women are referred to **22% less per person**, and the CI excludes
  zero.

**4. Pronoun features only work at shard level.** He/she pronouns occur about 18 times
per 1,000 words in MELD and about 2 per 1,000 in AMI. The referent feature is computable
on a single clip for only 18% of MELD episodes and 15% of AMI meetings (fig3). The scripts
store counts per clip and gender, so shard values are exact sums whatever shard design the
group adopts.

**5. A first answer to "why is text not redundant?"** On MELD, the negative control, floor
share is close to parity (consistent with C2), yet women are talked about less. Treat this
as **exploratory**. The referent feature was chosen after seeing these two corpora, so the
confirmatory test must run on data not used here: AVA, or held-out shards.

## Limitations to state

- **Gender labels:** binary labels taken from the datasets. MELD guests are excluded, so
  their lines are not scored.
- **Gender map:** provisional (`gender_map_provisional.csv`). It is replaced by the shared
  file once decision 7 is agreed.
- **What a MELD clip is:** the dialogues MELD sampled from an episode, not the whole
  episode.
- **Independence:** there are 17 independent identity groups (16 in AMI, 1 in MELD). The
  CIs resample only 9 MELD seasons and 28 AMI teams, so they are rough.
- **Pronoun noise:** "they/them" sometimes means things, and "you" is sometimes generic.
  In AMI, a generic "he" for a user or customer may inflate male references. AMI's he/she
  counts are too small to say anything about references.
- **Same-gender replies:** turn adjacency approximates who answers whom (spec risk R5).

## Decisions needed (C3 lead first, then the group where marked)

1. Drop F1 from the C3 equation, keep as a check. *(Group: affects the fusion.)*
2. Replace `gendered_reference_ratio` with `ref_log_parity`. *(Group: the F6 rename was
   declared a group-level change.)*
3. Same-gender replies: keep with activity-weighted chance, or drop given 34% AMI coverage
   and the residual link with floor share.
4. Shard design must pool counts (C3 needs many clips per shard for pronoun features).
