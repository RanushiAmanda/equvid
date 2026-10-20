# C3 results 03 — sentiment (F8): the model check, and why F8 is dropped

Owner: Fernando R. S. R. (IT23449282) · Run: 20 Oct 2026, on the C3 PC · Code:
`src/text/sentiment_check.py`

**The pass criteria were fixed before the run.** They are decision D6 in
`../C3_decision_log.md` and reached GitHub in commit `80e28cb` at 20:34 +0530. F8 stays
only if all three checks pass.

**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`, used as it is.

**Lines scored:** all 13,707 MELD lines, plus the 23,253 AMI lines that can enter the
feature (verified gender, scorable clip, 3+ words). The swap test adds 2,572 swapped
copies of MELD lines.

## The three checks

| Check | Measure | Result | Criterion | Passed |
|---|---|---|---|---|
| 1 Accuracy on MELD's human labels | accuracy | **60.6%** | ≥ 52% | yes |
| | macro-F1 | **0.58** | ≥ 0.50 | yes |
| | for reference | majority class 46.9%; VADER 43.0% / 0.41 | | |
| 2 Gender-word swap (2,572 MELD lines) | the label changes | **5.9% of lines** | ≤ 3% | **no** |
| | mean valence shift, female version minus male version | −0.014 | within ±0.02 | yes |
| 3 Agreement with the human labels (MELD) | model women − men | −0.019 [−0.036, −0.004] | inside the human 95% CI | yes |
| | human labels women − men | −0.006 [−0.059, +0.033] | | |

**Decision: F8 is dropped.** Check 2 failed, and the rule fixed in advance requires all
three to pass.

## What the checks say

**The model is a real improvement on VADER.** It gets 61% of MELD's human labels right,
against 43% for VADER and 47% for always answering "neutral".

| Human label | Called negative | Called neutral | Called positive |
|---|---|---|---|
| negative | 45% | 39% | 17% |
| neutral | 12% | 72% | 17% |
| positive | 11% | 30% | 60% |

**But it reacts to gender words.** Swapping only he/she, man/woman, the six main names and
similar words changes its answer on 5.9% of lines. On average it scores the female version
of a line 0.014 lower than the male version.

That is about the same size as the women-minus-men difference the feature is meant to
measure on MELD (−0.019). So a gender difference reported by this tool could come from
the tool, not from the script. A bias metric must not inherit a biased instrument: this
is the same argument the project makes about face detectors in C1.

**The model and the human labels agree on direction and rough size** (check 3). The human
labels alone are too noisy to show a difference: their interval runs from −0.059 to
+0.033.

## For the record (not used anywhere)

The model's women-minus-men valence is:

| Corpus | Difference | 95% CI |
|---|---|---|
| MELD | −0.019 | −0.036 to −0.004 |
| AMI | −0.034 | −0.063 to −0.003 |

Given check 2, these numbers cannot be read as differences in the dialogue.
`c3_sentiment_summary.csv` labels them `dropped`.

## Consequence for C3

C3 now has five features:

- `mean_utterance_length`
- `agency_score_mean`
- `power_score_mean`
- `same_gender_dialogue_share`
- `ref_log_parity`

At degree 2 that is 21 coefficients, under the 28 cap.

Sentiment goes to future work, with a concrete test any replacement must pass: the same
three checks, in particular the gender-word swap.

## Limitations of the check itself

- **The thresholds (3% flips, ±0.02 shift) are judgement calls.** They were written down
  before the run so they could not be tuned afterwards. With a 6% limit the model would
  have passed. That is exactly why the limit was fixed first.
- **The swap list leaves out her/him/his,** because "her" maps to both "him" and "his".
  The full swap effect may therefore be larger.
- **MELD's annotators also saw the video and heard the audio,** so no text-only tool can
  reach 100% on these labels.

## Files

- **This folder:** `c3_sentiment_check.csv` (every measure, with its criterion) and
  `c3_sentiment_summary.csv`.
- **`Research Project/derived/c3/`:** `c3_sentiment_scores.csv.gz` (per-line
  probabilities) and `c3_sentiment_stats.csv`.
