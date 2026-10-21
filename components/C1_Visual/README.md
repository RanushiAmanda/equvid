# C1 — EuqVid-Visual

**Hasangani WRA (IT23224384)** · group leader

Measures how much of the picture each gender gets.

**What belongs here:** proposal drafts, the C1 component specification, C1
diagrams, and the running results record. Code is in `src/visual/`.

## Where it stands

| Corpus | Status |
|---|---|
| AVA-ActiveSpeaker | done — 87 videos, 819 mixed-gender windows |
| MELD | dev and train done (371 + 3,033 single-face clips; test split not used). A presence control, not a finding |
| AMI | not started — contrast corpus, needs seat-based identity matching |

## Six features were specified; measurement reduced them to four dimensions

Presence (headcount), centrality, close-up share and solo framing. Mean face
area is a weaker estimator of the same quantity as close-up share, and screen
time per person carries no signal. The reasoning and the numbers are in
[`results/RESULTS.md`](results/RESULTS.md).

## Visual-EuqVid v1

A frozen closed form with no fitted parameters:

```
score_d = 10 · clip( parity_d ÷ div_d , −1, +1 )
```

Tiers: |score| < 2 Neutral, 2–6 Moderate, > 6 Strong. Constants, anchors and
their sources are in [`results/FROZEN_EQUATION.md`](results/FROZEN_EQUATION.md).
The proposal's polynomial Ridge equation is not frozen; it stays in the plan as
the comparison.

Sign: positive means women ahead. That is Temporal's proposal; EuqAud's
convention is the opposite (positive = male-dominant bias), and the group has
not decided which to use.

AVA, window-mean (positive = women higher; 95% CI bootstrapped over videos):

| Dimension | Parity | 95% CI | Score | Tier |
|---|---|---|---|---|
| presence | −0.635 | [−0.789, −0.493] | −4.58 | Moderate |
| centrality | +0.131 | [+0.094, +0.169] | +3.23 | Moderate |
| close-up | +0.095 | [+0.065, +0.126] | +3.80 | Moderate |
| solo | +0.053 | [+0.020, +0.084] | +2.11 | Moderate |

The dimensions are reported separately, never averaged: they disagree in sign.
The pooling rule behind these numbers is an open group decision (RESULTS.md,
Addendum 7). Tiers do not change under the three rules tested; magnitudes do.

## What C1 owes the group

- **Per-clip counts** keyed by `clip_id` (`AVA:<video>`, `MELD:S##E##`,
  `AMI:<meeting>`) with status, reason and `feature_version`, as the shared
  conventions propose. Not yet delivered.
- **AVA gender labels** are predicted by a face model, not annotated by hand.
  29.1% of entities have no usable label. A blind check of 167 faces (160
  usable, from 20 of the 88 videos, one annotator, no inter-rater agreement)
  gave 89.0% accuracy where model and human both decided (95% CI 80.9–93.9%,
  treating faces as independent), with errors concentrated on the female label.
  See RESULTS.md, Finding 1 and Addendum 2.
- **AVA face boxes** are the dataset authors' (automatic detector and tracker),
  so C1 adds no detector of its own but inherits theirs. Detector recall by
  gender is unmeasured.

## Layout

- `results/RESULTS.md` — running record, Findings 1–4 and Addenda 1–9
- `results/FROZEN_EQUATION.md` — the equation, its constants and sources
- `results/submetric.csv` — the four dimensions on AVA
