# C1 (EuqVid-Visual) — first extraction run on AVA-ActiveSpeaker

Run date: 19 Oct 2026 · Hasangani WRA (IT23224384)

*Running record, oldest first. Later addenda correct earlier ones, and each
correction is also noted where it applies. Addenda 8 (MELD train and checks)
and 9 (close-up cut sweep) were added on 20 Oct. The scripts are in
`src/visual/`.*

## What was run

| | |
|---|---|
| Corpus | AVA-ActiveSpeaker, 88 videos (all videos with gender labels) |
| Annotation rows read | 1,983,456 |
| Entities (one person inside one 60 s window) | 21,930 |
| Windows | 1,310 |
| Video decoded | **none** — AVA ships `entity_box`, so C1 runs no face detector of its own (see the note below) |
| Runtime | ~11 s |

**The AVA boxes are not detector-free.** AVA-ActiveSpeaker's face boxes were
produced by the dataset authors' automatic face detector and tracker, with
annotators discarding false-positive tracks (Roth et al., arXiv 1901.01342). C1
adds no detector of its own but inherits that one: a face it missed was never
annotated, and the paper describes no step that adds misses. Detector recall by
gender on AVA is unmeasured, which matters most for the presence dimension. The
check still to run is a hand count of visible against annotated faces, by
gender, on a sample of frames.

`entity_id` encodes the 60-second window (`<video>_<start>_<end>:<n>`), so the
same person in two consecutive windows carries two different ids. Per-person
quantities are therefore computed inside a window, where an entity really is
one person, and only then aggregated.

## Finding 1 — the gender labels are not missing at random

`entity_gender.csv` labels are **predicted** by a face model
(`insightface buffalo_l`), not annotated by hand. Coverage:

| Label | Entities | Share |
|---|---|---|
| M | 11,554 | 52.7% |
| F | 3,991 | 18.2% |
| unknown (model could not decide) | 4,118 | 18.8% |
| missing (no row at all) | 2,267 | 10.3% |
| **Unusable** | **6,385** | **29.1%** |

M:F among labelled entities = 2.90 : 1.

The failures are strongly geometry-dependent:

| | mean face area | mean centre distance |
|---|---|---|
| F | 0.1349 | 0.3141 |
| M | 0.1033 | 0.3443 |
| unknown | 0.0864 | 0.3505 |
| missing | 0.0120 | 0.4189 |

**Labelled faces are 1.86× larger than unlabelled ones** (0.1114 vs 0.0600),
and unlabelled faces sit further from frame centre.

This matters because C1 measures exactly face size and centrality. A labeller
that fails on small peripheral faces removes the small end of each gender's
distribution before the comparison is made. The measurement and the labelling
error are confounded.

## Finding 2 — most of the apparent disparity does not survive

Mean log parity per window, `log(F/M)`, positive = women higher.
Filtering to entities large enough for labelling to be reliable:

| Feature | no filter | area ≥ 0.02 | area ≥ 0.05 | area ≥ 0.10 |
|---|---|---|---|---|
| screen time (per person) | −0.031 | −0.016 | −0.024 | +0.000 |
| face area | +0.147 | +0.092 | +0.025 | +0.027 |
| centrality | **+0.131** | **+0.114** | **+0.092** | **+0.080** |
| close-up rate | +0.364 | +0.233 | +0.124 | +0.004 |
| solo framing | +0.155 | +0.104 | +0.038 | +0.037 |

Face area, close-up rate and solo framing collapse toward zero. The
unfiltered "women are framed larger and get more close-ups" reading is
largely an artefact of small female faces being dropped as `unknown`.

**Caveat, stated rather than hidden:** the close-up figure at the 0.10
threshold is not evidence. Close-up is defined as area ≥ 0.08, so filtering
at 0.10 makes nearly every surviving face a close-up and the feature
degenerates (sd 0.070). The informative comparison for that feature is
no-filter → 0.05, which still shows a 66% reduction. The filters also change
which windows qualify, so the columns are not a like-for-like sample.

## Finding 3 — centrality is the one signal that holds

Centrality retains 61% of its effect at the strictest usable filter, and the
direction is consistent across films rather than driven by outliers:

- 83 videos have ≥ 3 mixed-gender windows
- **69 of 83 (83%)** place women closer to frame centre than men
- per-video mean +0.144, median +0.137

Screen time per person is flat at zero: no per-person screen-time disparity
in AVA.

## What this means for the project

1. AVA gender labels can be used, but only with the confound reported
   alongside every score. M2's convention already allows predicted labels
   flagged as noisy; C1 now supplies the measured size of that noise.
2. The submitted proposal states gender would be annotated manually. The
   labels in use are predicted. Either the proposal's methodology is
   revised with this audit as the justification, or a manually annotated
   subset is produced and the predicted labels validated against it.
3. This mirrors M2's own lesson: high agreement on convenient data is not
   validity. An independent check is required before any score is believed.

## Files

| File | Contents |
|---|---|
| `src/visual/c1_extract_ava.py` | Parses annotations, joins labels, writes per-entity and per-window geometry |
| `src/visual/c1_audit.py` | Label-coverage and confound audit (Finding 1) |
| `src/visual/c1_score.py` | Log-parity features and the robustness sweep (Findings 2–3) |
| `out/entities.csv` | 21,930 rows — one per person-window |
| `out/windows.csv` | 1,310 rows — raw per-gender quantities per window |
| `out/window_scores.csv` | 1,310 rows — parities, with NA rows kept and a reason given |

## Reproduce

```
python3 -I src/visual/c1_extract_ava.py <ava_ann_dir> entity_gender.csv out
python3 -I src/visual/c1_audit.py out/entities.csv
python3 -I src/visual/c1_score.py out/entities.csv out
```

## Not yet done

*(As of the first run, 19 Oct. Since then the manual validation subset was
done (Addendum 2), the sub-metric was built (Addenda 4 and 5), and MELD dev and
train were extracted (Addenda 6 and 8). AMI and the regression are still open.)*

- MELD extraction (needs MTCNN; `pilot_meld.py` is the working single-clip pilot)
- AMI extraction (needs the manual annotation download and M2's `gender_map.csv`)
- Manual validation subset against the predicted labels
- Augmentation operators and the regression equation

---

# Addendum — stratified test (same run, 19 Oct)

## Finding 4 — the centrality result is not a size artefact

Face size was the measured confound, so the test is to compare F and M
*inside narrow size bands*. Within a band both genders have essentially
the same face size, so size-dependent label dropout cannot produce a gap.

| area band | n_F | n_M | F dist | M dist | diff | Cohen d |
|---|---|---|---|---|---|---|
| 0.00–0.01 | 352 | 1274 | 0.3927 | 0.4048 | −0.0121 | −0.080 |
| 0.01–0.02 | 346 | 1672 | 0.3945 | 0.4106 | −0.0162 | −0.118 |
| 0.02–0.04 | 471 | 2004 | 0.3711 | 0.3964 | −0.0253 | −0.208 |
| 0.04–0.08 | 754 | 2136 | 0.3649 | 0.3771 | −0.0122 | −0.111 |
| 0.08–0.16 | 770 | 1773 | 0.3123 | 0.3184 | −0.0061 | −0.054 |
| 0.16–0.32 | 869 | 1890 | 0.2390 | 0.2456 | −0.0066 | −0.066 |
| 0.32–1.01 | 429 | 805 | 0.1879 | 0.1822 | +0.0057 | +0.074 |

Negative = women closer to frame centre. **6 of 7 bands agree**; only the
largest-face band reverses.

Within mixed-gender windows — people filmed in the same scene, at the same
moment, which removes film, genre and era as explanations:

- 819 windows
- mean (F dist − M dist) = **−0.0329**, 95% CI **[−0.0411, −0.0247]**
- women more central in 506 of 819 windows (62%)

That interval is a normal approximation that treats the 819 windows as
independent. Windows from one film share cast, director and lighting, so the
level at which observations are exchangeable is the video. Resampling the 87
videos instead (2,000 draws, seed 0; `c1_stratify.py` now prints it) gives
**[−0.0438, −0.0218]**. It is wider, as it should be, and still excludes zero.

## What this does and does not establish

**Established:** the centrality gap is not produced by the size-dependent
label dropout measured in Finding 1, and is not produced by differences
between films. It is consistent across size bands, across windows, and
across 83% of videos.

**Not established:** that the labels are sound. Stratification controls the
*size* pathway because size was measured. It does not control a *pose*
pathway: central faces are more often frontal, and if the labeller is more
willing to answer "female" on a frontal face, the same pattern appears with
no difference in how anyone is filmed. Only labels produced without a face
model can settle this — MELD's manual character map, or a hand-annotated
AVA subset.

**Effect size:** small. Cohen's d between −0.05 and −0.21 by band. The
result is robust in direction, modest in magnitude, and should be reported
that way rather than as a headline disparity.

## Consequence for the sub-metric design

Of the five candidate features:

| Feature | Verdict |
|---|---|
| screen time per person | flat at zero — no signal in AVA |
| face area | collapses under size filter — mostly artefact |
| close-up rate | collapses under size filter — mostly artefact |
| solo framing | collapses under size filter — mostly artefact |
| centrality | survives stratification — small but real |

One feature carries signal. A degree-2 polynomial over five features needs
21 coefficients and would be fitted against augmentation operators, which
is precisely the failure M2 documented: their learned composite scored
Spearman −0.28 against DOME while their simple time-parity measure scored
+0.465. Fitting a richer model to weaker features is the known way to lose.

Recommended: build the simple anchored score first, report the regression
as a comparison rather than as the deliverable. That is a change from the
submitted proposal and needs supervisor sign-off.

---

# Addendum 2 — manual validation (19 Oct, 167 blind labels)

A blind, size-stratified sample of 167 entities was hand-labelled by the
researcher. The predicted label was hidden and the order shuffled. There was
one annotator, so no inter-rater agreement (no κ) was measured, and the crops
come from 20 of the 88 videos. Every interval below treats faces as
independent, so each is narrower than a video-clustered interval would be.

## Q1 — how good is the face model?

| model says | you say F | you say M | you say U | total |
|---|---|---|---|---|
| F | 36 | 8 | 1 | 45 |
| M | 2 | 45 | 0 | 47 |
| unknown | 19 | 24 | 5 | 48 |
| missing | 6 | 20 | 1 | 27 |

Accuracy where both decided: **81/91 = 89.0%** (95% CI 80.9–93.9%, Wilson).

The error is **asymmetric**: when the model says female it is wrong 18.2%
of the time (8 of 44); when it says male, 4.3% (2 of 47). It over-calls
female. Accuracy by
size band (82.6 / 95.5 / 91.7 / 86.4%) shows no clear size trend among the
faces it is willing to label.

## Q2 — what are the 29% it refused?

| pool | truly female | 95% CI |
|---|---|---|
| predicted F | 81.8% | 68–90% |
| predicted M | 4.3% | 1–14% |
| unknown | 44.2% | 30–59% |
| missing | 23.1% | 11–42% |

The refused pools are substantially female. Projected onto the corpus,
the headcount disparity is overstated by the model but remains large:

- model labels only: 2.90 men per woman (log parity −1.063)
- corrected for refusals: **2.60 men per woman** (log parity −0.954)

## Q3 — the strongest test available: the gap from human labels

Centre distance computed from the human gender labels, weighted back to the
population by (predicted group × size band) cell. The gender values compared
are human, but the weights still use the model's predicted group as strata and
the boxes are AVA's, so the model is not entirely out of the chain:

| label source | F | M | gap |
|---|---|---|---|
| model, all 15,545 entities | 0.3141 | 0.3443 | −0.0302 |
| **manual, population-weighted** | **0.3246** | **0.3760** | **−0.0514** |

95% CI on the weighted gap: **[−0.0981, −0.0084]**, excluding zero
(`c1_decide.py`; entities are resampled within cells, not videos, and the 160
labels come from 20 videos, so a video-level interval would be wider). The gap
does not shrink with human labels: the point estimate is larger, though the
interval is wide enough to include the model-label value (−0.0302). The model's
errors did not create the effect.

## Feature verdict on the best available evidence

Weighted manual-label comparison, 2,000 bootstrap draws over the sampling
cells:

| Feature | F | M | gap | 95% CI | verdict |
|---|---|---|---|---|---|
| centrality | 0.3246 | 0.3760 | −0.0514 | [−0.097, −0.005] | supported |
| close-up rate | 0.4897 | 0.3049 | +0.1848 | [+0.073, +0.313] | supported |
| screen time (frames) | 108.2 | 82.0 | +26.3 | [+0.7, +53.5] | weak |
| face area | 0.1101 | 0.0940 | +0.0161 | [−0.016, +0.049] | not supported |
| solo framing | 0.6145 | 0.5694 | +0.0452 | [−0.116, +0.198] | not supported |
| identity (headcount) | — | — | 2.60 men per woman | — | supported, large |

Reproducibility of this table. The point estimates reproduce exactly when the
`c1_decide.py` weighting is applied to each column (join `sample_key.csv` to
`entities.csv`), but only the centrality row is computed by a committed script.
The intervals above came from a different random seed: re-run with seed 0 they
move in the second or third decimal (centrality [−0.0981, −0.0084], close-up
[+0.065, +0.313], face area [−0.016, +0.049], solo [−0.113, +0.205]), and the
screen-time interval becomes [−0.37, +52.8], which includes zero and fits its
"weak" verdict.

**Correction to Addendum 1.** Close-up rate was reported there as collapsing
under the size filter and therefore an artefact. That conclusion was drawn
from a weaker test: the filter changes the sample, and at the 0.10 threshold
the feature degenerates because close-up is itself defined at area ≥ 0.08.
The manual-label comparison is the better evidence and supports the feature.
Addendum 1's table should be read with this correction.

**An inconsistency that needs explaining, not papering over.** Mean face
area shows no difference while the rate of faces above the close-up
threshold does. That is possible if the two distributions differ in shape
rather than in mean — for example if women appear either in close-up or far
in the background, while men occupy the middle of the range more evenly.
This should be checked directly on the area distributions before either
feature is used in a score.

## Where the component stands

Two effects are supported, and they point in opposite directions:

1. **Headcount** — roughly 2.6 men per woman on screen. Large, and
   consistent with the published presence findings (GD-IQ, Annenberg,
   Mazières) that this project set out to extend.
2. **Framing** — the women who are on screen are framed *more*
   prominently: closer to centre, more often in close-up.

So in this corpus the disparity is one of presence, not of framing, and
conditioning on presence reverses the naive expectation. That is a
reportable result, and it is the opposite of the hypothesis the proposal
was written around.

Sample size is 160 usable manual labels from 20 of 88 videos. This
establishes direction, not the population magnitude.

---

# Addendum 3 — resolving the area / close-up inconsistency

Addendum 2 flagged that mean face area shows no significant difference
while the close-up rate does, and said this needed explaining. It does not
need explaining away: the two are not independent features.

## Within an entity, face size barely moves

Re-extraction now records the within-entity standard deviation of relative
area. Population-weighted, on manually labelled entities:

| | F | M |
|---|---|---|
| mean area | 0.1101 | 0.0940 |
| sd of area within the entity | 0.0100 | 0.0070 |
| coefficient of variation | 0.091 | 0.075 |

A person's on-screen size is nearly constant inside a 60-second window.
The earlier guess — that women are cut between close-up and wide shots —
is wrong.

## So the close-up rate is an indicator, not a rate

| | never close-up | always close-up | mixed |
|---|---|---|---|
| F | 48% | 48% | 4% |
| M | 69% | 27% | 4% |

Only 4% of entities are ever mixed. `closeup_rate` is therefore
effectively `1[mean_rel_area >= 0.08]` — a thresholded version of the same
quantity mean area measures, not a second dimension.

## Why the threshold version detects the effect and the mean does not

Relative face area is strongly right-skewed (population mean 0.135 against
median 0.086 for F). For a skewed variable the sample mean is a
high-variance estimator, while the proportion above a cut-point is robust.
That is the whole difference:

| estimator | gap | 95% CI |
|---|---|---|
| mean area | +0.0161 | [−0.016, +0.049] |
| proportion above 0.08 | +0.1848 | [+0.073, +0.313] |

Same underlying quantity, different statistical power. Not a contradiction.

## Consequences

1. **Area and close-up must not both enter a score.** They are two
   estimators of face scale. The threshold form is the better one here and
   should be the one kept; the choice is recorded with this evidence.
2. **The proposal's claim that these cover separate dimensions ("size" and
   "shot scale") does not hold at this clip unit.** It may hold at a longer
   unit, where one person genuinely is cut between close-up and wide. The
   claim is unit-dependent and should be stated that way.
3. The six nominal features reduce to **three measured dimensions**:

   | dimension | feature kept | status |
   |---|---|---|
   | headcount | identity parity | supported, large (2.60 men per woman) |
   | face scale | close-up share | supported (+0.185) |
   | centrality | centre-distance parity | supported (−0.0514) |

   screen time per person carries no signal; solo framing is not
   supported; mean area is the weaker estimator of a dimension already
   covered. (Superseded in part: Addendum 4 brings solo framing back, as a
   difference of proportions, so four dimensions are reported.)

A three-dimension feature set derived from measurement is a far better
basis for the sub-metric than six assumed ones, and the reduction is
itself a reportable finding.

---

# Addendum 4 — the sub-metric, and two parameterisation bugs found building it

## A silent, one-sided drop

Close-up and solo framing were first written as log parities,
`log(rate_F / rate_M)`. Any window where one gender has no close-up at all
gives `log(0)` and was dropped. Those windows are not missing at random:

| | windows |
|---|---|
| both genders have close-ups (scored) | 527 |
| **women have none** | **145** |
| men have none | 82 |
| neither has any | 65 |

36% of mixed-gender windows were dropped, and 64% of the one-sided drops
were the windows where women get no close-up — exactly the most
female-disadvantaged cases. The feature was reading +0.364 because its
worst cases had been removed.

## The log ratio is the wrong form for a bounded rate

With every window kept, the two parameterisations disagree:

| feature | log parity | 95% CI | rate difference | 95% CI |
|---|---|---|---|---|
| close-up | +0.0814 | [−0.135, +0.293] | **+0.0950** | [+0.065, +0.126] |
| solo | −0.1902 | — | **+0.0527** | [+0.020, +0.083] |

A log ratio diverges as either rate approaches zero, so the mean is
dominated by a handful of near-empty windows. A difference of proportions
has none of that behaviour. The rule now applied:

- **ratio-scale quantities** (headcount, screen time, area, centre
  distance) → log parity
- **bounded proportions** (close-up share, solo share) → difference

This is a feature-definition decision with measured evidence behind it, and
it belongs in the decision log.

## Visual-EuqVid v1

Bootstrap CIs resample **videos**, not windows: windows inside one film
share cast, director and lighting, so resampling windows would give
intervals several times too narrow. This is the same point M2 makes about
bootstrapping over series rather than shards.

| dimension | parity | 95% CI (by video) | score | tier |
|---|---|---|---|---|
| presence (headcount) | −0.6354 | [−0.789, −0.493] | **−4.58** | Moderate |
| centrality | +0.1310 | [+0.094, +0.169] | **+3.23** | Moderate |
| close-up share | +0.0950 | [+0.065, +0.126] | **+3.80** | Moderate |
| solo framing | +0.0527 | [+0.020, +0.084] | **+2.11** | Moderate |

All four intervals exclude zero. 819 mixed-gender windows, 87 videos.

Anchors are declared conventions, not measurements: 4:1 headcount, 1.5:1
centre distance, and a 25-point gap in either rate, each saturating the
±10 scale. (The 4:1 headcount anchor was derived afterwards: Addendum 5.)

## The dimensions are reported separately, not averaged

They disagree in sign. Presence is −4.58; all three framing dimensions are
positive. Averaging would return roughly +1 and call this corpus neutral,
which would be false in both directions at once.

**What the corpus actually shows:** women are outnumbered on screen by
about 2.6 to 1 (counting every labelled person; inside scenes that hold both
genders the typical ratio is 1.89 to 1, which is what the −4.58 above reads
back as: see Addendum 5), and the women who are on screen are framed more
prominently than the men — closer to centre, more often in close-up, more
often alone in shot. A disparity of presence, not of framing, and
conditioning on presence reverses the direction the proposal assumed.

## Not yet done

- MELD extraction (since done: Addenda 6 and 8; single-face clips, no predicted gender labels)
- AMI
- Augmentation operators and the regression, to be compared with this baseline

---

# Addendum 5 — the presence anchor, derived

The co-supervisor's objection: the anchor cannot be picked, it has to be
proved from published research. That is correct, and it is the component's
stated novelty — *derived* rather than inherited thresholds. To be exact about
what is derived here: the presence anchor, and through it where the tier
cut-points fall on the presence scale. The cut-points themselves (2 and 6) are
EuqAud's and are used unchanged; the other three anchors are conventions.

## The two sources

**USC Annenberg, Inequality in 800 Popular Films** (Smith, Choueiti &
Pieper). 800 top-grossing films, 2007–2015, 35,205 characters:
**male-to-female speaking characters = 2.30 : 1**, 30.3% female.

**29 C.F.R. § 1607.4(D)**, Uniform Guidelines on Employee Selection
Procedures — the four-fifths rule: *"A selection rate for any race, sex,
or ethnic group which is less than four-fifths (4/5) (or eighty percent)
of the rate for the group with the highest rate will generally be regarded
by the Federal enforcement agencies as evidence of adverse impact."*
That is a ratio of **1.25 : 1**.

## The derivation

Set the Moderate/Strong boundary at the measured industry norm:

```
anchor = exp(10 × ln(2.30) ÷ 6) = 4.0076  →  4.0
```

The value previously assumed. It is now derived.

Reading the tier boundaries back out at anchor 4.0:

| Boundary | Ratio | Selection rate |
|---|---|---|
| Neutral / Moderate (|score| = 2) | 1.320 : 1 | 0.758 |
| Moderate / Strong (|score| = 6) | 2.297 : 1 | 0.435 |

So **Strong begins where a corpus is more unequal than the measured
industry average**, and **Neutral ends at 0.758** — marginally more lenient
than the legal 0.80 adverse-impact line: a selection rate of 0.78 (1.28 : 1)
is flagged by the four-fifths rule and is Neutral here. Both boundaries sit
beside a published figure. *(An earlier version of this paragraph said
"stricter". That was wrong: 0.758 is below 0.80.)*

Placed on that scale, the −4.58 above is the window-mean parity over the 819
windows that hold both a woman and a man. It reads back as 1 woman : 1.89 men
inside such a scene (pooled over those windows, 1.81 : 1, which is −4.27 and is
rule B of Addendum 7). The corpus-wide ratios used elsewhere in this record,
2.90 : 1 on the model's labels and 2.60 : 1 corrected for its refusals, are a
different quantity: they count every labelled person, including the 491 windows
that lack a labelled woman or a labelled man (254 women and 4,804 men between
them). On this scale those two ratios would read −7.68 and −6.89, both Strong.
*(An earlier version of this paragraph equated −4.58 with 2.60 : 1 and said the
label correction moved the score a full tier. Both statements were wrong: the
two ratios are different quantities, and on the corpus-wide one the correction
moves the reading from −7.68 to −6.89, inside the same tier.)*

## How this differs from EuqAud's scaling

EuqAud (Section III-E) scales its raw output to ±10 by min–max scaling, with
bounds built from extreme feature settings: for a positive raw score the
maximum comes from minimising the female features and the minimum from setting
male equal to female, and the reverse for a negative score. The paper does not
say whether those bounds are recomputed for each dataset, so it does not support
the claim that adding a corpus rescales earlier scores. *(An earlier version of
this section made that claim.)* What can be said is how this component differs:
the anchor is fixed against external figures and nothing in the equation is
estimated from the corpus being scored, so a score cannot move when a corpus is
added. Whether EuqAud's bounds move is a question for the supervisor.

## Declared limits

1. The four-fifths rule is United States employment law. Applying it to
   screen representation is an analogy. The ML fairness literature uses
   that analogy routinely, which is precedent, not proof, and it is
   declared rather than hidden.
2. Annenberg counts **speaking characters**, which matches the presence
   dimension and nothing else. **Centrality, close-up and solo have no
   published norm and remain conventions.** Grounding them needs the
   human-perception study or the downstream-disparity work, both in C4 —
   so this is a group dependency, not something C1 can close alone.

Sources:
- Smith, Choueiti & Pieper, *Inequality in 800 Popular Films*, USC
  Annenberg Inclusion Initiative.
- 29 C.F.R. § 1607.4(D), Uniform Guidelines on Employee Selection
  Procedures (1978).

---

# Addendum 6 — MELD, the second corpus (dev split complete, 19 Oct)

MELD gender labels are a hand-written map of six characters, so **no gender
model** is involved, unlike AVA. A face detector (MTCNN if installed, else
YuNet; the output file does not record which ran) still finds the face. 952
clips of the main cast; 371 (39%) have exactly one face in at least 80% of
their sampled frames, and the visible face is taken to be the speaker (an
assumption; not spot-checked). 175 female against 196 male, across 65
episodes.

*Status: the dev numbers below are as first reported. Addendum 8 repeats them
with a fixed seed, adds the train split, and supersedes the reading of the
framing results.*

## The presence control passes

| | |
|---|---|
| main cast | 3 F, 3 M — identity parity exactly 0 |
| single-face clips | 175 F, 196 M — parity −0.113 |
| **presence score** | **−0.82 → Neutral** |

A metric that scored a 3F/3M ensemble as biased on counting alone would be
producing false positives. It does not. This checks the counting arithmetic and
nothing else: the cast is balanced by construction, so a score near zero is the
expected result, and it says nothing about framing. It is a control, not a
finding. *(An earlier version called this "the negative control" and "the
false-positive check the proposal promised", which claimed more than a presence
count can show.)*

## The framing results point the other way from AVA — on dev only

*Superseded by Addendum 8: on the train split the close-up gap shrinks from
−0.128 to −0.029 and its interval includes zero, so "point the other way from
AVA" should not be read as a finding.*

Bootstrap clustered by episode, 2,000 draws:

| MELD, women vs men | F | M | gap | 95% CI |
|---|---|---|---|---|
| face area | 0.0261 | 0.0295 | **−0.0034** | [−0.0063, −0.0005] |
| close-up rate | 0.183 | 0.311 | **−0.1284** | [−0.238, −0.005] |
| centre distance | 0.3087 | 0.2979 | +0.0107 | [−0.008, +0.029] |

In the sitcom women are framed **smaller** and get **fewer close-ups** —
18% against 31%. On AVA the direction was the reverse, and centrality,
which was AVA's most robust signal, shows nothing here.

## Why this is not yet a contradiction

The two comparisons are not the same design, and saying otherwise would
be the easy mistake:

- **AVA** compares women and men **inside the same 60-second window** —
  same scene, same lighting, same director's choice in that moment.
- **MELD** compares across single-face clips, which are **not matched for
  scene**. A woman's clip and a man's clip may come from different
  moments with different framing.

So MELD opens a question; it does not answer one. Reconciling them needs
a within-scene comparison on MELD — multi-face clips with identity, which
is the work the single-face shortcut deliberately avoided.

## A portability defect the second corpus exposed

MELD faces are smaller than AVA's: mean relative area 0.027 on the 3,404
single-face clips (dev and train) against 0.111 on AVA's labelled people, about
a quarter; median 0.027 against 0.059, about a half. The close-up threshold of
0.08, tuned on film, finds a close-up in **none of the 371 single-face dev
clips** (and in 16 of the 3,033 train clips) — the feature dies on contact with
a second corpus. *(An earlier version compared a MELD median, 0.0285, with an
AVA mean, roughly 0.11, which is not like for like.)*

For the numbers above the threshold was set at MELD's own upper quartile
(0.0338). That fixes the feature but changes its meaning, from "a face
larger than X" to "the largest-framed quarter of this corpus".

**A metric claiming to be portable across datasets cannot use an absolute
pixel threshold.** This is a defect in the feature definition, found only
by running a second corpus, and it has to be settled before any score is
frozen.

## Status

dev split complete. The train split has since been extracted: see Addendum 8.

---

# Addendum 7 — the pooling rule is a decision, not a detail (19 Oct)

Prompted by: C1 works per window / utterance, M2 works in seconds — will
that break the final metric?

## What M2's contract actually says

Source: `shared_conventions_M2_2026-10-06.md` — a **proposal**; the note
says nothing is final until the group confirms it.

- M2's row is one `clip_id` = **one meeting (AMI) or one episode (MELD)**,
  built from speaker intervals. Seconds are the unit of the *quantity*,
  not of the row.
- Shards: 30 clips each, 200 shards, drawn from a pool of 247 clips
  (165 MELD + 82 AMI), overlapping, bootstrap over series.
- Output contract: one row per `clip_id` with score(s), status
  valid / NA, a reason when NA, and `feature_version`. **Never a silent
  drop** — the same defect found in C1 (Addendum 4: 36% of windows lost
  one-sidedly).
- C1's atoms are finer: an AVA 60-second window, a MELD utterance
  (~3.6 s). Aggregation only goes fine → coarse, so the unit gap is not
  by itself a problem. What can differ is HOW the counts are summed.

## The pooling rule moves the score

One set of per-window counts, three rules, one shared set of video
bootstrap resamples (2,000 draws, `src/visual/c1_agg_check.py`). Rule A
reproduces `submetric.csv` exactly.

- **A** window-mean — mean of per-window parity (reported so far)
- **B** pooled — add every count first, take one parity (ratio of sums)
- **C** video-mean — pool inside each video, then average videos
  (the video as M2's "clip")

| score (−10…+10) | A | B | C | B − A, paired 95% CI | C − A, paired 95% CI |
|---|---|---|---|---|---|
| presence   | −4.58 | −4.27 | −5.43 | +0.32 [−0.13, +0.87] | **−0.84 [−1.47, −0.27]** |
| centrality | +3.23 | +2.89 | +3.00 | −0.34 [−1.09, +0.39] | −0.23 [−1.07, +0.55] |
| close-up   | +3.80 | +5.04 | +4.32 | +1.24 [−0.19, +2.56] | +0.52 [−0.64, +1.64] |
| solo       | +2.11 | +4.47 | +3.27 | **+2.36 [+0.90, +3.73]** | **+1.16 [+0.01, +2.34]** |

**Every dimension is Moderate under every rule, with the same sign.** The
qualitative finding does not depend on the rule. The magnitude does:
solo moves from 2.11 to 4.47, and under A it sits 0.11 above the
Neutral/Moderate cut at 2.

## Recommendation: B, pooled (ratio of sums)

1. The presence anchor was derived from a **pooled** ratio (Annenberg
   2.30 : 1 over 35,205 characters). The like-for-like estimator is
   pooled. (Annenberg is also not conditional on co-presence, so that
   comparison stays approximate.)
2. A gives a window with 40 face-frames the same weight as one with
   1,000. B weights by evidence.
3. C3 already stores counts and takes the ratio of sums. Counts
   re-aggregate exactly to any grouping — clip, shard, series, corpus —
   with no re-extraction.

Cost, stated: B is noisier on the two proportions. 95% CI width, B
against A: close-up 0.0890 vs 0.0606, solo 0.0918 vs 0.0652; presence
and centrality are about equal.

## One video read alone is not a reading

SD of the single-video score across 87 videos: presence 5.5,
centrality 4.8, close-up 7.8, solo 7.2 score points, against dataset
intervals of about ±1. A per-clip row is an intermediate. Same lesson M2
reports for AMI meetings.

## Coverage is the larger constraint on the fused metric

As of M2's note (6 / 15 Oct):

| corpus | C1 | C2 |
|---|---|---|
| AVA  | done — 87 videos | **NA** (Decision 6) |
| MELD | dev done (65 episodes); train done since, Addendum 8 | done — 165 episodes |
| AMI  | not started | done — 82 meetings |

Today a C1 + C2 row exists only on MELD, the presence control. M2 gives
three reasons for NA on AVA: no gender column, speaker ids only inside
60-second windows, only on-screen speakers annotated. C1's labels answer
the first only. Unless that changes, the fused score cannot be evaluated
on the corpus where bias is expected.

## Open decisions

1. Pooling rule for dataset / shard values — recommend B; the group
   needs ONE rule or components differ partly by arithmetic.
2. C1 emits, per `clip_id` (`AVA:<video>`, `MELD:S##E##`, `AMI:<meeting>`):
   summed counts per gender, status, reason, `feature_version`.
3. Can C2's allocation score be computed on AVA windows once C1 supplies
   per-person labels (`gender_source = predicted`, with confidence)?

---

# Addendum 8 — MELD train, the frozen cut, and three checks (20 Oct)

Addendum 6 reported the dev split. The train split has since been extracted by
the same script (`c1_meld_extract.py`), and both splits are analysed by
`c1_meld_replicate.py`, which applies one fixed procedure to each and is
committed so that the numbers can be re-derived. MELD is a presence control and
an exploratory look at framing, **not a finding**; the limits are at the end.

## What was extracted

| | dev | train | both |
|---|---|---|---|
| main-cast clips (the six leads) | 952 | 8,306 | 9,258 |
| single-face clips | 371 | 3,033 | 3,404 |
| of which women / men | 175 / 196 | 1,456 / 1,577 | 1,631 / 1,773 |
| episodes with a single-face clip | 65 | 176 | 180 |

A clip is single-face when at least 80% of its sampled frames (every 5th) hold
exactly one face. One train clip could not be read (`MELD:train:d125_u3`). The
test split was not run. Guest speakers are excluded by construction: only the
six leads are mapped to a gender. Single-face clips are the close-coverage
clips, 37% of the main-cast clips, not a sample of MELD.

Dev and train are **not** episode-disjoint: 80 of dev's 81 episodes also have
train clips.

## The procedure, fixed before train was looked at

1. Unit: one single-face clip. The visible face is taken to be the speaker, and
   the speaker's gender (a hand-written map of six characters) labels it. That
   assumption has not been spot-checked; a reaction shot would break it.
2. Gap = mean(women) − mean(men). Negative means women smaller, or fewer
   close-ups.
3. Close-up: the clip's `mean_rel_area` is at or above **0.033815**, the 75th
   percentile of dev single-face `mean_rel_area`, frozen on dev before train was
   run. (The AVA cut of 0.08 finds a close-up in 0 of 371 dev single-face
   clips and 16 of 3,033 train ones.) The share of single-face clips at or
   above the cut is 25.1% on dev and 22.7% on train.
4. Interval: 95% percentile bootstrap that resamples whole episodes, 2,000
   draws, seed 20261020. Clips from one episode share a set, a director and a
   camera plan, so they are not independent.
5. Presence score: 10 · clip( ln(nF / nM) ÷ ln 4.00 , −1, +1 ), the frozen
   presence form applied to the single-face counts.

## Results

| women vs men | dev | train | train, episodes not in dev | dev + train |
|---|---|---|---|---|
| single-face clips, F / M | 175 / 196 | 1,456 / 1,577 | 871 / 849 | 1,631 / 1,773 |
| presence score | −0.82 [−3.3, +1.6] | −0.58 [−1.7, +0.5] | +0.18 [−1.2, +1.7] | −0.60 [−1.6, +0.4] |
| face-area gap | −0.0034 [−0.0065, −0.0006] | −0.0013 [−0.0026, −0.0002] | −0.0014 [−0.0029, +0.0001] | −0.0016 [−0.0026, −0.0004] |
| close-up gap (top-quarter cut) | −0.1284 [−0.2460, −0.0078] | −0.0294 [−0.0766, +0.0185] | −0.0316 [−0.0880, +0.0284] | −0.0403 [−0.0868, +0.0066] |
| close-up share, women / men | 18.3% / 31.1% | 21.2% / 24.1% | 21.8% / 25.0% | 20.8% / 24.9% |
| centre-distance gap | +0.0107 [−0.0074, +0.0296] | +0.0001 [−0.0081, +0.0084] | −0.0009 [−0.0106, +0.0087] | +0.0013 [−0.0063, +0.0084] |

The dev point estimates equal those in Addendum 6. The intervals differ slightly
from it, because the Addendum 6 bootstrap was run ad hoc and this script fixes
the seed. The "episodes not in dev" column is check A below.

Reading it:

- **Presence control.** Both splits score Neutral as point estimates
  (−0.82, −0.58); the intervals are wide and straddle zero. This tests counting
  only.
- **Close-up.** The dev gap (−0.128; 18% of women's clips against 31% of men's) is
  −0.029 on train (21% against 24%), and the interval now includes zero. Scored
  by the close-up rule with the same cut, the dev gap alone would read about
  −5.1 (Moderate) on a balanced cast and train −1.2 (Neutral). The sentence in
  Addendum 6 that the framing results "point the other way from AVA" is
  superseded.
- **Face area** keeps its direction (women's single-face boxes are smaller): the
  gap falls from −0.0034 to −0.0013; on train women's mean is about 5% below men's.
- **Centre distance** shows nothing in either split (+0.0107 on dev, +0.0001 on train;
  both intervals include zero).

## Three checks, run after the plan above

**A. Train restricted to episodes that do not appear in dev** (the third column above): 871 women and 849 men in 100 episodes. Face area −0.0014 [−0.0029, +0.0001], close-up −0.0316 [−0.0880, +0.0284], centre distance −0.0009 [−0.0106, +0.0087], presence score +0.18 [−1.2, +1.7]. Nothing reaches significance, and the direction of face area and close-up is unchanged.

**B. Within-episode gap (post hoc).** Women and men are compared inside the same episode, then averaged with weight nF·nM/(nF+nM), over episodes that have both genders. This removes season and director from the comparison.

| | face area | close-up | centre distance | episodes |
|---|---|---|---|---|
| dev | −0.0028 [−0.0056, −0.0004] | −0.1105 [−0.2326, −0.0023] | +0.0126 [−0.0086, +0.0312] | 33 |
| train | −0.0020 [−0.0031, −0.0008] | −0.0481 [−0.0949, −0.0000] | +0.0039 [−0.0039, +0.0113] | 150 |
| dev + train | −0.0021 [−0.0031, −0.0011] | −0.0559 [−0.0995, −0.0139] | +0.0048 [−0.0020, +0.0119] | 157 |

Inside an episode the female-smaller direction persists, and on dev + train the close-up and face-area intervals exclude zero. This is post hoc, and it does not answer check C.

**C. Character permutation.** MELD has three women and three men. Relabelling which three of the six characters are "F" in all C(6,3) = 20 ways, and asking how unusual the real labelling is, gives a two-sided p that cannot go below 2/20 = 0.10. Rank 1 means the most negative of the 20 labellings:

| | face area | close-up | centre distance |
|---|---|---|---|
| dev | −0.0034 · p = 0.10 · rank 1 of 20 | −0.1284 · p = 0.20 · rank 2 of 20 | +0.0107 · p = 0.50 · rank 16 of 20 |
| train | −0.0013 · p = 0.30 · rank 3 of 20 | −0.0294 · p = 0.60 · rank 6 of 20 | +0.0001 · p = 1.00 · rank 11 of 20 |

The spread between characters is several times the gap between the genders: on train the close-up share runs from 0.151 (Monica) to 0.304 (Phoebe), both women, against 0.209 (Chandler), 0.218 (Joey) and 0.292 (Ross) for the men. That is a spread of 0.153 against a gender gap of 0.029. With three women and three men, no gap can be told apart from six characters being framed differently.

## Limits

1. Dev and train share episodes, so train is a replication on other dialogues,
   not on other episodes. Check A restricts to episodes not in dev.
2. Three women and three men: no gender gap is separable from character
   (check C).
3. The single face is assumed to be the speaker. Not spot-checked.
4. The detector's error by gender has not been audited. The detector is MTCNN
   if installed, otherwise YuNet; the output file does not record which one ran.
5. Single-face clips are the close-coverage clips and are not matched for scene.
6. The test split was not used.

## Reproduce

```
python  src/visual/c1_meld_extract.py <MELD.Raw dir> out/meld_clips.csv
python3 -I src/visual/c1_meld_replicate.py out/meld_clips.csv out/meld_replication.csv
```

`out/meld_replication.csv` holds the estimates and intervals above. (The
extractor needs MELD.Raw and a face detector; the second script needs neither.)

---

# Addendum 9 — how much the close-up score depends on the cut (20 Oct)

The close-up dimension counts a face as a close-up when its box covers at least
`CLOSEUP_AREA` of the frame. The value 0.08 is a convention (Addendum 4), and
Addendum 6 showed that it does not travel to MELD. This addendum asks how far
the AVA close-up score moves with it.

**Method.** The unchanged chain (`c1_extract_ava.py`, `c1_features6.py`,
`c1_submetric.py`) was run 13 times on the same annotations and labels, editing
only `CLOSEUP_AREA`. The cut enters the close-up dimension alone: the presence,
centrality and solo rows are identical, to every printed digit, at all 13 cuts.
Bootstrap and seed are as in Addendum 4 (2,000 draws, seed 0, resampling
videos). The 0.08 row equals `submetric.csv`.

| cut | parity | 95% CI (by video) | score | score 95% CI | tier |
|---|---|---|---|---|---|
| 0.02 | +0.0658 | [+0.0431, +0.0885] | +2.63 | [+1.72, +3.54] | Moderate |
| 0.03 | +0.0968 | [+0.0696, +0.1239] | +3.87 | [+2.78, +4.96] | Moderate |
| 0.04 | +0.1095 | [+0.0810, +0.1393] | +4.38 | [+3.24, +5.57] | Moderate |
| 0.05 | +0.0972 | [+0.0664, +0.1286] | +3.89 | [+2.66, +5.14] | Moderate |
| 0.06 | +0.0915 | [+0.0608, +0.1237] | +3.66 | [+2.43, +4.95] | Moderate |
| 0.07 | +0.0918 | [+0.0629, +0.1226] | +3.67 | [+2.52, +4.90] | Moderate |
| **0.08** | +0.0950 | [+0.0648, +0.1262] | +3.80 | [+2.59, +5.05] | Moderate |
| 0.09 | +0.0881 | [+0.0570, +0.1212] | +3.52 | [+2.28, +4.85] | Moderate |
| 0.10 | +0.0833 | [+0.0518, +0.1177] | +3.33 | [+2.07, +4.71] | Moderate |
| 0.12 | +0.0785 | [+0.0474, +0.1117] | +3.14 | [+1.90, +4.47] | Moderate |
| 0.15 | +0.0718 | [+0.0437, +0.1026] | +2.87 | [+1.75, +4.11] | Moderate |
| 0.20 | +0.0445 | [+0.0211, +0.0695] | +1.78 | [+0.84, +2.78] | Neutral |
| 0.30 | +0.0270 | [+0.0053, +0.0488] | +1.08 | [+0.21, +1.95] | Neutral |

**Reading.** The sign is positive and the interval excludes zero at every cut
from 0.02 to 0.30. The tier is Moderate for cuts up to 0.15 (scores +2.63 to +4.38)
and Neutral from 0.20 (+1.78 at 0.20, +1.08 at 0.30). That follows from the
definition: as the cut rises, fewer faces of either gender qualify, both rates
fall toward zero, and so must their difference. The 0.08 used throughout sits
inside the Moderate range, but the close-up *tier* is a property of the
convention; what survives is the sign. A corpus-relative cut (a quantile) is the
fix proposed in `FROZEN_EQUATION.md` section 7; this sweep shows how far the
score can move with whichever cut is chosen.

**Reproduce.** Edit `CLOSEUP_AREA` at the top of `c1_extract_ava.py`, then re-run
`c1_extract_ava.py`, `c1_features6.py` and `c1_submetric.py` (run order in
`src/visual/README.md`). Only the close-up row changes. It needs the AVA
annotation archive and `entity_gender.csv`, which are not in the repository.
