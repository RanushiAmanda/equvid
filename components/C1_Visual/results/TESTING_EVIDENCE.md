# C1 (Visual) testing evidence

Status at PP1, 22 October 2026. This page lists what has been checked, how, and what has not been checked. Numbers come from `RESULTS.md` and the scripts in `src/visual/`.

## 1. Reproducibility

- Re-running the AVA chain (`c1_extract_ava.py` → `c1_features6.py` → `c1_submetric.py`) from the raw annotations reproduces `entities.csv`, `windows.csv`, `features6.csv` and `submetric.csv` byte for byte. Only three printed sentences differ.
- The Visual-EuqVid calculator (`demo/`) reproduces the four slide scores from the same equation: presence −4.58, centrality +3.23, close-up +3.80, solo +2.11.

## 2. Are the gender labels usable? (blind validation)

AVA ships no gender labels. The labels in use come from a face model that declined to label 29% of people, and the faces it declined are smaller and farther from the centre.

| Check | Result |
|---|---|
| Sample | 167 crops hand-labelled blind (prediction hidden, order shuffled), 160 judgeable, from 20 of 88 videos, one annotator |
| Design | Equal draws from 4 face-size bands × 4 pools (female, male, unknown, missing), weighted back to population share |
| Accuracy where model and annotator both decided | 81 of 91 = 89.0% (95% CI 80.9 to 93.9) |
| Error asymmetry | 18.2% of the model's "female" labels are wrong, against 4.3% of its "male" labels |
| Truly female, by pool | predicted female 81.8%, predicted male 4.3%, unknown 44.2%, missing 23.1% |
| Accuracy by size band | 82.6 / 95.5 / 91.7 / 86.4% |

Limits: one annotator, so no agreement statistic (kappa); 20 of 88 videos; intervals treat faces as independent.

## 3. Can label errors fake the centrality result?

- Within each of 7 face-size bands, the direction agrees in 6 of 7. Across films, 69 of 83 videos agree.
- Recomputed from human labels alone: gap −0.0514 (95% CI −0.0981 to −0.0084), against −0.0302 on model labels. The gap is larger with human labels. Limit: the weights still use the model's predicted pools, and the interval includes the model value.

## 4. Defects found in the score while building it

1. Written as log(rate_F / rate_M), any window where a gender had no close-up dropped silently: 292 of 819 mixed windows (36%), of which 145 had no close-up for women and 82 for men. Fixed.
2. A log ratio is the wrong form for a bounded rate. Rule now: log parity for ratio-scale quantities, difference for proportions.

## 5. Sensitivity

- **Pooling rule:** tiers do not change under window mean, pooled ratio of sums and video mean. Magnitudes do (solo goes from 2.11 to 4.47 when pooled). Open decision with Temporal.
- **Close-up cut:** sign positive and interval above zero at every cut from 0.02 to 0.30. Tier is Moderate up to 0.15 and Neutral from 0.20.
- **Resampling:** the bootstrap resamples videos, not windows, because scenes in one film share a cast and director.

## 6. Second corpus (MELD)

Gender is a hand-written map of six characters (three women, three men), so no gender model is involved. Single-face clips: 371 dev, 3,033 train. Presence control scores −0.82 (dev) and −0.58 (train), both Neutral. The dev close-up gap (−0.128) shrinks to −0.029 on train, with an interval that includes zero. MELD is a control, not a finding.

## 7. Not tested yet

- Scale invariance (same content at 1×, 5×, 10× volume), promised in the proposal.
- Detector recall by gender, on AVA's inherited boxes and on the MELD detector.
- Second annotator and kappa on the 167-face check.
- Whether the single face in a MELD clip is the speaker.
- MELD test split. AMI.
- The fitted regression, which is planned as a comparison with v1.
