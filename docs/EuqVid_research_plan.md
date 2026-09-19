# EuqVid — Research Execution Plan

Group J26-DS-343 · CoEAI · SLIIT · IT4010
Supervisor: Dr. Prasanna S. Haddela · Co-supervisor: Ms. Adya Dissanayake

**Status: draft for group agreement.** Nothing here is binding until all four members
and the supervisor have read it. Section 9 contains three findings that change what
the group previously agreed; read that section before the next meeting.

---

## 1. How to read this

**This plan is sequenced by dependency, not by date.** It is a set of gates. A gate
opens when its exit test passes, and work downstream of a gate does not start before
it opens. Drop your faculty's milestone dates onto the gates once you have them — the
ordering will not change.

Three kinds of item appear:

| Marker | Meaning |
|---|---|
| **[G]** | Group work. Blocks everyone. Cannot be delegated to one person quietly |
| **[Cn]** | That member's own work. Assessed individually |
| **[?]** | Not yet decided. Needs a name against it before work starts |

**The one rule that matters:** nobody extracts a feature before Gate 0 closes. Every
hour spent extracting against an unfrozen schema is an hour that gets thrown away.

---

## 2. Where the project stands

Proposals are submitted. The component structure is settled: four modalities, four
independent closed-form sub-metrics, one fused score.

What exists: corpus downloads (AVA partial, AMI ES-series, MELD tarballs still
compressed), the MELD speaking-time baseline, four component specifications, the
repository.

What does not exist yet: a single extracted feature, the identity-to-gender map, any
shard definition, the augmentation protocol, the calibration set, and an owner for
fusion and validation.

**That last one is the most dangerous gap in the project.** All four members are on a
modality. Fusion, cross-modal validation, the perception study and the dashboard
currently belong to nobody, and unowned work is unassessed work.

---

## 3. Gate 0 — what must close before anyone extracts

This is the critical path. Everything in section 4 waits on it. Ten items, roughly in
the order they unblock each other.

### G0.1 — Freeze the feature schema [G]

Every feature name, its exact definition, its units, and whether it is a mean or a
standard deviation. Signed by all four.

*Why it blocks:* a column renamed in month four costs four people a re-extraction.

*Exit test:* a versioned file in `docs/` that all four have committed against, and a
synthetic CSV conforming to it that every member's pipeline reads without modification.

*Already settled and not to be reopened:* means rather than standard deviations for
spatial, temporal and textual features. EuqAud's std rationale applies only to
intrinsically gender-linked acoustic quantities, which is why C4's `f0_std`,
`energy_std` and `amplitude_std` keep it and nothing else does. Argue this in your own
words in your own thesis — do not cite EuqAud as the reason.

### G0.2 — Settle the shard contract [G]

**Read section 9.1 first.** The ≥200-shard target is inherited from a design that no
longer exists and the arithmetic has changed.

*Exit test:* a written rule stating what one shard is, what the identity-partitioning
guarantee is, what the minimum clips-per-gender is, and the expected shard count per
corpus. Plus a shard manifest — an actual list of shard IDs with their member clips —
generated and committed before extraction.

### G0.3 — Build one identity-to-gender map [G]

One map. All four components resolve identity to gender through it.

*Why it blocks:* four private maps mean the four sub-scores describe four different
populations and cannot be fused. This is not a stylistic preference; it makes the
fusion arithmetically invalid.

*Exit test:* one file, one schema (`corpus`, `identity_id`, `gender`, `source`,
`confidence`), covering every identity in every shard, with the excluded/unknown share
reported per corpus.

Sources differ per corpus and each needs its own defence: MELD from character names
with a documented policy for the 260-speaker long tail; AMI from participant metadata;
AVA by hand (G0.4); LSMDC from shipped annotations if it ever arrives.

### G0.4 — Annotate AVA gender by hand [G]

The largest single labour item in the project, and it has no owner.

AVA-ActiveSpeaker ships `entity_id` but no gender. You label a person, not a frame.
Sort tracks by duration within each film and label the long ones first, because they
dominate every prominence feature.

*Scale, estimated from the corpus's own figures:* ~39K face tracks across 160 videos,
roughly 244 tracks per film. Labelling all of them is not happening. Labelling the top
20 per film is 3,200 tracks; at 20–30 seconds each that is roughly 18–27 hours in
total, or **4.5–7 hours per person split four ways**. That is feasible. Labelling the
top 30 per film roughly doubles it.

*Exit test:* every labelled track carries an annotator ID; a 10% subset is
double-annotated by a different member; Cohen's κ is computed and reported. If κ is
poor, the labels are the finding, not the features.

*Owner:* **[?]** — split four ways by film range, agreed in the meeting. Whoever
consolidates and computes κ should be named, because that is the part that gets
forgotten.

### G0.5 — Freeze one augmentation protocol [G]

One protocol for all four components. Four private label definitions cannot be fused.

The protocol must state: the manipulation level (source/clip resampling with
re-extraction, not arithmetic on the feature table — the circularity defence depends
on this), the label definition (resulting disparity state, not applied reduction
percentage), symmetry in both directions, the reduction depth cap, and the number of
stochastic replicates per level.

C2's three structural operators — O1 Fragmentation, O2 Interruption injection, O3
Temporal clustering — hold total speaking time constant. That property is what makes
C2's baseline comparison decisive by construction, and the group protocol must not
break it.

*Exit test:* one module in `src/shared/`, imported by all four pipelines, that turns a
shard into a labelled family of perturbed shards.

### G0.6 — Define one shared calibration set [G]

The four sub-scores are each min-max normalised to [−10, +10]. If each is normalised on
its own data, +6 from C1 does not mean the same disparity as +6 from C3, and the fusion
is adding incommensurable numbers.

*Exit test:* a named subset of shards, fixed and versioned, that all four components
normalise against. Held out of every component's fitting data.

### G0.7 — Run the coverage pilot [G]

Ten clips per corpus, every feature, before committing to anything. Three questions per
feature: can I compute it here at all, does it mean the same thing on every corpus, and
does it vary.

*Exit test:* a table with one row per feature and one column per corpus, filled in.

### G0.8 — Write the drop rule before seeing results [G]

A feature is dropped if it fails coverage, if VIF exceeds 10, or if its variance is
near zero. Written down and committed **before** G0.7's results are read.

*Why:* dropping features after seeing which ones perform well is researcher degrees of
freedom, and an examiner will name it as such. Committing the rule first costs nothing
and defends the whole feature-selection story.

### G0.9 — Name the integration lead [G] [?]

Fusion, cross-modal validation, the perception study and the audit dashboard have no
owner. One of the four takes them, and those phases go into the group thesis under that
name.

This is a scope question, not an admin one: whoever takes it is carrying the group
deliverable on top of their own component, and the group should decide whether that is
compensated by reducing their component's scope.

### G0.10 — Resolve the TAF [G] [?]

The signed TAF says "using visual and temporal features," which does not describe a
four-modality project. Confirm with Dr. Haddela whether it is revisable, and fix the
errors regardless of the answer: the false claim that EuqAud named video as future work
(it says "multimodal data," Section VI), the dataset count, the independence claim, the
per-corpus gender metadata claim, and the ethics wording that denies processing
identifiable data while processing faces and voices.

**Gate 0 exit test, all ten items:** a synthetic feature CSV conforming to the frozen
schema flows end to end through one member's pipeline — augmentation, screening, fit,
freeze, normalise, tier, bootstrap — and produces a number. The number is meaningless.
The point is that the pipeline runs before real data exists.

---

## 4. The four tracks

Once Gate 0 closes, the four run in parallel and do not block each other.

### 4.1 The common spine

Every component follows the same eleven steps. Keep them identical so the four theses
are comparable and the fusion receives four columns of the same shape.

1. Extract raw features per shard per gender
2. Convert everything to shares or ratios — no raw magnitude survives
3. Apply the shared augmentation protocol to produce labelled perturbed shards
4. Screen features **inside** cross-validation folds, never on the full set beforehand
5. Fit polynomial Ridge, degree capped at 2, grid-search alpha only
6. Freeze the equation in raw feature units, folding scalers into published coefficients
7. Normalise to [−10, +10] on the shared calibration set as an explicit closed form
8. Derive Neutral/Moderate/Strong from stated disparity levels, not inherited from EuqAud
9. Bootstrap 1,000× resampling at shard and identity level, never row level
10. Generalisation: leave-one-corpus-out CV, monotonicity along held-out perturbation
    curves, scale-invariance at 1× / 5× / 10×, Mahalanobis applicability-domain check
11. Hand off the sub-score column, equation, constants and thresholds

**Report nominal n and effective n.** Effective n is independent shards, not augmented
rows. Being caught quoting the augmented row count is worth negative marks, and
volunteering the distinction before being asked is worth positive ones.

### 4.2 C1 — Visual · Hasangani WRA

Features: `screen_time_share`, `mean_relative_face_area`, `mean_centre_proximity`,
`closeup_share`, `solo_frame_share`, `identity_share`.

Track-specific work, in order:

- **Detector-disparity measurement.** Measure MTCNN/RetinaFace detection rate and
  confidence by apparent gender, pose, occlusion and lighting on a hand-verified
  sample. This is not optional polish — "your bias metric depends on a biased detector"
  is the question you will be asked, and a measured answer beats a hedge.
- **Exploit AVA's ground truth.** `entity_box` is already normalised and `entity_id`
  already links people across frames, so on AVA you compute from annotation rather than
  detection, and detector bias is removed for the corpus that carries most of the shard
  count. Say this explicitly; it is a genuine strength.
- **Measurement validity.** Test whether `mean_relative_face_area` actually tracks
  human-judged visual prominence on a hand-labelled subset. A feature that does not
  measure what it claims is worse than a missing feature.
- **`closeup_share` needs a threshold.** Define what counts as a close-up in terms of
  relative face area, justify the cut, and show the result is not sensitive to it.
- Deliver `screen_time_share` to C2 early — `presence_voice_gap` depends on it.

### 4.3 C2 — Temporal · Harischandra RASR

Features: `speaking_time_share`, `turn_count_share`, `mean_turn_length`,
`interruption_asymmetry`, `speech_dispersion`, `presence_voice_gap`. Film-only extras
`mean_shot_duration` and `cut_share` stay outside the frozen equation.

Track-specific work, in order:

- **Expect to lose one of the first three features.** Total time = turns × mean length,
  so VIF will be severe. Decide in advance which two you keep and why, so the drop reads
  as a prediction confirmed rather than a scramble.
- **`interruption_asymmetry` needs an operational definition.** What counts as an
  interruption, in seconds of overlap or in turn-boundary terms, and how it is detected
  on each corpus. AVA-ActiveSpeaker gives frame-level speaking labels for two speakers
  at once, which makes overlap directly measurable — use that rather than inferring.
- **`speech_dispersion` needs a statistic.** Gini, entropy over timeline bins, or
  variance of inter-utterance gaps. Pick one, justify it, and show the ranking is stable
  under the alternatives.
- **AMI role confound.** Scenario meetings assign roles and the project manager speaks
  more by design. Check role-to-gender distribution *before* drawing any conclusion from
  AMI, and report the distribution whatever it shows.
- `presence_voice_gap` is the feature no other component can see. It is your strongest
  single claim — a character on screen constantly and silent. Give it its own analysis
  rather than burying it in a coefficient table.

### 4.4 C3 — Text · Fernando RSR

Features: `dialogue_volume_share`, `mean_utterance_length`, `agency_score_mean`,
`power_score_mean`, `same_gender_dialogue_share`, `gendered_reference_ratio`,
`sentiment_valence_mean`.

Track-specific work, in order:

- **The AVA ASR pass is on the group critical path, not just yours.** AVA is the primary
  fusion corpus and ships no dialogue. Until the transcript exists, the fused score on
  AVA is three-modality. Schedule this early and tell the group when it lands.
- **Quantify ASR word error against a hand-checked subset** and report it with every AVA
  text result. "We estimated features from an estimated transcript" is not an answer on
  its own.
- **Decide training vs held-out for AVA** [?]. Held-out keeps ASR off the fitting path
  and reads as a deliberate firewall; training gives a third fitting fold. This is a
  group decision because it changes what the fusion is fitted on.
- **Connotation-lexicon OOV rate.** The lexicon was built on screenplays; MELD is
  informal overlapping TV dialogue. Measure subject-verb coverage and report it. If
  coverage is poor, `agency_score_mean` and `power_score_mean` weaken and the component
  falls back to volume and interaction features — know that in advance.
- **`gendered_reference_ratio` is new and unvalidated.** It replaced `mention_ratio` for
  portability. Show on a hand-checked sample that it captures what `mention_ratio` was
  meant to capture, or the change reads as convenience rather than improvement.
- Retire `C3_dialogue_corpus_shortlist.md` or mark it superseded — it still argues for
  MovieNet and TVQA, both dropped.

### 4.5 C4 — Audio · Sathsarani WTTS

Contribution: `mean_voice_loudness`, `speech_to_background_ratio`, `dynamic_range`.
Baseline, transferred from EuqAud: `f0_std`, `energy_std`, `amplitude_std`.

Track-specific work, in order:

- **Write the contribution/baseline split down before the first progress presentation.**
  Dr. Haddela supervised EuqAud and AMI is inside its training base. Being able to say
  which three features are yours and which three are the transferred baseline, without
  hesitating, is the whole defence of this component's independence.
- **`speech_to_background_ratio` is the strongest claim** precisely because it is
  structurally impossible in EuqAud's data — Common Voice and LibriSpeech are people
  reading into a microphone in a quiet room. Lead with it.
- **Separating voice from background is the hard engineering step.** A film mix is not
  separable by thresholding. Decide early whether you use a source-separation model,
  restrict measurement to AVA-Speech's clean-speech regions, or both — and note that
  restricting to clean speech systematically excludes the loud scenes where the effect
  may be largest.
- **Design for a null result.** No academic prior work exists on gender and film sound
  mixing. If the effect is absent, "we measured it and it is not there, on this corpus,
  with this method" is a publishable finding — but only if the design anticipated it.
  Pre-register your expected effect direction and the power you have to detect it.
- **AMI is an unmixed corpus.** Meeting audio has no sound designer, so
  `speech_to_background_ratio` means something different there. Either exclude AMI from
  that feature or reinterpret it as room acoustics, explicitly.

---

## 5. Gate 1 — the unified feature dataset

*Exit test:* one versioned CSV, every shard, every gender, every feature from all four
components, no nulls that are not documented, and the shard manifest reconciling against
it exactly. Each component has also frozen its own equation and can score a shard
independently.

Nothing in section 6 starts before this.

---

## 6. Phase 3 — fusion [?]

1. Collect four sub-scores per shard.
2. Fit the fusion over the four scores, including pairwise interactions.
3. **Also fit one single equation over all features directly, as the comparison.** Late
   fusion cannot see cross-channel patterns — a woman on screen constantly but silent is
   invisible to any single modality. Reporting both is a result either way: if late
   fusion wins, the four-component structure is vindicated; if the single equation wins,
   that is an honest finding about the architecture.
4. Handle the reduced form. If C3's AVA transcript does not land, AVA scores on three
   modalities, and you must show the reduced form tracks the full one on corpora where
   both are computable.
5. Freeze EuqVid.

---

## 7. Phase 4 — validation [?]

- Compare against three baselines — face-count ratio, screen-time ratio,
  speaking-time ratio — using Kruskal-Wallis effect size. **Compare against** EuqAud's
  published figure; do not pre-commit to beating it.
- The face-count baseline calls MELD unbiased by construction (3F:3M). So if EuqVid
  finds disparity in MELD it cannot have come from counts. That single experiment answers
  "why not just count faces?" with a demonstration instead of an argument.
- Train downstream classifiers on subsets spanning the disparity range. MELD ships
  emotion labels; AVA-ActiveSpeaker ships active-speaker labels. Measure F1 disparity per
  subset and correlate against EuqVid tiers.
- **Run the mitigation experiment.** Rebalance the worst subsets by the score and
  re-measure the harm. This tests whether the metric is *actionable* rather than merely
  descriptive, and it is the thing EuqAud never did.
- Cross-check against AMI's DOME dominance annotations — independent human judgement on
  real data, free, no recruitment.
- Human perception study, 15–20 participants, clips spanning the range. Confirm SLIIT's
  ethics position rather than asserting that clearance is not required.
- Streamlit audit dashboard with SHAP per-feature attributions.

---

## 8. Phase 5 — write-up

Four individual theses, one group thesis. The group thesis carries Gate 0, fusion and
validation under the integration lead's name.

**Report the feature reduction as a finding, not an apology.** "We started with eight
textual features and kept six after collinearity screening inside folds" is a result
about the feature space. Written defensively it reads as failure; written plainly it
reads as method.

---

## 9. Three findings that change what was previously agreed

### 9.1 The ≥200-shard target is unreachable, and no longer necessary

Under strict identity partitioning — no character spans two shards — a shard cannot be
smaller than the unit over which characters recur.

| Corpus | Ceiling | Why |
|---|---|---|
| AVA-ActiveSpeaker | ~160 | 160 videos, each a different film with a different cast. Sub-sharding a film breaks identity disjointness, because its characters recur throughout |
| AMI (ES2002–ES2008 on disk) | ~7 | `ES2002a`–`d` are the same four people. Seven series, seven identity groups |
| MELD | ~1 | Six leads recur across nearly every episode. Strictly partitioned, all of MELD collapses into roughly one shard |
| **Total** | **~168** | Before dropping any shard that fails the minimum-clips rule |

So ~168, not 200+ — and about 95% of it from a single corpus.

**But the target no longer matters, because the coefficient budget collapsed.** The
200-shard figure was set when EuqVid was one equation over 12 features, needing 91
coefficients. Under the four-component split each equation covers about 6 features:

| Features | Degree-2 coefficients | Shards per coefficient at n=168 |
|---|---|---|
| 12 (old single-equation design) | 91 | 1.8 — non-viable |
| 7 | 36 | 4.7 |
| **6 (current per-component)** | **28** | **6.0** |
| 5 | 21 | 8.0 |

Six shards per coefficient is thin but defensible for a degree-2 Ridge fit, and it is
three times better than the design this project started from. **The four-component
restructure solved the shard problem.** Say so — it is a real methodological argument
for the architecture, not just a project-management convenience.

*Action:* replace the ≥200 target with a per-component budget of at most 6 features
after screening, and state the shard ceiling honestly in every thesis rather than
quoting an aspiration nobody reached.

### 9.2 The minimum-clips-per-gender rule biases the sample against the signal

The contract drops any shard with fewer than 30 clips per gender. Applied to AVA, where
one shard is one film, that drops films with few female speaking roles.

Those are exactly the films carrying the disparity the metric exists to detect. The rule
systematically removes the high-signal tail and shrinks the dynamic range of the training
set toward parity.

*Action:* lower the threshold, or make it asymmetric (a minimum on the *minority* gender
low enough to retain skewed films), or keep the rule and report the excluded films as a
named limitation with their count and their measured skew. An examiner who spots this
unprompted will make far more of it than one who reads your own account of it.

### 9.3 Leave-one-corpus-out CV is close to meaningless as currently planned

With ~95% of shards from AVA, holding out AVA leaves roughly eight shards to fit on, and
holding out anything else changes almost nothing.

*Action:* report LOCO anyway, with the shard counts visible, and lean on
**leave-one-film-out** within AVA as the real generalisation test. 160 films with
disjoint casts is a genuine held-out structure — arguably better evidence than LOCO
across three corpora of wildly unequal size. Reframe it as a strength rather than
reporting a LOCO number that cannot mean much.

---

## 10. Risk register

| | Risk | Consequence | Response |
|---|---|---|---|
| R1 | **AVA gender annotation stalls** | Blocks all four components. The single largest schedule risk | Split four ways in the first meeting, cap at top-20 tracks per film, track completion weekly |
| R2 | Gate 0 slips | Everything slips, because nothing downstream can start | Gate 0 is parallel across four people; its minimum viable output is two corpora, and tracks can start on partial data |
| R3 | **Fusion and validation stay unowned** | A whole phase goes unassessed and the group thesis has a hole | G0.9. Name someone before extraction starts, not after |
| R4 | C3's AVA transcript does not land | Fused score on the primary corpus drops to three modalities | Plan the reduced form now; show it tracks the full form elsewhere |
| R5 | C4 finds no effect | A component with no positive result | Design for a null result from the start; pre-register direction and power |
| R6 | Connotation lexicon has poor coverage on MELD | C3 loses its two strongest features | Measure OOV early; fall back to volume and interaction levels |
| R7 | Detector disparity is large | C1's features inherit a biased detector | Compute on AVA ground-truth boxes where possible; measure and report the disparity elsewhere |
| R8 | LSMDC never arrives | Loss of the only corpus with shipped gender labels | Already non-load-bearing. Keep it that way |
| R9 | Gender is binary and derived | A declared limitation of the whole project | Exclude ambiguous and non-binary identities rather than forcing them; report the excluded share per corpus |
| R10 | TAF cannot be revised | The signed document describes a different project | G0.10. Resolve with the supervisor before writing the thesis, not after |

---

## 11. Working rhythm

- **One shared feature CSV, versioned.** Never four private copies.
- **Logbook entries as decisions land**, not retrospectively at the end. D-002 — the
  dataset set differing from the signed TAF — is the highest-priority outstanding entry.
- **Group meeting cadence** with Gate 0 completion as a standing agenda item until it
  closes.
- **Anything that changes the shared schema, the gender map, the augmentation protocol
  or the calibration set is a group decision**, announced in writing, never a local fix.
  C3's F6 redefinition is the worked example: it started as one component's problem with
  AVA and ended as a change every component's feature list must pick up.

---

*Prepared for Group J26-DS-343. Feature definitions, sequencing and thresholds are
proposals for group agreement, not settled fact. The arithmetic in section 9 is computed
from corpus figures already on record in this repository; verify the corpus figures
themselves against their sources before quoting them in the TAF or a thesis. No results
in this document are our own measurements.*
