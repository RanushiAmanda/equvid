# EuqVid — Parallel Component Restructure (Option A)

Group J26-DS-343 · CoEAI · Supervisor: Dr. Prasanna S. Haddela
Proposal to remove sequential blocking from the four-component structure

**Status: draft for group discussion and supervisor approval. Nothing here is final until all four members and the supervisor agree.**

---

## 1. The problem, stated as a timeline

The current structure is a strict chain: M1 → M2 → M3 → M4. Every component's start date is another component's delivery date.

**Current plan — critical path runs through all four components:**

```
Month     1    2    3    4    5    6    7    8
M1      ████████████
M2           ████████████
M3                    ████████████
M4                             ████████████
                                        ^ zero slack
```

If M1 slips three weeks, M4 loses three weeks. M4 currently carries the human perception study, participant recruitment, classifier training, and the dashboard — the least compressible work in the project — in the last and most fragile slot.

**Proposed — critical path is Phase 0 plus the longest single component:**

```
Month     1    2    3    4    5    6    7    8
Phase 0 ██████
C1            ████████████████████
C2            ████████████████████
C3            ████████████████████
C4            ████████████████████
Integration                       ████████
```

Serial dependency drops from four links to one. Each component gains roughly two months of slack, and a slip in one component no longer propagates.

---

## 2. Phase 0 — shared group deliverable (weeks 1–6)

Phase 0 is **not** anyone's component and is **not** individually assessed. It is the shared substrate all four components run on.

**Key design choice: divide by dataset, not by feature.** Each member extracts the *complete* feature set from their assigned corpora.

Why this matters:

- No member can block another by being late with a feature column
- All four members learn the full extraction pipeline — four people who can debug it instead of one
- If one corpus fails (licence, download, metadata), the other three still produce data
- Components can start on partial Phase 0 output; they do not need all corpora present

### Week-by-week

| Weeks | Activity |
|---|---|
| 1–2 | Freeze the feature schema contract in writing, signed by all four. Agree the **shard naming convention**. Environment setup. Each member runs the full pipeline on one test clip. |
| 3–6 | Each member extracts the full feature set from two assigned corpora at shard granularity. |
| 6 | Merge into the shared feature CSV. Version it. |

### The shard decision — do not skip this

Natural splits give roughly 14 rows across the five named corpora (VoxCeleb2 dev/test, LRS3 ×3, CMU-MOSI ×3, MELD ×3, FF++ ×3). A degree-2 polynomial over 12 features needs 91 coefficients. **14 rows cannot support 91 coefficients.**

EuqAud got 420 base rows because Common Voice and Multilingual LibriSpeech ship dozens of language splits. Video corpora have no equivalent structure — this must be created deliberately.

**Contract terms to agree in week 2:**

- Naming: `voxceleb2_dev_shard_01` … `_shard_40`, not `voxceleb2_dev`
- Target: ≥200 base shard rows across all corpora
- Size-proportional (CMU-MOSI's ~93 videos cannot yield 40 shards)
- Minimum ≥30 clips per gender per shard, or the shard is dropped
- Shards partitioned by **identity**, never splitting one speaker across shards

Cost of agreeing this in week 2: a naming convention. Cost of discovering it in month 5: full re-extraction by four people.

### Dataset assignment

| Member | Corpora | Notes |
|---|---|---|
| Hasangani WRA | VoxCeleb2, FaceForensics++ | FF++ needs GBDF gender annotations — base release has none |
| Harischandra RASR | LRS3, MELD | LRS3 gender labels need verification; MELD derivable from character names |
| Fernando RSR | CMU-MOSI, + 1 | |
| Sathsarani WTTS | + 2 | |

**Open: the TAF says "8+ datasets" but names five.** Candidates to evaluate for gender metadata and licence — **verify each before committing**: CREMA-D (documented actor gender), RAVDESS (12M/12F by design), Celeb-DF with GBDF annotations. Either secure three more or change the number in the TAF to five.

---

## 3. The four components

### C1 — Hasangani WRA (IT23224384)

**Sub-objective:** Feature Validity and Bias Signal Characterisation for Video Datasets

Establish which visual and temporal features carry reliable gender-representation signal, and characterise their robustness against detector error, content style, and sampling variation.

**Tasks**

1. Review computer-vision fairness auditing literature (Gender Shades, GBDF, GD-IQ) to define candidate feature families and justify inclusions and exclusions.
2. Contribute assigned corpora to Phase 0 shared extraction at agreed shard granularity.
3. Quantify face-detector performance disparity — measure MTCNN/RetinaFace detection rate and confidence by apparent gender, pose, occlusion, and lighting on a manually verified sample.
4. Establish measurement validity — test whether each feature measures what it claims, e.g. whether bounding-box area statistics track manually annotated visual prominence on a labelled subset.
5. Compare mean-based against standard-deviation-based formulations of prominence and framing features, and determine empirically which retains bias signal.
6. Compute the correlation and redundancy structure of the full feature set (Pearson, Spearman, VIF) and identify collinear clusters.
7. Test feature stability across content styles — monologue (LRS3), scripted multi-party (MELD), interview (VoxCeleb2), manipulated (FF++).
8. Run shard-size sensitivity analysis to determine minimum clips-per-gender for stable feature estimates.
9. Perform feature-family ablation and rank families by retained bias signal.
10. Produce a feature specification report with the recommended set, exclusion justifications, and documented failure modes.
11. Publish the validated specification to the group as a versioned document.

**Novelty:** First systematic measurement-validity and robustness characterisation of visual and temporal gender-representation features for video datasets. GD-IQ and GBDF report feature *values*; neither establishes whether those features are valid, stable, or robust to detector bias — a prerequisite for any regression-derived metric built on them.

**Standalone success criterion:** A ranked, empirically justified feature specification reporting detector-disparity rates, cross-style stability coefficients, and minimum-sample thresholds. Evaluable with no metric, no labels, and no other component in existence.

**Dependencies:** Consumes Phase 0 only. Produces a recommendation C3 *may* adopt — advisory, never blocking, because C3 carries a default feature set.

---

### C2 — Harischandra RASR (IT23194908)

**Sub-objective:** Ground Truth Construction and Synthetic-to-Natural Gap Quantification

Design and validate a protocol for generating known-bias video datasets without exhaustive human annotation, and quantify how closely synthetic bias resembles naturally occurring bias.

**Tasks**

1. Review temporal gender-representation literature (GD-IQ speaking-time findings, VoxCeleb demographics) to characterise how gender bias actually manifests in real video corpora.
2. Contribute assigned corpora to Phase 0 shared extraction.
3. Measure and document the baseline gender disparity already present in each corpus — establishing that "unmodified" does not mean "unbiased."
4. Implement Paradigm A — feature-level multiplicative reduction, replicating EuqAud, as the baseline.
5. Implement Paradigm B — identity-level removal: progressively remove speakers of one gender and re-extract features from source.
6. Implement Paradigm C — structured prominence reduction: preferentially remove long shots, close-ups, and speaking segments for one gender while preserving clip counts.
7. Construct gender-balanced reference subsets per corpus to serve as a calibrated zero-bias anchor.
8. Quantify the synthetic-to-natural gap — measure distributional distance (Wasserstein, Mahalanobis) between synthetically biased feature vectors and naturally biased subsets at matched disparity levels.
9. Compare paradigms on realism, monotonicity, and feature-space coverage; determine which produces vectors closest to natural bias.
10. Define degeneracy limits — the reduction depth beyond which feature estimates become unstable — and document a defensible cap.
11. Deliver a versioned ground-truth generation protocol with labelled datasets at known disparity levels.

**Novelty:** First principled ground-truth construction protocol for video dataset gender bias, comparing multiple augmentation paradigms against naturally occurring bias. EuqAud's Section V names synthetic-label realism as its central unresolved limitation; this component confronts it rather than deferring it.

**Standalone success criterion:** A quantified three-paradigm comparison with measured distributional distance to natural bias, plus documented degeneracy limits. Requires no equation and no metric to evaluate.

**Dependencies:** Consumes Phase 0. Produces labelled datasets C3 adopts for robustness testing. Non-blocking: C3 implements a minimal Paradigm A on synthetic tables for its own pipeline development from day 1.

*Note on the deliberate overlap:* both C2 and C3 touch Paradigm A. This is intentional and is the price of non-blocking — C3 needs labels on day 1 and builds the simplest possible version on dummy data; C2 studies the paradigm space properly and hands back the production version. Declare this openly rather than letting it look like duplication.

---

### C3 — Fernando RSR (IT23449282)

**Sub-objective:** EuqVid Metric Development — Scale-Invariant Closed-Form Polynomial Ridge Equation

Derive a frozen, scale-invariant, closed-form regression equation mapping video feature vectors to a calibrated gender-bias score on [−10, +10], with derived severity tiers, characterised stability, and a declared applicability domain.

**Tasks**

1. Study EuqAud Sections III and IV in depth; document every methodological choice and every point where the published text and the published equation disagree, as a specification to resolve with the supervisor.
2. Contribute assigned corpora to Phase 0 shared extraction.
3. Build the augmentation and regression pipeline against a synthetic feature table conforming to the frozen schema — full end-to-end execution before any real data arrives.
4. Design and implement scale-invariant feature transforms so the score is unaffected by dataset size; verify with a 1× / 5× / 10× invariance test.
5. Run regression assumption diagnostics (VIF, Breusch-Pagan, Durbin-Watson), documenting which genuinely motivate regularisation and which are reported descriptively.
6. Compute Pearson and Spearman correlations between each feature and the bias label under both coupled and decoupled augmentation, producing a defensible predictor ranking.
7. Grid-search Ridge alpha with polynomial degree capped at 2, using nested cross-validation with feature screening performed inside folds.
8. Derive and freeze the closed-form equation in raw feature units, folding all scaling constants into the published coefficients.
9. Define the min–max normalisation to [−10, +10] as an explicit closed-form transform with stated bounds, and derive Neutral/Moderate/Strong thresholds from stated disparity levels rather than inheriting them.
10. Establish generalisation evidence internal to the component — leave-one-corpus-out cross-validation, monotonicity along held-out perturbation curves, and a Mahalanobis applicability-domain diagnostic flagging out-of-range inputs.
11. Run 1,000-iteration bootstrap stability analysis at shard and identity level, reporting score-level and tier-level stability, with coefficient stability as a secondary honestly-reported analysis.
12. Deliver the frozen equation, normalisation constants, tier thresholds, and applicability-domain check as a versioned specification.

**Novelty:** First closed-form, scale-invariant polynomial Ridge equation for video dataset gender bias, with a declared applicability domain and derived rather than inherited severity thresholds. Extends EuqAud's methodology from audio into video while resolving two limitations of the parent method — scale dependence of raw-magnitude features, and under-specified normalisation bounds.

**Standalone success criterion:** A frozen equation passing four internal tests requiring no other component — scale-invariance (score unchanged at 1×/5×/10×), monotonicity, leave-one-corpus-out R², and bootstrap tier stability.

**Dependencies:** Consumes Phase 0 features and its own minimal Paradigm A labels. C1's feature recommendation and C2's alternative paradigms enter as robustness studies, never as prerequisites.

---

### C4 — Sathsarani WTTS (IT23204102)

**Sub-objective:** Downstream Harm Correlation, Mitigation, and Audit Tooling

Establish whether dataset-level gender bias measurements predict downstream model harm, whether bias-guided rebalancing reduces that harm, and deliver a usable auditing tool.

**Tasks**

1. Study EuqAud Section IV's validation framework and design a video-appropriate equivalent.
2. Contribute assigned corpora to Phase 0 shared extraction.
3. Build the Streamlit audit dashboard against the frozen schema with a **pluggable scoring backend** — baseline measures first, EuqVid inserted when delivered.
4. Implement baseline bias measures (screen-time ratio, face-count ratio, speaking-duration ratio) as the comparison set.
5. Train video classifiers on subsets spanning a range of measured disparity.
6. Measure downstream F1 and accuracy disparity per subset and correlate against bias measures — baselines first, EuqVid when available.
7. Run the mitigation experiment — rebalance the highest-disparity subsets by the bias measure and re-measure downstream harm, testing whether the measure is *actionable* and not merely descriptive.
8. Design and deploy the human perception study — clips spanning the disparity range, 5-point scale, 15–20 participants.
9. Compute agreement between human perception ratings and both baseline measures and EuqVid tiers.
10. Run Kruskal-Wallis effect-size comparison between EuqVid and baselines across bias tiers.
11. Apply SHAP to the fitted metric to produce per-feature attributions in the dashboard.
12. Deliver the validated audit tool with a reproducibility report.

**Novelty:** First validation framework for a video dataset fairness metric that moves beyond correlation to **intervention** — testing whether bias-guided rebalancing measurably reduces downstream performance disparity. EuqAud correlated its score with WER but never tested whether acting on the score improves fairness outcomes.

**Standalone success criterion:** A measured correlation between dataset-level bias and downstream F1 disparity, plus a mitigation result — both obtainable using baseline measures alone, independent of whether EuqVid is finished.

**Dependencies:** Consumes Phase 0. EuqVid arrives as one additional column in an already-working comparison. Nothing blocks.

---

## 4. Dependency map

```
                    ┌─────────────┐
                    │   PHASE 0   │  weeks 1-6, all four members
                    │ shared CSV  │  divided by dataset
                    └──────┬──────┘
          ┌────────────┬───┴────┬────────────┐
          ▼            ▼        ▼            ▼
        ┌────┐      ┌────┐   ┌────┐       ┌────┐
        │ C1 │      │ C2 │   │ C3 │       │ C4 │
        └──┬─┘      └──┬─┘   └─┬──┘       └─┬──┘
           │           │       ▲            ▲
           └───────────┴───────┘            │
              advisory, non-blocking        │
                             └──────────────┘
                              pluggable, non-blocking
```

Solid line = hard dependency (one only). Dashed = advisory input that arrives as a refinement.

**The test for whether a dependency is truly non-blocking:** if the upstream component delivered *nothing at all*, could the downstream component still produce a complete, assessable result? For every arrow above, yes.

---

## 5. What actually changes from the current TAF

| | Current | Proposed |
|---|---|---|
| Extraction | M1 then M2, sequential, by feature type | Phase 0, parallel, by dataset |
| Row granularity | `dataset_split` (~14 rows) | Shards (~200+ rows) |
| Hasangani | Visual feature extraction | Feature validity and robustness |
| Harischandra | Temporal feature extraction | Ground truth construction |
| Fernando | Regression on received CSV | Metric with own generalisation evidence |
| Sathsarani | Validation of received equation | Downstream harm, mitigation, tooling |
| Hard dependencies | 3 (chained) | 1 (Phase 0) |
| Held-out evaluation in C3 | none | LOCO CV, monotonicity, scale-invariance |

The visual and temporal extraction work has not disappeared — it moved into Phase 0 and is shared four ways. What changed is that each member's *research question* is now their own.

---

## 6. Expect these pushbacks

**"Phase 0 means four people doing the same work."**
Six weeks of shared infrastructure to remove three months of serial dependency. It is also insurance — four people who can debug the pipeline instead of one. Individual contribution begins month 2 and runs five months instead of two.

**"C1 and C2 look like analysis, not engineering."**
C1 builds detector-disparity measurement and manual-annotation validation harnesses. C2 builds three distinct video-level augmentation pipelines with source-level re-extraction. Both are substantial implementation.

**"This is a bigger change than the problem warrants."**
Fair, and worth answering honestly: if the concern were purely cosmetic scheduling, a contract-first fix would do — freeze the schema in week 2, everyone extracts a small independent pilot, everyone codes against the contract. That keeps all four TAF components unchanged. Take that fallback to the meeting as Option C so the supervisor has a choice rather than an ultimatum.

**Real risk to name yourselves before someone else does:** if Phase 0 slips, everything slips. Mitigation — Phase 0 is parallel across four members, and the minimum viable output is two corpora. Components start on partial data and backfill.

---

## 7. Before anything is rewritten

1. Confirm whether the TAF is already submitted and whether revision is permitted.
2. Get all three teammates to agree — this rewrites their components, not just C3's.
3. Take Option A and the Option C fallback to the supervisor together.
4. Only then rewrite the TAF sections.

**Fix regardless of which option is chosen** — the current TAF states twice that EuqAud "explicitly identified video as future work." It does not. Section V's limitations cover demographic attributes, multi-speaker audio, far-field and noisy conditions, scarcity of female-biased corpora, and synthetic-label realism. Section VI says "other demographic attributes and **multimodal data**." Dr. Haddela supervised that paper.
