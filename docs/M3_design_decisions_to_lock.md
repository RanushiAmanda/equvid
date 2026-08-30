# M3 — Design decisions to lock before coding

Group J26-DS-343 · EuqVid Metric Development · Fernando RSR (IT23449282)
Sources: EuqAud (Jayawardena et al., IEEE Access 2026) Sections III-C/III-D/III-E/IV-D/V · New-TAF-J26-DS-343.pdf

Status key: `[ ]` open · `[x]` confirmed/locked · `[?]` needs supervisor confirmation · `[!]` **error to fix**

---

## 0. URGENT — act before M1/M2 finalise the schema

- [ ] **U1. Get into the dataset_split decision.**
  M1 task 2 and M2 task 2 are both *"agree on dataset splits with each other."* M3 is not named in that conversation, but the row count of M3's entire training set is decided there.

  Natural splits available: VoxCeleb2 (dev/test = 2), LRS3 (3), CMU-MOSI (3), MELD (3), FF++ (~3) ≈ **14 base rows**; ~24 at 8 datasets.
  EuqAud got 420 base rows free from Common Voice / MLS language splits. Video corpora have no equivalent.
  **24 independent units against 91 coefficients = 0.26 units per parameter. Non-viable.**

  Ask: shard-granular split names (`voxceleb2_dev_shard_01…40`), target ≥200 base rows, size-proportional (CMU-MOSI's ~93 videos cannot yield 40), minimum clips-per-gender-per-shard ≥30.
  **Cost now: a naming convention. Cost in three months: full re-extraction by two teammates.**

- [!] **U2. Fix the EuqAud future-work claim — it appears twice and it is false.**
  TAF Section 6: *"EuqAud... explicitly identifying video as unaddressed future work."*
  TAF M3 novelty column: *"a gap EuqAud Section V explicitly identified as future work."*
  **Verified against the paper:** Section V limitations cover demographic attributes, multi-speaker audio, far-field/noisy/child-elderly speech, scarcity of female-biased corpora, and synthetic-label limits. Video is never mentioned. Section VI says "other demographic attributes and **multimodal data**."
  Dr. Haddela supervised EuqAud. Reword to "multimodal data (Section VI)" before this is read.

- [!] **U3. "8+ datasets" vs 5 named.** Objective, M1 task 4 and M4 task 7 say 8+; Section 8 names five. Name the other three or change the number.

- [!] **U4. M4 task 7 says "8+ *independent* datasets"** but M1 extracts from the same corpora M3 trains on. Not independent without an explicit firewall. See F6.

- [!] **U5. Gender-metadata claim contradicts your own related work.** Section 8 says all five corpora "include gender metadata." Section 6 describes GBDF as *annotating gender on FaceForensics++* — which only makes sense if FF++ lacked it. Verify per corpus; expect LRS3 and FF++ to fail. Fix is already cited: use GBDF annotations.

- [!] **U6. Ethics wording overclaims.** "No collection of personal, sensitive, or identifiable data" while processing identifiable faces and inferring gender. Faces are biometric data. Position is defensible (public corpora, existing licences) — reword rather than assert.

- [ ] **U7. Add FF++'s real justification to the TAF.** Trinh & Liu found FF++ predominantly *female*. EuqAud Section V says female-biased corpora are almost impossible to find, which is why all their real-world scores clustered positive. FF++ gives you the negative half of your scale on real data. Strong reason, currently unstated.

---

## A. Ground truth & augmentation (Task 1)

- [ ] **A1. Manipulation level.** Source/clip-level resampling + re-run M1/M2 extraction, NOT arithmetic on the feature CSV.
  *EuqAud edits the feature table directly (III-C). That is what makes feature→label an algebraic inverse, and why their R² is 0.9998 at alpha 0.01. Re-extraction makes it a noisy inverse problem — the whole circularity defence rests on this.*
  Boundary flag: needs M1/M2 compute.

- [x] **A2. Unit of analysis = `dataset_split`.** Locked by TAF (M1 task 7, M2 task 8: "aggregate per dataset split"). The open question is *granularity* — see U1.

- [x] **A3. Sample counts held constant.** Confirmed in writing, TAF Section 8.
  *Consequence: `face_count_male/female` carry no label-linked variance; their coefficients will be spurious (EuqAud's are ±0.0001). Deviating now requires documented justification.*
  Optional: count-varying ablation.

- [ ] **A4. Joint vs decoupled feature reduction.** Run both as named modes.
  Joint → all features perfectly collinear → **Task 5's Pearson ranking is meaningless** (everything correlates ~±0.99 with the label) and your stated novelty evaporates.
  Decoupled (independent per-feature-group magnitudes) → breaks collinearity, fills the interior of feature space.

- [ ] **A5. Label definition.** Resulting **disparity state** (normalised difference / log-ratio), NOT applied reduction %.
  *EuqAud uses reduction÷10 (5% → 0.5), which labels an already-male-skewed corpus as 0 = Neutral. Their Table 2 shows real corpora at 0.86 / 1.58 / 2.21 while Section V says most corpora are male-dominant — visible under-detection.*

- [ ] **A6. Zero-bias reference.** Gender-balanced resampled subset per corpus, not the raw corpus.

- [x] **A7. Symmetry.** Both directions. EuqAud III-C splits the base set in half. Required by the signed [−10, +10] scale.

- [ ] **A8. Reduction depth.** Cap at 60–70%, not 100%.
  *At 100% with counts constant: female faces present, zero variance in their size. Physically unrealisable. Both ±10 anchors pinned by impossible data.*

- [ ] **A9. Stochastic replicates.** ≥10 seeds per (shard, level). No synthetic Gaussian noise on features.

- [?] **A10. Training target sign convention.** Signed −10…+10 vs unsigned 0–10.
  *EuqAud is internally inconsistent: III-E says targets are "0 to 10" but the published equation has sign-flipped coefficient pairs (E_male +0.0044 / E_female −0.0034), which only produces signed output. Ask Dr. Haddela what was actually run — this cannot be resolved from the PDF.*

---

## B. Feature semantics — raise with M1/M2 (boundary)

- [ ] **B1. Means, not only standard deviations.** *This challenges M1's written novelty claim — handle carefully, but it is the right call.*
  M1's TAF novelty: std used *"to isolate representational bias from natural gender appearance differences."*
  *EuqAud used std of pitch/energy/amplitude because those are intrinsically gender-linked — a mean would just detect "women are present." Bounding-box area and screen-centre proximity are not intrinsically gender-linked; the premise does not transfer.*
  "Why does lower variance in female bounding-box area mean more bias?" has no answer. Mean face size and mean centre-proximity do — those are exactly the *women are framed smaller and more peripherally* signal, and std removes it.

- [ ] **B2. Scale-free features. Now worse than EuqAud.**
  `face_count`, `screen_time`, `speaking_duration` are raw magnitudes that grow linearly with split size — **6 of 12 columns.** EuqAud had 4 of 10.
  Your corpora span ~2,200 CMU-MOSI segments to ~1.1M VoxCeleb2 utterances. A scale-dependent equation cannot survive that range. Use proportions/ratios.

- [ ] **B3. Gender label provenance per corpus.** See U5. Where labels come from a face-based classifier, the bias metric inherits that classifier's disparate error rates. Prepare the viva answer.

---

## C. Modelling (Tasks 3–4, 9)

- [ ] **C1. Assumption-check framing.** VIF genuinely motivates Ridge. Breusch–Pagan and Durbin–Watson are descriptive only.
  *DW tests autocorrelation in ordered data; on augmentation-sorted rows it is an artefact of your own row ordering. EuqAud reports DW = 1.027 as an independence violation. If retained for consistency, state the ordering it is computed over.*
  *Normality/heteroscedasticity violations invalidate OLS inference, not point estimates. "Assumptions violated → linear regression unsuitable" is logically loose reasoning you are about to inherit.*

- [ ] **C2. Feature pruning inside CV folds.** Screening on the full set then fitting is leakage — a classic viva gotcha. Target ≤8 features entering the expansion.

- [ ] **C3. Polynomial degree capped at 2. Do not put degree 3 in the grid** where a reviewer can see you considered it.

  | features | degree 2 | degree 3 |
  |---|---|---|
  | 12 (confirmed) | **91** | **455** |
  | 8 | 45 | 165 |
  | 6 | 28 | 84 |
  | 4 | 15 | 35 |

  *Effective n = independent shards, not augmented rows. All rows from one shard lie on a deterministic 1-D curve through feature space.*

- [?] **C4. Full vs selective expansion.** TAF M3 task 4 says "full grid search over polynomial degree and Ridge alpha." **EuqAud never gridded degree** — they searched alpha {0.001…10} and l1_ratio {0…1.0} only, and their final equation carries degree-2 terms in *pitch alone*. They hand-picked the expansion. Justify the deviation or drop it.

- [ ] **C5. Scaling order.** StandardScaler → PolynomialFeatures → Ridge, then algebraically fold the scalers back so the frozen equation prints in raw feature units.

- [ ] **C6. Report nominal n AND effective n.** Volunteer this before being asked. Being caught quoting the augmented row count is worth negative marks.

- [ ] **C7. Dummy-CSV caveat.** TAF tasks 2 and 4 build the pipeline on a synthetic dummy CSV — good sequencing, but grid-search results on dummy data are meaningless. Verify the code path only; never report those numbers.

---

## D. Normalisation & tiers (Task 6)

- [ ] **D1. Write the min–max transform out as an explicit closed form.**
  *EuqAud's Algorithm 1 says only `min_max_scale(raw, −10, 10)`. The prose describes per-dataset self-normalisation (bounds from minimising one gender's features vs setting them equal). That self-normalisation is the only thing rescuing their equation from the raw-scale bug — replicate it deliberately, not by accident.*
  **This is where you can be demonstrably better than the parent paper. Claim it as a contribution.**

- [?] **D2. Tier thresholds 2 / 6.** EuqAud gives no derivation. Ground them in a stated disparity level (e.g. 2 ↔ 20% representational gap).

- [ ] **D3. Scale-invariance test.** Score the same corpus at 1×, 5×, 10× size. If the score moves, B2/D1 are not done. Given B2, expect it to move badly on first attempt.

---

## E. Stability (Task 10)

- [ ] **E1. Lead with score- and tier-level stability**, matching EuqAud IV-D: resample 1,000×, recompute mean scaled score, report mean / SD / 95% CI, confirm no CI crosses zero or a tier boundary.

- [ ] **E2. Resample at shard/identity level, not row level.** Row-level bootstrap on synthetically expanded data produces fictitiously narrow CIs.

- [ ] **E3. Coefficient bootstrap = secondary, honestly reported.** EuqAud Section V names coefficient stability as an open need, so this is a genuine contribution — but with 45+ collinear polynomial terms, signs will flip across replicates. Keep it out of the abstract.

---

## F. Generalisation evidence M3 must own

**Gap in the TAF: M3 tasks 8–10 contain no held-out evaluation of any kind.** Your only reported number would be an in-sample R² ≈ 0.99 on data you generated yourself. If asked "how do you know it generalises?", your current answer is "M4 checked." Fix by adding:

- [ ] **F1. Cross-augmentation-type test.** Train on uniform reduction; test on structured reduction (preferentially remove long shots / close-ups / speaking segments for one gender). Real bias changes conditional distributions, not just levels.
- [ ] **F2. Leave-one-corpus-out CV.** Report alongside headline R². Expect much worse. Report it anyway — an honest LOCO number persuades more than a perfect in-sample one.
- [ ] **F3. Natural-contrast check.** Order naturally male-skewed vs balanced subsets within one corpus using existing metadata.
- [ ] **F4. Applicability-domain diagnostic.** Mahalanobis distance from the training manifold; flag out-of-range inputs. EuqAud has nothing like this.
- [ ] **F5. Monotonicity acceptance criterion.** Score must increase monotonically along held-out perturbation curves.
- [ ] **F6. M3/M4 firewall, agreed in writing.** Partition by identity, not clip. Resolves U4.

---

## G. Group-level flags (not yours, but they reflect on the project)

- [!] **G1. M4 task 8: "target effect size above 0.845 matching EuqAud."** Pre-committing to beating a specific result invites tuning until you hit it. Reword as "compare against."
- [ ] **G2. Ethical clearance.** "Formal ethical clearance is not required" for a 15–20 participant perception study — confirm against SLIIT policy rather than asserting it.

---

## Priority order

1. **U1** — dataset_split granularity. Time-critical; everything else is moot if you get 14 rows.
2. **U2** — the false EuqAud future-work claim, twice, in front of EuqAud's own supervisor.
3. **A5 + B2 + D1** — scale-free label, scale-free features, explicit normalisation. Highest technical leverage.
4. **A1** — source-level manipulation. The circularity defence depends on it.
5. **A4** — decoupled mode. Task 5 is worthless without it.
6. **B1** — means vs std. Changes the schema, so it must land before M1 extracts.
7. **F2 + C3 + C6** — held-out evaluation, degree cap, effective-n honesty.
