# Step 1 — Dataset access verification

Group J26-DS-343 · EuqVid
Domain: narrative multi-person video (film & scripted TV)

**Why this is first:** every downstream decision — feature schema, shard granularity, augmentation design, the equation itself — depends on which corpora you actually hold. The dataset list changed completely when the domain changed, and the TAF's claim that "all datasets are publicly available and freely accessible" was written about a different set. Film corpora are built on copyrighted material; some distribute features or keyframes rather than frames you can run a detector over.

Registration approvals have **external latency**. Start them now and do other work while they run.

---

## The verification table

Fill one row per corpus. A corpus is **usable** only if the first three columns are all yes.

Pre-filled from web verification on 2026-08-16. ✅ = confirmed from the official page · ⚠️ = problem found · ❓ = still needs checking by hand.

| Corpus | Raw video? | Gender labels | Access | Content | Size | M4 task labels |
|---|---|---|---|---|---|---|
| **MELD** | ✅ video clips shipped | Character names → derivable | ✅ **open, no registration** | Friends, multi-party scenes | 1,433 dialogues / ~13k utterances | ✅ **emotion + sentiment** |
| **AVA / AVA-ActiveSpeaker** | ✅ via cvdfoundation repo | ❌ needs annotation | ✅ **open** | movie clips | 38.5 h face tracks, 3.65M labelled frames | ✅ **active speaker + 80 actions** |
| **AMI Meeting Corpus** | ✅ room-view cameras — **Edinburgh form only, NOT OpenSLR** | ❓ participant metadata | ✅ open, CC BY 4.0 | 3–5 person meetings | 100 h | diarisation, ASR, + **DOME human dominance labels** |
| **LSMDC / MPII-MD Co-ref+Gender** | ✅ AVI + JPEG frames, via download script | ✅ **explicit M/F per character** | ⚠️ **application required** — username/password issued by MPII | film clips | ~118k clip-sentence pairs, ~158 h | video description |
| **TVQA** | ❓ verify — frames or full video? | Character names → derivable | ❓ | 6 TV shows | 21.8k clips / ~460 h | QA |
| **MovieNet** | ⚠️ **"all the data (excluding movies)"** — no raw film | Character identities → cast lists | registration via OpenDataLab | 1,100 movies | 1.1M character boxes | scene, action, place |
| **Charades** | ❓ | ❌ needs annotation | ❓ | home scenes | ~9,800 videos | action classification |
| **Condensed Movies** | ❓ | ❌ needs annotation | ❓ | film clips | ~33k clips | — |

### What this already tells you

**You can start M1 this week.** MELD + AVA + AMI are all openly downloadable with no application, and they clear the minimum bar: three corpora, two content styles (scripted TV + film, with meetings as a third contrast). Nothing is blocking extraction.

**MovieNet is probably out.** The project page states data is distributed *excluding movies* — you get annotations, keyframes, trailers and pretrained features, but not the films. M1 cannot run face detection on a feature vector. Keyframes might support per-shot face count and bounding-box statistics, but not screen time or motion energy. Check whether the keyframe density is usable before writing it off entirely, but do not plan around it.

**LSMDC is the one to apply for today.** It is the only corpus with explicit per-character M/F gender annotation, raw video is genuinely available (AVI plus JPEG frames via download script), but access needs an application to MPII for credentials. That approval is your longest lead time — start it before anything else.

### What to check in each column

**Raw video** — can you download actual frames, or only extracted features / keyframes / annotations? If features only, the corpus is dead for you: M1 cannot run face detection on a feature vector. This is the column most likely to kill a corpus.

**Gender labels** — native annotation (best), derivable from character names or cast lists (fine, document the mapping), or requires manual/classifier annotation (acceptable but adds work and, if you use a classifier, adds the "our bias metric depends on a biased classifier" viva question).

**Licence** — academic use permitted? Registration required, and what's the stated turnaround? Does it need an institutional email or supervisor signature? **Start any registration the same day you find it.**

**Multi-person** — confirm by eye on 5–10 sample clips that two or more people actually share frames. Do not take the paper's word for it.

**Downstream labels** — M4 must train a classifier and measure F1 disparity. A corpus that ships task labels (MELD emotion, AVA active-speaker) saves them weeks.

### Verification links

**Open — download now, no application**

| Corpus | Link |
|---|---|
| MELD — project page | https://affective-meld.github.io |
| MELD — code and data | https://github.com/declare-lab/MELD |
| MELD — paper (ACL 2019) | https://aclanthology.org/P19-1050/ |
| AVA — download page | https://research.google.com/ava/download.html |
| AVA — video download repo | https://github.com/cvdfoundation/ava-dataset |
| AVA-ActiveSpeaker — paper | https://arxiv.org/abs/1901.01342 |
| AMI — **video download form** | https://groups.inf.ed.ac.uk/ami/download |
| AMI — manual annotations v1.6.2 | https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip |
| AMI — DOME dominance annotations | https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/dome_annotations_M1.csv |

⚠️ **Do not use the OpenSLR AMI mirror (openslr.org/16) — it is audio only.** Category "Speech", every file is close-talking or array acoustic data, ~84GB of it, no video. Video comes only from the Edinburgh form above: select meetings → tick *Low-size DivX AVI videos* → press the button. Note also that OpenSLR states CC BY-**NC**-SA v2.0 while Edinburgh states CC BY 4.0 — cite Edinburgh's terms.

On AMI video: each meeting ships four closeup cameras plus corner and overhead room views. **Use the corner/room views** — closeups are single-face fixed shots, i.e. the framing problem the supervisor ruled out. Full-resolution video is not downloadable (firewire drive order only), so test face detection on one meeting at DivX resolution before committing to a batch.

**Application required — start today**

| Corpus | Link |
|---|---|
| LSMDC — download page | https://sites.google.com/site/describingmovies/download |
| MPII-MD — **access application** | https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset/access-to-mpii-movie-description-dataset/ |
| MPII-MD Co-ref+Gender — the gender annotations | https://arxiv.org/pdf/2008.09791 |

**Still to verify**

| Corpus | Link |
|---|---|
| TVQA — project page | http://tvqa.cs.unc.edu |
| TVQA — UNC data page | https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/index.html |
| TVQA — code and format | https://github.com/jayleicn/TVQA |
| MovieNet — project page | https://movienet.github.io |
| MovieNet — paper | https://arxiv.org/abs/2007.10937 |

For Charades and Condensed Movies, find the official Allen AI and Oxford VGG project pages rather than mirrors — both are optional and only worth chasing if the primary set falls short.

---

## Decision rule

**Minimum viable set to start M1:**

- **3 corpora** with raw video + gender labels confirmed
- covering **2 different content styles** (e.g. scripted TV *and* feature film) — you need this for the cross-style generalisation claim, and one style alone gives you a result nobody will believe transfers
- combined size sufficient for **≥200 shards** at ≥30 clips per gender per shard

If you clear that bar, M1 starts. Add corpora later — the pipeline doesn't care.

**If film corpora turn out to be restricted:** the domain still holds. MELD + TVQA + AVA carry it, with AMI as the contrast corpus. Do not silently drop to a single corpus and hope nobody notices.

**Update the TAF once the answer is known.** The current text says eight-plus datasets, names five, and asserts all are freely accessible. Whatever the real number is, make the TAF say that.

---

## Parallel track — what Fernando does while registrations run

None of this is blocked by the dataset answer.

**1. Send two emails today.** See below.

**2. Get into the split-granularity conversation with M1 and M2.** TAF tasks M1-2 and M2-2 are "agree on dataset splits with each other" — your name isn't on either, but the row count of your entire training set is decided there. Natural splits give ~14 rows against 91 coefficients. Ask for shard-granular naming (`meld_train_shard_01`…), ≥200 shards, size-proportional, ≥30 clips per gender, partitioned by identity so no character spans two shards. **This must land before M1 extracts anything.**

**3. Build the M3 pipeline against a synthetic feature table.** Already TAF task 2. Parameterise it over the schema so column-name changes cost nothing — the *shape* (six gendered feature pairs) is stable even if names aren't. Full end-to-end run: augmentation → assumption diagnostics → grid search → equation → normalisation → bootstrap. When real data arrives you swap the input file, nothing else.

**4. Fix the TAF error regardless of anything else.** It states twice that EuqAud "explicitly identified video as future work." It does not. Section V covers demographic attributes, multi-speaker audio, far-field and noisy conditions, scarcity of female-biased corpora, and synthetic-label realism. Section VI says "other demographic attributes and **multimodal data**." Dr. Haddela supervised that paper.

---

## The two emails

**To Dr. Haddela** — four questions and one request:

1. EuqAud III-E says training targets range 0 to 10, but the published equation has sign-flipped coefficient pairs (E_male +0.0044 / E_female −0.0034) that only produce signed output. Which was actually used? *Cannot be resolved from the PDF.*
2. How were the min–max bounds computed? Algorithm 1 gives only `min_max_scale(raw, −10, 10)`.
3. Was polynomial degree ever in the grid search, or selected manually? The published grid covers alpha and l1_ratio only, and the final equation carries degree-2 terms in pitch alone.
4. How were the 420 base rows expanded into the training set?
5. **Request:** an introduction to Sriwanthi Jayawardena, and whether the EuqAud augmentation code is still available.

That last one is worth more than the other four combined — it would resolve most of the open methodology questions in one go.

**To the group** — the domain change, the corpus shortlist, and the split-granularity ask. Send this before the next group meeting so people arrive having read it, not hearing it for the first time.
