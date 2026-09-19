# Step 1 — Dataset access verification

Group J26-DS-343 · EuqVid
Domain: narrative multi-person video (film & scripted TV)

**Status: the corpus set is settled.** This document is now the record of what was
verified and why, not a live to-do. Section 4 holds the methodology questions for
Dr. Haddela, which are still open and still group-level.

**Why this mattered:** every downstream decision — feature schema, shard granularity,
augmentation design, the four equations themselves — depends on which corpora we
actually hold. The dataset list changed completely when the domain changed, and the
TAF's claim that "all datasets are publicly available and freely accessible" was
written about a different set. Film corpora are built on copyrighted material; some
distribute features or keyframes rather than frames you can run a detector over.

---

## 1. The settled set

| Corpus | Raw video? | Gender labels | Access | Role |
|---|---|---|---|---|
| **AVA / AVA-ActiveSpeaker** | ✅ via cvdfoundation repo | ❌ manual annotation per `entity_id` | ✅ open | **Primary fusion corpus.** Feature film, 160 videos, 38.5 h face tracks, 3.65M labelled frames. The only corpus all four components run on |
| **MELD** | ✅ video clips shipped | Character names → derivable | ✅ open, no registration | **Negative control.** 1,433 dialogues / ~13k utterances. Ships emotion + sentiment labels |
| **AMI Meeting Corpus** | ✅ room-view cameras — **Edinburgh form only, NOT OpenSLR** | ❓ participant metadata | ✅ open, CC BY 4.0 | **Structural contrast corpus.** 100 h, 3–5 person meetings, plus DOME human dominance labels |
| **LSMDC / MPII-MD Co-ref+Gender** | ✅ AVI + JPEG frames | ✅ explicit M/F per character | ⚠️ application to MPII, pending | **Bonus.** Deliberately not load-bearing — nothing depends on approval arriving |

### Ruled out, with reasons

| Corpus | Why |
|---|---|
| MovieNet | Project page states data is distributed *excluding movies*. You get annotations, keyframes, trailers and pretrained features — no film. Face detection cannot run on a feature vector |
| TVQA | 3 FPS frames only. Optical flow between frames 333 ms apart is meaningless; lip activity needs ~25 FPS. Also behind a form requiring supervisor details |
| MAD | No usable raw video for our purposes |
| Condensed Movies | No gender labels, no advantage over AVA for the same content type |
| Cornell Movie-Dialogs | Dialogue only, no video — cannot support multimodal fusion |
| Bazinga! | Superseded by MELD for scripted TV |
| CCv2 | Single-subject problem. The domain is defined by two or more subjects sharing the frame; using it would require a `co_presence` flag |
| AVA-Kinetics | One annotated frame per clip — no temporal continuity, so no screen time and no motion energy |

### What this set does and does not give us

**It clears the minimum bar.** Three corpora, openly downloadable, two content styles
(feature film + scripted TV) with meetings as a third contrast. The cross-style
generalisation claim is supportable, and nothing blocks extraction.

**The known weakness is gender labels.** AVA — the corpus that carries the most weight —
ships none. Manual annotation per `entity_id` with inter-annotator agreement is declared
work in all four reports. LSMDC is the only corpus with the annotation already done,
and it is the one we may not get.

---

## 2. What to check in each column

Kept because the same checks apply to any corpus proposed later.

**Raw video** — can you download actual frames, or only extracted features / keyframes /
annotations? If features only, the corpus is dead for us. This is the column most likely
to kill a corpus.

**Gender labels** — native annotation (best), derivable from character names or cast lists
(fine, document the mapping), or requires manual annotation (acceptable but adds work). If
you use a classifier instead, you inherit its disparate error rates and acquire the "our
bias metric depends on a biased classifier" viva question. We annotate by hand for AVA
specifically to avoid that.

**Licence** — academic use permitted? Registration required, and what's the stated
turnaround? Does it need an institutional email or supervisor signature? Start any
registration the same day you find it.

**Multi-person** — confirm by eye on 5–10 sample clips that two or more people actually
share frames. Do not take the paper's word for it.

**Feature coverage** — can *every* feature in every component's set be computed on this
corpus? A frozen closed-form equation must evaluate every feature on every corpus it
scores. This is the check that ruled out TVQA and that forced C3's F6 redefinition.

---

## 3. Verification links

**Open — download now, no application**

| Corpus | Link |
|---|---|
| AVA — download page | https://research.google.com/ava/download.html |
| AVA — video download repo | https://github.com/cvdfoundation/ava-dataset |
| AVA-ActiveSpeaker — paper | https://arxiv.org/abs/1901.01342 |
| MELD — project page | https://affective-meld.github.io |
| MELD — code and data | https://github.com/declare-lab/MELD |
| MELD — paper (ACL 2019) | https://aclanthology.org/P19-1050/ |
| AMI — **video download form** | https://groups.inf.ed.ac.uk/ami/download |
| AMI — manual annotations v1.6.2 | https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip |
| AMI — DOME dominance annotations | https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/dome_annotations_M1.csv |

⚠️ **Do not use the OpenSLR AMI mirror (openslr.org/16) — it is audio only.** Category
"Speech", every file close-talking or array acoustic data, ~84 GB of it, no video. Video
comes only from the Edinburgh form above: select meetings → tick *Low-size DivX AVI
videos* → press the button. OpenSLR also states CC BY-**NC**-SA v2.0 while Edinburgh
states CC BY 4.0 — cite Edinburgh's terms.

On AMI video: each meeting ships four closeup cameras plus corner and overhead room
views. **Use the corner/room views** — closeups are single-face fixed shots, i.e. the
framing the supervisor ruled out. Full-resolution video is not downloadable (firewire
drive order only), so test face detection on one meeting at DivX resolution before
committing to a batch.

**Application required**

| Corpus | Link |
|---|---|
| LSMDC — download page | https://sites.google.com/site/describingmovies/download |
| MPII-MD — **access application** | https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset/access-to-mpii-movie-description-dataset/ |
| MPII-MD Co-ref+Gender — the gender annotations | https://arxiv.org/pdf/2008.09791 |

---

## 4. Open methodology questions for Dr. Haddela

**Still open, and group-level.** All four components fit a polynomial Ridge equation
against EuqAud's methodology, so these answers affect everyone, not one member. They
cannot be resolved from the published PDF.

1. **Sign convention.** EuqAud III-E says training targets range 0 to 10, but the
   published equation has sign-flipped coefficient pairs (E_male +0.0044 /
   E_female −0.0034) that only produce signed output. Which was actually used?
2. **Normalisation bounds.** How were the min–max bounds computed? Algorithm 1 gives
   only `min_max_scale(raw, −10, 10)`, while the prose describes per-dataset
   self-normalisation. The two are not the same thing, and the difference matters for
   whether +6 means the same disparity across corpora.
3. **Polynomial degree.** Was degree ever in the grid search, or selected manually? The
   published grid covers alpha and l1_ratio only, and the final equation carries
   degree-2 terms in pitch alone.
4. **Row expansion.** How were the 420 base rows expanded into the training set?
5. **Request:** an introduction to Sriwanthi Jayawardena, and whether the EuqAud
   augmentation code is still available. *This one is worth more than the other four
   combined — it would resolve most of the open methodology questions in one go.*

---

## 5. Fix regardless of anything else

The TAF states twice that EuqAud "explicitly identified video as future work." It does
not. Section V's limitations cover demographic attributes, multi-speaker audio,
far-field and noisy conditions, scarcity of female-biased corpora, and synthetic-label
realism. Section VI says "other demographic attributes and **multimodal data**."
Dr. Haddela supervised that paper.

The TAF also says eight-plus datasets while naming four, and asserts all are freely
accessible. Whatever the real number is, make the TAF say that — and note that the TAF
title, "using visual and temporal features," no longer describes a four-modality
project either. Whether the TAF is revisable is still unanswered.
