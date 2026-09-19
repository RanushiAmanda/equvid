# Data

**Nothing in this directory is committed except instructions and scripts.**

The corpora total roughly 43 GB and their licences do not permit redistribution. This file plus the download scripts is what makes the project reproducible: run them and you rebuild the same data from the original sources.

Put downloaded corpora in folders named `MELD/`, `AVA/`, `AMI/` at the repository root. `.gitignore` already blocks those names.

---

## Status

| Corpus | Size | Access | Gender labels | Status |
|---|---|---|---|---|
| AVA / ActiveSpeaker | ~22 GB | Open | ❌ Manual annotation per `entity_id` | ✅ Have it (partial batches) |
| MELD | ~11 GB | Open | Character names → derivable | ✅ Have it |
| AMI | ~9.6 GB | Open form | ❓ Check participant metadata | ✅ ES2002–ES2008 |
| LSMDC / MPII-MD | — | Application | ✅ Explicit M/F per character | ⏳ Awaiting credentials |

**Ruled out — do not spend time on these again:** Condensed Movies, TVQA, MAD, MovieNet, Cornell Movie-Dialogs, Bazinga!, CCv2. Reasons are in the table at the foot of this file.

**Which corpus carries which load:**

- **AVA is the primary fusion corpus.** It is the only feature-film source, it is where the disparity the metric measures is most expected, and it is the one corpus all four components can run on — so a fused EuqVid score is computed on the same material rather than on a union of partially-overlapping corpora.
- **MELD is the negative control**, not a bias demonstration. Friends is an unusually balanced ensemble (3 female / 3 male leads, main-cast speaking-time M:F = 1.094). A metric that correctly scores a balanced corpus near Neutral is evidence against false positives, and examiners ask for exactly that.
- **AMI is the structural contrast corpus** — non-narrative, and the only source of independent human dominance judgements.
- **LSMDC is a bonus.** Nothing is designed to depend on it, because approval is outside our control.

---

## AVA / AVA-ActiveSpeaker

Movie clips. 160 videos for ActiveSpeaker, 3.65M labelled frames, ~39K face tracks. **This is where identity diversity comes from** — every video is a different film with different actors.

**Annotations** (CC BY 4.0):

- `https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_train_v1.0.tar.bz2`
- `https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_val_v1.0.tar.bz2`
- AVA-Speech, for clean-speech regions: used by C3 to gate ASR segments so music and background noise do not enter the transcript
- Optional, for downstream action classification: `.../annotations/ava_v2.2.zip`

**Videos:** `python download_ava.py` (edit the batch slice inside). Source pattern is `https://s3.amazonaws.com/ava-dataset/trainval/[file_name]`, file list at `.../annotations/ava_speech_file_names_v1.txt`.

**Why ActiveSpeaker matters.** The CSV format is `video_id, frame_timestamp, entity_box, label, entity_id`:

| Field | Gives you |
|---|---|
| `entity_box` | Face bounding box, already normalised → C1 features without MTCNN |
| `entity_id` | Same person linked across frames → screen time, tracking solved |
| `label` | SPEAKING_AND_AUDIBLE / SPEAKING_BUT_NOT_AUDIBLE / NOT_SPEAKING per frame → C2 speaking duration without lip detection or VAD, and C3 speaker attribution without diarisation |

**Gender annotation is the outstanding work**, and it is declared work in all four reports rather than future work. Use `entity_id` — you label a person, not a frame. Sort tracks by duration and label the long ones first, since they dominate screen time. Inter-annotator agreement must be reported.

**AVA ships no dialogue.** C3 recovers a transcript with one ASR pass over clip audio, attributes speakers by aligning ASR segments to `SPEAKING_AND_AUDIBLE` spans, and gates on AVA-Speech. The resulting word error must be quantified against a hand-checked subset and reported with every AVA text result — see `docs/C3_AVA_decision_and_F6_change.md`.

> ⚠️ **Skip AVA-Kinetics.** One annotated frame per clip — no temporal continuity, so no screen time and no motion energy.

Verify clip count and current licence terms on the distributor's own page before submission rather than carrying forward an unchecked figure.

---

## MELD

Scripted TV (*Friends*). 1,433 dialogues, ~13k utterances. Ships emotion and sentiment labels, useful both as a downstream task for validation and as C3's `sentiment_valence_mean` feature.

**Video:** [MELD.Raw.tar.gz](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz) — `.mp4` clips in train/dev/test tarballs. Also mirrored at [huggingface.co/datasets/declare-lab/MELD](https://huggingface.co/datasets/declare-lab/MELD).

**Annotations (separate — you need these):** [github.com/declare-lab/MELD](https://github.com/declare-lab/MELD) → `/data/MELD/` → `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`

The `Speaker` column is the gender label source. Videos are named `diaX1_uttX2.mp4` where X1 = `Dialogue_ID` and X2 = `Utterance_ID` — that is the join key.

> ⚠️ **Do not download "Features."** Those are pickle files of GloVe and openSMILE vectors for reproducing someone else's emotion baseline. Nothing visual, nothing you can run a detector on.

**Notes:** clips average 3.59 s, one utterance each — any per-clip temporal statistic will be noisy, and shot-length features are censored because a shot cannot be longer than the clip containing it. Cast is 3 female / 3 male leads, but there are 260 speakers in train, so a name→gender map with a documented policy for the long tail is required. That map is group-level, not per-component.

Licence: GPL-3.0 on the repository. Cite Poria et al., ACL 2019.

---

## AMI Meeting Corpus

Multi-party meetings, 3–5 people. Contrast corpus only — see the caveats.

**Video:** the form at [groups.inf.ed.ac.uk/ami/download](https://groups.inf.ed.ac.uk/ami/download) → select meetings → tick **Low-size DivX AVI videos** → press the button → run the generated `.bat` (Windows) or `.sh`. Needs `wget` on PATH.

**Annotations:**

- [ami_public_manual_1.6.2.zip](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip) — transcription, speaker turns, participant metadata
- [DOME dominance annotations](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/dome_annotations_M1.csv) — **human judgements of who dominated each meeting**, plus Social Role annotations. Independent human validation signal on real data, potentially without recruiting participants.

> ⚠️ **Do not use the OpenSLR mirror (openslr.org/16).** It is audio only — ~84 GB of acoustic data, zero video. Licences also differ: OpenSLR states CC BY-**NC**-SA, Edinburgh states CC BY 4.0. Cite Edinburgh's terms.

**Three caveats:**

1. **Use Corner / room-view cameras.** Each meeting ships four Closeup cameras (one face each) plus Corner and Overhead. Closeups are single-face fixed shots — unusable for prominence, and the framing the supervisor ruled out.
2. **Independent units are sessions, not meetings.** `ES2002a/b/c/d` are the *same four people*. ES2002–ES2008 gives 28 meetings but only ~7 independent units.
3. **Role confound.** Scenario meetings assign roles, and the project manager speaks more *by design*. If PM assignment skews male, AMI reads as male-biased for reasons unrelated to representation. Check role-to-gender distribution before drawing conclusions.

Full-resolution video is not downloadable (firewire drive order only). Test face detection at DivX resolution before committing to a large batch.

---

## LSMDC / MPII-MD Co-ref+Gender

Film clips with **explicit per-character M/F gender annotation** — the only corpus that has this already done. Treat it as a bonus: nothing in the design depends on it arriving.

1. Apply for credentials: [MPII-MD access page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset/access-to-mpii-movie-description-dataset/)
2. Approval takes days to weeks
3. Download with the provided scripts using the issued username/password — AVI plus JPEG frames
4. Gender annotations: see [Identity-Aware Multi-Sentence Video Description](https://arxiv.org/pdf/2008.09791)

Download page: [sites.google.com/site/describingmovies/download](https://sites.google.com/site/describingmovies/download)

---

## Ruled out

| Corpus | Why |
|---|---|
| MovieNet | Distributes data *excluding movies* — annotations, keyframes, trailers and pretrained features, but no film. You cannot run a detector on a feature vector |
| TVQA | Frames at 3 FPS only. Optical flow between frames 333 ms apart is meaningless and lip activity needs ~25 FPS. Also behind a form requiring supervisor details, with an agreement stating *"please do not share the video frames with others"* |
| MAD | No usable raw video for our purposes |
| Condensed Movies | No gender labels and no advantage over AVA for the same content type |
| Cornell Movie-Dialogs | Dialogue only, no video — cannot support a multimodal fusion |
| Bazinga! | Superseded by MELD for scripted TV |
| CCv2 | Single-subject problem — would require a `co_presence` flag, and the domain is defined by two or more subjects sharing the frame |

> ⚠️ **The rule that ruled several of these out, and that forced C3's F6 change:** a frozen closed-form equation must evaluate **every feature on every corpus it scores**. A corpus that can supply only part of a component's feature set forces a choice — drop the corpus, or drop the feature. Resolve that in the schema decision, never later.

---

## Licence summary — for the TAF

| Corpus | Licence | Redistribution |
|---|---|---|
| AVA annotations | CC BY 4.0 | Permitted with attribution |
| AVA videos | Hosted by CVDF for research | Do not redistribute |
| MELD | GPL-3.0 (repo) | Cite Poria et al. 2019 |
| AMI | CC BY 4.0 (Edinburgh terms) | Permitted with attribution |
| LSMDC / MPII-MD | Signed agreement | Do not redistribute |

Record access dates and software versions alongside these for the reproducibility section.

Note for the ethics section: we process identifiable faces and voices and infer gender. Both are biometric-derived. The position is defensible on public corpora under existing licences — state it that way rather than claiming no sensitive data is involved.
