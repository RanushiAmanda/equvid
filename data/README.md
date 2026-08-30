# Data

**Nothing in this directory is committed except instructions and scripts.**

The corpora total roughly 43 GB and their licences do not permit redistribution — TVQA's agreement states explicitly *"please do not share the video frames with others."* This file plus the download scripts is what makes the project reproducible: run them and you rebuild the same data from the original sources.

Put downloaded corpora in folders named `MELD/`, `AVA/`, `AMI/` at the repository root. `.gitignore` already blocks those names.

---

## Status

| Corpus | Size | Access | Gender labels | Status |
|---|---|---|---|---|
| MELD | ~11 GB | Open | Character names → derivable | ✅ Have it |
| AVA / ActiveSpeaker | ~22 GB | Open | ❌ Needs annotation | ✅ Have it (partial batches) |
| AMI | ~9.6 GB | Open form | ❓ Check participant metadata | ✅ ES2002–ES2008 |
| LSMDC / MPII-MD | — | Application | ✅ Explicit M/F per character | ⏳ Awaiting credentials |
| TVQA | — | Form, ~1 week | Subtitles carry speaker names | ⏳ Not requested |
| ~~MovieNet~~ | — | — | — | ❌ Ruled out — distributes data *excluding movies* |

---

## MELD

Scripted TV (*Friends*). 1,433 dialogues, ~13k utterances. Ships emotion and sentiment labels, useful as M4's downstream task.

**Video:** [MELD.Raw.tar.gz](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz) — `.mp4` clips in train/dev/test tarballs. Also mirrored at [huggingface.co/datasets/declare-lab/MELD](https://huggingface.co/datasets/declare-lab/MELD).

**Annotations (separate — you need these):** [github.com/declare-lab/MELD](https://github.com/declare-lab/MELD) → `/data/MELD/` → `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`

The `Speaker` column is the gender label source. Videos are named `diaX1_uttX2.mp4` where X1 = `Dialogue_ID` and X2 = `Utterance_ID` — that is the join key.

> ⚠️ **Do not download "Features."** Those are pickle files of GloVe and openSMILE vectors for reproducing someone else's emotion baseline. Nothing visual, nothing you can run a detector on.

**Notes:** clips average 3.59 s, one utterance each — motion energy over that window will be noisy. Cast is 3 female / 3 male leads, but there are 260 speakers in train, so a name→gender map with a documented policy for the long tail is required.

Licence: GPL-3.0 on the repository. Cite Poria et al., ACL 2019.

---

## AVA / AVA-ActiveSpeaker

Movie clips. 160 videos for ActiveSpeaker, 3.65M labelled frames, ~39K face tracks. **This is where identity diversity comes from** — every video is a different film with different actors.

**Annotations** (CC BY 4.0):

- `https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_train_v1.0.tar.bz2`
- `https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_val_v1.0.tar.bz2`
- Optional, for M4's action-classification task: `.../annotations/ava_v2.2.zip`

**Videos:** `python download_ava.py` (edit the batch slice inside). Source pattern is `https://s3.amazonaws.com/ava-dataset/trainval/[file_name]`, file list at `.../annotations/ava_speech_file_names_v1.txt`.

**Why ActiveSpeaker matters.** The CSV format is `video_id, frame_timestamp, entity_box, label, entity_id`:

| Field | Gives you |
|---|---|
| `entity_box` | Face bounding box, already normalised → M1 features without MTCNN |
| `entity_id` | Same person linked across frames → screen time, tracking solved |
| `label` | SPEAKING_AND_AUDIBLE / SPEAKING_BUT_NOT_AUDIBLE / NOT_SPEAKING per frame → M2 speaking duration without lip detection or VAD |

**Gender annotation is the outstanding work.** Use `entity_id` — you label a person, not a frame. Sort tracks by duration, label the long ones first since they dominate screen time.

> ⚠️ **Skip AVA-Kinetics.** One annotated frame per clip — no temporal continuity, so no screen time and no motion energy.

---

## AMI Meeting Corpus

Multi-party meetings, 3–5 people. Contrast corpus only — see the caveats.

**Video:** the form at [groups.inf.ed.ac.uk/ami/download](https://groups.inf.ed.ac.uk/ami/download) → select meetings → tick **Low-size DivX AVI videos** → press the button → run the generated `.bat` (Windows) or `.sh`. Needs `wget` on PATH.

**Annotations:**

- [ami_public_manual_1.6.2.zip](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip) — transcription, speaker turns, participant metadata
- [DOME dominance annotations](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/dome_annotations_M1.csv) — **human judgements of who dominated each meeting.** Potential independent validation signal for M4.

> ⚠️ **Do not use the OpenSLR mirror (openslr.org/16).** It is audio only — ~84 GB of acoustic data, zero video. Licences also differ: OpenSLR states CC BY-**NC**-SA, Edinburgh states CC BY 4.0. Cite Edinburgh's terms.

**Three caveats:**

1. **Use Corner / room-view cameras.** Each meeting ships four Closeup cameras (one face each) plus Corner and Overhead. Closeups are single-face fixed shots — unusable for prominence.
2. **Independent units are sessions, not meetings.** `ES2002a/b/c/d` are the *same four people*. ES2002–ES2008 gives 28 meetings but only ~7 independent units.
3. **Role confound.** Scenario meetings assign roles, and the project manager speaks more *by design*. If PM assignment skews male, AMI reads as male-biased for reasons unrelated to representation. Check role-to-gender distribution before drawing conclusions.

Full-resolution video is not downloadable (firewire drive order only). Test face detection at DivX resolution before committing to a large batch.

---

## LSMDC / MPII-MD Co-ref+Gender

Film clips with **explicit per-character M/F gender annotation** — the only corpus that has this already done.

1. Apply for credentials: [MPII-MD access page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset/access-to-mpii-movie-description-dataset/)
2. Approval takes days to weeks
3. Download with the provided scripts using the issued username/password — AVI plus JPEG frames
4. Gender annotations: see [Identity-Aware Multi-Sentence Video Description](https://arxiv.org/pdf/2008.09791)

Download page: [sites.google.com/site/describingmovies/download](https://sites.google.com/site/describingmovies/download)

---

## TVQA

6 TV shows, 460 hours. **No raw video** — frames at 3 FPS only, 43 GB, behind a [form](https://goo.gl/forms/HJiFJSllupqeCbax1) requiring supervisor details and roughly a week's wait.

**Free and immediately useful:**

- [tvqa_subtitles.tar.gz](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/files/tvqa_subtitles.tar.gz) (14 MB) — `.srt` files with speaker names embedded, e.g. `(Ted:)Dude, that's my girlfriend.` Speaking duration per character with no video at all.
- [tvqa_qa_release.tar.gz](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/files/tvqa_qa_release.tar.gz) (15 MB)

**What 3 FPS breaks:** motion energy via optical flow (frames 333 ms apart) and lip-activity speaking detection (needs ~25 FPS). Face count, bounding-box area and centre proximity are fine; screen time is approximable.

> ⚠️ The frozen equation needs **all 12 features on every dataset it scores.** A corpus that can only supply 10 of 12 forces a choice: drop the corpus, or drop the feature from the equation. Resolve this in the schema decision, not later.

---

## Licence summary — for the TAF

| Corpus | Licence | Redistribution |
|---|---|---|
| MELD | GPL-3.0 (repo) | Cite Poria et al. 2019 |
| AVA annotations | CC BY 4.0 | Permitted with attribution |
| AVA videos | Hosted by CVDF for research | Do not redistribute |
| AMI | CC BY 4.0 (Edinburgh terms) | Permitted with attribution |
| LSMDC / MPII-MD | Signed agreement | Do not redistribute |
| TVQA frames | Signed agreement — *"do not share"* | **Explicitly prohibited** |

Record access dates and software versions alongside these for the reproducibility section.
