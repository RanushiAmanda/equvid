# Data

**Nothing in this directory is committed except instructions and scripts.**

The corpora total well over 100 GB if AVA is kept as full films, and their licences do not permit redistribution. This file plus the download scripts is what makes the project reproducible: run them and you rebuild the same data from the original sources.

Downloaded corpora live in `MELD/`, `AVA/` and `AMI/` **beside** the repository, not inside it (`Research Project/AVA/` next to `Research Project/equvid/`). The download scripts write there, so corpus files cannot be committed even by `git add -A`. `.gitignore` also blocks those names as a second line of defence.

---

## Status

| Corpus | Size | Access | Gender labels | Status |
|---|---|---|---|---|
| AVA / ActiveSpeaker | ~97 GB as full films (summed from S3 sizes); the labelled 15 minutes are roughly a tenth of that | Open | ❌ Manual annotation per `entity_id` | ⏳ Labels complete and verified (ActiveSpeaker + AVA-Speech). Full-film download running since 11 Oct: 40 of 153 films on disk, each exactly its S3 size; 73 GB to go |
| MELD | ~11 GB | Open | Character names → derivable | ✅ Have it |
| AMI | 12.0 GB room-view video + 3.7 GB headset-mix audio for the 16 usable series | Open | ✅ `sex` in `participants.xml` (see TS gotcha) | ✅ Complete for all 16 usable series: 148 room-view videos and 64 `Mix-Headset` tracks. Every file is size-checked against the server; samples decode cleanly (11 Oct) |

**Ruled out — do not spend time on these again:** Condensed Movies, TVQA, MAD, MovieNet, Cornell Movie-Dialogs, Bazinga!, CCv2, LSMDC. Reasons are in the table at the foot of this file.

**Which corpus carries which load:**

- **AVA is the primary fusion corpus.** It is the only feature-film source, it is where the disparity the metric measures is most expected, and it is the one corpus all four components can run on — so a fused EuqVid score is computed on the same material rather than on a union of partially-overlapping corpora.
- **MELD is the negative control**, not a bias demonstration. Friends is an unusually balanced ensemble (3 female / 3 male leads, main-cast speaking-time M:F = 1.094). A metric that correctly scores a balanced corpus near Neutral is evidence against false positives, and examiners ask for exactly that.
- **AMI is the structural contrast corpus** — non-narrative, and the only source of independent human dominance judgements.
- **Every corpus must be usable by all four members.** A corpus that only one member is licensed to use cannot feed a shared gender map, shared shards or a fused score. This rule is why LSMDC was dropped (see *Ruled out*).

---

## AVA / AVA-ActiveSpeaker

Movie clips. **153 videos carry public ActiveSpeaker labels** (120 train + 33 val — the val archive unpacks into a folder named `..._test_v1.0`, which is confusing but correct). The figure of 160 quoted elsewhere (from the dataset paper) is not what the public archives contain; use 153 in any count you report, and check the paper before explaining the difference. The paper quotes 3.65M labelled frames and ~39K face tracks; the released archives hold **3,444,621 frames and 37,738 tracks**, all between 900 s and 1800 s, so report those. **This is where identity diversity comes from** — every video is a different film with different actors.

**Annotations** (CC BY 4.0):

- `https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_train_v1.0.tar.bz2`
- `https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_val_v1.0.tar.bz2`
- AVA-Speech, `.../annotations/ava_speech_labels_v1.csv` (on disk, verified). It covers 160 films, each labelled for exactly 900–1800 s, including **all 153** ActiveSpeaker films. Labels are `CLEAN_SPEECH`, `SPEECH_WITH_MUSIC`, `SPEECH_WITH_NOISE` and `NO_SPEECH`.
  - C3 uses it to gate ASR segments, and C4 may use it to restrict or cross-check `speech_to_background_ratio` (research plan §4.5).
  - **Clean speech is only 27% of all speech time** (5.7 of 21.0 h). Gating on `CLEAN_SPEECH` alone would discard about three quarters of the dialogue, so choose the gate deliberately and report what it removes.
- Optional, for downstream action classification: `.../annotations/ava_v2.2.zip`

**Videos:** run from `equvid/data/`:

```
python download_ava.py --list      # 153 targets: what is on disk, what is missing
python download_ava.py --verify    # check files already on disk against the server
python download_ava.py             # download everything still missing (resumable)
python download_ava.py --limit 10  # or in batches
```

The target list is read from the ActiveSpeaker archives themselves, so "complete" means every video we have labels for. The earlier script walked the AVA-Speech file list alphabetically instead, which is how one unlabelled video (`8VZEwOCQ8bc`) ended up on disk. Videos are full films (0.1–3.7 GB each); only minutes 15–30 are annotated.

Downloads write to `<name>.part` and resume with HTTP Range requests, so an interruption costs only the unfinished tail. Existing files are checked against the server's size before being trusted — the earlier script wrote straight to the final name, so an interrupted download would have left a truncated file that looked complete.

**Why ActiveSpeaker matters.** The CSV format is `video_id, frame_timestamp, entity_box, label, entity_id`:

| Field | Gives you |
|---|---|
| `entity_box` | Face bounding box, already normalised → C1 features without MTCNN |
| `entity_id` | Same person linked across frames → screen time, tracking solved |
| `label` | `SPEAKING_AUDIBLE` / `SPEAKING_NOT_AUDIBLE` / `NOT_SPEAKING` per frame → C2 speaking duration without lip detection or VAD, and C3 speaker attribution without diarisation |

> ⚠️ **Use the label strings exactly as above.** Several of our documents write `SPEAKING_AND_AUDIBLE` / `SPEAKING_BUT_NOT_AUDIBLE`, and those strings do not occur in the files. Code filtering on them silently matches zero rows. Counts in the released CSVs: `NOT_SPEAKING` 2,536,949, `SPEAKING_AUDIBLE` 875,152, `SPEAKING_NOT_AUDIBLE` 32,520.

**Gender annotation is the outstanding work**, and it is declared work in all four reports rather than future work. Use `entity_id` — you label a person, not a frame. Sort tracks by duration and label the long ones first, since they dominate screen time. Inter-annotator agreement must be reported.

**AVA ships no dialogue.** C3 recovers a transcript with one ASR pass over clip audio, attributes speakers by aligning ASR segments to `SPEAKING_AUDIBLE` spans, and gates on AVA-Speech. The resulting word error must be quantified against a hand-checked subset and reported with every AVA text result — see `docs/C3_AVA_decision_and_F6_change.md`.

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

**Video and audio:** run from `equvid/data/` — no download form or manifest needed, the script builds the URLs from the meeting list inside the annotation zip:

```
python download_ami.py --list      # per-series gender mix and what is on disk
python download_ami.py             # room-view video for every mixed-gender series
python download_ami.py --audio     # ... plus headset-mix audio (C4: see the audio note below)
python download_ami.py --all       # every scenario series, single-gender included
```

> **If an earlier run reported 48 IS/TS files as 404:** that was the script, not the server. The first version asked every site for `Corner`/`Overhead`, which exist only in Edinburgh. Fixed in the current version. Re-running it fetches only what is missing (~5–6 GB for the six IS/TS series).

> **Also on disk, not used by anyone:** the four Closeup cameras for ES2002–ES2008, plus all six cameras for ES2003, which is all-male. These are 120 files, 6.9 GB, left over from the August manual download. Nothing in the plan reads them; delete them if the space is needed.

**Audio for C4 — the microphone is C4's decision.** `Mix-Headset` is on disk for all 64 meetings (16 kHz mono, verified 11 Oct); the other streams are not. Audio file names are the same at all three sites (checked on ES2002a, IS1000a, TS3005a). Sizes for the 16 usable series (32.6 h of meetings):

| Stream | What it is | Size | How to get it |
|---|---|---|---|
| `Mix-Headset` | the four headset mics mixed into one track | ~3.8 GB | `python download_ami.py --audio-only` |
| `Headset-0` … `Headset-3` | one close-talk mic per participant | ~15 GB | script change needed |
| one `Array1-0n` channel | far-field microphone on the table | ~3.8 GB | script change needed |

EuqAud describes its training audio as pre-segmented, single-speaker, near-field speech (Section V), so the per-participant headsets are the closest match for the transferred baseline features. A far-field channel is what a room-acoustics reading of `speech_to_background_ratio` would need. Speaker timings for attributing any of these to people are already on disk: `words/` and `segments/` inside the annotation zip.

**Why mixed-gender series only, by default.** EuqVid measures disparity between genders present in the same material, and a series in which all four people share one gender has nothing to compare. From AMI's own `participants.xml`, of the 35 scenario series only **16 are mixed-gender**; 14 are all-male and 5 all-female, and **9 of the 10 TS series are all-male**. Downloading the other 19 would add no usable shard.

| Site | Series | Mixed-gender (usable) | All-male | All-female |
|---|---|---|---|---|
| ES | 15 | 10 | 1 (ES2003) | 4 (ES2010, ES2011, ES2013, ES2015) |
| IS | 10 | 5 | 4 (IS1002, IS1004, IS1005, IS1007) | 1 (IS1009) |
| TS | 10 | 1 (TS3005) | 9 | 0 |

The transcripts C3 needs are already complete: word-level transcripts for all 171 meetings are inside `ami_public_manual_1.6.2.zip`.

> ⚠️ **The 40 TS (TNO) speakers have no verified gender.** `participants.xml` has no entries for TNO participants at all. It lists 189 people, and every one of them who speaks in a meeting (150) is from the Edinburgh or Idiap meetings. The TS speakers appear only in `meetings.xml`, with the role glued on (`MTD009PM`).
>
> - **What is known:** the only gender information is the first letter of the ID (`F`/`M`). For all 189 participants who do have an entry, that letter matches the recorded sex (checked October 2026).
> - **How to label them:** use the letter, record it as its own source (e.g. `gender_source = id_prefix`), and keep it out of the "verified" set. This is how the shared conventions treat them: 82 mixed-gender meetings with verified labels, 86 with TS3005 added.
> - **Correction:** an earlier version of this note claimed `participants.xml` lists the bare ID (`MTD009`). It does not.

**Annotations:**

- [ami_public_manual_1.6.2.zip](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip) — transcription, speaker turns, participant metadata
- [DOME dominance annotations](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/dome_annotations_M1.csv) — **human judgements of who dominated each meeting**, plus Social Role annotations. Independent human validation signal on real data, potentially without recruiting participants.

> ⚠️ **Do not use the OpenSLR mirror (openslr.org/16).** It is audio only — ~84 GB of acoustic data, zero video. Licences also differ: OpenSLR states CC BY-**NC**-SA, Edinburgh states CC BY 4.0. Cite Edinburgh's terms.

**Three caveats:**

1. **Use the room-view cameras.** Each meeting ships four Closeup cameras (one face each) plus room views, whose names differ by site: `Corner` and `Overhead` at Edinburgh (ES), `C`, `L` and `R` at Idiap (IS), `Overview1` and `Overview2` at TNO (TS). Closeups are single-face fixed shots — unusable for prominence, and the framing the supervisor ruled out. Look at a frame from each site before assuming the room views frame people the same way.
2. **Independent units are series, not meetings.** `ES2002a/b/c/d` are the *same four people*. ES2002–ES2008 gives 28 meetings and 7 series — but **only 6 usable**, because ES2003 is all-male. The whole scenario corpus caps at 16 usable series.
3. **Role confound — checked.** Scenario meetings assign roles, and the project manager speaks more *by design*. Across all 35 scenario series the PM is female in 11 of 35 (31%), the same as the corpus's overall female share (44 of 140 people, 31%). So role assignment does not add skew beyond AMI's own composition — but the composition itself is 69% male, and that, not the PM role, is the confound to report. C2 should re-check this on whichever series are finally used.

The files we download are 352×288, 25 fps, with no audio track. Test face detection at that resolution before committing to a large batch. Earlier notes said full-resolution video was available only by drive order. However, the mirror also lists an `<name>_orig.avi` for every camera, 3–8× the size of the normal file. **Its resolution has not been checked.** If faces turn out too small at 352×288, read the header of one `_orig` file first.

---

## Lexicons

**Connotation frames of agency and power** (Sap et al., EMNLP 2017) — the lexicon behind C3's F4 and F5. It lives in `Research Project/Lexicons/sap2017_agency_power/`, beside the corpora.

- **Source.** The paper's old project page now redirects to Riveter ([github.com/maartensap/riveter-nlp](https://github.com/maartensap/riveter-nlp)), so the file comes from there: `riveter/data/sap-lexicon/agency_power.csv`, commit `eef4f12`, retrieved 11 Oct 2026, unchanged.
- **Licence.** GPL-3.0, as distributed in Riveter; the licence file sits beside the CSV.
- **Contents.** 2,150 rows covering 2,148 verbs. Agency labels are `pos` / `neg` / `equal`, and power labels are `agent` / `theme` / `equal`.
- **Quirks the loader must handle.** These are listed in `SOURCE.md` in that folder:
  - three entries are not lemmatised (`assists`, `bars`, `pops`);
  - two verbs are split over two rows (`echo`, `go`);
  - 65 entries are multi-word, such as `apologize to`;
  - 414 verbs have no power label.

Only the CSV is used, not Riveter's pipeline. Riveter attributes verbs to characters through coreference, and pins Python 3.9 and an experimental spaCy coreference model. C3 attributes verbs to the speaker instead, so a spaCy parse plus this file is enough.

**Software for C3** is installed by `equvid/components/C3_Text/setup/setup_c3.bat`: spaCy with `en_core_web_sm`, VADER and ffmpeg (via `imageio-ffmpeg`). It finishes with a self-check that also measures the lexicon's coverage of MELD verbs and VADER's agreement with MELD's sentiment labels, and writes the results to `c3_setup_report.txt`.

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
| LSMDC / MPII-MD | Dropped October 2026. Access is granted to one named person under a signed agreement, and only one member holds it, so the group cannot use it. Its gender labels covered audio-description narration, not dialogue, so no fitted feature depended on it. The downloaded annotation files stay on that member's computer only, outside anything shared with the group |

> ⚠️ **The rule that ruled several of these out, and that forced C3's F6 change:** a frozen closed-form equation must evaluate **every feature on every corpus it scores**. A corpus that can supply only part of a component's feature set forces a choice — drop the corpus, or drop the feature. Resolve that in the schema decision, never later.

---

## Licence summary — for the TAF

| Corpus | Licence | Redistribution |
|---|---|---|
| AVA annotations | CC BY 4.0 | Permitted with attribution |
| AVA videos | Hosted by CVDF for research | Do not redistribute |
| MELD | GPL-3.0 (repo) | Cite Poria et al. 2019 |
| AMI | CC BY 4.0 (Edinburgh terms) | Permitted with attribution |
| Sap et al. agency/power lexicon | GPL-3.0 (as distributed in Riveter) | Permitted with the licence; cite Sap et al. 2017 |

Record access dates and software versions alongside these for the reproducibility section.

Note for the ethics section: we process identifiable faces and voices and infer gender. Both are biometric-derived. The position is defensible on public corpora under existing licences — state it that way rather than claiming no sensitive data is involved.
