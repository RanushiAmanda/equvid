# C2_Temporal: EuqVid-Temporal (M2, floor allocation)

Measures how the timeline is shared out: who holds the floor, when, and how evenly.
Core framing: **equal time is not equal treatment.**

Owner: Shashindi Harischandra (IT23194908). Supervisor: Dr. Prasanna S. Haddela.

## Datasets used in this component

| Corpus | What it provides here |
|---|---|
| MELD (TV dialogue, *Friends*) | speaker, start/end times per utterance; speaker gender from a hand-built cast map |
| AMI (meetings) | word/segment timing per speaker; gender from `participants.xml` |
| AVA-ActiveSpeaker (films) | per-frame speaking labels per face; gender is **predicted** from faces (see Notes) |

LSMDC (the group's fourth dataset) belongs to the text component. It has no speaker-turn timing, so it is not used here.
Raw data, videos and face crops are never committed (see `.gitignore`).

## Pipeline

1. **Ingestion to intervals:** every corpus becomes one table: session, speaker, gender, start, end, duration.
2. **Intervals to turns:** consecutive utterances of one speaker merge into a turn when the silence between them is at most `GAP = 1.0 s`; utterances of 2 words or fewer are treated as backchannels, not turns.
3. **Turns to shards (clips):** a clip needs at least `MIN_TURNS = 20` turns; shard table gives the null reference.
4. **Features** per clip (below).
5. **Structural augmentation** (below) creates controlled, labelled variations.
6. **Diagnostics, fit and freeze** gives the frozen model `results/m2_model_v0.1.json`.
7. **Validation** on real data (DOME check, final notebook, `presence_voice_gap` on real video data).

## Six features

| Feature | Meaning |
|---|---|
| `speaking_time_share` | share of total speaking time |
| `turn_count_share` | share of turns |
| `mean_turn_length` | average turn length |
| `interruption_asymmetry` | who is cut into, by whom (code: `interruption_asym`) |
| `speech_dispersion` | spread of turn lengths / positions (code: `len_sd_ratio`, `pos_sd_ratio`) [CONFIRM mapping] |
| `presence_voice_gap` | log ratio of speaking rate per on-screen second, women vs men; negative = seen but not heard |

Two film-only extras, `mean_shot_duration` and `cut_share`, are a sub-analysis and never enter the frozen equation. [CONFIRM status]

`presence_voice_gap` is reported as an **excess** over a null (observed gap minus the gap expected if both genders spoke at the same rate), because the raw pooled gap is not zero under "no bias". This definition was introduced after seeing the raw results (post hoc) and is disclosed as such.

## Structural augmentation operators

All three hold total speaking time constant and change only how it is arranged:

- **O1 Fragmentation:** splits turns into shorter pieces
- **O2 Interruption injection:** inserts cut-ins
- **O3 Temporal clustering:** moves a speaker's turns together in time

A signed level selects whose turns are modified (positive: women, negative: men).

## Contents of this folder

    M2_FLOOR_ALLOCATION.md       short overview of the component
    src/m2_temporal.py           turns, clip features, allocation and structure scores
    results/m2_model_v0.1.json   frozen structure model v0.1
    results/dome_final_check.csv final DOME check
    results/shard_table.csv      shard null reference table
    notebooks/01_meld.ipynb      MELD
    notebooks/02_ami.ipynb       AMI
    notebooks/03_ava.ipynb       AVA
    notebooks/04_m2_final.ipynb  final validation (outputs cleared)

The MELD speaking-time baseline (`src/meld_speaking_baseline.py`) lives in the shared `src/` tree: it sets MELD as the project's negative control for speaking time, which all four components cite. `presence_voice_gap` is a different measurement (speaking per on-screen second) and is reported separately.

## Reproducing

Place the datasets under `data/` (not in the repo), then run notebooks 01 to 04 in order.

## Status and next

- Done: shared pipeline for the three corpora, frozen model v0.1, validation notebooks.
- In progress: sensitivity of `GAP` and the backchannel rule, MELD episode-level resampling (episodes share one cast, so they are not independent), identity-partition check.
- Waiting for supervisor approval: scripts and results for `presence_voice_gap` (they use a face-based gender classifier).

## Notes

- Gender for AVA is **predicted (perceived) gender** from faces, checked on 100 faces by hand. It is not self-identified gender.
- The AVA train split has 120 videos; 88 could be downloaded and used.
- AMI has no person-level on-screen data, so `presence_voice_gap` is not available for AMI.
