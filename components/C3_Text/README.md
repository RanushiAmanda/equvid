# C3 — EuqVid-Text

**Fernando RSR (IT23449282)** · repository holder

Measures what the script writes for each gender. Deliberately
language-layer-agnostic: it extracts agency and power disparity from whichever
text layer a corpus provides, so it is not load-bearing on any single corpus.

**Features:** `dialogue_volume_share`, `mean_utterance_length`,
`agency_score_mean`, `power_score_mean`, `same_gender_dialogue_share`,
`gendered_reference_ratio` (F6), `sentiment_valence_mean`.

- `utterance_count_share` was deleted — it duplicates C2's `turn_count_share`
  and would be double-weighted in the fusion.
- **F6 changed, and it is a group-level change.** It was `mention_ratio`, which
  needs a cast roster AVA does not ship. It is now `gendered_reference_ratio`
  for all corpora: the share of third-person gendered references in each
  gender's speech, divided by that gender's total references to people.
  Roster-free, transcript-only, portable.

**Corpora:** MELD, AMI, and AVA / AVA-ActiveSpeaker. AVA ships no dialogue —
the transcript is recovered by ASR, with speaker attribution from
AVA-ActiveSpeaker's `SPEAKING_AND_AUDIBLE` spans (so no diarisation), quality
gating against AVA-Speech, and gender assigned per `entity_id` by hand.
ASR word error must be quantified against a hand-checked subset and reported
with every AVA result.

## Contents

| Path | What it is |
|---|---|
| `EuqVid_Text_Component_Specification.pdf` | The component specification. Markdown source is in the Claude project |
| `proposals/` | Four proposal drafts — `_v3` is the current one |
| `diagrams/` | Six draw.io figures with PNG exports: overall architecture, component architecture, use case, sequence, WBS, gantt |

## Still open

- Verify AVA clip count and current licence terms on the distributor's own page
  before submission.
- Decide with the group whether AVA is a **training** corpus or a **held-out
  application** corpus.
- Record the alternatives considered before AVA was selected in the decision
  log, Appendix 2.
