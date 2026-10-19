# C3 — EuqVid-Text

**Fernando RSR (IT23449282)** · repository holder

Measures what the script writes for each gender. Deliberately
language-layer-agnostic: it extracts agency and power disparity from whichever
text layer a corpus provides, so it is not load-bearing on any single corpus.

**Features in the C3 equation (6):** `mean_utterance_length`,
`agency_score_mean`, `power_score_mean`, `same_gender_dialogue_share`,
`ref_log_parity`, `sentiment_valence_mean` (tool not yet validated).

- **`dialogue_volume_share` (F1) left the equation on 19 Oct 2026.** Words per
  person repeat C2's speaking-time parity (r = 0.96 MELD, 0.97 AMI), so the
  fusion would count speaking share twice. It is still computed, as a pipeline
  check.
- **`ref_log_parity` replaces `gendered_reference_ratio`.** As written,
  `gendered_reference_ratio` showed no difference between women and men on
  either corpus. `ref_log_parity` measures how often each gender is referred to
  as he/she, per person. It is roster-free like its predecessor. The group
  still has to confirm this, because the earlier F6 rename was a group-level
  change.
- `utterance_count_share` was deleted earlier: it duplicates C2's
  `turn_count_share` and would be double-weighted in the fusion.

The reasons and evidence for each change are in `C3_decision_log.md`.

**Corpora:** MELD, AMI, and AVA / AVA-ActiveSpeaker. AVA ships no dialogue, so
the transcript has to be recovered:

- **Words:** recovered by ASR.
- **Speaker attribution:** from AVA-ActiveSpeaker's `SPEAKING_AUDIBLE` spans,
  so no diarisation is needed.
- **Quality gating:** against AVA-Speech.
- **Gender:** assigned per `entity_id` by hand.

ASR word error must be quantified against a hand-checked subset and reported
with every AVA result.

## Contents

| Path | What it is |
|---|---|
| `C3_decision_log.md` | Every feature and method decision, with its evidence and who still has to agree |
| `results/` | Results and charts. Start with `C3_results_01_lexical_features.md` |
| `setup/` | One-click install for this PC (`setup_c3.bat`) and its check report |
| `EuqVid_Text_Component_Specification.pdf` | The component specification. Markdown source is in the Claude project. Its feature table predates the decision log |
| `proposals/` | Four proposal drafts — `_v3` is the current one |
| `diagrams/` | Six draw.io figures with PNG exports: overall architecture, component architecture, use case, sequence, WBS, gantt |

The code is in `src/text/`. Double-click `run_c3_text.bat` there to rebuild
the tables, features and charts.

## Still open

- Agency and power features (`agency_score_mean`, `power_score_mean`): the
  spaCy script is next.
- Sentiment: VADER did not beat the majority class on MELD's labels. Choose a
  validated tool, or drop the feature.
- `same_gender_dialogue_share`: keep or drop. It can be computed on only 34% of
  AMI meetings, and on AMI it still tracks floor share.
- Verify AVA clip count and current licence terms on the distributor's own page
  before submission.
- Decide with the group whether AVA is a **training** corpus or a **held-out
  application** corpus.
- Record the alternatives considered before AVA was selected in the decision
  log, Appendix 2.
