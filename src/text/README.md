# src/text — C3 (EuqVid-Text)

Owner: Fernando R. S. R. (IT23449282)

On Windows, double-click `run_c3_text.bat`. It runs the seven steps below and stops at
the first error.

- **First run:** a few minutes, because step 3 parses every line with spaCy.
- **Later runs:** two or three minutes. Step 3 is skipped while its inputs are unchanged,
  and step 5 takes about a minute for its 1,000 bootstrap refits.

To run the steps by hand, use this folder and this order (use `py -3` if `python` is not
found):

```
python build_utterances.py       # 1  MELD + AMI utterance tables, gender map, clip list
python lexical_features.py       # 2  counts per clip and gender, parser-free features, CIs
python parse_frames.py           # 3  spaCy: who does what in each line (needs spaCy)
python agency_power_features.py  # 4  agency and power from the lexicon, CIs
python c3_equation.py            # 5  the C3 equation: fit, freeze, normalise, tiers, bootstrap
python plot_lexical.py           # 6  the charts
python test_text_features.py     # 7  hand-worked tests; all must pass
python test_c3_equation.py       # 7  the same for the equation
```

The scripts read the corpora and the lexicon from the `Research Project` folder, three
levels up, and never modify them. To use another location, pass
`--root "<path to Research Project>"`.

| File | What it does |
|---|---|
| `wordlists.py` | Tokenizer and pronoun lists, in one place. `FEATURE_VERSION` is written into every output row |
| `build_utterances.py` | One row per utterance for MELD (CSVs) and AMI (manual transcript XML, read straight from the zip). Also writes the provisional gender map and clip list |
| `lexical_features.py` | Counts per clip and gender, the parser-free features, and pooled corpus values with a bootstrap over series |
| `lexicon.py` | Loads the Sap et al. agency/power lexicon (with its known quirks) and turns labels into scores |
| `frames.py` | The rules that find frames: a verb plus the character doing it or receiving it ("I quit", "she fired him"). No spaCy inside, so the rules can be tested anywhere |
| `parse_frames.py` | Runs spaCy over the lines and applies `frames.py`. This is the only step that needs spaCy |
| `agency_power_features.py` | Scores the frames with the lexicon. Writes agency and power per corpus with CIs, the pre-declared checks, coverage and the most frequent verbs |
| `c3_equation.py` | The C3 equation (decision D7). Builds shards, biases them on purpose by known amounts, fits polynomial Ridge, freezes the closed form in `results/c3_equation.json`, normalises to −10…+10, derives the tiers and runs 1,000 bootstrap refits. `--dry-run` checks the pipeline without printing any result |
| `plot_lexical.py` | Black-and-white charts for slides (fig1–fig4) |
| `test_text_features.py` | Toy conversations and hand-built parses worked out by hand, so each rule can be explained in a viva |
| `test_c3_equation.py` | The equation's operators, labels, Ridge, normalisation, tiers and assumption statistics, each checked against numbers worked out by hand, plus the whole pipeline on made-up clips |

## Outputs

- **Tables** go to `Research Project/derived/c3/`, outside the repo, because they contain
  transcript text.
- **Summaries and charts** go to `components/C3_Text/results/`.

Definitions and the reasons behind them are in `components/C3_Text/C3_decision_log.md`.

## Provisional parts

- **Gender map.** `gender_map_provisional.csv` follows C2's labels and will be replaced by
  the group's shared `gender_map.csv` once decision 7 is agreed.
- **Clip unit.** A clip is one MELD episode or one AMI meeting, as C2 proposes.
- **Shards.** `c3_equation.py` draws its own provisional shards, 100 per corpus of 30
  clips each, within each corpus (D7). Because every count is stored per clip, the
  group's shard design can replace them by summing, with no re-extraction.
