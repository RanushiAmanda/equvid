# src/text — C3 (EuqVid-Text)

Owner: Fernando R. S. R. (IT23449282)

On Windows, double-click `run_c3_text.bat`: it runs all four steps and stops at the first
error. By hand, run from this folder in this order (use `py -3` if `python` is not found):

```
python build_utterances.py      # ~10 s: MELD + AMI utterance tables, gender map, clip list
python lexical_features.py      # ~30 s: counts per clip and gender, features, bootstrap CIs
python plot_lexical.py          # the three charts
python test_text_features.py    # hand-worked tests; all must pass
```

The scripts read the corpora from the `Research Project` folder, three levels up, and never
modify them. To point them at another location, pass `--root "<path to Research Project>"`.

| File | What it does |
|---|---|
| `wordlists.py` | Tokenizer and pronoun lists, in one place. Holds `FEATURE_VERSION`, which is written into every output row |
| `build_utterances.py` | One row per utterance for MELD (CSVs) and AMI (manual transcript XML, read straight from the zip). Also writes the provisional gender map and clip list |
| `lexical_features.py` | Counts per clip and gender, the parser-free features, and pooled corpus values with a bootstrap over series |
| `plot_lexical.py` | Black-and-white charts for slides |
| `test_text_features.py` | Toy conversations worked out by hand, so each formula can be explained in a viva |

## Outputs

- **Tables** go to `Research Project/derived/c3/`, outside the repo because they contain
  transcript text.
- **Summaries and charts** go to `components/C3_Text/results/`.

## Provisional parts

- **Gender map.** `gender_map_provisional.csv` follows C2's labels. It is replaced by the
  group's shared `gender_map.csv` once decision 7 is agreed.
- **Clip unit.** A clip is one MELD episode or one AMI meeting, as C2 proposes. Shards are
  not built yet. Because every count is stored per clip, any shard design can be computed
  by summing.
