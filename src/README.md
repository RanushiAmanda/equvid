# Source code

Pipeline modules live here.

## What exists now

- `meld_speaking_baseline.py` — speaking-time ratio from the MELD annotation CSVs
  alone, no video processing. Produces the main-cast M:F = 1.094 result that makes
  MELD the project's negative control.

## Planned layout

Organised by modality, matching the four components. Each component owns its own
extraction and fits its own sub-metric; the shared modules below the line are
group-level and must not be forked per component, or the four sub-scores stop
being comparable.

```
src/
├── visual/        C1 — screen_time_share, mean_relative_face_area,
│                       mean_centre_proximity, closeup_share,
│                       solo_frame_share, identity_share
├── temporal/      C2 — speaking_time_share, turn_count_share, mean_turn_length,
│                       interruption_asymmetry, speech_dispersion,
│                       presence_voice_gap
├── text/          C3 — dialogue_volume_share, mean_utterance_length,
│                       agency_score_mean, power_score_mean,
│                       same_gender_dialogue_share, gendered_reference_ratio,
│                       sentiment_valence_mean
├── audio/         C4 — mean_voice_loudness, speech_to_background_ratio,
│                       dynamic_range, f0_std, energy_std, amplitude_std
└── shared/        group-level, one implementation for all four
    ├── gender_map.py     one character-to-gender map, used by every component
    ├── sharding.py       identity-partitioned shard construction
    ├── augmentation.py   the single frozen augmentation protocol
    ├── metric.py         polynomial Ridge fitting, degree capped at 2
    ├── calibration.py    min-max normalisation on the shared calibration set
    ├── stability.py      bootstrap at shard and identity level, never row level
    └── fusion.py         the four sub-scores into one EuqVid score
```

## Three rules that apply to every module here

1. **Shares and ratios only.** No raw seconds or counts survive into a feature
   column. A raw magnitude grows with dataset size and would make a large balanced
   corpus score as more biased than a small skewed one.
2. **One gender map.** All four components resolve identity to gender through
   `shared/gender_map.py`. Four private maps mean the four sub-scores describe
   four different populations and cannot be fused.
3. **Screen features inside cross-validation folds, never on the full set
   beforehand.** Screening then fitting is leakage, and it is a standard viva
   question.
