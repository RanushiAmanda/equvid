# C2 — EuqVid-Temporal

**Harischandra RASR (IT23194908)**

Measures how the timeline is shared out — who holds the floor, when, and how
evenly. Core framing: *equal time is not equal treatment*.

**Six features:** `speaking_time_share`, `turn_count_share`, `mean_turn_length`,
`interruption_asymmetry`, `speech_dispersion`, `presence_voice_gap`.
Two film-only extras, `mean_shot_duration` and `cut_share`, are reported as a
sub-analysis and never enter the frozen equation.

**Three structural augmentation operators:** O1 Fragmentation, O2 Interruption
injection, O3 Temporal clustering — all holding total speaking time constant.

**What belongs here:** proposal drafts, the C2 component specification,
C2 diagrams, and any temporal-extraction code not shared with other components.

**Empty at present.** Nothing has been committed from this member's machine.

Note that `src/meld_speaking_baseline.py` sits in the shared `src/` tree rather
than here: it establishes MELD as the project's negative control, which all four
components cite.
