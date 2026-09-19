# Components

One folder per member. Each holds that member's own work — proposal drafts,
component specification, diagrams, and any code specific to their modality.

| Folder | Member | Reg No | Measures |
|---|---|---|---|
| `C1_Visual/` | Hasangani WRA | IT23224384 | What the frame gives each gender |
| `C2_Temporal/` | Harischandra RASR | IT23194908 | What the timeline gives each gender |
| `C3_Text/` | Fernando RSR | IT23449282 | What the script gives each gender |
| `C4_Audio/` | Sathsarani WTTS | IT23204102 | How each gender's voice is treated in the sound |

**What does NOT go in here.** Anything the group shares: `data/` (corpus
acquisition), `src/` (pipeline code, including the `shared/` modules every
component must use rather than fork), `docs/` (group documents), and
`docs/papers/` (EuqAud, the TAF, the template, the structure document).

**The boundary rule.** Time is Temporal. Sound is Audio. The frame is Visual.
Words are Text. Speaking duration belongs to C2 because it is seconds; voice
loudness belongs to C4 because it is decibels. This is the boundary most likely
to drift during implementation.

Only C3 has files at present — the other three members' work is on their own
machines and has not been committed here.
