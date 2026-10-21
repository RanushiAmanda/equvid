# EuqVid

**A Multimodal Approach to Gender Bias Detection in Video Datasets**

Group J26-DS-343 · CoEAI · SLIIT · IT4010 Research Project
Supervisor: Dr. Prasanna S. Haddela · Co-supervisor: Ms. Adya Dissanayake

Extends [EuqAud](https://doi.org/10.1109/ACCESS.2026.3674991) (Jayawardena et al., IEEE Access 2026) from the audio modality into video.

---

## What this is

A pre-training, model-agnostic metric that scores gender bias in video datasets before model development begins.

Each of four modalities — visual, temporal, textual, acoustic — produces its own frozen closed-form polynomial Ridge equation over scale-free features. The four sub-scores are then fused into a single EuqVid score on [−10, +10], with Neutral / Moderate / Strong severity tiers. No classifier is trained at any stage, and the result fits on one page and can be defended line by line.

**Domain:** narrative multi-person video — film and scripted television. Corpora where two or more subjects share the frame and shot composition reflects narrative importance.

**What it measures:** *within-content* representational disparity. Given subjects of both genders in the same material, is one systematically afforded greater visual, temporal, textual or acoustic prominence? Zero means parity — not "the naturally occurring distribution."

**Reading the score:** 0 = parity · positive = male-dominant · negative = female-dominant · below 2 = Neutral · 2 to 6 = Moderate · above 6 = Strong.

---

## Team

| Member | Reg No | Component | Measures |
|---|---|---|---|
| Hasangani WRA | IT23224384 | C1 — EuqVid-Visual | What the frame gives each gender |
| Harischandra RASR | IT23194908 | C2 — EuqVid-Temporal | What the timeline gives each gender |
| Fernando RSR | IT23449282 | C3 — EuqVid-Text | What the script gives each gender |
| Sathsarani WTTS | IT23204102 | C4 — EuqVid-Audio | How each gender's voice is treated in the sound |

Group leader: Hasangani WRA. Repository held by Fernando RSR.

**The boundary rule.** Time is Temporal. Sound is Audio. The frame is Visual. Words are Text. Speaking duration belongs to Temporal because it is seconds; voice loudness belongs to Audio because it is decibels. This is the boundary most likely to drift during implementation — it is in the group agreement for that reason.

---

## Repository layout

```
equvid/
├── README.md               you are here
├── .gitignore              blocks datasets from ever being committed
├── components/             one folder per member — individual work only
│   ├── C1_Visual/          Hasangani
│   ├── C2_Temporal/        Harischandra
│   ├── C3_Text/            Fernando — specification, proposals/, diagrams/
│   └── C4_Audio/           Sathsarani
├── docs/                   group documents
│   ├── claude_project_instructions.md
│   ├── Step1_dataset_verification.md
│   └── papers/             EuqAud, the TAF, the template, the structure document
├── data/                   how to obtain the corpora — NOT the corpora themselves
│   ├── README.md           per-corpus instructions, licences, traps
│   ├── download_ava.py
│   └── download_ami.py
├── src/                    shared pipeline code
└── notebooks/              exploration
```

Individual work lives in `components/`; anything the group shares stays outside it. Only `C3_Text/` has files at present — the other three members' work is on their own machines and has not been committed here. See [components/README.md](components/README.md).

---

## Getting the data

**The datasets are not in this repository and must never be added to it.** They total roughly 43 GB and their licences do not permit redistribution.

To set up on a new machine, see **[data/README.md](data/README.md)**. Every corpus is re-downloadable from its original source; that file plus the download scripts is what makes this project reproducible.

| Corpus | Role |
|---|---|
| AVA / AVA-ActiveSpeaker | Primary fusion corpus — feature film, 160 videos, supports all four components |
| MELD | Negative control — balanced ensemble, speaking-time M:F = 1.094 |
| AMI | Structural contrast corpus — non-narrative |

---

## Working documents

### In this repo

| Document | Purpose |
|---|---|
| [docs/papers/EuqVid_Research_Structure_and_Workflow.pdf](docs/papers/EuqVid_Research_Structure_and_Workflow.pdf) | **Start here.** The four components, their feature sets, and the end-to-end workflow from Phase 0 to write-up |
| [docs/claude_project_instructions.md](docs/claude_project_instructions.md) | Per-member Claude Project instruction blocks, shared corrections, known traps |
| [docs/Step1_dataset_verification.md](docs/Step1_dataset_verification.md) | Corpus verification record, and the open EuqAud methodology questions for the supervisor |
| [components/README.md](components/README.md) | What goes in a member folder and what stays group-level |
| [components/C3_Text/EuqVid_Text_Component_Specification.pdf](components/C3_Text/EuqVid_Text_Component_Specification.pdf) | C3 component specification |

### In the Claude project only — not yet mirrored here

| Document | Purpose |
|---|---|
| `C3_dialogue_corpus_shortlist.md` | Candidate dialogue corpora for C3 with selection criteria. **Superseded on its central recommendation** — it argues for MovieNet and TVQA, both since dropped; the AVA decision below replaced it |
| `C3_AVA_decision_and_F6_change.md` | Why AVA is C3's third corpus, how the transcript is recovered, and the group-level F6 redefinition it forced |

The markdown sources of the structure and specification documents also live in the
Claude project; the repo carries their typeset PDFs.

Live team hub, task tracking and decision log: **Notion** (link in team chat).
Group logbook: `docs/logbook/README.md`.

---

## Setup

```bash
git clone <repo-url>
cd equvid
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Then follow `data/README.md` to obtain the corpora.

---

## Citation

If this work is used, please cite the parent methodology:

> S. Jayawardena, P. S. Haddela, T. Shyamalee, A. Ekanayake, T. Mudalige, I. Dhanawardhana.
> "EuqAud: Detecting Gender Bias in Audio Datasets Using Polynomial Regression-Based Metric."
> *IEEE Access*, vol. 14, 2026. DOI: 10.1109/ACCESS.2026.3674991
