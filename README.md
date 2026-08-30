# EuqVid

**A Gender Bias Detection Metric for Video Datasets Using Visual and Temporal Features**

Group J26-DS-343 · CoEAI · SLIIT · IT4010 Research Project
Supervisor: Dr. Prasanna S. Haddela

Extends [EuqAud](https://doi.org/10.1109/ACCESS.2026.3674991) (Jayawardena et al., IEEE Access 2026) from the audio modality into video.

---

## What this is

A pre-training, model-agnostic metric that scores gender bias in video datasets before model development begins. It produces a frozen closed-form polynomial Ridge equation mapping 12 visual and temporal features to a bias score on [−10, +10], with Neutral / Moderate / Strong severity tiers.

**Domain:** narrative multi-person video — film and scripted television. Corpora where two or more subjects share the frame and shot composition reflects narrative importance.

**What it measures:** *within-content* representational disparity. Given subjects of both genders in the same material, is one systematically afforded greater visual and temporal prominence? Zero means parity — not "the naturally occurring distribution."

---

## Team

| Member | Reg No | Component |
|---|---|---|
| Hasangani WRA | IT23224384 | M1 — Visual Feature Extraction |
| Harischandra RASR | IT23194908 | M2 — Temporal Analysis |
| Fernando RSR | IT23449282 | M3 — EuqVid Metric (Polynomial Ridge) |
| Sathsarani WTTS | IT23204102 | M4 — Validation & Human Perception |

---

## Repository layout

```
euqvid/
├── README.md               you are here
├── .gitignore              blocks datasets from ever being committed
├── docs/                   design decisions, dataset verification, papers
├── data/                   how to obtain the corpora — NOT the corpora themselves
│   ├── README.md           per-corpus instructions, licences, traps
│   ├── download_ava.py
│   └── download_ami.py
├── src/                    pipeline code
└── notebooks/              exploration
```

---

## Getting the data

**The datasets are not in this repository and must never be added to it.** They total ~43 GB and their licences do not permit redistribution.

To set up on a new machine, see **[data/README.md](data/README.md)**. Every corpus is re-downloadable from its original source; that file plus the download scripts is what makes this project reproducible.

---

## Working documents

| Document | Purpose |
|---|---|
| [docs/M3_design_decisions_to_lock.md](docs/M3_design_decisions_to_lock.md) | Every open methodological decision, with recommendations and the reasoning |
| [docs/Step1_dataset_verification.md](docs/Step1_dataset_verification.md) | Corpus verification status — licence, gender labels, raw video availability |
| [docs/EuqVid_parallel_component_restructure.md](docs/EuqVid_parallel_component_restructure.md) | Alternative non-blocking component structure (not adopted; kept as a fallback) |

Live team hub, task tracking and decision log: **Notion** (link in team chat).

---

## Setup

```bash
git clone <repo-url>
cd euqvid
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
