# Keystroke Dynamics for Parkinson's Detection

**Question:** Why do keystroke-based Parkinson's classifiers perform well on
clinic-collected data but degrade sharply on data collected at home?

Status: in progress. Step 1 (data loading + exploratory plots) — not yet complete.

---

## Background

Fine motor control degrades early in Parkinson's Disease, often before visible
tremor. Typing is a dense stream of timed finger movements, so keystroke timing
acts as a passive motor assessment.

Two timing features carry most of the signal:

- **Hold time (HT)** — key press to key release
- **Latency / flight time (FT)** — release of one key to press of the next

Published work reports hold time alone reaching AUC ≈ 0.81, above the clinical
alternating-finger-tapping test (≈ 0.75).

## Datasets

| Dataset | Subjects | Setting | Text | Source |
|---|---|---|---|---|
| neuroQWERTY MIT-CSXPD | 85 (42 PD) | Clinic | Free-text | PhysioNet |
| Tappy | 200+ | Home, unsupervised | Free-text | PhysioNet |

Raw data is **not** committed. See `data/README.md` for download instructions.

## Baseline reproduction

| Metric | Published | This repo |
|---|---|---|
| AUC (hold-time features, neuroQWERTY) | ~0.81 | _TBD_ |

## Results

_TBD_

## What Didn't Work

_Record every failed attempt here — wrong assumptions, features with no signal,
bugs that produced impossible numbers. This section is not optional._

## Evaluation protocol

Leave-one-subject-out (LOSO) cross-validation. Random train/test splits leak the
same subject into both sides and inflate AUC substantially at this sample size.

## Reproducing

```bash
git clone <this repo>
cd keystroke-pd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download.py          # fetches data from PhysioNet
jupyter notebook notebooks/01_explore.ipynb
```

## References

- Giancardo et al. — neuroQWERTY index (nQi)
- Tappy Keystroke Data, PhysioNet
- Francesconi et al. (2025) — cross-dataset benchmark of 8 architectures
