# MedMNIST Training Pipeline

A reproducible PyTorch training pipeline for a MedMNIST sub-dataset, built to
demonstrate production-oriented ML engineering: configurable training,
checkpoint/resume support, reproducibility controls, and a lightweight
responsible-AI governance package.

> **Status:** work in progress.
## Project structure
_To be completed._

## Environment setup
```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell

# 2. Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Dataset

**Sub-dataset:** PneumoniaMNIST (from the MedMNIST v2 collection).

**Why this one:** a binary task (pneumonia vs. normal) on grayscale pediatric
chest X-rays. It is small enough to train end-to-end on CPU, its clinical framing
makes the responsible-AI discussion concrete, and it carries a natural class
imbalance that motivates deliberate metric and loss choices.

| Property | Value |
| --- | --- |
| Task | Binary classification (`binary-class`) |
| Image size | 28 × 28, 1 channel (grayscale) |
| Splits (train / val / test) | 4,708 / 524 / 624 |
| Classes | 0 = normal, 1 = pneumonia |

**Observed class balance:**

| Split | normal | pneumonia |
| --- | --- | --- |
| Train | 1,214 (25.8%) | 3,494 (74.2%) |
| Val | 135 (25.8%) | 389 (74.2%) |
| Test | 234 (37.5%) | 390 (62.5%) |

**Limitations this surfaces:**

1. **Class imbalance (~3:1 toward pneumonia in train/val).** Plain accuracy is
   misleading — always predicting "pneumonia" scores ~74% on train/val while
   learning nothing. We therefore evaluate with AUC and per-class
   recall/precision, and consider class-weighted loss.
2. **Train/test distribution shift.** Train and val share an identical balance
   (split from the same source pool), but the test set is less imbalanced
   (37.5% / 62.5%) because it derives from a different source split. Validation
   metrics will not perfectly predict test performance, and thresholds tuned on
   val may be slightly off on test — another reason to prefer threshold-independent
   metrics like AUC.

## Usage
### Train
### Evaluate
### Resume from a checkpoint

## Reproducibility controls
_To be completed._

## Performance / profiling notes
_To be completed._

## Distributed training plan
_To be completed._

## Model card & responsible-AI note
_To be completed._

## Assumptions and "what I'd improve with more time"
_To be completed._