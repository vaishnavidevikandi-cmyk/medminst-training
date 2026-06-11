# MedMNIST Training Pipeline

A reproducible PyTorch training pipeline for a MedMNIST sub-dataset, built to
demonstrate production-oriented ML engineering: configurable training,
checkpoint/resume support, reproducibility controls, and a lightweight
responsible-AI governance package.

## Project structure

```
medmnist-training/
├── configs/
│   ├── default.yaml            # main training config
│   ├── resume_demo.yaml        # short 3-epoch run for the resume demo
│   └── resume_demo_long.yaml   # 6-epoch run showing resume continues training
├── src/
│   ├── config.py               # dataclass config + YAML override + run capture
│   ├── reproducibility.py      # seeds all RNGs; optional deterministic mode
│   ├── data.py                 # DataLoaders + train-split class weights
│   ├── model.py                # SmallCNN (~105k params) for 28x28 grayscale
│   ├── checkpoint.py           # full-state save/load for exact resume
│   ├── train.py                # training loop with checkpoint/resume
│   ├── evaluate.py             # held-out test evaluation
│   ├── profile_run.py          # per-epoch data-vs-compute timing
│   └── explore_data.py         # dataset download + class-balance summary
├── PERFORMANCE.md              # profiling note and num_workers finding
├── DISTRIBUTED.md              # multi-GPU / DDP scale-out plan
├── MODEL_CARD.md               # model card + responsible-AI note
├── requirements.txt
└── README.md
```

(`data/`, `.venv/`, and `checkpoints*/` are gitignored — datasets and
checkpoints regenerate from the commands below.)

## Environment setup

```
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

| Property                    | Value                                  |
| --------------------------- | -------------------------------------- |
| Task                        | Binary classification (`binary-class`) |
| Image size                  | 28 × 28, 1 channel (grayscale)         |
| Splits (train / val / test) | 4,708 / 524 / 624                      |
| Classes                     | 0 = normal, 1 = pneumonia              |

**Observed class balance:**

| Split | normal        | pneumonia     |
| ----- | ------------- | ------------- |
| Train | 1,214 (25.8%) | 3,494 (74.2%) |
| Val   | 135 (25.8%)   | 389 (74.2%)   |
| Test  | 234 (37.5%)   | 390 (62.5%)   |

**Limitations this surfaces:**

1. **Class imbalance (~3:1 toward pneumonia in train/val).** Plain accuracy is
misleading — always predicting "pneumonia" scores ~74% on train/val while
learning nothing. We therefore evaluate with AUC and per-class
recall/precision, and apply a class-weighted loss.
2. **Train/test distribution shift.** Train and val share an identical balance,
but the test set is less imbalanced (37.5% / 62.5%) because it derives from a
different source split. Validation metrics will not perfectly predict test
performance — another reason to prefer threshold-independent metrics like AUC.

To download the data and print the summary above:

```
python src/explore_data.py
```

## Usage

### Train

```
python src/train.py --config configs/default.yaml
```

Trains the model, evaluating on the validation split each epoch. Writes
`checkpoints/last.pt` every epoch and `checkpoints/best.pt` whenever validation
AUC improves, plus `checkpoints/used_config.yaml` capturing the exact settings
used for the run.

### Evaluate

```
python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

Loads the checkpoint and reports test-set AUC, a confusion matrix, and a
per-class precision/recall/F1 report. The test split is touched only here.

### Resume from a checkpoint

```
python src/train.py --config configs/default.yaml --resume checkpoints/last.pt
```

Restores model weights, optimizer state, epoch index, and best-metric-so-far,
and continues training from the next epoch as if it was never interrupted.

## Reproducibility controls

- **Seeding:** `src/reproducibility.py` seeds Python's `random`, NumPy, and
  PyTorch from a single config value (`seed`), and sets `PYTHONHASHSEED`.
- **Optional determinism:** a `deterministic` mode forces deterministic
  algorithms (`torch.use_deterministic_algorithms`) and disables
  `cudnn.benchmark` — at a documented speed cost, so it is opt-in.
- **Config capture:** every run writes its fully-resolved config to
  `checkpoints/used_config.yaml`, so a run's exact settings are recoverable.
- **Pinned dependencies:** `requirements.txt` pins versions.

## Performance / profiling notes

See **[PERFORMANCE.md](PERFORMANCE.md)**. Summary: profiling one epoch showed a
roughly even data/compute split; raising `num_workers` to 4 was measured at
~12× slower on Windows+CPU due to process-spawn overhead, so `num_workers=0` is
the correct default here — kept configurable for Linux/GPU environments where it
helps.

## Distributed training plan

See **[DISTRIBUTED.md](DISTRIBUTED.md)** for how the pipeline extends to
multi-GPU / multi-node training with DistributedDataParallel: process-group
setup, `DistributedSampler` sharding, model wrapping, rank-0-gated
logging/checkpointing, and cross-process metric aggregation.

## Model card & responsible-AI note

See **[MODEL_CARD.md](MODEL_CARD.md)** for intended use and boundaries,
measured test-set error profiling, fairness/subgroup limitations, misuse risks,
and an operational-readiness checklist. In short: this is an engineering
artifact on a downsampled public benchmark and must not inform decisions about
real patients.

## Assumptions and "what I'd improve with more time"

**Assumptions:**
- CPU-only target hardware; PneumoniaMNIST chosen for fast CPU iteration.
- A small CNN is appropriate given the 28×28 resolution and CPU constraint;
  the goal is pipeline quality, not benchmark-topping accuracy.
- `num_workers=0` is correct for this dataset/OS (see performance note).

**What I'd improve with more time:**
- A `distributed` config flag so `train.py` runs both single-device and DDP.
- Probability calibration and explicit decision-threshold tuning.
- Structured logging / experiment tracking (e.g. TensorBoard or W&B).
- A small CI check (lint + a 1-epoch smoke-test run).
