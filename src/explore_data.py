"""
explore_data.py — Download and summarize the PneumoniaMNIST dataset.

Prints task type, image size, split sizes, and class distribution so we can
document the dataset and surface its class imbalance.

Run from the project root:
    python src/explore_data.py
"""

import os
import numpy as np
from medmnist import INFO, PneumoniaMNIST

DATASET_KEY = "pneumoniamnist"
DATA_DIR = "data"

# Make sure the (gitignored) data directory exists before downloading.
os.makedirs(DATA_DIR, exist_ok=True)

# 1) Metadata that MedMNIST ships for every sub-dataset.
info = INFO[DATASET_KEY]
print("=" * 52)
print(f"Dataset:  {DATASET_KEY}")
print(f"Task:     {info['task']}")
print(f"Channels: {info['n_channels']}  (1 = grayscale)")
print(f"Classes:  {info['label']}")          # e.g. {'0': 'normal', '1': 'pneumonia'}
print("=" * 52)

# 2) Download each split once into ./data and cache it.
train_ds = PneumoniaMNIST(split="train", download=True, root=DATA_DIR)
val_ds   = PneumoniaMNIST(split="val",   download=True, root=DATA_DIR)
test_ds  = PneumoniaMNIST(split="test",  download=True, root=DATA_DIR)

# 3) Split sizes.
print("\nSplit sizes:")
print(f"  train: {len(train_ds)}")
print(f"  val:   {len(val_ds)}")
print(f"  test:  {len(test_ds)}")

# 4) Image shape and a single sample.
print(f"\nImage array shape (train): {train_ds.imgs.shape}")  # (N, 28, 28)
img, label = train_ds[0]
print(f"Sample 0 -> image: {type(img).__name__}, label array: {label}")

# 5) Class distribution — the imbalance story.
def show_class_balance(name, dataset):
    labels = dataset.labels.flatten()                 # (N, 1) -> (N,)
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    print(f"\n{name} class balance ({total} samples):")
    for v, c in zip(values, counts):
        class_name = info["label"][str(v)]
        print(f"  {v} ({class_name}): {c:>5}  ({100 * c / total:5.1f}%)")

show_class_balance("Train", train_ds)
show_class_balance("Val",   val_ds)
show_class_balance("Test",  test_ds)