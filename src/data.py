"""
data.py — Build PneumoniaMNIST DataLoaders and training class weights.

Wraps the MedMNIST dataset in torchvision transforms (tensor + normalization)
and PyTorch DataLoaders. Also computes inverse-frequency class weights from the
TRAIN split only, so the ~3:1 pneumonia imbalance documented in the README can
be corrected in the loss without leaking val/test statistics.
"""
import numpy as np
import torch
from torchvision import transforms
from medmnist import PneumoniaMNIST


# PneumoniaMNIST is single-channel; these are the standard MedMNIST stats.
_NORMALIZE = transforms.Normalize(mean=[0.5], std=[0.5])
_TRANSFORM = transforms.Compose([transforms.ToTensor(), _NORMALIZE])


def _make_dataset(split, data_dir):
    """Load one split with the shared transform applied to each image."""
    return PneumoniaMNIST(split=split, root=data_dir, download=True,
                          transform=_TRANSFORM)


def get_dataloaders(cfg):
    """Return (train_loader, val_loader, test_loader) built from the config."""
    train_ds = _make_dataset("train", cfg.data_dir)
    val_ds   = _make_dataset("val",   cfg.data_dir)
    test_ds  = _make_dataset("test",  cfg.data_dir)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers)

    return train_loader, val_loader, test_loader


def compute_class_weights(cfg):
    """Inverse-frequency weights from the TRAIN split, for a weighted loss.

    Rarer classes get larger weights so they are not drowned out by the
    majority class. Computed on train only to avoid leaking val/test stats.
    """
    train_ds = PneumoniaMNIST(split="train", root=cfg.data_dir, download=True)
    labels = train_ds.labels.flatten()
    counts = np.bincount(labels, minlength=cfg.num_classes)
    # weight_c = total / (num_classes * count_c): scales each class to parity.
    weights = counts.sum() / (cfg.num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    from config import load_config
    cfg = load_config("configs/default.yaml")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    print(f"Batches -> train: {len(train_loader)}, "
          f"val: {len(val_loader)}, test: {len(test_loader)}")

    # Inspect one batch to confirm shapes and value range.
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {tuple(images.shape)}")   # (B, 1, 28, 28)
    print(f"Batch labels shape: {tuple(labels.shape)}")   # (B, 1)
    print(f"Pixel range: [{images.min():.2f}, {images.max():.2f}]")

    weights = compute_class_weights(cfg)
    print(f"Class weights (normal, pneumonia): {weights.tolist()}")