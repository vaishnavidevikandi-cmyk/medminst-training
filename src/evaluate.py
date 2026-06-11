import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report)

from config import load_config
from reproducibility import set_seed
from data import get_dataloaders
from model import build_model
from checkpoint import load_checkpoint


@torch.no_grad()
def run_eval(model, loader, device):
    """Collect predictions over a loader. Returns (labels, probs, preds)."""
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long()

        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]   # P(pneumonia)
        preds = logits.argmax(dim=1)                 # hard class prediction

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only the test loader is needed here.
    _, _, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    # Inference-only load: no optimizer needed.
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    labels, probs, preds = run_eval(model, test_loader, device)

    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    target_names = ["normal", "pneumonia"]

    print(f"\nTest AUC: {auc:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(f"              pred_normal  pred_pneumonia")
    print(f"true_normal      {cm[0,0]:>6}        {cm[0,1]:>6}")
    print(f"true_pneumonia   {cm[1,0]:>6}        {cm[1,1]:>6}")
    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=target_names,
                                digits=4))


if __name__ == "__main__":
    main()