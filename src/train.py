import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from config import load_config, save_config
from reproducibility import set_seed
from data import get_dataloaders, compute_class_weights
from model import build_model
from checkpoint import save_checkpoint, load_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None,
                        help="path to a checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_config(cfg, os.path.join(cfg.checkpoint_dir, "used_config.yaml"))

    train_loader, val_loader, _ = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    if cfg.use_class_weights:
        weights = compute_class_weights(cfg).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    start_epoch = 0
    best_auc = 0.0
    if args.resume:
        start_epoch, best_auc = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, "
              f"best AUC so far {best_auc:.4f}")

    last_path = os.path.join(cfg.checkpoint_dir, "last.pt")
    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")

    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:2d}/{cfg.epochs} | "
              f"train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_auc {val_auc:.4f}")
        
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc

        save_checkpoint(last_path, model, optimizer, epoch, best_auc)
        
        if is_best:
            save_checkpoint(best_path, model, optimizer, epoch, best_auc)
            print(f"  -> new best val_auc {best_auc:.4f}, saved {best_path}")

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()