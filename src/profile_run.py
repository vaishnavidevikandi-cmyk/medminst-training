import time
import torch
import torch.nn as nn

from config import load_config
from reproducibility import set_seed
from data import get_dataloaders, compute_class_weights
from model import build_model


def main():
    cfg = load_config("configs/default.yaml")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | batch_size={cfg.batch_size} "
          f"| num_workers={cfg.num_workers}")

    train_loader, _, _ = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    weights = compute_class_weights(cfg).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    data_time = 0.0      # time spent waiting for the next batch
    compute_time = 0.0   # time spent on forward + backward + step
    n_batches = 0

    epoch_start = time.perf_counter()
    batch_end = time.perf_counter()
    for images, labels in train_loader:
        # Time since the previous batch finished = dataloader wait.
        data_time += time.perf_counter() - batch_end

        compute_start = time.perf_counter()
        images = images.to(device)
        labels = labels.squeeze().long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        compute_time += time.perf_counter() - compute_start

        n_batches += 1
        batch_end = time.perf_counter()

    total = time.perf_counter() - epoch_start
    print(f"\nBatches: {n_batches}")
    print(f"Total epoch time:   {total:.3f}s")
    print(f"  Data  (loading):  {data_time:.3f}s  ({100*data_time/total:.1f}%)")
    print(f"  Compute (fwd/bwd): {compute_time:.3f}s  ({100*compute_time/total:.1f}%)")
    print(f"Throughput: {n_batches/total:.1f} batches/s")


if __name__ == "__main__":
    main()