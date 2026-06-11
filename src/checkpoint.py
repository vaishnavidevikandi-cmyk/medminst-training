"""
checkpoint.py — Save and restore full training state.

A checkpoint here is more than model weights: it captures the optimizer state,
the epoch index, and the best validation metric seen so far. That is the
difference between "load a model for inference" and "resume training exactly
where it stopped" — the latter is what makes a long run survivable across a
crash, a preempted instance, or a wall-clock limit.
"""
import os
import torch


def save_checkpoint(path, model, optimizer, epoch, best_metric):
    """Write full training state to `path` (creating parent dirs as needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Store metrics as native Python types so checkpoints load under
    # torch.load's secure weights_only=True default (PyTorch >= 2.6).
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """Restore state into `model` (and `optimizer` if given).

    Returns (start_epoch, best_metric) so the caller can continue the loop.
    Optimizer is optional: pass it when resuming training, omit it for
    inference-only loads where optimizer state is irrelevant.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    # Resume on the epoch AFTER the one we last completed.
    start_epoch = checkpoint["epoch"] + 1
    best_metric = checkpoint["best_metric"]
    return start_epoch, best_metric