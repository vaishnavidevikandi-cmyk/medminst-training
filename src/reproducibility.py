"""
reproducibility.py — Seed every RNG the pipeline touches.

PyTorch training draws randomness from three independent generators (Python's
`random`, NumPy, and torch), and CUDA adds non-deterministic kernels on top.
Seeding only one of them is a common mistake that leaves runs irreproducible.
This module seeds all of them from a single number and can optionally force
deterministic algorithms — at a documented speed cost — so a run can be
replayed exactly.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed, deterministic=False):
    """Seed all RNGs. When deterministic=True, also disable nondeterministic ops."""
    # The three independent generators.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # no-op on CPU, correct to set for GPU runs

    # Make hash-based ordering reproducible across processes.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Force deterministic kernels where available. This is slower, which is
        # why it is opt-in: reproducibility for debugging vs. speed for scale.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Prove it: seed, draw numbers, reseed, draw again — should match exactly.
    set_seed(42)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(42)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    print("First draw: ", a)
    print("Second draw:", b)
    print("Match:", a == b)