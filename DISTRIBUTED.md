## Reproducibility and performance under DDP

- **Seeding**: the existing `set_seed` is called per process; seeding each rank
  *identically* gives identical initial weights (required), while the
  `DistributedSampler` ensures the *data* differs per rank. For fully independent
  augmentation streams one would offset the seed by rank.
- **Effective batch size** scales with the number of GPUs (per-GPU batch ×
  world_size). The learning rate usually needs to scale with it (e.g. linear
  scaling rule with warmup), which would become a config concern.
- **num_workers** (see performance note) becomes a real lever here: on Linux GPU
  nodes with sharded on-disk data, worker prefetch overlaps loading with
  compute cheaply — the opposite of the Windows+CPU result measured earlier.
- **cudnn.benchmark = True** is worth enabling for fixed input sizes (our 28×28
  is fixed) to autotune convolutions, accepting the determinism trade-off.

## What I would add with more time

- A `distributed` flag in the config that conditionally wraps the model, swaps
  in the sampler, and guards rank-0 side effects, so the same `train.py` runs
  both single-device and DDP.
- Gradient accumulation for when target effective batch size exceeds GPU memory.
- A short test on 2 processes via `gloo` on CPU to validate the DDP code path
  without needing GPUs.