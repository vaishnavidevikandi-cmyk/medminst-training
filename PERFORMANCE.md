# Performance / Profiling Note

## What I measured

I instrumented a single training epoch (`src/profile_run.py`) to split wall-clock
time into two buckets: **data** (waiting for the dataloader to produce the next
batch) and **compute** (forward + backward + optimizer step). The ratio answers
the question that decides what optimization is worth doing: is the loop
**data-bound** or **compute-bound**?

Environment: CPU-only (Windows laptop), PneumoniaMNIST, batch size 128,
SmallCNN (~105k parameters), 37 train batches/epoch.

## Result (num_workers = 0)

| Bucket            | Time     | Share |
| ----------------- | -------- | ----- |
| Data (loading)    | 0.958 s  | 45.2% |
| Compute (fwd/bwd) | 1.160 s  | 54.8% |
| **Total**         | 2.119 s  | 100%  |

Throughput: ~17.5 batches/s.

The split is roughly even. With `num_workers=0` the dataloader runs *in the
training process*, so loading batch N+1 and computing on batch N happen
serially rather than in parallel. The ~45% "data" share is therefore time the
model sits idle waiting for input a structural inefficiency, not heavy I/O
(the dataset is tiny and effectively in memory).

## The optimization I tried — and why I rejected it

The textbook fix for serial loading is to raise `num_workers` so separate
processes prefetch the next batch while the current one is computed. I measured
it (`num_workers=4`, same machine):

| Bucket            | Time      | Share |
| ----------------- | --------- | ----- |
| Data (loading)    | 22.458 s  | 89.4% |
| Compute (fwd/bwd) | 1.201 s   | 4.8%  |
| **Total**         | 25.125 s  | 100%  |

Throughput collapsed to ~1.5 batches/s  **about 12× slower**.

Why: on Windows, worker processes use the "spawn" start method, which launches a
fresh Python interpreter and re-imports modules per worker. For a dataset this
small and fast, that fixed startup/IPC overhead dwarfs the loading work the
workers do. I was paying a large parallelization cost to speed up something that
was never the bottleneck.

**Conclusion:** `num_workers=0` is the correct default *for this dataset on this
OS*. It is kept as a config field rather than hardcoded precisely because the
right value is environment-dependent: on a Linux GPU host with a large on-disk
dataset, loading is genuinely heavy and workers are cheap, so `num_workers > 0`
would help there. The measurement, not a rule of thumb, drove the choice.

## What would change at scale (GPU / larger data)

These levers do little or nothing on this CPU setup but matter on GPU
infrastructure, and the pipeline is structured to adopt them:

- **num_workers > 0**: real win once data loading is the bottleneck and workers
  run cheaply (Linux, persistent workers, larger on-disk data).
- **Larger batch size**: GPUs are underutilized by tiny batches; bigger batches
  raise throughput until memory-bound.
- **Mixed precision (AMP)**: fp16/bf16 compute on GPU roughly halves memory and
  speeds up matmul-heavy layers; irrelevant on CPU.
- **cudnn.benchmark = True**: lets cuDNN autotune the fastest conv algorithm —
  a speed win, but it is non-deterministic, so it trades reproducibility for
  throughput (see reproducibility note). I would enable it only outside
  deterministic-debugging runs.


## One-line summary

Profiling showed an even data/compute split; the obvious `num_workers` fix was
empirically 12× slower on Windows+CPU due to process-spawn overhead, so the
honest optimization here was to *not* apply it and to keep it configurable for
the environments where it does pay off.