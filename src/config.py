from dataclasses import dataclass, asdict
import os
import yaml


@dataclass
class Config:
    # --- data ---
    dataset: str = "pneumoniamnist"
    data_dir: str = "data"
    num_classes: int = 2

    # --- training ---
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    use_class_weights: bool = True

    # --- dataloader / performance ---
    num_workers: int = 0

    # --- reproducibility ---
    seed: int = 42

    # --- checkpointing ---
    checkpoint_dir: str = "checkpoints"


def load_config(path=None):
    """Start from the dataclass defaults, then apply overrides from a YAML file."""
    cfg = Config()
    if path:
        with open(path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        # Reject unknown keys early so a typo in the YAML fails loudly instead
        # of being silently ignored.
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config key in {path}: {key!r}")
            setattr(cfg, key, value)
    return cfg


def save_config(cfg, path):
    """Write the resolved config to YAML — the per-run capture for reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


if __name__ == "__main__":
    # Load defaults merged with the committed YAML and print what we'd train with.
    config = load_config("configs/default.yaml")
    print("Resolved configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")