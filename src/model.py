import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()

        # Two conv blocks. Each: convolution -> ReLU -> 2x2 max-pool.
        # 28x28 -> (pool) 14x14 -> (pool) 7x7, with channels 1 -> 16 -> 32.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 28x28 -> 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 14x14 -> 7x7
        )

        # Classifier head. Flatten the 32x7x7 feature map, then two linear
        # layers with dropout for a little regularization on a small dataset.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x                                   # raw logits, shape (B, num_classes)


def build_model(cfg):
    """Construct the model from the config (keeps callers config-driven)."""
    return SmallCNN(num_classes=cfg.num_classes)


if __name__ == "__main__":
    from config import load_config
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg)

    # Parameter count — a quick sanity check that this really is a small model.
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\nTrainable parameters: {n_params:,}")

    # Forward a fake batch to confirm the output shape is (B, num_classes).
    dummy = torch.randn(4, 1, 28, 28)
    out = model(dummy)
    print(f"Output shape for a batch of 4: {tuple(out.shape)}")