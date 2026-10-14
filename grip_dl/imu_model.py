"""1D ConvNet over IMU windows: input (B, 6, T) acc+gyro."""

from __future__ import annotations

from torch import nn


class ImuWinNet(nn.Module):
    def __init__(self, n_ch: int = 6, n_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(n_ch, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(96, n_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        h = self.backbone(x)
        return self.head(h.flatten(1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
