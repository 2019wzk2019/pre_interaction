"""Full-grid TP ConvNet: input (B, T, 34, 21), time as channels."""

from __future__ import annotations

import torch
from torch import nn

from grip_dl import GRID_H, GRID_W, N_CLASSES, N_FRAMES


class TpGridNet(nn.Module):
    """
    Input:  (B, T, H, W) with T=N_FRAMES, H=34, W=21 (raw first-order diffs)
    Output: (B, 3) logits

    Time is treated as the channel dimension so a plain Conv2d stack stays
    ONNX / mobile friendly while keeping the full spatial layout.
    """

    def __init__(
        self,
        n_frames: int = N_FRAMES,
        grid_h: int = GRID_H,
        grid_w: int = GRID_W,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.backbone = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 17 x 10
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 x 5
            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W) — avoid Python shape asserts here for clean ONNX export
        h = self.backbone(x)
        return self.head(h.flatten(1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Back-compat alias used by older dry-run snippets.
TpEdgeNet = TpGridNet
