from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from grip_dl import GRID_H, GRID_W, N_CLASSES, N_FRAMES


class TpSequenceDataset(Dataset):
    """X is (N, T, H, W); tensors returned as (T, H, W) for Conv2d (time=channels)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 4:
            raise ValueError(f"expected X (N,T,H,W), got {X.shape}")
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])  # (T, H, W)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def class_sample_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w_class = counts.sum() / (N_CLASSES * counts)
    return torch.from_numpy(w_class[y].astype(np.float32))


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    weighted: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    ds = TpSequenceDataset(X, y)
    sampler = None
    if weighted and len(ds) > 0:
        sampler = WeightedRandomSampler(
            class_sample_weights(y), num_samples=len(ds), replacement=True
        )
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def dummy_batch(
    batch_size: int = 4,
    n_frames: int = N_FRAMES,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
):
    x = torch.randn(batch_size, n_frames, grid_h, grid_w)
    y = torch.randint(0, N_CLASSES, (batch_size,))
    return x, y
