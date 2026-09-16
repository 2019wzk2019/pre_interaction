from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    train_root: Path = Path("数据集")
    test_root: Path = Path("Wzk") / "测试集"
    out_dir: Path = Path("artifacts") / "exp_tp_dl_full"
    window_ms: float = 500.0
    stride_ms: float = 50.0
    n_frames: int = 42
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | cuda:0
