from __future__ import annotations

from pathlib import Path

import torch

from grip_dl import GRID_H, GRID_W, N_FRAMES
from grip_dl.model import TpGridNet


def export_onnx(
    model: TpGridNet,
    path: Path,
    n_frames: int = N_FRAMES,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, n_frames, grid_h, grid_w)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["tp_grid"],
        output_names=["logits"],
        dynamic_axes={"tp_grid": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=11,
    )
    return path
