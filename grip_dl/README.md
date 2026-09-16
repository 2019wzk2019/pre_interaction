# TP deep learning (full 34×21)

Branch: `feat/tp-dl`. Replaces RF handcrafted features with a small ConvNet over
raw first-order TP diffs for the full grid (not edge-only).

## Input contract

| item | value |
|------|-------|
| window | 500 ms |
| frames | 42 (time-resampled) |
| grid | 34 × 21 |
| tensor | `(B, T, H, W) = (B, 42, 34, 21)` |
| labels | `左to右` / `右to左` / `others` |
| train data | `数据集/` |
| eval data | `Wzk/测试集` |

## Setup (server)

```bash
pip install -r requirements-dl.txt
# GPU PyTorch: install the CUDA wheel for your driver, then the rest of requirements-dl.txt
```

## Dry-run (no training)

```bash
python train_tp_dl.py
```

Checks forward shape, param count, and writes a dummy ONNX under `artifacts/exp_tp_dl_full/`.

## Train on GPU

```bash
python train_tp_dl.py --train --device auto --epochs 30 --batch-size 64 --num-workers 4
```

Artifacts:

- `artifacts/exp_tp_dl_full/tp_grid_net.pt` — best checkpoint by holdout acc
- `artifacts/exp_tp_dl_full/tp_grid_net.onnx` — ONNX export of best
- `artifacts/exp_tp_dl_full/metrics.json` — report + confusion matrix + per-folder

## Model

`TpGridNet`: Conv2d stack with **time as channels** (keeps full spatial layout, ONNX-friendly).
About ~200k params (see dry-run output).
