# TP deep learning (full 34×21, single frame)

Shipped model: `delivery/hap_delivery_DL`. One raw diff frame, full grid, per-sample std.

## Input contract

| item | value |
|------|-------|
| window | 1 frame (`window_ms=0`) |
| tensor | `(B, T, H, W) = (B, 1, 34, 21)` |
| labels | `左to右` / `右to左` / `others` |
| train | `data/tp/train` |
| eval | `data/tp/test` |

## Setup

```bash
pip install -r requirements-dl.txt
```

## Dry-run

```bash
python scripts/train_tp_dl.py
```

## Train

```bash
python scripts/train_tp_dl.py --train --device auto --epochs 30 --batch-size 64 --num-workers 4
```

Artifacts in `artifacts/exp_tp_dl_single/`:

- `tp_grid_net.pt`
- `tp_grid_net.onnx`
- `metrics.json`

The older 500 ms / 42-frame run is:

```bash
python scripts/train_tp_dl.py --train --n-frames 42 --window-ms 500 --out-dir artifacts/exp_tp_dl_full
```
