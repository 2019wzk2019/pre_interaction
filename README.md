# pre_interaction

Grip / reach pre-interaction classification from IMU and TP (touch panel).

Classes: `左to右` / `右to左` / `others`.

## Retrain the latest TP model

Data is already split under `data/tp/`. Train on `data/tp/train` (121 recordings). Evaluate on `data/tp/test` (2 holdout recordings). The default command matches the shipped single-frame model in `delivery/hap_delivery_DL`.

```bash
pip install -r requirements-dl.txt
python scripts/train_tp_dl.py --train --device auto --epochs 30 --batch-size 64
```

Outputs land in `artifacts/exp_tp_dl_single/`. See `data/tp/README.txt` for the class folders.

## Layout

| path | what |
|------|------|
| `data/tp/train` `data/tp/test` | latest TP train / holdout |
| `data/wzk/` | older random-forest recordings |
| `data/imu_old/` `data/imu_new/` | IMU recordings |
| `data/datatp/` | continuous sessions used for TP videos |
| `grip_classify/` `grip_dl/` | RF and ConvNet libraries |
| `scripts/` | train, export, and plot entry points |
| `delivery/` | HarmonyOS ONNX packages |
| `artifacts/` | checkpoints and metrics |
| `docs/` | colleague notes |

## Other training

```bash
python scripts/train_grip_models.py --modality both --window-ms 150
python scripts/train_imu_dl.py
```

## HarmonyOS app

HAP 工程不在本仓库。适配说明见 `docs/HAP_TP_ADAPT.md`。上机包在 `delivery/`。

本地若仍保留 `harmony_app/`：

```bash
python scripts/sync_harmony_models.py
```
